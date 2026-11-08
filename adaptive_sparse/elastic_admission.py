"""Live request-level residency decisions under an aggregate KV budget.

The offline planner chooses an optimistic mode from predicted runtimes.  This
module applies the same choice to the controller's observed in-flight set, so a
serving process can preserve full-GPU residency while load is light and demote
only new requests when active KV would cross the configured budget.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .future_plan import PlannerConfig, RolloutSubmission
from .sglang_future_hook import FutureHint


@dataclass(frozen=True, slots=True)
class ElasticKVConfig:
    """Conservative byte model used only by the rollout controller."""

    memory_budget_bytes: int
    kv_bytes_per_token: int
    sparse_hot_tokens: int
    auto_resident_output_tokens: int = 128
    lookahead_ms: int = 0
    max_future_reservations: int = 4
    prefill_workspace_bytes_per_token: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("memory_budget_bytes", self.memory_budget_bytes),
            ("kv_bytes_per_token", self.kv_bytes_per_token),
            ("sparse_hot_tokens", self.sparse_hot_tokens),
            ("auto_resident_output_tokens", self.auto_resident_output_tokens),
            ("max_future_reservations", self.max_future_reservations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.lookahead_ms, bool)
            or not isinstance(self.lookahead_ms, int)
            or self.lookahead_ms < 0
        ):
            raise ValueError("lookahead_ms must be a nonnegative integer")
        if (
            isinstance(self.prefill_workspace_bytes_per_token, bool)
            or not isinstance(self.prefill_workspace_bytes_per_token, int)
            or self.prefill_workspace_bytes_per_token < 0
        ):
            raise ValueError(
                "prefill_workspace_bytes_per_token must be a nonnegative integer"
            )

    def hold_bytes(self, submission: RolloutSubmission, residence: str) -> int:
        """Return the predicted per-rank KV bytes for one placement."""

        sequence_tokens = submission.prompt_tokens + submission.output_tokens
        held_tokens = (
            sequence_tokens
            if residence == "resident"
            else min(sequence_tokens, self.sparse_hot_tokens)
        )
        return held_tokens * self.kv_bytes_per_token

    def prefill_workspace_bytes(self, submission: RolloutSubmission) -> int:
        return submission.prompt_tokens * self.prefill_workspace_bytes_per_token


def elastic_config_from_planner(config: PlannerConfig) -> ElasticKVConfig:
    """Project the offline resource model into the live admission ledger."""

    return ElasticKVConfig(
        memory_budget_bytes=config.memory_budget_bytes,
        kv_bytes_per_token=config.kv_bytes_per_token,
        sparse_hot_tokens=config.sparse_hot_tokens,
        prefill_workspace_bytes_per_token=config.prefill_workspace_bytes_per_token,
        auto_resident_output_tokens=config.auto_resident_output_threshold,
        lookahead_ms=config.lookahead_ms,
    )


@dataclass(frozen=True, slots=True)
class ElasticAdmissionDecision:
    request_id: str
    planned_residence: str
    chosen_residence: str
    reason: str
    active_before_bytes: int
    active_after_bytes: int
    resident_bytes: int
    sparse_bytes: int
    defer_for_budget: bool
    over_budget: bool
    future_reserved_bytes: int = 0
    future_prefill_bytes: int = 0
    future_reserved_requests: int = 0
    future_effective_after_bytes: int = 0
    future_over_budget: bool = False
    prefill_workspace_bytes: int = 0
    prefill_peak_bytes: int = 0
    future_effective_peak_bytes: int = 0
    prefill_over_budget: bool = False
    future_peak_over_budget: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "planned_residence": self.planned_residence,
            "chosen_residence": self.chosen_residence,
            "reason": self.reason,
            "active_before_bytes": self.active_before_bytes,
            "active_after_bytes": self.active_after_bytes,
            "resident_bytes": self.resident_bytes,
            "sparse_bytes": self.sparse_bytes,
            "defer_for_budget": self.defer_for_budget,
            "over_budget": self.over_budget,
            "future_reserved_bytes": self.future_reserved_bytes,
            "future_prefill_bytes": self.future_prefill_bytes,
            "future_reserved_requests": self.future_reserved_requests,
            "future_effective_after_bytes": self.future_effective_after_bytes,
            "future_over_budget": self.future_over_budget,
            "prefill_workspace_bytes": self.prefill_workspace_bytes,
            "prefill_peak_bytes": self.prefill_peak_bytes,
            "future_effective_peak_bytes": self.future_effective_peak_bytes,
            "prefill_over_budget": self.prefill_over_budget,
            "future_peak_over_budget": self.future_peak_over_budget,
        }


def _preferred_residence(
    submission: RolloutSubmission, config: ElasticKVConfig
) -> tuple[str, str]:
    """Return the placement implied by one request before live contention."""

    hint = FutureHint.from_payload(dict(submission.hint))
    sequence_tokens = submission.prompt_tokens + submission.output_tokens
    if hint.residence == "host_backed":
        return "host_backed", "planned-host-backed"
    if sequence_tokens <= config.sparse_hot_tokens:
        return "host_backed", "fits-native-hot-set"
    if hint.residence == "auto":
        if submission.output_tokens >= config.auto_resident_output_tokens:
            return "resident", "auto-long"
        return "host_backed", "auto-short"
    return "resident", "planned-resident"


def _future_reservation(
    submission: RolloutSubmission,
    future_submissions: Iterable[RolloutSubmission],
    *,
    now_ms: float,
    config: ElasticKVConfig,
) -> tuple[int, int, int]:
    """Reserve bounded capacity for more urgent requests that can overlap now.

    The executor receives a known rollout schedule. Counting every queued
    request would serialize the whole trace, so this conservative guard only
    includes requests that may start before the current request finishes,
    arrive inside the configured horizon, and carry a deadline or a strictly
    larger soft business weight. A fixed request cap bounds both the search and
    the over-reservation caused by an unusually dense trace.
    """

    if config.lookahead_ms == 0:
        return 0, 0, 0
    current_hint = FutureHint.from_payload(dict(submission.hint))
    current_finish_ms = now_ms + max(
        1, submission.estimated_finish_ms - submission.submit_at_ms
    )
    horizon_end_ms = now_ms + config.lookahead_ms
    reserved_kv_bytes = 0
    reserved_prefill_bytes = 0
    reserved_requests = 0
    resident_prefixes: dict[str, list[int]] = {}
    for future in future_submissions:
        if future.request_id == submission.request_id:
            continue
        if future.submit_at_ms > horizon_end_ms:
            continue
        if future.submit_at_ms > current_finish_ms:
            continue
        future_hint = FutureHint.from_payload(dict(future.hint))
        protected = (
            future_hint.deadline_after_ms is not None
            or future_hint.weight > current_hint.weight
        )
        if not protected:
            continue
        placement, _ = _preferred_residence(future, config)
        reserved_kv_bytes += config.hold_bytes(future, placement)
        reserved_prefill_bytes += config.prefill_workspace_bytes(future)
        if (
            placement == "resident"
            and future_hint.prefix_group is not None
            and future_hint.shared_prefix_tokens > 0
        ):
            resident_prefixes.setdefault(future_hint.prefix_group, []).append(
                future_hint.shared_prefix_tokens
            )
        reserved_requests += 1
        if reserved_requests >= config.max_future_reservations:
            break
    reserved_kv_bytes -= sum(
        (len(lengths) - 1) * min(lengths) * config.kv_bytes_per_token
        for lengths in resident_prefixes.values()
        if len(lengths) > 1
    )
    return reserved_kv_bytes, reserved_prefill_bytes, reserved_requests


def decide_elastic_admission(
    submission: RolloutSubmission,
    *,
    active_bytes: int,
    config: ElasticKVConfig,
    future_submissions: Iterable[RolloutSubmission] = (),
    now_ms: float | None = None,
) -> ElasticAdmissionDecision:
    """Choose a mode and report whether admission should wait for capacity.

    ``future_submissions`` is optional. When present with a positive
    ``config.lookahead_ms``, bounded capacity is reserved before the current
    request claims a resident slot. Omitting it preserves observed-load-only
    admission.
    """

    if isinstance(active_bytes, bool) or not isinstance(active_bytes, int):
        raise ValueError("active_bytes must be an integer")
    if active_bytes < 0:
        raise ValueError("active_bytes must be nonnegative")
    queued = tuple(future_submissions)
    if now_ms is None:
        if config.lookahead_ms > 0 and queued:
            raise ValueError("now_ms is required when future reservations are enabled")
        now_ms = 0.0
    if isinstance(now_ms, bool) or not isinstance(now_ms, (int, float)):
        raise ValueError("now_ms must be finite")
    now_ms = float(now_ms)
    if not math.isfinite(now_ms) or now_ms < 0:
        raise ValueError("now_ms must be finite and nonnegative")
    hint = FutureHint.from_payload(dict(submission.hint))
    planned = hint.residence
    resident_bytes = config.hold_bytes(submission, "resident")
    sparse_bytes = config.hold_bytes(submission, "host_backed")
    workspace_bytes = config.prefill_workspace_bytes(submission)
    chosen, reason = _preferred_residence(submission, config)
    future_kv_bytes, future_prefill_bytes, future_count = _future_reservation(
        submission, queued, now_ms=now_ms, config=config
    )
    future_bytes = future_kv_bytes + future_prefill_bytes

    if chosen == "resident":
        resident_after = active_bytes + resident_bytes
        resident_peak = resident_after + workspace_bytes
        if resident_after > config.memory_budget_bytes:
            chosen = "host_backed"
            reason = "aggregate-budget-demotion"
        elif resident_peak > config.memory_budget_bytes:
            chosen = "host_backed"
            reason = "prefill-headroom-demotion"
        elif resident_after + future_kv_bytes > config.memory_budget_bytes:
            chosen = "host_backed"
            reason = "imminent-future-demotion"
        elif resident_peak + future_bytes > config.memory_budget_bytes:
            chosen = "host_backed"
            reason = "imminent-future-prefill-demotion"
    added = sparse_bytes if chosen == "host_backed" else resident_bytes
    after = active_bytes + added
    prefill_peak = after + workspace_bytes
    effective_after = after + future_kv_bytes
    effective_peak = prefill_peak + future_bytes
    return ElasticAdmissionDecision(
        request_id=submission.request_id,
        planned_residence=planned,
        chosen_residence=chosen,
        reason=reason,
        active_before_bytes=active_bytes,
        active_after_bytes=after,
        resident_bytes=resident_bytes,
        sparse_bytes=sparse_bytes,
        defer_for_budget=effective_peak > config.memory_budget_bytes,
        over_budget=after > config.memory_budget_bytes,
        future_reserved_bytes=future_kv_bytes,
        future_prefill_bytes=future_prefill_bytes,
        future_reserved_requests=future_count,
        future_effective_after_bytes=effective_after,
        future_over_budget=effective_after > config.memory_budget_bytes,
        prefill_workspace_bytes=workspace_bytes,
        prefill_peak_bytes=prefill_peak,
        future_effective_peak_bytes=effective_peak,
        prefill_over_budget=prefill_peak > config.memory_budget_bytes,
        future_peak_over_budget=effective_peak > config.memory_budget_bytes,
    )
