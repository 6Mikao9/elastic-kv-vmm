"""Live request-level residency decisions under an aggregate KV budget.

The offline planner chooses an optimistic mode from predicted runtimes.  This
module applies the same choice to the controller's observed in-flight set, so a
serving process can preserve full-GPU residency while load is light and demote
only new requests when active KV would cross the configured budget.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .future_plan import (
    KindServiceProfile,
    PlannerConfig,
    RolloutSubmission,
    normalize_kind_profiles,
)
from .service_time import estimate_service_duration_ms
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
    kind_profiles: (
        tuple[tuple[str, KindServiceProfile], ...]
        | Mapping[str, KindServiceProfile | Mapping[str, object]]
    ) = ()

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
        object.__setattr__(
            self, "kind_profiles", normalize_kind_profiles(self.kind_profiles)
        )

    def profile_for(self, kind: str) -> KindServiceProfile:
        for candidate, profile in self.kind_profiles:
            if candidate == kind:
                return profile
        return KindServiceProfile()

    def sparse_hot_tokens_for(self, kind: str) -> int:
        override = self.profile_for(kind).sparse_hot_tokens
        return self.sparse_hot_tokens if override is None else override

    def prefill_workspace_rate(self, kind: str) -> int:
        override = self.profile_for(kind).prefill_workspace_bytes_per_token
        if override is None:
            return self.prefill_workspace_bytes_per_token
        return override

    def auto_resident_threshold(self, kind: str) -> int:
        override = self.profile_for(kind).auto_resident_output_threshold
        return self.auto_resident_output_tokens if override is None else override

    def prefill_rate_tokens_per_s(self, kind: str) -> float | None:
        override = self.profile_for(kind).prefill_tokens_per_ms
        return None if override is None else override * 1000.0

    def decode_rate_tokens_per_s(
        self, kind: str, residence: str
    ) -> float | None:
        profile = self.profile_for(kind)
        override = (
            profile.resident_decode_tokens_per_ms
            if residence == "resident"
            else profile.sparse_decode_tokens_per_ms
        )
        return None if override is None else override * 1000.0

    def hold_bytes(self, submission: RolloutSubmission, residence: str) -> int:
        """Return the predicted per-rank KV bytes for one placement."""

        sequence_tokens = submission.prompt_tokens + submission.output_tokens
        held_tokens = (
            sequence_tokens
            if residence == "resident"
            else min(sequence_tokens, self.sparse_hot_tokens_for(submission.kind))
        )
        return held_tokens * self.kv_bytes_per_token

    def prefill_workspace_bytes(self, submission: RolloutSubmission) -> int:
        return (
            submission.prompt_tokens
            * self.prefill_workspace_rate(submission.kind)
        )


def elastic_config_from_planner(config: PlannerConfig) -> ElasticKVConfig:
    """Project the offline resource model into the live admission ledger."""

    return ElasticKVConfig(
        memory_budget_bytes=config.memory_budget_bytes,
        kv_bytes_per_token=config.kv_bytes_per_token,
        sparse_hot_tokens=config.sparse_hot_tokens,
        prefill_workspace_bytes_per_token=config.prefill_workspace_bytes_per_token,
        auto_resident_output_tokens=config.auto_resident_output_threshold,
        lookahead_ms=config.lookahead_ms,
        kind_profiles=config.kind_profiles,
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
    service_time_feedback_applied: bool = False
    service_time_current_duration_ms: float = 0.0
    service_time_calibrated_future_requests: int = 0

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
            "service_time_feedback_applied": self.service_time_feedback_applied,
            "service_time_current_duration_ms": (
                self.service_time_current_duration_ms
            ),
            "service_time_calibrated_future_requests": (
                self.service_time_calibrated_future_requests
            ),
        }


def _preferred_residence(
    submission: RolloutSubmission, config: ElasticKVConfig
) -> tuple[str, str]:
    """Return the placement implied by one request before live contention."""

    hint = FutureHint.from_payload(dict(submission.hint))
    sequence_tokens = submission.prompt_tokens + submission.output_tokens
    if hint.residence == "host_backed":
        return "host_backed", "planned-host-backed"
    if sequence_tokens <= config.sparse_hot_tokens_for(submission.kind):
        return "host_backed", "fits-native-hot-set"
    if hint.residence == "auto":
        if submission.output_tokens >= config.auto_resident_threshold(
            submission.kind
        ):
            return "resident", "auto-long"
        return "host_backed", "auto-short"
    return "resident", "planned-resident"


def _predicted_duration_ms(
    submission: RolloutSubmission,
    hint: FutureHint,
    *,
    fallback_ms: float,
    prefill_tokens_per_s: float | None,
    decode_tokens_per_s: float | None,
    config: ElasticKVConfig,
    residence: str,
) -> float:
    return estimate_service_duration_ms(
        prompt_tokens=submission.prompt_tokens,
        output_tokens=submission.output_tokens,
        fallback_ms=fallback_ms,
        prefill_tokens_per_s=(
            prefill_tokens_per_s
            if prefill_tokens_per_s is not None
            else (
                hint.prefill_tokens_per_s
                if hint.prefill_tokens_per_s is not None
                else config.prefill_rate_tokens_per_s(submission.kind)
            )
        ),
        decode_tokens_per_s=(
            decode_tokens_per_s
            if decode_tokens_per_s is not None
            else (
                hint.decode_tokens_per_s
                if hint.decode_tokens_per_s is not None
                else config.decode_rate_tokens_per_s(
                    submission.kind, residence
                )
            )
        ),
    )


def _future_reservation(
    submission: RolloutSubmission,
    future_submissions: Iterable[RolloutSubmission],
    *,
    now_ms: float,
    config: ElasticKVConfig,
    current_duration_ms: float,
    prefill_tokens_per_s: float | None,
    decode_tokens_per_s: float | None,
) -> tuple[int, int, int, int]:
    """Reserve bounded capacity for more urgent requests that can overlap now.

    The executor receives a known rollout schedule. Counting every queued
    request would serialize the whole trace, so this guard only includes
    requests that may start before the current request finishes, arrive inside
    the configured horizon, and carry a deadline or a strictly larger soft
    business weight. A deadline is protected only when waiting behind the
    current request would consume otherwise feasible slack. A fixed request cap
    bounds the search and the effect of an unusually dense trace.

    The returned bytes are interval peaks, not sums over the whole horizon.
    Two requests that appear and finish at different times do not need two
    simultaneous reservations merely because both lie before the current
    request finishes. This reduces false demotions while retaining the
    controller's conservative assumption that prefill workspace remains
    charged until a request completes.
    """

    if config.lookahead_ms == 0:
        return 0, 0, 0, 0
    current_hint = FutureHint.from_payload(dict(submission.hint))
    current_finish_ms = now_ms + current_duration_ms
    horizon_end_ms = now_ms + config.lookahead_ms
    reservations: list[tuple[int, int, int, int, str | None, int]] = []
    calibrated_future_requests = 0
    for future in future_submissions:
        if future.request_id == submission.request_id:
            continue
        if future.submit_at_ms > horizon_end_ms:
            continue
        if future.submit_at_ms > current_finish_ms:
            continue
        future_hint = FutureHint.from_payload(dict(future.hint))
        if future_hint.shared_prefix_tokens > future.prompt_tokens:
            raise ValueError(
                f"{future.request_id}: shared prefix exceeds prompt length"
            )
        has_deadline = future_hint.deadline_after_ms is not None
        has_higher_weight = future_hint.weight > current_hint.weight
        if not has_deadline and not has_higher_weight:
            continue
        placement, _ = _preferred_residence(future, config)
        predicted_duration_ms = _predicted_duration_ms(
            future,
            future_hint,
            fallback_ms=max(
                1, future.estimated_finish_ms - future.submit_at_ms
            ),
            prefill_tokens_per_s=prefill_tokens_per_s,
            decode_tokens_per_s=decode_tokens_per_s,
            config=config,
            residence=placement,
        )
        profile = config.profile_for(future.kind)
        profile_residence_rate = (
            profile.resident_decode_tokens_per_ms
            if placement == "resident"
            else profile.sparse_decode_tokens_per_ms
        )
        if (
            future_hint.prefill_tokens_per_s is not None
            or future_hint.decode_tokens_per_s is not None
            or prefill_tokens_per_s is not None
            or decode_tokens_per_s is not None
            or profile.prefill_tokens_per_ms is not None
            or profile_residence_rate is not None
        ):
            calibrated_future_requests += 1
        deadline_needs_protection = has_deadline
        if has_deadline and future.deadline_ms is not None:
            earliest_finish_ms = (
                max(now_ms, future.submit_at_ms) + predicted_duration_ms
            )
            deadline_needs_protection = (
                earliest_finish_ms <= future.deadline_ms
                and current_finish_ms + predicted_duration_ms >= future.deadline_ms
            )
        if not has_higher_weight and not deadline_needs_protection:
            # A deadline with positive slack after the current request, or one
            # that is already infeasible even when started immediately, should
            # not demote unrelated work. The weight signal remains independent.
            continue
        start_ms = max(now_ms, future.submit_at_ms)
        # A queued request may have crossed its original estimated finish
        # before it was admitted. Preserve its predicted remaining service time
        # instead of creating an interval whose finish precedes its start.
        finish_ms = min(current_finish_ms, start_ms + predicted_duration_ms)
        if finish_ms <= start_ms:
            continue
        reservations.append(
            (
                start_ms,
                finish_ms,
                config.hold_bytes(future, placement),
                config.prefill_workspace_bytes(future),
                future_hint.prefix_group if placement == "resident" else None,
                future_hint.shared_prefix_tokens if placement == "resident" else 0,
            )
        )
        if len(reservations) >= config.max_future_reservations:
            break

    if not reservations:
        return 0, 0, 0, calibrated_future_requests

    events: dict[int, tuple[list[int], list[int]]] = {}
    for index, (start_ms, finish_ms, *_rest) in enumerate(reservations):
        starts = events.setdefault(start_ms, ([], []))[0]
        finishes = events.setdefault(finish_ms, ([], []))[1]
        starts.append(index)
        finishes.append(index)

    active: set[int] = set()
    peak_kv_bytes = 0
    peak_prefill_bytes = 0
    for event_ms in sorted(events):
        starts, finishes = events[event_ms]
        # Intervals are half-open. A request that finishes as another starts
        # does not overlap it.
        active.difference_update(finishes)
        active.update(starts)
        kv_bytes = sum(reservations[index][2] for index in active)
        resident_prefixes: dict[str, list[int]] = {}
        for index in active:
            _, _, _, _, prefix_group, shared_tokens = reservations[index]
            if prefix_group is not None and shared_tokens > 0:
                resident_prefixes.setdefault(prefix_group, []).append(shared_tokens)
        shared_credit = sum(
            (len(lengths) - 1) * min(lengths) * config.kv_bytes_per_token
            for lengths in resident_prefixes.values()
            if len(lengths) > 1
        )
        peak_kv_bytes = max(peak_kv_bytes, kv_bytes - shared_credit)
        peak_prefill_bytes = max(
            peak_prefill_bytes,
            sum(reservations[index][3] for index in active),
        )

    return (
        peak_kv_bytes,
        peak_prefill_bytes,
        len(reservations),
        calibrated_future_requests,
    )


def decide_elastic_admission(
    submission: RolloutSubmission,
    *,
    active_bytes: int,
    config: ElasticKVConfig,
    future_submissions: Iterable[RolloutSubmission] = (),
    now_ms: float | None = None,
    prefill_tokens_per_s: float | None = None,
    decode_tokens_per_s: float | None = None,
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
    if hint.shared_prefix_tokens > submission.prompt_tokens:
        raise ValueError(
            f"{submission.request_id}: shared prefix exceeds prompt length"
        )
    planned = hint.residence
    resident_bytes = config.hold_bytes(submission, "resident")
    sparse_bytes = config.hold_bytes(submission, "host_backed")
    workspace_bytes = config.prefill_workspace_bytes(submission)
    chosen, reason = _preferred_residence(submission, config)
    static_duration_ms = max(
        1, submission.estimated_finish_ms - submission.submit_at_ms
    )
    current_duration_ms = _predicted_duration_ms(
        submission,
        hint,
        fallback_ms=static_duration_ms,
        prefill_tokens_per_s=prefill_tokens_per_s,
        decode_tokens_per_s=decode_tokens_per_s,
        config=config,
        residence=chosen,
    )
    profile = config.profile_for(submission.kind)
    profile_residence_rate = (
        profile.resident_decode_tokens_per_ms
        if chosen == "resident"
        else profile.sparse_decode_tokens_per_ms
    )
    feedback_applied = bool(
        hint.prefill_tokens_per_s is not None
        or hint.decode_tokens_per_s is not None
        or prefill_tokens_per_s is not None
        or decode_tokens_per_s is not None
        or profile.prefill_tokens_per_ms is not None
        or profile_residence_rate is not None
    )
    (
        future_kv_bytes,
        future_prefill_bytes,
        future_count,
        calibrated_future_requests,
    ) = _future_reservation(
        submission,
        queued,
        now_ms=now_ms,
        config=config,
        current_duration_ms=current_duration_ms,
        prefill_tokens_per_s=prefill_tokens_per_s,
        decode_tokens_per_s=decode_tokens_per_s,
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
        service_time_feedback_applied=feedback_applied,
        service_time_current_duration_ms=current_duration_ms,
        service_time_calibrated_future_requests=calibrated_future_requests,
    )
