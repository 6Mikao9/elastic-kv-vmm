"""Live request-level residency decisions under an aggregate KV budget.

The offline planner chooses an optimistic mode from predicted runtimes.  This
module applies the same choice to the controller's observed in-flight set, so a
serving process can preserve full-GPU residency while load is light and demote
only new requests when active KV would cross the configured budget.
"""

from __future__ import annotations

from dataclasses import dataclass

from .future_plan import PlannerConfig, RolloutSubmission
from .sglang_future_hook import FutureHint


@dataclass(frozen=True, slots=True)
class ElasticKVConfig:
    """Conservative byte model used only by the rollout controller."""

    memory_budget_bytes: int
    kv_bytes_per_token: int
    sparse_hot_tokens: int
    auto_resident_output_tokens: int = 128

    def __post_init__(self) -> None:
        for name, value in (
            ("memory_budget_bytes", self.memory_budget_bytes),
            ("kv_bytes_per_token", self.kv_bytes_per_token),
            ("sparse_hot_tokens", self.sparse_hot_tokens),
            ("auto_resident_output_tokens", self.auto_resident_output_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def hold_bytes(self, submission: RolloutSubmission, residence: str) -> int:
        """Return the predicted per-rank KV bytes for one placement."""

        sequence_tokens = submission.prompt_tokens + submission.output_tokens
        held_tokens = (
            sequence_tokens
            if residence == "resident"
            else min(sequence_tokens, self.sparse_hot_tokens)
        )
        return held_tokens * self.kv_bytes_per_token


def elastic_config_from_planner(config: PlannerConfig) -> ElasticKVConfig:
    """Project the offline resource model into the live admission ledger."""

    return ElasticKVConfig(
        memory_budget_bytes=config.memory_budget_bytes,
        kv_bytes_per_token=config.kv_bytes_per_token,
        sparse_hot_tokens=config.sparse_hot_tokens,
        auto_resident_output_tokens=config.auto_resident_output_threshold,
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
        }


def decide_elastic_admission(
    submission: RolloutSubmission,
    *,
    active_bytes: int,
    config: ElasticKVConfig,
) -> ElasticAdmissionDecision:
    """Choose a mode and report whether admission should wait for capacity."""

    if isinstance(active_bytes, bool) or not isinstance(active_bytes, int):
        raise ValueError("active_bytes must be an integer")
    if active_bytes < 0:
        raise ValueError("active_bytes must be nonnegative")
    hint = FutureHint.from_payload(dict(submission.hint))
    planned = hint.residence
    resident_bytes = config.hold_bytes(submission, "resident")
    sparse_bytes = config.hold_bytes(submission, "host_backed")
    sequence_tokens = submission.prompt_tokens + submission.output_tokens

    if planned == "host_backed":
        chosen = "host_backed"
        reason = "planned-host-backed"
    elif sequence_tokens <= config.sparse_hot_tokens:
        chosen = "host_backed"
        reason = "fits-native-hot-set"
    elif planned == "auto":
        chosen = (
            "resident"
            if submission.output_tokens >= config.auto_resident_output_tokens
            else "host_backed"
        )
        reason = "auto-long" if chosen == "resident" else "auto-short"
    else:
        chosen = "resident"
        reason = "planned-resident"

    if chosen == "resident" and active_bytes + resident_bytes > config.memory_budget_bytes:
        chosen = "host_backed"
        reason = "aggregate-budget-demotion"
    added = sparse_bytes if chosen == "host_backed" else resident_bytes
    after = active_bytes + added
    return ElasticAdmissionDecision(
        request_id=submission.request_id,
        planned_residence=planned,
        chosen_residence=chosen,
        reason=reason,
        active_before_bytes=active_bytes,
        active_after_bytes=after,
        resident_bytes=resident_bytes,
        sparse_bytes=sparse_bytes,
        defer_for_budget=after > config.memory_budget_bytes,
        over_budget=after > config.memory_budget_bytes,
    )
