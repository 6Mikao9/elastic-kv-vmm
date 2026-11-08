"""Runtime value carrying for active-request residency decisions.

The future scheduler uses ``kv_hints`` to order waiting work.  The same
immutable hints remain attached to a request while it runs.  This module turns
that metadata into a small, deterministic retention value for the eager VMM
demotion policy.  It deliberately has no CUDA or SGLang runtime side effects.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any

DEFAULT_DEADLINE_GUARD_MS = 500


@dataclass(frozen=True, slots=True)
class RuntimeRequestValue:
    """Bounded business value retained by one active request.

    ``retention_score`` is used only as a tie-breaker under the existing
    pressure, dwell, byte-cap and joint-release rules.  The score is monotone
    in business weight, remaining decode work, imminent deadline pressure and
    potentially reusable shared-prefix bytes.  ``kind`` is preserved as
    evidence; it has no implicit priority because the planner already maps
    workload meaning into ``weight``.
    """

    request_id: str
    weight: float = 1.0
    deadline_remaining_ms: int | None = None
    remaining_output_tokens: int | None = None
    kind: str = "generic"
    prefix_group: str | None = None
    shared_prefix_tokens: int = 0
    active_reuse_consumers: int = 0
    future_reuse_consumers: int = 0
    hinted: bool = False

    def deadline_guarded(self, guard_ms: int = DEFAULT_DEADLINE_GUARD_MS) -> bool:
        if guard_ms < 0:
            raise ValueError("deadline guard must be nonnegative")
        return (
            self.deadline_remaining_ms is not None
            and self.deadline_remaining_ms <= guard_ms
        )

    def retention_score(self, guard_ms: int = DEFAULT_DEADLINE_GUARD_MS) -> float:
        """Return a bounded score where larger means retain on the GPU."""

        if guard_ms < 0:
            raise ValueError("deadline guard must be nonnegative")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("runtime request weight must be positive")

        # These logarithmic factors make metadata useful without letting a
        # single very large hint starve all ordinary requests indefinitely.
        weight_factor = 1.0 + min(math.log2(max(1.0, self.weight)), 8.0)
        remaining = max(0, self.remaining_output_tokens or 0)
        work_factor = 1.0 + min(math.log2(1.0 + remaining / 64.0), 8.0)
        if self.deadline_remaining_ms is None:
            deadline_factor = 1.0
        else:
            remaining_deadline = max(0, self.deadline_remaining_ms)
            deadline_factor = 1.0 + min(
                guard_ms / max(1.0, float(remaining_deadline)), 8.0
            )
        # A prefix length alone is not a benefit. Keeping the donor resident
        # only matters when another live or imminent host-backed request can
        # reuse it. The forecast only orders demotions; the runtime still checks
        # the complete token/adapter fingerprint before direct reuse.
        active_reuse = self.shared_prefix_tokens * self.active_reuse_consumers
        future_reuse = self.shared_prefix_tokens * self.future_reuse_consumers
        reusable_tokens = max(0, active_reuse, future_reuse)
        shared_factor = 1.0 + min(reusable_tokens / 2048.0, 4.0)
        return weight_factor * work_factor * deadline_factor * shared_factor


@dataclass(frozen=True, slots=True)
class PrefixReuseDemand:
    """Evidence that a resident request can avoid another request's fetch."""

    tokens: int
    active_consumers: int = 0
    future_consumers: int = 0

    def __post_init__(self) -> None:
        for name in ("tokens", "active_consumers", "future_consumers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")

    @property
    def total_consumers(self) -> int:
        return self.active_consumers + self.future_consumers


def attach_prefix_reuse(
    value: RuntimeRequestValue, demand: PrefixReuseDemand
) -> RuntimeRequestValue:
    """Return ``value`` annotated with live and forecast consumer evidence."""

    tokens = min(value.shared_prefix_tokens, demand.tokens)
    return replace(
        value,
        active_reuse_consumers=demand.active_consumers if tokens else 0,
        future_reuse_consumers=demand.future_consumers if tokens else 0,
    )


def forecast_prefix_reuse(
    requests: Iterable[Any],
) -> dict[tuple[str, int], int]:
    """Count potentially host-backed waiting consumers by identity geometry.

    Token/adapter identity is intentionally not hashed on every scheduler tick.
    This forecast only orders demotions, while the shared-prefix registry still
    requires an exact fingerprint before a direct Top-K reuse. Explicit resident
    requests are excluded; ``auto`` and ``host_backed`` hints remain candidates
    because pressure may force ``auto`` onto the native HiSparse path.
    """

    from .sglang_future_hook import _extract_hint

    demand: dict[tuple[str, int], int] = {}
    for request in requests:
        try:
            hint = _extract_hint(request)
        except (AttributeError, TypeError, ValueError):
            continue
        if hint is None or hint.residence == "resident":
            continue
        if hint.prefix_group is None or hint.shared_prefix_tokens < 1:
            continue
        key = (hint.prefix_group, hint.shared_prefix_tokens)
        demand[key] = demand.get(key, 0) + 1
    return demand


@dataclass(frozen=True, slots=True)
class DemotionRankValue:
    retention_score: float
    deadline_guarded: bool
    request_values: tuple[RuntimeRequestValue, ...]


def merge_demotion_values(
    values: Iterable[RuntimeRequestValue],
    *,
    guard_ms: int = DEFAULT_DEADLINE_GUARD_MS,
) -> DemotionRankValue:
    """Protect a request group when any member deserves protection."""

    materialized = tuple(values)
    if not materialized:
        return DemotionRankValue(1.0, False, ())
    return DemotionRankValue(
        max(value.retention_score(guard_ms) for value in materialized),
        any(value.deadline_guarded(guard_ms) for value in materialized),
        materialized,
    )


def rank_demotions(
    rank_candidates: Sequence[Mapping[Hashable, tuple[int, int]]],
    rank_values: Sequence[Mapping[Hashable, DemotionRankValue]] | None = None,
    *,
    guard_ms: int = DEFAULT_DEADLINE_GUARD_MS,
) -> list[Hashable]:
    """Rank request ids common to every TP rank for demotion.

    The existing worst-rank byte efficiency remains the primary control after
    the deadline guard tier.  Runtime value then lowers the priority of work
    that should stay resident.  Missing runtime metadata is neutral.
    """

    ranks = tuple(rank_candidates)
    if not ranks:
        return []
    common = set(ranks[0])
    for rank in ranks[1:]:
        common.intersection_update(rank)

    value_ranks = tuple(rank_values or ())
    merged_values: dict[Hashable, DemotionRankValue] = {}
    for request_id in common:
        observations = [
            rank[request_id] for rank in value_ranks if request_id in rank
        ]
        if not observations:
            merged_values[request_id] = DemotionRankValue(1.0, False, ())
            continue
        merged_values[request_id] = DemotionRankValue(
            max(value.retention_score for value in observations),
            any(value.deadline_guarded for value in observations),
            tuple(
                item
                for value in observations
                for item in value.request_values
            ),
        )

    def key(request_id: Hashable):
        reclaim = min(rank[request_id][0] for rank in ranks)
        transfer = max(1, max(rank[request_id][1] for rank in ranks))
        efficiency = Fraction(reclaim, transfer)
        value = merged_values[request_id]
        value_adjusted = efficiency / max(value.retention_score, 1e-12)
        return (value.deadline_guarded, -value_adjusted, str(request_id))

    return sorted(common, key=key)


def _request_id(request: Any) -> str | None:
    request_id = getattr(request, "rid", None)
    return request_id if isinstance(request_id, str) and request_id else None


def _remaining_output_tokens(request: Any, hint: Any) -> int | None:
    expected = getattr(hint, "expected_output_tokens", None)
    if expected is None:
        return None
    current = getattr(getattr(request, "kv", None), "kv_committed_len", None)
    prompt_ids = getattr(request, "origin_input_ids", None)
    if isinstance(current, int) and not isinstance(current, bool) and prompt_ids is not None:
        try:
            prompt = len(prompt_ids)
        except TypeError:
            prompt = 0
        generated = max(0, current - prompt)
        return max(0, expected - generated)
    return expected


def runtime_value_from_hint(request: Any, hint: Any, *, now_s: float) -> RuntimeRequestValue | None:
    """Convert one validated ``FutureHint`` into active-request metadata."""

    request_id = _request_id(request)
    if request_id is None:
        return None
    deadline = getattr(hint, "deadline_after_ms", None)
    if deadline is not None:
        from .sglang_future_hook import _request_age_ms

        try:
            age_ms = _request_age_ms(request, now_s)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "deadline hint requires a scheduler receive timestamp"
            ) from exc
        deadline = max(0, math.ceil(deadline - age_ms))
    return RuntimeRequestValue(
        request_id=request_id,
        weight=float(getattr(hint, "weight", 1.0)),
        deadline_remaining_ms=deadline,
        remaining_output_tokens=_remaining_output_tokens(request, hint),
        kind=str(getattr(hint, "kind", "generic")),
        prefix_group=getattr(hint, "prefix_group", None),
        shared_prefix_tokens=int(getattr(hint, "shared_prefix_tokens", 0) or 0),
        hinted=True,
    )


def runtime_values(
    requests: Iterable[Any], *, now_s: float
) -> tuple[dict[str, RuntimeRequestValue], dict[str, int]]:
    """Read live requests, returning neutral values on absent/invalid hints."""

    from .sglang_future_hook import _extract_hint

    values: dict[str, RuntimeRequestValue] = {}
    stats = {"hinted": 0, "invalid": 0}
    for request in requests:
        request_id = _request_id(request)
        if request_id is None:
            continue
        try:
            hint = _extract_hint(request)
        except (AttributeError, TypeError, ValueError):
            stats["invalid"] += 1
            values[request_id] = RuntimeRequestValue(request_id)
            continue
        if hint is None:
            values[request_id] = RuntimeRequestValue(request_id)
            continue
        try:
            value = runtime_value_from_hint(request, hint, now_s=now_s)
        except (AttributeError, TypeError, ValueError):
            stats["invalid"] += 1
            values[request_id] = RuntimeRequestValue(request_id)
            continue
        if value is not None:
            values[request_id] = value
            stats["hinted"] += 1
    return values, stats


def evidence(value: RuntimeRequestValue | None, *, guard_ms: int) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "weight": value.weight,
        "deadline_remaining_ms": value.deadline_remaining_ms,
        "remaining_output_tokens": value.remaining_output_tokens,
        "kind": value.kind,
        "prefix_group": value.prefix_group,
        "shared_prefix_tokens": value.shared_prefix_tokens,
        "active_reuse_consumers": value.active_reuse_consumers,
        "future_reuse_consumers": value.future_reuse_consumers,
        "deadline_guarded": value.deadline_guarded(guard_ms),
        "retention_score": value.retention_score(guard_ms),
        "hinted": value.hinted,
    }
