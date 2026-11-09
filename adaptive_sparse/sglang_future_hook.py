"""Opt-in SGLang ordering hook for controlled, known-future rollouts.

This uses SGLang's native ``kv_hints`` envelope when available and a narrow
transport shim on 0.5.2. It only reorders the waiting queue after SGLang's
native policy has run; ``PrefillAdder`` and all native memory, grammar, LoRA,
and request-slot checks remain authoritative. The hook is disabled by default.
"""

from __future__ import annotations

import math
import os
import time
from collections import OrderedDict, deque
from collections.abc import Iterable
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any

from .service_time import (
    DEFAULT_DECODE_TOKENS_PER_S,
    DEFAULT_PREFILL_TOKENS_PER_S,
    MAX_SERVICE_RATE,
    MIN_SERVICE_RATE,
)

ACTION_TYPE = "elastic.future"
ACTION_VERSION = "4"
RATE_ACTION_VERSION = "3"
LEGACY_ACTION_VERSION = "2"
SUPPORTED_ACTION_VERSIONS = frozenset(
    ("1", LEGACY_ACTION_VERSION, RATE_ACTION_VERSION, ACTION_VERSION)
)
POLICY_ENV = "EKV_FUTURE_POLICY"
STATS_PATH_ENV = "EKV_FUTURE_STATS_PATH"
HORIZON_ENV = "EKV_FUTURE_ORDERING_HORIZON_MS"
COHORT_WAIT_TIMEOUT_ENV = "EKV_COHORT_WAIT_TIMEOUT_MS"
HARD_RELEASE_GATE_ENV = "EKV_HARD_RELEASE_GATE"
POLICIES = ("off", "deadline")
_STATS_INTERVAL_S = 0.25
_DEFAULT_COHORT_WAIT_TIMEOUT_MS = 1000.0
_MAX_DECODE_REQUEST_TRACES = 4096
_MAX_DECODE_TRACE_EVENTS = 4096

_installed = False
_last_stats_publish = 0.0
_stats = {
    "ticks": 0,
    "reordered_ticks": 0,
    "queue_requests": 0,
    "released_requests": 0,
    "unreleased_requests": 0,
    "fallback_missing_hint": 0,
    "fallback_invalid_hint": 0,
    "fallback_horizon": 0,
    "fallback_exception": 0,
    "telemetry_errors": 0,
    "prefill_batches": 0,
    "decode_batches": 0,
    "decode_requests_observed": 0,
    "decode_mixed_batches": 0,
    "decode_request_trace_evictions": 0,
    "decode_trace_evictions": 0,
    "cohort_batches": 0,
    "complete_cohort_batches": 0,
    "partial_cohort_batches": 0,
    "cohort_members_observed": 0,
    "cohort_gate_ticks": 0,
    "cohort_gate_holds": 0,
    "cohort_members_held": 0,
    "cohort_gate_timeouts": 0,
    "cohort_metadata_conflicts": 0,
    "cohort_gate_errors": 0,
    "hard_release_gate_ticks": 0,
    "hard_release_gate_holds": 0,
    "hard_release_members_held": 0,
    "hard_release_gate_errors": 0,
}
_last_cohort_batch: dict[str, object] | None = None
_complete_cohort_ids: set[str] = set()
_last_decode_batch: dict[str, object] | None = None
_decode_batch_sizes: dict[int, int] = {}
_decode_batch_sizes_by_request: OrderedDict[str, dict[int, int]] = OrderedDict()
_decode_trace: deque[dict[str, object]] = deque(maxlen=_MAX_DECODE_TRACE_EVENTS)
_last_held_cohorts: list[dict[str, object]] = []
_last_held_releases: list[dict[str, object]] = []
_cohort_first_seen_s: dict[str, float] = {}
_cohort_arrival_complete: set[str] = set()
_cohort_timed_out: dict[str, tuple[int, tuple[int, ...]]] = {}
_stats_identity: dict[str, int] = {}
_hard_release_gate = False


@dataclass(slots=True)
class _PrefillGateFrame:
    ordered_queue: list[Any] | None = None
    held: list[Any] = field(default_factory=list)


_active_prefill_gate: ContextVar[_PrefillGateFrame | None] = ContextVar(
    "elastic_prefill_gate", default=None
)


@dataclass(frozen=True, slots=True)
class FutureHint:
    release_after_ms: int = 0
    release_clock: str | None = None
    release_absolute_ms: int | None = None
    deadline_after_ms: int | None = None
    deadline_clock: str | None = None
    deadline_absolute_ms: int | None = None
    expected_output_tokens: int | None = None
    weight: float = 1.0
    kind: str = "generic"
    residence: str | None = None
    prefix_group: str | None = None
    shared_prefix_tokens: int = 0
    prefill_tokens_per_s: float | None = None
    decode_tokens_per_s: float | None = None
    cohort_id: str | None = None
    cohort_size: int = 1

    @classmethod
    def from_payload(cls, payload: Any) -> FutureHint:
        if not isinstance(payload, dict):
            raise ValueError("elastic.future payload must be an object")  # noqa: TRY004
        allowed = {
            "release_after_ms",
            "release_clock",
            "release_absolute_ms",
            "deadline_after_ms",
            "deadline_clock",
            "deadline_absolute_ms",
            "expected_output_tokens",
            "weight",
            "kind",
            "residence",
            "prefix_group",
            "shared_prefix_tokens",
            "prefill_tokens_per_s",
            "decode_tokens_per_s",
            "cohort_id",
            "cohort_size",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError("elastic.future payload has unknown keys")
        release = _bounded_int(payload.get("release_after_ms", 0), "release_after_ms")
        release_clock = payload.get("release_clock")
        release_absolute_raw = payload.get("release_absolute_ms")
        if (release_clock is None) != (release_absolute_raw is None):
            raise ValueError(
                "release_clock and release_absolute_ms must be supplied together"
            )
        if release_clock is not None:
            if release_clock != "perf_counter":
                raise ValueError("unsupported release_clock")
            release_absolute = _bounded_int64(
                release_absolute_raw, "release_absolute_ms"
            )
        else:
            release_absolute = None
        deadline_raw = payload.get("deadline_after_ms")
        deadline = None if deadline_raw is None else _bounded_int(deadline_raw, "deadline_after_ms")
        if deadline is not None and deadline < release:
            raise ValueError("deadline_after_ms precedes release_after_ms")
        deadline_clock = payload.get("deadline_clock")
        deadline_absolute_raw = payload.get("deadline_absolute_ms")
        if (deadline_clock is None) != (deadline_absolute_raw is None):
            raise ValueError(
                "deadline_clock and deadline_absolute_ms must be supplied together"
            )
        if deadline_clock is not None:
            if deadline is None:
                raise ValueError(
                    "absolute deadline requires deadline_after_ms for compatibility"
                )
            if deadline_clock != "perf_counter":
                raise ValueError("unsupported deadline_clock")
            deadline_absolute = _bounded_int64(
                deadline_absolute_raw, "deadline_absolute_ms"
            )
            if (
                release_absolute is not None
                and deadline_absolute < release_absolute
            ):
                raise ValueError("absolute deadline precedes absolute release")
        else:
            deadline_absolute = None
        expected_raw = payload.get("expected_output_tokens")
        expected = (
            None
            if expected_raw is None
            else _bounded_int(expected_raw, "expected_output_tokens", minimum=1)
        )
        weight = payload.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("weight must be numeric")  # noqa: TRY004
        weight = float(weight)
        if not math.isfinite(weight) or not 0 < weight <= 1e6:
            raise ValueError("weight must be finite and in (0, 1e6]")
        kind = payload.get("kind", "generic")
        if not isinstance(kind, str) or not kind or len(kind) > 64:
            raise ValueError("kind must be a short non-empty string")
        residence = payload.get("residence")
        if residence is not None and residence not in ("auto", "resident", "host_backed"):
            raise ValueError("invalid residence hint")
        prefix_group = payload.get("prefix_group")
        if prefix_group is not None and (
            not isinstance(prefix_group, str)
            or not prefix_group
            or len(prefix_group) > 64
        ):
            raise ValueError("prefix_group must be a short non-empty string")
        shared_prefix_tokens = _bounded_int(
            payload.get("shared_prefix_tokens", 0),
            "shared_prefix_tokens",
        )
        if shared_prefix_tokens and prefix_group is None:
            raise ValueError("prefix_group is required for shared_prefix_tokens")
        prefill_tokens_per_s = _optional_rate(
            payload.get("prefill_tokens_per_s"), "prefill_tokens_per_s"
        )
        decode_tokens_per_s = _optional_rate(
            payload.get("decode_tokens_per_s"), "decode_tokens_per_s"
        )
        cohort_id = payload.get("cohort_id")
        cohort_size = _bounded_int(
            payload.get("cohort_size", 1), "cohort_size", minimum=1
        )
        if cohort_id is None:
            if cohort_size != 1:
                raise ValueError("cohort_size greater than 1 requires cohort_id")
        elif (
            not isinstance(cohort_id, str)
            or not cohort_id
            or len(cohort_id) > 128
            or cohort_size < 2
        ):
            raise ValueError(
                "cohort_id must be a short non-empty string and requires cohort_size >= 2"
            )
        return cls(
            release_after_ms=release,
            release_clock=release_clock,
            release_absolute_ms=release_absolute,
            deadline_after_ms=deadline,
            deadline_clock=deadline_clock,
            deadline_absolute_ms=deadline_absolute,
            expected_output_tokens=expected,
            weight=weight,
            kind=kind,
            residence=residence,
            prefix_group=prefix_group,
            shared_prefix_tokens=shared_prefix_tokens,
            prefill_tokens_per_s=prefill_tokens_per_s,
            decode_tokens_per_s=decode_tokens_per_s,
            cohort_id=cohort_id,
            cohort_size=cohort_size,
        )


def _bounded_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")  # noqa: TRY004
    if not minimum <= value <= 86_400_000:
        raise ValueError(f"{name} is outside the supported range")
    return value


def _bounded_int64(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")  # noqa: TRY004
    if not 0 <= value <= (1 << 63) - 1:
        raise ValueError(f"{name} is outside the supported range")
    return value


def _optional_rate(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not MIN_SERVICE_RATE <= float(value) <= MAX_SERVICE_RATE
    ):
        raise ValueError(
            f"{name} must be finite and in [{MIN_SERVICE_RATE}, {MAX_SERVICE_RATE}]"
        )
    return float(value)


def _extract_hint(request: Any) -> FutureHint | None:
    envelope = getattr(request, "kv_hints", None)
    actions = getattr(envelope, "actions", None)
    if not actions:
        return None
    matches = [
        action
        for action in actions
        if getattr(action, "action_type", None) == ACTION_TYPE
    ]
    if len(matches) != 1:
        raise ValueError("request must carry exactly one elastic.future action")
    action = matches[0]
    action_version = getattr(action, "action_version", None)
    if action_version not in SUPPORTED_ACTION_VERSIONS:
        raise ValueError("unsupported elastic.future action version")
    hint = FutureHint.from_payload(getattr(action, "payload", None))
    if action_version == "1" and hint.deadline_clock is not None:
        raise ValueError("action version 1 cannot carry an absolute deadline clock")
    if action_version not in (RATE_ACTION_VERSION, ACTION_VERSION) and (
        hint.prefill_tokens_per_s is not None
        or hint.decode_tokens_per_s is not None
    ):
        raise ValueError("service-time rates require action version 3")
    if action_version != ACTION_VERSION and hint.cohort_id is not None:
        raise ValueError("cohort metadata requires action version 4")
    return hint


def _request_age_ms(request: Any, now_s: float) -> float:
    stats = getattr(request, "time_stats", None)
    received = getattr(stats, "scheduler_recv_time", None)
    if received in (None, 0.0):
        received = getattr(stats, "wait_queue_entry_time", None)
    if received in (None, 0.0):
        received = getattr(request, "_ekv_scheduler_recv_time", None)
    if not isinstance(received, (int, float)) or not math.isfinite(received) or received <= 0:
        raise ValueError("request scheduler receive time is unavailable")
    return max(0.0, (now_s - float(received)) * 1000.0)


def deadline_remaining_ms(
    hint: FutureHint,
    *,
    now_s: float | None = None,
    age_ms: float | None = None,
) -> int | None:
    """Resolve the remaining deadline without confusing dispatch and receipt clocks."""
    if hint.deadline_after_ms is None:
        return None
    if hint.deadline_clock == "perf_counter":
        if now_s is None or not math.isfinite(now_s):
            raise ValueError("absolute deadline requires a finite perf_counter time")
        return max(0, math.ceil(hint.deadline_absolute_ms - now_s * 1000.0))
    if age_ms is None or not math.isfinite(age_ms):
        raise ValueError("relative deadline requires request age")
    return max(0, math.ceil(hint.deadline_after_ms - age_ms))


def release_remaining_ms(
    hint: FutureHint,
    *,
    now_s: float | None = None,
    age_ms: float | None = None,
) -> float:
    """Resolve release time against the dispatch clock when available."""
    if hint.release_clock == "perf_counter":
        if now_s is None or not math.isfinite(now_s):
            raise ValueError("absolute release requires a finite perf_counter time")
        return hint.release_absolute_ms - now_s * 1000.0
    if age_ms is None or not math.isfinite(age_ms):
        raise ValueError("relative release requires request age")
    return hint.release_after_ms - age_ms


def _prompt_tokens(request: Any) -> int:
    ids = getattr(request, "origin_input_ids", None)
    if ids is None:
        raise ValueError("request prompt length is unavailable")
    try:
        total = len(ids)
    except TypeError as exc:
        raise ValueError("request prompt ids are not a sequence") from exc
    if total < 1:
        raise ValueError("request prompt cannot be empty")
    cached = getattr(request, "num_matched_prefix_tokens", 0) or 0
    if isinstance(cached, bool) or not isinstance(cached, int) or cached < 0:
        raise ValueError("matched prefix length is invalid")
    return max(1, total - min(cached, total))


def order_requests(
    requests: Iterable[Any],
    *,
    now_s: float,
    prefill_tokens_per_s: float = DEFAULT_PREFILL_TOKENS_PER_S,
    decode_tokens_per_s: float = DEFAULT_DECODE_TOKENS_PER_S,
    priority_sign: int | None = None,
    horizon_ms: int | None = None,
) -> list[Any]:
    """Return a stable deadline-aware order or the exact native order.

    All requests in the queue must carry a valid hint.  A mixed queue falls
    back without side effects, so enabling the feature cannot reorder legacy
    traffic that has no planning metadata.
    """

    queue = list(requests)
    if not queue:
        return queue
    prefill_tokens_per_s = _optional_rate(
        prefill_tokens_per_s, "prefill_tokens_per_s"
    )
    decode_tokens_per_s = _optional_rate(
        decode_tokens_per_s, "decode_tokens_per_s"
    )
    if prefill_tokens_per_s is None or decode_tokens_per_s is None:
        raise ValueError("default service-time rates must be finite and positive")
    decorated_hints = []
    for request in queue:
        hint = _extract_hint(request)
        if hint is None:
            _stats["fallback_missing_hint"] += 1
            return queue
        try:
            age_ms = _request_age_ms(request, now_s)
        except (AttributeError, TypeError, ValueError):
            if (
                hint.release_after_ms != 0
                or horizon_ms is not None
                or (
                    hint.deadline_after_ms is not None
                    and hint.deadline_clock != "perf_counter"
                )
            ):
                raise
            age_ms = 0.0
        release_in_ms = release_remaining_ms(hint, now_s=now_s, age_ms=age_ms)
        if horizon_ms is not None and release_in_ms > horizon_ms:
            _stats["fallback_horizon"] += 1
            return queue
        decorated_hints.append((hint, age_ms))

    decorated = []
    for native_order, (request, (hint, age_ms)) in enumerate(
        zip(queue, decorated_hints)
    ):
        release_in_ms = release_remaining_ms(hint, now_s=now_s, age_ms=age_ms)
        released = release_in_ms <= 0
        request_prefill_rate = (
            hint.prefill_tokens_per_s or prefill_tokens_per_s
        )
        request_decode_rate = hint.decode_tokens_per_s or decode_tokens_per_s
        prefill_ms = 1000.0 * _prompt_tokens(request) / request_prefill_rate
        if hint.deadline_after_ms is not None and hint.expected_output_tokens is None:
            raise ValueError("deadline planning requires expected_output_tokens")
        decode_ms = (
            1000.0
            * (hint.expected_output_tokens or 1)
            / request_decode_rate
        )
        estimated_request_ms = prefill_ms + decode_ms
        remaining_deadline_ms = deadline_remaining_ms(
            hint, now_s=now_s, age_ms=age_ms
        )
        latest_start = (
            float("inf")
            if remaining_deadline_ms is None
            else remaining_deadline_ms - estimated_request_ms
        )
        residency_rank = {
            "resident": 0,
            None: 1,
            "auto": 1,
            "host_backed": 2,
        }[hint.residence]
        decorated.append(
            (
                0 if released else 1,
                0 if hint.deadline_after_ms is not None else 1,
                latest_start,
                -hint.weight,
                residency_rank,
                estimated_request_ms,
                release_in_ms,
                native_order,
                request,
            )
        )

    if priority_sign is None:
        groups = [decorated]
    else:
        groups = []
        current = []
        current_priority = None
        for item in decorated:
            priority = getattr(item[-1], "priority", None)
            if (
                isinstance(priority, bool)
                or not isinstance(priority, (int, float))
                or not math.isfinite(priority)
            ):
                raise ValueError("native priority scheduling has an invalid request priority")
            if current and priority != current_priority:
                groups.append(current)
                current = []
            current_priority = priority
            current.append(item)
        if current:
            groups.append(current)

    for group in groups:
        # A release hint is a soft admission preference. If no request in this
        # native-priority tier is released, retain the exact native order.
        if all(item[0] for item in group):
            continue
        group.sort(key=lambda item: item[:-1])
    result = [item[-1] for group in groups for item in group]
    if result != queue:
        _stats["reordered_ticks"] += 1
    return result


def apply_to_waiting_queue(
    waiting_queue: list[Any],
    *,
    now_s: float | None = None,
    priority_sign: int | None = None,
    horizon_ms: int | None = None,
) -> None:
    """Mutate a queue only after a complete successful planning pass."""
    _stats["ticks"] += 1
    _stats["queue_requests"] = len(waiting_queue)
    try:
        planned = order_requests(
            waiting_queue,
            now_s=time.perf_counter() if now_s is None else now_s,
            priority_sign=priority_sign,
            horizon_ms=horizon_ms,
        )
        planned_hints = [_extract_hint(request) for request in planned]
        if all(hint is not None for hint in planned_hints):
            clock_now = time.perf_counter() if now_s is None else now_s
            released = sum(
                release_remaining_ms(
                    hint,
                    now_s=clock_now,
                    age_ms=_request_age_ms(request, clock_now),
                )
                <= 0
                for request, hint in zip(planned, planned_hints)
            )
            _stats["released_requests"] = released
            _stats["unreleased_requests"] = len(planned) - released
        else:
            # Missing hints preserve native order and are already counted as a
            # fallback; release telemetry is meaningful only for an all-hint queue.
            _stats["released_requests"] = 0
            _stats["unreleased_requests"] = 0
    except (TypeError, ValueError):
        _stats["fallback_invalid_hint"] += 1
        _publish_stats()
        return
    except Exception:  # noqa: BLE001 - hint failures must preserve native scheduling
        _stats["fallback_exception"] += 1
        _publish_stats()
        return
    waiting_queue[:] = planned
    _publish_stats()


def withhold_unreleased_requests(
    waiting_queue: list[Any],
    *,
    now_s: float | None = None,
) -> list[Any]:
    """Hold preannounced hints outside native prefill until their release time."""

    global _last_held_releases
    if not waiting_queue:
        _last_held_releases = []
        return []
    now = time.perf_counter() if now_s is None else now_s
    if not math.isfinite(now):
        raise ValueError("hard release clock must be finite")

    held_members: list[Any] = []
    held_records: list[dict[str, object]] = []
    for request in waiting_queue:
        try:
            hint = _extract_hint(request)
            if hint is None:
                continue
            age_ms = _request_age_ms(request, now)
        except (AttributeError, TypeError, ValueError):
            _stats["hard_release_gate_errors"] += 1
            continue
        remaining_ms = release_remaining_ms(hint, now_s=now, age_ms=age_ms)
        if remaining_ms <= 0:
            continue
        held_members.append(request)
        held_records.append(
            {
                "request_id": str(
                    getattr(request, "request_id", "<missing>")
                ),
                "release_after_ms": hint.release_after_ms,
                "age_ms": round(age_ms, 3),
                "remaining_ms": round(remaining_ms, 3),
            }
        )

    _stats["hard_release_gate_ticks"] += 1
    _last_held_releases = held_records
    if not held_members:
        return []
    _stats["hard_release_gate_holds"] += 1
    _stats["hard_release_members_held"] += len(held_members)
    held_ids = {id(member) for member in held_members}
    waiting_queue[:] = [
        member for member in waiting_queue if id(member) not in held_ids
    ]
    return held_members


def observe_prefill_batch(batch: Any) -> None:
    """Record whether planned cohorts landed together in a real prefill batch."""

    global _last_cohort_batch
    batch = _prefill_batch(batch)
    if batch is None:
        return
    _stats["prefill_batches"] += 1
    requests = getattr(batch, "reqs", None) or ()
    cohorts: dict[str, list[Any]] = {}
    for request in requests:
        try:
            hint = _extract_hint(request)
        except (AttributeError, TypeError, ValueError):
            continue
        if hint is not None and hint.cohort_id is not None:
            cohorts.setdefault(hint.cohort_id, []).append(request)
    for cohort_id, members in cohorts.items():
        hints = [_extract_hint(member) for member in members]
        expected_sizes = {
            hint.cohort_size for hint in hints if hint is not None
        }
        expected = next(iter(expected_sizes)) if len(expected_sizes) == 1 else None
        observed = len(members)
        complete = expected is not None and observed == expected
        _stats["cohort_batches"] += 1
        _stats["cohort_members_observed"] += observed
        if complete:
            _stats["complete_cohort_batches"] += 1
            _complete_cohort_ids.add(cohort_id)
        else:
            _stats["partial_cohort_batches"] += 1
        _last_cohort_batch = {
            "cohort_id": cohort_id,
            "expected_members": expected,
            "observed_members": observed,
            "request_ids": sorted(
                str(getattr(member, "request_id", "<missing>"))
                for member in members
            ),
            "complete": complete,
            "reason": None if complete else "native_prefill_selected_partial_members",
        }


def observe_decode_batch(result: Any, *, now_s: float | None = None) -> None:
    """Record the actual occupancy of one SGLang decode scheduling batch."""

    global _last_decode_batch
    batch = _prefill_batch(result)
    if batch is None:
        return

    # Mixed chunked-prefill keeps its running decode rows in ``decoding_reqs``
    # while the batch-wide forward mode remains EXTEND. Counting only
    # ``forward_mode.is_decode()`` would silently hide those decode occupants.
    decoding_reqs = getattr(batch, "decoding_reqs", None)
    forward_mode = getattr(batch, "forward_mode", None)
    is_decode = getattr(forward_mode, "is_decode", None)
    decode_mode = callable(is_decode) and is_decode()
    if decoding_reqs is not None:
        requests = tuple(decoding_reqs or ())
        mixed = bool(requests) and not decode_mode
    elif decode_mode:
        requests = tuple(getattr(batch, "reqs", None) or ())
        mixed = False
    else:
        return
    if not requests:
        return

    size = len(requests)
    _stats["decode_batches"] += 1
    _stats["decode_requests_observed"] += size
    if mixed:
        _stats["decode_mixed_batches"] += 1
    _decode_batch_sizes[size] = _decode_batch_sizes.get(size, 0) + 1
    request_ids: list[str] = []
    cohort_ids: set[str] = set()
    for request in requests:
        request_id = str(getattr(request, "request_id", "<missing>"))
        request_ids.append(request_id)
        by_size = _decode_batch_sizes_by_request.get(request_id)
        if by_size is None:
            while len(_decode_batch_sizes_by_request) >= _MAX_DECODE_REQUEST_TRACES:
                _decode_batch_sizes_by_request.popitem(last=False)
                _stats["decode_request_trace_evictions"] += 1
            by_size = {}
            _decode_batch_sizes_by_request[request_id] = by_size
        by_size[size] = by_size.get(size, 0) + 1
        _decode_batch_sizes_by_request.move_to_end(request_id)
        try:
            hint = _extract_hint(request)
        except (AttributeError, TypeError, ValueError):
            hint = None
        if hint is not None and hint.cohort_id is not None:
            cohort_ids.add(hint.cohort_id)

    observed_at_s = time.monotonic() if now_s is None else float(now_s)
    if not math.isfinite(observed_at_s):
        raise ValueError("decode observation time must be finite")
    event = {
        "sequence": _stats["decode_batches"],
        "observed_at_s": observed_at_s,
        "batch_size": size,
        "request_ids": sorted(request_ids),
        "cohort_ids": sorted(cohort_ids),
        "mixed_prefill_decode": mixed,
    }
    if len(_decode_trace) == _decode_trace.maxlen:
        _stats["decode_trace_evictions"] += 1
    _decode_trace.append(event)
    _last_decode_batch = {
        "batch_size": size,
        "request_ids": event["request_ids"],
        "cohort_ids": event["cohort_ids"],
        "mixed_prefill_decode": mixed,
    }


def planned_cohort_sizes(submissions: Iterable[Any]) -> dict[str, int]:
    """Return and validate the cohort membership of an executable plan."""

    expected: dict[str, int] = {}
    members: dict[str, int] = {}
    for submission in submissions:
        cohort_id = getattr(submission, "cohort_id", None)
        if cohort_id is None:
            continue
        size = getattr(submission, "prefill_batch_size", None)
        if not isinstance(cohort_id, str) or not cohort_id:
            raise ValueError("planned cohort ids must be non-empty strings")
        if isinstance(size, bool) or not isinstance(size, int) or size < 2:
            raise ValueError("planned cohort sizes must be integers of at least 2")
        previous = expected.setdefault(cohort_id, size)
        if previous != size:
            raise ValueError(f"cohort {cohort_id}: inconsistent planned size")
        members[cohort_id] = members.get(cohort_id, 0) + 1
    for cohort_id, size in expected.items():
        if members[cohort_id] != size:
            count = members[cohort_id]
            noun = "submission" if count == 1 else "submissions"
            raise ValueError(
                f"cohort {cohort_id}: planned size {size} has "
                f"{count} {noun}"
            )
    return expected


def cohort_completion_evidence(
    expected_sizes: dict[str, int],
    stats: dict[str, object],
) -> dict[str, object]:
    """Summarize whether every planned cohort reached one complete batch."""

    for cohort_id, size in expected_sizes.items():
        if not isinstance(cohort_id, str) or not cohort_id:
            raise ValueError("expected cohort ids must be non-empty strings")
        if isinstance(size, bool) or not isinstance(size, int) or size < 2:
            raise ValueError("expected cohort sizes must be integers of at least 2")
    raw_complete_ids = stats.get("complete_cohort_ids", [])
    if not isinstance(raw_complete_ids, list) or any(
        not isinstance(cohort_id, str) for cohort_id in raw_complete_ids
    ):
        raise ValueError("complete cohort ids must be a list of strings")
    complete_ids = set(raw_complete_ids)
    expected_ids = set(expected_sizes)
    missing_ids = sorted(expected_ids - complete_ids)
    complete = int(stats.get("complete_cohort_batches", 0))
    partial = int(stats.get("partial_cohort_batches", 0))
    timeouts = int(stats.get("cohort_gate_timeouts", 0))
    conflicts = int(stats.get("cohort_metadata_conflicts", 0))
    errors = int(stats.get("cohort_gate_errors", 0))
    passed = bool(expected_sizes) and (
        complete >= len(expected_sizes)
        and not missing_ids
        and partial == 0
        and timeouts == 0
        and conflicts == 0
        and errors == 0
    )
    return {
        "passed": passed,
        "expected_cohorts": expected_sizes,
        "expected_cohort_count": len(expected_sizes),
        "expected_request_count": sum(expected_sizes.values()),
        "complete_cohort_ids": sorted(complete_ids),
        "missing_cohort_ids": missing_ids,
        "complete_cohort_batches": complete,
        "partial_cohort_batches": partial,
        "cohort_gate_timeouts": timeouts,
        "cohort_metadata_conflicts": conflicts,
        "cohort_gate_errors": errors,
    }


def _prefill_batch(result: Any) -> Any:
    """Extract a batch from the 0.5.2 batch and 0.5.17 NextBatchPlan APIs."""

    if result is None or hasattr(result, "reqs"):
        return result
    return getattr(result, "batch_to_run", None)


def withhold_incomplete_cohorts(
    waiting_queue: list[Any],
    *,
    now_s: float | None = None,
    timeout_ms: float | None = None,
) -> list[Any]:
    """Temporarily remove partial cohorts while native admission serves others.

    The caller must restore the returned requests after native prefill scheduling.
    This gate only controls cohort arrival. Native ``PrefillAdder`` still decides
    request count, tokens, radix, grammar, LoRA, and every other capacity limit.
    """

    global _cohort_timed_out, _last_held_cohorts
    if not waiting_queue:
        _cohort_first_seen_s.clear()
        _cohort_arrival_complete.clear()
        _cohort_timed_out.clear()
        _last_held_cohorts = []
        return []

    now = time.monotonic() if now_s is None else now_s
    if not math.isfinite(now):
        raise ValueError("cohort gate clock must be finite")
    if timeout_ms is None:
        raw_timeout = os.environ.get(COHORT_WAIT_TIMEOUT_ENV)
        timeout_ms = (
            _DEFAULT_COHORT_WAIT_TIMEOUT_MS
            if raw_timeout in (None, "")
            else float(raw_timeout)
        )
    if not math.isfinite(timeout_ms) or timeout_ms < 0:
        raise ValueError(f"{COHORT_WAIT_TIMEOUT_ENV} must be finite and non-negative")

    cohorts: dict[str, list[Any]] = {}
    for request in waiting_queue:
        try:
            hint = _extract_hint(request)
        except (AttributeError, TypeError, ValueError):
            continue
        if hint is not None and hint.cohort_id is not None:
            cohorts.setdefault(hint.cohort_id, []).append(request)

    _cohort_arrival_complete.intersection_update(cohorts)
    _cohort_timed_out = {
        cohort_id: signature
        for cohort_id, signature in _cohort_timed_out.items()
        if cohort_id in cohorts
    }
    incomplete: dict[str, tuple[int, list[Any]]] = {}
    for cohort_id, members in cohorts.items():
        sizes = {_extract_hint(member).cohort_size for member in members}
        if len(sizes) != 1:
            _stats["cohort_metadata_conflicts"] += 1
            _cohort_first_seen_s.pop(cohort_id, None)
            _cohort_arrival_complete.discard(cohort_id)
            _cohort_timed_out.pop(cohort_id, None)
            continue
        expected = next(iter(sizes))
        if len(members) > expected:
            _stats["cohort_metadata_conflicts"] += 1
            _cohort_first_seen_s.pop(cohort_id, None)
            _cohort_arrival_complete.discard(cohort_id)
            _cohort_timed_out.pop(cohort_id, None)
        elif len(members) == expected:
            _cohort_arrival_complete.add(cohort_id)
            _cohort_first_seen_s.pop(cohort_id, None)
            _cohort_timed_out.pop(cohort_id, None)
        elif cohort_id not in _cohort_arrival_complete:
            incomplete[cohort_id] = (expected, members)

    active_ids = set(incomplete)
    for stale_id in set(_cohort_first_seen_s) - active_ids:
        del _cohort_first_seen_s[stale_id]

    held_members: list[Any] = []
    held_records: list[dict[str, object]] = []
    for cohort_id, (expected, members) in incomplete.items():
        signature = (expected, tuple(sorted(id(member) for member in members)))
        if _cohort_timed_out.get(cohort_id) == signature:
            continue
        if cohort_id in _cohort_timed_out:
            # A newly arrived member changes the admission problem; retry the
            # bounded wait once from the new membership set.
            del _cohort_timed_out[cohort_id]
        first_seen = _cohort_first_seen_s.setdefault(cohort_id, now)
        waited_ms = max(0.0, (now - first_seen) * 1000.0)
        if waited_ms >= timeout_ms:
            _stats["cohort_gate_timeouts"] += 1
            del _cohort_first_seen_s[cohort_id]
            _cohort_timed_out[cohort_id] = signature
            continue
        held_members.extend(members)
        held_records.append(
            {
                "cohort_id": cohort_id,
                "expected_members": expected,
                "observed_members": len(members),
                "waited_ms": round(waited_ms, 3),
                "request_ids": sorted(
                    str(getattr(member, "request_id", "<missing>"))
                    for member in members
                ),
            }
        )

    _stats["cohort_gate_ticks"] += 1
    _last_held_cohorts = held_records
    if not held_members:
        return []
    _stats["cohort_gate_holds"] += 1
    _stats["cohort_members_held"] += len(held_members)
    held_ids = {id(member) for member in held_members}
    waiting_queue[:] = [
        member for member in waiting_queue if id(member) not in held_ids
    ]
    return held_members


def _gate_active_prefill_queue(
    waiting_queue: list[Any], *, now_s: float | None = None
) -> None:
    """Apply the arrival gate after SGLang has published grammar-ready work."""

    frame = _active_prefill_gate.get()
    if frame is None or frame.ordered_queue is not None:
        return
    frame.ordered_queue = list(waiting_queue)
    held: list[Any] = []
    if _hard_release_gate:
        try:
            held.extend(withhold_unreleased_requests(waiting_queue, now_s=now_s))
        except (AttributeError, TypeError, ValueError, OSError):
            _stats["hard_release_gate_errors"] += 1
    try:
        held.extend(withhold_incomplete_cohorts(waiting_queue))
    except (AttributeError, TypeError, ValueError, OSError):
        # A malformed optional gate must not suppress native prefill.
        _stats["cohort_gate_errors"] += 1
    frame.held = held


def _restore_prefill_queue(
    waiting_queue: list[Any], *, frame: _PrefillGateFrame
) -> None:
    """Restore held requests without changing the native order of survivors."""

    if frame.ordered_queue is None or not frame.held:
        return
    current = list(waiting_queue)
    current_ids = {id(request) for request in current}
    held_ids = {id(request) for request in frame.held}
    previous_ids = {id(request) for request in frame.ordered_queue}
    restored = [
        request
        for request in frame.ordered_queue
        if id(request) in current_ids or id(request) in held_ids
    ]
    restored.extend(
        request for request in current if id(request) not in previous_ids
    )
    waiting_queue[:] = restored


def snapshot_stats() -> dict[str, object]:
    snapshot: dict[str, object] = dict(_stats)
    snapshot["last_cohort_batch"] = _last_cohort_batch
    snapshot["complete_cohort_ids"] = sorted(_complete_cohort_ids)
    snapshot["last_decode_batch"] = _last_decode_batch
    snapshot["decode_batch_size_histogram"] = {
        str(size): count for size, count in sorted(_decode_batch_sizes.items())
    }
    snapshot["decode_batch_sizes_by_request"] = {
        request_id: {
            str(size): count for size, count in sorted(by_size.items())
        }
        for request_id, by_size in sorted(_decode_batch_sizes_by_request.items())
    }
    snapshot["decode_trace"] = [dict(event) for event in _decode_trace]
    snapshot["last_held_cohorts"] = list(_last_held_cohorts)
    snapshot["last_held_releases"] = list(_last_held_releases)
    snapshot["stats_identity"] = dict(_stats_identity)
    return snapshot


def _capture_stats_identity(scheduler: Any) -> None:
    """Capture the scheduler's TP/PP/DP coordinates once per process."""

    source = getattr(scheduler, "ps", scheduler)
    identity = {}
    for name in (
        "tp_rank",
        "pp_rank",
        "dp_rank",
        "attn_tp_rank",
        "attn_dp_rank",
        "moe_dp_rank",
    ):
        value = getattr(source, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            identity[name] = value
    if identity:
        _stats_identity.clear()
        _stats_identity.update(identity)


def _stats_target(path: str | Path) -> Path:
    target = Path(path).expanduser()
    if not _stats_identity:
        return target
    tag = ".".join(
        f"{name}-{_stats_identity.get(name, 0)}"
        for name in (
            "tp_rank",
            "pp_rank",
            "dp_rank",
            "attn_tp_rank",
            "attn_dp_rank",
            "moe_dp_rank",
        )
    )
    return target.with_name(
        f"{target.stem}.rank-{tag}.pid-{os.getpid()}{target.suffix}"
    )


def find_stats_paths(path: str | Path) -> list[Path]:
    """Return a legacy unsplit file or every per-rank shard for ``path``."""

    target = Path(path).expanduser()
    paths = []
    if target.is_file():
        paths.append(target)
    prefix = f"{target.stem}.rank-"
    paths.extend(
        candidate
        for candidate in target.parent.glob(f"{prefix}*{target.suffix}")
        if candidate.is_file()
    )
    return sorted(set(paths), key=lambda item: item.name)


def load_stats(path: str | Path) -> dict[str, object]:
    """Load all rank shards without pretending they are additive counters."""

    import json

    paths = find_stats_paths(path)
    if not paths:
        raise FileNotFoundError(path)
    documents = [json.loads(item.read_text(encoding="utf-8")) for item in paths]
    selected = max(
        documents,
        key=lambda document: (
            int(document.get("complete_cohort_batches", 0)),
            int(document.get("prefill_batches", 0)),
            int(document.get("ticks", 0)),
        ),
    )
    result = dict(selected)
    for name in (
        "fallback_exception",
        "fallback_invalid_hint",
        "telemetry_errors",
        "cohort_gate_errors",
        "hard_release_gate_errors",
    ):
        result[name] = sum(int(document.get(name, 0)) for document in documents)
    result["shard_count"] = len(paths)
    result["stats_files"] = [str(item) for item in paths]
    result["stats_shards"] = [
        {"path": str(item), "stats": document}
        for item, document in zip(paths, documents, strict=True)
    ]
    return result


def _publish_stats(*, force: bool = False, now_s: float | None = None) -> None:
    global _last_stats_publish
    path = os.environ.get(STATS_PATH_ENV)
    if not path:
        return
    now = time.monotonic() if now_s is None else now_s
    if not force and now - _last_stats_publish < _STATS_INTERVAL_S:
        return
    try:
        target = _stats_target(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Scheduler workers publish the same aggregate telemetry path. A
        # process-unique temporary file prevents one rank from replacing or
        # deleting another rank's in-flight snapshot before os.replace().
        temp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        import json

        temp.write_text(json.dumps(snapshot_stats(), sort_keys=True), encoding="utf-8")
        temp.replace(target)
        _last_stats_publish = now
    except OSError:
        _stats["telemetry_errors"] += 1


def install() -> None:
    """Install the opt-in hook after native priority calculation."""
    global _hard_release_gate, _installed
    if _installed:
        return
    policy = os.environ.get(POLICY_ENV, "off")
    if policy not in POLICIES:
        raise ValueError(f"unknown {POLICY_ENV}")
    if policy == "off":
        _hard_release_gate = False
        return
    hard_release = os.environ.get(HARD_RELEASE_GATE_ENV, "0").strip().lower()
    if hard_release not in ("0", "1", "false", "true", "no", "yes", "off", "on"):
        raise ValueError(f"unknown {HARD_RELEASE_GATE_ENV}")
    _hard_release_gate = hard_release in ("1", "true", "yes", "on")
    horizon_raw = os.environ.get(HORIZON_ENV)
    horizon_ms = None if horizon_raw in (None, "") else int(horizon_raw)
    if horizon_ms is not None and horizon_ms < 0:
        raise ValueError(f"{HORIZON_ENV} must be non-negative")
    try:
        # Do not replace the native transport on revisions that already expose
        # it (including development images used by the real scheduler smoke).
        import importlib
        import sys

        if "sglang.srt.managers.kv_hints" not in sys.modules:
            importlib.import_module("sglang.srt.managers.kv_hints")
    except (ImportError, AttributeError):
        # SGLang 0.5.2 predates kv_hints. Install the transport before
        # http_server imports its request dataclass and before workers start.
        from adaptive_sparse import kv_hints_compat

        kv_hints_compat.install()

    from sglang.srt.managers.schedule_policy import SchedulePolicy

    original = SchedulePolicy.calc_priority
    if getattr(original, "_elastic_future_hook", False):
        _installed = True
        return

    @wraps(original)
    def elastic_calc_priority(self, waiting_queue, *args, **kwargs):
        result = original(self, waiting_queue, *args, **kwargs)
        priority_sign = (
            getattr(self, "priority_sign", None)
            if getattr(self, "enable_priority_scheduling", False)
            else None
        )
        apply_to_waiting_queue(
            waiting_queue, priority_sign=priority_sign, horizon_ms=horizon_ms
        )
        _gate_active_prefill_queue(waiting_queue)
        return result

    elastic_calc_priority._elastic_future_hook = True
    SchedulePolicy.calc_priority = elastic_calc_priority

    try:
        from sglang.srt.managers.scheduler import Scheduler
    except (ImportError, AttributeError):
        Scheduler = None
    if Scheduler is not None and hasattr(Scheduler, "get_new_batch_prefill"):
        original_prefill = Scheduler.get_new_batch_prefill
        if not getattr(original_prefill, "_elastic_cohort_observer", False):

            @wraps(original_prefill)
            def elastic_get_new_batch_prefill(self, *args, **kwargs):
                _capture_stats_identity(self)
                frame = _PrefillGateFrame()
                token = _active_prefill_gate.set(frame)
                try:
                    result = original_prefill(self, *args, **kwargs)
                finally:
                    _restore_prefill_queue(self.waiting_queue, frame=frame)
                    _active_prefill_gate.reset(token)
                observe_prefill_batch(_prefill_batch(result))
                _publish_stats()
                return result

            elastic_get_new_batch_prefill._elastic_cohort_observer = True
            Scheduler.get_new_batch_prefill = elastic_get_new_batch_prefill
    if Scheduler is not None and hasattr(Scheduler, "get_next_batch_to_run"):
        original_next_batch = Scheduler.get_next_batch_to_run
        if not getattr(original_next_batch, "_elastic_decode_observer", False):

            @wraps(original_next_batch)
            def elastic_get_next_batch_to_run(self, *args, **kwargs):
                _capture_stats_identity(self)
                result = original_next_batch(self, *args, **kwargs)
                observe_decode_batch(result)
                _publish_stats()
                return result

            elastic_get_next_batch_to_run._elastic_decode_observer = True
            Scheduler.get_next_batch_to_run = elastic_get_next_batch_to_run
    _installed = True
