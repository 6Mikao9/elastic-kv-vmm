"""Opt-in SGLang ordering hook for controlled, known-future rollouts.

This uses SGLang's native ``kv_hints`` envelope when available and a narrow
transport shim on 0.5.2. It only reorders the waiting queue after SGLang's
native policy has run; ``PrefillAdder`` and all native memory, grammar, LoRA,
and request-slot checks remain authoritative. The hook is disabled by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import math
import os
from pathlib import Path
import time
from typing import Any, Iterable


ACTION_TYPE = "elastic.future"
ACTION_VERSION = "1"
POLICY_ENV = "EKV_FUTURE_POLICY"
STATS_PATH_ENV = "EKV_FUTURE_STATS_PATH"
POLICIES = ("off", "deadline")
_STATS_INTERVAL_S = 0.25

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
    "fallback_exception": 0,
    "telemetry_errors": 0,
}


@dataclass(frozen=True, slots=True)
class FutureHint:
    release_after_ms: int = 0
    deadline_after_ms: int | None = None
    expected_output_tokens: int | None = None
    weight: float = 1.0
    kind: str = "generic"
    residence: str | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> "FutureHint":
        if not isinstance(payload, dict):
            raise ValueError("elastic.future payload must be an object")
        allowed = {
            "release_after_ms",
            "deadline_after_ms",
            "expected_output_tokens",
            "weight",
            "kind",
            "residence",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError("elastic.future payload has unknown keys")
        release = _bounded_int(payload.get("release_after_ms", 0), "release_after_ms")
        deadline_raw = payload.get("deadline_after_ms")
        deadline = None if deadline_raw is None else _bounded_int(deadline_raw, "deadline_after_ms")
        if deadline is not None and deadline < release:
            raise ValueError("deadline_after_ms precedes release_after_ms")
        expected_raw = payload.get("expected_output_tokens")
        expected = (
            None
            if expected_raw is None
            else _bounded_int(expected_raw, "expected_output_tokens", minimum=1)
        )
        weight = payload.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("weight must be numeric")
        weight = float(weight)
        if not math.isfinite(weight) or not 0 < weight <= 1e6:
            raise ValueError("weight must be finite and in (0, 1e6]")
        kind = payload.get("kind", "generic")
        if not isinstance(kind, str) or not kind or len(kind) > 64:
            raise ValueError("kind must be a short non-empty string")
        residence = payload.get("residence")
        if residence is not None and residence not in ("auto", "resident", "host_backed"):
            raise ValueError("invalid residence hint")
        return cls(release, deadline, expected, weight, kind, residence)


def _bounded_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= 86_400_000:
        raise ValueError(f"{name} is outside the supported range")
    return value


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
    if getattr(action, "action_version", None) != ACTION_VERSION:
        raise ValueError("unsupported elastic.future action version")
    return FutureHint.from_payload(getattr(action, "payload", None))


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


def _prompt_tokens(request: Any) -> int:
    ids = getattr(request, "origin_input_ids", None)
    if ids is None:
        return 1
    total = len(ids)
    cached = getattr(request, "num_matched_prefix_tokens", 0) or 0
    if isinstance(cached, bool) or not isinstance(cached, int) or cached < 0:
        raise ValueError("matched prefix length is invalid")
    return max(1, total - min(cached, total))


def order_requests(
    requests: Iterable[Any],
    *,
    now_s: float,
    prefill_tokens_per_s: float = 8_000.0,
    decode_tokens_per_s: float = 20_000.0,
    priority_sign: int | None = None,
) -> list[Any]:
    """Return a stable deadline-aware order or the exact native order.

    All requests in the queue must carry a valid hint.  A mixed queue falls
    back without side effects, so enabling the feature cannot reorder legacy
    traffic that has no planning metadata.
    """

    queue = list(requests)
    if not queue:
        return queue
    hints = []
    for request in queue:
        hint = _extract_hint(request)
        if hint is None:
            _stats["fallback_missing_hint"] += 1
            return queue
        hints.append(hint)

    decorated = []
    for native_order, (request, hint) in enumerate(zip(queue, hints)):
        age_ms = _request_age_ms(request, now_s)
        released = age_ms >= hint.release_after_ms
        prefill_ms = 1000.0 * _prompt_tokens(request) / prefill_tokens_per_s
        decode_ms = 1000.0 * (hint.expected_output_tokens or 1) / decode_tokens_per_s
        estimated_request_ms = prefill_ms + decode_ms
        latest_start = (
            float("inf")
            if hint.deadline_after_ms is None
            else hint.deadline_after_ms - age_ms - estimated_request_ms
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
                hint.release_after_ms,
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
) -> None:
    """Mutate a queue only after a complete successful planning pass."""
    _stats["ticks"] += 1
    _stats["queue_requests"] = len(waiting_queue)
    try:
        planned = order_requests(
            waiting_queue,
            now_s=time.perf_counter() if now_s is None else now_s,
            priority_sign=priority_sign,
        )
        planned_hints = [_extract_hint(request) for request in planned]
        if all(hint is not None for hint in planned_hints):
            released = sum(
                _request_age_ms(request, time.perf_counter() if now_s is None else now_s)
                >= hint.release_after_ms
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
    except Exception:
        _stats["fallback_exception"] += 1
        _publish_stats()
        return
    waiting_queue[:] = planned
    _publish_stats()


def snapshot_stats() -> dict[str, int]:
    return dict(_stats)


def _publish_stats(*, force: bool = False, now_s: float | None = None) -> None:
    global _last_stats_publish
    path = os.environ.get(STATS_PATH_ENV)
    if not path:
        return
    now = time.monotonic() if now_s is None else now_s
    if not force and now - _last_stats_publish < _STATS_INTERVAL_S:
        return
    try:
        target = Path(path).expanduser()
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
    global _installed
    if _installed:
        return
    policy = os.environ.get(POLICY_ENV, "off")
    if policy not in POLICIES:
        raise ValueError(f"unknown {POLICY_ENV}")
    if policy == "off":
        return
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
        apply_to_waiting_queue(waiting_queue, priority_sign=priority_sign)
        return result

    elastic_calc_priority._elastic_future_hook = True
    SchedulePolicy.calc_priority = elastic_calc_priority
    _installed = True
