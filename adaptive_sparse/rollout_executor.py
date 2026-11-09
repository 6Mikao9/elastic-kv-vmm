"""Execute an offline rollout plan against a live SGLang HTTP server.

The planner intentionally stays offline so it can be reproduced without CUDA.
This module is the narrow live boundary: it wakes at planned submission times,
attaches the versioned ``elastic.future`` envelope, records actual latency and
deadline outcomes, and optionally reconciles per-request residency telemetry
from every tensor-parallel rank.

The executor does not infer success from a completed HTTP response alone.  A
late submission, malformed hint, missing timing field, or conflicting TP
residency decision is a hard failure.  A missed application deadline is kept
as a measured result because it can be a valid policy outcome rather than a
transport error.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .elastic_admission import (
    ElasticAdmissionDecision,
    ElasticKVConfig,
    decide_elastic_admission,
)
from .future_plan import RolloutPlan, RolloutSubmission
from .service_time import ServiceTimeEstimator, estimate_service_duration_ms
from .sglang_future_hook import (
    ACTION_VERSION,
    LEGACY_ACTION_VERSION,
    RATE_ACTION_VERSION,
    FutureHint,
)

FIRST_TOKEN_CALLBACK_KEY = "_elastic_first_token_callback"


class RolloutExecutionError(RuntimeError):
    """A live plan violated a transport, timing, or evidence invariant."""


def _execution_groups(
    submissions: Sequence[RolloutSubmission],
) -> tuple[tuple[RolloutSubmission, ...], ...]:
    """Return chronological atomic submission groups for execution."""

    groups: list[tuple[RolloutSubmission, ...] | list[RolloutSubmission]] = []
    cohort_positions: dict[str, int] = {}
    for item in submissions:
        if item.cohort_id is None:
            groups.append((item,))
            continue
        position = cohort_positions.get(item.cohort_id)
        if position is None:
            cohort_positions[item.cohort_id] = len(groups)
            groups.append([item])
        else:
            group = groups[position]
            assert isinstance(group, list)
            group.append(item)
    normalized = tuple(
        tuple(group) if isinstance(group, list) else group for group in groups
    )
    normalized = tuple(
        sorted(
            normalized,
            key=lambda group: (
                group[0].submit_at_ms,
                min(item.request_id for item in group),
            ),
        )
    )
    return normalized


def _cohort_from_queue(
    head: RolloutSubmission,
    queue: Sequence[RolloutSubmission],
) -> tuple[RolloutSubmission, ...]:
    """Return all still-queued members of ``head``'s atomic cohort."""

    if head.cohort_id is None:
        return (head,)
    return tuple(item for item in queue if item.cohort_id == head.cohort_id)


def _admission_hold_bytes(admission: ElasticAdmissionDecision) -> int:
    """Return the conservative in-flight charge used by managed admission."""

    return (
        admission.active_after_bytes
        - admission.active_before_bytes
        + admission.prefill_workspace_bytes
    )


def _barrier_generate(
    cohort_id: str | None,
    barrier: threading.Barrier | None,
    timeout_s: float,
    generate: Callable[[str, dict[str, object], float, bool], dict[str, object]],
) -> Callable[[str, dict[str, object], float, bool], dict[str, object]]:
    """Release all cohort workers into ``generate`` together."""

    if barrier is None:
        return generate

    def synchronized(
        url: str,
        spec: dict[str, object],
        timeout: float,
        stream: bool,
    ) -> dict[str, object]:
        try:
            barrier.wait(timeout=timeout_s)
        except threading.BrokenBarrierError as exc:
            raise RolloutExecutionError(
                f"cohort {cohort_id}: dispatch barrier failed"
            ) from exc
        return generate(url, spec, timeout, stream)

    return synchronized


def _ceil_milliseconds(value: float) -> int:
    """Ceil a clock-derived duration without adding a spurious millisecond."""

    return math.ceil(value - 1e-7)


def _announce_time_ms(
    submission: RolloutSubmission,
    preannounce_lead_ms: int,
) -> int:
    """Return the monotonic execution deadline for a preannounced request."""

    return max(0, submission.submit_at_ms - preannounce_lead_ms)


@dataclass(frozen=True, slots=True)
class ReplayClock:
    """Injectable monotonic clock used by both live and fake-clock tests."""

    now: Callable[[], float] = time.perf_counter
    sleep: Callable[[float], None] = time.sleep


def _future_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate one rollout payload before any request is dispatched."""

    if not isinstance(payload, Mapping):
        raise RolloutExecutionError("rollout hint must be an object")
    try:
        FutureHint.from_payload(dict(payload))
    except (TypeError, ValueError) as exc:
        raise RolloutExecutionError(f"invalid rollout hint: {exc}") from exc
    return dict(payload)


def _service_time_residence(value: object) -> str | None:
    """Return a placement only when service-time feedback can be segmented."""

    return value if value in ("resident", "host_backed") else None


def _submission_service_rates(
    estimator: ServiceTimeEstimator | None,
    submission: RolloutSubmission,
    *,
    residence: object | None = None,
) -> dict[str, float]:
    """Read feedback for one request's actual batch shape and placement."""

    if estimator is None:
        return {}
    planned_residence = (
        FutureHint.from_payload(dict(submission.hint)).residence
        if residence is None
        else residence
    )
    decode_residence = _service_time_residence(planned_residence)
    rates = estimator.rates(
        residence=decode_residence,
        kind=submission.kind,
        prefill_batch_size=submission.prefill_batch_size,
        decode_batch_size=(
            submission.decode_batch_size
            if decode_residence is not None
            else None
        ),
    )
    if decode_residence is None:
        rates.pop("decode_tokens_per_s", None)
    return rates


def make_kv_hints(
    submission: RolloutSubmission,
    *,
    deadline_after_ms: int | None = None,
    release_after_ms: int | None = None,
    residence_override: str | None = None,
    clock_now_s: float | None = None,
    prefill_tokens_per_s: float | None = None,
    decode_tokens_per_s: float | None = None,
) -> dict[str, object]:
    """Wrap a planner payload in the SGLang versioned hint envelope."""

    payload = _future_payload(submission.hint)
    dispatch_clock_s = time.perf_counter() if clock_now_s is None else clock_now_s
    if not math.isfinite(dispatch_clock_s) or dispatch_clock_s < 0:
        raise RolloutExecutionError("dispatch clock must be finite and nonnegative")
    if submission.deadline_ms is not None:
        remaining_ms = (
            max(0, submission.deadline_ms - submission.submit_at_ms)
            if deadline_after_ms is None
            else deadline_after_ms
        )
        payload["deadline_after_ms"] = remaining_ms
        payload["deadline_clock"] = "perf_counter"
        payload["deadline_absolute_ms"] = (
            _ceil_milliseconds(dispatch_clock_s * 1000.0 + remaining_ms)
        )
    if release_after_ms is not None:
        if (
            isinstance(release_after_ms, bool)
            or not isinstance(release_after_ms, int)
            or not 0 <= release_after_ms <= 86_400_000
        ):
            raise RolloutExecutionError(
                "runtime release_after_ms must be a nonnegative integer"
            )
        payload["release_after_ms"] = release_after_ms
        payload["release_clock"] = "perf_counter"
        payload["release_absolute_ms"] = _ceil_milliseconds(
            dispatch_clock_s * 1000.0 + release_after_ms
        )
    if residence_override is not None:
        if residence_override not in ("resident", "host_backed"):
            raise RolloutExecutionError(
                f"invalid runtime residence override: {residence_override!r}"
            )
        payload["residence"] = residence_override
    if prefill_tokens_per_s is not None:
        payload["prefill_tokens_per_s"] = prefill_tokens_per_s
    if decode_tokens_per_s is not None:
        payload["decode_tokens_per_s"] = decode_tokens_per_s
    if submission.cohort_id is not None:
        payload["cohort_id"] = submission.cohort_id
        payload["cohort_size"] = submission.prefill_batch_size
    # Revalidate the complete payload after runtime overrides.  A malformed
    # feedback value must fail before any request reaches the server.
    try:
        FutureHint.from_payload(payload)
    except (TypeError, ValueError) as exc:
        raise RolloutExecutionError(f"invalid runtime service-time hint: {exc}") from exc
    has_service_rates = any(
        name in payload for name in ("prefill_tokens_per_s", "decode_tokens_per_s")
    )
    if submission.cohort_id is not None:
        action_version = ACTION_VERSION
    elif has_service_rates:
        action_version = RATE_ACTION_VERSION
    else:
        action_version = LEGACY_ACTION_VERSION
    request_id = submission.request_id
    return {
        "protocol_version": "1",
        "message_id": request_id,
        "actions": [
            {
                "action_id": f"{request_id}:future",
                "action_type": "elastic.future",
                "action_version": action_version,
                "payload": payload,
            }
        ],
    }


def validate_submissions(submissions: Iterable[RolloutSubmission]) -> tuple[RolloutSubmission, ...]:
    """Return stable submission order after checking all plan invariants."""

    ordered = tuple(submissions)
    if not ordered:
        raise RolloutExecutionError("rollout contains no submissions")
    prefix_lengths: dict[str, set[int]] = {}
    cohorts: dict[str, list[RolloutSubmission]] = {}
    for item in ordered:
        if not isinstance(item.request_id, str) or not item.request_id:
            raise RolloutExecutionError("each rollout submission needs a request ID")
        if any(
            type(value) is not int
            for value in (
                item.planned_arrival_ms,
                item.submit_at_ms,
                item.estimated_finish_ms,
                item.prompt_tokens,
                item.output_tokens,
            )
        ):
            raise RolloutExecutionError(f"{item.request_id}: invalid integer plan field")
        if item.submit_at_ms < 0 or item.estimated_finish_ms < item.submit_at_ms:
            raise RolloutExecutionError(f"{item.request_id}: inconsistent plan times")
        if item.planned_arrival_ms > item.submit_at_ms:
            raise RolloutExecutionError(
                f"{item.request_id}: submit time precedes planned arrival"
            )
        if item.prompt_tokens < 1 or item.output_tokens < 1:
            raise RolloutExecutionError(f"{item.request_id}: token counts must be positive")
        if (
            type(item.decode_batch_size) is not int
            or item.decode_batch_size < 1
        ):
            raise RolloutExecutionError(
                f"{item.request_id}: decode batch size must be positive"
            )
        if (
            type(item.prefill_batch_size) is not int
            or item.prefill_batch_size < 1
        ):
            raise RolloutExecutionError(
                f"{item.request_id}: prefill batch size must be positive"
            )
        if item.cohort_id is not None:
            if (
                not isinstance(item.cohort_id, str)
                or not item.cohort_id
                or len(item.cohort_id) > 128
            ):
                raise RolloutExecutionError(
                    f"{item.request_id}: invalid cohort ID"
                )
            cohorts.setdefault(item.cohort_id, []).append(item)
        elif item.prefill_batch_size != 1:
            raise RolloutExecutionError(
                f"{item.request_id}: batched prefill requires a cohort ID"
            )
        hint = FutureHint.from_payload(_future_payload(item.hint))
        if hint.release_after_ms != 0:
            raise RolloutExecutionError(
                f"{item.request_id}: controller-owned submissions require release_after_ms=0"
            )
        if hint.kind != item.kind:
            raise RolloutExecutionError(
                f"{item.request_id}: hint kind does not match submission kind"
            )
        if hint.expected_output_tokens != item.output_tokens:
            raise RolloutExecutionError(
                f"{item.request_id}: hint output estimate does not match output_tokens"
            )
        if hint.shared_prefix_tokens > item.prompt_tokens:
            raise RolloutExecutionError(
                f"{item.request_id}: shared prefix exceeds prompt length"
            )
        if hint.prefix_group is not None:
            prefix_lengths.setdefault(hint.prefix_group, set()).add(
                hint.shared_prefix_tokens
            )
        if item.deadline_ms is not None:
            if type(item.deadline_ms) is not int or item.deadline_ms < item.planned_arrival_ms:
                raise RolloutExecutionError(
                    f"{item.request_id}: deadline must be an integer at or after arrival"
                )
            if hint.deadline_after_ms != max(0, item.deadline_ms - item.submit_at_ms):
                raise RolloutExecutionError(
                    f"{item.request_id}: hint deadline is inconsistent with absolute deadline"
                )
        elif hint.deadline_after_ms is not None:
            raise RolloutExecutionError(
                f"{item.request_id}: deadline hint lacks the original absolute deadline"
            )
        make_kv_hints(item)
    inconsistent = next(
        (group for group, lengths in prefix_lengths.items() if len(lengths) > 1),
        None,
    )
    if inconsistent is not None:
        raise RolloutExecutionError(
            f"prefix group {inconsistent}: inconsistent shared prefix lengths"
        )
    ids = [item.request_id for item in ordered]
    if len(set(ids)) != len(ids):
        raise RolloutExecutionError("rollout request IDs must be unique")
    for cohort_id, members in cohorts.items():
        starts = {item.submit_at_ms for item in members}
        sizes = {item.prefill_batch_size for item in members}
        decodes = {item.decode_batch_size for item in members}
        if len(starts) != 1 or len(sizes) != 1 or len(decodes) != 1:
            raise RolloutExecutionError(
                f"cohort {cohort_id}: inconsistent timing or batch metadata"
            )
        if next(iter(sizes)) != len(members):
            raise RolloutExecutionError(
                f"cohort {cohort_id}: prefill batch size does not match members"
            )
    return tuple(
        sorted(ordered, key=lambda item: (item.submit_at_ms, item.request_id))
    )


def load_plan(path: str | Path) -> tuple[RolloutSubmission, ...]:
    """Load either a bare rollout or ``future_plan --output`` JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RolloutExecutionError("plan file must contain a JSON object")
    rollout = payload.get("rollout", payload)
    if not isinstance(rollout, Mapping):
        raise RolloutExecutionError("plan rollout must be an object")
    raw_submissions = rollout.get("submissions")
    if not isinstance(raw_submissions, list):
        raise RolloutExecutionError("plan rollout lacks a submissions list")

    submissions = []
    for raw in raw_submissions:
        if not isinstance(raw, Mapping):
            raise RolloutExecutionError("rollout submission must be an object")
        hint = raw.get("hint")
        if not isinstance(hint, Mapping):
            raise RolloutExecutionError("rollout submission hint must be an object")
        try:
            submission = RolloutSubmission(
                request_id=raw["request_id"],
                planned_arrival_ms=raw["planned_arrival_ms"],
                submit_at_ms=raw["submit_at_ms"],
                estimated_finish_ms=raw["estimated_finish_ms"],
                kind=raw["kind"],
                prompt_tokens=raw["prompt_tokens"],
                output_tokens=raw["output_tokens"],
                hint=dict(hint),
                deadline_ms=raw.get("deadline_ms"),
                decode_batch_size=raw.get("decode_batch_size", 1),
                prefill_batch_size=raw.get("prefill_batch_size", 1),
                cohort_id=raw.get("cohort_id"),
            )
        except KeyError as exc:
            raise RolloutExecutionError(
                f"plan submission lacks {exc.args[0]!r}; regenerate it with the current planner"
            ) from exc
        submissions.append(submission)
    return validate_submissions(submissions)


def _wait_until(target: float, clock: ReplayClock) -> None:
    while True:
        remaining = target - clock.now()
        if remaining <= 0:
            return
        clock.sleep(remaining)


def _default_generate(url: str, spec: dict[str, object], timeout: float, stream: bool) -> dict[str, object]:
    from benchmarks.client import generate

    wire_spec = dict(spec)
    first_token_callback = wire_spec.pop(FIRST_TOKEN_CALLBACK_KEY, None)
    return generate(
        url,
        wire_spec,
        timeout,
        stream,
        first_token_callback=first_token_callback,
    )


def _deadline_target_ms(submission: RolloutSubmission) -> int | None:
    return submission.deadline_ms


def _residency_request_rows(evidence: Mapping[str, object]) -> Mapping[str, object]:
    requests = evidence.get("requests")
    if not isinstance(requests, Mapping):
        raise RolloutExecutionError("residency evidence lacks a requests object")
    return requests


def collect_residency_evidence(stats_dir: str | Path) -> dict[str, object]:
    """Reconcile per-request residency decisions from all visible TP ranks."""

    root = Path(stats_dir)
    paths = sorted(root.glob("tp-*.json"))
    if not paths:
        return {"available": False, "reason": "no tp-*.json files", "requests": {}}
    by_request: dict[str, list[dict[str, object]]] = {}
    parsed_ranks = 0
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "available": False,
                "reason": f"invalid {path.name}: {type(exc).__name__}",
                "requests": {},
            }
        if not isinstance(payload, Mapping):
            return {"available": False, "reason": f"invalid {path.name}", "requests": {}}
        rank = payload.get("tp_rank")
        residency = payload.get("request_residency")
        if isinstance(rank, bool) or not isinstance(rank, int):
            return {"available": False, "reason": f"missing rank in {path.name}", "requests": {}}
        if not isinstance(residency, Mapping):
            return {
                "available": False,
                "reason": f"missing request_residency in {path.name}",
                "requests": {},
            }
        decisions = residency.get("requests", {})
        if not isinstance(decisions, Mapping):
            return {
                "available": False,
                "reason": f"invalid request decisions in {path.name}",
                "requests": {},
            }
        parsed_ranks += 1
        for request_id, decision in decisions.items():
            if not isinstance(request_id, str) or not isinstance(decision, Mapping):
                continue
            actual = decision.get("actual_residence")
            if actual not in ("resident", "host_backed"):
                continue
            by_request.setdefault(request_id, []).append(
                {"rank": rank, **dict(decision)}
            )

    reconciled: dict[str, object] = {}
    for request_id, rank_rows in by_request.items():
        distinct_ranks = {row["rank"] for row in rank_rows}
        actual_values = {row.get("actual_residence") for row in rank_rows}
        declared_world_sizes = {
            row.get("tp_world_size")
            for row in rank_rows
            if type(row.get("tp_world_size")) is int and row.get("tp_world_size") > 0
        }
        reconciled[request_id] = {
            "actual_residence": (
                next(iter(actual_values)) if len(actual_values) == 1 else "conflict"
            ),
            "all_ranks_agree": (
                len(actual_values) == 1
                and len(distinct_ranks) == parsed_ranks
                and declared_world_sizes == {parsed_ranks}
            ),
            "rank_count": len(distinct_ranks),
            "per_rank": sorted(rank_rows, key=lambda row: int(row["rank"])),
        }
    return {"available": True, "rank_count": parsed_ranks, "requests": reconciled}


def _residency_evidence_ready(
    evidence: Mapping[str, object],
    request_ids: set[str],
) -> bool:
    if not evidence.get("available"):
        return False
    requests = evidence.get("requests")
    if not isinstance(requests, Mapping):
        return False
    for request_id in request_ids:
        decision = requests.get(request_id)
        if not isinstance(decision, Mapping) or not decision.get("all_ranks_agree"):
            return False
    return True


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    at = (len(ordered) - 1) * fraction
    low, high = int(at), min(len(ordered) - 1, int(at) + 1)
    weight = at - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _efficiency(rows: Sequence[dict[str, object]]) -> dict[str, float]:
    serving_wall = max(row["actual_finish_ms"] for row in rows) - min(
        row["actual_submit_ms"] for row in rows
    )
    makespan = max(row["actual_finish_ms"] for row in rows) - min(
        row["planned_arrival_ms"] for row in rows
    )
    ttft = [float(row["ttft_s"]) * 1000.0 for row in rows if row["ttft_s"] is not None]
    tpot = [float(row["tpot_s"]) * 1000.0 for row in rows if row["tpot_s"] is not None]
    return {
        "serving_wall_ms": serving_wall,
        "makespan_ms": makespan,
        "output_tokens_per_s": (
            0.0
            if makespan <= 0
            else 1000.0 * sum(int(row["output_tokens"]) for row in rows) / makespan
        ),
        "serving_output_tokens_per_s": (
            0.0
            if serving_wall <= 0
            else 1000.0 * sum(int(row["output_tokens"]) for row in rows) / serving_wall
        ),
        "ttft_p50_ms": _percentile(ttft, 0.5),
        "ttft_p95_ms": _percentile(ttft, 0.95),
        "tpot_p50_ms": _percentile(tpot, 0.5),
        "tpot_p95_ms": _percentile(tpot, 0.95),
    }


@dataclass(slots=True)
class _PendingRequest:
    submission: RolloutSubmission
    future: Any | None
    dispatch_lateness_ms: float
    capacity_delay_ms: float
    admission: ElasticAdmissionDecision | None
    prefill_tokens_per_s_hint: float | None = None
    decode_tokens_per_s_hint: float | None = None
    decode_feedback_residence: str | None = None
    prefill_released: threading.Event = field(default_factory=threading.Event)
    prefill_release_perf_s: float | None = None

    def mark_prefill_released(self, now_s: float) -> None:
        if self.prefill_released.is_set():
            return
        if not math.isfinite(now_s):
            raise ValueError("first-token clock must be finite")
        self.prefill_release_perf_s = now_s
        self.prefill_released.set()

    @property
    def hold_bytes(self) -> int:
        if self.admission is None:
            return 0
        kv_bytes = (
            self.admission.active_after_bytes
            - self.admission.active_before_bytes
        )
        if self.prefill_released.is_set():
            return kv_bytes
        return kv_bytes + self.admission.prefill_workspace_bytes


def _duration_ms(
    submission: RolloutSubmission,
    hint: FutureHint,
    *,
    fallback_ms: float,
    prefill_tokens_per_s: float | None,
    decode_tokens_per_s: float | None,
) -> float:
    return estimate_service_duration_ms(
        prompt_tokens=submission.prompt_tokens,
        output_tokens=submission.output_tokens,
        fallback_ms=fallback_ms,
        prefill_tokens_per_s=(
            prefill_tokens_per_s
            if prefill_tokens_per_s is not None
            else hint.prefill_tokens_per_s
        ),
        decode_tokens_per_s=(
            decode_tokens_per_s
            if decode_tokens_per_s is not None
            else hint.decode_tokens_per_s
        ),
    )


def _select_due(
    due: Sequence[RolloutSubmission],
    policy: str,
    *,
    prefill_tokens_per_s: float | None = None,
    decode_tokens_per_s: float | None = None,
    decode_tokens_per_s_by_residence: Mapping[str, float | None] | None = None,
    service_time_estimator: ServiceTimeEstimator | None = None,
) -> RolloutSubmission:
    if policy == "timeline":
        return min(due, key=lambda item: (item.submit_at_ms, item.request_id))
    if policy != "deadline":
        raise ValueError("queue policy must be timeline or deadline")
    return min(
        due,
        key=lambda item: _deadline_order_key(
            item,
            prefill_tokens_per_s=prefill_tokens_per_s,
            decode_tokens_per_s=decode_tokens_per_s,
            decode_tokens_per_s_by_residence=(
                decode_tokens_per_s_by_residence
            ),
            service_time_estimator=service_time_estimator,
        ),
    )


def _deadline_order_key(
    item: RolloutSubmission,
    *,
    prefill_tokens_per_s: float | None = None,
    decode_tokens_per_s: float | None = None,
    decode_tokens_per_s_by_residence: Mapping[str, float | None] | None = None,
    service_time_estimator: ServiceTimeEstimator | None = None,
) -> tuple[object, ...]:
    """Return the immutable deadline-policy key for one rollout request."""

    hint = FutureHint.from_payload(dict(item.hint))
    item_feedback = _submission_service_rates(service_time_estimator, item)
    if service_time_estimator is not None:
        item_prefill = item_feedback.get("prefill_tokens_per_s")
        item_decode = item_feedback.get("decode_tokens_per_s")
    else:
        item_prefill = prefill_tokens_per_s
        item_decode = (
            decode_tokens_per_s
            if decode_tokens_per_s_by_residence is None
            else decode_tokens_per_s_by_residence.get(hint.residence)
        )
        if item_decode is None:
            item_decode = decode_tokens_per_s
    if item.deadline_ms is not None:
        predicted_duration = _duration_ms(
            item,
            hint,
            fallback_ms=max(1, item.estimated_finish_ms - item.submit_at_ms),
            prefill_tokens_per_s=item_prefill,
            decode_tokens_per_s=item_decode,
        )
        return (
            0,
            item.deadline_ms,
            item.deadline_ms - item.submit_at_ms - predicted_duration,
            -hint.weight,
            item.submit_at_ms,
            item.request_id,
        )
    return (1, 0, 0, -hint.weight, item.submit_at_ms, item.request_id)


def _ordered_due_cohorts(
    due: Sequence[RolloutSubmission],
    queue: Sequence[RolloutSubmission],
    policy: str,
    *,
    prefill_tokens_per_s: float | None = None,
    decode_tokens_per_s: float | None = None,
    decode_tokens_per_s_by_residence: Mapping[str, float | None] | None = None,
    service_time_estimator: ServiceTimeEstimator | None = None,
) -> tuple[tuple[RolloutSubmission, ...], ...]:
    """Return atomic cohorts in policy order for this scheduling pass."""

    if policy == "timeline":
        prioritized = sorted(
            due, key=lambda item: (item.submit_at_ms, item.request_id)
        )
    elif policy == "deadline":
        prioritized = sorted(
            due,
            key=lambda item: _deadline_order_key(
                item,
                prefill_tokens_per_s=prefill_tokens_per_s,
                decode_tokens_per_s=decode_tokens_per_s,
                decode_tokens_per_s_by_residence=(
                    decode_tokens_per_s_by_residence
                ),
                service_time_estimator=service_time_estimator,
            ),
        )
    else:
        raise ValueError("queue policy must be timeline or deadline")

    cohort_members: dict[str, list[RolloutSubmission]] = {}
    for item in queue:
        if item.cohort_id is not None:
            cohort_members.setdefault(item.cohort_id, []).append(item)
    remaining_ids = {item.request_id for item in due}
    ordered: list[tuple[RolloutSubmission, ...]] = []
    for head in prioritized:
        if head.request_id not in remaining_ids:
            continue
        cohort = (
            (head,)
            if head.cohort_id is None
            else tuple(cohort_members[head.cohort_id])
        )
        cohort_ids = {item.request_id for item in cohort}
        ordered.append(cohort)
        remaining_ids.difference_update(cohort_ids)
    return tuple(ordered)


def _admit_managed_cohort(
    cohort: Sequence[RolloutSubmission],
    *,
    queue: Sequence[RolloutSubmission],
    active_bytes: int,
    now_ms: float,
    config: ElasticKVConfig | None,
    allow_deadline_overcommit: bool,
    prefill_tokens_per_s: float | None,
    decode_tokens_per_s: float | None,
    decode_tokens_per_s_by_residence: Mapping[str, float | None],
    service_time_estimator: ServiceTimeEstimator | None = None,
) -> tuple[
    dict[str, ElasticAdmissionDecision | None],
    list[RolloutSubmission],
    bool,
]:
    """Simulate all-or-nothing admission for one atomic scheduling cohort."""

    cohort_ids = {item.request_id for item in cohort}
    other_queue = tuple(
        item for item in queue if item.request_id not in cohort_ids
    )
    provisional_bytes = active_bytes
    admissions: dict[str, ElasticAdmissionDecision | None] = {}
    deferred: list[RolloutSubmission] = []
    deadline_forced = False
    submission_rate_resolver = None
    if service_time_estimator is not None:
        def resolve_submission_rates(future, residence):
            return _submission_service_rates(
                service_time_estimator,
                future,
                residence=residence,
            )

        submission_rate_resolver = resolve_submission_rates
    for member in cohort:
        admission = None
        if config is not None:
            planned_residence = FutureHint.from_payload(dict(member.hint)).residence
            item_feedback = _submission_service_rates(
                service_time_estimator,
                member,
            )
            if service_time_estimator is not None:
                item_prefill = item_feedback.get("prefill_tokens_per_s")
                item_decode = item_feedback.get("decode_tokens_per_s")
            else:
                item_prefill = prefill_tokens_per_s
                item_decode = decode_tokens_per_s_by_residence.get(
                    planned_residence
                )
                if item_decode is None:
                    item_decode = decode_tokens_per_s
            admission = decide_elastic_admission(
                member,
                active_bytes=provisional_bytes,
                config=config,
                future_submissions=other_queue,
                now_ms=now_ms,
                prefill_tokens_per_s=item_prefill,
                decode_tokens_per_s=item_decode,
                submission_rate_resolver=submission_rate_resolver,
            )
            if admission.defer_for_budget:
                hint = FutureHint.from_payload(dict(member.hint))
                predicted_duration = _duration_ms(
                    member,
                    hint,
                    fallback_ms=max(
                        1,
                        member.estimated_finish_ms - member.submit_at_ms,
                    ),
                    prefill_tokens_per_s=item_prefill,
                    decode_tokens_per_s=item_decode,
                )
                forced = bool(
                    allow_deadline_overcommit
                    and member.deadline_ms is not None
                    and now_ms + predicted_duration > member.deadline_ms
                )
                deadline_forced = deadline_forced or forced
                if not forced:
                    deferred.append(member)
            provisional_bytes += _admission_hold_bytes(admission)
        admissions[member.request_id] = admission
    return admissions, deferred, deadline_forced


def _measure_result(
    submission: RolloutSubmission,
    record: object,
    *,
    origin: float,
    dispatch_lateness_ms: float,
    capacity_delay_ms: float,
    admission: ElasticAdmissionDecision | None,
    max_submit_lateness_ms: float,
    strict_submit_timing: bool,
) -> dict[str, object]:
    if not isinstance(record, Mapping):
        raise RolloutExecutionError(
            f"{submission.request_id}: generator returned a non-object result"
        )
    for key in ("_started_perf_s", "end_to_end_s", "output_tokens", "ttft_s", "tpot_s"):
        if key not in record:
            raise RolloutExecutionError(
                f"{submission.request_id}: result lacks {key}"
            )
    if int(record["output_tokens"]) != submission.output_tokens:
        raise RolloutExecutionError(
            f"{submission.request_id}: output token count differs from plan"
        )
    actual_submit_ms = (float(record["_started_perf_s"]) - origin) * 1000.0
    submit_lateness_ms = actual_submit_ms - submission.submit_at_ms
    if strict_submit_timing and submit_lateness_ms > max_submit_lateness_ms:
        raise RolloutExecutionError(
            f"{submission.request_id}: actual submit late by {submit_lateness_ms:.1f} ms"
        )
    actual_finish_ms = actual_submit_ms + float(record["end_to_end_s"]) * 1000.0
    deadline_target_ms = _deadline_target_ms(submission)
    if deadline_target_ms is not None:
        planned_deadline_hit = submission.estimated_finish_ms <= deadline_target_ms
        actual_deadline_hit = actual_finish_ms <= deadline_target_ms
    else:
        planned_deadline_hit = actual_deadline_hit = None
    planned_residence = FutureHint.from_payload(dict(submission.hint)).residence
    row: dict[str, object] = {
        "request_id": submission.request_id,
        "kind": submission.kind,
        "planned_arrival_ms": submission.planned_arrival_ms,
        "planned_submit_ms": submission.submit_at_ms,
        "planned_finish_ms": submission.estimated_finish_ms,
        "actual_submit_ms": actual_submit_ms,
        "preannounce_lead_actual_ms": max(
            0.0, submission.submit_at_ms - actual_submit_ms
        ),
        "actual_finish_ms": actual_finish_ms,
        "submit_lateness_ms": submit_lateness_ms,
        "dispatch_lateness_ms": dispatch_lateness_ms,
        "capacity_delay_ms": capacity_delay_ms,
        "finish_error_ms": actual_finish_ms - submission.estimated_finish_ms,
        "deadline_target_ms": deadline_target_ms,
        "planned_deadline_hit": planned_deadline_hit,
        "actual_deadline_hit": actual_deadline_hit,
        "planned_residence": planned_residence,
        "runtime_residence": (
            admission.chosen_residence if admission is not None else planned_residence
        ),
        "prompt_tokens": submission.prompt_tokens,
        "output_tokens": int(record["output_tokens"]),
        "cohort_id": submission.cohort_id,
        "prefill_batch_size": submission.prefill_batch_size,
        "decode_batch_size": submission.decode_batch_size,
        "ttft_s": record["ttft_s"],
        "tpot_s": record["tpot_s"],
        "end_to_end_s": float(record["end_to_end_s"]),
    }
    if admission is not None:
        row.update(
            {
                "elastic_admission_reason": admission.reason,
                "elastic_active_before_bytes": admission.active_before_bytes,
                "elastic_active_after_bytes": admission.active_after_bytes,
                "elastic_over_budget": admission.over_budget,
                "elastic_future_reserved_bytes": admission.future_reserved_bytes,
                "elastic_future_prefill_bytes": admission.future_prefill_bytes,
                "elastic_future_reserved_requests": admission.future_reserved_requests,
                "elastic_future_effective_after_bytes": (
                    admission.future_effective_after_bytes
                ),
                "elastic_future_over_budget": admission.future_over_budget,
                "elastic_prefill_workspace_bytes": (
                    admission.prefill_workspace_bytes
                ),
                "elastic_prefill_peak_bytes": admission.prefill_peak_bytes,
                "elastic_future_effective_peak_bytes": (
                    admission.future_effective_peak_bytes
                ),
                "elastic_prefill_over_budget": admission.prefill_over_budget,
                "elastic_future_peak_over_budget": (
                    admission.future_peak_over_budget
                ),
                "service_time_feedback_applied": (
                    admission.service_time_feedback_applied
                ),
                "service_time_current_duration_ms": (
                    admission.service_time_current_duration_ms
                ),
                "service_time_calibrated_future_requests": (
                    admission.service_time_calibrated_future_requests
                ),
            }
        )
    return row


def _execute_managed_rollout(
    url: str,
    submissions: tuple[RolloutSubmission, ...],
    *,
    timeout_s: float,
    stream: bool,
    emit_hints: bool,
    max_parallel: int | None,
    max_inflight: int | None,
    queue_policy: str,
    elastic_kv_config: ElasticKVConfig | None,
    allow_deadline_overcommit: bool,
    max_submit_lateness_ms: float,
    clock: ReplayClock,
    generate: Callable[[str, dict[str, object], float, bool], dict[str, object]],
    poll_interval_s: float,
    service_time_estimator: ServiceTimeEstimator | None = None,
    cohort_barrier_timeout_s: float = 5.0,
    preannounce_lead_ms: int = 0,
) -> tuple[
    list[dict[str, object]],
    list[str],
    dict[str, int],
]:
    """Dispatch through an observed in-flight set and aggregate byte budget."""

    if queue_policy not in ("timeline", "deadline"):
        raise ValueError("queue_policy must be timeline or deadline")
    if poll_interval_s <= 0 or not math.isfinite(poll_interval_s):
        raise ValueError("poll_interval_s must be finite and positive")
    if max_inflight is not None and (
        isinstance(max_inflight, bool)
        or not isinstance(max_inflight, int)
        or max_inflight < 1
    ):
        raise ValueError("max_inflight must be a positive integer")
    configured_parallel = max_parallel or len(submissions)
    worker_count = max(configured_parallel, max_inflight or 1)
    origin = clock.now()
    queue = list(submissions)
    pending: list[_PendingRequest] = []
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    scheduler_stats = {
        "selection_passes": 0,
        "capacity_blocked_groups": 0,
        "budget_blocked_groups": 0,
        "head_of_line_capacity_bypasses": 0,
        "head_of_line_budget_bypasses": 0,
        "deadline_forced_groups": 0,
        "dispatched_groups": 0,
    }

    def harvest() -> None:
        for item in tuple(pending):
            if not item.future.done():
                continue
            pending.remove(item)
            try:
                record = item.future.result()
                row = _measure_result(
                    item.submission,
                    record,
                    origin=origin,
                    dispatch_lateness_ms=item.dispatch_lateness_ms,
                    capacity_delay_ms=item.capacity_delay_ms,
                    admission=item.admission,
                    max_submit_lateness_ms=max_submit_lateness_ms,
                    strict_submit_timing=False,
                )
                row["service_time_prefill_tokens_per_s_hint"] = (
                    item.prefill_tokens_per_s_hint
                )
                row["service_time_decode_tokens_per_s_hint"] = (
                    item.decode_tokens_per_s_hint
                )
                row["service_time_decode_feedback_residence"] = (
                    item.decode_feedback_residence
                )
                release_s = item.prefill_release_perf_s
                response_finished_s = float(record["_started_perf_s"]) + float(
                    record["end_to_end_s"]
                )
                has_prefill_workspace = bool(
                    item.admission is not None
                    and item.admission.prefill_workspace_bytes > 0
                )
                row["prefill_workspace_released_early"] = (
                    None
                    if not has_prefill_workspace
                    else (
                        release_s is not None
                        and release_s < response_finished_s
                    )
                )
                row["prefill_workspace_release_after_ms"] = (
                    None
                    if release_s is None or not has_prefill_workspace
                    else max(
                        0.0,
                        (release_s - float(record["_started_perf_s"]))
                        * 1000.0,
                    )
                )
                if service_time_estimator is not None:
                    try:
                        service_time_estimator.observe(
                            prompt_tokens=item.submission.prompt_tokens,
                            output_tokens=item.submission.output_tokens,
                            ttft_s=float(record["ttft_s"]),
                            tpot_s=float(record["tpot_s"]),
                            residence=_service_time_residence(
                                item.admission.chosen_residence
                                if item.admission is not None
                                else FutureHint.from_payload(
                                    dict(item.submission.hint)
                                ).residence
                            ),
                            kind=item.submission.kind,
                            prefill_batch_size=(
                                item.submission.prefill_batch_size
                            ),
                            decode_batch_size=item.submission.decode_batch_size,
                        )
                    except (TypeError, ValueError):
                        service_time_estimator.record_rejected_sample()
                rows.append(row)
            except Exception as exc:  # noqa: BLE001 - worker failures are evidence
                failures.append(
                    f"{item.submission.request_id}: {type(exc).__name__}: {exc}"
                )

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        while queue or pending:
            harvest()
            now_ms = (clock.now() - origin) * 1000.0
            if not queue:
                clock.sleep(poll_interval_s)
                continue
            due = [
                item
                for item in queue
                if _announce_time_ms(item, preannounce_lead_ms) <= now_ms
            ]
            if not due:
                next_ms = min(
                    _announce_time_ms(item, preannounce_lead_ms)
                    for item in queue
                )
                clock.sleep(
                    max(0.0, min(poll_interval_s, (next_ms - now_ms) / 1000.0))
                )
                continue
            if max_inflight is not None and len(pending) >= max_inflight:
                clock.sleep(poll_interval_s)
                continue

            cohorts = _ordered_due_cohorts(
                due,
                queue,
                queue_policy,
                service_time_estimator=service_time_estimator,
            )
            capacity = worker_count if max_inflight is None else max_inflight
            active_bytes = sum(item.hold_bytes for item in pending)
            scheduler_stats["selection_passes"] += 1
            selected: tuple[
                tuple[RolloutSubmission, ...],
                dict[str, ElasticAdmissionDecision | None],
                bool,
            ] | None = None
            first_block: tuple[str, list[RolloutSubmission]] | None = None
            for candidate_index, cohort in enumerate(cohorts):
                if len(pending) + len(cohort) > capacity:
                    scheduler_stats["capacity_blocked_groups"] += 1
                    if first_block is None:
                        first_block = ("capacity", list(cohort))
                    continue
                (
                    admissions,
                    deferred,
                    cohort_deadline_forced,
                ) = _admit_managed_cohort(
                    cohort,
                    queue=queue,
                    active_bytes=active_bytes,
                    now_ms=now_ms,
                    config=elastic_kv_config,
                    allow_deadline_overcommit=allow_deadline_overcommit,
                    prefill_tokens_per_s=None,
                    decode_tokens_per_s=None,
                    decode_tokens_per_s_by_residence={},
                    service_time_estimator=service_time_estimator,
                )
                if deferred and not cohort_deadline_forced:
                    scheduler_stats["budget_blocked_groups"] += 1
                    if first_block is None:
                        first_block = ("budget", deferred)
                    continue
                if candidate_index > 0 and first_block is not None:
                    blocked_reason, _ = first_block
                    if blocked_reason == "capacity":
                        scheduler_stats[
                            "head_of_line_capacity_bypasses"
                        ] += 1
                    else:
                        scheduler_stats[
                            "head_of_line_budget_bypasses"
                        ] += 1
                if cohort_deadline_forced:
                    scheduler_stats["deadline_forced_groups"] += 1
                selected = (cohort, admissions, cohort_deadline_forced)
                break
            if selected is None:
                if pending:
                    clock.sleep(poll_interval_s)
                    continue
                if first_block is not None and first_block[0] == "budget":
                    raise RolloutExecutionError(
                        f"{first_block[1][0].request_id}: bounded KV hot set plus "
                        "prefill workspace exceeds the total elastic budget"
                    )
                raise RolloutExecutionError(
                    "managed scheduler made no progress with an empty in-flight set"
                )
            cohort, admissions, _ = selected
            scheduler_stats["dispatched_groups"] += 1

            dispatch_ms = (clock.now() - origin) * 1000.0
            dispatch_lateness_ms = dispatch_ms - cohort[0].submit_at_ms
            capacity_delay_ms = max(0.0, dispatch_lateness_ms)
            dispatch_barrier = (
                threading.Barrier(len(cohort)) if len(cohort) > 1 else None
            )
            synchronized_generate = _barrier_generate(
                cohort[0].cohort_id,
                dispatch_barrier,
                cohort_barrier_timeout_s,
                generate,
            )
            dispatched_ids: set[str] = set()
            for member in cohort:
                admission = admissions[member.request_id]
                spec: dict[str, object] = {
                    "request_id": member.request_id,
                    "input_tokens": member.prompt_tokens,
                    "requested_output_tokens": member.output_tokens,
                    "kind": member.kind,
                }
                emitted_residence = (
                    admission.chosen_residence
                    if admission is not None
                    else FutureHint.from_payload(dict(member.hint)).residence
                )
                emitted_feedback = _submission_service_rates(
                    service_time_estimator,
                    member,
                    residence=emitted_residence,
                )
                if emit_hints:
                    dispatch_clock_s = clock.now()
                    deadline_after_ms = (
                        None
                        if member.deadline_ms is None
                        else max(0, member.deadline_ms - dispatch_ms)
                    )
                    spec["future_hint"] = make_kv_hints(
                        member,
                        deadline_after_ms=(
                            None
                            if deadline_after_ms is None
                            else _ceil_milliseconds(deadline_after_ms)
                        ),
                        release_after_ms=_ceil_milliseconds(
                            max(0.0, member.submit_at_ms - dispatch_ms)
                        ),
                        clock_now_s=dispatch_clock_s,
                        residence_override=(
                            None
                            if admission is None
                            or admission.chosen_residence
                            == FutureHint.from_payload(dict(member.hint)).residence
                            else admission.chosen_residence
                        ),
                        prefill_tokens_per_s=emitted_feedback.get(
                            "prefill_tokens_per_s"
                        ),
                        decode_tokens_per_s=emitted_feedback.get(
                            "decode_tokens_per_s"
                        ),
                    )
                pending_item = _PendingRequest(
                    submission=member,
                    future=None,
                    dispatch_lateness_ms=dispatch_lateness_ms,
                    capacity_delay_ms=capacity_delay_ms,
                    admission=admission,
                    prefill_tokens_per_s_hint=emitted_feedback.get(
                        "prefill_tokens_per_s"
                    ),
                    decode_tokens_per_s_hint=emitted_feedback.get(
                        "decode_tokens_per_s"
                    ),
                    decode_feedback_residence=_service_time_residence(
                        emitted_residence
                    ),
                )
                spec[FIRST_TOKEN_CALLBACK_KEY] = pending_item.mark_prefill_released
                future = pool.submit(
                    synchronized_generate, url, spec, timeout_s, stream
                )
                pending_item.future = future
                pending.append(pending_item)
                dispatched_ids.add(member.request_id)
            queue = [
                item for item in queue if item.request_id not in dispatched_ids
            ]
        harvest()
    return rows, failures, scheduler_stats


def _finalize_rollout(
    rows: list[dict[str, object]],
    submissions: tuple[RolloutSubmission, ...],
    *,
    clock: ReplayClock,
    residency_stats_dir: str | Path | None,
    require_residency_evidence: bool,
    residency_evidence_timeout_s: float,
    max_cohort_start_skew_ms: float,
) -> dict[str, object]:
    rows.sort(key=lambda row: (int(row["actual_submit_ms"]), str(row["request_id"])))
    residency: dict[str, object] | None = None
    if residency_stats_dir is not None:
        request_ids = {str(row["request_id"]) for row in rows}
        evidence_deadline = clock.now() + residency_evidence_timeout_s
        while True:
            residency = collect_residency_evidence(residency_stats_dir)
            if _residency_evidence_ready(residency, request_ids):
                break
            remaining = evidence_deadline - clock.now()
            if remaining <= 0:
                break
            clock.sleep(min(0.05, remaining))
        assert residency is not None
        if require_residency_evidence and not _residency_evidence_ready(
            residency, request_ids
        ):
            raise RolloutExecutionError(
                "residency evidence incomplete: "
                f"{residency.get('reason', 'missing request or TP-rank decision')}"
            )
        if residency.get("available"):
            decisions = _residency_request_rows(residency)
            for row in rows:
                evidence = decisions.get(row["request_id"])
                if not isinstance(evidence, Mapping):
                    if require_residency_evidence:
                        raise RolloutExecutionError(
                            f"{row['request_id']}: missing per-request residency evidence"
                        )
                    row["actual_residence"] = None
                    row["residency_match"] = None
                    continue
                if not evidence.get("all_ranks_agree"):
                    if evidence.get("actual_residence") == "conflict":
                        raise RolloutExecutionError(
                            f"{row['request_id']}: TP ranks disagree on residency"
                        )
                    row["actual_residence"] = None
                    row["residency_match"] = None
                    row["residency_incomplete"] = True
                    continue
                actual = evidence.get("actual_residence")
                row["actual_residence"] = actual
                row["residency_match"] = actual == row["runtime_residence"]
                row["residency_rank_count"] = evidence.get("rank_count")

    deadline_rows = [row for row in rows if row["deadline_target_ms"] is not None]
    cohorts: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        cohort_id = row.get("cohort_id")
        if isinstance(cohort_id, str):
            cohorts.setdefault(cohort_id, []).append(row)
    violations: list[str] = []
    max_observed_skew_ms = 0.0
    for cohort_id, members in cohorts.items():
        starts = [float(row["actual_submit_ms"]) for row in members]
        skew_ms = max(starts) - min(starts)
        max_observed_skew_ms = max(max_observed_skew_ms, skew_ms)
        violated = len(members) > 1 and skew_ms > max_cohort_start_skew_ms
        if violated:
            violations.append(
                f"{cohort_id}: actual cohort start skew {skew_ms:.1f} ms "
                f"exceeds {max_cohort_start_skew_ms:.1f} ms"
            )
        for row in members:
            row["cohort_member_count"] = len(members)
            row["cohort_actual_start_skew_ms"] = skew_ms
            row["cohort_start_skew_ok"] = not violated
    if violations:
        raise RolloutExecutionError("; ".join(violations))
    summary = {
        "request_count": len(rows),
        "completed": len(rows) == len(submissions),
        "max_submit_lateness_ms": max(float(row["submit_lateness_ms"]) for row in rows),
        "max_actual_preannounce_lead_ms": max(
            float(row["preannounce_lead_actual_ms"]) for row in rows
        ),
        "max_capacity_delay_ms": max(
            float(row.get("capacity_delay_ms", 0.0)) for row in rows
        ),
        "max_abs_finish_error_ms": max(abs(float(row["finish_error_ms"])) for row in rows),
        "deadline_count": len(deadline_rows),
        "deadline_hits": sum(row["actual_deadline_hit"] is True for row in deadline_rows),
        "deadline_misses": sum(row["actual_deadline_hit"] is False for row in deadline_rows),
        "scheduled_misses": sum(
            row["planned_deadline_hit"] is False for row in deadline_rows
        ),
        "residency_mismatches": sum(
            row.get("residency_match") is False for row in rows
        ),
        "runtime_demotions": sum(
            row.get("planned_residence") == "resident"
            and row.get("runtime_residence") == "host_backed"
            for row in rows
        ),
        "over_budget_admissions": sum(
            row.get("elastic_over_budget") is True for row in rows
        ),
        "future_reservation_admissions": sum(
            int(row.get("elastic_future_reserved_requests", 0)) > 0 for row in rows
        ),
        "future_reservation_demotions": sum(
            row.get("elastic_admission_reason")
            in ("imminent-future-demotion", "imminent-future-prefill-demotion")
            for row in rows
        ),
        "prefill_headroom_demotions": sum(
            row.get("elastic_admission_reason") == "prefill-headroom-demotion"
            for row in rows
        ),
        "prefill_workspace_released_early": sum(
            row.get("prefill_workspace_released_early") is True for row in rows
        ),
        "prefill_workspace_held_until_completion": sum(
            row.get("prefill_workspace_released_early") is False for row in rows
        ),
        "cohort_count": len(cohorts),
        "batched_cohort_count": sum(
            len(members) > 1 for members in cohorts.values()
        ),
        "batched_request_count": sum(
            len(members) for members in cohorts.values() if len(members) > 1
        ),
        "max_cohort_actual_start_skew_ms": max_observed_skew_ms,
        "cohort_start_skew_violations": 0,
        "efficiency": _efficiency(rows),
    }
    return {
        "requests": rows,
        "summary": summary,
        "residency_evidence": residency,
    }


def execute_rollout(
    url: str,
    rollout: RolloutPlan | Iterable[RolloutSubmission],
    *,
    timeout_s: float = 180.0,
    stream: bool = True,
    emit_hints: bool = True,
    max_parallel: int | None = None,
    max_submit_lateness_ms: float = 100.0,
    clock: ReplayClock | None = None,
    generator: Callable[[str, dict[str, object], float, bool], dict[str, object]] | None = None,
    residency_stats_dir: str | Path | None = None,
    require_residency_evidence: bool = False,
    residency_evidence_timeout_s: float = 2.0,
    max_inflight: int | None = None,
    queue_policy: str = "timeline",
    elastic_kv_config: ElasticKVConfig | None = None,
    allow_deadline_overcommit: bool = True,
    poll_interval_ms: float = 10.0,
    service_time_estimator: ServiceTimeEstimator | None = None,
    cohort_barrier_timeout_s: float = 5.0,
    max_cohort_start_skew_ms: float = 100.0,
    preannounce_lead_ms: int = 0,
) -> dict[str, object]:
    """Run every planned request and return measured, JSON-safe evidence."""

    timing_values = (
        timeout_s,
        max_submit_lateness_ms,
        residency_evidence_timeout_s,
        cohort_barrier_timeout_s,
        max_cohort_start_skew_ms,
        poll_interval_ms,
    )
    if any(not math.isfinite(value) for value in timing_values):
        raise ValueError("timing parameters must be finite")
    if (
        timeout_s <= 0
        or max_submit_lateness_ms < 0
        or residency_evidence_timeout_s < 0
        or cohort_barrier_timeout_s <= 0
        or max_cohort_start_skew_ms < 0
    ):
        raise ValueError(
            "request and barrier timeouts must be positive; timing tolerances "
            "must be nonnegative"
        )
    submissions = validate_submissions(
        rollout.submissions if isinstance(rollout, RolloutPlan) else rollout
    )
    replay_clock = clock or ReplayClock()
    generate = generator or _default_generate
    execution_groups = _execution_groups(submissions)
    max_cohort_size = max(len(group) for group in execution_groups)
    worker_count = (
        len(submissions) if max_parallel is None else max_parallel
    )
    if worker_count < 1:
        raise ValueError("max_parallel must be positive")
    if worker_count < max_cohort_size:
        raise ValueError(
            "max_parallel must be at least the largest cohort size"
        )
    if max_inflight is not None and (
        isinstance(max_inflight, bool)
        or not isinstance(max_inflight, int)
        or max_inflight < 1
    ):
        raise ValueError("max_inflight must be a positive integer")
    if max_inflight is not None and max_inflight < max_cohort_size:
        raise ValueError(
            "max_inflight must be at least the largest cohort size"
        )
    if queue_policy not in ("timeline", "deadline"):
        raise ValueError("queue_policy must be timeline or deadline")
    if poll_interval_ms <= 0 or not math.isfinite(poll_interval_ms):
        raise ValueError("poll_interval_ms must be finite and positive")
    if (
        isinstance(preannounce_lead_ms, bool)
        or not isinstance(preannounce_lead_ms, int)
        or not 0 <= preannounce_lead_ms <= 60_000
    ):
        raise ValueError("preannounce_lead_ms must be an integer between 0 and 60000")
    if preannounce_lead_ms and not emit_hints:
        raise ValueError("preannouncement requires emitted future hints")
    managed = (
        service_time_estimator is not None
        or max_inflight is not None
        or queue_policy != "timeline"
        or elastic_kv_config is not None
    )
    if managed:
        rows, failures, scheduler_stats = _execute_managed_rollout(
            url,
            submissions,
            timeout_s=timeout_s,
            stream=stream,
            emit_hints=emit_hints,
            max_parallel=max_parallel,
            max_inflight=max_inflight,
            queue_policy=queue_policy,
            elastic_kv_config=elastic_kv_config,
            allow_deadline_overcommit=allow_deadline_overcommit,
            max_submit_lateness_ms=max_submit_lateness_ms,
            clock=replay_clock,
            generate=generate,
            poll_interval_s=poll_interval_ms / 1000.0,
            service_time_estimator=service_time_estimator,
            cohort_barrier_timeout_s=cohort_barrier_timeout_s,
            preannounce_lead_ms=preannounce_lead_ms,
        )
        if failures:
            raise RolloutExecutionError("; ".join(failures))
        result = _finalize_rollout(
            rows,
            submissions,
            clock=replay_clock,
            residency_stats_dir=residency_stats_dir,
            require_residency_evidence=require_residency_evidence,
            residency_evidence_timeout_s=residency_evidence_timeout_s,
            max_cohort_start_skew_ms=max_cohort_start_skew_ms,
        )
        result["summary"]["managed_scheduler"] = scheduler_stats
        result["summary"]["preannounce_lead_ms"] = preannounce_lead_ms
        if service_time_estimator is not None:
            result["summary"]["service_time_estimator"] = (
                service_time_estimator.snapshot().as_dict()
            )
        return result
    origin = replay_clock.now()
    pending = []
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        for group in execution_groups:
            head = group[0]
            announce_at_ms = _announce_time_ms(head, preannounce_lead_ms)
            _wait_until(origin + announce_at_ms / 1000.0, replay_clock)
            dispatch_s = replay_clock.now() - origin
            dispatch_lateness_ms = dispatch_s * 1000.0 - head.submit_at_ms
            if dispatch_lateness_ms > max_submit_lateness_ms:
                raise RolloutExecutionError(
                    f"{head.request_id}: dispatch late by {dispatch_lateness_ms:.1f} ms"
                )
            dispatch_barrier = (
                threading.Barrier(len(group)) if len(group) > 1 else None
            )
            synchronized_generate = _barrier_generate(
                head.cohort_id,
                dispatch_barrier,
                cohort_barrier_timeout_s,
                generate,
            )
            for submission in group:
                spec: dict[str, object] = {
                    "request_id": submission.request_id,
                    "input_tokens": submission.prompt_tokens,
                    "requested_output_tokens": submission.output_tokens,
                    "kind": submission.kind,
                }
                if emit_hints:
                    dispatch_clock_s = replay_clock.now()
                    deadline_after_ms = (
                        None
                        if submission.deadline_ms is None
                        else max(0, submission.deadline_ms - dispatch_s * 1000.0)
                    )
                    spec["future_hint"] = make_kv_hints(
                        submission,
                        deadline_after_ms=(
                            None
                            if deadline_after_ms is None
                            else _ceil_milliseconds(deadline_after_ms)
                        ),
                        release_after_ms=_ceil_milliseconds(
                            max(0.0, submission.submit_at_ms - dispatch_s * 1000.0)
                        ),
                        clock_now_s=dispatch_clock_s,
                    )
                future = pool.submit(
                    synchronized_generate, url, spec, timeout_s, stream
                )
                pending.append((submission, dispatch_lateness_ms, future))

        rows: list[dict[str, object]] = []
        failures: list[str] = []
        for submission, dispatch_lateness_ms, future in pending:
            try:
                record = future.result()
                rows.append(
                    _measure_result(
                        submission,
                        record,
                        origin=origin,
                        dispatch_lateness_ms=dispatch_lateness_ms,
                        capacity_delay_ms=0.0,
                        admission=None,
                        max_submit_lateness_ms=max_submit_lateness_ms,
                        strict_submit_timing=True,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - worker failures are evidence
                failures.append(
                    f"{submission.request_id}: {type(exc).__name__}: {exc}"
                )

    if failures:
        raise RolloutExecutionError("; ".join(failures))
    result = _finalize_rollout(
        rows,
        submissions,
        clock=replay_clock,
        residency_stats_dir=residency_stats_dir,
        require_residency_evidence=require_residency_evidence,
        residency_evidence_timeout_s=residency_evidence_timeout_s,
        max_cohort_start_skew_ms=max_cohort_start_skew_ms,
    )
    result["summary"]["preannounce_lead_ms"] = preannounce_lead_ms
    return result
