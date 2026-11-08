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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .elastic_admission import ElasticAdmissionDecision, ElasticKVConfig, decide_elastic_admission
from .future_plan import RolloutPlan, RolloutSubmission
from .sglang_future_hook import FutureHint


class RolloutExecutionError(RuntimeError):
    """A live plan violated a transport, timing, or evidence invariant."""


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


def make_kv_hints(
    submission: RolloutSubmission,
    *,
    deadline_after_ms: int | None = None,
    residence_override: str | None = None,
) -> dict[str, object]:
    """Wrap a planner payload in the SGLang versioned hint envelope."""

    payload = _future_payload(submission.hint)
    if submission.deadline_ms is not None:
        payload["deadline_after_ms"] = (
            max(0, submission.deadline_ms - submission.submit_at_ms)
            if deadline_after_ms is None
            else deadline_after_ms
        )
    if residence_override is not None:
        if residence_override not in ("resident", "host_backed"):
            raise RolloutExecutionError(
                f"invalid runtime residence override: {residence_override!r}"
            )
        payload["residence"] = residence_override
    request_id = submission.request_id
    return {
        "protocol_version": "1",
        "message_id": request_id,
        "actions": [
            {
                "action_id": f"{request_id}:future",
                "action_type": "elastic.future",
                "action_version": "1",
                "payload": payload,
            }
        ],
    }


def validate_submissions(submissions: Iterable[RolloutSubmission]) -> tuple[RolloutSubmission, ...]:
    """Return stable submission order after checking all plan invariants."""

    ordered = tuple(submissions)
    if not ordered:
        raise RolloutExecutionError("rollout contains no submissions")
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
    ids = [item.request_id for item in ordered]
    if len(set(ids)) != len(ids):
        raise RolloutExecutionError("rollout request IDs must be unique")
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

    return generate(url, spec, timeout, stream)


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
    future: Any
    dispatch_lateness_ms: float
    capacity_delay_ms: float
    admission: ElasticAdmissionDecision | None

    @property
    def hold_bytes(self) -> int:
        if self.admission is None:
            return 0
        return self.admission.active_after_bytes - self.admission.active_before_bytes


def _select_due(
    due: Sequence[RolloutSubmission], policy: str
) -> RolloutSubmission:
    if policy == "timeline":
        return min(due, key=lambda item: (item.submit_at_ms, item.request_id))
    if policy != "deadline":
        raise ValueError("queue policy must be timeline or deadline")

    def key(item: RolloutSubmission) -> tuple[object, ...]:
        hint = FutureHint.from_payload(dict(item.hint))
        if item.deadline_ms is not None:
            predicted_duration = max(1, item.estimated_finish_ms - item.submit_at_ms)
            return (
                0,
                item.deadline_ms,
                item.deadline_ms - item.submit_at_ms - predicted_duration,
                -hint.weight,
                item.submit_at_ms,
                item.request_id,
            )
        return (1, 0, 0, -hint.weight, item.submit_at_ms, item.request_id)

    return min(due, key=key)


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
                "elastic_future_reserved_requests": admission.future_reserved_requests,
                "elastic_future_effective_after_bytes": (
                    admission.future_effective_after_bytes
                ),
                "elastic_future_over_budget": admission.future_over_budget,
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
) -> tuple[list[dict[str, object]], list[str]]:
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

    def harvest() -> None:
        for item in tuple(pending):
            if not item.future.done():
                continue
            pending.remove(item)
            try:
                record = item.future.result()
                rows.append(
                    _measure_result(
                        item.submission,
                        record,
                        origin=origin,
                        dispatch_lateness_ms=item.dispatch_lateness_ms,
                        capacity_delay_ms=item.capacity_delay_ms,
                        admission=item.admission,
                        max_submit_lateness_ms=max_submit_lateness_ms,
                        strict_submit_timing=False,
                    )
                )
            except Exception as exc:
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
            due = [item for item in queue if item.submit_at_ms <= now_ms]
            if not due:
                next_ms = min(item.submit_at_ms for item in queue)
                clock.sleep(
                    max(0.0, min(poll_interval_s, (next_ms - now_ms) / 1000.0))
                )
                continue
            if max_inflight is not None and len(pending) >= max_inflight:
                clock.sleep(poll_interval_s)
                continue

            submission = _select_due(due, queue_policy)
            active_bytes = sum(item.hold_bytes for item in pending)
            admission = None
            if elastic_kv_config is not None:
                admission = decide_elastic_admission(
                    submission,
                    active_bytes=active_bytes,
                    config=elastic_kv_config,
                    future_submissions=queue,
                    now_ms=now_ms,
                )
                if admission.defer_for_budget:
                    predicted_duration = max(
                        1, submission.estimated_finish_ms - submission.submit_at_ms
                    )
                    deadline_forced = bool(
                        allow_deadline_overcommit
                        and submission.deadline_ms is not None
                        and now_ms + predicted_duration > submission.deadline_ms
                    )
                    if pending and not deadline_forced:
                        clock.sleep(poll_interval_s)
                        continue
                    if not pending and not deadline_forced:
                        raise RolloutExecutionError(
                            f"{submission.request_id}: sparse hot set exceeds the total elastic KV budget"
                        )

            dispatch_ms = (clock.now() - origin) * 1000.0
            dispatch_lateness_ms = dispatch_ms - submission.submit_at_ms
            capacity_delay_ms = max(0.0, dispatch_lateness_ms)
            spec: dict[str, object] = {
                "request_id": submission.request_id,
                "input_tokens": submission.prompt_tokens,
                "requested_output_tokens": submission.output_tokens,
                "kind": submission.kind,
            }
            if emit_hints:
                deadline_after_ms = (
                    None
                    if submission.deadline_ms is None
                    else max(0, submission.deadline_ms - dispatch_ms)
                )
                spec["future_hint"] = make_kv_hints(
                    submission,
                    deadline_after_ms=(
                        None if deadline_after_ms is None else int(deadline_after_ms)
                    ),
                    residence_override=(
                        None
                        if admission is None
                        or admission.chosen_residence
                        == FutureHint.from_payload(dict(submission.hint)).residence
                        else admission.chosen_residence
                    ),
                )
            queue.remove(submission)
            future = pool.submit(generate, url, spec, timeout_s, stream)
            pending.append(
                _PendingRequest(
                    submission=submission,
                    future=future,
                    dispatch_lateness_ms=dispatch_lateness_ms,
                    capacity_delay_ms=capacity_delay_ms,
                    admission=admission,
                )
            )
        harvest()
    return rows, failures


def _finalize_rollout(
    rows: list[dict[str, object]],
    submissions: tuple[RolloutSubmission, ...],
    *,
    clock: ReplayClock,
    residency_stats_dir: str | Path | None,
    require_residency_evidence: bool,
    residency_evidence_timeout_s: float,
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
    summary = {
        "request_count": len(rows),
        "completed": len(rows) == len(submissions),
        "max_submit_lateness_ms": max(float(row["submit_lateness_ms"]) for row in rows),
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
            row.get("elastic_admission_reason") == "imminent-future-demotion"
            for row in rows
        ),
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
) -> dict[str, object]:
    """Run every planned request and return measured, JSON-safe evidence."""

    if timeout_s <= 0 or max_submit_lateness_ms < 0 or residency_evidence_timeout_s < 0:
        raise ValueError(
            "request timeout must be positive; timing tolerances must be nonnegative"
        )
    submissions = validate_submissions(
        rollout.submissions if isinstance(rollout, RolloutPlan) else rollout
    )
    replay_clock = clock or ReplayClock()
    generate = generator or _default_generate
    worker_count = len(submissions) if max_parallel is None else max_parallel
    if worker_count < 1:
        raise ValueError("max_parallel must be positive")
    if max_inflight is not None and (
        isinstance(max_inflight, bool)
        or not isinstance(max_inflight, int)
        or max_inflight < 1
    ):
        raise ValueError("max_inflight must be a positive integer")
    if queue_policy not in ("timeline", "deadline"):
        raise ValueError("queue_policy must be timeline or deadline")
    if poll_interval_ms <= 0 or not math.isfinite(poll_interval_ms):
        raise ValueError("poll_interval_ms must be finite and positive")
    managed = (
        max_inflight is not None
        or queue_policy != "timeline"
        or elastic_kv_config is not None
    )
    if managed:
        rows, failures = _execute_managed_rollout(
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
        )
        if failures:
            raise RolloutExecutionError("; ".join(failures))
        return _finalize_rollout(
            rows,
            submissions,
            clock=replay_clock,
            residency_stats_dir=residency_stats_dir,
            require_residency_evidence=require_residency_evidence,
            residency_evidence_timeout_s=residency_evidence_timeout_s,
        )
    origin = replay_clock.now()
    pending = []
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        for submission in submissions:
            _wait_until(origin + submission.submit_at_ms / 1000.0, replay_clock)
            dispatch_s = replay_clock.now() - origin
            dispatch_lateness_ms = dispatch_s * 1000.0 - submission.submit_at_ms
            if dispatch_lateness_ms > max_submit_lateness_ms:
                raise RolloutExecutionError(
                    f"{submission.request_id}: dispatch late by {dispatch_lateness_ms:.1f} ms"
                )
            spec: dict[str, object] = {
                "request_id": submission.request_id,
                "input_tokens": submission.prompt_tokens,
                "requested_output_tokens": submission.output_tokens,
                "kind": submission.kind,
            }
            if emit_hints:
                deadline_after_ms = (
                    None
                    if submission.deadline_ms is None
                    else max(0, submission.deadline_ms - dispatch_s * 1000.0)
                )
                spec["future_hint"] = make_kv_hints(
                    submission,
                    deadline_after_ms=(
                        None if deadline_after_ms is None else int(deadline_after_ms)
                    ),
                )
            future = pool.submit(generate, url, spec, timeout_s, stream)
            pending.append((submission, dispatch_lateness_ms, future))

        rows: list[dict[str, object]] = []
        failures: list[str] = []
        for submission, dispatch_lateness_ms, future in pending:
            try:
                record = future.result()
            except Exception as exc:
                failures.append(f"{submission.request_id}: {type(exc).__name__}: {exc}")
                continue
            for key in ("_started_perf_s", "end_to_end_s", "output_tokens", "ttft_s", "tpot_s"):
                if key not in record:
                    failures.append(f"{submission.request_id}: result lacks {key}")
                    break
            else:
                if int(record["output_tokens"]) != submission.output_tokens:
                    failures.append(
                        f"{submission.request_id}: output token count differs from plan"
                    )
                    continue
                actual_submit_ms = (float(record["_started_perf_s"]) - origin) * 1000.0
                submit_lateness_ms = actual_submit_ms - submission.submit_at_ms
                if submit_lateness_ms > max_submit_lateness_ms:
                    failures.append(
                        f"{submission.request_id}: actual submit late by {submit_lateness_ms:.1f} ms"
                    )
                    continue
                actual_finish_ms = actual_submit_ms + float(record["end_to_end_s"]) * 1000.0
                deadline_target_ms = _deadline_target_ms(submission)
                if deadline_target_ms is not None:
                    planned_deadline_hit = submission.estimated_finish_ms <= deadline_target_ms
                    actual_deadline_hit = actual_finish_ms <= deadline_target_ms
                else:
                    planned_deadline_hit = actual_deadline_hit = None
                rows.append(
                    {
                        "request_id": submission.request_id,
                        "kind": submission.kind,
                        "planned_arrival_ms": submission.planned_arrival_ms,
                        "planned_submit_ms": submission.submit_at_ms,
                        "planned_finish_ms": submission.estimated_finish_ms,
                        "actual_submit_ms": actual_submit_ms,
                        "actual_finish_ms": actual_finish_ms,
                        "submit_lateness_ms": submit_lateness_ms,
                        "dispatch_lateness_ms": dispatch_lateness_ms,
                        "finish_error_ms": actual_finish_ms - submission.estimated_finish_ms,
                        "deadline_target_ms": deadline_target_ms,
                        "planned_deadline_hit": planned_deadline_hit,
                        "actual_deadline_hit": actual_deadline_hit,
                        "planned_residence": FutureHint.from_payload(
                            submission.hint
                        ).residence,
                        "prompt_tokens": submission.prompt_tokens,
                        "output_tokens": int(record["output_tokens"]),
                        "ttft_s": record["ttft_s"],
                        "tpot_s": record["tpot_s"],
                        "end_to_end_s": float(record["end_to_end_s"]),
                    }
                )

    if failures:
        raise RolloutExecutionError("; ".join(failures))
    rows.sort(key=lambda row: (int(row["actual_submit_ms"]), str(row["request_id"])))

    residency: dict[str, object] | None = None
    if residency_stats_dir is not None:
        request_ids = {str(row["request_id"]) for row in rows}
        evidence_deadline = replay_clock.now() + residency_evidence_timeout_s
        while True:
            residency = collect_residency_evidence(residency_stats_dir)
            if _residency_evidence_ready(
                residency,
                request_ids,
            ):
                break
            remaining = evidence_deadline - replay_clock.now()
            if remaining <= 0:
                break
            replay_clock.sleep(min(0.05, remaining))
        assert residency is not None
        if require_residency_evidence and not _residency_evidence_ready(
            residency,
            request_ids,
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
                row["residency_match"] = actual == row["planned_residence"]
                row["residency_rank_count"] = evidence.get("rank_count")

    deadline_rows = [row for row in rows if row["deadline_target_ms"] is not None]
    summary = {
        "request_count": len(rows),
        "completed": len(rows) == len(submissions),
        "max_submit_lateness_ms": max(float(row["submit_lateness_ms"]) for row in rows),
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
        "efficiency": _efficiency(rows),
    }
    return {
        "requests": rows,
        "summary": summary,
        "residency_evidence": residency,
    }
