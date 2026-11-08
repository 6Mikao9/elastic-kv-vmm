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
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

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
) -> dict[str, object]:
    """Wrap a planner payload in the SGLang versioned hint envelope."""

    payload = _future_payload(submission.hint)
    if submission.deadline_ms is not None:
        payload["deadline_after_ms"] = (
            max(0, submission.deadline_ms - submission.submit_at_ms)
            if deadline_after_ms is None
            else deadline_after_ms
        )
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
