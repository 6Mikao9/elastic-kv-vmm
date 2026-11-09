"""Compare planned decode width with observed SGLang batch occupancy."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _histogram(value: object, request_id: str) -> dict[int, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            f"decode batch histogram for {request_id} must be an object"
        )
    histogram: dict[int, int] = {}
    for raw_size, raw_count in value.items():
        try:
            size = int(raw_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"decode batch size for {request_id} must be an integer"
            ) from exc
        if str(size) != str(raw_size):
            raise ValueError(
                f"decode batch size for {request_id} must be an integer"
            )
        histogram[_positive_integer(size, "decode batch size")] = (
            _positive_integer(raw_count, "decode batch count")
        )
    return histogram


def _trace_summary(value: object) -> dict[str, object]:
    if value is None:
        return {
            "event_count": 0,
            "mixed_event_count": 0,
            "batch_size_histogram": {},
            "start_s": None,
            "end_s": None,
            "duration_s": None,
            "attributed_duration_s": 0.0,
            "request_occupancy": [],
        }
    if not isinstance(value, list):
        raise TypeError("decode trace must be a list")
    times: list[float] = []
    sizes: dict[int, int] = {}
    mixed = 0
    normalized_events: list[tuple[float, int, tuple[str, ...]]] = []
    for index, event in enumerate(value):
        if not isinstance(event, Mapping):
            raise TypeError(f"decode trace event {index} must be an object")
        observed_at_s = event.get("observed_at_s")
        if (
            isinstance(observed_at_s, bool)
            or not isinstance(observed_at_s, (int, float))
            or not math.isfinite(observed_at_s)
        ):
            raise ValueError(
                f"decode trace event {index} has an invalid observation time"
            )
        times.append(float(observed_at_s))
        size = _positive_integer(event.get("batch_size"), "decode trace batch size")
        sizes[size] = sizes.get(size, 0) + 1
        mixed += int(event.get("mixed_prefill_decode") is True)
        raw_request_ids = event.get("request_ids", [])
        if not isinstance(raw_request_ids, list) or any(
            not isinstance(request_id, str) or not request_id
            for request_id in raw_request_ids
        ):
            raise ValueError(
                f"decode trace event {index} has invalid request IDs"
            )
        normalized_events.append(
            (
                float(observed_at_s),
                size,
                tuple(sorted(set(raw_request_ids))),
            )
        )
    if any(
        current[0] < previous[0]
        for previous, current in itertools.pairwise(normalized_events)
    ):
        raise ValueError("decode trace observation times must be nondecreasing")

    durations_by_request: dict[str, dict[int, float]] = {}
    attributed_duration_s = 0.0
    for current, following in itertools.pairwise(normalized_events):
        duration_s = following[0] - current[0]
        if duration_s <= 0:
            continue
        attributed_duration_s += duration_s
        size = current[1]
        for request_id in current[2]:
            by_size = durations_by_request.setdefault(request_id, {})
            by_size[size] = by_size.get(size, 0.0) + duration_s
    start = min(times) if times else None
    end = max(times) if times else None
    return {
        "event_count": len(value),
        "mixed_event_count": mixed,
        "batch_size_histogram": {
            str(size): count for size, count in sorted(sizes.items())
        },
        "start_s": start,
        "end_s": end,
        "duration_s": None if start is None else end - start,
        "attributed_duration_s": attributed_duration_s,
        "request_occupancy": [
            {
                "request_id": request_id,
                "attributed_duration_s": sum(by_size.values()),
                "batch_size_durations_s": {
                    str(size): duration
                    for size, duration in sorted(by_size.items())
                },
            }
            for request_id, by_size in sorted(durations_by_request.items())
        ],
    }


def compare_decode_occupancy(
    rows: Iterable[Mapping[str, object]],
    stats: Mapping[str, object],
) -> dict[str, object]:
    """Join rollout plans with hook telemetry without claiming performance.

    The result is structural evidence: it identifies when a planned decode
    width was not observed and gives a bounded event timeline for replay. It
    does not turn occupancy mismatches into a throughput claim.
    """

    if not isinstance(stats, Mapping):
        raise TypeError("decode occupancy stats must be an object")
    raw_by_request = stats.get("decode_batch_sizes_by_request", {})
    if not isinstance(raw_by_request, Mapping):
        raise TypeError("decode batch sizes by request must be an object")

    trace = _trace_summary(stats.get("decode_trace"))
    trace_by_request = {
        str(record["request_id"]): record
        for record in trace["request_occupancy"]
    }
    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"rollout row {index} must be an object")
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"rollout row {index} needs a request ID")
        if request_id in seen_ids:
            raise ValueError(f"duplicate rollout request ID: {request_id}")
        seen_ids.add(request_id)
        planned = _positive_integer(
            row.get("decode_batch_size"), "planned decode batch size"
        )
        histogram = _histogram(raw_by_request.get(request_id), request_id)
        observed_events = sum(histogram.values())
        exact_events = histogram.get(planned, 0)
        weighted_size = sum(size * count for size, count in histogram.items())
        trace_occupancy = trace_by_request.get(request_id)
        trace_attributed_s = (
            0.0
            if trace_occupancy is None
            else float(trace_occupancy["attributed_duration_s"])
        )
        trace_exact_s = (
            0.0
            if trace_occupancy is None
            else float(
                trace_occupancy["batch_size_durations_s"].get(str(planned), 0.0)
            )
        )
        dominant_size = (
            None
            if not histogram
            else min(histogram.items(), key=lambda item: (-item[1], item[0]))[0]
        )
        records.append(
            {
                "request_id": request_id,
                "kind": row.get("kind"),
                "planned_batch_size": planned,
                "observed_events": observed_events,
                "observed_batch_sizes": {
                    str(size): count
                    for size, count in sorted(histogram.items())
                },
                "dominant_batch_size": dominant_size,
                "weighted_mean_batch_size": (
                    None if observed_events == 0 else weighted_size / observed_events
                ),
                "exact_events": exact_events,
                "exact_event_fraction": (
                    None if observed_events == 0 else exact_events / observed_events
                ),
                "exactly_matches_plan": (
                    observed_events > 0 and exact_events == observed_events
                ),
                "trace_attributed_duration_s": trace_attributed_s,
                "trace_exact_duration_s": trace_exact_s,
                "trace_exact_time_fraction": (
                    None
                    if trace_attributed_s == 0
                    else trace_exact_s / trace_attributed_s
                ),
            }
        )

    observed = [record for record in records if record["observed_events"] > 0]
    observed_events = sum(int(record["observed_events"]) for record in observed)
    exact_events = sum(int(record["exact_events"]) for record in observed)
    missing_ids = [
        str(record["request_id"])
        for record in records
        if record["observed_events"] == 0
    ]
    unplanned_ids = sorted(set(raw_by_request) - seen_ids)
    traced = [
        record
        for record in records
        if record["trace_attributed_duration_s"] > 0
    ]
    traced_duration_s = sum(
        float(record["trace_attributed_duration_s"]) for record in traced
    )
    exact_duration_s = sum(
        float(record["trace_exact_duration_s"]) for record in traced
    )
    return {
        "available": bool(observed),
        "requests": records,
        "summary": {
            "planned_request_count": len(records),
            "observed_request_count": len(observed),
            "missing_request_ids": missing_ids,
            "unplanned_observed_request_ids": unplanned_ids,
            "observed_decode_events": observed_events,
            "exact_decode_events": exact_events,
            "exact_event_fraction": (
                None if observed_events == 0 else exact_events / observed_events
            ),
            "requests_fully_matching": sum(
                record["exactly_matches_plan"] is True for record in records
            ),
            "requests_with_mismatch": sum(
                record["observed_events"] > 0
                and record["exactly_matches_plan"] is False
                for record in records
            ),
            "complete_request_coverage": len(observed) == len(records),
            "requests_with_trace_occupancy": len(traced),
            "requests_with_trace_mismatch": sum(
                record["trace_exact_time_fraction"] < 1 for record in traced
            ),
            "trace_exact_time_fraction": (
                None
                if traced_duration_s == 0
                else exact_duration_s / traced_duration_s
            ),
        },
        "trace": trace,
    }
