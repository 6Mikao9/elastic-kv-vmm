"""Compare planned decode width with observed SGLang batch occupancy."""

from __future__ import annotations

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
        }
    if not isinstance(value, list):
        raise TypeError("decode trace must be a list")
    times: list[float] = []
    sizes: dict[int, int] = {}
    mixed = 0
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
        },
        "trace": _trace_summary(stats.get("decode_trace")),
    }
