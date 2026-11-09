"""Turn observed rollout traces into conservative planner profiles."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping
from typing import cast


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _request_histogram(
    by_request: Mapping[object, object], request_id: str
) -> dict[int, int]:
    raw = by_request.get(request_id, {})
    if not isinstance(raw, Mapping):
        raise TypeError(f"decode histogram for {request_id} must be an object")
    histogram: dict[int, int] = {}
    for raw_size, raw_count in raw.items():
        if isinstance(raw_size, str) and raw_size.isdecimal():
            size = int(raw_size)
        elif isinstance(raw_size, bool) or not isinstance(raw_size, int):
            raise ValueError(f"decode batch size for {request_id} must be an integer")
        else:
            size = raw_size
        histogram[_positive_int(size, "decode batch size")] = _positive_int(
            raw_count, "decode batch count"
        )
    return histogram


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def calibrate_profiles(
    rows: Iterable[Mapping[str, object]],
    stats: Mapping[str, object],
    *,
    min_samples: int = 3,
) -> dict[str, object]:
    """Build planner profiles from complete and unambiguous measurements.

    Decode samples are accepted only when that request was observed at exactly
    one real batch size. Mixed-width requests are excluded instead of assigning
    their single TPOT to one width. Prefill samples require a complete native
    cohort. Missing samples are retained as evidence and never interpolated.
    """

    min_samples = _positive_int(min_samples, "min_samples")
    if not isinstance(stats, Mapping):
        raise TypeError("rollout stats must be an object")
    raw_by_request = stats.get("decode_batch_sizes_by_request", {})
    if not isinstance(raw_by_request, Mapping):
        raise TypeError("decode batch sizes by request must be an object")

    normalized_rows: list[Mapping[str, object]] = []
    decode_samples: dict[tuple[str, int], list[float]] = {}
    kind_decode_samples: dict[tuple[str, str, int], list[float]] = {}
    decode_counts = {
        "candidate_requests": 0,
        "accepted_requests": 0,
        "missing_observation": 0,
        "ambiguous_batch_size": 0,
        "invalid_row": 0,
    }
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"rollout row {index} must be an object")
        normalized_rows.append(row)
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            decode_counts["invalid_row"] += 1
            continue
        decode_counts["candidate_requests"] += 1
        histogram = _request_histogram(raw_by_request, request_id)
        if not histogram:
            decode_counts["missing_observation"] += 1
            continue
        if len(histogram) != 1:
            decode_counts["ambiguous_batch_size"] += 1
            continue
        residence = row.get("runtime_residence", row.get("planned_residence"))
        if residence not in ("resident", "host_backed"):
            decode_counts["invalid_row"] += 1
            continue
        try:
            tpot_s = _positive_float(row.get("tpot_s"), "tpot_s")
        except ValueError:
            decode_counts["invalid_row"] += 1
            continue
        batch_size = next(iter(histogram))
        aggregate_tokens_per_ms = batch_size / (tpot_s * 1000.0)
        decode_samples.setdefault((residence, batch_size), []).append(
            aggregate_tokens_per_ms
        )
        kind = row.get("kind")
        if isinstance(kind, str) and kind and len(kind) <= 64:
            kind_decode_samples.setdefault(
                (residence, kind, batch_size), []
            ).append(aggregate_tokens_per_ms)
        decode_counts["accepted_requests"] += 1

    decode_profiles: dict[str, dict[str, object]] = {}
    decode_evidence: dict[str, dict[str, object]] = {}
    for (residence, batch_size), values in sorted(
        decode_samples.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        if len(values) < min_samples:
            continue
        profile = decode_profiles.setdefault(str(batch_size), {})
        key = (
            "resident_tokens_per_ms"
            if residence == "resident"
            else "sparse_tokens_per_ms"
        )
        profile[key] = _median(values)
        decode_evidence[f"{batch_size}:{residence}"] = {
            "batch_size": batch_size,
            "residence": residence,
            "samples": len(values),
            "aggregate_tokens_per_ms": _median(values),
        }

    kind_decode_evidence: dict[str, dict[str, object]] = {}
    for (residence, kind, batch_size), values in sorted(
        kind_decode_samples.items(),
        key=lambda item: (item[0][2], item[0][0], item[0][1]),
    ):
        if len(values) < min_samples:
            continue
        batch_profile = decode_profiles[str(batch_size)]
        raw_kind_profiles = cast(
            dict[str, dict[str, float]],
            batch_profile.setdefault("kind_profiles", {}),
        )
        raw_kind_profiles[kind] = {
            "resident_tokens_per_ms"
            if residence == "resident"
            else "sparse_tokens_per_ms": _median(values)
        }
        kind_decode_evidence[f"{batch_size}:{residence}:{kind}"] = {
            "batch_size": batch_size,
            "residence": residence,
            "kind": kind,
            "samples": len(values),
            "aggregate_tokens_per_ms": _median(values),
        }

    complete_ids = stats.get("complete_cohort_ids", [])
    if not isinstance(complete_ids, list) or any(
        not isinstance(cohort_id, str) for cohort_id in complete_ids
    ):
        raise ValueError("complete cohort IDs must be a list of strings")
    by_cohort: dict[str, list[Mapping[str, object]]] = {}
    for row in normalized_rows:
        cohort_id = row.get("cohort_id")
        if isinstance(cohort_id, str) and cohort_id:
            by_cohort.setdefault(cohort_id, []).append(row)

    prefill_samples: dict[int, list[float]] = {}
    prefill_evidence = {
        "complete_cohorts": 0,
        "accepted_cohorts": 0,
        "incomplete_cohorts": 0,
        "invalid_cohorts": 0,
    }
    complete_set = set(complete_ids)
    for cohort_id, members in sorted(by_cohort.items()):
        if cohort_id not in complete_set:
            prefill_evidence["incomplete_cohorts"] += 1
            continue
        prefill_evidence["complete_cohorts"] += 1
        try:
            sizes = {
                _positive_int(
                    member.get("prefill_batch_size"), "prefill_batch_size"
                )
                for member in members
            }
        except ValueError:
            prefill_evidence["invalid_cohorts"] += 1
            continue
        if len(sizes) != 1:
            prefill_evidence["invalid_cohorts"] += 1
            continue
        batch_size = next(iter(sizes))
        if len(members) != batch_size:
            prefill_evidence["invalid_cohorts"] += 1
            continue
        try:
            tokens = sum(
                _positive_int(member.get("prompt_tokens"), "prompt_tokens")
                for member in members
            )
            ttfts = [
                _positive_float(member.get("ttft_s"), "ttft_s")
                for member in members
            ]
        except ValueError:
            prefill_evidence["invalid_cohorts"] += 1
            continue
        aggregate_tokens_per_ms = tokens / (_median(ttfts) * 1000.0)
        prefill_samples.setdefault(batch_size, []).append(
            aggregate_tokens_per_ms
        )
        prefill_evidence["accepted_cohorts"] += 1

    prefill_profiles: dict[str, float] = {}
    prefill_profile_evidence: dict[str, dict[str, object]] = {}
    for batch_size, values in sorted(prefill_samples.items()):
        if len(values) < min_samples:
            continue
        rate = _median(values)
        prefill_profiles[str(batch_size)] = rate
        prefill_profile_evidence[str(batch_size)] = {
            "samples": len(values),
            "aggregate_tokens_per_ms": rate,
        }

    return {
        "min_samples": min_samples,
        "available": bool(decode_profiles or prefill_profiles),
        "decode_batch_profiles": decode_profiles,
        "prefill_batch_profiles": prefill_profiles,
        "evidence": {
            "decode": {
                **decode_counts,
                "profiles": decode_evidence,
                "kind_profiles": kind_decode_evidence,
                "below_min_samples": sorted(
                    f"{batch_size}:{residence}"
                    for (residence, batch_size), values in decode_samples.items()
                    if len(values) < min_samples
                ),
                "kind_below_min_samples": sorted(
                    f"{batch_size}:{residence}:{kind}"
                    for (residence, kind, batch_size), values in (
                        kind_decode_samples.items()
                    )
                    if len(values) < min_samples
                ),
            },
            "prefill": {
                **prefill_evidence,
                "profiles": prefill_profile_evidence,
                "below_min_samples": sorted(
                    str(batch_size)
                    for batch_size, values in prefill_samples.items()
                    if len(values) < min_samples
                ),
            },
        },
    }
