#!/usr/bin/env python3
"""Measure future-reservation accounting without CUDA or model execution."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from functools import partial
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_sparse.future_plan import RolloutSubmission
from adaptive_sparse.future_prefill import FuturePrefillEntry, _interval_peak
from adaptive_sparse.residency_broker import ResidencyBroker
from adaptive_sparse.rollout_executor import (
    _cohort_from_queue,
    _deadline_order_key,
    _execution_groups,
    _ordered_due_cohorts,
)


def _reference_interval_peak(entries, byte_field, finish_field, *, shared_prefix):
    """Previous event-scan implementation retained as the equivalence oracle."""

    if not entries:
        return 0
    by_id = {entry.request_id: entry for entry in entries}
    events = {}
    for entry in entries:
        start = float(getattr(entry, "start_in_ms", 0.0))
        finish = float(getattr(entry, finish_field, float("inf")))
        value = int(getattr(entry, byte_field))
        if finish <= start or value <= 0:
            continue
        events.setdefault(start, []).append((1, entry.request_id))
        events.setdefault(finish, []).append((-1, entry.request_id))

    active = set()
    peak = 0
    for instant in sorted(events):
        changes = events[instant]
        for direction, request_id in changes:
            if direction < 0:
                active.discard(request_id)
        for direction, request_id in changes:
            if direction > 0:
                active.add(request_id)
        if not active:
            continue
        total = sum(
            int(getattr(by_id[request_id], byte_field))
            for request_id in active
        )
        if shared_prefix and byte_field == "kv_bytes":
            groups = {}
            for request_id in active:
                entry = by_id[request_id]
                if (
                    entry.residence == "resident"
                    and entry.prefix_group
                    and entry.shared_prefix_bytes > 0
                ):
                    groups.setdefault(entry.prefix_group, []).append(
                        entry.shared_prefix_bytes
                    )
            total -= sum(
                (len(values) - 1) * min(values)
                for values in groups.values()
            )
        peak = max(peak, total)
    return peak


def _entries(count):
    return [
        FuturePrefillEntry(
            request_id=f"request-{index}",
            kind="rollout",
            residence="resident",
            prompt_tokens=128,
            expected_output_tokens=256,
            kv_bytes=1_000 + index * 17,
            activation_bytes=0,
            release_in_ms=1_000,
            native_order=index,
            start_in_ms=(index % 37) * 10,
            kv_finish_in_ms=(index % 37) * 10 + 100 + (index % 11) * 10,
            prefix_group=f"group-{index % 16}" if index % 3 else None,
            shared_prefix_tokens=16 + index % 7,
            shared_prefix_bytes=16 + index % 7,
        )
        for index in range(count)
    ]


def _median_us(function, *, repeats):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - started) / 1_000.0)
    return statistics.median(samples)


def _rollout_requests(count, *, cohort_size=1):
    return tuple(
        RolloutSubmission(
            request_id=f"rollout-{index}",
            planned_arrival_ms=0,
            submit_at_ms=0,
            estimated_finish_ms=100 + index % 31,
            kind="rollout",
            prompt_tokens=128 + index % 17,
            output_tokens=64 + index % 13,
            hint={
                "release_after_ms": 0,
                "deadline_after_ms": 5000 + index % 97,
                "expected_output_tokens": 64 + index % 13,
                "weight": 1.0 + index % 3,
                "kind": "rollout",
                "residence": (
                    "resident" if index % 4 else "host_backed"
                ),
            },
            deadline_ms=5000 + index % 97,
            cohort_id=(
                f"cohort-{index // cohort_size}"
                if cohort_size > 1
                else None
            ),
            prefill_batch_size=cohort_size,
        )
        for index in range(count)
    )


def _reference_ordered_due_cohorts(due, queue):
    """Previous restart-from-the-head ordering implementation."""

    remaining = list(due)
    ordered = []
    while remaining:
        head = min(
            remaining,
            key=lambda item: _deadline_order_key(item),
        )
        cohort = _cohort_from_queue(head, queue)
        ordered.append(cohort)
        cohort_ids = {item.request_id for item in cohort}
        remaining = [
            item for item in remaining if item.request_id not in cohort_ids
        ]
    return tuple(ordered)


def _reference_execution_groups(submissions):
    """Previous one-full-scan-per-cohort grouping implementation."""

    groups = []
    emitted_cohorts = set()
    for item in submissions:
        if item.cohort_id is None:
            groups.append((item,))
            continue
        if item.cohort_id in emitted_cohorts:
            continue
        emitted_cohorts.add(item.cohort_id)
        groups.append(
            tuple(
                candidate
                for candidate in submissions
                if candidate.cohort_id == item.cohort_id
            )
        )
    groups.sort(
        key=lambda group: (
            group[0].submit_at_ms,
            min(item.request_id for item in group),
        )
    )
    return tuple(groups)


def run(sizes, rollout_sizes, repeats, scans):
    interval = []
    for size in sizes:
        entries = _entries(size)
        reference = _reference_interval_peak(
            entries, "kv_bytes", "kv_finish_in_ms", shared_prefix=True
        )
        optimized = _interval_peak(
            entries, "kv_bytes", "kv_finish_in_ms", shared_prefix=True
        )
        if reference != optimized:
            raise AssertionError((size, reference, optimized))
        interval.append(
            {
                "requests": size,
                "peak_bytes": optimized,
                "reference_median_us": _median_us(
                    lambda entries=entries: _reference_interval_peak(
                        entries,
                        "kv_bytes",
                        "kv_finish_in_ms",
                        shared_prefix=True,
                    ),
                    repeats=repeats,
                ),
                "incremental_median_us": _median_us(
                    lambda entries=entries: _interval_peak(
                        entries,
                        "kv_bytes",
                        "kv_finish_in_ms",
                        shared_prefix=True,
                    ),
                    repeats=repeats,
                ),
            }
        )

    reservations = []
    for size in sizes:
        broker = ResidencyBroker()
        for index in range(size):
            broker.reserve(f"reservation-{index}", predicted_growth_bytes=index + 1)
        expected = sum(broker._reservations.values())
        if broker.reserved_bytes != expected:
            raise AssertionError((size, expected, broker.reserved_bytes))

        reservations.append(
            {
                "reservations": size,
                "scans_per_sample": scans,
                "total_bytes": expected,
                "reference_median_us": _median_us(
                    partial(_scan_reserved_bytes, broker, expected, scans),
                    repeats=repeats,
                ),
                "incremental_median_us": _median_us(
                    partial(_read_reserved_bytes, broker, expected, scans),
                    repeats=repeats,
                ),
            }
        )
    cohorts = []
    groups = []
    for size in rollout_sizes:
        requests = _rollout_requests(size)
        reference = _reference_ordered_due_cohorts(requests, requests)
        optimized = _ordered_due_cohorts(
            requests,
            requests,
            "deadline",
        )
        if reference != optimized:
            raise AssertionError((size, "cohort ordering mismatch"))
        cohorts.append(
            {
                "requests": size,
                "cohorts": len(optimized),
                "reference_median_us": _median_us(
                    lambda requests=requests: _reference_ordered_due_cohorts(
                        requests, requests
                    ),
                    repeats=repeats,
                ),
                "single_sort_median_us": _median_us(
                    lambda requests=requests: _ordered_due_cohorts(
                        requests, requests, "deadline"
                    ),
                    repeats=repeats,
                ),
            }
        )
        grouped_requests = _rollout_requests(size, cohort_size=8)
        reference_groups = _reference_execution_groups(grouped_requests)
        optimized_groups = _execution_groups(grouped_requests)
        if reference_groups != optimized_groups:
            raise AssertionError((size, "execution grouping mismatch"))
        groups.append(
            {
                "requests": size,
                "cohorts": len(optimized_groups),
                "reference_median_us": _median_us(
                    lambda requests=grouped_requests: (
                        _reference_execution_groups(requests)
                    ),
                    repeats=repeats,
                ),
                "single_index_median_us": _median_us(
                    lambda requests=grouped_requests: _execution_groups(
                        requests
                    ),
                    repeats=repeats,
                ),
            }
        )
    return {
        "environment": "local CPU control-plane microbenchmark; no CUDA or model execution",
        "repeats": repeats,
        "interval_peak": interval,
        "reserved_bytes": reservations,
        "cohort_ordering": cohorts,
        "execution_groups": groups,
    }


def _scan_reserved_bytes(broker, expected, scans):
    for _ in range(scans):
        if sum(broker._reservations.values()) != expected:
            raise AssertionError("reference scan mismatch")


def _read_reserved_bytes(broker, expected, scans):
    for _ in range(scans):
        if broker.reserved_bytes != expected:
            raise AssertionError("incremental total mismatch")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=(64, 256, 1024))
    parser.add_argument(
        "--rollout-sizes", nargs="+", type=int, default=(64, 256, 1024)
    )
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--reservation-scans", type=int, default=1_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.reservation_scans < 1:
        parser.error("repeats and reservation scans must be positive")
    if any(size < 1 for size in (*args.sizes, *args.rollout_sizes)):
        parser.error("sizes must be positive")

    gc.disable()
    try:
        result = run(
            args.sizes,
            args.rollout_sizes,
            args.repeats,
            args.reservation_scans,
        )
    finally:
        gc.enable()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
