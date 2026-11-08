"""Compare exhaustive and owner-indexed joint-demotion candidate search."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_sparse.scheduler_budget import (
    PhysicalDemandIndex,
    _bounded_demotion_groups,
    _indexed_demotion_groups,
)


def make_case(request_count: int, scenario: str):
    requests = {}
    mapped = set()
    refs = Counter()
    for index in range(request_count):
        pages = {index + 1}
        chunks = Counter()
        if scenario == "many-pairs" and index // 2 * 2 + 1 < request_count:
            chunk = index // 2 + 1
            chunks[chunk] = 1
            mapped.add(chunk)
            refs[chunk] = 2
        elif scenario == "shared-tail" and index >= request_count - 2:
            chunks[1] = 1
            mapped.add(1)
            refs[1] = 2
        requests[f"r{index:02d}"] = (None, pages, chunks, set())
    tracker = SimpleNamespace(
        backing=SimpleNamespace(mapped_chunks=frozenset(mapped)),
        refs=refs,
        chunk_bytes=2 << 20,
    )
    return tracker, requests


def measure(callable_, repeats: int):
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = callable_()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    assert result is not None
    return result, {
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[min(len(samples) - 1, int(len(samples) * 0.95))],
    }


def run_case(args, scenario):
    tracker, requests = make_case(args.requests, scenario)
    cost = lambda _request_id: 1

    def exhaustive():
        return _bounded_demotion_groups(
            tracker,
            requests,
            cost,
            max_group=4,
            byte_cap=1 << 60,
            search_limit=args.search_limit,
        )

    def indexed():
        index = PhysicalDemandIndex()
        updates = index.sync(requests)
        measured = _indexed_demotion_groups(
            index,
            tracker,
            requests,
            cost,
            max_group=4,
            byte_cap=1 << 60,
            search_limit=args.search_limit,
        )
        return updates, *measured

    exhaustive_result, exhaustive_timing = measure(exhaustive, args.repeats)
    indexed_result, indexed_timing = measure(indexed, args.repeats)
    old_groups, _, old_examined, old_truncated, _ = exhaustive_result
    updates, new_groups, _, new_examined, new_truncated, _ = indexed_result
    old_group_lists = [list(group) for group in sorted(old_groups)]
    new_group_lists = [list(group) for group in sorted(new_groups)]
    return {
        "scenario": scenario,
        "requests": args.requests,
        "search_limit": args.search_limit,
        "repeats": args.repeats,
        "exhaustive": {
            **exhaustive_timing,
            "examined": old_examined,
            "candidate_groups": len(old_groups),
            "groups_preview": old_group_lists[:8],
            "groups_preview_truncated": len(old_group_lists) > 8,
            "truncated": old_truncated,
        },
        "indexed": {
            **indexed_timing,
            "index_updates": updates,
            "examined": new_examined,
            "candidate_groups": len(new_groups),
            "groups_preview": new_group_lists[:8],
            "groups_preview_truncated": len(new_group_lists) > 8,
            "truncated": new_truncated,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument(
        "--scenario", choices=("all", "no-candidate", "shared-tail", "many-pairs"),
        default="shared-tail",
    )
    parser.add_argument("--search-limit", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args(argv)
    if args.requests < 4:
        parser.error("--requests must be at least 4")
    if args.search_limit < 1 or args.repeats < 1:
        parser.error("search limit and repeats must be positive")

    scenarios = (
        ("no-candidate", "shared-tail", "many-pairs")
        if args.scenario == "all"
        else (args.scenario,)
    )
    payload = [run_case(args, scenario) for scenario in scenarios]
    if args.scenario != "all":
        payload = payload[0]
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
