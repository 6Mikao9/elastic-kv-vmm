#!/usr/bin/env python3
"""Validate and summarize the three capacity experiment arms."""

import argparse
import csv
import json
import math
from pathlib import Path
import re
from statistics import median


ARMS = {"baseline": 327680, "v5": 327680, "baseline-tuned": 32768}
TTFT_LIMITS = (5, 10, 20, 30)
E2E_LIMIT = 60


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def number(value, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            f"invalid numeric value {value!r}")
    return value


def describe(values):
    return {"median": median(values), "min": min(values), "max": max(values)}


def latency(values):
    ordered = sorted(values)
    return {"median": median(ordered), "p95": ordered[math.ceil(.95 * len(ordered)) - 1]}


def summarize(phases):
    elapsed, throughput, peaks, idle = [], [], [], []
    timings = {key: [] for key in ("ttft_s", "end_to_end_s", "tpot_s")}
    successes = {limit: [] for limit in TTFT_LIMITS}
    goodput = {limit: [] for limit in TTFT_LIMITS}
    prompts = {}
    for phase in phases:
        requests = phase["requests"]
        require(len(requests) == phase["completed_requests"] == 40, "expected 40 completions")
        require(phase["output_tokens"] == 40 * 512, "incorrect phase output token count")
        duration = number(phase["elapsed_s"])
        require(duration > 0, "phase elapsed time must be positive")
        elapsed.append(duration)
        throughput.append(number(phase["output_tokens_per_s"]))
        begin, end = map(number, phase["active_window_s"])
        require(begin < end, "empty active window")
        peaks.append(max(number(sample["device_used_bytes"]) for sample in phase["samples"]
                         if begin <= number(sample["offset_s"]) <= end))
        idle.append(number(phase["after"]["device_used_bytes"]))
        for req in requests:
            require(req["input_tokens"] == 6144 and
                    req["requested_output_tokens"] == req["output_tokens"] == 512,
                    "incorrect request token count")
            request_id, digest = req["request_id"], req["prompt_sha256"]
            require(isinstance(request_id, str) and request_id and request_id not in prompts,
                    f"invalid or duplicate request ID {request_id!r}")
            require(re.fullmatch("[0-9a-f]{64}", digest), "invalid prompt hash")
            prompts[request_id] = digest
            for key, values in timings.items():
                values.append(number(req[key]))
        for limit in TTFT_LIMITS:
            count = sum(req["ttft_s"] <= limit and req["end_to_end_s"] <= E2E_LIMIT
                        for req in requests)
            successes[limit].append(count)
            goodput[limit].append(count / duration)
    summary = {
        "elapsed_s": describe(elapsed), "output_tokens_per_s": describe(throughput),
        "active_peak_device_used_bytes": describe(peaks),
        "post_idle_device_used_bytes": describe(idle),
        **{key: latency(values) for key, values in timings.items()},
        "thresholds": {
            str(limit): {
                "request_fraction": sum(successes[limit]) / len(prompts),
                "phase_success_requests_per_s_median": median(goodput[limit]),
            }
            for limit in TTFT_LIMITS
        },
    }
    return summary, prompts


def flatten(values, prefix=""):
    for key, value in values.items():
        name = prefix + key
        if isinstance(value, dict):
            yield from flatten(value, name + ".")
        else:
            yield name, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parents[1]
                        / "results" / "phase2" / "capacity")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = {
        "metadata": {
            "ttft_limits_s": TTFT_LIMITS, "e2e_limit_s": E2E_LIMIT,
            "threshold_rule": "TTFT and E2E must both meet inclusive limits",
            "p95_method": "nearest rank",
            "scope": "Startup admission experiment. Static pool occupancy influences peer profiling and actual capacities.",
            "baseline_tuned": "Uses prior traffic knowledge to manually lower the resident model A capacity.",
        },
        "arms": {},
    }
    reference = None
    for arm, requested_a in ARMS.items():
        try:
            data = json.loads((args.input_dir / arm / "results.json").read_text(encoding="utf-8"))
            require(data["status"] == "complete", f"status is {data['status']!r}")
            require(data["mode"] == ("v5" if arm == "v5" else "baseline"), "incorrect mode")
            require(data["requested_capacities"] == {"a": requested_a, "b": 327680},
                    "incorrect requested capacity")
            actual = data["actual_capacity"]
            require(set(actual) == {"a", "b"} and all(type(value) is int and value > 0
                                                     for value in actual.values()),
                    "expected two positive actual capacities")
            phases = data["phases"]
            require(len(phases) == 3, "expected three steady phases")
            steady, prompts = summarize(phases)
            cold, cold_prompts = summarize([data["first_burst"]])
            require(not prompts.keys() & cold_prompts.keys(), "first burst overlaps steady request IDs")
            if reference is None:
                reference = prompts, cold_prompts
            require(reference == (prompts, cold_prompts), "request IDs or prompt hashes differ across arms")
            report["arms"][arm] = {
                "requested_capacities": data["requested_capacities"],
                "actual_capacity": actual, "steady": steady, "first_burst": cold,
            }
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            parser.exit(1, f"Rejected {arm}: {error}\n")

    output = args.output_dir or args.input_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = []
    for arm, data in report["arms"].items():
        for group in ("steady", "first_burst"):
            rows.append(dict(arm=arm, group=group, e2e_limit_s=E2E_LIMIT, **dict(flatten({
                "requested_capacities": data["requested_capacities"],
                "actual_capacity": data["actual_capacity"], **data[group],
            }))))
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
