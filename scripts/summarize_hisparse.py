"""Summarize complete MiniMax HiSparse performance arms using the standard library."""

import argparse
import json
import math
from pathlib import Path
import statistics


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def distribution(values):
    require(values and all(type(value) in (float, int) and math.isfinite(value)
                           and value >= 0 for value in values), "Missing or invalid measurements")
    values = sorted(values)
    index = (len(values) - 1) * 0.95
    lower, upper = math.floor(index), math.ceil(index)
    return {"median": statistics.median(values),
            "p95": values[lower] + (values[upper] - values[lower]) * (index - lower)}


def load(directory):
    path = directory / "performance.json" if directory.is_dir() else directory
    data = json.loads(path.read_text(encoding="utf-8"))
    require(data["status"] == "complete", f"{path}: run is incomplete")
    workloads, repeats, ranks = data["workloads"], data["repeats"], data["tp_size"]
    require(type(repeats) is int and repeats > 0 and type(ranks) is int and ranks > 0,
            "Invalid repeat or rank count")
    require(workloads and len(set(workloads)) == len(workloads), "Invalid workload list")
    forced = data.get("forced_replay", False)
    require(type(forced) is bool, "Invalid replay flag")
    warmup_policy = data.get("warmup_policy", "each_repeat")
    require(warmup_policy in ("once_all_shapes", "each_repeat"), "Invalid warmup policy")
    by_phase = {}
    for key, warmup in (("warmups", True), ("phases", False)):
        phase_repeats = 1 if warmup and warmup_policy == "once_all_shapes" else repeats
        expected = {(name, repeat) for name in workloads for repeat in range(phase_repeats)}
        phases = data[key]
        index = {(phase["workload"], phase["repeat"]): phase for phase in phases}
        require(len(phases) == len(expected) and index.keys() == expected,
                f"{path}: incomplete or duplicate {key}")
        for phase in phases:
            requests = phase["requests"]
            label = f"{phase['workload']}/{phase['repeat']}"
            require(phase["warmup"] is warmup and phase["request_count"] == len(requests) == 4,
                    f"{label}: incomplete requests")
            require(len({r["request_id"] for r in requests}) == len(requests)
                    and all(r["output_tokens"] == r["requested_output_tokens"] > 0 for r in requests)
                    and phase["output_tokens"] == sum(r["output_tokens"] for r in requests)
                    and phase["phase_wall_s"] > 0
                    and math.isclose(phase["aggregate_output_tokens_per_s"],
                                     phase["output_tokens"] / phase["phase_wall_s"]),
                    f"{label}: inconsistent request or throughput records")
            require(all(r.get("forced_replay", False) is forced for r in requests),
                    f"{label}: mixed sampling modes")
        if not warmup:
            by_phase = index

    result = {}
    for name in workloads:
        phases = [by_phase[name, repeat] for repeat in range(repeats)]
        requests = [request for phase in phases for request in phase["requests"]]
        rates = [phase["aggregate_output_tokens_per_s"] for phase in phases]
        memory = []
        for phase in phases:
            detail = phase["memory"]
            begin, end = detail["active_window_s"]
            samples = [s for s in detail["samples"] if begin <= s["offset_s"] <= end]
            require(samples and len(samples) == detail["active_sample_count"]
                    and all(len(s["rank_used_bytes"]) == ranks
                            and all(type(v) is int and v >= 0 for v in s["rank_used_bytes"])
                            and sum(s["rank_used_bytes"]) == s["sum_used_bytes"] for s in samples),
                    f"{name}: invalid active NVML sample")
            memory.extend(samples)
        result[name] = {
            "repeats": repeats, "requests": len(requests),
            "throughput_output_tokens_per_s": {
                "by_repeat": rates, "mean": statistics.mean(rates),
                "median": statistics.median(rates), "min": min(rates), "max": max(rates),
            },
            "pooled_request_ttft_s": distribution([r["ttft_s"] for r in requests]),
            "pooled_request_tpot_s": distribution([r["tpot_s"] for r in requests]),
            "nvml_peak_synchronized_sum_bytes": max(s["sum_used_bytes"] for s in memory),
            "nvml_peak_per_rank_bytes": [max(s["rank_used_bytes"][rank] for s in memory)
                                         for rank in range(ranks)],
        }
    return {"arm": path.parent.name, "sampling": "forced" if forced else "greedy",
            "tp_size": ranks, "warmup_policy": warmup_policy, "workloads": result}


def markdown(report):
    for mode, cases in report["sampling_groups"].items():
        if not cases:
            continue
        print(f"### {mode}\n")
        print("| Arm | Workload | Repeats | Output tok/s median | TTFT median / p95 (ms) | TPOT median / p95 (ms) | NVML peak sum (GiB) |")
        print("|---|---|---:|---:|---:|---:|---:|")
        for case in cases:
            for name, row in case["workloads"].items():
                rate = row["throughput_output_tokens_per_s"]["median"]
                ttft, tpot = row["pooled_request_ttft_s"], row["pooled_request_tpot_s"]
                memory = row["nvml_peak_synchronized_sum_bytes"] / (1 << 30)
                print(f"| {case['arm']} | {name} | {row['repeats']} | {rate:.2f} | "
                      f"{ttft['median']*1000:.1f} / {ttft['p95']*1000:.1f} | "
                      f"{tpot['median']*1000:.1f} / {tpot['p95']*1000:.1f} | {memory:.2f} |")
        print()
    print(report["note"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="+", type=Path, help="Result directories or performance.json files")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()
    report = {"sampling_groups": {"greedy": [], "forced": []},
              "note": "Small repeat counts limit inference. Request percentiles pool measured requests. "
                      "The NVML sum peak uses one shared sample, while per-rank peaks may occur at different times."}
    try:
        for directory in args.directory:
            case = load(directory)
            report["sampling_groups"][case["sampling"]].append(case)
        if args.format == "markdown":
            markdown(report)
        else:
            print(json.dumps(report, indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Summary rejected: {exc}\n")


if __name__ == "__main__":
    main()
