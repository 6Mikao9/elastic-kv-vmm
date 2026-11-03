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


COMMON_ARGS = (
    "tp-size", "dtype", "kv-cache-dtype", "attention-backend", "mm-attention-backend",
    "moe-runner-backend", "page-size", "context-length", "chunked-prefill-size",
    "max-running-requests", "mem-fraction-static", "random-seed",
)
EAGER_FLAGS = ("disable-cuda-graph", "disable-prefill-cuda-graph",
               "disable-overlap-schedule", "disable-radix-cache")
DIAGNOSTICS = ("diagnostic_captures", "cuda_launch_blocking_diagnostic",
               "layer_trace_diagnostic", "qknorm_fusion_disabled",
               "attention_capture_diagnostic", "logical_shadow_diagnostic")
COMPAT_SOURCE = "adaptive_sparse/minimax_sm80.py"


def comparison_signature(directory, performance):
    outer = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "source_manifest.json").read_text(encoding="utf-8"))
    require(outer.get("status") == "complete", f"{directory}: outer run is incomplete")
    require(outer.get("mode") == manifest["mode"], f"{directory}: mode evidence differs")
    for key in DIAGNOSTICS:
        require(key in manifest and not manifest[key], f"{directory}: missing or intrusive {key}")
    settings = manifest["adaptive_settings"] or {}
    require(not any(settings.get(key, 0) for key in (
        "EKV_ADAPTIVE_FORCE_DECODE_STEP", "EKV_ADAPTIVE_FORCE_PROMOTE_STEP",
        "EKV_ADAPTIVE_PLAN_CAPTURE",
    )), f"{directory}: forced migration or plan capture")
    require(performance["repeats"] == manifest["benchmark_repeats"]
            and (manifest["workloads"] is None or performance["workloads"] == manifest["workloads"]),
            f"{directory}: workload or repeat manifest mismatch")
    require(performance.get("forced_replay", False) == manifest.get("forced_replay", False)
            and (performance.get("warmup_policy", "each_repeat") == "once_all_shapes")
            == manifest.get("warmup_once", False),
            f"{directory}: replay or warmup manifest mismatch")

    capacity = outer["capacity"]
    logical = capacity["derived_logical_capacity"]
    pool, ratio = capacity["reported_pool_capacity"], capacity["host_to_device_ratio"]
    require(all(type(value) is int and value > 0 for value in (logical, pool, ratio))
            and pool == capacity["resolved_max_total_tokens"]
            and logical == pool * ratio == manifest["requested_logical_capacity"],
            f"{directory}: actual logical capacity disagrees with requested capacity")

    argv = manifest["server_args"]
    def option(name):
        flag = "--" + name
        require(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv)
                and not argv[argv.index(flag) + 1].startswith("--"),
                f"{directory}: missing or ambiguous {flag}")
        return argv[argv.index(flag) + 1]

    require(pool == manifest["requested_pool_capacity"] == int(option("max-total-tokens")),
            f"{directory}: actual physical capacity disagrees with requested pool")
    mode = manifest["mode"]
    hisparse = None
    if mode == "native":
        require(argv.count("--enable-hisparse") == 1, f"{directory}: native mode requires HiSparse")
        hisparse = json.loads(option("hisparse-config"))
        require(all(type(hisparse.get(key)) is int and hisparse[key] > 0 for key in
                    ("top_k", "device_buffer_size", "host_to_device_ratio"))
                and hisparse["host_to_device_ratio"] == ratio,
                f"{directory}: HiSparse config ratio or dimensions disagree")
    else:
        require(mode == "resident" and ratio == 1 and "--enable-hisparse" not in argv
                and "--hisparse-config" not in argv,
                f"{directory}: resident mode requires no HiSparse and capacity ratio one")
    signature = {name: option(name) for name in COMMON_ARGS}
    for name in ("enable-deterministic-inference", "disable-custom-all-reduce",
                 "enable-custom-logit-processor"):
        flag = "--" + name
        require(argv.count(flag) <= 1, f"{directory}: duplicate {flag}")
        signature[name] = flag in argv
    require(signature["enable-custom-logit-processor"] == performance.get("forced_replay", False),
            f"{directory}: custom logit processor and replay evidence disagree")
    require(all(argv.count("--" + name) == 1 for name in EAGER_FLAGS),
            f"{directory}: missing eager, no-overlap or no-radix flags")
    require(int(signature["tp-size"]) == performance["tp_size"] == outer["tp_size"],
            f"{directory}: TP degree mismatch")
    compat = manifest["sm80_bf16_projections"]
    require(type(compat) is bool and outer.get("sm80_bf16_projections") is compat,
            f"{directory}: missing or inconsistent SM80 compatibility mode")
    source = manifest["sha256"].get(COMPAT_SOURCE)
    require(not compat or (isinstance(source, str) and len(source) == 64
                           and all(char in "0123456789abcdef" for char in source)),
            f"{directory}: missing SM80 compatibility source hash")
    return {**signature, **dict.fromkeys(EAGER_FLAGS, True),
            "logical_capacity": logical, "sm80_bf16_projections": compat,
            "sm80_compat_source_sha256": source if compat else None}, hisparse


def require_matched(reference, case):
    require(all(case[key] == reference[key] for key in
                ("sampling", "tp_size", "warmup_policy", "comparison_signature")),
            "Sampling, TP degree, warmup policy or common inference configuration differs")
    left, right = reference["hisparse_config"], case["hisparse_config"]
    require(left is None or right is None or left == right, "HiSparse configuration differs")


def load(directory):
    path = directory / "performance.json" if directory.is_dir() else directory
    data = json.loads(path.read_text(encoding="utf-8"))
    require(data["status"] == "complete", f"{path}: run is incomplete")
    signature, hisparse = comparison_signature(path.parent, data)
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
            "tp_size": ranks, "warmup_policy": warmup_policy, "workloads": result,
            "comparison_signature": signature, "hisparse_config": hisparse}


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
