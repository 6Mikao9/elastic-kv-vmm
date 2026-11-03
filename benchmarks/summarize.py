"""Summarize explicitly selected completed inference runs and check greedy IDs."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median

# Measured request counts in the fixed benchmark protocol.
WORKLOADS = {"short_serial": 6, "short_concurrent": 8, "long_concurrent": 4, "mixed": 8}


def distribution(values):
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return None
    position = (len(ordered) - 1) * 0.95
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return {"median": median(ordered),
            "p95": ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(directory, model, mode):
    perf = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    probe_path = directory / "results.correctness-v2.json"
    if not probe_path.exists():
        probe_path = directory / "results.correctness.json"
    correctness = json.loads(probe_path.read_text(encoding="utf-8"))
    for result in (perf, correctness):
        require(result["status"] == "complete", f"{model}/{mode}: {result['status']}")
        require(result["model_label"] == model, f"{model}/{mode}: model label mismatch")
        require(result["repeats"] == 3, f"{model}/{mode}: expected three repeats")
    require(correctness["schema_version"] == 2, "Expected schema 2 correctness")
    expected = {f"probe-v2/{n}/{m}/{seed}" for n in (128, 1024, 4096)
                for m in (64, 128) for seed in range(2)}
    probes = {r["request_id"]: r for r in correctness["probes"]}
    require(len(correctness["probes"]) == 12 and probes.keys() == expected,
            "Expected all twelve distinct fixed probes")
    for record in probes.values():
        ids = record["output_ids"]
        require(record["output_ids_complete"] and isinstance(ids, list)
                and len(ids) == record["output_tokens"] > 0
                and all(type(token) is int for token in ids)
                and bool(record["prompt_sha256"]), f"Invalid probe: {record['request_id']}")
    require(set(perf["workloads"]) == set(WORKLOADS), "Incomplete workload coverage")
    phases = {(p["workload"], p["repeat"]): p for p in perf["phases"]}
    expected_phases = {(name, repeat) for name in WORKLOADS for repeat in range(3)}
    require(len(perf["phases"]) == len(expected_phases) and phases.keys() == expected_phases,
            "Expected three measured phases for every workload")
    for (name, repeat), phase in phases.items():
        expected_ids = [f"measured/{name}/{repeat}/{i}" for i in range(WORKLOADS[name])]
        require(not phase["warmup"]
                and [r["request_id"] for r in phase["requests"]] == expected_ids,
                f"Incomplete measured phase: {name}/{repeat}")
    return phases, probes


def metrics(phases, name, stats_enabled):
    active, idle, requests, rates = [], [], [], []
    for repeat in range(3):
        phase = phases[name, repeat]
        requests.extend(phase["requests"])
        rates.append(phase["aggregate_output_tokens_per_s"])
        samples = phase["memory_samples"]
        for target, period in ((active, "active"), (idle, "idle_after")):
            selected = [s for s in samples if s["period"] == period]
            require(selected, f"No {period} samples for {name}/{repeat}")
            target.extend(selected)
    result = {"request_count": len(requests), "repeats": len(rates)}
    for suffix, value in (("median", median(rates)), ("min", min(rates)), ("max", max(rates))):
        result[f"output_tokens_per_s_{suffix}"] = value
    for metric in ("ttft_s", "tpot_s"):
        values = distribution([r[metric] for r in requests])
        require(values is not None, f"No {metric} measurements for {name}")
        for stat in ("median", "p95"):
            result[f"{metric}_{stat}"] = values[stat]
    result["nvml_peak_bytes"] = max(s["device_used_bytes"] for s in active)
    result["nvml_idle_median_bytes"] = median(s["device_used_bytes"] for s in idle)
    result["kv_backed_peak_bytes"] = None
    result["kv_backed_idle_median_bytes"] = None
    if stats_enabled:
        result["kv_backed_peak_bytes"] = max(s["kv_stats"]["backed_bytes"] for s in active)
        result["kv_backed_idle_median_bytes"] = median(
            s["kv_stats"]["backed_bytes"] for s in idle)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, metavar="MODEL:MODE:DIR")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cases = {}
    rows = []
    checks = []
    try:
        for value in args.case:
            model, mode, directory = value.split(":", 2)
            key = (model, mode)
            require(all((model, mode, directory)) and key not in cases, f"Invalid case: {value}")
            cases[key] = load(Path(directory), model, mode)
        for (model, mode), (phases, probes) in cases.items():
            require((model, "baseline") in cases, f"Missing baseline for {model}")
            baseline_phases, baseline_probes = cases[model, "baseline"]
            mismatches = []
            for request_id, probe in probes.items():
                reference = baseline_probes[request_id]
                for field in ("prompt_sha256", "output_ids"):
                    if probe[field] != reference[field]:
                        mismatches.append({"request_id": request_id, "field": field})
            for key, phase in phases.items():
                for request, reference in zip(phase["requests"], baseline_phases[key]["requests"]):
                    if request["prompt_sha256"] != reference["prompt_sha256"]:
                        mismatches.append({"request_id": request["request_id"], "field": "prompt_sha256"})
            checks.append({"model": model, "mode": mode, "probe_count": len(probes),
                           "passed": not mismatches, "mismatches": mismatches})
            for name in WORKLOADS:
                result = metrics(phases, name, mode != "baseline")
                baseline = metrics(baseline_phases, name, False)
                reductions = {}
                for metric, value in result.items():
                    if metric.endswith("_bytes"):
                        reference = baseline[metric]
                        reductions[f"{metric}_reduction_pct"] = (
                            100 * (1 - value / reference) if value is not None and reference else None)
                rows.append({"model": model, "mode": mode, "workload": name,
                             "correctness_passed": not mismatches, **result, **reductions})
        args.output_dir.mkdir(parents=True, exist_ok=True)
        passed = all(check["passed"] for check in checks)
        (args.output_dir / "summary.json").write_text(
            json.dumps({"correctness_passed": passed, "correctness": checks, "workloads": rows},
                       indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Summary rejected: {exc}\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
