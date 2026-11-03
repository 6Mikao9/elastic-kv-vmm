"""Validate Phase 2 arms and summarize matched workloads and correctness probes."""

import argparse
import csv
import json
from pathlib import Path

from summarize import WORKLOADS, load, metrics, require


MODEL_LABELS = {"qwen06": "Qwen3-0.6B-Base", "qwen4": "Qwen3-4B"}


COMPARISON_METRICS = (
    "nvml_peak_bytes", "nvml_idle_median_bytes", "output_tokens_per_s_median",
    "ttft_s_median", "ttft_s_p95", "tpot_s_median", "tpot_s_p95",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(record, reference, fields):
    return [{"request_id": record["request_id"], "field": field}
            for field in fields if record[field] != reference[field]]


def padded_requests(directory):
    path = directory / "results.correctness-v2.json"
    if not path.exists():
        path = directory / "results.correctness.json"
    result = {}
    batches = read_json(path).get("padded_batches", [])
    require([batch["batch_size"] for batch in batches] == [3, 5, 3]
            and len({batch["round"] for batch in batches}) == 3,
            f"Expected padded batches of sizes 3, 5, 3 in {path}")
    for batch in batches:
        batch_key = (batch["batch_size"], batch["round"])
        require(len(batch["requests"]) == batch_key[0], f"Incomplete padded batch: {batch_key}")
        for record in batch["requests"]:
            key = (*batch_key, record["request_id"])
            ids = record["output_ids"]
            require(key not in result and record["output_ids_complete"]
                    and isinstance(ids, list) and len(ids) == record["output_tokens"] > 0
                    and all(type(token) is int for token in ids) and bool(record["prompt_sha256"]),
                    f"Invalid padded probe: {key}")
            result[key] = record
    return result


def compare_padding(candidate, reference, mode):
    mismatches = []
    matches = 0
    for key in sorted(candidate.keys() | reference.keys()):
        batch_size, round_number, request_id = key
        details = {"batch_size": batch_size, "round": round_number, "request_id": request_id}
        if key not in candidate or key not in reference:
            mismatches.append({**details, "field": "missing_request",
                               "missing_from": mode if key not in candidate else "baseline"})
            continue
        actual, expected = candidate[key], reference[key]
        differences = compare(actual, expected, ("prompt_sha256", "output_ids"))
        if not differences:
            matches += 1
        for difference in differences:
            field = difference["field"]
            mismatches.append({**details, **difference, "baseline": expected[field],
                               mode: actual[field]})
    return {"baseline_count": len(reference), "v5_count": len(candidate),
            "match_count": matches, "mismatches": mismatches}


def validate_stats(stats, execution, mode, label, arm):
    actual_mode = "v5" if mode == "v5-keep8" else mode
    require(stats["mode"] == actual_mode, f"{label}: adapter stats mode mismatch")
    if actual_mode == "v5":
        expected = 8 if mode == "v5-keep8" else 1
        require(stats.get("retain_chunks", 1) == expected,
                f"{label}: expected retain_chunks={expected}")
    if execution.startswith("graph"):
        evidence = None
        if stats.get("actual_graph_replay_calls", 0) <= 0:
            require(execution == "graph" and mode == "baseline"
                    and stats.get("graph_capture_calls", 0) > 0,
                    f"{label}: no graph replays")
            evidence = read_json(arm / "graph_evidence.json")
            digest = evidence["server_log_sha256"]
            require(evidence["source"] == "SGLang decode log"
                    and evidence["engine_decode_log_graph_true_count"] > 0
                    and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                    f"{label}: invalid Graph decode log evidence")
        require(stats["live_tokens"] == 0, f"{label}: KV slots remain live at idle")
        if mode != "baseline":
            for field in ("ensure_calls", "reclaim_calls"):
                require(stats.get(field, 0) > 0, f"{label}: no {field}")
        if execution == "graph-pad":
            require(stats.get("padded_graph_replay_calls", 0) > 0, f"{label}: no padded replays")
        return evidence
    return None


def summarize(directory, include_padding=False, include_retention=False):
    require(not include_retention or include_padding, "--include-retention requires --include-padding")
    groups = [("qwen06", "eager"), ("qwen06", "graph"), ("qwen4", "graph")]
    if include_padding:
        groups.extend((model, "graph-pad") for model in ("qwen06", "qwen4"))
    rows, checks = [], []
    fixed_probes = {}
    for model, execution in groups:
        modes = ("baseline", "v5") if execution == "graph-pad" else ("baseline", "v4", "v5")
        if execution == "graph-pad" and include_retention:
            modes += ("v5-keep8",)
        cases, measured = {}, {}
        for mode in modes:
            label = f"{model}/{execution}/{mode}"
            arm = directory / label
            phases, probes = load(arm, MODEL_LABELS[model], "v5" if mode == "v5-keep8" else mode)
            stats = read_json(arm / "adapter_stats.json")
            evidence = validate_stats(stats, execution, mode, label, arm)
            cases[mode] = phases, probes, stats, evidence
            fixed_probes[model, mode, execution] = probes
            measured[mode] = {name: metrics(phases, name, mode != "baseline")
                              for name in WORKLOADS}

        baseline_phases, baseline_probes, _, _ = cases["baseline"]
        for mode, (phases, probes, stats, evidence) in cases.items():
            mismatches = [difference for key, record in probes.items()
                          for difference in compare(record, baseline_probes[key],
                                                    ("prompt_sha256", "output_ids"))]
            for key, phase in phases.items():
                for record, reference in zip(phase["requests"], baseline_phases[key]["requests"]):
                    mismatches.extend(compare(record, reference, ("prompt_sha256",)))
            identifiers = {"model": model, "execution": execution, "mode": mode}
            check = {**identifiers, "probe_count": len(probes), "passed": not mismatches,
                     "mismatches": mismatches, "adapter_stats": stats}
            if evidence is not None:
                check["graph_evidence"] = evidence
            if execution == "graph-pad" and mode != "baseline":
                group_path = directory / model / execution
                padding = compare_padding(padded_requests(group_path / mode),
                                          padded_requests(group_path / "baseline"), mode)
                check["padded_probes"] = padding
                check["passed"] = check["passed"] and not padding["mismatches"]
            checks.append(check)
            for workload, result in measured[mode].items():
                row = {**identifiers, "workload": workload,
                       "correctness_passed": check["passed"], **result}
                for reference_mode in ("baseline", "v4", "v5") if include_retention else ("baseline", "v4"):
                    reference = measured.get(reference_mode, {}).get(workload, {})
                    for field in COMPARISON_METRICS:
                        value = reference.get(field)
                        row[f"{field}_change_vs_{reference_mode}_pct"] = (
                            100 * (result[field] / value - 1) if value else None)
                rows.append(row)

    cross_execution = []
    references = {}
    for (model, mode, execution), probes in fixed_probes.items():
        key = (model, mode)
        if key not in references:
            references[key] = execution, probes
            continue
        reference_execution, reference = references[key]
        mismatches = [difference for request_id, record in probes.items()
                      for difference in compare(record, reference[request_id],
                                                ("prompt_sha256", "output_ids"))]
        cross_execution.append({"model": model, "mode": mode, "execution": execution,
                                "reference_execution": reference_execution,
                                "probe_count": len(probes), "passed": not mismatches,
                                "mismatches": mismatches})
    return {"correctness_passed": all(check["passed"] for check in checks + cross_execution),
            "correctness": checks, "cross_execution_correctness": cross_execution,
            "workloads": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Phase 2 results root")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--include-padding", action="store_true")
    parser.add_argument("--include-retention", action="store_true")
    args = parser.parse_args()
    try:
        result = summarize(args.directory, args.include_padding, args.include_retention)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        rows = result["workloads"]
        with (args.output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Summary rejected: {exc}\n")
    return 0 if result["correctness_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
