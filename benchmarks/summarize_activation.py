"""Summarize completed Qwen3-4B activation experiments without a GPU."""

import argparse
import json
from pathlib import Path

from summarize import WORKLOADS, load, metrics, require
from summarize_phase2 import compare_padding, padded_requests


def summarize(directory):
    calibration = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
    cases = {}
    for policy, chunk in (("telemetry", 512), ("telemetry", 2048),
                          ("telemetry", 8192), ("both", 8192)):
        name = f"{policy}-{chunk}"
        path = directory / name
        phases, probes = load(path, "Qwen3-4B", "v5")
        manifest = json.loads((path / "source_manifest.json").read_text(encoding="utf-8"))
        expected_headroom = calibration["prefill_headroom_mib"] if policy == "both" else 0
        expected = {"mode": "v5", "execution": "graph-pad", "retain_chunks": 8,
                    "prefill_chunk": chunk, "activation_policy": policy,
                    "prefill_headroom_mib": expected_headroom}
        require(all(manifest[key] == value for key, value in expected.items()),
                f"Configuration mismatch for {name}")
        stats = json.loads((path / "adapter_stats.json").read_text(encoding="utf-8"))
        require(stats["activation_prefill_calls"] > 0, f"No prefill telemetry for {name}")
        require(stats["live_tokens"] == 0 and stats.get("padded_graph_replay_calls", 0) > 0,
                f"Missing idle or padded Graph evidence for {name}")
        padded_requests(path)
        cases[name] = phases, probes, stats

    ref_phases, ref_probes, ref_stats = cases["telemetry-8192"]
    peak = ref_stats["activation_prefill_peak_growth_bytes"]
    quantum = 64 << 20
    require(peak == calibration["activation_prefill_peak_growth_bytes"]
            and calibration["prefill_headroom_mib"] == ((peak + quantum - 1) // quantum) * 64 + 128,
            "Calibration does not match the measured 8192-chunk peak")
    phases, probes, _ = cases["both-8192"]
    mismatches = [{"request_id": request_id, "field": field}
                  for request_id, probe in probes.items() for field in ("prompt_sha256", "output_ids")
                  if probe[field] != ref_probes[request_id][field]]
    for key, phase in phases.items():
        for request, reference in zip(phase["requests"], ref_phases[key]["requests"]):
            if request["prompt_sha256"] != reference["prompt_sha256"]:
                mismatches.append({"request_id": request["request_id"], "field": "prompt_sha256"})

    padded = compare_padding(padded_requests(directory / "both-8192"),
                             padded_requests(directory / "telemetry-8192"), "both")
    return {
        "calibration": calibration,
        "same_chunk_correctness": {"passed": not mismatches and not padded["mismatches"],
                                   "fixed_probe_count": len(probes),
                                   "mismatches": mismatches, "padded_probes": padded},
        "cases": [{
            "case": name,
            "activation": {key: value for key, value in stats.items() if key.startswith("activation_")},
            "workloads": [{"workload": workload, **metrics(phases, workload, True)}
                          for workload in WORKLOADS],
        } for name, (phases, _, stats) in cases.items()],
        "note": "These model runs characterize measured pressure and policy actions. "
                "Successful runs alone do not establish OOM prevention.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Activation results directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = summarize(args.directory)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Summary rejected: {exc}\n")
    return 0 if report["same_chunk_correctness"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
