"""Run isolated Qwen3-4B activation calibration and comparison cases."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from summarize import load, require

MODEL = "Qwen3-4B"
SNAPSHOT = "models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
MIB = 1 << 20


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_case(gpu, hub, directory, policy, chunk, headroom, resume):
    config = {"model_label": MODEL, "mode": "v5", "execution": "graph-pad",
              "retain_chunks": 8, "activation_policy": policy, "prefill_chunk": chunk,
              "prefill_headroom_mib": headroom}
    if directory.exists():
        require(resume, f"Output already exists: {directory}")
    else:
        env = {**os.environ, "EKV_EXECUTION": "graph-pad", "EKV_RETAIN_CHUNKS": "8",
               "EKV_ACTIVATION_POLICY": policy, "EKV_PREFILL_CHUNK": str(chunk),
               "EKV_PREFILL_HEADROOM_MIB": str(headroom)}
        time.sleep(3)
        subprocess.run(["bash", "scripts/container_case.sh", gpu, str(hub), SNAPSHOT,
                        MODEL, "v5", directory.relative_to(ROOT).as_posix(), "3"],
                       cwd=ROOT, env=env, check=True)
    manifest = read_json(directory / "source_manifest.json")
    require(all(manifest.get(key) == value for key, value in config.items()),
            f"Configuration mismatch: {directory}")
    phases, probes = load(directory, MODEL, "v5")
    return phases, probes, read_json(directory / "adapter_stats.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpu")
    parser.add_argument("hub", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    directory = (ROOT / args.output_prefix / "activation").resolve()
    require(directory.is_relative_to(ROOT) and directory != ROOT,
            "Output prefix must resolve inside the mounted experiment directory")
    calibration = directory / "calibration.json"
    require(args.resume or not directory.exists(), f"Output already exists: {directory}")
    for chunk in (512, 2048, 8192):
        phases, probes, stats = run_case(args.gpu, args.hub.resolve(),
                                         directory / f"telemetry-{chunk}",
                                         "telemetry", chunk, 0, args.resume)
    peak = stats["activation_prefill_peak_growth_bytes"]
    require(type(peak) is int and peak > 0, "Missing measured prefill allocation growth")
    quantum = 64 * MIB
    headroom = ((peak + quantum - 1) // quantum) * 64 + 128
    calibration_result = {
        "source": "telemetry-8192/adapter_stats.json",
        "activation_prefill_peak_growth_bytes": peak,
        "prefill_headroom_mib": headroom,
        "formula": "ceil(peak_bytes / (64 MiB)) * 64 MiB + 128 MiB",
        "note": "Empirical experiment setting. This does not guarantee memory safety.",
    }
    if calibration.exists():
        require(args.resume and read_json(calibration) == calibration_result,
                "Existing calibration does not match measured inputs")
    else:
        with calibration.open("x", encoding="utf-8") as stream:
            json.dump(calibration_result, stream, indent=2)
            stream.write("\n")
    both_phases, both_probes, _ = run_case(args.gpu, args.hub.resolve(),
                                          directory / "both-8192",
                                          "both", 8192, headroom, args.resume)
    for request_id, record in both_probes.items():
        require(all(record[key] == probes[request_id][key]
                    for key in ("prompt_sha256", "output_ids")),
                f"Same-chunk probe mismatch: {request_id}")
    for phase_key, phase in both_phases.items():
        require(all(a["prompt_sha256"] == b["prompt_sha256"]
                    for a, b in zip(phase["requests"], phases[phase_key]["requests"])),
                f"Same-chunk timed prompt mismatch: {phase_key}")


if __name__ == "__main__":
    main()
