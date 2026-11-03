"""Validate and summarize the fixed post-warmup prefill pressure scenario."""

import argparse
import json
from pathlib import Path

from summarize import require

MIB = 1 << 20
FIELDS = ("prompt_sha256", "output_ids")
ACTIONS = ("activation_trim_bytes", "activation_flush_freed_bytes",
           "activation_trim_calls", "activation_flush_calls")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_case(directory, policy):
    result = read(directory / "results.json")
    manifest = read(directory / "source_manifest.json")
    require(result["outcome"] == "complete" and result["policy"] == manifest["policy"] == policy,
            f"{policy}: incomplete or incorrect policy")
    require(result["driver_free_target_bytes"] == manifest["target_free_mib"] * MIB
            and result["prefill_headroom_bytes"] == manifest["headroom_mib"] * MIB,
            f"{policy}: reported configuration differs from manifest")
    ref, probe = result["reference"], result["pressure_correctness"]
    for record in (ref, probe):
        require(record["output_ids_complete"] and record["output_tokens"] == 64
                and len(record["output_ids"]) == 64
                and all(type(token) is int for token in record["output_ids"])
                and bool(record["prompt_sha256"]), f"{policy}: incomplete 64-token probe")
    require(all(ref[key] == probe[key] for key in FIELDS), f"{policy}: probe mismatch")

    ballast = result["ballast"]
    ready_free = ballast["driver_free_ready_bytes"]
    require(abs(ready_free - result["driver_free_target_bytes"]) <= 64 * MIB
            and 0 < ballast["ballast_allocated_bytes"] <= manifest["ballast_limit_mib"] * MIB,
            f"{policy}: ballast missed bounds")
    requests = result["pressure_requests"]
    require([request["index"] for request in requests] == [0, 1, 2],
            f"{policy}: expected three timed requests")
    rows = []
    for request in requests:
        timed = request["timed"]
        require(timed["output_tokens"] == 64 and timed["prompt_sha256"] == ref["prompt_sha256"]
                and timed["ttft_s"] > 0 and timed["tpot_s"] > 0,
                f"{policy}: invalid timed request")
        if timed["output_ids_complete"]:
            require(timed["output_ids"] == ref["output_ids"], f"{policy}: timed token mismatch")
        delta = request["activation_delta"]
        actions = {key: delta.get(key, 0) for key in ACTIONS}
        require(all(value >= 0 and value == request["after"].get(key, 0)
                    - request["before"].get(key, 0) for key, value in actions.items()),
                f"{policy}: invalid action delta")
        rows.append({"index": request["index"], "ttft_s": timed["ttft_s"],
                     "tpot_s": timed["tpot_s"], "activation_delta": actions})
    if policy == "both":
        require(rows[0]["activation_delta"]["activation_trim_bytes"] > 0,
                "both: first pressure request did not trim backing")
    else:
        require(all(not any(row["activation_delta"].values()) for row in rows),
                "telemetry: unexpected trim or cache release")

    nvml = [sample["device_used_bytes"] for sample in result["whole_gpu_nvml_samples"]]
    require(len(nvml) >= 2 and all(type(value) is int and value > 0 for value in nvml),
            f"{policy}: missing whole-GPU NVML samples")
    summary = {"driver_free_ready_bytes": ready_free, "ballast": ballast,
               "first_pressure_request": rows[0], "subsequent_pressure_requests": rows[1:],
               "whole_gpu_nvml_before_bytes": nvml[0], "whole_gpu_nvml_after_bytes": nvml[-1],
               "whole_gpu_nvml_peak_bytes": max(nvml)}
    return manifest, probe, summary


def summarize(directory):
    control, reference, telemetry = load_case(directory / "telemetry", "telemetry")
    treatment, probe, both = load_case(directory / "both", "both")
    require({k: v for k, v in control.items() if k != "policy"}
            == {k: v for k, v in treatment.items() if k != "policy"},
            "Pressure arms have different configurations or source hashes")
    require(all(probe[key] == reference[key] for key in FIELDS),
            "Cross-arm pressure probe mismatch")
    difference = abs(telemetry["driver_free_ready_bytes"] - both["driver_free_ready_bytes"])
    require(difference <= 64 * MIB, "Cross-arm measured driver free differs by more than 64 MiB")
    return {"probe_tokens_matched": 64, "driver_free_difference_bytes": difference,
            "target_free_mib": control["target_free_mib"], "headroom_mib": control["headroom_mib"],
            "telemetry": telemetry, "both": both,
            "note": "Fixed mechanism scenario. Whole-GPU NVML samples include ballast."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = summarize(args.directory)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Summary rejected: {exc}\n")


if __name__ == "__main__":
    main()
