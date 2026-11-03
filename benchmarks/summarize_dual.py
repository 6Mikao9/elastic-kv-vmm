"""Validate and summarize the fixed two-instance workload using recorded samples."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median

from summarize import require


SCENARIOS = {"a-hot": ("a",), "b-hot": ("b",), "both-active": ("a", "b")}


def summarize(data, mode):
    require(data["mode"] == mode and data["models"] == {"a": "0.6B", "b": "4B"},
            "Unexpected mode or model assignment")
    phases = {(p["scenario"], p["repeat"]): p for p in data["phases"]}
    require(len(data["phases"]) == 9
            and phases.keys() == {(s, r) for s in SCENARIOS for r in range(1, 4)},
            "Expected three repeats of all three scenarios")
    for scenario, instances in SCENARIOS.items():
        rates, requests, active, idle = [], [], [], []
        for repeat in range(1, 4):
            phase = phases[scenario, repeat]
            records = phase["requests"]
            expected = {f"measured/{scenario}/{repeat}/{i}/{n}"
                        for i in instances for n in range(4)}
            require(len(records) == len(expected)
                    and {r["request_id"] for r in records} == expected,
                    "Missing or duplicate measured requests")
            require(all(r["output_tokens"] == r["requested_output_tokens"] == 128
                        and r["input_tokens"] == 1024 and r["output_ids_complete"]
                        and r["output_ids_sha256"] and r["ttft_s"] > 0 and r["tpot_s"] > 0
                        for r in records), "Incomplete measured response")
            output_tokens = sum(r["output_tokens"] for r in records)
            require(output_tokens == phase["output_tokens"] and phase["elapsed_s"] > 0,
                    "Invalid phase timing or token count")
            count = phase["active_sample_count"]
            require(0 < count < len(phase["samples"])
                    and phase["idle"] == phase["samples"][-1], "Incomplete memory sampling")
            if mode == "v2":
                stats = phase["idle"]["instances"]
                require(stats.keys() == {"a", "b"}
                        and all(s["live_tokens"] == 0 for s in stats.values()),
                        "Both V2 instances must return to zero live tokens")
            if "active_window_s" in phase:
                start, end = phase["active_window_s"]
                require(0 <= start < end, "Invalid active window")
                active_samples = [s for s in phase["samples"]
                                  if start <= s["offset_s"] <= end]
                require(len(active_samples) == count, "Invalid active sample count")
            else:
                active_samples = phase["samples"][:count]
            active.extend(s["device_used_bytes"] for s in active_samples)
            idle.append(phase["idle"]["device_used_bytes"])
            rates.append(output_tokens / phase["elapsed_s"])
            requests.extend(records)
        yield {
            "mode": mode, "scenario": scenario, "repeats": 3, "requests": len(requests),
            "nvml_idle_median_bytes": median(idle), "nvml_active_peak_bytes": max(active),
            "output_tokens_per_s_median": median(rates),
            "output_tokens_per_s_min": min(rates), "output_tokens_per_s_max": max(rates),
            "ttft_ms_median": 1000 * median(r["ttft_s"] for r in requests),
            "tpot_ms_median": 1000 * median(r["tpot_s"] for r in requests),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory with baseline and v2 results")
    args = parser.parse_args()
    rows = []
    prompts = {}
    try:
        for mode in ("baseline", "v2"):
            data = json.loads((args.directory / mode / "results.json").read_text(encoding="utf-8"))
            rows.extend(summarize(data, mode))
            prompts[mode] = {
                (phase["scenario"], phase["repeat"], record["request_id"]):
                record["prompt_sha256"]
                for phase in data["phases"] for record in phase["requests"]
            }
        require(all(prompts["baseline"].values()) and prompts["baseline"] == prompts["v2"],
                "Measured request IDs or prompt hashes differ between modes")
        (args.directory / "summary.json").write_text(
            json.dumps({"scenarios": rows}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (args.directory / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(2, f"Dual summary rejected: {exc}\n")


if __name__ == "__main__":
    main()
