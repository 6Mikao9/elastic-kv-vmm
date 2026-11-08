import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.summarize_dual import SCENARIOS, summarize


class DualSummaryTest(unittest.TestCase):
    def test_mutable_workload_lengths_are_validated_against_each_phase(self):
        workload = {
            "requests_per_instance": 1,
            "input_tokens": 8,
            "output_tokens": 4,
            "arrival_delay_ms": {"a": 0, "b": 0},
        }
        data = {
            "status": "complete",
            "arm": "baseline",
            "mode": "baseline",
            "models": {"a": "0.6B", "b": "4B"},
            "workload": workload,
            "phases": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for repeat in range(1, 4):
                for scenario, instances in SCENARIOS.items():
                    stem = f"repeat-{repeat:02d}-{scenario}"
                    requests = [{
                        "request_id": f"measured/{scenario}/{repeat}/{instance}/0",
                        "prompt_sha256": "a" * 64,
                        "input_tokens": 8,
                        "requested_output_tokens": 4,
                        "output_tokens": 4,
                        "output_ids_complete": True,
                        "output_ids_sha256": "b" * 64,
                        "ttft_s": 0.01,
                        "tpot_s": 0.02,
                    } for instance in instances]
                    samples = [
                        {"offset_s": 0.1, "device_used_bytes": 4096},
                        {"offset_s": 0.3, "device_used_bytes": 2048, "instances": {}},
                    ]
                    (output / f"{stem}.requests.json").write_text(
                        json.dumps(requests), encoding="utf-8"
                    )
                    (output / f"{stem}.samples.json").write_text(
                        json.dumps(samples), encoding="utf-8"
                    )
                    data["phases"].append({
                        "name": scenario,
                        "scenario": scenario,
                        "repeat": repeat,
                        "active_window_s": [0.0, 0.2],
                        "idle": samples[-1],
                        "summary": {
                            "elapsed_s": 0.5,
                            "output_tokens": 4 * len(instances),
                            "active_sample_count": 1,
                        },
                    })
            rows = list(summarize(data, "baseline", output))
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            {row["scenario"]: row["requests"] for row in rows},
            {"a-hot": 3, "b-hot": 3, "both-active": 6},
        )


if __name__ == "__main__":
    unittest.main()
