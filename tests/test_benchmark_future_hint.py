from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.client import future_hint, payload_for
from benchmarks.hisparse_client import run_benchmark


class BenchmarkFutureHintTest(unittest.TestCase):
    def test_payload_omits_future_hint_by_default(self):
        payload = payload_for(
            {
                "request_id": "plain",
                "input_tokens": 16,
                "requested_output_tokens": 4,
            },
            stream=True,
        )
        self.assertNotIn("kv_hints", payload)

    def test_payload_carries_the_authored_future_hint(self):
        hint = future_hint("hinted", 7)
        payload = payload_for(
            {
                "request_id": "hinted",
                "input_tokens": 16,
                "requested_output_tokens": 7,
                "future_hint": hint,
            },
            stream=True,
        )
        self.assertIs(payload["kv_hints"], hint)
        self.assertEqual(
            payload["kv_hints"]["actions"][0]["payload"]["expected_output_tokens"],
            7,
        )

    def test_hisparse_correctness_probe_receives_a_matching_hint(self):
        seen = []

        def fake_generate(_url, request, _timeout, stream=True):
            self.assertFalse(stream)
            seen.append(request)
            return {
                "request_id": request["request_id"],
                "output_tokens": request["requested_output_tokens"],
                "requested_output_tokens": request["requested_output_tokens"],
                "output_ids_complete": True,
                "_started_perf_s": 0.0,
            }

        with tempfile.TemporaryDirectory() as directory, patch(
            "benchmarks.hisparse_client.run_phase", return_value={}
        ), patch("benchmarks.hisparse_client.generate", side_effect=fake_generate):
            output = Path(directory) / "performance.json"
            results, correctness = run_benchmark(
                "http://127.0.0.1:1",
                output,
                ["GPU-a", "GPU-b", "GPU-c", "GPU-d"],
                repeats=1,
                workloads=["short_serial"],
                warmup_once=True,
                emit_future_hints=True,
            )

            self.assertEqual(len(results["phases"]), 1)
            self.assertEqual(len(correctness["probes"]), 3)
            self.assertEqual(len(seen), 3)
            for request in seen:
                action = request["future_hint"]["actions"][0]
                self.assertEqual(
                    action["payload"]["expected_output_tokens"],
                    request["requested_output_tokens"],
                )
            persisted = json.loads(
                output.with_name(output.stem + ".correctness.json").read_text()
            )
            self.assertTrue(persisted["probes"])


if __name__ == "__main__":
    unittest.main()
