from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.hisparse_locality_replay import replay, trace_rows


def row(step, selected, request_id="r", layer_id=0):
    return dict(request_id=request_id, layer_id=layer_id, step=step, selected=selected)


class LocalityReplayTests(unittest.TestCase):
    def test_batch_pins_late_hits_and_reports_capacity_effect(self):
        rows = [row(0, [0, 1]), row(1, [2, 0]), row(2, [1, 0])]
        result = replay(rows, capacities=(2, 3), window_size=2)
        self.assertEqual(result["policy"], "simulated_batch_token_lru")
        first, second = result["windows"]
        self.assertEqual(first["capacities"]["2"], {"misses": 3, "miss_rate": 0.75})
        self.assertEqual(first["turnover_rate"], 0.5)
        self.assertEqual(second["capacities"]["2"], {"misses": 1, "miss_rate": 0.5})
        self.assertEqual(second["capacities"]["3"], {"misses": 0, "miss_rate": 0})
        self.assertEqual(second["turnover_rate"], 0.5)

    def test_request_layer_and_gap_have_independent_cold_starts(self):
        rows = [
            row(0, [1]), row(0, [1], request_id="other"), row(0, [1], layer_id=1),
            row(1, [1]), row(3, [1]),
        ]
        windows = replay(rows, capacities=(2,))["windows"]
        self.assertEqual([window["capacities"]["2"]["misses"] for window in windows],
                         [1, 1, 1, 1])
        self.assertEqual([window["selected"] for window in windows], [2, 1, 1, 1])
        self.assertEqual(windows[0]["turnover_rate"], 0)
        self.assertEqual(windows[-1]["segment_start_step"], 3)
        self.assertIsNone(windows[-1]["turnover_rate"])

    def test_invalid_or_oversized_selection(self):
        for invalid in ([1, 1], [-1], [True], [0, 1, 2]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replay([row(0, invalid)], capacities=(2,))

    def test_rejects_duplicate_or_out_of_order_steps(self):
        with self.assertRaisesRegex(ValueError, "out-of-order"):
            replay([row(2, [1]), row(2, [2])], capacities=(2,))

    def test_native_capture_deduplicates_in_order_and_reports_truncation(self):
        events = [
            {"type": "metadata", "schema": "minimax-hisparse-access-v1", "rank": 0},
            {"type": "access", "rank": 0, "rid": "r", "layer": 5, "step": 0,
             "generation": 1, "selected_tokens": [3, 1, 3, 2, 1]},
            {"type": "truncation", "reason": "max_steps"},
        ]
        stats = {"duplicate_token_occurrences": 0, "truncation_markers": Counter()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            path.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
            access, = trace_rows([path], stats)
        self.assertEqual(access["selected"], [3, 1, 2])
        self.assertEqual(stats["duplicate_token_occurrences"], 2)
        self.assertEqual(stats["truncation_markers"], {"max_steps": 1})
