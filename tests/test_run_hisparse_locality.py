import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_hisparse_locality import CAPTURE_LAYERS, CAPTURE_RANKS, capture_evidence


class CaptureCoverageTest(unittest.TestCase):
    def test_requires_each_configured_layer_and_contiguous_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for rank in CAPTURE_RANKS:
                events = [
                    {"type": "metadata", "schema": "minimax-hisparse-access-v1",
                     "rank": rank, "layers": CAPTURE_LAYERS},
                    *({"type": "access", "rid": "r", "layer": layer, "rank": rank,
                       "step": step} for layer in CAPTURE_LAYERS for step in range(2)),
                ]
                path = directory / f"hisparse-rank{rank}.jsonl"
                path.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
            self.assertTrue(capture_evidence(directory, {"r": 3})["request_rank_coverage_ok"])
            events[-1]["step"] = 2
            path.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
            evidence = capture_evidence(directory, {"r": 3})
        self.assertFalse(evidence["request_rank_coverage_ok"])
        entry, = evidence["incomplete_request_rank_layers"]
        self.assertEqual((entry["rank"], entry["layer"]), (CAPTURE_RANKS[-1], CAPTURE_LAYERS[-1]))
