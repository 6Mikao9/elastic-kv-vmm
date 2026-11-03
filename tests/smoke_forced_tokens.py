"""Run explicitly in the benchmark SGLang image with CPU tensors."""

import unittest
from types import SimpleNamespace

import torch

from benchmarks.forced_tokens import ForceTokens, forced_ids, forced_payload, verify_forced_output
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


def params(ids, step, limit=None):
    return {"forced_ids": ids, "__req__": SimpleNamespace(
        output_ids=[0] * step,
        sampling_params=SimpleNamespace(max_new_tokens=len(ids) if limit is None else limit),
    )}


class ForcedTokensTest(unittest.TestCase):
    def test_row_steps_and_nan_replacement(self):
        logits = torch.full((2, 5), torch.nan)
        processor = CustomLogitProcessor.from_str(forced_payload(128, 16)["custom_logit_processor"])
        result = processor(logits, [params([3, 1], 0), params([0, 4], 1)])
        self.assertEqual(result.argmax(1).tolist(), [3, 4])
        self.assertEqual(torch.isfinite(result).sum(1).tolist(), [1, 1])
        self.assertTrue(torch.isnan(logits).all())

    def test_invalid_inputs(self):
        for value in (params([5], 0), params([-1], 0), params([True], 0),
                      params([2], 1), params([1], 0, limit=2)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ForceTokens()(torch.zeros(1, 5), [value])

    def test_complete_record(self):
        expected = forced_ids(8192, 128)
        record = {"output_ids_complete": True, "output_ids": expected, "output_tokens": 128}
        verify_forced_output(record, expected)
        with self.assertRaises(ValueError):
            verify_forced_output({**record, "output_ids": expected[:-1]}, expected)


if __name__ == "__main__":
    unittest.main()
