import unittest

from adaptive_sparse.future_residency import decide_resident_claim


class FutureResidencyTest(unittest.TestCase):
    def decide(self, **overrides):
        arguments = {
            "sequence_tokens": 8192,
            "native_buffer_tokens": 4096,
            "slots_available": True,
            "minimum_hint_output_tokens": 256,
        }
        arguments.update(overrides)
        return decide_resident_claim(**arguments)

    def test_no_hint_preserves_long_request_default(self):
        decision = self.decide()
        self.assertTrue(decision.claim)
        self.assertEqual(decision.reason, "long-request-default")

    def test_request_that_fits_native_hot_set_never_claims_full_kv(self):
        decision = self.decide(sequence_tokens=4096)
        self.assertFalse(decision.claim)
        self.assertEqual(decision.reason, "fits-native-hot-set")

    def test_explicit_resident_overrides_short_prediction(self):
        decision = self.decide(
            hint_residence="resident", expected_output_tokens=8
        )
        self.assertTrue(decision.claim)
        self.assertEqual(decision.reason, "hint-resident")

    def test_explicit_host_backed_yields_slot(self):
        decision = self.decide(
            hint_residence="host_backed", expected_output_tokens=7000
        )
        self.assertFalse(decision.claim)
        self.assertEqual(decision.reason, "hint-host-backed")

    def test_auto_uses_predicted_output_length(self):
        short = self.decide(hint_residence="auto", expected_output_tokens=64)
        long = self.decide(hint_residence="auto", expected_output_tokens=1024)
        self.assertEqual((short.claim, short.reason), (False, "hint-short-auto"))
        self.assertEqual((long.claim, long.reason), (True, "hint-long-auto"))

    def test_auto_without_prediction_preserves_long_request_default(self):
        decision = self.decide(hint_residence="auto")
        self.assertTrue(decision.claim)
        self.assertEqual(decision.reason, "hint-auto-missing-output")

    def test_slot_pressure_falls_back_to_native(self):
        decision = self.decide(hint_residence="resident", slots_available=False)
        self.assertFalse(decision.claim)
        self.assertEqual(decision.reason, "no-resident-slot")

    def test_invalid_values_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "unknown residence"):
            self.decide(hint_residence="pinned")
        with self.assertRaisesRegex(ValueError, "expected_output_tokens"):
            self.decide(expected_output_tokens=0)
        with self.assertRaisesRegex(ValueError, "minimum_hint_output_tokens"):
            self.decide(minimum_hint_output_tokens=0)


if __name__ == "__main__":
    unittest.main()
