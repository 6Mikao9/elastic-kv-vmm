import unittest
from collections import OrderedDict

from adaptive_sparse.request_residency import remember_request_decision


class RequestResidencyEvidenceTest(unittest.TestCase):
    def test_keeps_bounded_latest_rank_local_decisions(self):
        stats = {"requests": OrderedDict()}
        for index in range(3):
            remember_request_decision(
                stats,
                f"request-{index}",
                actual_residence="resident" if index == 2 else "host_backed",
                reason="test",
                requested_residence="auto",
                expected_output_tokens=8,
                sequence_tokens=128,
                native_buffer_tokens=64,
                slots_available=True,
                tp_world_size=2,
                weight=3,
                deadline_after_ms=400,
                kind="interactive",
                prefix_group="system",
                shared_prefix_tokens=128,
            )

        decisions = stats["requests"]
        self.assertEqual(list(decisions), ["request-0", "request-1", "request-2"])
        self.assertEqual(decisions["request-2"]["actual_residence"], "resident")
        self.assertEqual(decisions["request-1"]["tp_world_size"], 2)
        self.assertEqual(decisions["request-2"]["weight"], 3)
        self.assertEqual(decisions["request-2"]["kind"], "interactive")
        self.assertEqual(decisions["request-2"]["prefix_group"], "system")

    def test_ignores_a_missing_request_id(self):
        stats = {"requests": OrderedDict()}
        remember_request_decision(
            stats,
            None,
            actual_residence="host_backed",
            reason="test",
            requested_residence=None,
            expected_output_tokens=None,
            sequence_tokens=1,
            native_buffer_tokens=0,
            slots_available=False,
            tp_world_size=1,
        )
        self.assertEqual(stats["requests"], OrderedDict())


if __name__ == "__main__":
    unittest.main()
