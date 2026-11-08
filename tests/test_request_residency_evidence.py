import unittest
from collections import OrderedDict
from types import SimpleNamespace

from adaptive_sparse.request_residency import (
    coordinator_broker_snapshot,
    remember_request_decision,
    resident_admission_fairness,
)


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

    def test_fair_resident_admission_observes_then_enforces_pressure(self):
        self.assertEqual(resident_admission_fairness(None, "off"), (False, False))
        observed = {
            "fair_mode": "observe",
            "fair_share_would_yield": True,
            "fairness_pressure": False,
        }
        self.assertEqual(
            resident_admission_fairness(observed, "observe"), (True, False)
        )
        enforced = {
            "fair_mode": "enforce",
            "fair_share_would_yield": True,
            "fairness_pressure": True,
        }
        self.assertEqual(
            resident_admission_fairness(enforced, "enforce"), (True, True)
        )

    def test_fair_resident_admission_rejects_stale_or_malformed_state(self):
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            resident_admission_fairness(
                {"fair_mode": "observe", "fairness_pressure": False},
                "enforce",
            )
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            resident_admission_fairness(
                {
                    "fair_mode": "enforce",
                    "fair_share_would_yield": 1,
                    "fairness_pressure": True,
                },
                "enforce",
            )
        with self.assertRaisesRegex(RuntimeError, "requires"):
            resident_admission_fairness(None, "enforce")

    def test_coordinator_snapshot_reads_callable_backing_property(self):
        calls = []
        backing = SimpleNamespace(
            broker_snapshot=lambda: calls.append(True) or {"fair_mode": "off"}
        )
        coordinator = SimpleNamespace(
            mem_pool_device=SimpleNamespace(
                adaptive_vmm=SimpleNamespace(backing=backing)
            )
        )
        self.assertEqual(coordinator_broker_snapshot(coordinator), {"fair_mode": "off"})
        self.assertEqual(calls, [True])


if __name__ == "__main__":
    unittest.main()
