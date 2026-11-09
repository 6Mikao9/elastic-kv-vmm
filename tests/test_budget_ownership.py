import unittest
from types import SimpleNamespace

from adaptive_sparse.budget_ownership import (
    build_budget_ownership_snapshot,
)
from adaptive_sparse.vmm_runtime import attach_budget_ownership


class BudgetOwnershipTest(unittest.TestCase):
    def test_attributes_physical_broker_and_advisory_request_growth(self):
        snapshot = build_budget_ownership_snapshot(
            physical_broker_snapshot={
                "capacity_bytes": 100,
                "outstanding_bytes": 70,
                "available_bytes": 30,
                "lease_bytes": 60,
                "claim_bytes": 10,
                "claims": {
                    "prefill-a": {"kind": "prefill", "bytes": 10}
                },
                "fair_mode": "observe",
            },
            request_growth_snapshot={
                "outstanding_bytes": 20,
                "active_reservations": 2,
                "assessments": 4,
                "would_admit": 3,
                "would_decline": 1,
                "decline_reasons": {"insufficient-growth-headroom": 1},
            },
            request_growth_mode="enforce",
            vmm_backing_snapshot={
                "backed_bytes": 60,
                "live_bytes": 45,
                "reclaimable_bytes": 15,
                "mapped_chunks": 3,
            },
            activation_counters={
                "activation_global_prefill_claim_admissions": 2,
                "unrelated": 9,
            },
        )

        self.assertEqual(
            snapshot["physical_broker"]["active_claim_kinds"],
            {"prefill": 10},
        )
        self.assertEqual(
            snapshot["request_growth"]["authority"],
            "predictive-advisory",
        )
        self.assertFalse(
            snapshot["relationship"][
                "request_growth_is_additive_to_physical_usage"
            ]
        )
        self.assertEqual(
            snapshot["activation"]["counters"],
            {"activation_global_prefill_claim_admissions": 2},
        )

    def test_rejects_inconsistent_broker_partition(self):
        with self.assertRaisesRegex(ValueError, "partition"):
            build_budget_ownership_snapshot(
                physical_broker_snapshot={
                    "capacity_bytes": 100,
                    "outstanding_bytes": 70,
                    "available_bytes": 30,
                    "lease_bytes": 60,
                    "claim_bytes": 20,
                }
            )

    def test_disabled_layers_are_explicit(self):
        snapshot = build_budget_ownership_snapshot()

        self.assertEqual(snapshot["physical_broker"]["status"], "disabled")
        self.assertEqual(snapshot["request_growth"]["status"], "disabled")
        self.assertIsNone(snapshot["vmm_backing"])
        self.assertEqual(snapshot["activation"]["status"], "disabled")
        self.assertFalse(
            snapshot["relationship"]["physical_broker_is_authoritative"]
        )

    def test_vmm_runtime_attachment_is_best_effort(self):
        tracker = SimpleNamespace(
            backing=SimpleNamespace(
                metrics={
                    "global_broker": {
                        "capacity_bytes": 100,
                        "outstanding_bytes": 40,
                        "available_bytes": 60,
                        "lease_bytes": 40,
                        "claim_bytes": 0,
                        "claims": {},
                    }
                }
            ),
            counters={},
        )
        stats = {
            "request_residency": {
                "resident_broker_mode": "enforce",
                "resident_broker": {
                    "outstanding_bytes": 10,
                    "active_reservations": 1,
                },
            }
        }

        attached = attach_budget_ownership(
            stats,
            tracker=tracker,
            tracker_stats={"backed_bytes": 40},
        )

        self.assertEqual(
            attached["budget_ownership"]["physical_broker"]["lease_bytes"],
            40,
        )
        self.assertNotIn("budget_ownership_error", attached)


if __name__ == "__main__":
    unittest.main()
