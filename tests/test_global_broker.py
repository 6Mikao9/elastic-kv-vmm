import multiprocessing
import tempfile
import threading
import time
import unittest
from pathlib import Path

from elastic_kv.global_broker import GlobalAdmissionBroker


def _race_for_lease(path, index, barrier, results):
    broker = GlobalAdmissionBroker(
        path,
        3,
        owner_id=f"worker-{index}-{__import__('os').getpid()}",
        heartbeat_seconds=None,
    )
    try:
        barrier.wait(timeout=10)
        decision = broker.reserve({f"owner-{index}/chunk-0": 1})
        results.put(decision.admit)
        barrier.wait(timeout=10)
    finally:
        broker.close()


def _race_for_mixed_capacity(path, index, barrier, results):
    broker = GlobalAdmissionBroker(
        path,
        3,
        owner_id=f"mixed-worker-{index}-{__import__('os').getpid()}",
        heartbeat_seconds=None,
    )
    try:
        barrier.wait(timeout=10)
        if index % 2:
            decision = broker.claim(f"claim-{index}", 1)
        else:
            decision = broker.reserve({f"lease-{index}": 1})
        results.put(decision.admit)
        barrier.wait(timeout=10)
    finally:
        broker.close()


class GlobalAdmissionBrokerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "gpu0.sqlite"

    def tearDown(self):
        self.temporary.cleanup()

    def broker(self, owner, capacity=10, **kwargs):
        return GlobalAdmissionBroker(
            self.path, capacity, owner_id=owner, heartbeat_seconds=None, **kwargs
        )

    def test_two_owners_share_one_capacity_without_partial_leases(self):
        first = self.broker("first")
        second = self.broker("second")
        try:
            decision = first.reserve({"chunk-0": 6})
            self.assertTrue(decision.admit)
            self.assertEqual(decision.available_bytes, 4)
            denied = second.reserve({"chunk-1": 3, "chunk-2": 3})
            self.assertFalse(denied.admit)
            self.assertEqual(denied.reason, "global-capacity-exhausted")
            self.assertEqual(second.outstanding_bytes(), 6)
            admitted = second.reserve({"chunk-1": 3, "chunk-2": 1})
            self.assertTrue(admitted.admit)
            self.assertEqual(admitted.available_bytes, 0)
        finally:
            first.close()
            second.close()

    def test_release_and_close_return_capacity(self):
        first = self.broker("first")
        second = self.broker("second")
        try:
            self.assertTrue(first.reserve({"chunk-0": 8}).admit)
            self.assertFalse(second.reserve({"chunk-1": 3}).admit)
            self.assertEqual(first.release(["chunk-0", "missing"]), 8)
            self.assertTrue(second.reserve({"chunk-1": 3}).admit)
            first.close()
            self.assertTrue(second.reserve({"chunk-2": 7}).admit)
            second.close()
            observer = self.broker("observer")
            try:
                self.assertEqual(observer.available_bytes(), 10)
            finally:
                observer.close()
        finally:
            first.close()
            second.close()

    def test_reserving_same_owner_key_is_idempotent(self):
        broker = self.broker("owner")
        try:
            first = broker.reserve({"chunk-0": 4})
            second = broker.reserve({"chunk-0": 4})
            self.assertEqual(first.acquired_keys, ("chunk-0",))
            self.assertEqual(second.reason, "already-held")
            self.assertEqual(second.required_bytes, 0)
            self.assertEqual(broker.outstanding_bytes(), 4)
        finally:
            broker.close()

    def test_prefill_claim_competes_with_vmm_leases(self):
        lease_owner = self.broker("lease-owner", capacity=100)
        prefill_owner = self.broker("prefill-owner", capacity=100)
        try:
            self.assertTrue(lease_owner.reserve({"kv-0": 60}).admit)
            claimed = prefill_owner.claim("decode/prefill-0", 30)
            self.assertTrue(claimed.admit)
            denied = lease_owner.reserve({"kv-1": 20})
            self.assertFalse(denied.admit)
            self.assertEqual(denied.reason, "global-capacity-exhausted")
            snapshot = lease_owner.snapshot()
            self.assertEqual(snapshot["outstanding_bytes"], 90)
            self.assertEqual(snapshot["lease_bytes"], 60)
            self.assertEqual(snapshot["claim_bytes"], 30)
            self.assertEqual(
                snapshot["claims"]["decode/prefill-0"],
                {"kind": "prefill", "bytes": 30},
            )

            self.assertEqual(prefill_owner.release_claim("decode/prefill-0"), 30)
            self.assertTrue(lease_owner.reserve({"kv-1": 20}).admit)
        finally:
            lease_owner.close()
            prefill_owner.close()

    def test_claim_is_idempotent_and_conflicts_across_kinds(self):
        broker = self.broker("owner", capacity=10)
        try:
            first = broker.claim("prefill-0", 4, kind="prefill")
            second = broker.claim("prefill-0", 4, kind="prefill")
            self.assertEqual(first.reason, "admitted")
            self.assertEqual(second.reason, "already-held")
            self.assertEqual(second.required_bytes, 0)
            self.assertEqual(broker.outstanding_bytes(), 4)
            with self.assertRaisesRegex(RuntimeError, "claim key conflict"):
                broker.claim("prefill-0", 5, kind="prefill")
            with self.assertRaisesRegex(RuntimeError, "claim key conflict"):
                broker.claim("prefill-0", 4, kind="rollout")
            with self.assertRaisesRegex(RuntimeError, "lease key conflict"):
                broker.reserve({"prefill-0": 4})
        finally:
            broker.close()

    def test_release_all_and_close_release_temporary_claims(self):
        first = self.broker("first", capacity=10)
        second = self.broker("second", capacity=10)
        try:
            first.reserve({"kv-0": 3})
            first.claim("prefill-0", 4)
            self.assertEqual(first.release_all(), 7)
            self.assertTrue(second.reserve({"kv-1": 7}).admit)
            first.claim("prefill-1", 2)
            first.close()
            self.assertEqual(second.outstanding_bytes(), 7)
            self.assertTrue(second.reserve({"kv-2": 2}).admit)
        finally:
            first.close()
            second.close()

    def test_process_race_combines_leases_and_claims_atomically(self):
        context = multiprocessing.get_context("spawn")
        workers = 8
        barrier = context.Barrier(workers)
        results = context.Queue()
        processes = [
            context.Process(
                target=_race_for_mixed_capacity,
                args=(str(self.path), index, barrier, results),
            )
            for index in range(workers)
        ]
        for process in processes:
            process.start()
        outcomes = [results.get(timeout=15) for _ in processes]
        for process in processes:
            process.join(timeout=15)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(sum(outcomes), 3)

    def test_same_key_cannot_be_claimed_by_two_owners(self):
        first = self.broker("first")
        second = self.broker("second")
        try:
            first.reserve({"device-0/chunk-0": 2})
            with self.assertRaisesRegex(RuntimeError, "lease key conflict"):
                second.reserve({"device-0/chunk-0": 2})
            self.assertEqual(second.outstanding_bytes(), 2)
        finally:
            first.close()
            second.close()

    def test_process_race_admits_only_capacity_count(self):
        context = multiprocessing.get_context("spawn")
        workers = 8
        barrier = context.Barrier(workers)
        results = context.Queue()
        processes = [
            context.Process(
                target=_race_for_lease,
                args=(str(self.path), index, barrier, results),
            )
            for index in range(workers)
        ]
        for process in processes:
            process.start()
        outcomes = [results.get(timeout=15) for _ in processes]
        for process in processes:
            process.join(timeout=15)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(sum(outcomes), 3)

    def test_capacity_mismatch_is_rejected(self):
        broker = self.broker("owner", capacity=10)
        try:
            with self.assertRaisesRegex(ValueError, "capacity does not match"):
                GlobalAdmissionBroker(
                    self.path, 11, owner_id="other", heartbeat_seconds=None
                )
        finally:
            broker.close()

    def test_resource_mismatch_is_rejected(self):
        broker = GlobalAdmissionBroker(
            self.path, 10, owner_id="owner", heartbeat_seconds=None,
            resource_id="gpu-0",
        )
        try:
            with self.assertRaisesRegex(ValueError, "different physical GPU"):
                GlobalAdmissionBroker(
                    self.path, 10, owner_id="other", heartbeat_seconds=None,
                    resource_id="gpu-1",
                )
        finally:
            broker.close()

    def test_dead_owner_leases_are_recovered_after_stale_interval(self):
        now = [100.0]
        from unittest.mock import patch

        with patch("elastic_kv.global_broker.time.time", side_effect=lambda: now[0]):
            dead = self.broker(
                "dead", stale_after_seconds=1, process_exists=lambda _: False,
                boot_id="boot",
            )
            dead.reserve({"dead-chunk": 7})
            now[0] = 102.0
            survivor = self.broker(
                "survivor", stale_after_seconds=1, process_exists=lambda _: False,
                boot_id="boot",
            )
            try:
                decision = survivor.reserve({"new-chunk": 7})
                self.assertTrue(decision.admit)
                self.assertEqual(survivor.outstanding_bytes(), 7)
            finally:
                survivor.close()
                dead.close()

    def test_fair_share_blocks_over_share_growth_for_starved_tenant(self):
        first = self.broker(
            "first", capacity=10, tenant_id="model-a", fair_mode="enforce"
        )
        second = self.broker(
            "second", capacity=10, tenant_id="model-b", fair_mode="enforce"
        )
        try:
            self.assertTrue(first.reserve({"a-0": 10}).admit)
            denied = second.reserve({"b-0": 2})
            self.assertEqual(denied.reason, "global-capacity-exhausted")
            self.assertEqual(first.release(["a-0"]), 10)
            self.assertTrue(first.reserve({"a-1": 5}).admit)
            yielded = first.reserve({"a-2": 1})
            self.assertFalse(yielded.admit)
            self.assertEqual(yielded.reason, "fair-share-yield")
            self.assertEqual(yielded.starved_tenants, ("model-b",))
            self.assertTrue(second.reserve({"b-0": 2}).admit)
        finally:
            first.close()
            second.close()

    def test_fair_share_rejects_batch_that_would_overshoot_target(self):
        first = self.broker(
            "first", capacity=10, tenant_id="model-a", fair_mode="enforce"
        )
        second = self.broker(
            "second", capacity=10, tenant_id="model-b", fair_mode="enforce"
        )
        try:
            self.assertTrue(first.reserve({"a-0": 10}).admit)
            self.assertFalse(second.reserve({"b-0": 1}).admit)
            first.release(["a-0"])
            self.assertTrue(first.reserve({"a-1": 4}).admit)

            yielded = first.reserve({"a-2": 2})

            self.assertFalse(yielded.admit)
            self.assertEqual(yielded.reason, "fair-share-yield")
            self.assertEqual(yielded.tenant_usage_bytes, 4)
            self.assertEqual(yielded.starved_tenants, ("model-b",))
            self.assertEqual(first.outstanding_bytes(), 4)
            self.assertTrue(second.reserve({"b-1": 5}).admit)
        finally:
            first.close()
            second.close()

    def test_fair_share_reports_weighted_tp_targets(self):
        first = self.broker(
            "first", capacity=100, tenant_id="model-a", tenant_weight=3,
            fair_mode="observe",
        )
        second = self.broker(
            "second", capacity=100, tenant_id="model-b", tenant_weight=1,
            fair_mode="observe",
        )
        try:
            first.reserve({"a-0": 100})
            self.assertFalse(second.reserve({"b-0": 1}).admit)
            self.assertEqual(first.snapshot()["fair_target_bytes"], 75)
            self.assertEqual(second.snapshot()["fair_target_bytes"], 25)
            self.assertEqual(second.snapshot()["tenant_deficit_bytes"], 25)
        finally:
            first.close()
            second.close()

    def test_observe_mode_records_would_yield_but_does_not_block(self):
        first = self.broker(
            "first", capacity=10, tenant_id="model-a", fair_mode="observe"
        )
        second = self.broker(
            "second", capacity=10, tenant_id="model-b", fair_mode="observe"
        )
        try:
            first.reserve({"a-0": 10})
            self.assertFalse(second.reserve({"b-0": 1}).admit)
            first.release(["a-0"])
            observed = first.reserve({"a-1": 8})
            self.assertTrue(observed.admit)
            self.assertEqual(observed.reason, "fair-share-observed")
            self.assertTrue(first.snapshot()["fair_share_would_yield"])
        finally:
            first.close()
            second.close()

    def test_bounded_wait_succeeds_after_cooperating_owner_releases(self):
        first = self.broker("first", capacity=1)
        second = self.broker("second", capacity=1, reserve_wait_seconds=0.5)
        self.assertTrue(first.reserve({"a-0": 1}).admit)

        def release():
            time.sleep(0.05)
            first.release(["a-0"])

        thread = threading.Thread(target=release)
        thread.start()
        try:
            decision = second.reserve({"b-0": 1})
            self.assertTrue(decision.admit)
            self.assertGreater(decision.retry_count, 0)
            self.assertGreater(decision.waited_seconds, 0)
        finally:
            thread.join(timeout=2)
            first.close()
            second.close()

    def test_expired_demand_stops_protecting_an_idle_tenant(self):
        from unittest.mock import patch

        now = [100.0]
        with patch("elastic_kv.global_broker.time.time", side_effect=lambda: now[0]):
            first = self.broker(
                "first", capacity=10, tenant_id="model-a", fair_mode="enforce",
                demand_ttl_seconds=5,
            )
            second = self.broker(
                "second", capacity=10, tenant_id="model-b", fair_mode="enforce",
                demand_ttl_seconds=5,
            )
            try:
                first.reserve({"a-0": 10})
                self.assertFalse(second.reserve({"b-0": 1}).admit)
                first.release(["a-0"])
                self.assertTrue(first.reserve({"a-1": 5}).admit)
                now[0] = 106.0
                self.assertTrue(first.reserve({"a-2": 2}).admit)
            finally:
                first.close()
                second.close()

    def test_snapshot_ignores_expired_demand_without_a_write_transaction(self):
        from unittest.mock import patch

        now = [100.0]
        with patch("elastic_kv.global_broker.time.time", side_effect=lambda: now[0]):
            first = self.broker(
                "first", capacity=10, tenant_id="model-a", fair_mode="enforce",
                demand_ttl_seconds=5,
            )
            second = self.broker(
                "second", capacity=10, tenant_id="model-b", fair_mode="enforce",
                demand_ttl_seconds=5,
            )
            try:
                first.reserve({"a-0": 10})
                self.assertFalse(second.reserve({"b-0": 1}).admit)
                first.release(["a-0"])
                self.assertTrue(first.reserve({"a-1": 5}).admit)
                self.assertTrue(first.snapshot()["fair_share_would_yield"])

                now[0] = 106.0
                self.assertFalse(first.snapshot()["fair_share_would_yield"])
                self.assertTrue(first.reserve({"a-2": 2}).admit)
            finally:
                first.close()
                second.close()

    def test_fair_configuration_must_be_unambiguous(self):
        with self.assertRaisesRegex(ValueError, "stable tenant_id"):
            GlobalAdmissionBroker(
                self.path, 10, fair_mode="enforce", heartbeat_seconds=None
            )
        owner = self.broker(
            "owner", capacity=10, tenant_id="model", fair_mode="off"
        )
        try:
            with self.assertRaisesRegex(ValueError, "fair mode does not match"):
                self.broker(
                    "other", capacity=10, tenant_id="model", fair_mode="enforce"
                )
        finally:
            owner.close()
        with self.assertRaisesRegex(ValueError, r"within \[0, 5\]"):
            GlobalAdmissionBroker(
                self.path, 10, heartbeat_seconds=None, reserve_wait_seconds=5.1
            )

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            GlobalAdmissionBroker(self.path, 0, heartbeat_seconds=None)
        broker = self.broker("owner")
        try:
            with self.assertRaises(ValueError):
                broker.reserve({"chunk": 0})
            with self.assertRaises(ValueError):
                broker.reserve({"": 1})
        finally:
            broker.close()


if __name__ == "__main__":
    unittest.main()
