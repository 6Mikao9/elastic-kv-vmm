import multiprocessing
import tempfile
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
