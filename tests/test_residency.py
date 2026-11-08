"""Slot ownership tests that do not need a CUDA device."""

import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from elastic_kv import adapter

CHUNK = 2 << 20
SLOT = 3 * CHUNK // 4


class FakeArena:
    def __init__(self):
        self.mapped_chunks = {0}
        self.precommitted_chunks = set()
        self.reclaimed = []

    @property
    def backed_bytes(self):
        return len(self.mapped_chunks | self.precommitted_chunks) * CHUNK

    @property
    def allocated_chunks(self):
        return self.mapped_chunks | self.precommitted_chunks

    def ensure(self, indices):
        self.mapped_chunks.update(indices)
        self.precommitted_chunks.difference_update(indices)

    def precommit(self, indices):
        pending = set(indices) - self.allocated_chunks
        self.precommitted_chunks.update(pending)
        return len(pending)

    def reclaim(self, indices, *, synchronize=True):
        if synchronize:
            raise AssertionError("the controller must synchronize the batch once")
        assert indices <= self.allocated_chunks
        self.reclaimed.append(set(indices))
        self.mapped_chunks.difference_update(indices)
        self.precommitted_chunks.difference_update(indices)


class ResidencyTest(unittest.TestCase):
    def setUp(self):
        for setting in (
            patch.dict(os.environ, {"EKV_CHUNK_MIB": "2", "EKV_STATS_PATH": ""}),
            patch.object(adapter, "MODE", "v1"),
            patch.object(adapter, "CONTROLLERS", []),
        ):
            setting.start()
            self.addCleanup(setting.stop)
        synchronization = patch.object(torch.cuda, "synchronize")
        self.synchronize = synchronization.start()
        self.addCleanup(synchronization.stop)
        pool = SimpleNamespace(
            size=8192,
            page_size=1,
            head_num=1,
            head_dim=1,
            dtype=torch.float16,
            layer_num=1,
            device="cuda:0",
        )
        self.arenas = [FakeArena(), FakeArena()]
        self.ctrl = adapter.Residency(pool, self.arenas, SLOT, "test")

    def assert_mapping(self, expected):
        for arena in self.arenas:
            self.assertEqual(arena.mapped_chunks, expected)

    def test_idle_policy_retains_only_configured_mapped_prefix(self):
        with patch.object(adapter, "MODE", "v5"):
            self.ctrl = adapter.Residency(self.ctrl.pool, self.arenas, SLOT, "test")
            self.ctrl.retain_chunks = 2
            self.ctrl.admit_ranges([(1, 4)])
            self.assertEqual(self.ctrl.live_count, 3)
            self.assert_mapping({0, 1, 2})
            self.ctrl.release_idle_only(np.array([1, 2]))
            self.ctrl.reclaim_idle()
            self.assert_mapping({0, 1, 2})
            self.synchronize.assert_not_called()
            self.ctrl.release_idle_only(np.array([3]))
            self.ctrl.reclaim_idle()
            self.assertEqual(self.ctrl.live_count, 0)
            self.assert_mapping({0, 1})
            self.synchronize.assert_called_once()

    def test_idle_policy_rejects_duplicate_free_before_mutation(self):
        with patch.object(adapter, "MODE", "v5"):
            self.ctrl = adapter.Residency(self.ctrl.pool, self.arenas, SLOT, "test")
        self.ctrl.admit_ranges([(1, 3)])
        before = self.ctrl.live.copy()
        with self.assertRaises(RuntimeError):
            self.ctrl.release_idle_only(np.array([1, 1]))
        np.testing.assert_array_equal(self.ctrl.live, before)
        self.assertEqual(self.ctrl.live_count, 2)

    def test_optional_lookahead_maps_one_chunk_ahead_and_reclaims_it(self):
        with (
            patch.object(adapter, "MODE", "v5"),
            patch.dict(os.environ, {"EKV_PRECOMMIT_CHUNKS": "1"}),
        ):
            self.ctrl = adapter.Residency(self.ctrl.pool, self.arenas, SLOT, "test")
        self.ctrl.retain_chunks = 2
        self.ctrl.admit_ranges([(1, 4)])
        self.assertEqual(self.ctrl.precommitted, {3})
        for arena in self.arenas:
            self.assertEqual(arena.mapped_chunks, {0, 1, 2})
            self.assertEqual(arena.precommitted_chunks, {3})

        self.ctrl.admit_ranges([(4, 5)])
        self.assertEqual(self.ctrl.precommitted, set())
        for arena in self.arenas:
            self.assertEqual(arena.mapped_chunks, {0, 1, 2, 3})

        self.ctrl.release_idle_only(np.array([1, 2, 3, 4]))
        self.ctrl.reclaim_idle()
        self.assertEqual(self.ctrl.live_count, 0)
        for arena in self.arenas:
            self.assertEqual(arena.mapped_chunks, {0, 1})
            self.assertEqual(arena.precommitted_chunks, set())

    def test_precommit_configuration_is_bounded(self):
        for value in ("-1", "5"):
            with self.subTest(value=value):
                with (
                    patch.object(adapter, "MODE", "v5"),
                    patch.dict(os.environ, {"EKV_PRECOMMIT_CHUNKS": value}),
                ):
                    with self.assertRaisesRegex(ValueError, "EKV_PRECOMMIT_CHUNKS"):
                        adapter.Residency(self.ctrl.pool, self.arenas, SLOT, "test")

    def test_predictive_prewarm_requires_bounded_v5_configuration(self):
        with (
            patch.object(adapter, "MODE", "v1"),
            patch.dict(os.environ, {"EKV_PREDICTIVE_PREWARM_MODE": "enforce"}),
        ):
            with self.assertRaisesRegex(ValueError, "requires EKV_MODE=v5"):
                adapter.Residency(self.ctrl.pool, self.arenas, SLOT, "test")
        for values in (
            {"EKV_PREDICTIVE_PREWARM_MODE": "invalid"},
            {
                "EKV_PREDICTIVE_PREWARM_MODE": "off",
                "EKV_PREDICTIVE_PREWARM_TTL_MS": "-1",
            },
            {
                "EKV_PREDICTIVE_PREWARM_MODE": "dry_run",
                "EKV_PREDICTIVE_PREWARM_TTL_MS": "0",
            },
            {
                "EKV_PREDICTIVE_PREWARM_MODE": "enforce",
                "EKV_PREDICTIVE_PREWARM_MAX_CHUNKS": "0",
            },
        ):
            with self.subTest(values=values):
                with (
                    patch.object(adapter, "MODE", "v5"),
                    patch.dict(os.environ, values),
                ):
                    with self.assertRaises(ValueError):
                        adapter.Residency(self.ctrl.pool, self.arenas, SLOT, "test")

    def test_predictive_prewarm_rejects_invalid_demand_and_time(self):
        for demand in (-1, True, 1.5):
            with self.subTest(demand=demand):
                with self.assertRaisesRegex(ValueError, "demand"):
                    self.ctrl.predictive_prewarm(demand)
        with self.assertRaisesRegex(ValueError, "time"):
            self.ctrl.predictive_prewarm(1, now_s=float("nan"))

    def test_predictive_prewarm_dry_run_does_not_claim_or_map_slots(self):
        with (
            patch.object(adapter, "MODE", "v5"),
            patch.dict(
                os.environ,
                {
                    "EKV_PREDICTIVE_PREWARM_MODE": "dry_run",
                    "EKV_PREDICTIVE_PREWARM_MAX_CHUNKS": "2",
                },
            ),
        ):
            self.ctrl = adapter.Residency(
                self.ctrl.pool, self.arenas, SLOT, "test"
            )
        self.ctrl.free_slots = adapter.FreeSlots(self.ctrl.pool.size)
        free_before = self.ctrl.free_slots.count

        selected = self.ctrl.predictive_prewarm(
            SLOT * 4, now_s=time.monotonic()
        )

        self.assertGreater(selected, 0)
        self.assertEqual(self.ctrl.precommitted, set())
        self.assertEqual(self.ctrl.free_slots.count, free_before)
        self.assertEqual(self.ctrl.live_count, 0)
        self.assert_mapping({0})
        self.assertEqual(
            self.ctrl.counters["predictive_prewarm_dry_run_chunks"], selected
        )

    def test_predictive_prewarm_is_consumed_by_native_slot_admission(self):
        with (
            patch.object(adapter, "MODE", "v5"),
            patch.dict(
                os.environ,
                {
                    "EKV_PREDICTIVE_PREWARM_MODE": "enforce",
                    "EKV_PREDICTIVE_PREWARM_TTL_MS": "1000",
                    "EKV_PREDICTIVE_PREWARM_MAX_CHUNKS": "2",
                },
            ),
        ):
            self.ctrl = adapter.Residency(
                self.ctrl.pool, self.arenas, SLOT, "test"
            )
        self.ctrl.free_slots = adapter.FreeSlots(self.ctrl.pool.size)
        free_before = self.ctrl.free_slots.count

        selected = self.ctrl.predictive_prewarm(
            SLOT * 4, now_s=time.monotonic()
        )
        self.assertGreater(selected, 0)
        self.assertEqual(self.ctrl.free_slots.count, free_before)
        self.assertEqual(len(self.ctrl.precommitted), selected)

        ranges = self.ctrl.free_slots.peek(2)
        self.ctrl.free_slots.claim(ranges)
        self.ctrl.admit_ranges(ranges)

        self.assertGreater(
            self.ctrl.counters["predictive_prewarm_hit_chunks"], 0
        )
        self.assertEqual(self.ctrl._predictive_precommitted, set())
        self.assertEqual(self.ctrl.free_slots.count, free_before - 2)
        self.assertEqual(self.ctrl.live_count, 2)

    def test_predictive_prewarm_expiry_reclaims_only_idle_lease(self):
        with (
            patch.object(adapter, "MODE", "v5"),
            patch.dict(
                os.environ,
                {
                    "EKV_PREDICTIVE_PREWARM_MODE": "enforce",
                    "EKV_PREDICTIVE_PREWARM_TTL_MS": "10",
                    "EKV_PREDICTIVE_PREWARM_MAX_CHUNKS": "2",
                },
            ),
        ):
            self.ctrl = adapter.Residency(
                self.ctrl.pool, self.arenas, SLOT, "test"
            )
        self.ctrl.free_slots = adapter.FreeSlots(self.ctrl.pool.size)
        now = time.monotonic()
        selected = self.ctrl.predictive_prewarm(SLOT * 4, now_s=now)

        expired = self.ctrl.expire_predictive_prewarm(now_s=now + 0.02)

        self.assertEqual(expired, selected)
        self.assertEqual(self.ctrl.precommitted, set())
        self.assertEqual(self.ctrl._predictive_precommitted, set())
        self.assert_mapping({0})
        self.assertEqual(
            self.ctrl.counters["predictive_prewarm_expired_chunks"], selected
        )

    def test_repeated_predictive_plan_does_not_extend_existing_lease(self):
        with (
            patch.object(adapter, "MODE", "v5"),
            patch.dict(
                os.environ,
                {
                    "EKV_PREDICTIVE_PREWARM_MODE": "enforce",
                    "EKV_PREDICTIVE_PREWARM_TTL_MS": "100",
                    "EKV_PREDICTIVE_PREWARM_MAX_CHUNKS": "4",
                },
            ),
        ):
            self.ctrl = adapter.Residency(
                self.ctrl.pool, self.arenas, SLOT, "test"
            )
        self.ctrl.free_slots = adapter.FreeSlots(self.ctrl.pool.size)
        now = time.monotonic()
        self.ctrl.predictive_prewarm(SLOT * 8, now_s=now)
        original = {
            chunk: self.ctrl._predictive_prewarm_deadlines[chunk]
            for chunk in self.ctrl._predictive_precommitted
        }

        self.ctrl.predictive_prewarm(SLOT * 8, now_s=now + 0.05)

        for chunk, deadline in original.items():
            self.assertEqual(
                self.ctrl._predictive_prewarm_deadlines[chunk], deadline
            )
        self.assertEqual(len(self.ctrl._predictive_precommitted), 4)

        # A repeated plan consumes no additional speculative budget once the
        # outstanding lease set reaches its configured cap.
        self.assertEqual(
            self.ctrl.predictive_prewarm(SLOT * 8, now_s=now + 0.075), 0
        )
        self.assertEqual(len(self.ctrl._predictive_precommitted), 4)

    def test_crossing_slots_retain_shared_chunks(self):
        # Slot 1 touches chunks 0,1. Slot 2 touches chunks 1,2.
        # Slot 3 touches only chunk 2, leaving it live after 1 and 2 finish.
        self.ctrl.admit(np.array([1, 2, 3], dtype=np.int64))
        np.testing.assert_array_equal(self.ctrl.counts[:3], [2, 2, 2])
        self.assert_mapping({0, 1, 2})

        self.ctrl.release(np.array([2], dtype=np.int32))
        np.testing.assert_array_equal(self.ctrl.counts[:3], [2, 1, 1])
        self.assert_mapping({0, 1, 2})
        self.synchronize.assert_not_called()

        self.ctrl.release(np.array([1], dtype=np.int32))
        np.testing.assert_array_equal(self.ctrl.counts[:3], [1, 0, 1])
        self.assert_mapping({0, 2})
        self.assertTrue(self.ctrl.live[3])
        self.synchronize.assert_called_once()

        self.ctrl.release(np.array([3], dtype=np.int32))
        self.assert_mapping({0})
        for arena in self.arenas:
            self.assertEqual(arena.reclaimed, [{1}, {2}])
        self.assertEqual(self.synchronize.call_count, 2)

    def test_int32_release_past_two_gib(self):
        slot = 1366
        self.assertGreater(slot * SLOT, 2**31)
        self.ctrl.admit(np.array([slot], dtype=np.int64))
        self.assert_mapping({0, 1024, 1025})
        self.ctrl.release(np.array([slot], dtype=np.int32))
        self.assert_mapping({0})
        self.assertTrue(np.all(self.ctrl.counts[1:] == 0))

    def test_invalid_and_duplicate_frees_leave_state_intact(self):
        self.ctrl.admit(np.array([1], dtype=np.int64))
        before = self.ctrl.counts.copy()
        for ids in ([0], [-1], [8193], [2], [1, 1], [1, 2]):
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(RuntimeError, "non-live|duplicate"):
                    self.ctrl.release(np.array(ids, dtype=np.int32))
                np.testing.assert_array_equal(self.ctrl.counts, before)
                self.assertTrue(self.ctrl.live[1])
                self.assert_mapping({0, 1})
        self.synchronize.assert_not_called()
        self.ctrl.release(np.array([1], dtype=np.int32))
        with self.assertRaisesRegex(RuntimeError, "non-live"):
            self.ctrl.release(np.array([1], dtype=np.int32))
        self.assert_mapping({0})


if __name__ == "__main__":
    unittest.main()
