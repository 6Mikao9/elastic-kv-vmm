from types import SimpleNamespace as B
import unittest
from unittest.mock import Mock, patch

from adaptive_sparse.scheduler_budget import SchedulerBudget, _rank_demotions, _releasable


class SchedulerBudgetTest(unittest.TestCase):
    def test_common_rank_uses_worst_rank_benefit_and_cost(self):
        ranks = [{"a": (100, 10), "b": (50, 10), "only": (999, 1)},
                 {"a": (20, 20), "b": (50, 10)}]
        self.assertEqual(_rank_demotions(ranks), ["b", "a"])
        self.assertEqual(_rank_demotions(ranks[::-1]), ["b", "a"])

    def test_state_change_restarts_dwell_and_force_bypasses_cap(self):
        policy = SchedulerBudget(100, 0, 3, 0, byte_cap=10)
        request = B(rid="a", kv=B(req_pool_idx=1, kv_committed_len=5))
        batch = B(reqs=[request])
        pool = B(layer_num=1, bytes_per_token_k=1, bytes_per_token_v=1)
        policy.observe_residency(batch, {1}, 0)
        self.assertFalse(policy.eligible(request, pool, 2, False))
        self.assertTrue(policy.eligible(request, pool, 3, False))
        policy.observe_residency(batch, set(), 3)
        self.assertFalse(policy.eligible(request, pool, 5, False))
        self.assertTrue(policy.eligible(request, pool, 6, False))
        request.kv.kv_committed_len += 1
        self.assertFalse(policy.eligible(request, pool, 7, False))
        self.assertTrue(policy.eligible(request, pool, 7, True))
        self.assertEqual(policy.stats["rejected_dwell"], 2)
        self.assertEqual(policy.stats["rejected_cap"], 1)

    def test_low_water_requires_stability_after_excursion(self):
        policy = SchedulerBudget(100, 0, 0, 0, low=60, stable=2)
        policy.promote = Mock()
        backing = B(backed_bytes=50)
        ctrl = B(_ekv_resident=set(), mem_pool_device=B(adaptive_vmm=B(backing=backing, refs={}, chunk_bytes=1)))
        for now, used in [(0, 50), (1, 60), (2, 50), (3, 50), (4, 50)]:
            backing.backed_bytes = used
            policy.boundary(B(reqs=[]), ctrl, lambda value: (value, value),
                            lambda: None, prefill=True, now=now)
            self.assertEqual(policy.promote.call_count, int(now == 4))

    def test_pressure_trims_slack_without_migration(self):
        policy = SchedulerBudget(100, 0, 0, 0)
        backing = B(backed_bytes=120)
        tracker = B(backing=backing, refs={1: 1}, chunk_bytes=80)
        def trim():
            backing.backed_bytes = 80
            return 40
        tracker.trim_quiescent = trim
        wait = Mock()
        ctrl = B(_ekv_resident=set(), wait_for_pending_backup=wait,
                 mem_pool_device=B(adaptive_vmm=tracker))
        policy.boundary(B(reqs=[]), ctrl, lambda value: (value, value),
                        lambda: None, prefill=True, now=0)
        wait.assert_called_once_with()
        self.assertEqual(policy.stats["pressure_observations"], 1)
        self.assertEqual(policy.stats["slack_reclaim_bytes"], 40)
        self.assertEqual(policy.stats["slack_reclaim_actions"], 1)
        self.assertEqual(policy.stats["min_live_bytes"], 80)
        self.assertEqual(policy.stats["max_reclaimable_bytes"], 40)
        self.assertEqual(policy.stats["demotions"], 0)
    def test_shared_pages_reclaim_only_as_union_and_publish_together(self):
        policy = SchedulerBudget(100, 0, 0, 0, max_group=2)
        ctrl = B(_ekv_resident={1, 2})
        backing = B(backed_bytes=10, mapped_chunks={9})
        tracker = B(backing=backing, refs={9: 2}, chunks=lambda _: {9}, chunk_bytes=10)
        self.assertEqual(_releasable(tracker, [1]), set())
        self.assertEqual(_releasable(tracker, [2]), set())
        self.assertEqual(_releasable(tracker, [1, 2]), {9})

        def commit(row, publish, synchronize):
            self.assertEqual(ctrl._ekv_resident, {1, 2})
            tracker.refs[9] -= 1
            if not tracker.refs[9]:
                del tracker.refs[9]
            publish(row)

        def reclaim(chunks, *, synchronize):
            self.assertEqual(chunks, {9})
            self.assertEqual(ctrl._ekv_resident, {1, 2})
            self.assertFalse(tracker.refs)
            backing.backed_bytes = 0
            return 1

        backing.reclaim = reclaim
        with patch("adaptive_sparse.scheduler_budget.prepare_demote", side_effect=[1, 2]), patch(
            "adaptive_sparse.scheduler_budget.commit_demote", side_effect=commit
        ):
            policy.demote_group(("a", "b"), {"a": (1,), "b": (2,)}, {9},
                                ctrl, tracker, lambda value: (value, value),
                                lambda: None, False, True, 20)
        self.assertEqual(ctrl._ekv_resident, set())
        self.assertEqual(policy.stats["demotions"], 2)
        self.assertEqual(policy.stats["group_actions"], 1)
        self.assertEqual(policy.stats["group_sizes"], {"2": 1})
        self.assertEqual(policy.stats["reasons"], {"prefill_pressure_group": 1})
        self.assertEqual(policy.stats["released_bytes"], 10)
    def test_low_water_slack_trim_starts_stability_after_actual_release(self):
        policy = SchedulerBudget(100, 0, 0, 0, low=60, stable=2, trim_low_water=True)
        policy.promote = Mock()
        backing = B(backed_bytes=60)
        tracker = B(backing=backing, refs={1: 1}, chunk_bytes=40)
        def trim():
            backing.backed_bytes = 40
            return 20
        tracker.trim_quiescent = trim
        ctrl = B(_ekv_resident=set(), wait_for_pending_backup=Mock(),
                 mem_pool_device=B(adaptive_vmm=tracker), device_buffer_size=4)
        req = B(rid="a", kv=B(req_pool_idx=1, kv_committed_len=8, kv_allocated_len=8),
                finished=lambda: False, hisparse_staging=False)
        for now in range(3):
            policy.boundary(B(reqs=[req]), ctrl, lambda value: (value, value),
                            lambda: None, prefill=True, now=now)
            self.assertEqual(policy.promote.call_count, int(now == 2))
        self.assertEqual(policy.stats["low_water_slack_actions"], 1)
        self.assertEqual(policy.stats["low_water_slack_bytes"], 20)
        self.assertEqual(policy.stats["demotions"], 0)