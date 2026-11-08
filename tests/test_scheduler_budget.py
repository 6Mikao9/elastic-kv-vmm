from collections import Counter
from itertools import combinations
from random import Random
from types import SimpleNamespace as B
import unittest
from unittest.mock import Mock, call, patch

from adaptive_sparse.scheduler_budget import (
    SchedulerBudget, _chunk_release_counts, _rank_demotions, _releasable,
    _releasable_counts, _releasable_pair,
)
from adaptive_sparse.future_prefill import empty_plan


class SchedulerBudgetTest(unittest.TestCase):
    def test_future_reservation_telemetry_is_latest_snapshot(self):
        policy = SchedulerBudget(100, 0, 0, 0)
        policy.observe_future_prefill(empty_plan("dry_run"), 0)
        self.assertEqual(policy.stats["future_prefill_ticks"], 1)
        self.assertEqual(policy.stats["future_prefill_selected"], 0)

    def test_future_headroom_can_trigger_prefill_pressure(self):
        policy = SchedulerBudget(100, 10, 0, 0)
        backing = B(backed_bytes=85)
        tracker = B(backing=backing, refs={1: 1}, chunk_bytes=85)
        ctrl = B(_ekv_resident=set(), padded_buffer_size=4,
                 mem_pool_device=B(adaptive_vmm=tracker, page_size=1))
        policy.boundary(
            B(reqs=[]), ctrl, lambda value: (value, value), lambda: None,
            prefill=True, now=0, future_headroom=20,
        )
        self.assertEqual(policy.stats["pressure_observations"], 1)
        self.assertEqual(policy.stats["future_prefill_pressure_observations"], 1)

    def test_future_headroom_uses_maximum_of_base_and_reservation(self):
        policy = SchedulerBudget(100, 10, 0, 0)
        backing = B(backed_bytes=89)
        tracker = B(backing=backing, refs={1: 1}, chunk_bytes=89)
        ctrl = B(_ekv_resident=set(),
                 mem_pool_device=B(adaptive_vmm=tracker))
        policy.boundary(
            B(reqs=[]), ctrl, lambda value: (value, value), lambda: None,
            prefill=True, now=0, future_headroom=11,
        )
        self.assertEqual(policy.stats["pressure_observations"], 0)

    def test_cached_release_counts_match_page_union(self):
        random = Random(137)
        for trial in range(100):
            layout = [set(random.sample(range(9), random.randint(1, 3))) for _ in range(16)]
            refs = Counter(chunk for chunks in layout for chunk in chunks)
            mapped = set(random.sample(range(9), 8))
            tracker = B(chunks=layout.__getitem__, refs=refs, backing=B(mapped_chunks=mapped))
            candidates = [set(random.sample(range(1, 16), 7)) for _ in range(5)]
            if trial % 2:
                # Disjoint page ownership, while chunks can still be shared.
                candidates = [set(range(start, start + 3)) for start in range(1, 16, 3)]

            def reference(released):
                return {
                    chunk for chunk in mapped
                    if any(chunk in layout[page] for page in released)
                    and all(page in released or chunk not in chunks
                            for page, chunks in enumerate(layout))
                }

            counts = [_chunk_release_counts(tracker, pages) for pages in candidates]
            for pages, count in zip(candidates, counts):
                self.assertEqual(_releasable_counts(tracker, count), reference(pages))
            for left, right in combinations(range(len(candidates)), 2):
                self.assertEqual(
                    _releasable_pair(tracker, candidates[left], counts[left],
                                     candidates[right], counts[right]),
                    reference(candidates[left] | candidates[right]),
                )

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
        release_broker = Mock()
        ctrl = B(_ekv_resident={1, 2}, _ekv_broker_release_request=release_broker)
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
        self.assertEqual(
            release_broker.call_args_list, [call(1), call(2)]
        )
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

    def test_broker_rejection_filters_promotion_before_prepare(self):
        policy = SchedulerBudget(0, 0, 0, 0)
        request = B(
            rid="a",
            kv=B(req_pool_idx=1, kv_committed_len=8, kv_allocated_len=8),
            finished=lambda: False,
            hisparse_staging=False,
        )
        policy.residency_since["a"] = (False, 0)
        backing = B(backed_bytes=0, mapped_chunks=set())
        tracker = B(backing=backing, refs={}, chunk_bytes=1)
        physical = B(free_pages=[], available_size=lambda: 100)
        gate = Mock(return_value=False)
        ctrl = B(
            _ekv_resident=set(),
            mem_pool_device=B(
                adaptive_vmm=tracker, page_size=2, layer_num=1,
                bytes_per_token_k=1, bytes_per_token_v=1,
            ),
            token_to_kv_pool_allocator=B(hisparse_attn_allocator=physical),
            req_device_buffer_size={1: 4},
            padded_buffer_size=4,
            device_buffer_size=4,
            _ekv_broker_can_promote=gate,
        )
        with patch("adaptive_sparse.scheduler_budget.prepare_promote") as prepare:
            policy.promote(
                B(reqs=[request]), ctrl, lambda value: (value, value),
                lambda: None, False, 0,
            )
        gate.assert_called_once_with(request, 100)
        prepare.assert_not_called()
        self.assertEqual(policy.stats["broker_rejected_promotions"], 1)

    def test_broker_promotion_captures_growth_before_native_cleanup(self):
        policy = SchedulerBudget(0, 0, 0, 0)
        request = B(
            rid="a",
            kv=B(req_pool_idx=1, kv_committed_len=8, kv_allocated_len=8),
            finished=lambda: False,
            hisparse_staging=False,
        )
        policy.residency_since["a"] = (False, 0)
        backing = B(backed_bytes=0, mapped_chunks=set())
        tracker = B(
            backing=backing,
            refs={},
            chunk_bytes=1,
            chunks=lambda page: {page},
        )
        class PageList(list):
            def __getitem__(self, key):
                value = super().__getitem__(key)
                return PageList(value) if isinstance(key, slice) else value

            def tolist(self):
                return list(self)

        physical = B(free_pages=PageList([0, 1, 2, 3]), available_size=lambda: 100)
        plan = Mock(return_value=(8, 4, 7, 1))
        commit_broker = Mock()
        ctrl = B(
            _ekv_resident=set(),
            mem_pool_device=B(
                adaptive_vmm=tracker, page_size=2, layer_num=1,
                bytes_per_token_k=1, bytes_per_token_v=1,
            ),
            token_to_kv_pool_allocator=B(hisparse_attn_allocator=physical),
            req_device_buffer_size={1: 4},
            padded_buffer_size=4,
            device_buffer_size=4,
            _ekv_broker_can_promote=Mock(return_value=True),
            _ekv_broker_promotion_plan=plan,
            _ekv_broker_commit_promotion=commit_broker,
        )

        def commit(prepared, publish, synchronize):
            publish(1)
            return 0

        with patch(
            "adaptive_sparse.scheduler_budget.prepare_promote", return_value="prepared"
        ), patch("adaptive_sparse.scheduler_budget.commit_promote", side_effect=commit):
            policy.promote(
                B(reqs=[request]), ctrl, lambda value: (value, value),
                lambda: None, False, 0,
            )

        plan.assert_called_once_with(request)
        commit_broker.assert_called_once_with(request, 7)
        self.assertIn(1, ctrl._ekv_resident)
