import unittest
from collections import Counter
from itertools import combinations
from random import Random
from types import SimpleNamespace as B
from unittest.mock import Mock, call, patch

from adaptive_sparse.scheduler_budget import (
    PhysicalDemandIndex, SchedulerBudget, _bounded_demotion_groups,
    _chunk_release_counts, _demand_index_demotion_groups,
    _indexed_demotion_groups, _rank_demotions,
    _releasable, _releasable_counts, _releasable_group, _releasable_pair,
    _should_plan_future_waiting,
)
from adaptive_sparse.future_prefill import empty_plan
from adaptive_sparse.runtime_value import RuntimeRequestValue, merge_demotion_values


class SchedulerBudgetTest(unittest.TestCase):
    def test_future_reservation_telemetry_is_latest_snapshot(self):
        policy = SchedulerBudget(100, 0, 0, 0)
        policy.observe_future_prefill(empty_plan("dry_run"), 0)
        self.assertEqual(policy.stats["future_prefill_ticks"], 1)
        self.assertEqual(policy.stats["future_plan_ticks"], 1)
        self.assertEqual(policy.stats["last_future_plan_boundary"], "prefill")
        self.assertEqual(policy.stats["future_prefill_selected"], 0)

    def test_decode_future_plan_telemetry_is_distinct_from_prefill(self):
        policy = SchedulerBudget(100, 0, 0, 0)
        policy.observe_future_prefill(
            empty_plan("dry_run"), 0, boundary="decode"
        )
        self.assertEqual(policy.stats["future_plan_ticks"], 1)
        self.assertEqual(policy.stats["future_prefill_ticks"], 0)
        self.assertEqual(policy.stats["last_future_plan_boundary"], "decode")

    def test_future_waiting_cadence_is_immediate_for_prefill_and_periodic_for_decode(self):
        self.assertTrue(_should_plan_future_waiting(
            "enforce", [object()], prefill=True, steps=3, observe=16
        ))
        self.assertFalse(_should_plan_future_waiting(
            "enforce", [object()], prefill=False, steps=14, observe=16
        ))
        self.assertTrue(_should_plan_future_waiting(
            "enforce", [object()], prefill=False, steps=15, observe=16
        ))
        self.assertFalse(_should_plan_future_waiting(
            "enforce", [], prefill=False, steps=15, observe=16
        ))
        self.assertFalse(_should_plan_future_waiting(
            "off", [object()], prefill=True, steps=15, observe=16
        ))

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

    def test_future_headroom_can_trigger_pressure_during_decode(self):
        policy = SchedulerBudget(100, 10, 0, 0, observe=16)
        backing = B(backed_bytes=85)
        tracker = B(
            backing=backing,
            refs={},
            chunk_bytes=85,
            trim_quiescent=lambda: 0,
        )
        ctrl = B(
            _ekv_resident=set(),
            wait_for_pending_backup=lambda: None,
            padded_buffer_size=4,
            mem_pool_device=B(adaptive_vmm=tracker, page_size=1),
        )
        batch = B(reqs=[], is_empty=lambda: False)
        policy.boundary(
            batch, ctrl, lambda value: (value, value), lambda: None,
            prefill=False, now=0, future_headroom=20,
        )
        self.assertEqual(policy.stats["pressure_observations"], 1)
        self.assertEqual(policy.stats["future_decode_pressure_observations"], 1)
        self.assertEqual(policy.stats["future_prefill_pressure_observations"], 0)

    def test_global_broker_headroom_can_trigger_prefill_pressure(self):
        policy = SchedulerBudget(
            0, 0, 0, 0, broker_headroom_bytes=10, broker_low_bytes=20
        )
        backing = B(
            backed_bytes=40,
            broker_snapshot={
                "capacity_bytes": 100,
                "outstanding_bytes": 91,
                "available_bytes": 9,
            },
        )
        tracker = B(
            backing=backing, refs={}, chunk_bytes=40,
            trim_quiescent=lambda: 0,
        )
        ctrl = B(
            _ekv_resident=set(),
            wait_for_pending_backup=lambda: None,
            padded_buffer_size=0,
            mem_pool_device=B(adaptive_vmm=tracker, page_size=1),
        )
        policy.boundary(
            B(reqs=[]), ctrl, lambda value: (value, value), lambda: None,
            prefill=True, now=0,
        )
        self.assertEqual(policy.stats["pressure_observations"], 1)
        self.assertEqual(policy.stats["global_broker_pressure_observations"], 1)
        self.assertEqual(policy.stats["global_broker_min_available_bytes"], 9)
        self.assertEqual(policy.stats["global_broker_max_outstanding_bytes"], 91)

    def test_global_broker_recovery_requires_low_watermark_stability(self):
        policy = SchedulerBudget(
            0, 0, 0, 0, stable=2,
            broker_headroom_bytes=10, broker_low_bytes=20,
        )
        policy.promote = Mock()
        backing = B(backed_bytes=40)
        tracker = B(backing=backing, refs={}, chunk_bytes=40)
        ctrl = B(
            _ekv_resident=set(),
            mem_pool_device=B(adaptive_vmm=tracker),
        )
        for now, available in ((0, 20), (1, 19), (2, 20), (3, 25), (4, 25)):
            backing.broker_snapshot = {
                "capacity_bytes": 100,
                "outstanding_bytes": 100 - available,
                "available_bytes": available,
            }
            policy.boundary(
                B(reqs=[]), ctrl, lambda value: (value, value), lambda: None,
                prefill=True, now=now,
            )
            self.assertEqual(policy.promote.call_count, int(now == 4))

    def test_fair_share_pressure_triggers_before_global_capacity_exhaustion(self):
        policy = SchedulerBudget(
            0, 0, 0, 0, broker_headroom_bytes=10, broker_low_bytes=20
        )
        backing = B(
            backed_bytes=40,
            broker_snapshot={
                "capacity_bytes": 100,
                "outstanding_bytes": 70,
                "available_bytes": 30,
                "fairness_pressure": True,
            },
        )
        tracker = B(
            backing=backing, refs={}, chunk_bytes=40,
            trim_quiescent=lambda: 0,
        )
        ctrl = B(
            _ekv_resident=set(),
            wait_for_pending_backup=lambda: None,
            padded_buffer_size=0,
            mem_pool_device=B(adaptive_vmm=tracker, page_size=1),
        )
        policy.boundary(
            B(reqs=[]), ctrl, lambda value: (value, value), lambda: None,
            prefill=True, now=0,
        )
        self.assertEqual(policy.stats["pressure_observations"], 1)
        self.assertEqual(
            policy.stats["global_broker_fair_pressure_observations"], 2
        )

    def test_global_broker_pressure_requires_snapshot_on_every_rank(self):
        policy = SchedulerBudget(
            0, 0, 0, 0, broker_headroom_bytes=10, broker_low_bytes=20
        )
        tracker = B(
            backing=B(
                backed_bytes=40,
                broker_snapshot={
                    "capacity_bytes": 100,
                    "outstanding_bytes": 80,
                    "available_bytes": 20,
                },
            ),
            refs={},
            chunk_bytes=40,
        )
        ctrl = B(
            _ekv_resident=set(),
            mem_pool_device=B(adaptive_vmm=tracker),
        )

        def asymmetric_exchange(value):
            return value, (*value[:4], None, None)

        with self.assertRaisesRegex(RuntimeError, "every TP rank"):
            policy.boundary(
                B(reqs=[]), ctrl, asymmetric_exchange, lambda: None,
                prefill=True, now=0,
            )

    def test_base_prefill_headroom_does_not_inflate_decode_pressure(self):
        policy = SchedulerBudget(100, 10, 0, 0, observe=1)
        backing = B(backed_bytes=95)
        tracker = B(backing=backing, refs={}, chunk_bytes=95, trim_quiescent=lambda: 0)
        ctrl = B(
            _ekv_resident=set(),
            wait_for_pending_backup=lambda: None,
            mem_pool_device=B(adaptive_vmm=tracker, page_size=1),
        )
        policy.boundary(
            B(reqs=[], is_empty=lambda: False), ctrl,
            lambda value: (value, value), lambda: None,
            prefill=False, now=0,
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
        self.assertEqual(_rank_demotions(()), [])

    def test_common_rank_uses_runtime_value_and_deadline_guard(self):
        ranks = [{"a": (100, 10), "b": (100, 10)}]
        values = [{
            "a": merge_demotion_values([
                RuntimeRequestValue("a", weight=64, remaining_output_tokens=1024)
            ]),
            "b": merge_demotion_values([
                RuntimeRequestValue("b", deadline_remaining_ms=50)
            ]),
        }]
        self.assertEqual(_rank_demotions(ranks, values, deadline_guard_ms=500), ["a", "b"])

    def test_releasable_group_requires_every_shared_chunk_owner(self):
        tracker = B(
            backing=B(mapped_chunks={9}),
            refs={9: 3},
            chunk_bytes=10,
        )
        candidates = [
            ({1}, Counter({9: 1})),
            ({2}, Counter({9: 1})),
            ({3}, Counter({9: 1})),
        ]
        self.assertEqual(_releasable_group(tracker, candidates[:2]), set())
        self.assertEqual(_releasable_group(tracker, candidates), {9})

    def test_bounded_group_search_finds_three_request_release(self):
        tracker = B(
            backing=B(mapped_chunks={9}),
            refs={9: 3},
            chunk_bytes=10,
        )
        requests = {
            "a": (None, {1}, Counter({9: 1}), set()),
            "b": (None, {2}, Counter({9: 1}), set()),
            "c": (None, {3}, Counter({9: 1}), set()),
        }
        costs = {"a": 4, "b": 5, "c": 6}
        groups, chunks, examined, truncated, rejected = _bounded_demotion_groups(
            tracker, requests, costs.__getitem__, max_group=3,
            byte_cap=20, search_limit=8,
        )
        self.assertEqual(groups[("a", "b", "c")], (10, 15))
        self.assertEqual(chunks[("a", "b", "c")], {9})
        self.assertFalse(truncated)
        self.assertEqual(examined, 4)
        self.assertEqual(rejected, 0)

        groups, _, examined, truncated, rejected = _bounded_demotion_groups(
            tracker, requests, costs.__getitem__, max_group=3,
            byte_cap=20, search_limit=3, rotation=0,
        )
        self.assertEqual(groups[("a", "b", "c")], (10, 15))
        self.assertTrue(truncated)
        self.assertEqual(examined, 3)
        self.assertEqual(rejected, 0)

        groups, _, examined, truncated, rejected = _bounded_demotion_groups(
            tracker, requests, costs.__getitem__, max_group=3,
            byte_cap=12, search_limit=8,
        )
        self.assertFalse(groups)
        self.assertFalse(truncated)
        self.assertEqual(examined, 4)
        self.assertEqual(rejected, 1)

    def test_physical_demand_index_updates_changed_requests_incrementally(self):
        index = PhysicalDemandIndex()
        requests = {
            "a": (None, {1}, Counter({9: 1}), set()),
            "b": (None, {2}, Counter({9: 1}), set()),
        }
        self.assertEqual(index.sync(requests), 2)
        self.assertEqual(index.page_owners, {1: {"a"}, 2: {"b"}})
        self.assertEqual(index.chunk_owners, {9: {"a": 1, "b": 1}})
        self.assertFalse(index.has_page_overlap)
        self.assertEqual(index.sync(requests), 0)

        requests["b"] = (None, {1}, Counter({9: 1}), set())
        self.assertEqual(index.sync(requests), 1)
        self.assertTrue(index.has_page_overlap)
        self.assertEqual(index.page_owners[1], {"a", "b"})

        self.assertEqual(index.sync({}), 2)
        self.assertEqual(index.records, {})
        self.assertEqual(index.chunk_owners, {})
        self.assertEqual(index.page_owners, {})

    def test_indexed_group_search_follows_shared_chunk_owners(self):
        tracker = B(
            backing=B(mapped_chunks={9}),
            refs={9: 3},
            chunk_bytes=10,
        )
        requests = {
            "a": (None, {1}, Counter({9: 1}), set()),
            "b": (None, {2}, Counter({9: 1}), set()),
            "c": (None, {3}, Counter({9: 1}), set()),
        }
        costs = {"a": 4, "b": 5, "c": 6}
        index = PhysicalDemandIndex()
        self.assertEqual(index.sync(requests), 3)
        groups, chunks, examined, truncated, rejected = _indexed_demotion_groups(
            index, tracker, requests, costs.__getitem__, max_group=3,
            byte_cap=20, search_limit=8,
        )
        self.assertEqual(groups[("a", "b", "c")], (10, 15))
        self.assertEqual(chunks[("a", "b", "c")], {9})
        self.assertEqual(examined, 1)
        self.assertFalse(truncated)
        self.assertEqual(rejected, 0)

    def test_indexed_search_uses_exact_fallback_for_overlapping_pages(self):
        tracker = B(
            backing=B(mapped_chunks={9}),
            refs={9: 1},
            chunk_bytes=10,
            chunks=lambda _: {9},
        )
        requests = {
            "a": (None, {1}, Counter({9: 1}), set()),
            "b": (None, {1}, Counter({9: 1}), set()),
        }
        index = PhysicalDemandIndex()
        index.sync(requests)
        (groups, chunks, indexed_examined, fallback_examined, index_truncated,
         fallback_truncated, rejected, used_fallback) = (
            _demand_index_demotion_groups(
                index,
                tracker,
                requests,
                lambda _: 1,
                max_group=2,
                byte_cap=10,
                search_limit=4,
            )
        )
        self.assertEqual(groups[("a", "b")], (10, 2))
        self.assertEqual(chunks[("a", "b")], {9})
        self.assertEqual(indexed_examined, 0)
        self.assertEqual(fallback_examined, 1)
        self.assertFalse(index_truncated)
        self.assertFalse(fallback_truncated)
        self.assertEqual(rejected, 0)
        self.assertTrue(used_fallback)

    def test_indexed_search_does_not_miss_disjoint_page_groups(self):
        random = Random(911)
        for trial in range(25):
            request_ids = [f"r{index}" for index in range(8)]
            per_request = {request_id: Counter() for request_id in request_ids}
            refs = Counter()
            for chunk in range(12):
                owners = random.sample(request_ids, random.randint(2, 4))
                for request_id in owners:
                    per_request[request_id][chunk] += 1
                    refs[chunk] += 1
            requests = {
                request_id: (
                    None,
                    {int(request_id[1:]) + 1},
                    per_request[request_id],
                    set(),
                )
                for request_id in request_ids
            }
            tracker = B(
                backing=B(mapped_chunks=set(refs)),
                refs=refs,
                chunk_bytes=1,
            )
            exhaustive, _, _, _, _ = _bounded_demotion_groups(
                tracker,
                requests,
                lambda _: 1,
                max_group=4,
                byte_cap=1 << 30,
                search_limit=1000,
            )
            index = PhysicalDemandIndex()
            index.sync(requests)
            (indexed, _, _, fallback_examined, _, _, _, used_fallback) = (
                _demand_index_demotion_groups(
                    index,
                    tracker,
                    requests,
                    lambda _: 1,
                    max_group=4,
                    byte_cap=1 << 30,
                    search_limit=256,
                )
            )
            with self.subTest(trial=trial):
                self.assertTrue(exhaustive)
                self.assertTrue(indexed)
                self.assertTrue(set(indexed).issubset(exhaustive))
                self.assertEqual(fallback_examined, 0)
                self.assertFalse(used_fallback)

    def test_group_search_rotation_reaches_later_shared_pair(self):
        tracker = B(
            backing=B(mapped_chunks={9}),
            refs={9: 2},
            chunk_bytes=10,
        )
        requests = {
            "a": (None, {1}, Counter(), set()),
            "b": (None, {2}, Counter(), set()),
            "c": (None, {3}, Counter({9: 1}), set()),
            "d": (None, {4}, Counter({9: 1}), set()),
        }
        cost = lambda _: 1
        first, _, _, truncated, _ = _bounded_demotion_groups(
            tracker, requests, cost, max_group=2, byte_cap=10,
            search_limit=1, rotation=0,
        )
        self.assertFalse(first)
        self.assertTrue(truncated)
        later, chunks, _, _, _ = _bounded_demotion_groups(
            tracker, requests, cost, max_group=2, byte_cap=10,
            search_limit=1, rotation=2,
        )
        self.assertEqual(later[("c", "d")], (10, 2))
        self.assertEqual(chunks[("c", "d")], {9})

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

    def test_migration_time_cap_filters_request_and_force_bypasses_it(self):
        policy = SchedulerBudget(
            100, 0, 0, 0, byte_cap=100, migration_time_cap_ms=1_000,
            initial_migration_bandwidth_bytes_per_s=10,
        )
        request = B(rid="a", kv=B(req_pool_idx=1, kv_committed_len=11))
        batch = B(reqs=[request])
        pool = B(layer_num=1, bytes_per_token_k=1, bytes_per_token_v=0)
        policy.observe_residency(batch, {1}, 0)
        self.assertEqual(policy.migration_byte_cap("demotion"), 10)
        self.assertFalse(policy.eligible(request, pool, 1, False))
        self.assertEqual(policy.stats["rejected_cap"], 1)
        self.assertEqual(policy.stats["time_cap_rejections"], 1)
        self.assertEqual(policy.stats["time_cap_rejections_demotion"], 1)
        self.assertEqual(policy.stats["time_cap_rejections_promotion"], 0)
        self.assertTrue(policy.eligible(request, pool, 1, True))

    def test_fixed_byte_cap_and_time_cap_rejections_are_distinct(self):
        policy = SchedulerBudget(
            100, 0, 0, 0, byte_cap=10, migration_time_cap_ms=1_000,
            initial_migration_bandwidth_bytes_per_s=5,
        )
        request = B(rid="a", kv=B(req_pool_idx=1, kv_committed_len=11))
        batch = B(reqs=[request])
        pool = B(layer_num=1, bytes_per_token_k=1, bytes_per_token_v=0)
        policy.observe_residency(batch, {1}, 0)
        self.assertFalse(policy.eligible(request, pool, 1, False))
        self.assertEqual(policy.stats["rejected_cap"], 1)
        self.assertEqual(policy.stats["time_cap_rejections"], 0)

    def test_group_cap_observer_distinguishes_time_rejections(self):
        tracker = B(
            backing=B(mapped_chunks={9}),
            refs={9: 2},
            chunk_bytes=10,
        )
        requests = {
            "a": (None, {1}, Counter({9: 1}), set()),
            "b": (None, {2}, Counter({9: 1}), set()),
        }
        capacities = []
        groups, _, _, _, rejected = _bounded_demotion_groups(
            tracker, requests, lambda _: 9, max_group=2, byte_cap=15,
            search_limit=4, cap_observer=capacities.append,
        )
        self.assertFalse(groups)
        self.assertEqual(rejected, 1)
        self.assertEqual(capacities, [18])

    def test_record_updates_directional_bandwidth_and_time_cap_telemetry(self):
        policy = SchedulerBudget(
            100, 0, 0, 0, byte_cap=100,
            migration_time_cap_ms=1_000,
            initial_migration_bandwidth_bytes_per_s=10,
        )
        with patch("adaptive_sparse.scheduler_budget.time.monotonic", return_value=10.5):
            policy.record("demotion", "prefill_pressure", 10, 20, 10)
        self.assertEqual(policy.stats["demotion" + "s"], 1)
        self.assertEqual(policy.stats["migration_seconds"], 0.5)
        self.assertEqual(
            policy.stats["migration_bandwidth_bytes_per_s"]["demotion"], 40
        )
        self.assertEqual(
            policy.stats["migration_bandwidth_bytes_per_s"]["promotion"], 10
        )
        self.assertEqual(
            policy.stats["migration_bandwidth_samples"],
            {"demotion": 1, "promotion": 0},
        )
        self.assertEqual(
            policy.stats["migration_effective_byte_cap"],
            {"demotion": 40, "promotion": 10},
        )

    def test_different_rank_time_caps_leave_only_common_candidates(self):
        request = B(rid="a", kv=B(req_pool_idx=1, kv_committed_len=20))
        batch = B(reqs=[request])
        pool = B(layer_num=1, bytes_per_token_k=1, bytes_per_token_v=0)
        rank_caps = (50, 10)
        rank_metrics = []
        for cap in rank_caps:
            policy = SchedulerBudget(
                100, 0, 0, 0, byte_cap=100,
                migration_time_cap_ms=1_000,
                initial_migration_bandwidth_bytes_per_s=cap,
            )
            policy.observe_residency(batch, {1}, 0)
            metrics = {}
            if policy.eligible(request, pool, 1, False):
                metrics["a"] = (10, 20)
            rank_metrics.append(metrics)
        self.assertEqual(rank_metrics, [{"a": (10, 20)}, {}])
        self.assertEqual(_rank_demotions(rank_metrics), [])

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
