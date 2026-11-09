import unittest
from types import SimpleNamespace

from elastic_kv.request_affine import (
    RequestAffinePlanner,
    RequestDemand,
    extract_request_demands,
    observe_packed_allocation,
)
from elastic_kv.slots import FreeSlots


class RequestAffinePlannerTest(unittest.TestCase):
    def setUp(self):
        self.pool_size = 256
        self.chunk_slots = 64
        self.live = {0}
        self.planner = RequestAffinePlanner(self.pool_size, self.chunk_slots)
        self.planner.prime(lambda slot: slot in self.live)
        self.free_ranges = [(1, self.pool_size + 1)]

    def plan(self, *demands):
        return self.planner.plan(
            demands,
            free_ranges=self.free_ranges,
            is_live=lambda slot: slot in self.live,
        )

    def commit(self, plan):
        self.planner.commit(plan)
        self.live.update(slot for request in plan.requests for slot in request.slot_ids)

    def test_boundary_requests_switch_to_private_at_one_chunk(self):
        plan = self.plan(
            RequestDemand("short-63", 1, 63),
            RequestDemand("at-chunk", 1, 64),
            RequestDemand("over-chunk", 1, 65),
        )
        self.assertIsNotNone(plan)
        by_id = {request.request_id: request for request in plan.requests}
        self.assertFalse(by_id["short-63"].private)
        self.assertTrue(by_id["at-chunk"].private)
        self.assertTrue(by_id["over-chunk"].private)
        self.assertEqual(plan.new_reserved_chunks, (1, 2))
        self.assertEqual(by_id["short-63"].slot_ids, (1,))
        self.assertEqual(by_id["at-chunk"].slot_ids, (64,))
        self.assertEqual(by_id["over-chunk"].slot_ids, (128,))

    def test_private_request_growth_reuses_only_its_own_tail(self):
        first = self.plan(RequestDemand("long", 32, 100))
        self.commit(first)
        self.free_ranges = [(1, 64), (96, 257)]

        second = self.plan(RequestDemand("long", 20, 100))
        by_id = {request.request_id: request for request in second.requests}
        self.assertEqual(by_id["long"].slot_ids, tuple(range(96, 116)))
        self.assertEqual(second.new_reserved_chunks, ())

    def test_new_long_requests_never_share_a_chunk(self):
        plan = self.plan(
            RequestDemand("a", 8, 80),
            RequestDemand("b", 8, 80),
        )
        self.assertEqual(plan.new_reserved_chunks, (1, 2))
        by_id = {request.request_id: request for request in plan.requests}
        self.assertEqual(self.planner.chunk_for_slot(by_id["a"].slot_ids[0]), 1)
        self.assertEqual(self.planner.chunk_for_slot(by_id["b"].slot_ids[0]), 2)

    def test_release_returns_only_an_empty_private_chunk(self):
        plan = self.plan(
            RequestDemand("a", 16, 80),
            RequestDemand("b", 16, 80),
        )
        self.commit(plan)
        by_id = {request.request_id: request for request in plan.requests}
        released_a = by_id["a"].slot_ids
        self.live.difference_update(released_a)
        ranges = self.planner.release_slots(
            released_a, is_live=lambda slot: slot in self.live
        )
        self.assertEqual(ranges, ((64, 128),))
        self.assertIn(1, self.planner.free_chunks)
        self.assertNotIn(2, self.planner.free_chunks)

    def test_partial_private_release_stays_withheld_until_chunk_is_empty(self):
        owner = self.plan(RequestDemand("owner", 2, 64))
        self.commit(owner)
        owner_slots = owner.requests[0].slot_ids
        self.live.remove(owner_slots[0])
        self.assertEqual(
            self.planner.release_slots(
                [owner_slots[0]], is_live=lambda slot: slot in self.live
            ),
            (),
        )
        self.assertTrue(self.planner.is_slot_withheld(owner_slots[0]))
        self.assertNotIn(1, self.planner.free_chunks)
        self.live.remove(owner_slots[1])
        self.assertEqual(
            self.planner.release_slots(
                [owner_slots[1]], is_live=lambda slot: slot in self.live
            ),
            ((64, 128),),
        )
        self.assertIn(1, self.planner.free_chunks)

    def test_incomplete_tail_chunk_is_never_released_as_private(self):
        planner = RequestAffinePlanner(130, 64)
        self.assertEqual(
            planner.release_slots([128], is_live=lambda slot: False),
            (),
        )

    def test_free_slot_count_returns_to_capacity_exactly_once(self):
        free_slots = FreeSlots(self.pool_size)
        plan = self.plan(RequestDemand("owner", 2, 64))
        free_slots.remove(plan.reserved_chunk_ranges)
        self.commit(plan)
        slots = plan.requests[0].slot_ids

        self.live.remove(slots[0])
        self.assertEqual(
            self.planner.release_slots(
                [slots[0]], is_live=lambda slot: slot in self.live
            ),
            (),
        )
        self.assertEqual(free_slots.count, self.pool_size - self.chunk_slots)

        self.live.remove(slots[1])
        released = self.planner.release_slots(
            [slots[1]], is_live=lambda slot: slot in self.live
        )
        for start, stop in released:
            free_slots.release(range(start, stop))
        self.assertEqual(free_slots.count, self.pool_size)
        self.assertEqual(free_slots.ranges, [(1, self.pool_size + 1)])

    def test_commit_validates_before_mutating_free_chunks(self):
        plan = self.plan(RequestDemand("long", 1, 64))
        self.commit(plan)
        before = self.planner.free_chunks
        with self.assertRaisesRegex(RuntimeError, "not free|already committed"):
            self.planner.commit(plan)
        self.assertEqual(self.planner.free_chunks, before)

    def test_capacity_shortfall_returns_none_without_state_change(self):
        planner = RequestAffinePlanner(128, 64)
        free = [(1, 129)]
        live = {0}
        planner.prime(lambda slot: slot in live)
        plan = planner.plan(
            [
                RequestDemand("a", 1, 80),
                RequestDemand("b", 1, 80),
            ],
            free_ranges=free,
            is_live=lambda slot: slot in live,
        )
        self.assertIsNone(plan)
        self.assertEqual(planner.private_request_count, 0)
        self.assertEqual(planner.free_chunks, frozenset({1}))

    def test_small_requests_remain_packed(self):
        plan = self.plan(
            RequestDemand("a", 10, 20),
            RequestDemand("b", 10, 20),
        )
        self.assertEqual(plan.private_requests, 0)
        self.assertEqual(plan.packed_requests, 2)
        self.assertEqual(plan.all_ranges, ((1, 11), (11, 21)))

    def test_duplicate_request_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self.plan(
                RequestDemand("same", 1, 1),
                RequestDemand("same", 1, 1),
            )


class PackedObservationTest(unittest.TestCase):
    def test_counts_cross_request_tail_sharing(self):
        observation = observe_packed_allocation(
            [
                RequestDemand("a", 40, 40),
                RequestDemand("b", 40, 40),
            ],
            [range(1, 41), range(41, 81)],
            chunk_slots=64,
        )
        self.assertEqual(observation.mapped_chunks, 2)
        self.assertEqual(observation.shared_chunks, 1)
        self.assertEqual(observation.maximum_owners, 2)


class RequestDemandExtractionTest(unittest.TestCase):
    def test_projects_generation_and_decode_allocation(self):
        reqs = [
            SimpleNamespace(
                rid="r",
                sampling_params=SimpleNamespace(max_new_tokens=90),
                output_ids=(),
            )
        ]
        demands = extract_request_demands(reqs, [20], [10], decode=True)
        self.assertEqual(demands[0], RequestDemand("r", 10, 110))

    def test_decode_does_not_add_generated_tokens_twice(self):
        req = SimpleNamespace(
            rid="r",
            sampling_params=SimpleNamespace(max_new_tokens=10),
            output_ids=[0] * 5,
        )
        demands = extract_request_demands([req], [55], [1], decode=True)
        self.assertEqual(demands[0], RequestDemand("r", 1, 60))

    def test_rejects_missing_request_id(self):
        self.assertIsNone(
            extract_request_demands([SimpleNamespace(rid=None)], [1], [1], decode=False)
        )


if __name__ == "__main__":
    unittest.main()
