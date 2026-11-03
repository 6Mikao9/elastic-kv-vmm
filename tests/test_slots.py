import random
import unittest

from elastic_kv.slots import FreeSlots


class FreeSlotsTest(unittest.TestCase):
    def test_fragmented_claim_and_bridge_merge(self):
        slots = FreeSlots(8)
        slots.claim(slots.peek(8))
        slots.release([1, 2, 4, 5, 7, 8])
        ranges = slots.peek(3)
        self.assertEqual(ranges, [(1, 3), (4, 5)])
        slots.claim(ranges)
        self.assertEqual(slots.ranges, [(5, 6), (7, 9)])
        slots.release([6])
        self.assertEqual(slots.ranges, [(5, 9)])
        self.assertEqual(slots.count, 4)

    def test_peek_bounds_leave_state_intact(self):
        slots = FreeSlots(2)
        self.assertEqual(slots.peek(0), [])
        self.assertIsNone(slots.peek(3))
        with self.assertRaises(ValueError):
            slots.peek(-1)
        self.assertEqual(slots.ranges, [(1, 3)])
        self.assertEqual(slots.count, 2)

    def test_randomized_conservation(self):
        rng = random.Random(504)
        capacity = 97
        slots = FreeSlots(capacity)
        free = set(range(1, capacity + 1))
        for _ in range(3000):
            live = sorted(set(range(1, capacity + 1)) - free)
            if live and rng.randrange(2):
                ids = sorted(rng.sample(live, rng.randint(1, min(9, len(live)))))
                slots.release(ids)
                free.update(ids)
            else:
                count = rng.randint(0, 10)
                before = slots.ranges.copy()
                ranges = slots.peek(count)
                self.assertEqual(slots.ranges, before)
                if count > len(free):
                    self.assertIsNone(ranges)
                else:
                    selected = [slot for start, stop in ranges for slot in range(start, stop)]
                    self.assertEqual(selected, sorted(free)[:count])
                    slots.claim(ranges)
                    free.difference_update(selected)
            self.assertEqual(slots.count, len(free))
            self.assertEqual(
                [slot for start, stop in slots.ranges for slot in range(start, stop)],
                sorted(free),
            )
            self.assertTrue(all(a[1] < b[0] for a, b in zip(slots.ranges, slots.ranges[1:])))


if __name__ == "__main__":
    unittest.main()
