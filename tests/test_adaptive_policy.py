import unittest
from concurrent.futures import ThreadPoolExecutor

from adaptive_sparse.policy import BudgetPolicy, Candidate, State


class PolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = BudgetPolicy(100, 85, 65, 5, 100, 2)

    def test_concurrent_grants_and_atomic_commit(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            grants = list(pool.map(lambda key: self.policy.reserve_prefill(key, 30), range(8)))
        self.assertEqual(sum(grants), 3)
        key = grants.index(True)
        self.policy.commit_prefill(key, "r", 25, 0)
        self.assertEqual(self.policy.snapshot()["committed_bytes"], 25)
        self.assertEqual(self.policy.snapshot()["reserved_bytes"], 60)
        self.assertFalse(self.policy.add("other", 16, 0))

    def test_demote_release_only_after_confirmation_and_failed_promote(self):
        p = self.policy
        self.assertTrue(p.add("r", 75, 0))
        (migration,) = p.plan(20, [Candidate("r", 50, 50, 20)], 5)
        self.assertFalse(p.reserve_prefill("pending", 40))
        p.finish(migration, True, 45, 6)
        self.assertTrue(p.reserve_prefill("pending", 40))
        self.assertIsNone(p.begin_promote("r", 30, 25, 9))
        promotion = p.begin_promote("r", 30, 25, 11)
        self.assertFalse(p.reserve_prefill("other", 1))
        p.finish(promotion, False, 0, 12)
        self.assertEqual(p.snapshot()["states"]["r"], State.HOST_BACKED)
        self.assertEqual(p.snapshot()["reserved_bytes"], 40)
        self.assertEqual(p.snapshot()["committed_bytes"], 30)
        with self.assertRaisesRegex(ValueError, "stale"):
            p.finish(promotion, True, 30, 13)

    def test_dwell_pressure_hysteresis_and_inflight_limits(self):
        p = self.policy
        for key in ("a", "b", "c"):
            self.assertTrue(p.add(key, 25, 0))
        candidates = [Candidate(key, 20, 20, 40) for key in ("a", "b", "c")]
        self.assertEqual(p.plan(35, candidates, 4), [])
        migrations = p.plan(60, candidates, 5)
        self.assertEqual(len(migrations), 2)
        self.assertEqual(p.plan(60, candidates, 6), [])
        for migration in migrations:
            p.finish(migration, True, 20, 6)
        self.assertEqual(p.plan(0, candidates, 10), [])
        (promotion,) = p.plan(0, candidates, 11)
        p.finish(promotion, True, 20, 12)
        self.assertEqual(p.plan(0, candidates, 20), [])
        self.assertEqual(p.snapshot()["committed_bytes"], 55)


    def test_temporary_reservation_released_on_failed_demotion(self):
        p = self.policy
        self.assertTrue(p.add("r", 80, 0))
        self.assertIsNone(p.begin_demote("r", 50, 1, 5, temporary_gpu_bytes=21))
        migration = p.begin_demote("r", 50, 1, 5, temporary_gpu_bytes=20)
        self.assertEqual(p.snapshot()["reserved_bytes"], 20)
        self.assertFalse(p.reserve_prefill("pending", 1))
        p.finish(migration, False, 0, 6)
        self.assertEqual(p.snapshot()["states"]["r"], State.GPU_RESIDENT)
        self.assertEqual(p.snapshot()["reserved_bytes"], 0)
        self.assertEqual(p.snapshot()["committed_bytes"], 80)

    def test_stable_low_window_resets_and_prefers_remaining_benefit(self):
        p = BudgetPolicy(100, 85, 65, 0, 100, 2, low_stable_s=5)
        for key in ("a", "b"):
            self.assertTrue(p.add(key, 40, 0))
            migration = p.begin_demote(key, 35, 5, 0)
            p.finish(migration, True, 35, 0)
        candidates = [Candidate("a", 35, 35, 5, 1), Candidate("b", 35, 35, 5, 3)]
        self.assertEqual(p.plan(0, candidates, 1), [])
        self.assertEqual(p.plan(60, candidates, 4), [])
        self.assertEqual(p.plan(0, candidates, 5), [])
        self.assertEqual(p.plan(0, candidates, 9), [])
        (migration,) = p.plan(0, candidates, 10)
        self.assertEqual(migration.request_id, "b")


    def test_growth_reserves_headroom_and_cannot_commit_during_migration(self):
        p = self.policy
        self.assertTrue(p.add("r", 55, 0))
        self.assertTrue(p.reserve_prefill("growth", 40))
        self.assertFalse(p.reserve_prefill("competing", 6))
        migration = p.begin_demote("r", 30, 2, 1, emergency=True)
        with self.assertRaisesRegex(ValueError, "growth"):
            p.commit_growth("growth", "r", 35)
        p.finish(migration, False, 0, 2)
        p.commit_growth("growth", "r", 35)
        self.assertEqual(p.snapshot()["committed_bytes"], 90)
        self.assertEqual(p.snapshot()["reserved_bytes"], 0)
        self.assertIsNone(p.begin_demote("r", 30, 2, 3))


if __name__ == "__main__":
    unittest.main()
