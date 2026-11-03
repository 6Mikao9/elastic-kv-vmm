from types import SimpleNamespace as Bag
import unittest
from unittest.mock import patch

from adaptive_sparse.adaptive_runtime import AdaptiveRuntime, MigrationFatal
from adaptive_sparse.policy import BudgetPolicy, Candidate, State


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.policy = BudgetPolicy(100, 80, 60, 2, 100, 1)
        self.policy.add("request", 70, 0)
        self.ctrl = Bag(_ekv_resident={3})
        self.request = Bag(kv=Bag(req_pool_idx=3), hisparse_staging=False)
        self.candidate = Candidate("request", 30, 0, 40)
        self.runtime = AdaptiveRuntime(self.policy, self.ctrl, lambda x: (x, x),
                                       lambda: None, lambda: 20, eager=True, overlap=False)

    def run_boundary(self, now):
        return self.runtime.at_prefill_boundary(20, [self.candidate],
                                                {"request": self.request}, now)

    def test_dwell_and_minimum_physical_feedback(self):
        self.assertEqual(self.run_boundary(1), [])
        self.runtime.exchange = lambda x: (20, 18) if type(x) is int else (x, x)

        def commit(prepared, publish, synchronize, trim):
            publish(3)
            return trim()

        with patch("adaptive_sparse.adaptive_runtime.prepare_demote", return_value=object()), patch(
            "adaptive_sparse.adaptive_runtime.commit_demote", side_effect=commit
        ):
            result, = self.run_boundary(2)
        self.assertEqual(result["released_bytes"], 18)
        self.assertEqual(self.policy.snapshot()["committed_bytes"], 52)
        self.assertEqual(self.policy.snapshot()["states"]["request"], State.HOST_BACKED)
        self.assertFalse(self.ctrl._ekv_resident)

    def test_rank_prepare_failure_aborts_local_and_poison(self):
        decisions = iter(((True, True), (True, False)))
        self.runtime.exchange = lambda x: next(decisions) if type(x) is bool else (x, x)
        prepared = object()
        with patch("adaptive_sparse.adaptive_runtime.prepare_demote", return_value=prepared), patch(
            "adaptive_sparse.adaptive_runtime.abort_demote"
        ) as abort, self.assertRaises(MigrationFatal):
            self.run_boundary(2)
        abort.assert_called_once_with(prepared)
        self.assertTrue(self.runtime.poisoned)
        self.assertEqual(self.ctrl._ekv_resident, {3})

    def test_rank_commit_failure_is_terminal(self):
        self.runtime.exchange = lambda x: (x, None) if type(x) is int else (x, x)
        with patch("adaptive_sparse.adaptive_runtime.prepare_demote", return_value=object()), patch(
            "adaptive_sparse.adaptive_runtime.commit_demote", return_value=20
        ), self.assertRaises(MigrationFatal):
            self.run_boundary(2)
        self.assertEqual(self.policy.snapshot()["committed_bytes"], 70)


if __name__ == "__main__":
    unittest.main()
