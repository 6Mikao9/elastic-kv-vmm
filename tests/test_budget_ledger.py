import unittest

from experimental.budget_ledger import BudgetLedger


class BudgetLedgerTest(unittest.TestCase):
    def test_migration_charges_new_and_old_until_quiescence(self):
        ledger = BudgetLedger(100, safety_margin_bytes=10, external_bytes=10)
        ledger.register("old", "kv", 45)
        self.assertFalse(ledger.reserve("too_large", 36))
        self.assertTrue(ledger.reserve("migration", 35))
        ledger.commit("migration", "new", "hot", 25)
        self.assertEqual(ledger.snapshot()["committed_by_category"], {"kv": 45, "hot": 25})
        with self.assertRaises(ValueError):
            ledger.release("old", quiescent=False)
        self.assertFalse(ledger.reserve("workspace", 11))
        self.assertEqual(ledger.release("old", quiescent=True), 45)
        self.assertTrue(ledger.reserve("workspace", 55))

    def test_failed_commit_keeps_ticket_and_allocation(self):
        ledger = BudgetLedger(100)
        ledger.register("x", "kv", 50)
        self.assertTrue(ledger.reserve("ticket", 30))
        expected = ledger.snapshot()
        for key, actual in (("x", 10), ("new", 31), ("new", -1)):
            with self.assertRaises(ValueError):
                ledger.commit("ticket", key, "hot", actual)
            self.assertEqual(ledger.snapshot(), expected)
        ledger.commit("ticket", "new", "hot", 20)
        self.assertEqual(ledger.snapshot()["available_bytes"], 30)
        self.assertEqual(ledger.snapshot()["pending_bytes"], 0)

    def test_observed_external_growth_never_fakes_release(self):
        ledger = BudgetLedger(100, 10)
        ledger.register("kv", "kv", 60)
        self.assertTrue(ledger.reserve("pending", 20))
        ledger.set_external_bytes(20)
        self.assertEqual(ledger.snapshot()["overcommitted_bytes"], 10)
        self.assertFalse(ledger.reserve("even_zero", 0))
        ledger.commit("pending", "graph", "graph", 20)
        self.assertEqual(ledger.snapshot()["committed_bytes"], 80)
        ledger.set_external_bytes(0)
        self.assertFalse(ledger.snapshot()["overcommitted"])
        self.assertTrue(ledger.reserve("now_fits", 10))
        self.assertEqual(ledger.cancel("now_fits"), 10)

    def test_existing_registration_and_bad_inputs(self):
        ledger = BudgetLedger(10)
        ledger.register("real", "indexer", 11)
        self.assertTrue(ledger.snapshot()["overcommitted"])
        with self.assertRaises(ValueError):
            ledger.register("real", "other", 0)
        ledger.release("real", quiescent=True)
        self.assertTrue(ledger.reserve("ticket", 1))
        with self.assertRaises(ValueError):
            ledger.reserve("ticket", 1)
        with self.assertRaises(ValueError):
            ledger.reserve("negative", -1)
        with self.assertRaises(ValueError):
            ledger.set_external_bytes(-1)


if __name__ == "__main__":
    unittest.main()
