import unittest

from adaptive_sparse.residency_broker import (
    ResidencyBroker,
    page_aligned_growth_slots,
    page_aligned_slots,
    remaining_output_tokens,
)


class ResidencyBrokerTest(unittest.TestCase):
    def test_incremental_reserved_total_tracks_mixed_updates(self):
        broker = ResidencyBroker()
        for key, amount in (("a", 10), ("b", 20), ("c", 30)):
            broker.try_admit(
                key,
                available_bytes=100,
                fixed_reserve_bytes=0,
                predicted_growth_bytes=amount,
            )

        self.assertEqual(broker.reserved_bytes, 60)
        self.assertEqual(broker.consume("a", bytes_consumed=4), 4)
        self.assertEqual(broker.reserved_bytes, 56)
        self.assertEqual(broker.release("b"), 20)
        self.assertEqual(broker.reserved_bytes, 36)
        self.assertEqual(broker.release("missing"), 0)
        self.assertEqual(broker.reserved_bytes, 36)

    def test_dry_run_assessment_does_not_mutate_ledger(self):
        broker = ResidencyBroker()
        decision = broker.assess(
            "a", available_bytes=100, fixed_reserve_bytes=20,
            predicted_growth_bytes=30,
        )
        self.assertTrue(decision.admit)
        self.assertEqual(decision.required_bytes, 50)
        self.assertEqual(broker.reserved_bytes, 0)

    def test_dry_run_can_track_actual_claim_after_a_would_decline(self):
        broker = ResidencyBroker()
        decision = broker.assess(
            "a", available_bytes=9, fixed_reserve_bytes=0,
            predicted_growth_bytes=10,
        )
        self.assertFalse(decision.admit)
        self.assertEqual(
            decision.reason, "insufficient-growth-headroom"
        )
        self.assertEqual(broker.reserved_bytes, 0)

        # Dry-run preserves actual placement, then reserves the growth that
        # placement really promised so later assessments see the same state.
        broker.reserve("a", predicted_growth_bytes=10)
        self.assertEqual(broker.reserved_bytes, 10)
        self.assertEqual(broker.snapshot()["would_decline"], 1)

    def test_later_claim_cannot_spend_promised_growth_twice(self):
        broker = ResidencyBroker()
        self.assertTrue(broker.try_admit(
            "a", available_bytes=100, fixed_reserve_bytes=10,
            predicted_growth_bytes=60,
        ).admit)
        declined = broker.try_admit(
            "b", available_bytes=100, fixed_reserve_bytes=10,
            predicted_growth_bytes=40,
        )
        self.assertFalse(declined.admit)
        self.assertEqual(declined.reason, "insufficient-growth-headroom")
        self.assertEqual(declined.required_bytes, 110)

    def test_exact_boundary_fits_and_prior_reservations_are_aggregated(self):
        broker = ResidencyBroker()
        self.assertTrue(broker.try_admit(
            "a", available_bytes=100, fixed_reserve_bytes=10,
            predicted_growth_bytes=40,
        ).admit)
        exact = broker.try_admit(
            "b", available_bytes=100, fixed_reserve_bytes=10,
            predicted_growth_bytes=50,
        )
        self.assertTrue(exact.admit)
        self.assertEqual(exact.required_bytes, 100)
        self.assertEqual(exact.reserved_bytes, 90)
        self.assertEqual(broker.reserved_bytes, 90)

        over = broker.assess(
            "c", available_bytes=100, fixed_reserve_bytes=10,
            predicted_growth_bytes=1,
        )
        self.assertFalse(over.admit)
        self.assertEqual(over.required_bytes, 101)
        self.assertEqual(broker.reserved_bytes, 90)

    def test_rejected_atomic_admission_does_not_leave_a_reservation(self):
        broker = ResidencyBroker()
        declined = broker.try_admit(
            "a", available_bytes=9, fixed_reserve_bytes=0,
            predicted_growth_bytes=10,
        )
        self.assertFalse(declined.admit)
        self.assertEqual(broker.reserved_bytes, 0)
        self.assertEqual(broker.snapshot()["active_reservations"], 0)

        admitted = broker.try_admit(
            "a", available_bytes=10, fixed_reserve_bytes=0,
            predicted_growth_bytes=10,
        )
        self.assertTrue(admitted.admit)
        self.assertEqual(broker.reserved_bytes, 10)

    def test_consumption_turns_reservation_into_observed_allocator_bytes(self):
        broker = ResidencyBroker()
        broker.try_admit(
            "a", available_bytes=100, fixed_reserve_bytes=0,
            predicted_growth_bytes=60,
        )
        broker.set_last_length("a", 8)
        self.assertEqual(broker.consume_new_tokens(
            "a", length=10, bytes_per_token=9,
        ), 18)
        self.assertEqual(broker.reserved_bytes, 42)
        self.assertEqual(broker.consume_new_tokens(
            "a", length=20, bytes_per_token=9,
        ), 42)
        self.assertEqual(broker.release("a"), 0)

    def test_release_returns_only_unconsumed_promise(self):
        broker = ResidencyBroker()
        broker.try_admit(
            "a", available_bytes=100, fixed_reserve_bytes=0,
            predicted_growth_bytes=50,
        )
        broker.consume("a", bytes_consumed=20)
        self.assertEqual(broker.release("a"), 30)
        self.assertEqual(broker.reserved_bytes, 0)

    def test_consumption_saturates_and_unused_zero_reservation_can_be_released(self):
        broker = ResidencyBroker()
        broker.try_admit(
            "a", available_bytes=100, fixed_reserve_bytes=0,
            predicted_growth_bytes=10,
        )
        self.assertEqual(broker.consume("a", bytes_consumed=15), 10)
        self.assertEqual(broker.reserved_bytes, 0)
        self.assertEqual(broker.release("a"), 0)

        broker.try_admit(
            "zero", available_bytes=0, fixed_reserve_bytes=0,
            predicted_growth_bytes=0,
        )
        self.assertEqual(broker.reserved_bytes, 0)
        self.assertEqual(broker.release("zero"), 0)
        self.assertEqual(broker.consume("missing", bytes_consumed=1), 0)
        self.assertEqual(broker.release("missing"), 0)

    def test_token_consumption_is_monotonic_across_length_regressions(self):
        broker = ResidencyBroker()
        broker.try_admit(
            "a", available_bytes=100, fixed_reserve_bytes=0,
            predicted_growth_bytes=30,
        )
        broker.set_last_length("a", 10)
        self.assertEqual(broker.consume_new_tokens(
            "a", length=8, bytes_per_token=2,
        ), 0)
        self.assertEqual(broker.consume_new_tokens(
            "a", length=10, bytes_per_token=2,
        ), 0)
        self.assertEqual(broker.consume_new_tokens(
            "a", length=12, bytes_per_token=2,
        ), 4)
        self.assertEqual(broker.reserved_bytes, 26)

    def test_page_aligned_growth_rounds_up_once_per_allocator_page(self):
        self.assertEqual(page_aligned_growth_slots(
            current_tokens=128, target_tokens=130, page_size=128,
        ), 128)
        self.assertEqual(page_aligned_growth_slots(
            current_tokens=129, target_tokens=255, page_size=128,
        ), 0)
        self.assertEqual(page_aligned_growth_slots(
            current_tokens=255, target_tokens=256, page_size=128,
        ), 0)
        self.assertEqual(page_aligned_growth_slots(
            current_tokens=255, target_tokens=257, page_size=128,
        ), 128)

    def test_page_aligned_consumption_matches_predictions(self):
        broker = ResidencyBroker()
        broker.try_admit(
            "a", available_bytes=1000, fixed_reserve_bytes=0,
            predicted_growth_bytes=256,
        )
        broker.set_last_length("a", 128)
        self.assertEqual(broker.consume_new_page_slots(
            "a", length=129, bytes_per_token=1, page_size=128,
        ), 128)
        self.assertEqual(broker.consume_new_page_slots(
            "a", length=255, bytes_per_token=1, page_size=128,
        ), 0)
        self.assertEqual(broker.consume_new_page_slots(
            "a", length=257, bytes_per_token=1, page_size=128,
        ), 128)
        self.assertEqual(broker.reserved_bytes, 0)

    def test_promotion_counts_current_history_plus_future_page_growth(self):
        current = 200
        future = 70
        page = 128
        required = page_aligned_slots(current, page) + page_aligned_growth_slots(
            current_tokens=current, target_tokens=current + future, page_size=page
        )
        self.assertEqual(required, 384)

    def test_promotion_prediction_discounts_tokens_already_generated(self):
        self.assertEqual(remaining_output_tokens(
            predicted_total=512, current_tokens=456, baseline_tokens=256,
        ), 312)
        self.assertEqual(remaining_output_tokens(
            predicted_total=64, current_tokens=456, baseline_tokens=256,
        ), 0)

    def test_snapshot_is_defensive_and_reports_isolated_counters(self):
        first = ResidencyBroker()
        second = ResidencyBroker()
        first.try_admit(
            "a", available_bytes=100, fixed_reserve_bytes=0,
            predicted_growth_bytes=30,
        )
        first.consume("a", bytes_consumed=10)
        snapshot = first.snapshot()
        self.assertEqual(snapshot["active_reservations"], 1)
        self.assertEqual(snapshot["outstanding_bytes"], 20)
        self.assertEqual(snapshot["consumed_bytes"], 10)
        snapshot["outstanding_bytes"] = -1
        snapshot["decline_reasons"]["fabricated"] = 1

        fresh = first.snapshot()
        self.assertEqual(fresh["outstanding_bytes"], 20)
        self.assertNotIn("fabricated", fresh["decline_reasons"])
        self.assertEqual(second.snapshot()["active_reservations"], 0)

    def test_snapshot_accounts_for_admission_consumption_and_release(self):
        broker = ResidencyBroker()
        broker.assess(
            "declined", available_bytes=9, fixed_reserve_bytes=0,
            predicted_growth_bytes=10,
        )
        broker.try_admit(
            "a", available_bytes=100, fixed_reserve_bytes=0,
            predicted_growth_bytes=30,
        )
        broker.consume("a", bytes_consumed=12)
        self.assertEqual(broker.release("a"), 18)
        stats = broker.snapshot()
        self.assertEqual(stats["assessments"], 2)
        self.assertEqual(stats["would_admit"], 1)
        self.assertEqual(stats["would_decline"], 1)
        self.assertEqual(stats["reservations"], 1)
        self.assertEqual(stats["consumed_bytes"], 12)
        self.assertEqual(stats["released_bytes"], 18)
        self.assertEqual(
            stats["decline_reasons"], {"insufficient-growth-headroom": 1}
        )

    def test_duplicate_and_invalid_inputs_are_rejected(self):
        broker = ResidencyBroker()
        broker.try_admit(
            "a", available_bytes=100, fixed_reserve_bytes=0,
            predicted_growth_bytes=10,
        )
        duplicate = broker.assess(
            "a", available_bytes=100, fixed_reserve_bytes=0,
            predicted_growth_bytes=10,
        )
        self.assertFalse(duplicate.admit)
        self.assertEqual(duplicate.reason, "duplicate-request")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            broker.reserve("a", predicted_growth_bytes=10)
        with self.assertRaisesRegex(ValueError, "available_bytes"):
            broker.assess(
                "b", available_bytes=-1, fixed_reserve_bytes=0,
                predicted_growth_bytes=0,
            )

    def test_every_broker_input_rejects_bool_float_and_negative_values(self):
        for value in (-1, True, 1.5):
            with self.subTest(value=value):
                broker = ResidencyBroker()
                with self.assertRaises(ValueError):
                    broker.assess(
                        "a", available_bytes=value, fixed_reserve_bytes=0,
                        predicted_growth_bytes=0,
                    )
                with self.assertRaises(ValueError):
                    broker.reserve("a", predicted_growth_bytes=value)
                with self.assertRaises(ValueError):
                    broker.consume("a", bytes_consumed=value)
                with self.assertRaises(ValueError):
                    broker.set_last_length("a", value)
                with self.assertRaises(ValueError):
                    broker.consume_new_tokens(
                        "a", length=value, bytes_per_token=1,
                    )


if __name__ == "__main__":
    unittest.main()
