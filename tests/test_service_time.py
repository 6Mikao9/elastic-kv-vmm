import unittest

from adaptive_sparse.service_time import (
    MAX_SERVICE_RATE,
    MIN_SERVICE_RATE,
    ServiceTimeEstimator,
    estimate_service_duration_ms,
)


class ServiceTimeEstimatorTest(unittest.TestCase):
    def test_duration_estimate_preserves_static_fallback_without_feedback(self):
        self.assertEqual(
            estimate_service_duration_ms(
                prompt_tokens=100,
                output_tokens=50,
                fallback_ms=123.0,
            ),
            123.0,
        )

    def test_duration_estimate_uses_measured_rates(self):
        self.assertEqual(
            estimate_service_duration_ms(
                prompt_tokens=1_000,
                output_tokens=500,
                fallback_ms=999.0,
                prefill_tokens_per_s=10_000,
                decode_tokens_per_s=5_000,
            ),
            200.0,
        )

    def test_duration_estimate_fills_each_missing_rate_with_its_default(self):
        with self.subTest(missing="decode"):
            self.assertEqual(
                estimate_service_duration_ms(
                    prompt_tokens=8_000,
                    output_tokens=100,
                    fallback_ms=999.0,
                    prefill_tokens_per_s=8_000,
                ),
                1_005.0,
            )
        with self.subTest(missing="prefill"):
            self.assertEqual(
                estimate_service_duration_ms(
                    prompt_tokens=100,
                    output_tokens=20_000,
                    fallback_ms=999.0,
                    decode_tokens_per_s=20_000,
                ),
                1_012.5,
            )

    def test_starts_without_guessing_a_rate(self):
        estimator = ServiceTimeEstimator()

        self.assertEqual(estimator.rates(), {})
        self.assertEqual(estimator.snapshot().prefill_samples, 0)
        self.assertEqual(estimator.snapshot().decode_samples, 0)

    def test_uses_independent_ewma_state_for_prefill_and_decode(self):
        estimator = ServiceTimeEstimator(alpha=0.5)
        first = estimator.observe(
            prompt_tokens=1_000,
            output_tokens=10,
            ttft_s=0.1,
            tpot_s=0.001,
        )
        second = estimator.observe(
            prompt_tokens=400,
            output_tokens=10,
            ttft_s=0.2,
            tpot_s=0.002,
        )

        self.assertEqual(first.prefill_tokens_per_s, 10_000.0)
        self.assertEqual(first.decode_tokens_per_s, 1_000.0)
        self.assertAlmostEqual(second.prefill_tokens_per_s, 6_000.0)
        self.assertAlmostEqual(second.decode_tokens_per_s, 750.0)
        self.assertEqual(second.prefill_samples, 2)
        self.assertEqual(second.decode_samples, 2)

    def test_waits_for_minimum_samples_before_emitting_feedback(self):
        estimator = ServiceTimeEstimator(alpha=1.0, min_samples=2)
        estimator.observe(
            prompt_tokens=1_000,
            output_tokens=10,
            ttft_s=0.1,
            tpot_s=0.001,
        )

        self.assertEqual(estimator.rates(), {})

        estimator.observe(
            prompt_tokens=400,
            output_tokens=10,
            ttft_s=0.2,
            tpot_s=0.002,
        )

        self.assertEqual(
            estimator.rates(),
            {
                "prefill_tokens_per_s": 2_000.0,
                "decode_tokens_per_s": 500.0,
            },
        )

    def test_clips_pathological_rates_and_counts_both_directions(self):
        estimator = ServiceTimeEstimator(alpha=1.0)
        snapshot = estimator.observe(
            prompt_tokens=1,
            output_tokens=1,
            ttft_s=1e-12,
            tpot_s=1e6,
        )

        self.assertEqual(snapshot.prefill_tokens_per_s, MAX_SERVICE_RATE)
        self.assertEqual(snapshot.decode_tokens_per_s, MIN_SERVICE_RATE)
        self.assertEqual(snapshot.clipped_prefill_samples, 1)
        self.assertEqual(snapshot.clipped_decode_samples, 1)

    def test_rejects_invalid_sample_without_changing_rates(self):
        estimator = ServiceTimeEstimator()

        with self.assertRaisesRegex(ValueError, "ttft_s"):
            estimator.observe(
                prompt_tokens=10,
                output_tokens=1,
                ttft_s=0.0,
                tpot_s=0.001,
            )
        self.assertEqual(estimator.rates(), {})

        estimator.record_rejected_sample()
        self.assertEqual(estimator.snapshot().rejected_samples, 1)

    def test_seed_values_are_validated_and_transformable_to_hint_fields(self):
        estimator = ServiceTimeEstimator(
            prefill_tokens_per_s=1_000,
            decode_tokens_per_s=2_000,
        )
        self.assertEqual(
            estimator.rates(),
            {
                "prefill_tokens_per_s": 1_000.0,
                "decode_tokens_per_s": 2_000.0,
            },
        )
        with self.assertRaisesRegex(ValueError, "alpha"):
            ServiceTimeEstimator(alpha=0)
        with self.assertRaisesRegex(ValueError, "min_samples"):
            ServiceTimeEstimator(min_samples=0)


if __name__ == "__main__":
    unittest.main()
