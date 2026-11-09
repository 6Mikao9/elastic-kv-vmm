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

    def test_decode_feedback_is_separated_by_residency_mode(self):
        estimator = ServiceTimeEstimator(alpha=1.0, min_samples=1)

        resident = estimator.observe(
            prompt_tokens=100,
            output_tokens=10,
            ttft_s=0.02,
            tpot_s=0.001,
            residence="resident",
        )

        self.assertEqual(resident.resident_decode_tokens_per_s, 1_000.0)
        self.assertIsNone(resident.host_backed_decode_tokens_per_s)
        self.assertEqual(
            estimator.rates(residence="resident")["decode_tokens_per_s"],
            1_000.0,
        )
        self.assertNotIn(
            "decode_tokens_per_s", estimator.rates(residence="host_backed")
        )

        estimator.observe(
            prompt_tokens=100,
            output_tokens=10,
            ttft_s=0.02,
            tpot_s=0.004,
            residence="host_backed",
        )

        self.assertEqual(
            estimator.rates(residence="resident")["decode_tokens_per_s"],
            1_000.0,
        )
        self.assertEqual(
            estimator.rates(residence="host_backed")["decode_tokens_per_s"],
            250.0,
        )
        snapshot = estimator.snapshot()
        self.assertEqual(snapshot.resident_decode_samples, 1)
        self.assertEqual(snapshot.host_backed_decode_samples, 1)
        self.assertEqual(
            snapshot.as_dict()["decode_tokens_per_s_by_residence"],
            {"resident": 1_000.0, "host_backed": 250.0},
        )

    def test_prefill_feedback_is_isolated_by_kind_and_batch_size(self):
        estimator = ServiceTimeEstimator(alpha=1.0, min_samples=1)
        samples = (
            ("chat", 1, 10_000.0),
            ("chat", 8, 4_000.0),
            ("tool", 1, 2_000.0),
        )
        for kind, batch_size, rate in samples:
            estimator.observe(
                prompt_tokens=100,
                output_tokens=8,
                ttft_s=100.0 / rate,
                tpot_s=0.001,
                kind=kind,
                prefill_batch_size=batch_size,
            )

        for kind, batch_size, rate in samples:
            with self.subTest(kind=kind, batch_size=batch_size):
                self.assertEqual(
                    estimator.rates(
                        kind=kind,
                        prefill_batch_size=batch_size,
                    )["prefill_tokens_per_s"],
                    rate,
                )

    def test_decode_feedback_is_isolated_by_residence_kind_and_batch(self):
        estimator = ServiceTimeEstimator(alpha=1.0, min_samples=1)
        samples = (
            ("resident", "chat", 1, 1_000.0),
            ("host_backed", "chat", 1, 250.0),
            ("resident", "tool", 1, 500.0),
            ("resident", "chat", 4, 125.0),
        )
        for residence, kind, batch_size, rate in samples:
            estimator.observe(
                prompt_tokens=100,
                output_tokens=8,
                ttft_s=0.01,
                tpot_s=1.0 / rate,
                residence=residence,
                kind=kind,
                prefill_batch_size=1,
                decode_batch_size=batch_size,
            )

        for residence, kind, batch_size, rate in samples:
            with self.subTest(
                residence=residence,
                kind=kind,
                batch_size=batch_size,
            ):
                self.assertEqual(
                    estimator.rates(
                        residence=residence,
                        kind=kind,
                        prefill_batch_size=1,
                        decode_batch_size=batch_size,
                    )["decode_tokens_per_s"],
                    rate,
                )

    def test_immature_exact_bucket_withholds_coarse_feedback(self):
        estimator = ServiceTimeEstimator(alpha=1.0, min_samples=2)
        estimator.observe(
            prompt_tokens=1_000,
            output_tokens=8,
            ttft_s=0.1,
            tpot_s=0.001,
            residence="resident",
            kind="chat",
            prefill_batch_size=1,
            decode_batch_size=2,
        )
        estimator.observe(
            prompt_tokens=200,
            output_tokens=8,
            ttft_s=0.1,
            tpot_s=0.002,
            residence="resident",
            kind="chat",
            prefill_batch_size=2,
            decode_batch_size=4,
        )

        self.assertEqual(
            estimator.rates(
                residence="resident",
                kind="chat",
                prefill_batch_size=1,
                decode_batch_size=2,
            ),
            {},
        )
        unseen = estimator.rates(
            residence="resident",
            kind="chat",
            prefill_batch_size=3,
            decode_batch_size=8,
        )
        self.assertEqual(unseen["prefill_tokens_per_s"], 2_000.0)
        self.assertEqual(unseen["decode_tokens_per_s"], 500.0)

    def test_rejects_partial_or_invalid_service_bucket_arguments(self):
        estimator = ServiceTimeEstimator()
        invalid_calls = (
            (
                {"kind": "chat"},
                "kind and prefill_batch_size must be supplied together",
            ),
            (
                {"prefill_batch_size": 1},
                "kind and prefill_batch_size must be supplied together",
            ),
            (
                {"kind": "chat", "prefill_batch_size": 1, "decode_batch_size": 2},
                "decode_batch_size requires residence and kind",
            ),
            (
                {"kind": "x" * 65, "prefill_batch_size": 1},
                "kind must be a short non-empty string",
            ),
            (
                {"kind": "chat", "prefill_batch_size": 0},
                "prefill_batch_size must be a positive integer",
            ),
        )
        for kwargs, message in invalid_calls:
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(ValueError, message),
            ):
                estimator.observe(
                    prompt_tokens=10,
                    output_tokens=1,
                    ttft_s=0.01,
                    tpot_s=0.001,
                    **kwargs,
                )

        with self.assertRaisesRegex(
            ValueError,
            "decode_batch_size requires residence and kind",
        ):
            estimator.rates(
                residence="resident",
                decode_batch_size=2,
            )

    def test_snapshot_serializes_service_time_buckets(self):
        estimator = ServiceTimeEstimator(alpha=1.0, min_samples=1)
        estimator.observe(
            prompt_tokens=100,
            output_tokens=8,
            ttft_s=0.01,
            tpot_s=0.001,
            residence="resident",
            kind="chat",
            prefill_batch_size=2,
            decode_batch_size=4,
        )

        snapshot = estimator.snapshot().as_dict()
        self.assertEqual(
            snapshot["prefill_buckets"],
            [
                {
                    "kind": "chat",
                    "batch_size": 2,
                    "tokens_per_s": 10_000.0,
                    "samples": 1,
                }
            ],
        )
        self.assertEqual(
            snapshot["decode_buckets"],
            [
                {
                    "residence": "resident",
                    "kind": "chat",
                    "batch_size": 4,
                    "tokens_per_s": 1_000.0,
                    "samples": 1,
                }
            ],
        )

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

        with self.assertRaisesRegex(ValueError, "residence"):
            estimator.rates(residence="auto")

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
