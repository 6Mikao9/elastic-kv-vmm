import math
import unittest

from adaptive_sparse.migration_budget import MigrationBandwidthEstimator


class MigrationBandwidthEstimatorTest(unittest.TestCase):
    def test_uses_fallback_until_a_real_sample_exists(self):
        estimator = MigrationBandwidthEstimator(1_000, 0.25)
        self.assertEqual(estimator.bandwidth("demotion"), 1_000)
        self.assertEqual(estimator.effective_byte_cap(10_000, 0), 10_000)
        self.assertEqual(estimator.effective_byte_cap(10_000, 1_500), 1_500)
        self.assertEqual(estimator.samples("demotion"), 0)

    def test_real_sample_replaces_fallback_and_ewma_follows_it(self):
        estimator = MigrationBandwidthEstimator(10_000, 0.25)
        self.assertEqual(estimator.observe("demotion", 2_000, 2), 1_000)
        self.assertEqual(estimator.bandwidth("demotion"), 1_000)
        self.assertEqual(estimator.observe("demotion", 4_000, 1), 4_000)
        self.assertEqual(estimator.bandwidth("demotion"), 1_750)
        self.assertEqual(estimator.samples("demotion"), 2)
        self.assertEqual(estimator.bandwidth("promotion"), 10_000)

    def test_directions_are_independent(self):
        estimator = MigrationBandwidthEstimator(1_000, 0.5)
        estimator.observe("demotion", 2_000, 1)
        estimator.observe("promotion", 8_000, 2)
        self.assertEqual(estimator.bandwidth("demotion"), 2_000)
        self.assertEqual(estimator.bandwidth("promotion"), 4_000)

    def test_zero_transfer_or_nonpositive_time_is_ignored(self):
        estimator = MigrationBandwidthEstimator(1_000, 0.5)
        self.assertIsNone(estimator.observe("demotion", 0, 1))
        self.assertIsNone(estimator.observe("demotion", 10, 0))
        self.assertIsNone(estimator.observe("demotion", 10, -1))
        self.assertEqual(estimator.bandwidth("demotion"), 1_000)
        self.assertEqual(estimator.samples("demotion"), 0)

    def test_time_cap_never_exceeds_byte_cap(self):
        estimator = MigrationBandwidthEstimator(2_000, 1)
        self.assertEqual(estimator.effective_byte_cap(500, 1_000), 500)
        self.assertEqual(estimator.effective_byte_cap(5_000, 1_000), 2_000)
        self.assertEqual(estimator.effective_byte_cap(5_000, 1), 2)

    def test_invalid_configuration_is_rejected(self):
        for initial, alpha in ((0, 0.5), (1, 0), (1, 1.1), (1, math.nan)):
            with self.subTest(initial=initial, alpha=alpha):
                with self.assertRaises(ValueError):
                    MigrationBandwidthEstimator(initial, alpha)
        estimator = MigrationBandwidthEstimator(1, 0.5)
        with self.assertRaises(ValueError):
            estimator.observe("unknown", 1, 1)
        with self.assertRaises(ValueError):
            estimator.effective_byte_cap(0, 1)
        with self.assertRaises(ValueError):
            estimator.effective_byte_cap(1, -1)


if __name__ == "__main__":
    unittest.main()
