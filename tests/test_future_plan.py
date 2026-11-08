import math
import unittest

from adaptive_sparse.future_plan import (
    PlanResult,
    PlannerConfig,
    RequestProfile,
    ResidencyMode,
    simulate,
)


class FuturePlanTest(unittest.TestCase):
    def setUp(self):
        self.config = PlannerConfig(
            memory_budget_bytes=8_500_000,
            kv_bytes_per_token=1000,
            prefill_workspace_bytes_per_token=0,
            sparse_hot_tokens=2000,
            prefill_tokens_per_ms=1000,
            resident_decode_tokens_per_ms=2,
            sparse_decode_tokens_per_ms=1.6,
            max_active_requests=8,
            lookahead_ms=50,
        )
        self.workload = [
            RequestProfile(
                "long", arrival_ms=0, prompt_tokens=1000, output_tokens=7000
            ),
            *[
                RequestProfile(
                    f"short-{index}",
                    arrival_ms=10,
                    prompt_tokens=100,
                    output_tokens=400,
                    kind="tool",
                    priority=2,
                    deadline_ms=500,
                )
                for index in range(5)
            ],
        ]

    def test_future_policy_protects_imminent_deadline_work(self):
        fifo = simulate(self.workload, self.config, "fifo")
        future = simulate(self.workload, self.config, "future")

        self.assertEqual(fifo.admissions[0].mode, ResidencyMode.RESIDENT)
        self.assertEqual(future.admissions[0].mode, ResidencyMode.HI_SPARSE)
        self.assertGreater(fifo.deadline_misses, future.deadline_misses)
        self.assertEqual(future.deadline_misses, 0)
        self.assertGreaterEqual(future.weighted_completion_ms, 0)

    def test_prefill_workspace_can_force_bounded_hot_set(self):
        config = PlannerConfig(
            memory_budget_bytes=8_000_000,
            kv_bytes_per_token=1000,
            prefill_workspace_bytes_per_token=1,
            sparse_hot_tokens=2000,
            prefill_tokens_per_ms=1000,
            resident_decode_tokens_per_ms=1,
            sparse_decode_tokens_per_ms=1,
        )
        request = RequestProfile(
            "exact", arrival_ms=0, prompt_tokens=1000, output_tokens=7000
        )
        result = simulate([request], config, "mode-aware")
        self.assertEqual(result.admissions[0].mode, ResidencyMode.HI_SPARSE)
        self.assertEqual(result.admissions[0].prefill_workspace_bytes, 1000)
        self.assertLessEqual(result.peak_memory_bytes, config.memory_budget_bytes)

    def test_deadline_urgency_orders_equal_arrivals(self):
        workload = [
            RequestProfile(
                "a-low", arrival_ms=0, prompt_tokens=10, output_tokens=10
            ),
            RequestProfile(
                "z-deadline",
                arrival_ms=0,
                prompt_tokens=10,
                output_tokens=10,
                deadline_ms=15,
            ),
        ]
        result = simulate(workload, self.config, "mode-aware")
        self.assertEqual(result.admissions[0].request_id, "z-deadline")

    def test_simulation_is_deterministic_and_never_reports_overcommit(self):
        first = simulate(self.workload, self.config, "future")
        second = simulate(self.workload, self.config, "future")
        self.assertIsInstance(first, PlanResult)
        self.assertEqual(first, second)
        self.assertLessEqual(first.peak_memory_bytes, self.config.memory_budget_bytes)
        self.assertEqual(first.peak_backed_bytes, first.peak_memory_bytes)
        self.assertEqual(first.mode_switches, 0)

    def test_invalid_shared_prefix_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "prefix_group"):
            RequestProfile(
                "bad",
                arrival_ms=0,
                prompt_tokens=10,
                output_tokens=1,
                shared_prefix_tokens=5,
            )

    def test_non_finite_rates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite positive"):
            PlannerConfig(
                memory_budget_bytes=1,
                kv_bytes_per_token=1,
                prefill_workspace_bytes_per_token=0,
                prefill_tokens_per_ms=math.inf,
            )


if __name__ == "__main__":
    unittest.main()
