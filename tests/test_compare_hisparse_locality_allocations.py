import unittest

from scripts.compare_hisparse_locality_allocations import CANDIDATES, compare_allocations


def summary_fixture():
    rows = []
    for rank in (0, 1):
        for index in range(1, 7):
            for layer in (5, 15):
                selected = 500 if index == 1 else 100
                if index == 6:
                    selected = 0
                row = dict(rank=rank, request=f"R{index:02d}", layer=layer, selected=selected)
                for capacity in CANDIDATES:
                    row[f"simulated_misses_{capacity}"] = (
                        50 if index == rank + 1 and capacity < 6144 else 0
                    )
                rows.append(row)
    return {
        "coverage": "complete", "warmup_steps": 16,
        "selection_limits": {"capacities": {
            str(capacity): {"status": "feasible"} for capacity in CANDIDATES
        }},
        "rank_request_layer_after_first_16": rows,
    }


class AllocationComparisonTests(unittest.TestCase):
    def test_fixed_budget_optimizes_each_rank_and_uses_weighted_denominator(self):
        result = compare_allocations(summary_fixture())
        self.assertEqual(result["request_aliases"], ["R01", "R02", "R03", "R04", "R05"])
        for rank, row in enumerate(result["rank_results"]):
            self.assertEqual(row["rank"], rank)
            self.assertEqual(row["selected_tokens"], 1800)
            comparison, = row["comparisons"]
            self.assertEqual(comparison["total_token_budget"], 20480)
            self.assertGreater(comparison["exact_allocations"], 1)
            self.assertEqual(sum(comparison["best"]["allocation"].values()), 20480)
            self.assertGreaterEqual(comparison["best"]["allocation"][f"R{rank + 1:02d}"], 6144)
            self.assertEqual(comparison["best"]["simulated_misses"], 0)
            self.assertEqual(comparison["uniform"]["simulated_misses"], 100)
            self.assertAlmostEqual(comparison["best_improvement_over_uniform_pp"], 10000 / 1800)

    def test_exact_budget_and_infeasible_capacity_are_not_silently_relaxed(self):
        fixture = summary_fixture()
        fixture["selection_limits"]["capacities"]["2560"]["status"] = "infeasible"
        result = compare_allocations(fixture, budgets=(20480, 20481))
        self.assertNotIn(2560, result["feasible_candidates"])
        exact, impossible = result["rank_results"][0]["comparisons"]
        self.assertGreater(exact["exact_allocations"], 0)
        self.assertEqual(impossible["exact_allocations"], 0)
        self.assertIsNone(impossible["best"])

    def test_missing_rank_request_or_simulation_fails(self):
        fixture = summary_fixture()
        row = fixture["rank_request_layer_after_first_16"][0]
        row["simulated_misses_3072"] = None
        with self.assertRaisesRegex(ValueError, "invalid simulated miss count"):
            compare_allocations(fixture)

        fixture = summary_fixture()
        fixture["rank_request_layer_after_first_16"] = [
            row for row in fixture["rank_request_layer_after_first_16"]
            if (row["rank"], row["request"]) != (1, "R01")
        ]
        with self.assertRaisesRegex(ValueError, "different post-warmup request sets"):
            compare_allocations(fixture)

    def test_joint_objectives_use_shared_allocation_but_can_favor_different_requests(self):
        fixture = summary_fixture()
        cost = {"R01": 50, "R02": 40, "R03": 30}
        for row in fixture["rank_request_layer_after_first_16"]:
            for capacity in CANDIDATES:
                row[f"simulated_misses_{capacity}"] = (
                    cost.get(row["request"], 0) if capacity < 6144 else 0
                )
        joint = compare_allocations(fixture)["sampled_rank_joint_oracle"]
        self.assertEqual(joint["sampled_ranks"], [0, 1])
        self.assertEqual(joint["unsampled_tp4_ranks"], [2, 3])
        self.assertEqual(joint["weighted"]["selected_tokens"], 3600)
        weighted, = joint["weighted"]["comparisons"]
        equal, = joint["mean_per_request"]["comparisons"]
        self.assertEqual(sum(weighted["best"]["allocation"].values()), 20480)
        self.assertEqual(sum(equal["best"]["allocation"].values()), 20480)
        self.assertEqual(weighted["best"]["simulated_misses"], 120)
        self.assertEqual(equal["best"]["simulated_misses"], 200)
        self.assertAlmostEqual(weighted["best"]["mean_per_request_simulated_miss_rate"], 0.06)
        self.assertAlmostEqual(equal["best"]["mean_per_request_simulated_miss_rate"], 0.02)
