import unittest

from adaptive_sparse.elastic_admission import (
    ElasticKVConfig,
    decide_elastic_admission,
    elastic_config_from_planner,
)
from adaptive_sparse.future_plan import PlannerConfig, RolloutSubmission


def submission(
    request_id,
    *,
    prompt=100,
    output=100,
    residence="resident",
):
    return RolloutSubmission(
        request_id=request_id,
        planned_arrival_ms=0,
        submit_at_ms=0,
        estimated_finish_ms=100,
        kind="rollout",
        prompt_tokens=prompt,
        output_tokens=output,
        hint={
            "release_after_ms": 0,
            "expected_output_tokens": output,
            "weight": 1.0,
            "kind": "rollout",
            "residence": residence,
        },
    )


class ElasticAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.config = ElasticKVConfig(
            memory_budget_bytes=1_000,
            kv_bytes_per_token=2,
            sparse_hot_tokens=64,
        )

    def test_low_load_keeps_planned_residency(self):
        decision = decide_elastic_admission(
            submission("a"), active_bytes=100, config=self.config
        )

        self.assertEqual(decision.chosen_residence, "resident")
        self.assertEqual(decision.active_after_bytes, 500)
        self.assertFalse(decision.defer_for_budget)

    def test_high_load_demotes_new_request_without_evicting_active(self):
        decision = decide_elastic_admission(
            submission("b"), active_bytes=700, config=self.config
        )

        self.assertEqual(decision.planned_residence, "resident")
        self.assertEqual(decision.chosen_residence, "host_backed")
        self.assertEqual(decision.reason, "aggregate-budget-demotion")
        self.assertEqual(decision.active_after_bytes, 828)
        self.assertFalse(decision.defer_for_budget)

    def test_short_or_explicit_host_request_uses_native_hot_set(self):
        short = decide_elastic_admission(
            submission("short", prompt=20, output=20), active_bytes=0, config=self.config
        )
        explicit = decide_elastic_admission(
            submission("host", residence="host_backed"),
            active_bytes=0,
            config=self.config,
        )

        self.assertEqual(short.chosen_residence, "host_backed")
        self.assertEqual(short.reason, "fits-native-hot-set")
        self.assertEqual(explicit.chosen_residence, "host_backed")
        self.assertEqual(explicit.reason, "planned-host-backed")

    def test_auto_mode_uses_known_output_length(self):
        short = decide_elastic_admission(
            submission("short", output=32, residence="auto"),
            active_bytes=0,
            config=self.config,
        )
        long = decide_elastic_admission(
            submission("long", output=128, residence="auto"),
            active_bytes=0,
            config=self.config,
        )

        self.assertEqual(short.chosen_residence, "host_backed")
        self.assertEqual(long.chosen_residence, "resident")

    def test_requests_wait_when_even_hot_set_exceeds_budget(self):
        decision = decide_elastic_admission(
            submission("c"), active_bytes=980, config=self.config
        )

        self.assertEqual(decision.chosen_residence, "host_backed")
        self.assertTrue(decision.defer_for_budget)
        self.assertTrue(decision.over_budget)

    def test_invalid_active_bytes_are_rejected(self):
        for value in (-1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "active_bytes"):
                    decide_elastic_admission(
                        submission("bad"), active_bytes=value, config=self.config
                    )

    def test_planner_config_round_trips_into_live_budget(self):
        planner = PlannerConfig(
            memory_budget_bytes=self.config.memory_budget_bytes,
            kv_bytes_per_token=self.config.kv_bytes_per_token,
            prefill_workspace_bytes_per_token=0,
            sparse_hot_tokens=self.config.sparse_hot_tokens,
            auto_resident_output_threshold=self.config.auto_resident_output_tokens,
        )
        live = elastic_config_from_planner(planner)

        self.assertEqual(live, self.config)


if __name__ == "__main__":
    unittest.main()
