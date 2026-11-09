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
    weight=1.0,
    submit_at_ms=0,
    estimated_finish_ms=100,
    deadline_after_ms=None,
    prefix_group=None,
    shared_prefix_tokens=0,
    kind="rollout",
    prefill_tokens_per_s=None,
    decode_tokens_per_s=None,
):
    hint = {
        "release_after_ms": 0,
        "expected_output_tokens": output,
        "weight": weight,
        "kind": kind,
        "residence": residence,
    }
    if deadline_after_ms is not None:
        hint["deadline_after_ms"] = deadline_after_ms
    if prefix_group is not None:
        hint["prefix_group"] = prefix_group
        hint["shared_prefix_tokens"] = shared_prefix_tokens
    if prefill_tokens_per_s is not None:
        hint["prefill_tokens_per_s"] = prefill_tokens_per_s
    if decode_tokens_per_s is not None:
        hint["decode_tokens_per_s"] = decode_tokens_per_s
    return RolloutSubmission(
        request_id=request_id,
        planned_arrival_ms=0,
        submit_at_ms=submit_at_ms,
        estimated_finish_ms=estimated_finish_ms,
        kind=kind,
        prompt_tokens=prompt,
        output_tokens=output,
        hint=hint,
        deadline_ms=(
            None
            if deadline_after_ms is None
            else submit_at_ms + deadline_after_ms
        ),
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

    def test_prefill_workspace_can_demote_before_persistent_kv_overflows(self):
        config = ElasticKVConfig(
            memory_budget_bytes=1_000,
            kv_bytes_per_token=2,
            sparse_hot_tokens=64,
            prefill_workspace_bytes_per_token=2,
        )
        decision = decide_elastic_admission(
            submission("long", prompt=100, output=100),
            active_bytes=410,
            config=config,
        )

        self.assertEqual(decision.chosen_residence, "host_backed")
        self.assertEqual(decision.reason, "prefill-headroom-demotion")
        self.assertEqual(decision.active_after_bytes, 538)
        self.assertEqual(decision.prefill_workspace_bytes, 200)
        self.assertEqual(decision.prefill_peak_bytes, 738)
        self.assertFalse(decision.prefill_over_budget)
        self.assertFalse(decision.over_budget)
        self.assertFalse(decision.defer_for_budget)

    def test_prefill_workspace_waits_when_sparse_peak_also_overflows(self):
        config = ElasticKVConfig(
            memory_budget_bytes=1_000,
            kv_bytes_per_token=2,
            sparse_hot_tokens=64,
            prefill_workspace_bytes_per_token=2,
        )
        decision = decide_elastic_admission(
            submission("long", prompt=100, output=100),
            active_bytes=700,
            config=config,
        )

        self.assertEqual(decision.chosen_residence, "host_backed")
        self.assertEqual(decision.prefill_peak_bytes, 1_028)
        self.assertTrue(decision.prefill_over_budget)
        self.assertFalse(decision.over_budget)
        self.assertFalse(decision.future_over_budget)
        self.assertTrue(decision.future_peak_over_budget)
        self.assertTrue(decision.defer_for_budget)

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

    def test_imminent_urgent_request_demotes_current_resident_request(self):
        current = submission("background", weight=1.0, estimated_finish_ms=100)
        urgent = submission(
            "urgent",
            weight=2.0,
            submit_at_ms=20,
            estimated_finish_ms=120,
            deadline_after_ms=180,
        )
        config = ElasticKVConfig(
            memory_budget_bytes=700,
            kv_bytes_per_token=2,
            sparse_hot_tokens=64,
            lookahead_ms=200,
        )

        decision = decide_elastic_admission(
            current,
            active_bytes=0,
            config=config,
            future_submissions=(current, urgent),
            now_ms=0,
        )

        self.assertEqual(decision.chosen_residence, "host_backed")
        self.assertEqual(decision.reason, "imminent-future-demotion")
        self.assertEqual(decision.future_reserved_bytes, 400)
        self.assertEqual(decision.future_reserved_requests, 1)
        self.assertEqual(decision.active_after_bytes, 128)
        self.assertEqual(decision.future_effective_after_bytes, 528)
        self.assertFalse(decision.defer_for_budget)

    def test_future_reservation_includes_future_prefill_workspace(self):
        current = submission("background", weight=1.0, estimated_finish_ms=100)
        urgent = submission(
            "urgent",
            weight=2.0,
            submit_at_ms=20,
            estimated_finish_ms=120,
            deadline_after_ms=180,
        )
        config = ElasticKVConfig(
            memory_budget_bytes=950,
            kv_bytes_per_token=2,
            sparse_hot_tokens=64,
            prefill_workspace_bytes_per_token=1,
            lookahead_ms=200,
        )

        decision = decide_elastic_admission(
            current,
            active_bytes=0,
            config=config,
            future_submissions=(current, urgent),
            now_ms=0,
        )

        self.assertEqual(decision.chosen_residence, "host_backed")
        self.assertEqual(decision.reason, "imminent-future-prefill-demotion")
        self.assertEqual(decision.future_reserved_bytes, 400)
        self.assertEqual(decision.future_prefill_bytes, 100)
        self.assertEqual(decision.future_effective_after_bytes, 528)
        self.assertEqual(decision.future_effective_peak_bytes, 728)
        self.assertFalse(decision.defer_for_budget)

    def test_calibrated_fast_completion_removes_nonoverlapping_future_guard(self):
        current = submission("background", weight=1.0, estimated_finish_ms=100)
        urgent = submission(
            "urgent",
            weight=2.0,
            submit_at_ms=50,
            estimated_finish_ms=150,
        )
        config = ElasticKVConfig(
            memory_budget_bytes=700,
            kv_bytes_per_token=2,
            sparse_hot_tokens=64,
            lookahead_ms=700,
        )

        static = decide_elastic_admission(
            current,
            active_bytes=0,
            config=config,
            future_submissions=(current, urgent),
            now_ms=0,
        )
        calibrated = decide_elastic_admission(
            current,
            active_bytes=0,
            config=config,
            future_submissions=(current, urgent),
            now_ms=0,
            prefill_tokens_per_s=1_000_000,
            decode_tokens_per_s=1_000_000,
        )

        self.assertEqual(static.reason, "imminent-future-demotion")
        self.assertEqual(static.future_reserved_requests, 1)
        self.assertFalse(static.service_time_feedback_applied)
        self.assertEqual(static.service_time_current_duration_ms, 100.0)
        self.assertEqual(static.service_time_calibrated_future_requests, 0)
        self.assertEqual(calibrated.chosen_residence, "resident")
        self.assertEqual(calibrated.future_reserved_requests, 0)
        self.assertTrue(calibrated.service_time_feedback_applied)
        self.assertLessEqual(calibrated.service_time_current_duration_ms, 1.0)

    def test_calibrated_slow_completion_extends_overlap_and_reserves_future_kv(self):
        current = submission("background", weight=1.0, estimated_finish_ms=10)
        urgent = submission(
            "urgent",
            weight=2.0,
            submit_at_ms=50,
            estimated_finish_ms=100,
        )
        config = ElasticKVConfig(
            memory_budget_bytes=700,
            kv_bytes_per_token=2,
            sparse_hot_tokens=64,
            lookahead_ms=700,
        )

        static = decide_elastic_admission(
            current,
            active_bytes=0,
            config=config,
            future_submissions=(current, urgent),
            now_ms=0,
        )
        calibrated = decide_elastic_admission(
            current,
            active_bytes=0,
            config=config,
            future_submissions=(current, urgent),
            now_ms=0,
            prefill_tokens_per_s=10,
            decode_tokens_per_s=10,
        )

        self.assertEqual(static.chosen_residence, "resident")
        self.assertEqual(static.future_reserved_requests, 0)
        self.assertEqual(calibrated.reason, "imminent-future-demotion")
        self.assertEqual(calibrated.future_reserved_bytes, 400)
        self.assertEqual(calibrated.future_reserved_requests, 1)
        self.assertEqual(calibrated.service_time_calibrated_future_requests, 1)

    def test_future_prefill_only_pressure_gets_prefill_reason(self):
        current = submission("current", prompt=1, output=199, weight=1.0)
        urgent = submission("urgent", weight=2.0, submit_at_ms=20)
        config = ElasticKVConfig(
            memory_budget_bytes=850,
            kv_bytes_per_token=2,
            sparse_hot_tokens=64,
            prefill_workspace_bytes_per_token=1,
            lookahead_ms=200,
        )

        decision = decide_elastic_admission(
            current,
            active_bytes=0,
            config=config,
            future_submissions=(current, urgent),
            now_ms=0,
        )

        self.assertEqual(decision.reason, "imminent-future-prefill-demotion")
        self.assertEqual(decision.future_reserved_bytes, 400)
        self.assertEqual(decision.future_prefill_bytes, 100)

    def test_future_reservation_ignores_nonoverlap_equal_weight_and_far_work(self):
        current = submission("current", weight=1.0, estimated_finish_ms=50)
        same_weight = submission(
            "same-weight", weight=1.0, submit_at_ms=10, estimated_finish_ms=60
        )
        later = submission(
            "later",
            weight=2.0,
            submit_at_ms=100,
            estimated_finish_ms=150,
            deadline_after_ms=200,
        )
        beyond_horizon = submission(
            "beyond-horizon",
            weight=2.0,
            submit_at_ms=250,
            estimated_finish_ms=300,
            deadline_after_ms=350,
        )
        config = ElasticKVConfig(
            memory_budget_bytes=700,
            kv_bytes_per_token=2,
            sparse_hot_tokens=64,
            lookahead_ms=200,
        )

        decision = decide_elastic_admission(
            current,
            active_bytes=0,
            config=config,
            future_submissions=(current, same_weight, later, beyond_horizon),
            now_ms=0,
        )

        self.assertEqual(decision.chosen_residence, "resident")
        self.assertEqual(decision.future_reserved_bytes, 0)
        self.assertEqual(decision.future_reserved_requests, 0)

    def test_future_reservation_skips_work_with_slack_after_current_finishes(self):
        current = submission("current", weight=1.0, estimated_finish_ms=50)
        urgent = submission(
            "urgent",
            weight=1.0,
            submit_at_ms=20,
            estimated_finish_ms=120,
            deadline_after_ms=300,
        )
        config = ElasticKVConfig(
            memory_budget_bytes=700,
            kv_bytes_per_token=2,
            sparse_hot_tokens=64,
            lookahead_ms=200,
        )

        decision = decide_elastic_admission(
            current,
            active_bytes=0,
            config=config,
            future_submissions=(current, urgent),
            now_ms=0,
        )

        self.assertEqual(decision.chosen_residence, "resident")
        self.assertEqual(decision.future_reserved_bytes, 0)
        self.assertEqual(decision.future_reserved_requests, 0)

    def test_future_reservation_keeps_higher_weight_when_deadline_has_slack(self):
        current = submission("current", weight=1.0, estimated_finish_ms=50)
        urgent = submission(
            "urgent",
            weight=2.0,
            submit_at_ms=20,
            estimated_finish_ms=120,
            deadline_after_ms=300,
        )
        config = ElasticKVConfig(
            memory_budget_bytes=700,
            kv_bytes_per_token=2,
            sparse_hot_tokens=64,
            lookahead_ms=200,
        )

        decision = decide_elastic_admission(
            current,
            active_bytes=0,
            config=config,
            future_submissions=(current, urgent),
            now_ms=0,
        )

        self.assertEqual(decision.chosen_residence, "host_backed")
        self.assertEqual(decision.future_reserved_bytes, 400)
        self.assertEqual(decision.future_reserved_requests, 1)

    def test_future_reservation_keeps_higher_weight_when_deadline_is_infeasible(self):
        current = submission("current", weight=1.0, estimated_finish_ms=50)
        urgent = submission(
            "urgent",
            weight=2.0,
            submit_at_ms=20,
            estimated_finish_ms=120,
            deadline_after_ms=20,
        )
        config = ElasticKVConfig(
            memory_budget_bytes=700,
            kv_bytes_per_token=2,
            sparse_hot_tokens=64,
            lookahead_ms=200,
        )

        decision = decide_elastic_admission(
            current,
            active_bytes=0,
            config=config,
            future_submissions=(current, urgent),
            now_ms=0,
        )

        self.assertEqual(decision.chosen_residence, "host_backed")
        self.assertEqual(decision.future_reserved_bytes, 400)
        self.assertEqual(decision.future_reserved_requests, 1)

    def test_future_reservation_uses_interval_peak_instead_of_horizon_sum(self):
        current = submission(
            "current",
            prompt=1,
            output=199,
            weight=1.0,
            estimated_finish_ms=300,
        )
        disjoint = [
            submission(
                request_id,
                weight=2.0,
                submit_at_ms=start,
                estimated_finish_ms=finish,
                deadline_after_ms=deadline_after,
            )
            for request_id, start, finish, deadline_after in (
                ("early", 20, 120, 250),
                ("late", 150, 190, 185),
            )
        ]
        config = ElasticKVConfig(
            memory_budget_bytes=2_000,
            kv_bytes_per_token=2,
            sparse_hot_tokens=64,
            prefill_workspace_bytes_per_token=1,
            lookahead_ms=200,
        )

        decision = decide_elastic_admission(
            current,
            active_bytes=0,
            config=config,
            future_submissions=(current, *disjoint),
            now_ms=0,
        )

        self.assertEqual(decision.future_reserved_requests, 2)
        self.assertEqual(decision.future_reserved_bytes, 400)
        self.assertEqual(decision.future_prefill_bytes, 100)

    def test_future_reservation_uses_remaining_duration_after_backlog(self):
        current = submission("current", weight=1.0, estimated_finish_ms=300)
        backlogged = submission(
            "backlogged",
            weight=2.0,
            submit_at_ms=20,
            estimated_finish_ms=120,
            deadline_after_ms=400,
        )
        config = ElasticKVConfig(
            memory_budget_bytes=700,
            kv_bytes_per_token=2,
            sparse_hot_tokens=64,
            lookahead_ms=400,
        )

        decision = decide_elastic_admission(
            current,
            active_bytes=0,
            config=config,
            future_submissions=(current, backlogged),
            now_ms=200,
        )

        self.assertEqual(decision.future_reserved_requests, 1)
        self.assertEqual(decision.future_reserved_bytes, 400)

    def test_direct_admission_rejects_shared_prefix_above_prompt(self):
        invalid = submission(
            "invalid-prefix",
            prompt=10,
            prefix_group="shared",
            shared_prefix_tokens=11,
        )

        with self.assertRaisesRegex(ValueError, "shared prefix exceeds prompt"):
            decide_elastic_admission(
                invalid, active_bytes=0, config=self.config
            )

    def test_future_reservation_requires_time_and_fits_planner_projection(self):
        config = ElasticKVConfig(
            memory_budget_bytes=1_000,
            kv_bytes_per_token=2,
            sparse_hot_tokens=64,
            lookahead_ms=200,
        )
        with self.assertRaisesRegex(ValueError, "now_ms"):
            decide_elastic_admission(
                submission("current"),
                active_bytes=0,
                config=config,
                future_submissions=(submission("other"),),
            )

        planner = PlannerConfig(
            memory_budget_bytes=1_000,
            kv_bytes_per_token=2,
            prefill_workspace_bytes_per_token=0,
            sparse_hot_tokens=64,
            lookahead_ms=200,
            auto_resident_output_threshold=128,
        )
        self.assertEqual(elastic_config_from_planner(planner).lookahead_ms, 200)

    def test_future_reservation_credits_shared_resident_prefixes(self):
        current = submission("current", weight=1.0, estimated_finish_ms=100)
        shared = [
            submission(
                request_id,
                weight=2.0,
                submit_at_ms=20,
                estimated_finish_ms=120,
                deadline_after_ms=180,
                prefix_group="shared-prefix",
                shared_prefix_tokens=80,
            )
            for request_id in ("shared-a", "shared-b")
        ]
        config = ElasticKVConfig(
            memory_budget_bytes=1_000,
            kv_bytes_per_token=2,
            sparse_hot_tokens=64,
            lookahead_ms=200,
        )

        decision = decide_elastic_admission(
            current,
            active_bytes=0,
            config=config,
            future_submissions=(current, *shared),
            now_ms=0,
        )

        self.assertEqual(decision.future_reserved_requests, 2)
        self.assertEqual(decision.future_reserved_bytes, 640)

    def test_invalid_active_bytes_are_rejected(self):
        for value in (-1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "active_bytes"):
                    decide_elastic_admission(
                        submission("bad"), active_bytes=value, config=self.config
                    )

    def test_invalid_prefill_workspace_rate_is_rejected(self):
        for value in (-1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "prefill_workspace_bytes_per_token"
                ):
                    ElasticKVConfig(
                        memory_budget_bytes=1_000,
                        kv_bytes_per_token=2,
                        sparse_hot_tokens=64,
                        prefill_workspace_bytes_per_token=value,
                    )

    def test_kind_profile_changes_hot_set_workspace_and_auto_threshold(self):
        config = ElasticKVConfig(
            memory_budget_bytes=2_000,
            kv_bytes_per_token=2,
            sparse_hot_tokens=64,
            auto_resident_output_tokens=128,
            kind_profiles={
                "tool": {
                    "sparse_hot_tokens": 50,
                    "prefill_workspace_bytes_per_token": 3,
                    "auto_resident_output_threshold": 32,
                }
            },
        )

        decision = decide_elastic_admission(
            submission(
                "tool",
                prompt=20,
                output=40,
                residence="auto",
                kind="tool",
            ),
            active_bytes=0,
            config=config,
        )

        self.assertEqual(config.sparse_hot_tokens_for("unknown"), 64)
        self.assertEqual(decision.chosen_residence, "resident")
        self.assertEqual(decision.reason, "auto-long")
        self.assertEqual(decision.prefill_workspace_bytes, 60)

        sparse = decide_elastic_admission(
            submission(
                "tool-sparse",
                prompt=20,
                output=60,
                kind="tool",
                residence="host_backed",
            ),
            active_bytes=0,
            config=config,
        )
        self.assertEqual(sparse.sparse_bytes, 100)

    def test_measured_rate_overrides_hint_and_kind_profile_for_duration(self):
        config = ElasticKVConfig(
            memory_budget_bytes=10_000,
            kv_bytes_per_token=2,
            sparse_hot_tokens=64,
            kind_profiles={
                "tool": {
                    "prefill_tokens_per_ms": 1,
                    "resident_decode_tokens_per_ms": 2,
                }
            },
        )
        request = submission(
            "tool", prompt=100, output=100, kind="tool", residence="resident"
        )

        profile_decision = decide_elastic_admission(
            request,
            active_bytes=0,
            config=config,
        )
        explicit = submission(
            "explicit",
            prompt=100,
            output=100,
            kind="tool",
            residence="resident",
            prefill_tokens_per_s=10_000,
            decode_tokens_per_s=20_000,
        )
        hint_decision = decide_elastic_admission(
            explicit, active_bytes=0, config=config
        )
        measured_decision = decide_elastic_admission(
            explicit,
            active_bytes=0,
            config=config,
            prefill_tokens_per_s=100_000,
            decode_tokens_per_s=100_000,
        )

        self.assertEqual(profile_decision.service_time_current_duration_ms, 150)
        self.assertTrue(profile_decision.service_time_feedback_applied)
        self.assertEqual(hint_decision.service_time_current_duration_ms, 15)
        self.assertEqual(measured_decision.service_time_current_duration_ms, 2.0)

    def test_kind_profiles_survive_live_planner_projection(self):
        planner = PlannerConfig(
            memory_budget_bytes=1_000,
            kv_bytes_per_token=2,
            prefill_workspace_bytes_per_token=0,
            sparse_hot_tokens=64,
            kind_profiles={
                "tool": {
                    "sparse_hot_tokens": 48,
                    "prefill_workspace_bytes_per_token": 4,
                    "prefill_tokens_per_ms": 3,
                }
            },
        )

        live = elastic_config_from_planner(planner)

        self.assertEqual(live.kind_profiles, planner.kind_profiles)
        self.assertEqual(live.sparse_hot_tokens_for("tool"), 48)
        self.assertEqual(live.prefill_workspace_rate("tool"), 4)
        self.assertEqual(live.prefill_rate_tokens_per_s("tool"), 3000)

    def test_invalid_live_kind_profiles_are_rejected(self):
        common = {
            "memory_budget_bytes": 1_000,
            "kv_bytes_per_token": 2,
            "sparse_hot_tokens": 64,
        }
        invalid = (
            ({"tool": {"sparse_hot_tokens": 0}}, "sparse_hot_tokens"),
            ({"tool": {"sparse_decode_tokens_per_ms": -1}}, "positive"),
            ((('tool', {}), ('tool', {})), "duplicate kind profile"),
        )
        for profiles, message in invalid:
            with self.subTest(profiles=profiles), self.assertRaisesRegex(
                ValueError, message
            ):
                ElasticKVConfig(**common, kind_profiles=profiles)

    def test_new_prefill_field_preserves_legacy_positional_argument_order(self):
        config = ElasticKVConfig(1_000, 2, 64, 128, 5, 3)

        self.assertEqual(config.auto_resident_output_tokens, 128)
        self.assertEqual(config.lookahead_ms, 5)
        self.assertEqual(config.max_future_reservations, 3)
        self.assertEqual(config.prefill_workspace_bytes_per_token, 0)

    def test_planner_config_round_trips_into_live_budget(self):
        planner = PlannerConfig(
            memory_budget_bytes=self.config.memory_budget_bytes,
            kv_bytes_per_token=self.config.kv_bytes_per_token,
            prefill_workspace_bytes_per_token=3,
            sparse_hot_tokens=self.config.sparse_hot_tokens,
            auto_resident_output_threshold=self.config.auto_resident_output_tokens,
        )
        live = elastic_config_from_planner(planner)

        self.assertEqual(
            live,
            ElasticKVConfig(
                memory_budget_bytes=self.config.memory_budget_bytes,
                kv_bytes_per_token=self.config.kv_bytes_per_token,
                sparse_hot_tokens=self.config.sparse_hot_tokens,
                prefill_workspace_bytes_per_token=3,
                auto_resident_output_tokens=self.config.auto_resident_output_tokens,
            ),
        )


if __name__ == "__main__":
    unittest.main()
