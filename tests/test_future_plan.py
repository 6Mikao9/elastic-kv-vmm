import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from adaptive_sparse.future_plan import (
    DecodeBatchProfile,
    KindServiceProfile,
    PlannerConfig,
    PlanResult,
    RequestProfile,
    ResidencyMode,
    RuntimeCapabilities,
    compare,
    main,
    plan_rollout,
    simulate,
    validate_runtime_capabilities,
)
from adaptive_sparse.sglang_future_hook import FutureHint


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

    def test_oracle_beats_fifo_and_explores_both_residency_modes(self):
        fifo = simulate(self.workload, self.config, "fifo")
        oracle = simulate(self.workload, self.config, "oracle")

        self.assertGreater(fifo.deadline_misses, 0)
        self.assertEqual(oracle.deadline_misses, 0)
        self.assertLess(
            (oracle.deadline_misses, oracle.weighted_completion_ms),
            (fifo.deadline_misses, fifo.weighted_completion_ms),
        )
        self.assertEqual(
            {item.mode for item in oracle.admissions},
            {ResidencyMode.RESIDENT, ResidencyMode.HI_SPARSE},
        )
        self.assertGreater(oracle.search_expansions, 0)
        self.assertFalse(oracle.search_truncated)
        self.assertTrue(oracle.search_exact)

    def test_oracle_can_intentionally_idle_to_protect_deadline(self):
        config = PlannerConfig(
            memory_budget_bytes=20_000_000,
            kv_bytes_per_token=1000,
            prefill_workspace_bytes_per_token=0,
            sparse_hot_tokens=200,
            prefill_tokens_per_ms=1,
            resident_decode_tokens_per_ms=1,
            sparse_decode_tokens_per_ms=1,
            max_active_requests=2,
            lookahead_ms=50,
        )
        workload = [
            RequestProfile(
                "background",
                arrival_ms=0,
                prompt_tokens=80,
                output_tokens=10,
            ),
            RequestProfile(
                "tool",
                arrival_ms=10,
                prompt_tokens=10,
                output_tokens=100,
                priority=2,
                deadline_ms=130,
            ),
        ]

        oracle = simulate(workload, config, "oracle")

        self.assertEqual(oracle.deadline_misses, 0)
        by_id = {item.request_id: item for item in oracle.admissions}
        self.assertEqual(by_id["tool"].start_ms, 10)
        self.assertGreaterEqual(by_id["background"].start_ms, 10)

    def test_oracle_expansion_cap_returns_deployable_fallback(self):
        bounded = PlannerConfig(
            **{
                **self.config.as_dict(),
                "oracle_max_expansions": 1,
            }
        )
        result = simulate(self.workload, bounded, "oracle")

        self.assertEqual(result.policy, "oracle")
        self.assertTrue(result.search_truncated)
        self.assertFalse(result.search_exact)
        self.assertIn(result.fallback_policy, {"future", "online"})
        self.assertGreater(result.search_preparation_rollouts, 0)
        self.assertLessEqual(
            result.search_preparation_rollouts,
            bounded.online_max_rollouts,
        )
        self.assertLessEqual(result.search_expansions, 1)
        self.assertEqual(len(result.admissions), len(self.workload))
        self.assertLessEqual(result.peak_memory_bytes, bounded.memory_budget_bytes)

    def test_online_rollouts_close_a_mode_choice_gap(self):
        config = PlannerConfig(
            memory_budget_bytes=900_000,
            kv_bytes_per_token=1000,
            prefill_workspace_bytes_per_token=1,
            sparse_hot_tokens=64,
            prefill_tokens_per_ms=8,
            resident_decode_tokens_per_ms=10,
            sparse_decode_tokens_per_ms=16,
            max_active_requests=3,
            lookahead_ms=40,
            auto_resident_output_threshold=32,
            oracle_max_expansions=20_000,
        )
        workload = [
            RequestProfile(
                "r0", 20, 64, 32, priority=1.0, deadline_ms=140
            ),
            RequestProfile(
                "r1", 30, 320, 64, priority=2.0, deadline_ms=110
            ),
            RequestProfile(
                "r2", 50, 192, 32, priority=0.5, deadline_ms=230
            ),
        ]

        future = simulate(workload, config, "future")
        online = simulate(workload, config, "online")
        oracle = simulate(workload, config, "oracle")

        self.assertTrue(oracle.search_exact)
        self.assertLess(
            online.weighted_completion_ms,
            future.weighted_completion_ms,
        )
        self.assertEqual(
            (
                online.deadline_misses,
                online.weighted_completion_ms,
                online.makespan_ms,
            ),
            (
                oracle.deadline_misses,
                oracle.weighted_completion_ms,
                oracle.makespan_ms,
            ),
        )
        self.assertGreater(online.search_expansions, 0)
        self.assertFalse(online.search_truncated)

    def test_online_rollout_budget_falls_back_to_future(self):
        bounded = PlannerConfig(
            **{
                **self.config.as_dict(),
                "online_max_rollouts": 1,
            }
        )
        result = simulate(self.workload, bounded, "online")

        self.assertEqual(result.policy, "online")
        self.assertTrue(result.search_truncated)
        self.assertEqual(result.fallback_policy, "future")
        self.assertEqual(len(result.admissions), len(self.workload))
        self.assertLessEqual(result.peak_memory_bytes, bounded.memory_budget_bytes)

    def test_online_budget_after_commit_labels_hybrid_fallback(self):
        config = PlannerConfig(
            memory_budget_bytes=900_000,
            kv_bytes_per_token=1000,
            prefill_workspace_bytes_per_token=1,
            sparse_hot_tokens=64,
            prefill_tokens_per_ms=8,
            resident_decode_tokens_per_ms=10,
            sparse_decode_tokens_per_ms=16,
            max_active_requests=3,
            lookahead_ms=40,
            auto_resident_output_threshold=32,
            online_max_rollouts=3,
        )
        workload = [
            RequestProfile(
                "r0", 20, 64, 32, priority=1.0, deadline_ms=140
            ),
            RequestProfile(
                "r1", 30, 320, 64, priority=2.0, deadline_ms=110
            ),
            RequestProfile(
                "r2", 50, 192, 32, priority=0.5, deadline_ms=230
            ),
        ]

        result = simulate(workload, config, "online")

        self.assertTrue(result.search_truncated)
        self.assertGreater(result.search_expansions, 0)
        self.assertEqual(
            result.fallback_policy,
            "online_prefix_then_future",
        )
        self.assertEqual(len(result.admissions), len(workload))

    def test_future_policy_can_idle_to_protect_an_imminent_deadline(self):
        workload = [
            RequestProfile(
                "background",
                arrival_ms=0,
                prompt_tokens=5000,
                output_tokens=1000,
                kind="background",
            ),
            RequestProfile(
                "tool",
                arrival_ms=2,
                prompt_tokens=100,
                output_tokens=200,
                kind="tool",
                priority=2,
                deadline_ms=130,
            ),
        ]
        future = simulate(workload, self.config, "future")

        self.assertEqual(future.admissions[0].request_id, "tool")
        self.assertEqual(future.admissions[0].start_ms, 2)
        self.assertEqual(future.deadline_misses, 0)
        background = next(
            item for item in future.admissions if item.request_id == "background"
        )
        self.assertGreaterEqual(background.start_ms, 3)

    def test_idle_forecast_accounts_for_future_decode_residency(self):
        config = PlannerConfig(
            memory_budget_bytes=110,
            kv_bytes_per_token=1,
            prefill_workspace_bytes_per_token=0,
            sparse_hot_tokens=40,
            prefill_tokens_per_ms=5,
            resident_decode_tokens_per_ms=1,
            sparse_decode_tokens_per_ms=1,
            max_active_requests=2,
            lookahead_ms=10,
        )
        workload = [
            RequestProfile(
                "current",
                arrival_ms=0,
                prompt_tokens=60,
                output_tokens=40,
                deadline_ms=100,
                residence_hint="resident",
            ),
            RequestProfile(
                "future",
                arrival_ms=2,
                prompt_tokens=40,
                output_tokens=40,
                priority=2,
                deadline_ms=50,
            ),
        ]
        future = simulate(workload, config, "future")

        self.assertEqual(future.admissions[0].request_id, "current")
        self.assertLessEqual(future.admissions[0].finish_ms, 100)

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

    def test_explicit_host_backed_hint_uses_bounded_hot_set(self):
        request = RequestProfile(
            "background",
            arrival_ms=0,
            prompt_tokens=3000,
            output_tokens=100,
            residence_hint="host_backed",
        )
        result = simulate([request], self.config, "mode-aware")
        self.assertEqual(result.admissions[0].mode, ResidencyMode.HI_SPARSE)
        self.assertEqual(
            result.admissions[0].reason, "hint-requires-bounded-hot-set"
        )

    def test_explicit_resident_hint_claims_an_available_slot(self):
        request = RequestProfile(
            "pinned",
            arrival_ms=0,
            prompt_tokens=3000,
            output_tokens=100,
            residence_hint="resident",
        )
        result = simulate([request], self.config, "mode-aware")
        self.assertEqual(result.admissions[0].mode, ResidencyMode.RESIDENT)
        self.assertEqual(result.admissions[0].reason, "hint-resident")

    def test_explicit_resident_hint_falls_back_only_when_capacity_requires_it(self):
        config = PlannerConfig(
            memory_budget_bytes=2_000_000,
            kv_bytes_per_token=1000,
            prefill_workspace_bytes_per_token=0,
            sparse_hot_tokens=1000,
        )
        request = RequestProfile(
            "pinned",
            arrival_ms=0,
            prompt_tokens=100,
            output_tokens=2000,
            residence_hint="resident",
        )
        result = simulate([request], config, "mode-aware")
        self.assertEqual(result.admissions[0].mode, ResidencyMode.HI_SPARSE)
        self.assertEqual(result.admissions[0].reason, "hint-resident-falls-back")

    def test_auto_hint_uses_output_threshold(self):
        config = PlannerConfig(
            memory_budget_bytes=8_500_000,
            kv_bytes_per_token=1000,
            prefill_workspace_bytes_per_token=0,
            sparse_hot_tokens=2000,
            auto_resident_output_threshold=128,
        )
        request = RequestProfile(
            "short",
            arrival_ms=0,
            prompt_tokens=3000,
            output_tokens=64,
        )
        result = simulate([request], config, "mode-aware")
        self.assertEqual(result.admissions[0].mode, ResidencyMode.HI_SPARSE)
        self.assertEqual(result.admissions[0].reason, "hint-short-auto")

    def test_invalid_residence_hint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid residence_hint"):
            RequestProfile(
                "bad",
                arrival_ms=0,
                prompt_tokens=1,
                output_tokens=1,
                residence_hint="pinned",
            )

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

    def test_resident_shared_prefix_backing_is_counted_once(self):
        config = PlannerConfig(
            memory_budget_bytes=2_600,
            kv_bytes_per_token=1,
            prefill_workspace_bytes_per_token=0,
            sparse_hot_tokens=200,
            prefill_tokens_per_ms=1000,
            resident_decode_tokens_per_ms=2,
            sparse_decode_tokens_per_ms=0.5,
            max_active_requests=2,
        )
        shared = [
            RequestProfile(
                request_id,
                arrival_ms=0,
                prompt_tokens=1500,
                output_tokens=500,
                residence_hint="resident",
                prefix_group="shared-system-prompt",
                shared_prefix_tokens=1500,
            )
            for request_id in ("shared-a", "shared-b")
        ]
        unshared = [
            replace(
                item,
                prefix_group=None,
                shared_prefix_tokens=0,
            )
            for item in shared
        ]

        with_sharing = simulate(shared, config, "mode-aware")
        without_sharing = simulate(unshared, config, "mode-aware")

        self.assertEqual(
            {item.mode for item in with_sharing.admissions},
            {ResidencyMode.RESIDENT},
        )
        self.assertEqual(with_sharing.peak_memory_bytes, 2500)
        self.assertEqual(with_sharing.peak_shared_prefix_credit_bytes, 1500)
        self.assertIn(
            ResidencyMode.HI_SPARSE,
            {item.mode for item in without_sharing.admissions},
        )
        self.assertEqual(without_sharing.peak_shared_prefix_credit_bytes, 0)
        self.assertLess(with_sharing.makespan_ms, without_sharing.makespan_ms)

    def test_inconsistent_shared_prefix_group_is_rejected(self):
        workload = [
            RequestProfile(
                "a",
                arrival_ms=0,
                prompt_tokens=100,
                output_tokens=100,
                prefix_group="group",
                shared_prefix_tokens=50,
            ),
            RequestProfile(
                "b",
                arrival_ms=0,
                prompt_tokens=100,
                output_tokens=100,
                prefix_group="group",
                shared_prefix_tokens=60,
            ),
        ]
        with self.assertRaisesRegex(ValueError, "inconsistent shared prefix"):
            simulate(workload, self.config, "future")

    def test_non_finite_rates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite positive"):
            PlannerConfig(
                memory_budget_bytes=1,
                kv_bytes_per_token=1,
                prefill_workspace_bytes_per_token=0,
                prefill_tokens_per_ms=math.inf,
            )

    def test_future_policy_accounts_for_same_arrival_competition(self):
        workload = [
            RequestProfile(
                "long", arrival_ms=0, prompt_tokens=1000, output_tokens=7000
            ),
            *[
                RequestProfile(
                    f"short-{index}",
                    arrival_ms=0,
                    prompt_tokens=100,
                    output_tokens=400,
                    priority=2,
                    deadline_ms=500,
                )
                for index in range(5)
            ],
        ]
        result = simulate(workload, self.config, "future")
        self.assertEqual(result.admissions[0].mode, ResidencyMode.HI_SPARSE)

    def test_non_finite_priority_is_rejected(self):
        for priority in (math.nan, math.inf, -math.inf):
            with self.subTest(priority=priority), self.assertRaisesRegex(
                ValueError, "priority must be finite"
            ):
                RequestProfile(
                    "bad-priority",
                    arrival_ms=0,
                    prompt_tokens=1,
                    output_tokens=1,
                    priority=priority,
                )

    def test_kind_and_priority_match_runtime_hint_bounds(self):
        for kind in ("", "x" * 65):
            with self.subTest(kind=kind), self.assertRaisesRegex(
                ValueError, "kind must be a short non-empty string"
            ):
                RequestProfile(
                    "bad-kind",
                    arrival_ms=0,
                    prompt_tokens=1,
                    output_tokens=1,
                    kind=kind,
                )
        with self.assertRaisesRegex(ValueError, "priority must be finite"):
            RequestProfile(
                "too-heavy",
                arrival_ms=0,
                prompt_tokens=1,
                output_tokens=1,
                priority=1_000_001,
            )

    def test_request_fields_are_strict_runtime_compatible_types(self):
        invalid = (
            ({"prompt_tokens": 0}, "prompt_tokens must be >= 1"),
            ({"arrival_ms": 0.5}, "arrival_ms must be an integer"),
            ({"prompt_tokens": 1.5}, "prompt_tokens must be an integer"),
            ({"output_tokens": True}, "output_tokens must be an integer"),
            ({"shared_prefix_tokens": False}, "shared_prefix_tokens must be an integer"),
            ({"priority": True}, "priority must be finite"),
            ({"priority": "2"}, "priority must be finite"),
            ({"deadline_ms": 2.5}, "deadline_ms must be an integer"),
            ({"deadline_ms": 86_400_001}, "deadline_ms must be between"),
        )
        base = {
            "request_id": "bad",
            "arrival_ms": 0,
            "prompt_tokens": 1,
            "output_tokens": 1,
        }
        for update, message in invalid:
            with self.subTest(update=update), self.assertRaisesRegex(ValueError, message):
                RequestProfile(**{**base, **update})

    def test_config_fields_are_strict_integer_and_number_types(self):
        with self.assertRaisesRegex(ValueError, "memory_budget_bytes must be an integer"):
            PlannerConfig(
                memory_budget_bytes=1.5,
                kv_bytes_per_token=1,
                prefill_workspace_bytes_per_token=0,
            )
        with self.assertRaisesRegex(ValueError, "lookahead_ms must be an integer"):
            PlannerConfig(
                memory_budget_bytes=1,
                kv_bytes_per_token=1,
                prefill_workspace_bytes_per_token=0,
                lookahead_ms=True,
            )
        with self.assertRaisesRegex(
            ValueError, "active_prefix_prefill_reuse must be a boolean"
        ):
            PlannerConfig(
                memory_budget_bytes=1,
                kv_bytes_per_token=1,
                prefill_workspace_bytes_per_token=0,
                active_prefix_prefill_reuse=1,
            )

    def test_live_resident_prefix_can_reduce_known_future_prefill(self):
        base = PlannerConfig(
            memory_budget_bytes=50_000,
            kv_bytes_per_token=1,
            prefill_workspace_bytes_per_token=0,
            sparse_hot_tokens=100,
            prefill_tokens_per_ms=1,
            resident_decode_tokens_per_ms=100,
            sparse_decode_tokens_per_ms=50,
            max_active_requests=2,
            lookahead_ms=50,
        )
        workload = (
            RequestProfile(
                "anchor",
                arrival_ms=0,
                prompt_tokens=1000,
                output_tokens=10_000,
                kind="chat",
                prefix_group="system-prefix",
                shared_prefix_tokens=1000,
                residence_hint="resident",
            ),
            RequestProfile(
                "reuse",
                arrival_ms=10,
                prompt_tokens=1000,
                output_tokens=10,
                kind="chat",
                prefix_group="system-prefix",
                shared_prefix_tokens=1000,
                residence_hint="resident",
            ),
        )

        disabled = simulate(workload, base, "future")
        enabled = simulate(
            workload,
            replace(base, active_prefix_prefill_reuse=True),
            "future",
        )
        disabled_by_id = {item.request_id: item for item in disabled.admissions}
        enabled_by_id = {item.request_id: item for item in enabled.admissions}

        self.assertFalse(base.active_prefix_prefill_reuse)
        self.assertEqual(disabled_by_id["reuse"].finish_ms, 2001)
        self.assertEqual(enabled_by_id["reuse"].finish_ms, 1002)
        self.assertEqual(
            enabled_by_id["reuse"].finish_ms - disabled_by_id["reuse"].finish_ms,
            -999,
        )
        self.assertEqual(
            enabled_by_id["reuse"].reason,
            "hint-resident+active-prefix-reuse",
        )
        self.assertEqual(enabled_by_id["anchor"], disabled_by_id["anchor"])
        self.assertEqual(enabled.peak_memory_bytes, disabled.peak_memory_bytes)
        self.assertEqual(enabled.peak_memory_bytes, 11_010)
        self.assertEqual(enabled.peak_shared_prefix_credit_bytes, 1000)

    def test_active_prefix_reuse_updates_deadline_urgency_ranking(self):
        config = PlannerConfig(
            memory_budget_bytes=50_000,
            kv_bytes_per_token=1,
            prefill_workspace_bytes_per_token=0,
            sparse_hot_tokens=100,
            prefill_tokens_per_ms=1,
            resident_decode_tokens_per_ms=100,
            sparse_decode_tokens_per_ms=50,
            max_active_requests=3,
            active_prefix_prefill_reuse=True,
        )
        workload = (
            RequestProfile(
                "donor",
                arrival_ms=0,
                prompt_tokens=1000,
                output_tokens=10_000,
                kind="chat",
                prefix_group="system-prefix",
                shared_prefix_tokens=1000,
                residence_hint="resident",
            ),
            RequestProfile(
                "reuse",
                arrival_ms=1000,
                prompt_tokens=1000,
                output_tokens=1000,
                kind="chat",
                deadline_ms=2999,
                prefix_group="system-prefix",
                shared_prefix_tokens=1000,
                residence_hint="resident",
            ),
            RequestProfile(
                "fresh",
                arrival_ms=1000,
                prompt_tokens=1000,
                output_tokens=1000,
                kind="chat",
                deadline_ms=3000,
                residence_hint="resident",
            ),
        )

        disabled = simulate(
            workload,
            replace(config, active_prefix_prefill_reuse=False),
            "mode-aware",
        )
        result = simulate(workload, config, "mode-aware")

        self.assertEqual(
            [item.request_id for item in disabled.admissions],
            ["donor", "reuse", "fresh"],
        )
        self.assertEqual(
            [item.request_id for item in result.admissions],
            ["donor", "fresh", "reuse"],
        )

    def test_kind_profiles_drive_resources_timing_and_round_trip(self):
        config = replace(
            self.config,
            kind_profiles={
                "tool": {
                    "prefill_tokens_per_ms": 4,
                    "sparse_decode_tokens_per_ms": 4,
                    "sparse_hot_tokens": 100,
                    "prefill_workspace_bytes_per_token": 3,
                    "auto_resident_output_threshold": 64,
                }
            },
        )
        restored = PlannerConfig(**config.as_dict())
        request = RequestProfile(
            "tool-request",
            arrival_ms=0,
            prompt_tokens=100,
            output_tokens=100,
            kind="tool",
            residence_hint="host_backed",
        )

        result = simulate((request,), restored, "fifo")
        admission = result.admissions[0]

        self.assertEqual(restored.kind_profiles, config.kind_profiles)
        self.assertEqual(restored.prefill_rate("tool"), 4)
        self.assertEqual(
            restored.decode_rate("tool", ResidencyMode.HI_SPARSE), 4
        )
        self.assertEqual(restored.sparse_hot_tokens_for("tool"), 100)
        self.assertEqual(restored.prefill_workspace_rate("tool"), 3)
        self.assertEqual(restored.auto_resident_threshold("tool"), 64)
        self.assertEqual(restored.profile_for("unknown"), KindServiceProfile())
        self.assertEqual(admission.mode, ResidencyMode.HI_SPARSE)
        self.assertEqual(admission.resident_kv_bytes, 100_000)
        self.assertEqual(admission.prefill_workspace_bytes, 300)
        self.assertEqual(admission.finish_ms - admission.start_ms, 50)
        self.assertFalse(restored.active_prefix_prefill_reuse)

    def test_active_prefix_reuse_round_trips_in_planner_config(self):
        config = replace(self.config, active_prefix_prefill_reuse=True)

        restored = PlannerConfig(**config.as_dict())

        self.assertTrue(restored.active_prefix_prefill_reuse)
        self.assertEqual(restored.as_dict(), config.as_dict())

    def test_measured_batch_profile_changes_coactive_decode_timing(self):
        config = replace(
            self.config,
            memory_budget_bytes=1_000_000,
            kv_bytes_per_token=1,
            sparse_hot_tokens=100,
            max_active_requests=2,
            decode_batch_profiles={
                "1": {
                    "resident_tokens_per_ms": 2,
                    "sparse_tokens_per_ms": 1.6,
                },
                "2": {"resident_tokens_per_ms": 10},
            },
        )
        workload = (
            RequestProfile(
                "anchor",
                arrival_ms=0,
                prompt_tokens=10,
                output_tokens=200,
                residence_hint="resident",
            ),
            RequestProfile(
                "coactive",
                arrival_ms=1,
                prompt_tokens=10,
                output_tokens=200,
                residence_hint="resident",
            ),
        )

        result = simulate(workload, config, "fifo")
        by_id = {item.request_id: item for item in result.admissions}

        self.assertEqual(by_id["anchor"].finish_ms - by_id["anchor"].start_ms, 101)
        self.assertEqual(by_id["coactive"].finish_ms - by_id["coactive"].start_ms, 41)
        self.assertEqual(by_id["anchor"].decode_batch_size, 1)
        self.assertEqual(by_id["coactive"].decode_batch_size, 2)
        rollout = plan_rollout(workload, config, "fifo", simulation=result)
        submission = next(
            item for item in rollout.submissions if item.request_id == "coactive"
        )
        self.assertEqual(submission.decode_batch_size, 2)
        self.assertEqual(submission.hint["decode_tokens_per_s"], 5000)
        self.assertEqual(
            result.as_dict()["admissions"][1]["decode_batch_size"], 2
        )
        self.assertEqual(
            rollout.as_dict()["submissions"][1]["decode_batch_size"], 2
        )
        self.assertEqual(
            config.decode_rate_for_batch(
                "generic", ResidencyMode.RESIDENT, 2
            ),
            5,
        )
        self.assertEqual(
            config.decode_rate_for_batch(
                "generic", ResidencyMode.RESIDENT, 3
            ),
            config.resident_decode_tokens_per_ms,
        )
        restored = PlannerConfig(**config.as_dict())
        self.assertEqual(restored.decode_batch_profiles, config.decode_batch_profiles)
        self.assertIsInstance(
            restored.decode_batch_profiles[0][1], DecodeBatchProfile
        )

    def test_invalid_decode_batch_profiles_are_rejected(self):
        common = {
            "memory_budget_bytes": 1,
            "kv_bytes_per_token": 1,
            "prefill_workspace_bytes_per_token": 0,
        }
        invalid = (
            (
                ((0, {"resident_tokens_per_ms": 1}),),
                "decode batch size",
            ),
            (
                ((True, {"resident_tokens_per_ms": 1}),),
                "integer",
            ),
            (
                ((1, {}),),
                "requires resident or sparse",
            ),
            (
                ((1, {"resident_tokens_per_ms": math.inf}),),
                "finite positive",
            ),
            (
                (("1", {"resident_tokens_per_ms": 1}),
                 (1, {"resident_tokens_per_ms": 2})),
                "duplicate decode batch profile",
            ),
            (
                ((1, {
                    "resident_tokens_per_ms": 1,
                    "kind_profiles": {1: {"resident_tokens_per_ms": 2}},
                }),),
                "keys must be short non-empty strings",
            ),
            (
                ((1, {
                    "resident_tokens_per_ms": 1,
                    "kind_profiles": (("tool", {"resident_tokens_per_ms": 2}),
                                     ("tool", {"resident_tokens_per_ms": 3})),
                }),),
                "duplicate kind decode batch profile",
            ),
            (
                ((1, {
                    "resident_tokens_per_ms": 1,
                    "kind_profiles": {"tool": {}},
                }),),
                "requires resident or sparse",
            ),
        )
        for profiles, message in invalid:
            with self.subTest(profiles=profiles), self.assertRaisesRegex(
                ValueError, message
            ):
                PlannerConfig(**common, decode_batch_profiles=profiles)

    def test_kind_decode_profiles_override_exact_batch_with_safe_fallback(self):
        config = replace(
            self.config,
            decode_batch_profiles={
                4: {
                    "resident_tokens_per_ms": 8,
                    "sparse_tokens_per_ms": 4,
                    "kind_profiles": {
                        "chat": {"resident_tokens_per_ms": 6},
                        "tool": {
                            "resident_tokens_per_ms": 10,
                            "sparse_tokens_per_ms": 3,
                        },
                    },
                }
            },
        )

        restored = PlannerConfig(**config.as_dict())

        self.assertEqual(
            restored.decode_rate_for_batch("chat", ResidencyMode.RESIDENT, 4),
            1.5,
        )
        self.assertEqual(
            restored.decode_rate_for_batch("tool", ResidencyMode.RESIDENT, 4),
            2.5,
        )
        self.assertEqual(
            restored.decode_rate_for_batch(
                "tool", ResidencyMode.HI_SPARSE, 4
            ),
            0.75,
        )
        self.assertEqual(
            restored.decode_rate_for_batch(
                "generic", ResidencyMode.RESIDENT, 4
            ),
            2.0,
        )
        self.assertEqual(
            restored.decode_rate_for_batch("chat", ResidencyMode.HI_SPARSE, 4),
            1.0,
        )
        self.assertEqual(
            restored.decode_rate_for_batch("chat", ResidencyMode.RESIDENT, 3),
            config.resident_decode_tokens_per_ms,
        )
        self.assertEqual(restored.as_dict(), config.as_dict())
        self.assertEqual(
            restored.as_dict()["decode_batch_profiles"]["4"]["kind_profiles"],
            {
                "chat": {"resident_tokens_per_ms": 6.0},
                "tool": {
                    "resident_tokens_per_ms": 10.0,
                    "sparse_tokens_per_ms": 3.0,
                },
            },
        )

    def test_batched_policy_waits_only_when_measured_cohort_is_faster(self):
        config = replace(
            self.config,
            memory_budget_bytes=20_000,
            kv_bytes_per_token=10,
            sparse_hot_tokens=5,
            prefill_tokens_per_ms=10,
            resident_decode_tokens_per_ms=10,
            max_active_requests=2,
            max_batch_requests=2,
            batch_window_ms=10,
            prefill_batch_profiles={2: 30},
            decode_batch_profiles={
                2: {
                    "resident_tokens_per_ms": 30,
                    "sparse_tokens_per_ms": 20,
                }
            },
        )
        workload = (
            RequestProfile(
                "first",
                0,
                100,
                20,
                residence_hint="resident",
            ),
            RequestProfile(
                "second",
                5,
                100,
                20,
                residence_hint="resident",
            ),
        )

        fifo = simulate(workload, config, "fifo")
        batched = simulate(workload, config, "batched")

        self.assertEqual({item.start_ms for item in batched.admissions}, {5})
        self.assertEqual(len({item.cohort_id for item in batched.admissions}), 1)
        self.assertEqual(
            {item.prefill_batch_size for item in batched.admissions}, {2}
        )
        self.assertLess(batched.weighted_completion_ms, fifo.weighted_completion_ms)
        self.assertLess(batched.makespan_ms, fifo.makespan_ms)
        rollout = plan_rollout(
            workload,
            config,
            "batched",
            simulation=batched,
            cohort_namespace="run-a",
        )
        second_rollout = plan_rollout(
            workload,
            config,
            "batched",
            simulation=batched,
            cohort_namespace="run-b",
        )
        runtime_cohorts = {
            item.cohort_id for item in rollout.submissions
        }
        self.assertEqual(len(runtime_cohorts), 1)
        self.assertTrue(next(iter(runtime_cohorts)).endswith("-run-a"))
        self.assertTrue(
            runtime_cohorts.isdisjoint(
                item.cohort_id for item in second_rollout.submissions
            )
        )
        self.assertEqual(rollout.cohort_namespace, "run-a")
        self.assertEqual(
            {item.hint["prefill_tokens_per_s"] for item in rollout.submissions},
            {15_000.0},
        )

    def test_batched_policy_does_not_trade_a_met_deadline_for_a_cohort(self):
        config = replace(
            self.config,
            memory_budget_bytes=20_000,
            kv_bytes_per_token=10,
            sparse_hot_tokens=5,
            prefill_tokens_per_ms=10,
            resident_decode_tokens_per_ms=10,
            max_active_requests=2,
            max_batch_requests=2,
            batch_window_ms=5,
            prefill_batch_profiles={2: 30},
        )
        workload = (
            RequestProfile(
                "deadline",
                0,
                10,
                10,
                deadline_ms=3,
                residence_hint="resident",
            ),
            RequestProfile(
                "later",
                4,
                10,
                10,
                residence_hint="resident",
            ),
        )

        result = simulate(workload, config, "batched")
        by_id = {item.request_id: item for item in result.admissions}

        self.assertEqual(by_id["deadline"].start_ms, 0)
        self.assertLessEqual(by_id["deadline"].finish_ms, 3)
        self.assertEqual(result.deadline_misses, 0)

    def test_batched_policy_idles_without_profiles_to_protect_deadline(self):
        config = PlannerConfig(
            memory_budget_bytes=10_000,
            kv_bytes_per_token=1,
            prefill_workspace_bytes_per_token=0,
            sparse_hot_tokens=10,
            prefill_tokens_per_ms=1,
            resident_decode_tokens_per_ms=1,
            sparse_decode_tokens_per_ms=1,
            max_active_requests=2,
            max_batch_requests=2,
            batch_window_ms=50,
            lookahead_ms=50,
        )
        workload = (
            RequestProfile(
                "background",
                arrival_ms=0,
                prompt_tokens=80,
                output_tokens=10,
            ),
            RequestProfile(
                "deadline",
                arrival_ms=10,
                prompt_tokens=10,
                output_tokens=10,
                priority=2,
                deadline_ms=30,
            ),
        )

        result = simulate(workload, config, "batched")
        by_id = {item.request_id: item for item in result.admissions}

        self.assertEqual(result.deadline_misses, 0)
        self.assertEqual(by_id["deadline"].start_ms, 10)
        self.assertLessEqual(by_id["deadline"].finish_ms, 30)

    def test_batched_whole_cohort_cannot_delay_known_deadline(self):
        config = PlannerConfig(
            memory_budget_bytes=10_000,
            kv_bytes_per_token=1,
            prefill_workspace_bytes_per_token=0,
            sparse_hot_tokens=10,
            prefill_tokens_per_ms=1,
            resident_decode_tokens_per_ms=1,
            sparse_decode_tokens_per_ms=1,
            max_active_requests=4,
            max_batch_requests=3,
            batch_window_ms=50,
            lookahead_ms=50,
        )
        workload = (
            *(
                RequestProfile(
                    f"background-{index}",
                    arrival_ms=0,
                    prompt_tokens=8,
                    output_tokens=10,
                )
                for index in range(3)
            ),
            RequestProfile(
                "deadline",
                arrival_ms=10,
                prompt_tokens=10,
                output_tokens=10,
                priority=2,
                deadline_ms=30,
            ),
        )

        result = simulate(workload, config, "batched")
        by_id = {item.request_id: item for item in result.admissions}

        self.assertEqual(result.deadline_misses, 0)
        self.assertEqual(by_id["deadline"].start_ms, 10)
        self.assertLessEqual(by_id["deadline"].finish_ms, 30)
        self.assertEqual(by_id["deadline"].prefill_batch_size, 1)

    def test_missing_prefill_profile_never_synthesizes_batch_benefit(self):
        config = replace(
            self.config,
            memory_budget_bytes=100_000,
            kv_bytes_per_token=10,
            sparse_hot_tokens=20,
            max_active_requests=2,
            max_batch_requests=2,
            batch_window_ms=10,
        )
        workload = (
            RequestProfile(
                "first", 0, 100, 20, residence_hint="resident"
            ),
            RequestProfile(
                "second", 5, 100, 20, residence_hint="resident"
            ),
        )

        result = simulate(workload, config, "batched")
        by_id = {item.request_id: item for item in result.admissions}

        self.assertEqual(by_id["first"].start_ms, 0)
        self.assertIsNone(by_id["first"].cohort_id)
        self.assertIsNone(by_id["second"].cohort_id)

    def test_batched_ready_set_stays_serial_without_exact_cohort_profile(self):
        config = replace(
            self.config,
            max_batch_requests=2,
            batch_window_ms=0,
        )
        workload = (
            RequestProfile("first", 0, 100, 20),
            RequestProfile("second", 0, 100, 20),
        )

        result = simulate(workload, config, "batched")

        self.assertEqual(result.deadline_misses, 0)
        self.assertEqual({item.prefill_batch_size for item in result.admissions}, {1})
        self.assertEqual({item.cohort_id for item in result.admissions}, {None})

    def test_batched_joint_placement_can_mix_resident_and_hisparse(self):
        config = replace(
            self.config,
            memory_budget_bytes=350,
            kv_bytes_per_token=10,
            prefill_workspace_bytes_per_token=0,
            sparse_hot_tokens=10,
            max_active_requests=2,
            max_batch_requests=2,
            batch_window_ms=0,
        )
        workload = (
            RequestProfile(
                "resident", 0, 10, 10, residence_hint="resident"
            ),
            RequestProfile(
                "sparse", 0, 10, 10, residence_hint="resident"
            ),
        )

        result = simulate(workload, config, "batched")
        by_id = {item.request_id: item for item in result.admissions}

        self.assertEqual(by_id["resident"].mode, ResidencyMode.RESIDENT)
        self.assertEqual(by_id["sparse"].mode, ResidencyMode.HI_SPARSE)
        self.assertEqual(len(result.admissions), 2)
        self.assertLessEqual(result.peak_memory_bytes, config.memory_budget_bytes)

    def test_batched_deadline_deferral_advances_a_safe_ready_subset(self):
        config = PlannerConfig(
            memory_budget_bytes=100_000,
            kv_bytes_per_token=1000,
            prefill_workspace_bytes_per_token=0,
            sparse_hot_tokens=10,
            prefill_tokens_per_ms=1,
            resident_decode_tokens_per_ms=1,
            sparse_decode_tokens_per_ms=1,
            max_active_requests=3,
            max_batch_requests=3,
            batch_window_ms=10,
            lookahead_ms=10,
        )
        workload = (
            RequestProfile(
                "foreground",
                arrival_ms=0,
                prompt_tokens=5,
                output_tokens=5,
                priority=2,
                residence_hint="resident",
            ),
            RequestProfile(
                "long-background",
                arrival_ms=0,
                prompt_tokens=60,
                output_tokens=100,
                priority=1,
                residence_hint="resident",
            ),
            RequestProfile(
                "deadline",
                arrival_ms=5,
                prompt_tokens=5,
                output_tokens=10,
                priority=3,
                deadline_ms=20,
                residence_hint="resident",
            ),
        )

        result = simulate(workload, config, "batched")
        by_id = {item.request_id: item for item in result.admissions}

        self.assertEqual(result.deadline_misses, 0)
        self.assertEqual(by_id["foreground"].start_ms, 0)
        self.assertLessEqual(by_id["deadline"].finish_ms, 20)
        self.assertIsNone(by_id["foreground"].cohort_id)
        self.assertIsNone(by_id["long-background"].cohort_id)

    def test_batched_config_round_trips_and_rejects_invalid_values(self):
        config = replace(
            self.config,
            batch_window_ms=7,
            max_batch_requests=3,
            prefill_batch_profiles={"1": 10, "3": 24},
        )

        restored = PlannerConfig(**config.as_dict())

        self.assertEqual(restored.prefill_batch_profiles, ((1, 10), (3, 24)))
        self.assertEqual(restored.prefill_rate_for_batch("generic", 1), 10)
        self.assertEqual(restored.prefill_rate_for_batch("generic", 3), 8)
        self.assertEqual(restored.prefill_rate_for_batch("missing", 2), 1000)
        self.assertEqual(restored.as_dict(), config.as_dict())

        common = {
            "memory_budget_bytes": 1,
            "kv_bytes_per_token": 1,
            "prefill_workspace_bytes_per_token": 0,
            "max_active_requests": 2,
        }
        with self.assertRaisesRegex(ValueError, "batch_window_ms"):
            PlannerConfig(**common, batch_window_ms=-1)
        with self.assertRaisesRegex(ValueError, "max_batch_requests"):
            PlannerConfig(**common, max_batch_requests=0)
        with self.assertRaisesRegex(ValueError, "max_batch_requests"):
            PlannerConfig(**common, max_batch_requests=3)
        for profiles, message in (
            ({"0": 1}, "prefill batch size"),
            ({True: 1}, "integer"),
            ({1: 0}, "finite positive"),
            ({1: math.inf}, "finite positive"),
            ((1, 2, 3), "entries"),
            ((("1", 1), (1, 2)), "duplicate"),
        ):
            with self.subTest(profiles=profiles), self.assertRaisesRegex(
                ValueError, message
            ):
                PlannerConfig(**common, prefill_batch_profiles=profiles)

    def test_active_prefix_reuse_requires_explicit_radix_capability(self):
        config = replace(self.config, active_prefix_prefill_reuse=True)
        workload = (
            RequestProfile(
                "shared",
                arrival_ms=0,
                prompt_tokens=100,
                output_tokens=100,
                prefix_group="system",
                shared_prefix_tokens=80,
            ),
        )
        simulation = simulate(workload, config, "fifo")

        with self.assertRaisesRegex(ValueError, "radix cache"):
            plan_rollout(workload, config, "fifo", simulation=simulation)
        with self.assertRaisesRegex(ValueError, "radix cache"):
            validate_runtime_capabilities(
                config, RuntimeCapabilities(radix_cache_enabled=False)
            )

        plan = plan_rollout(
            workload,
            config,
            "fifo",
            simulation=simulation,
            runtime_capabilities=RuntimeCapabilities(radix_cache_enabled=True),
        )
        self.assertEqual(
            plan.as_dict()["runtime_capabilities"],
            {"radix_cache_enabled": True},
        )

    def test_rollout_hint_uses_selected_kind_and_residency_rates(self):
        config = replace(
            self.config,
            kind_profiles={
                "tool": {
                    "prefill_tokens_per_ms": 4,
                    "sparse_decode_tokens_per_ms": 3,
                }
            },
        )
        request = RequestProfile(
            "tool-request",
            arrival_ms=0,
            prompt_tokens=10,
            output_tokens=200,
            kind="tool",
            residence_hint="host_backed",
        )

        hint = plan_rollout((request,), config, "fifo").submissions[0].hint

        self.assertEqual(hint["residence"], "host_backed")
        self.assertEqual(hint["prefill_tokens_per_s"], 4000)
        self.assertEqual(hint["decode_tokens_per_s"], 3000)

    def test_invalid_kind_profiles_are_rejected(self):
        common = {
            "memory_budget_bytes": 1,
            "kv_bytes_per_token": 1,
            "prefill_workspace_bytes_per_token": 0,
        }
        invalid = (
            ({"tool": {"sparse_hot_tokens": 0}}, "sparse_hot_tokens"),
            ({"tool": {"prefill_tokens_per_ms": math.inf}}, "finite positive"),
            ({"": {}}, "short non-empty"),
            ((('tool', {}), ('tool', {})), "duplicate kind profile"),
            ((('tool', 1),), "must be an object"),
        )
        for profiles, message in invalid:
            with self.subTest(profiles=profiles), self.assertRaisesRegex(
                ValueError, message
            ):
                PlannerConfig(**common, kind_profiles=profiles)

    def test_rollout_plan_emits_controlled_submissions_and_runtime_hints(self):
        plan = plan_rollout(self.workload, self.config, "future")
        submissions = {item.request_id: item for item in plan.submissions}

        self.assertEqual(len(plan.submissions), len(self.workload))
        self.assertLessEqual(
            plan.simulation.peak_memory_bytes, self.config.memory_budget_bytes
        )
        for item in plan.submissions:
            self.assertGreaterEqual(item.submit_at_ms, item.planned_arrival_ms)
            self.assertEqual(item.hint["release_after_ms"], 0)
            self.assertEqual(
                item.hint["residence"],
                "resident"
                if next(
                    admission.mode
                    for admission in plan.simulation.admissions
                    if admission.request_id == item.request_id
                ) is ResidencyMode.RESIDENT
                else "host_backed",
            )

        long = submissions["long"]
        self.assertEqual(long.kind, "generic")
        self.assertEqual(long.hint["expected_output_tokens"], 7000)
        self.assertNotIn("deadline_after_ms", long.hint)

        short = submissions["short-0"]
        self.assertEqual(short.kind, "tool")
        self.assertEqual(short.hint["weight"], 2.0)
        self.assertEqual(
            short.hint["deadline_after_ms"],
            max(0, 500 - short.submit_at_ms),
        )
        for item in plan.submissions:
            FutureHint.from_payload(item.hint)

    def test_rollout_plan_carries_shared_prefix_identity(self):
        workload = [
            RequestProfile(
                "shared",
                arrival_ms=0,
                prompt_tokens=100,
                output_tokens=100,
                prefix_group="system",
                shared_prefix_tokens=80,
            )
        ]
        submission = plan_rollout(workload, self.config, "future").submissions[0]

        self.assertEqual(submission.hint["prefix_group"], "system")
        self.assertEqual(submission.hint["shared_prefix_tokens"], 80)
        parsed = FutureHint.from_payload(submission.hint)
        self.assertEqual(parsed.prefix_group, "system")
        self.assertEqual(parsed.shared_prefix_tokens, 80)

    def test_rollout_plan_is_json_serializable_and_rejects_unknown_policy(self):
        payload = plan_rollout(self.workload, self.config, "future").as_dict()
        rendered = json.dumps(payload, sort_keys=True)
        self.assertIn('"submissions"', rendered)
        self.assertIn('"kind": "tool"', rendered)
        self.assertEqual(
            payload["planner_config"]["memory_budget_bytes"],
            self.config.memory_budget_bytes,
        )
        self.assertEqual(
            payload["planner_config"]["sparse_hot_tokens"],
            self.config.sparse_hot_tokens,
        )
        with self.assertRaisesRegex(ValueError, "unknown planning policy"):
            plan_rollout(self.workload, self.config, "missing")

    def test_compare_and_rollout_support_oracle(self):
        raw = {
            "config": self.config.as_dict(),
            "requests": [
                {
                    "request_id": item.request_id,
                    "arrival_ms": item.arrival_ms,
                    "prompt_tokens": item.prompt_tokens,
                    "output_tokens": item.output_tokens,
                    "kind": item.kind,
                    "priority": item.priority,
                    "deadline_ms": item.deadline_ms,
                }
                for item in self.workload
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            workload = Path(directory) / "workload.json"
            workload.write_text(json.dumps(raw), encoding="utf-8")
            comparisons = compare(workload, ("fifo", "oracle"))
        self.assertEqual(set(comparisons), {"fifo", "oracle"})
        self.assertEqual(comparisons["oracle"]["policy"], "oracle")
        self.assertTrue(comparisons["oracle"]["oracle_gap"]["is_oracle"])
        self.assertEqual(
            comparisons["oracle"]["oracle_gap"]["makespan_abs_ms"], 0
        )
        self.assertEqual(
            comparisons["fifo"]["oracle_gap"]["makespan_abs_ms"],
            comparisons["fifo"]["makespan_ms"] - comparisons["oracle"]["makespan_ms"],
        )
        oracle_simulation = simulate(self.workload, self.config, "oracle")
        self.assertEqual(
            len(
                plan_rollout(
                    self.workload,
                    self.config,
                    "oracle",
                    simulation=oracle_simulation,
                ).submissions
            ),
            len(self.workload),
        )

    def test_planner_priority_is_emitted_only_as_a_soft_hint_weight(self):
        rollout = plan_rollout(self.workload, self.config, "future")
        tool = next(
            item for item in rollout.submissions if item.request_id == "short-0"
        )

        self.assertEqual(tool.hint["weight"], 2)
        self.assertNotIn("priority", tool.hint)

    def test_cli_can_emit_a_selected_rollout(self):
        raw = {
            "config": {
                "memory_budget_bytes": self.config.memory_budget_bytes,
                "kv_bytes_per_token": self.config.kv_bytes_per_token,
                "prefill_workspace_bytes_per_token": (
                    self.config.prefill_workspace_bytes_per_token
                ),
                "sparse_hot_tokens": self.config.sparse_hot_tokens,
                "prefill_tokens_per_ms": self.config.prefill_tokens_per_ms,
                "resident_decode_tokens_per_ms": (
                    self.config.resident_decode_tokens_per_ms
                ),
                "sparse_decode_tokens_per_ms": self.config.sparse_decode_tokens_per_ms,
                "max_active_requests": self.config.max_active_requests,
                "lookahead_ms": self.config.lookahead_ms,
            },
            "requests": [
                {
                    "request_id": item.request_id,
                    "arrival_ms": item.arrival_ms,
                    "prompt_tokens": item.prompt_tokens,
                    "output_tokens": item.output_tokens,
                    "kind": item.kind,
                    "priority": item.priority,
                    "deadline_ms": item.deadline_ms,
                }
                for item in self.workload
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            workload = Path(directory) / "workload.json"
            output = Path(directory) / "plan.json"
            workload.write_text(json.dumps(raw), encoding="utf-8")
            self.assertEqual(
                main([
                    str(workload), "--policies", "fifo",
                    "--emit-policy", "future",
                    "--output", str(output),
                ]),
                0,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["rollout"]["policy"], "future")
        self.assertEqual(set(payload) - {"rollout"}, {"fifo", "future"})
        self.assertEqual(
            payload["rollout"]["planner_config"]["kv_bytes_per_token"],
            self.config.kv_bytes_per_token,
        )
        self.assertEqual(
            len(payload["rollout"]["submissions"]), len(self.workload)
        )


if __name__ == "__main__":
    unittest.main()
