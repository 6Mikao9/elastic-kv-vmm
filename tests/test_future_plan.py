import json
import math
from pathlib import Path
import tempfile
import unittest

from adaptive_sparse.future_plan import (
    PlanResult,
    PlannerConfig,
    RequestProfile,
    ResidencyMode,
    compare,
    main,
    plan_rollout,
    simulate,
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
        self.assertEqual(result.fallback_policy, "future")
        self.assertEqual(len(result.admissions), len(self.workload))
        self.assertLessEqual(result.peak_memory_bytes, bounded.memory_budget_bytes)

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
