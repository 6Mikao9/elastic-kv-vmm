import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from adaptive_sparse.elastic_admission import ElasticKVConfig
from adaptive_sparse.future_plan import (
    PlannerConfig,
    RequestProfile,
    RolloutSubmission,
    plan_rollout,
)
from adaptive_sparse.rollout_executor import (
    ReplayClock,
    RolloutExecutionError,
    collect_residency_evidence,
    execute_rollout,
    load_plan,
    make_kv_hints,
)


class FakeClock:
    def __init__(self):
        self.value = 10.0
        self.overshoot_ms = 0.0

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds + self.overshoot_ms / 1000.0


def submission(deadline_after_ms=100):
    return RolloutSubmission(
        request_id="request-1",
        planned_arrival_ms=5,
        submit_at_ms=10,
        estimated_finish_ms=70,
        kind="tool",
        prompt_tokens=32,
        output_tokens=8,
        hint={
            "release_after_ms": 0,
            "deadline_after_ms": deadline_after_ms,
            "expected_output_tokens": 8,
            "weight": 2.0,
            "kind": "tool",
            "residence": "resident",
        },
        deadline_ms=10 + deadline_after_ms,
    )


class RolloutExecutorTest(unittest.TestCase):
    def test_checked_in_known_future_plan_matches_executor_schema(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "results/future-planning/known-future-rollout.json"
        )
        submissions = load_plan(path)

        self.assertEqual(len(submissions), 7)
        self.assertEqual(
            {item.request_id for item in submissions},
            {
                "background-long",
                "tool-0",
                "tool-1",
                "rollout-short",
                "chat-0",
                "tool-2",
                "rollout-long",
            },
        )

    def test_wraps_payload_in_the_runtime_envelope(self):
        envelope = make_kv_hints(submission())

        self.assertEqual(envelope["protocol_version"], "1")
        self.assertEqual(envelope["message_id"], "request-1")
        self.assertEqual(envelope["actions"][0]["action_type"], "elastic.future")
        self.assertEqual(envelope["actions"][0]["payload"]["residence"], "resident")
        adjusted = make_kv_hints(submission(), deadline_after_ms=17)
        self.assertEqual(adjusted["actions"][0]["payload"]["deadline_after_ms"], 17)
        demoted = make_kv_hints(submission(), residence_override="host_backed")
        self.assertEqual(demoted["actions"][0]["payload"]["residence"], "host_backed")

    def test_runtime_budget_demotes_only_the_new_request(self):
        base = submission()
        first = RolloutSubmission(
            request_id="first",
            planned_arrival_ms=0,
            submit_at_ms=0,
            estimated_finish_ms=100,
            kind=base.kind,
            prompt_tokens=100,
            output_tokens=100,
            hint={
                key: value
                for key, value in base.hint.items()
                if key != "deadline_after_ms"
            }
            | {"expected_output_tokens": 100},
        )
        second = RolloutSubmission(
            request_id="second",
            planned_arrival_ms=0,
            submit_at_ms=0,
            estimated_finish_ms=100,
            kind=base.kind,
            prompt_tokens=100,
            output_tokens=100,
            hint={
                key: value
                for key, value in base.hint.items()
                if key != "deadline_after_ms"
            }
            | {"expected_output_tokens": 100},
        )
        first_started = threading.Event()
        seen = []

        def generate(url, spec, timeout, stream):
            seen.append((spec["request_id"], spec["future_hint"]))
            if spec["request_id"] == "first":
                first_started.set()
                time.sleep(0.03)
            else:
                self.assertTrue(first_started.is_set())
            started = time.perf_counter()
            return {
                "_started_perf_s": started,
                "end_to_end_s": 0.01,
                "output_tokens": 100,
                "ttft_s": 0.004,
                "tpot_s": 0.0001,
            }

        result = execute_rollout(
            "http://server/generate",
            [first, second],
            generator=generate,
            elastic_kv_config=ElasticKVConfig(
                memory_budget_bytes=600,
                kv_bytes_per_token=2,
                sparse_hot_tokens=64,
            ),
            poll_interval_ms=1,
        )

        by_id = {row["request_id"]: row for row in result["requests"]}
        self.assertEqual(by_id["first"]["runtime_residence"], "resident")
        self.assertEqual(by_id["second"]["runtime_residence"], "host_backed")
        second_hint = next(
            hint for request_id, hint in seen if request_id == "second"
        )
        self.assertEqual(second_hint["actions"][0]["payload"]["residence"], "host_backed")
        self.assertEqual(result["summary"]["runtime_demotions"], 1)

    def test_managed_rollout_reserves_for_imminent_urgent_request(self):
        background = RolloutSubmission(
            request_id="background",
            planned_arrival_ms=0,
            submit_at_ms=0,
            estimated_finish_ms=100,
            kind="background",
            prompt_tokens=100,
            output_tokens=100,
            hint={
                "release_after_ms": 0,
                "expected_output_tokens": 100,
                "weight": 1.0,
                "kind": "background",
                "residence": "resident",
            },
        )
        urgent = RolloutSubmission(
            request_id="urgent",
            planned_arrival_ms=10,
            submit_at_ms=10,
            estimated_finish_ms=110,
            kind="tool",
            prompt_tokens=100,
            output_tokens=100,
            hint={
                "release_after_ms": 0,
                "deadline_after_ms": 190,
                "expected_output_tokens": 100,
                "weight": 2.0,
                "kind": "tool",
                "residence": "resident",
            },
            deadline_ms=200,
        )
        seen = {}

        def generate(url, spec, timeout, stream):
            seen[spec["request_id"]] = spec["future_hint"]
            started = time.perf_counter()
            return {
                "_started_perf_s": started,
                "end_to_end_s": 0.01,
                "output_tokens": 100,
                "ttft_s": 0.004,
                "tpot_s": 0.0001,
            }

        result = execute_rollout(
            "http://server/generate",
            [background, urgent],
            generator=generate,
            elastic_kv_config=ElasticKVConfig(
                memory_budget_bytes=700,
                kv_bytes_per_token=2,
                sparse_hot_tokens=64,
                lookahead_ms=200,
            ),
            poll_interval_ms=1,
        )

        by_id = {row["request_id"]: row for row in result["requests"]}
        self.assertEqual(by_id["background"]["runtime_residence"], "host_backed")
        self.assertEqual(
            by_id["background"]["elastic_admission_reason"],
            "imminent-future-demotion",
        )
        self.assertEqual(
            seen["background"]["actions"][0]["payload"]["residence"],
            "host_backed",
        )
        self.assertEqual(result["summary"]["future_reservation_demotions"], 1)

    def test_deadline_queue_policy_orders_contended_requests(self):
        base = submission()
        background = RolloutSubmission(
            request_id="background",
            planned_arrival_ms=0,
            submit_at_ms=0,
            estimated_finish_ms=50,
            kind=base.kind,
            prompt_tokens=32,
            output_tokens=8,
            hint={
                key: value
                for key, value in base.hint.items()
                if key != "deadline_after_ms"
            },
        )
        urgent = RolloutSubmission(
            request_id="urgent",
            planned_arrival_ms=0,
            submit_at_ms=0,
            estimated_finish_ms=50,
            kind=base.kind,
            prompt_tokens=32,
            output_tokens=8,
            hint={**base.hint, "deadline_after_ms": 200, "weight": 2.0},
            deadline_ms=200,
        )
        seen = []

        def generate(url, spec, timeout, stream):
            seen.append(spec["request_id"])
            time.sleep(0.01)
            return {
                "_started_perf_s": time.perf_counter(),
                "end_to_end_s": 0.005,
                "output_tokens": 8,
                "ttft_s": 0.002,
                "tpot_s": 0.0001,
            }

        result = execute_rollout(
            "http://server/generate",
            [background, urgent],
            max_inflight=1,
            queue_policy="deadline",
            generator=generate,
            poll_interval_ms=1,
        )

        self.assertEqual(seen, ["urgent", "background"])
        self.assertEqual(result["requests"][0]["request_id"], "urgent")
        self.assertGreater(result["summary"]["max_capacity_delay_ms"], 0)

    def test_executes_at_planned_time_and_measures_deadline(self):
        clock = FakeClock()
        seen = []

        def generate(url, spec, timeout, stream):
            seen.append((url, spec, timeout, stream))
            started = clock.now()
            return {
                "_started_perf_s": started,
                "end_to_end_s": 0.02,
                "output_tokens": 8,
                "ttft_s": 0.005,
                "tpot_s": 0.002,
            }

        result = execute_rollout(
            "http://server/generate",
            [submission(deadline_after_ms=40)],
            timeout_s=3,
            clock=ReplayClock(clock.now, clock.sleep),
            generator=generate,
        )

        row = result["requests"][0]
        self.assertAlmostEqual(row["actual_submit_ms"], 10.0)
        self.assertAlmostEqual(row["actual_finish_ms"], 30.0)
        self.assertTrue(row["actual_deadline_hit"])
        self.assertEqual(result["summary"]["deadline_hits"], 1)
        self.assertEqual(seen[0][1]["future_hint"]["message_id"], "request-1")
        self.assertEqual(seen[0][3], True)

    def test_throughput_includes_planned_initial_idle(self):
        clock = FakeClock()

        def generate(url, spec, timeout, stream):
            return {
                "_started_perf_s": clock.now(),
                "end_to_end_s": 0.02,
                "output_tokens": 8,
                "ttft_s": 0.005,
                "tpot_s": 0.002,
            }

        late_arrival = RolloutSubmission(
            request_id="late-arrival",
            planned_arrival_ms=100,
            submit_at_ms=110,
            estimated_finish_ms=150,
            kind="tool",
            prompt_tokens=32,
            output_tokens=8,
            hint={
                "release_after_ms": 0,
                "deadline_after_ms": 40,
                "expected_output_tokens": 8,
                "weight": 1.0,
                "kind": "tool",
                "residence": "host_backed",
            },
            deadline_ms=150,
        )
        result = execute_rollout(
            "http://server/generate",
            [late_arrival],
            clock=ReplayClock(clock.now, clock.sleep),
            generator=generate,
        )

        efficiency = result["summary"]["efficiency"]
        self.assertAlmostEqual(efficiency["serving_wall_ms"], 20.0)
        self.assertAlmostEqual(efficiency["makespan_ms"], 30.0)
        self.assertAlmostEqual(efficiency["output_tokens_per_s"], 8000.0 / 30.0)
        self.assertAlmostEqual(efficiency["serving_output_tokens_per_s"], 400.0)

    def test_can_measure_an_unhinted_control_arm(self):
        clock = FakeClock()
        seen = []

        def generate(url, spec, timeout, stream):
            seen.append(spec)
            return {
                "_started_perf_s": clock.now(),
                "end_to_end_s": 0.01,
                "output_tokens": 8,
                "ttft_s": 0.004,
                "tpot_s": 0.001,
            }

        execute_rollout(
            "http://server/generate",
            [submission()],
            emit_hints=False,
            clock=ReplayClock(clock.now, clock.sleep),
            generator=generate,
        )

        self.assertNotIn("future_hint", seen[0])

    def test_late_dispatch_fails_loudly(self):
        clock = FakeClock()
        clock.overshoot_ms = 200

        with self.assertRaisesRegex(RolloutExecutionError, "dispatch late"):
            execute_rollout(
                "http://server/generate",
                [submission()],
                max_submit_lateness_ms=50,
                clock=ReplayClock(clock.now, clock.sleep),
                generator=lambda *args: self.fail("request should not be sent"),
            )

    def test_invalid_hint_is_rejected_before_dispatch(self):
        bad = submission()
        bad = RolloutSubmission(
            request_id=bad.request_id,
            planned_arrival_ms=bad.planned_arrival_ms,
            submit_at_ms=bad.submit_at_ms,
            estimated_finish_ms=bad.estimated_finish_ms,
            kind=bad.kind,
            prompt_tokens=bad.prompt_tokens,
            output_tokens=bad.output_tokens,
            hint={**bad.hint, "unknown": 1},
            deadline_ms=bad.deadline_ms,
        )

        with self.assertRaisesRegex(RolloutExecutionError, "invalid rollout hint"):
            execute_rollout(
                "http://server/generate",
                [bad],
                generator=lambda *args: self.fail("request should not be sent"),
            )

    def test_rejects_semantically_inconsistent_plans_before_dispatch(self):
        base = submission()
        cases = (
            (
                {"planned_arrival_ms": 11},
                "submit time precedes planned arrival",
            ),
            (
                {"kind": "chat"},
                "hint kind does not match",
            ),
            (
                {"output_tokens": 9},
                "hint output estimate does not match",
            ),
            (
                {"hint": {**base.hint, "release_after_ms": 1, "deadline_after_ms": 40}},
                "release_after_ms=0",
            ),
        )
        for update, message in cases:
            with self.subTest(message=message):
                arguments = {
                    "request_id": base.request_id,
                    "planned_arrival_ms": base.planned_arrival_ms,
                    "submit_at_ms": base.submit_at_ms,
                    "estimated_finish_ms": base.estimated_finish_ms,
                    "kind": base.kind,
                    "prompt_tokens": base.prompt_tokens,
                    "output_tokens": base.output_tokens,
                    "hint": base.hint,
                    "deadline_ms": base.deadline_ms,
                    **update,
                }
                with self.assertRaisesRegex(RolloutExecutionError, message):
                    execute_rollout(
                        "http://server/generate",
                        [RolloutSubmission(**arguments)],
                        generator=lambda *args: self.fail("request should not be sent"),
                    )

    def test_rejects_zero_parallelism(self):
        with self.assertRaisesRegex(ValueError, "max_parallel must be positive"):
            execute_rollout(
                "http://server/generate",
                [submission()],
                max_parallel=0,
                generator=lambda *args: self.fail("request should not be sent"),
            )

    def test_plan_round_trips_with_execution_inputs(self):
        plan = plan_rollout(
            [
                RequestProfile(
                    "request-1",
                    arrival_ms=0,
                    prompt_tokens=32,
                    output_tokens=8,
                    deadline_ms=100,
                )
            ],
            PlannerConfig(
                memory_budget_bytes=1_000_000,
                kv_bytes_per_token=100,
                prefill_workspace_bytes_per_token=1,
            ),
            "future",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps({"rollout": plan.as_dict()}), encoding="utf-8")
            loaded = load_plan(path)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].prompt_tokens, 32)
        self.assertEqual(loaded[0].output_tokens, 8)
        self.assertEqual(loaded[0].hint["expected_output_tokens"], 8)

    def test_residency_evidence_reconciles_all_tp_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rank in (0, 1):
                payload = {
                    "tp_rank": rank,
                    "request_residency": {
                        "requests": {
                            "request-1": {
                                "actual_residence": "host_backed",
                                "reason": "hint-short-auto",
                                "tp_world_size": 2,
                            }
                        }
                    },
                }
                (root / f"tp-{rank}.json").write_text(json.dumps(payload), encoding="utf-8")
            evidence = collect_residency_evidence(root)

        row = evidence["requests"]["request-1"]
        self.assertTrue(evidence["available"])
        self.assertTrue(row["all_ranks_agree"])
        self.assertEqual(row["actual_residence"], "host_backed")
        self.assertEqual(row["rank_count"], 2)

    def test_residency_evidence_detects_tp_disagreement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rank, actual in ((0, "resident"), (1, "host_backed")):
                payload = {
                    "tp_rank": rank,
                    "request_residency": {
                        "requests": {
                            "request-1": {
                                "actual_residence": actual,
                                "tp_world_size": 2,
                            }
                        }
                    },
                }
                (root / f"tp-{rank}.json").write_text(json.dumps(payload), encoding="utf-8")
            evidence = collect_residency_evidence(root)

        row = evidence["requests"]["request-1"]
        self.assertFalse(row["all_ranks_agree"])
        self.assertEqual(row["actual_residence"], "conflict")

    def test_residency_evidence_requires_every_declared_tp_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "tp_rank": 0,
                "request_residency": {
                    "requests": {
                        "request-1": {
                            "actual_residence": "resident",
                            "tp_world_size": 4,
                        }
                    }
                },
            }
            (root / "tp-0.json").write_text(json.dumps(payload), encoding="utf-8")
            evidence = collect_residency_evidence(root)

        row = evidence["requests"]["request-1"]
        self.assertFalse(row["all_ranks_agree"])
        self.assertEqual(row["rank_count"], 1)

    def test_required_residency_evidence_times_out_without_stats(self):
        clock = FakeClock()

        def generate(url, spec, timeout, stream):
            return {
                "_started_perf_s": clock.now(),
                "end_to_end_s": 0.02,
                "output_tokens": 8,
                "ttft_s": 0.005,
                "tpot_s": 0.002,
            }

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RolloutExecutionError, "evidence incomplete"):
                execute_rollout(
                    "http://server/generate",
                    [submission()],
                    clock=ReplayClock(clock.now, clock.sleep),
                    generator=generate,
                    residency_stats_dir=directory,
                    require_residency_evidence=True,
                    residency_evidence_timeout_s=0.2,
                )

    def test_optional_partial_residency_evidence_is_marked_incomplete(self):
        clock = FakeClock()

        def generate(url, spec, timeout, stream):
            return {
                "_started_perf_s": clock.now(),
                "end_to_end_s": 0.02,
                "output_tokens": 8,
                "ttft_s": 0.005,
                "tpot_s": 0.002,
            }

        with tempfile.TemporaryDirectory() as directory:
            payload = {
                "tp_rank": 0,
                "request_residency": {
                    "requests": {
                        "request-1": {
                            "actual_residence": "resident",
                            "tp_world_size": 2,
                        }
                    }
                },
            }
            (Path(directory) / "tp-0.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            result = execute_rollout(
                "http://server/generate",
                [submission()],
                clock=ReplayClock(clock.now, clock.sleep),
                generator=generate,
                residency_stats_dir=directory,
                residency_evidence_timeout_s=0.1,
            )

        row = result["requests"][0]
        self.assertIsNone(row["actual_residence"])
        self.assertTrue(row["residency_incomplete"])


if __name__ == "__main__":
    unittest.main()
