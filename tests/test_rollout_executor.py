import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from adaptive_sparse import rollout_executor as rollout_module
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
    validate_submissions,
)
from adaptive_sparse.service_time import ServiceTimeEstimator


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


def without_deadline(item):
    return replace(
        item,
        deadline_ms=None,
        hint={
            key: value
            for key, value in item.hint.items()
            if key != "deadline_after_ms"
        },
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
        envelope = make_kv_hints(submission(), clock_now_s=10.0)

        self.assertEqual(envelope["protocol_version"], "1")
        self.assertEqual(envelope["message_id"], "request-1")
        self.assertEqual(envelope["actions"][0]["action_type"], "elastic.future")
        self.assertEqual(envelope["actions"][0]["action_version"], "2")
        self.assertEqual(envelope["actions"][0]["payload"]["residence"], "resident")
        payload = envelope["actions"][0]["payload"]
        self.assertEqual(payload["deadline_clock"], "perf_counter")
        self.assertEqual(payload["deadline_absolute_ms"], 10100)
        adjusted = make_kv_hints(
            submission(), deadline_after_ms=17, clock_now_s=10.0
        )
        self.assertEqual(adjusted["actions"][0]["payload"]["deadline_after_ms"], 17)
        self.assertEqual(
            adjusted["actions"][0]["payload"]["deadline_absolute_ms"], 10017
        )
        preannounced = make_kv_hints(
            submission(deadline_after_ms=500), release_after_ms=250
        )
        self.assertEqual(
            preannounced["actions"][0]["payload"]["release_after_ms"], 250
        )
        self.assertEqual(
            preannounced["actions"][0]["payload"]["release_clock"],
            "perf_counter",
        )
        demoted = make_kv_hints(submission(), residence_override="host_backed")
        self.assertEqual(demoted["actions"][0]["payload"]["residence"], "host_backed")

    def test_rejects_invalid_runtime_service_time_rate_before_dispatch(self):
        with self.assertRaisesRegex(
            RolloutExecutionError,
            "invalid runtime service-time hint",
        ):
            make_kv_hints(submission(), prefill_tokens_per_s=0)

    def test_payload_rates_force_v3_action_version(self):
        base = submission()
        planned = replace(
            base,
            hint={
                **base.hint,
                "prefill_tokens_per_s": 4_000.0,
                "decode_tokens_per_s": 1_000.0,
            },
        )

        envelope = make_kv_hints(planned)

        self.assertEqual(envelope["actions"][0]["action_version"], "3")

    def test_cohort_payload_uses_v4_and_carries_batch_identity(self):
        planned = replace(
            submission(),
            prefill_batch_size=2,
            cohort_id="batch-0",
        )

        envelope = make_kv_hints(planned)

        action = envelope["actions"][0]
        self.assertEqual(action["action_version"], "4")
        self.assertEqual(action["payload"]["cohort_id"], "batch-0")
        self.assertEqual(action["payload"]["cohort_size"], 2)

    def test_measured_rates_upgrade_only_later_hints_to_v3(self):
        base = submission()
        hints = {
            key: value
            for key, value in base.hint.items()
            if key != "deadline_after_ms"
        }
        rows = [
            RolloutSubmission(
                request_id=request_id,
                planned_arrival_ms=0,
                submit_at_ms=100 if request_id == "fourth" else 0,
                estimated_finish_ms=120,
                kind="tool",
                prompt_tokens=32,
                output_tokens=8,
                hint=dict(hints),
            )
            for request_id in ("first", "second", "third", "fourth")
        ]
        seen = {}

        def generate(url, spec, timeout, stream):
            seen[spec["request_id"]] = spec["future_hint"]
            return {
                "_started_perf_s": time.perf_counter(),
                "end_to_end_s": 0.01,
                "output_tokens": 8,
                "ttft_s": 0.008,
                "tpot_s": 0.001,
            }

        estimator = ServiceTimeEstimator(alpha=1.0, min_samples=3)
        result = execute_rollout(
            "http://server/generate",
            rows,
            generator=generate,
            max_inflight=1,
            poll_interval_ms=1,
            max_submit_lateness_ms=10_000,
            clock=FakeClock(),
            service_time_estimator=estimator,
        )

        self.assertEqual(seen["first"]["actions"][0]["action_version"], "2")
        self.assertEqual(seen["second"]["actions"][0]["action_version"], "2")
        self.assertEqual(seen["third"]["actions"][0]["action_version"], "2")
        fourth_action = seen["fourth"]["actions"][0]
        self.assertEqual(fourth_action["action_version"], "3")
        self.assertEqual(fourth_action["payload"]["prefill_tokens_per_s"], 4_000.0)
        self.assertEqual(fourth_action["payload"]["decode_tokens_per_s"], 1_000.0)
        telemetry = result["summary"]["service_time_estimator"]
        self.assertEqual(telemetry["prefill_samples"], 4)
        self.assertEqual(telemetry["decode_samples"], 4)
        self.assertEqual(telemetry["min_samples"], 3)
        by_id = {row["request_id"]: row for row in result["requests"]}
        self.assertIsNone(by_id["first"]["service_time_prefill_tokens_per_s_hint"])
        self.assertIsNone(by_id["second"]["service_time_prefill_tokens_per_s_hint"])
        self.assertIsNone(by_id["third"]["service_time_prefill_tokens_per_s_hint"])
        self.assertEqual(
            by_id["fourth"]["service_time_prefill_tokens_per_s_hint"],
            4_000.0,
        )

    def test_measured_decode_rates_follow_the_runtime_residency_mode(self):
        tpot_by_id = {
            "a-resident": 0.001,
            "b-host": 0.004,
            "c-resident": 0.001,
            "d-host": 0.004,
        }
        submissions = [
            RolloutSubmission(
                request_id=request_id,
                planned_arrival_ms=0,
                submit_at_ms=0,
                estimated_finish_ms=100,
                kind="tool",
                prompt_tokens=32,
                output_tokens=8,
                hint={
                    "release_after_ms": 0,
                    "expected_output_tokens": 8,
                    "weight": 1.0,
                    "kind": "tool",
                    "residence": residence,
                },
            )
            for request_id, residence in (
                ("a-resident", "resident"),
                ("b-host", "host_backed"),
                ("c-resident", "resident"),
                ("d-host", "host_backed"),
            )
        ]
        seen = {}

        def generate(url, spec, timeout, stream):
            request_id = spec["request_id"]
            seen[request_id] = spec["future_hint"]
            return {
                "_started_perf_s": time.perf_counter(),
                "end_to_end_s": 0.01,
                "output_tokens": 8,
                "ttft_s": 0.008,
                "tpot_s": tpot_by_id[request_id],
            }

        result = execute_rollout(
            "http://server/generate",
            submissions,
            generator=generate,
            max_inflight=1,
            poll_interval_ms=1,
            max_submit_lateness_ms=10_000,
            clock=FakeClock(),
            service_time_estimator=ServiceTimeEstimator(
                alpha=1.0,
                min_samples=1,
            ),
        )

        def payload(request_id):
            return seen[request_id]["actions"][0]["payload"]

        self.assertNotIn("decode_tokens_per_s", payload("a-resident"))
        self.assertNotIn("decode_tokens_per_s", payload("b-host"))
        self.assertEqual(payload("c-resident")["decode_tokens_per_s"], 1_000.0)
        self.assertEqual(payload("d-host")["decode_tokens_per_s"], 250.0)
        telemetry = result["summary"]["service_time_estimator"]
        self.assertEqual(
            telemetry["decode_samples_by_residence"],
            {"resident": 2, "host_backed": 2},
        )

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

    def test_runtime_activation_peak_demotes_new_request_and_exports_ledger(self):
        base = submission()
        hint = {
            key: value for key, value in base.hint.items() if key != "deadline_after_ms"
        }
        first = RolloutSubmission(
            request_id="first",
            planned_arrival_ms=0,
            submit_at_ms=0,
            estimated_finish_ms=100,
            kind=base.kind,
            prompt_tokens=50,
            output_tokens=50,
            hint={**hint, "expected_output_tokens": 50},
        )
        second = RolloutSubmission(
            request_id="second",
            planned_arrival_ms=0,
            submit_at_ms=0,
            estimated_finish_ms=100,
            kind=base.kind,
            prompt_tokens=100,
            output_tokens=50,
            hint={**hint, "expected_output_tokens": 50},
        )
        first_started = threading.Event()
        second_started = threading.Event()
        seen = {}

        def generate(url, spec, timeout, stream):
            seen[spec["request_id"]] = spec["future_hint"]
            if spec["request_id"] == "first":
                first_started.set()
                self.assertTrue(second_started.wait(1.0))
            else:
                self.assertTrue(first_started.is_set())
                second_started.set()
            return {
                "_started_perf_s": time.perf_counter(),
                "end_to_end_s": 0.01,
                "output_tokens": 50,
                "ttft_s": 0.004,
                "tpot_s": 0.0001,
            }

        result = execute_rollout(
            "http://server/generate",
            [first, second],
            generator=generate,
            elastic_kv_config=ElasticKVConfig(
                memory_budget_bytes=1_400,
                kv_bytes_per_token=2,
                sparse_hot_tokens=64,
                prefill_workspace_bytes_per_token=7,
            ),
            poll_interval_ms=1,
        )

        by_id = {row["request_id"]: row for row in result["requests"]}
        first_row = by_id["first"]
        second_row = by_id["second"]
        self.assertEqual(first_row["runtime_residence"], "resident")
        self.assertEqual(second_row["runtime_residence"], "host_backed")
        self.assertEqual(
            second_row["elastic_admission_reason"],
            "prefill-headroom-demotion",
        )
        self.assertEqual(second_row["elastic_prefill_workspace_bytes"], 700)
        self.assertEqual(second_row["elastic_prefill_peak_bytes"], 1378)
        self.assertFalse(second_row["elastic_prefill_over_budget"])
        self.assertEqual(
            seen["second"]["actions"][0]["payload"]["residence"],
            "host_backed",
        )
        self.assertEqual(result["summary"]["prefill_headroom_demotions"], 1)

    def test_first_token_releases_prefill_workspace_for_later_admission(self):
        base = submission()
        hint = {
            key: value
            for key, value in base.hint.items()
            if key != "deadline_after_ms"
        }
        first = RolloutSubmission(
            request_id="first",
            planned_arrival_ms=0,
            submit_at_ms=0,
            estimated_finish_ms=100,
            kind=base.kind,
            prompt_tokens=50,
            output_tokens=50,
            hint={**hint, "expected_output_tokens": 50},
        )
        second = RolloutSubmission(
            request_id="second",
            planned_arrival_ms=0,
            submit_at_ms=0,
            estimated_finish_ms=100,
            kind=base.kind,
            prompt_tokens=100,
            output_tokens=50,
            hint={**hint, "expected_output_tokens": 50},
        )
        first_started = threading.Event()
        second_started = threading.Event()

        def generate(url, spec, timeout, stream):
            started = time.perf_counter()
            first_token_callback = spec[rollout_module.FIRST_TOKEN_CALLBACK_KEY]
            first_token_callback(started + 0.004)
            if spec["request_id"] == "first":
                first_started.set()
                self.assertTrue(second_started.wait(1.0))
            else:
                self.assertTrue(first_started.is_set())
                second_started.set()
            return {
                "_started_perf_s": started,
                "end_to_end_s": 0.02,
                "output_tokens": 50,
                "ttft_s": 0.004,
                "tpot_s": 0.0001,
            }

        result = execute_rollout(
            "http://server/generate",
            [first, second],
            generator=generate,
            elastic_kv_config=ElasticKVConfig(
                memory_budget_bytes=1_400,
                kv_bytes_per_token=2,
                sparse_hot_tokens=64,
                prefill_workspace_bytes_per_token=7,
            ),
            poll_interval_ms=1,
        )

        by_id = {row["request_id"]: row for row in result["requests"]}
        self.assertEqual(by_id["first"]["runtime_residence"], "resident")
        self.assertEqual(by_id["second"]["runtime_residence"], "resident")
        self.assertEqual(
            result["summary"]["prefill_headroom_demotions"],
            0,
        )
        self.assertEqual(
            result["summary"]["prefill_workspace_released_early"],
            2,
        )
        self.assertEqual(
            result["summary"]["prefill_workspace_held_until_completion"],
            0,
        )
        self.assertTrue(by_id["second"]["prefill_workspace_released_early"])

    def test_runtime_activation_over_budget_reports_peak_cause(self):
        request = RolloutSubmission(
            request_id="large-activation",
            planned_arrival_ms=0,
            submit_at_ms=0,
            estimated_finish_ms=100,
            kind="chat",
            prompt_tokens=100,
            output_tokens=50,
            hint={
                "release_after_ms": 0,
                "expected_output_tokens": 50,
                "weight": 1.0,
                "kind": "chat",
                "residence": "resident",
            },
        )

        with self.assertRaisesRegex(
            RolloutExecutionError,
            "bounded KV hot set plus prefill workspace",
        ):
            execute_rollout(
                "http://server/generate",
                [request],
                generator=lambda *args: self.fail("request should not be sent"),
                elastic_kv_config=ElasticKVConfig(
                    memory_budget_bytes=1_000,
                    kv_bytes_per_token=2,
                    sparse_hot_tokens=64,
                    prefill_workspace_bytes_per_token=9,
                ),
                poll_interval_ms=1,
            )

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

    def test_calibrated_duration_breaks_equal_static_deadline_slack(self):
        def timed(request_id, prompt_tokens, output_tokens):
            return RolloutSubmission(
                request_id=request_id,
                planned_arrival_ms=0,
                submit_at_ms=0,
                estimated_finish_ms=100,
                kind="tool",
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                hint={
                    "release_after_ms": 0,
                    "deadline_after_ms": 5_000,
                    "expected_output_tokens": output_tokens,
                    "weight": 1.0,
                    "kind": "tool",
                    "residence": "resident",
                },
                deadline_ms=5_000,
            )

        requests = [timed("a-short", 1, 1), timed("z-long", 100, 100)]
        seen = []

        def generate(url, spec, timeout, stream):
            seen.append(spec["request_id"])
            return {
                "_started_perf_s": time.perf_counter(),
                "end_to_end_s": 0.001,
                "output_tokens": spec["requested_output_tokens"],
                "ttft_s": 0.0005,
                "tpot_s": 0.0001,
            }

        execute_rollout(
            "http://server/generate",
            requests,
            generator=generate,
            max_inflight=1,
            queue_policy="deadline",
            poll_interval_ms=1,
            max_submit_lateness_ms=10_000,
            service_time_estimator=ServiceTimeEstimator(
                prefill_tokens_per_s=100,
                decode_tokens_per_s=100,
            ),
        )

        self.assertEqual(seen, ["z-long", "a-short"])

    def test_measured_mode_rate_overrides_static_hint_for_deadline_order(self):
        def timed(request_id, residence):
            return RolloutSubmission(
                request_id=request_id,
                planned_arrival_ms=0,
                submit_at_ms=0,
                estimated_finish_ms=100,
                kind="tool",
                prompt_tokens=100,
                output_tokens=100,
                hint={
                    "release_after_ms": 0,
                    "deadline_after_ms": 5_000,
                    "expected_output_tokens": 100,
                    "weight": 1.0,
                    "kind": "tool",
                    "residence": residence,
                    "prefill_tokens_per_s": 10_000.0,
                    "decode_tokens_per_s": 10_000.0,
                },
                deadline_ms=5_000,
            )

        estimator = ServiceTimeEstimator(alpha=1.0, min_samples=1)
        estimator.observe(
            prompt_tokens=100,
            output_tokens=100,
            ttft_s=0.01,
            tpot_s=0.1,
            residence="resident",
        )
        estimator.observe(
            prompt_tokens=100,
            output_tokens=100,
            ttft_s=0.01,
            tpot_s=0.001,
            residence="host_backed",
        )
        seen = []

        def generate(url, spec, timeout, stream):
            seen.append(spec["request_id"])
            return {
                "_started_perf_s": time.perf_counter(),
                "end_to_end_s": 0.001,
                "output_tokens": 100,
                "ttft_s": 0.01,
                "tpot_s": 0.001,
            }

        execute_rollout(
            "http://server/generate",
            [
                timed("a-host", "host_backed"),
                timed("z-resident", "resident"),
            ],
            generator=generate,
            max_inflight=1,
            queue_policy="deadline",
            poll_interval_ms=1,
            max_submit_lateness_ms=10_000,
            service_time_estimator=estimator,
        )

        self.assertEqual(seen, ["z-resident", "a-host"])

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
        absolute_deadline = seen[0][1]["future_hint"]["actions"][0]["payload"][
            "deadline_absolute_ms"
        ]
        self.assertEqual(absolute_deadline, 10050)
        self.assertEqual(seen[0][3], True)

    def test_managed_executor_preannounces_without_moving_deadline(self):
        clock = FakeClock()
        seen = []
        preannounced = replace(
            submission(deadline_after_ms=40),
            submit_at_ms=100,
            estimated_finish_ms=160,
            deadline_ms=140,
        )

        def generate(url, spec, timeout, stream):
            seen.append(spec)
            return {
                "_started_perf_s": clock.now(),
                "end_to_end_s": 0.02,
                "output_tokens": 8,
                "ttft_s": 0.005,
                "tpot_s": 0.002,
            }

        result = execute_rollout(
            "http://server/generate",
            [preannounced],
            max_inflight=1,
            poll_interval_ms=1,
            clock=ReplayClock(clock.now, clock.sleep),
            generator=generate,
            preannounce_lead_ms=100,
        )

        payload = seen[0]["future_hint"]["actions"][0]["payload"]
        self.assertEqual(payload["release_after_ms"], 100)
        self.assertEqual(payload["release_clock"], "perf_counter")
        self.assertEqual(payload["release_absolute_ms"], 10100)
        self.assertEqual(payload["deadline_after_ms"], 140)
        self.assertEqual(payload["deadline_absolute_ms"], 10140)
        self.assertEqual(result["requests"][0]["actual_submit_ms"], 0.0)
        self.assertEqual(
            result["requests"][0]["preannounce_lead_actual_ms"], 100.0
        )
        self.assertEqual(result["summary"]["preannounce_lead_ms"], 100)
        self.assertEqual(
            result["summary"]["max_actual_preannounce_lead_ms"], 100.0
        )

    def test_default_executor_preannounces_without_moving_deadline(self):
        clock = FakeClock()
        seen = []
        preannounced = replace(
            submission(deadline_after_ms=40),
            submit_at_ms=100,
            estimated_finish_ms=160,
            deadline_ms=140,
        )

        def generate(url, spec, timeout, stream):
            seen.append(spec)
            return {
                "_started_perf_s": clock.now(),
                "end_to_end_s": 0.02,
                "output_tokens": 8,
                "ttft_s": 0.005,
                "tpot_s": 0.002,
            }

        result = execute_rollout(
            "http://server/generate",
            [preannounced],
            clock=ReplayClock(clock.now, clock.sleep),
            generator=generate,
            preannounce_lead_ms=100,
        )

        payload = seen[0]["future_hint"]["actions"][0]["payload"]
        self.assertEqual(payload["release_after_ms"], 100)
        self.assertEqual(payload["release_clock"], "perf_counter")
        self.assertEqual(payload["release_absolute_ms"], 10100)
        self.assertEqual(payload["deadline_after_ms"], 140)
        self.assertEqual(payload["deadline_absolute_ms"], 10140)
        self.assertEqual(result["summary"]["preannounce_lead_ms"], 100)

    def test_preannouncement_requires_hints(self):
        with self.assertRaisesRegex(ValueError, "requires emitted future hints"):
            execute_rollout(
                "http://server/generate",
                [submission()],
                emit_hints=False,
                preannounce_lead_ms=100,
                generator=lambda *args: self.fail("request should not be sent"),
            )

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
            (
                {"hint": {**base.hint, "prefix_group": "group", "shared_prefix_tokens": 33}},
                "shared prefix exceeds prompt length",
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

    def test_rejects_inconsistent_shared_prefix_plan_before_dispatch(self):
        base = submission()
        first = replace(
            base,
            request_id="first",
            hint={
                **base.hint,
                "prefix_group": "group",
                "shared_prefix_tokens": 16,
            },
        )
        second = replace(
            base,
            request_id="second",
            hint={
                **base.hint,
                "prefix_group": "group",
                "shared_prefix_tokens": 8,
            },
        )

        with self.assertRaisesRegex(
            RolloutExecutionError, "inconsistent shared prefix lengths"
        ):
            execute_rollout(
                "http://server/generate",
                [first, second],
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

    def test_cohort_members_enter_default_executor_together(self):
        base = submission()
        cohort = [
            without_deadline(replace(
                base,
                request_id=request_id,
                planned_arrival_ms=0,
                submit_at_ms=0,
                prefill_batch_size=2,
                cohort_id="batch-0",
            ))
            for request_id in ("first", "second")
        ]
        starts = {}
        lock = threading.Lock()

        def generate(url, spec, timeout, stream):
            started = time.perf_counter()
            with lock:
                starts[spec["request_id"]] = started
            return {
                "_started_perf_s": started,
                "end_to_end_s": 0.001,
                "output_tokens": 8,
                "ttft_s": 0.0005,
                "tpot_s": 0.0001,
            }

        result = execute_rollout(
            "http://server/generate",
            cohort,
            max_parallel=2,
            generator=generate,
            cohort_barrier_timeout_s=1.0,
        )

        self.assertEqual(set(starts), {"first", "second"})
        self.assertLess(abs(starts["first"] - starts["second"]), 0.05)
        self.assertEqual(result["summary"]["batched_cohort_count"], 1)
        self.assertEqual(result["summary"]["batched_request_count"], 2)
        self.assertTrue(
            all(row["cohort_start_skew_ok"] for row in result["requests"])
        )

    def test_managed_executor_admits_cohort_as_one_inflight_group(self):
        base = submission()
        singleton = without_deadline(replace(
            base,
            request_id="singleton",
            planned_arrival_ms=0,
            submit_at_ms=0,
        ))
        cohort = [
            without_deadline(replace(
                base,
                request_id=request_id,
                planned_arrival_ms=20,
                submit_at_ms=20,
                estimated_finish_ms=50,
                prefill_batch_size=2,
                cohort_id="batch-0",
            ))
            for request_id in ("cohort-first", "cohort-second")
        ]
        singleton_started = threading.Event()
        release_singleton = threading.Event()
        cohort_started = threading.Event()

        def generate(url, spec, timeout, stream):
            started = time.perf_counter()
            if spec["request_id"] == "singleton":
                singleton_started.set()
                release_singleton.wait(timeout=2.0)
            else:
                cohort_started.set()
            return {
                "_started_perf_s": started,
                "end_to_end_s": 0.001,
                "output_tokens": 8,
                "ttft_s": 0.0005,
                "tpot_s": 0.0001,
            }

        outcome = {}

        def run():
            try:
                outcome["result"] = execute_rollout(
                    "http://server/generate",
                    [singleton, *cohort],
                    max_inflight=2,
                    max_parallel=2,
                    max_submit_lateness_ms=1_000,
                    generator=generate,
                    poll_interval_ms=1,
                    cohort_barrier_timeout_s=1.0,
                )
            except Exception as exc:  # noqa: BLE001 - re-raised below
                outcome["error"] = exc

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(singleton_started.wait(timeout=1.0))
        self.assertFalse(cohort_started.wait(timeout=0.05))
        release_singleton.set()
        worker.join(timeout=3.0)
        self.assertFalse(worker.is_alive())
        if "error" in outcome:
            raise outcome["error"]
        self.assertTrue(cohort_started.is_set())
        self.assertEqual(
            outcome["result"]["summary"]["batched_request_count"], 2
        )

    def test_managed_executor_runs_singleton_past_capacity_blocked_cohort(self):
        base = without_deadline(submission())
        blocker = replace(
            base,
            request_id="a-blocker",
            planned_arrival_ms=0,
            submit_at_ms=0,
            estimated_finish_ms=500,
        )
        cohort = [
            replace(
                base,
                request_id=f"b-cohort-{index}",
                planned_arrival_ms=20,
                submit_at_ms=20,
                estimated_finish_ms=60,
                prefill_batch_size=2,
                cohort_id="batch-0",
            )
            for index in range(2)
        ]
        singleton = replace(
            base,
            request_id="c-ready",
            planned_arrival_ms=20,
            submit_at_ms=20,
            estimated_finish_ms=60,
        )
        blocker_started = threading.Event()
        release_blocker = threading.Event()
        singleton_started = threading.Event()
        cohort_started = threading.Event()
        starts = []
        lock = threading.Lock()

        def generate(url, spec, timeout, stream):
            started = time.perf_counter()
            with lock:
                starts.append(spec["request_id"])
            if spec["request_id"] == "a-blocker":
                blocker_started.set()
                release_blocker.wait(timeout=2.0)
            elif spec["request_id"] == "c-ready":
                singleton_started.set()
            else:
                cohort_started.set()
            return {
                "_started_perf_s": started,
                "end_to_end_s": 0.001,
                "output_tokens": 8,
                "ttft_s": 0.0005,
                "tpot_s": 0.0001,
            }

        outcome = {}

        def run():
            try:
                outcome["result"] = execute_rollout(
                    "http://server/generate",
                    [blocker, *cohort, singleton],
                    max_inflight=2,
                    max_parallel=4,
                    generator=generate,
                    poll_interval_ms=1,
                    cohort_barrier_timeout_s=1.0,
                )
            except Exception as exc:  # noqa: BLE001 - re-raised below
                outcome["error"] = exc

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(blocker_started.wait(timeout=1.0))
        singleton_ran = singleton_started.wait(timeout=0.5)
        cohort_wait_while_full = not cohort_started.wait(timeout=0.05)
        release_blocker.set()
        worker.join(timeout=3.0)

        self.assertFalse(worker.is_alive())
        if "error" in outcome:
            raise outcome["error"]
        self.assertTrue(singleton_ran)
        self.assertTrue(cohort_wait_while_full)
        self.assertLess(
            starts.index("c-ready"),
            min(
                index
                for index, request_id in enumerate(starts)
                if request_id.startswith("b-cohort-")
            ),
        )
        scheduler = outcome["result"]["summary"]["managed_scheduler"]
        self.assertGreaterEqual(
            scheduler["head_of_line_capacity_bypasses"], 1
        )

    def test_managed_executor_runs_singleton_past_budget_blocked_group(self):
        base = without_deadline(submission())
        blocker = replace(
            base,
            request_id="a-blocker",
            planned_arrival_ms=0,
            submit_at_ms=0,
            estimated_finish_ms=500,
        )
        blocked = replace(
            base,
            request_id="b-blocked",
            planned_arrival_ms=20,
            submit_at_ms=20,
            estimated_finish_ms=60,
        )
        singleton = replace(
            base,
            request_id="c-ready",
            planned_arrival_ms=20,
            submit_at_ms=20,
            estimated_finish_ms=60,
        )
        blocker_started = threading.Event()
        release_blocker = threading.Event()
        blocked_started = threading.Event()
        singleton_started = threading.Event()
        starts = []
        lock = threading.Lock()

        def admit(member, *, active_bytes, **kwargs):
            hold = 10
            return SimpleNamespace(
                request_id=member.request_id,
                planned_residence="resident",
                chosen_residence="resident",
                reason="test-admission",
                active_before_bytes=active_bytes,
                active_after_bytes=active_bytes + hold,
                resident_bytes=hold,
                sparse_bytes=4,
                defer_for_budget=(
                    member.request_id == "b-blocked" and active_bytes > 0
                ),
                over_budget=False,
                future_reserved_bytes=0,
                future_prefill_bytes=0,
                future_reserved_requests=0,
                future_effective_after_bytes=active_bytes + hold,
                future_over_budget=False,
                prefill_workspace_bytes=0,
                prefill_peak_bytes=active_bytes + hold,
                future_effective_peak_bytes=active_bytes + hold,
                prefill_over_budget=False,
                future_peak_over_budget=False,
                service_time_feedback_applied=False,
                service_time_current_duration_ms=None,
                service_time_calibrated_future_requests=0,
            )

        def generate(url, spec, timeout, stream):
            started = time.perf_counter()
            with lock:
                starts.append(spec["request_id"])
            if spec["request_id"] == "a-blocker":
                blocker_started.set()
                release_blocker.wait(timeout=2.0)
            elif spec["request_id"] == "b-blocked":
                blocked_started.set()
            else:
                singleton_started.set()
            return {
                "_started_perf_s": started,
                "end_to_end_s": 0.001,
                "output_tokens": 8,
                "ttft_s": 0.0005,
                "tpot_s": 0.0001,
            }

        outcome = {}

        def run():
            try:
                with patch.object(
                    rollout_module,
                    "decide_elastic_admission",
                    side_effect=admit,
                ):
                    outcome["result"] = execute_rollout(
                        "http://server/generate",
                        [blocker, blocked, singleton],
                        max_inflight=2,
                        max_parallel=3,
                        elastic_kv_config=ElasticKVConfig(
                            memory_budget_bytes=1024,
                            kv_bytes_per_token=1,
                            sparse_hot_tokens=16,
                            lookahead_ms=200,
                        ),
                        generator=generate,
                        poll_interval_ms=1,
                    )
            except Exception as exc:  # noqa: BLE001 - re-raised below
                outcome["error"] = exc

        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(blocker_started.wait(timeout=1.0))
        singleton_ran = singleton_started.wait(timeout=0.5)
        blocked_waited = not blocked_started.wait(timeout=0.05)
        release_blocker.set()
        worker.join(timeout=3.0)

        self.assertFalse(worker.is_alive())
        if "error" in outcome:
            raise outcome["error"]
        self.assertTrue(singleton_ran)
        self.assertTrue(blocked_waited)
        self.assertLess(starts.index("c-ready"), starts.index("b-blocked"))
        scheduler = outcome["result"]["summary"]["managed_scheduler"]
        self.assertGreaterEqual(
            scheduler["head_of_line_budget_bypasses"], 1
        )

    def test_rejects_parallelism_smaller_than_cohort(self):
        base = submission()
        cohort = [
            without_deadline(replace(
                base,
                request_id=f"member-{index}",
                planned_arrival_ms=0,
                submit_at_ms=0,
                prefill_batch_size=2,
                cohort_id="batch-0",
            ))
            for index in range(2)
        ]
        for kwargs, message in (
            ({"max_parallel": 1}, "max_parallel must be at least"),
            ({"max_inflight": 1}, "max_inflight must be at least"),
        ):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaisesRegex(ValueError, message),
            ):
                execute_rollout(
                    "http://server/generate",
                    cohort,
                    generator=lambda *args: self.fail("request should not be sent"),
                    **kwargs,
                )

    def test_rejects_cohort_start_skew_above_tolerance(self):
        base = submission()
        cohort = [
            without_deadline(replace(
                base,
                request_id=request_id,
                planned_arrival_ms=0,
                submit_at_ms=0,
                prefill_batch_size=2,
                cohort_id="batch-0",
            ))
            for request_id in ("first", "second")
        ]

        def generate(url, spec, timeout, stream):
            offset = 0.0 if spec["request_id"] == "first" else 0.01
            return {
                "_started_perf_s": time.perf_counter() + offset,
                "end_to_end_s": 0.001,
                "output_tokens": 8,
                "ttft_s": 0.0005,
                "tpot_s": 0.0001,
            }

        with self.assertRaisesRegex(RolloutExecutionError, "cohort start skew"):
            execute_rollout(
                "http://server/generate",
                cohort,
                max_parallel=2,
                max_submit_lateness_ms=1000.0,
                generator=generate,
                max_cohort_start_skew_ms=1.0,
            )

    def test_rejects_nonfinite_timing_parameters_before_dispatch(self):
        with self.assertRaisesRegex(ValueError, "timing parameters must be finite"):
            execute_rollout(
                "http://server/generate",
                [submission()],
                max_cohort_start_skew_ms=float("nan"),
                generator=lambda *args: self.fail("request should not be sent"),
            )

    def test_rejects_invalid_decode_batch_size(self):
        with self.assertRaisesRegex(
            RolloutExecutionError, "decode batch size must be positive"
        ):
            validate_submissions(
                [replace(submission(), decode_batch_size=0)]
            )

    def test_rejects_inconsistent_batch_cohort_metadata(self):
        first = replace(
            submission(),
            request_id="first",
            prefill_batch_size=2,
            cohort_id="batch-0",
        )
        second = replace(
            submission(),
            request_id="second",
            submit_at_ms=11,
            deadline_ms=111,
            prefill_batch_size=2,
            cohort_id="batch-0",
        )

        with self.assertRaisesRegex(RolloutExecutionError, "inconsistent timing"):
            validate_submissions([first, second])
        with self.assertRaisesRegex(RolloutExecutionError, "requires a cohort"):
            validate_submissions(
                [replace(submission(), prefill_batch_size=2)]
            )

    def test_batched_plan_round_trips_cohort_inputs(self):
        plan = plan_rollout(
            [
                RequestProfile(
                    "first", 0, 10, 10, residence_hint="resident"
                ),
                RequestProfile(
                    "second", 0, 10, 10, residence_hint="resident"
                ),
            ],
            PlannerConfig(
                memory_budget_bytes=100_000,
                kv_bytes_per_token=10,
                prefill_workspace_bytes_per_token=0,
                sparse_hot_tokens=10,
                max_active_requests=2,
                max_batch_requests=2,
            ),
            "batched",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(
                json.dumps({"rollout": plan.as_dict()}), encoding="utf-8"
            )
            loaded = load_plan(path)

        self.assertEqual({item.prefill_batch_size for item in loaded}, {2})
        self.assertEqual(len({item.cohort_id for item in loaded}), 1)
        self.assertEqual(len({item.submit_at_ms for item in loaded}), 1)

    def test_plan_round_trips_with_execution_inputs(self):
        plan = plan_rollout(
            [
                RequestProfile(
                    "request-1",
                    arrival_ms=0,
                    prompt_tokens=32,
                    output_tokens=8,
                    deadline_ms=100,
                    prefix_group="request-prefix",
                    shared_prefix_tokens=16,
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
        self.assertEqual(loaded[0].hint["prefix_group"], "request-prefix")
        self.assertEqual(loaded[0].hint["shared_prefix_tokens"], 16)
        self.assertEqual(loaded[0].decode_batch_size, 1)
        self.assertEqual(loaded[0].prefill_batch_size, 1)
        self.assertIsNone(loaded[0].cohort_id)

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

        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(RolloutExecutionError, "evidence incomplete"),
        ):
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
