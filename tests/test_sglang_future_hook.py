import json
import math
import os
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from adaptive_sparse import sglang_future_hook as hook
from adaptive_sparse.future_residency import decide_resident_claim
from adaptive_sparse.sglang_future_hook import (
    apply_to_waiting_queue,
    cohort_completion_evidence,
    observe_decode_batch,
    observe_prefill_batch,
    order_requests,
    planned_cohort_sizes,
    snapshot_stats,
    withhold_incomplete_cohorts,
    withhold_unreleased_requests,
)


def request(name, prompt, created, payload, *, version="1"):
    action = SimpleNamespace(
        action_type="elastic.future",
        action_version=version,
        payload=payload,
    )
    return SimpleNamespace(
        request_id=name,
        origin_input_ids=list(range(prompt)),
        num_matched_prefix_tokens=0,
        time_stats=SimpleNamespace(scheduler_recv_time=created),
        priority=0,
        kv_hints=SimpleNamespace(actions=[action]),
    )


class SGLangFutureHookTest(unittest.TestCase):
    def setUp(self):
        hook._cohort_first_seen_s.clear()
        hook._cohort_arrival_complete.clear()
        hook._cohort_timed_out.clear()
        hook._complete_cohort_ids.clear()
        hook._decode_batch_sizes.clear()
        hook._decode_batch_sizes_by_request.clear()
        hook._stats_identity.clear()
        hook._last_cohort_batch = None
        hook._last_decode_batch = None
        hook._last_held_cohorts = []
        hook._last_held_releases = []
        hook._hard_release_gate = False

    def test_absolute_deadline_uses_dispatch_clock_not_request_age(self):
        hint = hook.FutureHint.from_payload(
            {
                "deadline_after_ms": 50,
                "deadline_clock": "perf_counter",
                "deadline_absolute_ms": 1050,
            }
        )

        self.assertEqual(
            hook.deadline_remaining_ms(hint, now_s=1.02, age_ms=500),
            30,
        )

    def test_cohort_completion_requires_every_planned_group(self):
        expected = {"batch-0": 2, "batch-1": 3}
        passed = cohort_completion_evidence(
            expected,
            {
                "complete_cohort_batches": 2,
                "complete_cohort_ids": ["batch-0", "batch-1"],
                "partial_cohort_batches": 0,
                "cohort_gate_timeouts": 0,
                "cohort_metadata_conflicts": 0,
                "cohort_gate_errors": 0,
            },
        )
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["expected_request_count"], 5)

        missing = cohort_completion_evidence(
            expected,
            {
                "complete_cohort_batches": 1,
                "complete_cohort_ids": ["batch-0"],
                "partial_cohort_batches": 0,
                "cohort_gate_timeouts": 0,
                "cohort_metadata_conflicts": 0,
                "cohort_gate_errors": 0,
            },
        )
        self.assertFalse(missing["passed"])

        timed_out = cohort_completion_evidence(
            expected,
            {
                "complete_cohort_batches": 2,
                "complete_cohort_ids": ["batch-0", "batch-1"],
                "partial_cohort_batches": 0,
                "cohort_gate_timeouts": 1,
                "cohort_metadata_conflicts": 0,
                "cohort_gate_errors": 0,
            },
        )
        self.assertFalse(timed_out["passed"])

        wrong_identity = cohort_completion_evidence(
            expected,
            {
                "complete_cohort_batches": 2,
                "complete_cohort_ids": ["batch-0", "batch-0-other"],
                "partial_cohort_batches": 0,
                "cohort_gate_timeouts": 0,
                "cohort_metadata_conflicts": 0,
                "cohort_gate_errors": 0,
            },
        )
        self.assertFalse(wrong_identity["passed"])
        self.assertEqual(wrong_identity["missing_cohort_ids"], ["batch-1"])

    def test_planned_cohort_sizes_validate_membership(self):
        submissions = [
            SimpleNamespace(cohort_id="batch-0", prefill_batch_size=2),
            SimpleNamespace(cohort_id="batch-0", prefill_batch_size=2),
            SimpleNamespace(cohort_id=None, prefill_batch_size=1),
        ]
        self.assertEqual(planned_cohort_sizes(submissions), {"batch-0": 2})

        with self.assertRaisesRegex(ValueError, "1 submission"):
            planned_cohort_sizes(submissions[:1])
        with self.assertRaisesRegex(ValueError, "inconsistent planned size"):
            planned_cohort_sizes(
                [
                    *submissions[:2],
                    SimpleNamespace(cohort_id="batch-0", prefill_batch_size=3),
                ]
            )

    def test_decode_observer_tracks_dynamic_batch_occupancy(self):
        first = request("first", 10, 1.0, {})
        second = request("second", 10, 1.0, {})

        class DecodeMode:
            def is_decode(self):
                return True

        before = snapshot_stats()
        observe_decode_batch(
            SimpleNamespace(
                batch_to_run=SimpleNamespace(
                    reqs=[first], forward_mode=DecodeMode()
                )
            )
        )
        observe_decode_batch(
            SimpleNamespace(
                batch_to_run=SimpleNamespace(
                    reqs=[first, second], forward_mode=DecodeMode()
                )
            )
        )
        after = snapshot_stats()

        self.assertEqual(
            after["decode_batches"] - before["decode_batches"], 2
        )
        self.assertEqual(
            after["decode_requests_observed"]
            - before["decode_requests_observed"],
            3,
        )
        self.assertEqual(after["last_decode_batch"]["batch_size"], 2)
        self.assertEqual(
            after["decode_batch_sizes_by_request"]["first"],
            {"1": 1, "2": 1},
        )
        self.assertEqual(
            after["decode_batch_sizes_by_request"]["second"], {"2": 1}
        )

    def test_decode_observer_ignores_prefill_batches(self):
        item = request("prefill", 10, 1.0, {})

        class PrefillMode:
            def is_decode(self):
                return False

        before = snapshot_stats()["decode_batches"]
        observe_decode_batch(
            SimpleNamespace(
                batch_to_run=SimpleNamespace(
                    reqs=[item], forward_mode=PrefillMode()
                )
            )
        )
        self.assertEqual(snapshot_stats()["decode_batches"], before)

    def test_decode_observer_counts_running_rows_in_mixed_batch(self):
        prefill = request("prefill", 10, 1.0, {})
        first = request("first", 10, 1.0, {})
        second = request("second", 10, 1.0, {})

        class ExtendMode:
            def is_decode(self):
                return False

        before = snapshot_stats()
        observe_decode_batch(
            SimpleNamespace(
                batch_to_run=SimpleNamespace(
                    reqs=[prefill, first, second],
                    decoding_reqs=[first, second],
                    forward_mode=ExtendMode(),
                )
            )
        )
        after = snapshot_stats()

        self.assertEqual(after["decode_batches"] - before["decode_batches"], 1)
        self.assertEqual(
            after["decode_requests_observed"]
            - before["decode_requests_observed"],
            2,
        )
        self.assertEqual(
            after["decode_mixed_batches"] - before["decode_mixed_batches"], 1
        )
        self.assertEqual(after["last_decode_batch"]["batch_size"], 2)
        self.assertEqual(
            after["last_decode_batch"]["request_ids"], ["first", "second"]
        )
        self.assertTrue(after["last_decode_batch"]["mixed_prefill_decode"])

    def test_decode_request_trace_is_bounded_to_recent_requests(self):
        class DecodeMode:
            def is_decode(self):
                return True

        old_limit = hook._MAX_DECODE_REQUEST_TRACES
        hook._MAX_DECODE_REQUEST_TRACES = 2
        try:
            first = request("first", 10, 1.0, {})
            second = request("second", 10, 1.0, {})
            third = request("third", 10, 1.0, {})
            observe_decode_batch(
                SimpleNamespace(reqs=[first, second], forward_mode=DecodeMode())
            )
            observe_decode_batch(
                SimpleNamespace(reqs=[first], forward_mode=DecodeMode())
            )
            observe_decode_batch(
                SimpleNamespace(reqs=[third], forward_mode=DecodeMode())
            )
        finally:
            hook._MAX_DECODE_REQUEST_TRACES = old_limit

        self.assertEqual(
            list(hook._decode_batch_sizes_by_request), ["first", "third"]
        )

    def test_absolute_deadline_orders_without_scheduler_receive_time(self):
        long = request(
            "long",
            1000,
            0.0,
            {"expected_output_tokens": 7000},
            version="2",
        )
        short = request(
            "short",
            100,
            0.0,
            {
                "deadline_after_ms": 50,
                "deadline_clock": "perf_counter",
                "deadline_absolute_ms": 1050,
                "expected_output_tokens": 400,
            },
            version="2",
        )
        long.time_stats = SimpleNamespace()
        short.time_stats = SimpleNamespace()

        ordered = order_requests(
            [long, short],
            now_s=1.01,
            prefill_tokens_per_s=1000,
            decode_tokens_per_s=2000,
        )

        self.assertEqual([item.request_id for item in ordered], ["short", "long"])

    def test_absolute_deadline_rejects_partial_or_unknown_clock(self):
        cases = (
            {"deadline_after_ms": 50, "deadline_clock": "perf_counter"},
            {"deadline_after_ms": 50, "deadline_absolute_ms": 1050},
            {
                "deadline_after_ms": 50,
                "deadline_clock": "unix",
                "deadline_absolute_ms": 1050,
            },
            {"deadline_clock": "perf_counter", "deadline_absolute_ms": 1050},
        )
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                hook.FutureHint.from_payload(payload)

        legacy = request(
            "legacy",
            10,
            1.0,
            {
                "deadline_after_ms": 50,
                "deadline_clock": "perf_counter",
                "deadline_absolute_ms": 1050,
            },
        )
        with self.assertRaisesRegex(ValueError, "version 1"):
            hook._extract_hint(legacy)

    def test_legacy_hint_fields_remain_distinguishable_from_omitted_values(self):
        hint = hook.FutureHint.from_payload({"deadline_after_ms": 50})
        self.assertIsNone(hint.residence)
        self.assertIsNone(hint.expected_output_tokens)
        decision = decide_resident_claim(
            sequence_tokens=8192,
            native_buffer_tokens=4096,
            slots_available=True,
            hint_residence=hint.residence,
            expected_output_tokens=hint.expected_output_tokens,
            minimum_hint_output_tokens=128,
        )
        self.assertTrue(decision.claim)
        self.assertEqual(decision.reason, "long-request-default")

    def test_shared_prefix_identity_is_validated_and_preserved(self):
        hint = hook.FutureHint.from_payload(
            {
                "prefix_group": "system-prefix",
                "shared_prefix_tokens": 4096,
            }
        )
        self.assertEqual(hint.prefix_group, "system-prefix")
        self.assertEqual(hint.shared_prefix_tokens, 4096)
        with self.assertRaisesRegex(ValueError, "prefix_group"):
            hook.FutureHint.from_payload({"shared_prefix_tokens": 1})
        with self.assertRaisesRegex(ValueError, "shared_prefix_tokens"):
            hook.FutureHint.from_payload(
                {"prefix_group": "group", "shared_prefix_tokens": True}
            )

    def test_v3_measured_rates_override_defaults_per_request(self):
        slow = request(
            "slow",
            100,
            1.0,
            {
                "expected_output_tokens": 100,
                "prefill_tokens_per_s": 100,
                "decode_tokens_per_s": 100,
            },
            version="3",
        )
        fast = request(
            "fast",
            100,
            1.0,
            {
                "expected_output_tokens": 100,
                "prefill_tokens_per_s": 10_000,
                "decode_tokens_per_s": 10_000,
            },
            version="3",
        )

        ordered = order_requests(
            [slow, fast],
            now_s=1.01,
            prefill_tokens_per_s=1,
            decode_tokens_per_s=1,
        )

        self.assertEqual([item.request_id for item in ordered], ["fast", "slow"])

    def test_v4_cohort_metadata_requires_version_four(self):
        cohort_payload = {"cohort_id": "batch-0", "cohort_size": 2}
        hint = hook.FutureHint.from_payload(cohort_payload)
        self.assertEqual(hint.cohort_id, "batch-0")
        self.assertEqual(hint.cohort_size, 2)
        supported = request("member", 10, 1.0, cohort_payload, version="4")
        self.assertEqual(hook._extract_hint(supported).cohort_id, "batch-0")
        rejected = request("member", 10, 1.0, cohort_payload, version="3")
        with self.assertRaisesRegex(ValueError, "version 4"):
            hook._extract_hint(rejected)
        with self.assertRaisesRegex(ValueError, "requires cohort_id"):
            hook.FutureHint.from_payload({"cohort_size": 2})

    def test_observes_complete_and_partial_server_prefill_cohorts(self):
        first = request(
            "first",
            10,
            1.0,
            {"cohort_id": "batch-0", "cohort_size": 2},
            version="4",
        )
        second = request(
            "second",
            10,
            1.0,
            {"cohort_id": "batch-0", "cohort_size": 2},
            version="4",
        )
        before = snapshot_stats()

        observe_prefill_batch(SimpleNamespace(reqs=[first]))
        partial = snapshot_stats()
        observe_prefill_batch(SimpleNamespace(reqs=[first, second]))
        complete = snapshot_stats()

        self.assertEqual(
            partial["partial_cohort_batches"]
            - before["partial_cohort_batches"],
            1,
        )
        self.assertFalse(partial["last_cohort_batch"]["complete"])
        self.assertEqual(
            complete["complete_cohort_batches"]
            - partial["complete_cohort_batches"],
            1,
        )
        self.assertIn("batch-0", complete["complete_cohort_ids"])
        self.assertTrue(complete["last_cohort_batch"]["complete"])
        self.assertEqual(
            complete["last_cohort_batch"]["request_ids"],
            ["first", "second"],
        )

    def test_observer_unwraps_new_scheduler_prefill_plan(self):
        first = request(
            "first",
            10,
            1.0,
            {"cohort_id": "batch-0", "cohort_size": 2},
            version="4",
        )
        second = request(
            "second",
            10,
            1.0,
            {"cohort_id": "batch-0", "cohort_size": 2},
            version="4",
        )
        before = snapshot_stats()

        observe_prefill_batch(
            SimpleNamespace(
                batch_to_run=SimpleNamespace(reqs=[first, second]),
                running_batch=object(),
            )
        )

        after = snapshot_stats()
        self.assertEqual(
            after["complete_cohort_batches"]
            - before["complete_cohort_batches"],
            1,
        )

    def test_partial_cohort_is_withheld_without_blocking_other_requests(self):
        first = request(
            "first",
            10,
            1.0,
            {"cohort_id": "batch-0", "cohort_size": 2},
            version="4",
        )
        other = request("other", 10, 1.0, {})
        queue = [first, other]

        held = withhold_incomplete_cohorts(queue, now_s=10.0, timeout_ms=100.0)

        self.assertEqual(held, [first])
        self.assertEqual(queue, [other])
        state = snapshot_stats()
        self.assertEqual(state["last_held_cohorts"][0]["cohort_id"], "batch-0")
        self.assertEqual(state["last_held_cohorts"][0]["expected_members"], 2)

    def test_partial_cohort_timeout_releases_members_fail_safe(self):
        first = request(
            "first",
            10,
            1.0,
            {"cohort_id": "batch-0", "cohort_size": 2},
            version="4",
        )
        queue = [first]

        with patch.dict("os.environ", {hook.COHORT_WAIT_TIMEOUT_ENV: "10"}):
            initially_held = withhold_incomplete_cohorts(queue, now_s=1.0)
            self.assertEqual(initially_held, [first])
            queue.extend(initially_held)
            before = snapshot_stats()["cohort_gate_timeouts"]
            held = withhold_incomplete_cohorts(queue, now_s=1.011)

        self.assertEqual(held, [])
        self.assertEqual(queue, [first])
        self.assertEqual(snapshot_stats()["cohort_gate_timeouts"] - before, 1)
        self.assertEqual(
            withhold_incomplete_cohorts(queue, now_s=2.0),
            [],
        )
        self.assertEqual(snapshot_stats()["cohort_gate_timeouts"] - before, 1)

    def test_zero_cohort_timeout_does_not_hold_a_partial_cohort(self):
        first = request(
            "first",
            10,
            1.0,
            {"cohort_id": "batch-0", "cohort_size": 2},
            version="4",
        )
        queue = [first]
        before = snapshot_stats()["cohort_gate_timeouts"]

        held = withhold_incomplete_cohorts(queue, now_s=1.0, timeout_ms=0)

        self.assertEqual(held, [])
        self.assertEqual(queue, [first])
        self.assertEqual(snapshot_stats()["cohort_gate_timeouts"] - before, 1)

    def test_completed_cohort_is_not_reheld_after_native_partial_admission(self):
        first = request(
            "first",
            10,
            1.0,
            {"cohort_id": "batch-0", "cohort_size": 2},
            version="4",
        )
        second = request(
            "second",
            10,
            1.0,
            {"cohort_id": "batch-0", "cohort_size": 2},
            version="4",
        )
        queue = [first]
        queue.extend(withhold_incomplete_cohorts(queue, now_s=1.0, timeout_ms=100))

        queue.append(second)
        self.assertEqual(
            withhold_incomplete_cohorts(queue, now_s=1.01, timeout_ms=100), []
        )
        queue.remove(first)
        self.assertEqual(
            withhold_incomplete_cohorts(queue, now_s=1.02, timeout_ms=100), []
        )
        self.assertEqual(queue, [second])

    def test_oversubscribed_cohort_metadata_fails_open(self):
        members = [
            request(
                request_id,
                10,
                1.0,
                {"cohort_id": "batch-0", "cohort_size": 2},
                version="4",
            )
            for request_id in ("first", "second", "extra")
        ]
        original = list(members)
        before = snapshot_stats()["cohort_metadata_conflicts"]

        held = withhold_incomplete_cohorts(members, now_s=1.0, timeout_ms=100)

        self.assertEqual(held, [])
        self.assertEqual(members, original)
        self.assertEqual(
            snapshot_stats()["cohort_metadata_conflicts"] - before,
            1,
        )

    def test_invalid_cohort_timeout_leaves_native_scheduling_available(self):
        first = request(
            "first",
            10,
            1.0,
            {"cohort_id": "batch-0", "cohort_size": 2},
            version="4",
        )
        queue = [first]

        with patch.dict(
            "os.environ", {hook.COHORT_WAIT_TIMEOUT_ENV: "not-a-number"}
        ), self.assertRaises(ValueError):
            withhold_incomplete_cohorts(queue, now_s=1.0)

        self.assertEqual(queue, [first])

    def test_service_rates_are_rejected_from_older_action_versions(self):
        legacy = request(
            "legacy",
            100,
            1.0,
            {"prefill_tokens_per_s": 100, "decode_tokens_per_s": 100},
            version="2",
        )
        with self.assertRaisesRegex(ValueError, "version 3"):
            hook._extract_hint(legacy)

    def test_service_rates_require_finite_positive_values(self):
        for value in (0, -1, True, float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                hook.FutureHint.from_payload({"prefill_tokens_per_s": value})

    def test_deadline_reorders_after_native_priority(self):
        long = request(
            "long",
            1000,
            1.0,
            {"expected_output_tokens": 7000, "weight": 1},
        )
        short = request(
            "short",
            100,
            1.0,
            {
                "expected_output_tokens": 400,
                "deadline_after_ms": 500,
                "weight": 2,
            },
        )
        ordered = order_requests(
            [long, short], now_s=1.01, prefill_tokens_per_s=1000, decode_tokens_per_s=2000
        )
        self.assertEqual([item.request_id for item in ordered], ["short", "long"])

    def test_unreleased_request_stays_behind_ready_work(self):
        ready = request(
            "ready", 10, 1.0,
            {"deadline_after_ms": 0, "expected_output_tokens": 10},
        )
        future = request("future", 10, 1.0, {"release_after_ms": 2000})
        ordered = order_requests([future, ready], now_s=1.1)
        self.assertEqual([item.request_id for item in ordered], ["ready", "future"])

    def test_hard_release_gate_holds_only_until_release_time(self):
        ready = request("ready", 10, 1.0, {})
        future = request("future", 10, 1.0, {"release_after_ms": 200})
        queue = [future, ready]
        before = snapshot_stats()["hard_release_members_held"]

        held = withhold_unreleased_requests(queue, now_s=1.1)

        self.assertEqual(held, [future])
        self.assertEqual(queue, [ready])
        self.assertEqual(
            snapshot_stats()["hard_release_members_held"] - before,
            1,
        )
        self.assertEqual(
            snapshot_stats()["last_held_releases"][0]["remaining_ms"],
            100.0,
        )
        self.assertEqual(withhold_unreleased_requests(queue, now_s=1.2), [])
        self.assertEqual(queue, [ready])

    def test_absolute_release_uses_dispatch_clock(self):
        future = request(
            "future",
            10,
            1.0,
            {
                "release_after_ms": 200,
                "release_clock": "perf_counter",
                "release_absolute_ms": 1_150,
            },
        )
        queue = [future]

        self.assertEqual(withhold_unreleased_requests(queue, now_s=1.1), [future])
        self.assertEqual(
            snapshot_stats()["last_held_releases"][0]["remaining_ms"],
            50.0,
        )
        self.assertEqual(withhold_unreleased_requests(queue, now_s=1.15), [])

    def test_release_clock_requires_absolute_timestamp(self):
        with self.assertRaisesRegex(
            ValueError, "release_clock and release_absolute_ms"
        ):
            hook.FutureHint.from_payload(
                {"release_after_ms": 100, "release_clock": "perf_counter"}
            )

    def test_hard_release_gate_restores_preannounced_queue_order(self):
        ready = request("ready", 10, 1.0, {})
        future = request("future", 10, 1.0, {"release_after_ms": 200})
        queue = [future, ready]
        hook._hard_release_gate = True
        frame = hook._PrefillGateFrame()
        token = hook._active_prefill_gate.set(frame)
        try:
            hook._gate_active_prefill_queue(queue, now_s=1.1)
        finally:
            hook._active_prefill_gate.reset(token)

        self.assertEqual(queue, [ready])
        self.assertEqual(frame.held, [future])
        hook._restore_prefill_queue(queue, frame=frame)
        self.assertEqual(queue, [future, ready])

    def test_unreleased_tier_preserves_exact_native_order(self):
        first = request("first", 1000, 1.0, {"release_after_ms": 2000})
        second = request("second", 10, 1.0, {"release_after_ms": 3000})
        self.assertEqual(order_requests([first, second], now_s=1.1), [first, second])

    def test_missing_hint_preserves_native_order_atomically(self):
        first = request("first", 10, 1.0, {})
        second = SimpleNamespace(
            request_id="second",
            origin_input_ids=list(range(10)),
            num_matched_prefix_tokens=0,
            time_stats=SimpleNamespace(scheduler_recv_time=1.0),
            priority=0,
            kv_hints=None,
        )
        self.assertEqual(order_requests([first, second], now_s=1.1), [first, second])

    def test_invalid_hint_does_not_mutate_queue_or_raise(self):
        bad = request("bad", 10, 1.0, {"release_after_ms": -1})
        queue = [bad]
        apply_to_waiting_queue(queue, now_s=1.1)
        self.assertEqual(queue, [bad])
        self.assertGreater(snapshot_stats()["fallback_invalid_hint"], 0)

    def test_missing_hint_fallback_does_not_fail_release_telemetry(self):
        hinted = request("hinted", 10, 1.0, {})
        missing = SimpleNamespace(
            request_id="missing",
            origin_input_ids=list(range(10)),
            num_matched_prefix_tokens=0,
            time_stats=SimpleNamespace(scheduler_recv_time=1.0),
            priority=0,
            kv_hints=None,
        )
        queue = [hinted, missing]
        apply_to_waiting_queue(queue, now_s=1.1)
        self.assertEqual(queue, [hinted, missing])

    def test_equal_keys_follow_native_order(self):
        first = request(
            "first", 10, 1.0,
            {"deadline_after_ms": 100, "expected_output_tokens": 10},
        )
        second = request(
            "second", 10, 1.0,
            {"deadline_after_ms": 100, "expected_output_tokens": 10},
        )
        ordered = order_requests([first, second], now_s=1.0)
        self.assertEqual([item.request_id for item in ordered], ["first", "second"])

    def test_residency_hint_orders_equal_priority_work(self):
        background = request(
            "background",
            100,
            1.0,
            {"expected_output_tokens": 1000, "residence": "host_backed"},
        )
        automatic = request(
            "automatic",
            100,
            1.0,
            {"expected_output_tokens": 1000, "residence": "auto"},
        )
        critical = request(
            "critical",
            100,
            1.0,
            {"expected_output_tokens": 1000, "residence": "resident"},
        )
        ordered = order_requests(
            [background, automatic, critical], now_s=1.01
        )
        self.assertEqual(
            [item.request_id for item in ordered],
            ["critical", "automatic", "background"],
        )

    def test_deadline_still_precedes_residency_tie_break(self):
        resident = request(
            "resident",
            100,
            1.0,
            {"expected_output_tokens": 1000, "residence": "resident"},
        )
        urgent = request(
            "urgent",
            100,
            1.0,
            {
                "expected_output_tokens": 1000,
                "deadline_after_ms": 50,
                "residence": "host_backed",
            },
        )
        ordered = order_requests([resident, urgent], now_s=1.01)
        self.assertEqual([item.request_id for item in ordered], ["urgent", "resident"])

    def test_missing_matched_prefix_count_is_treated_as_zero(self):
        item = request(
            "fresh", 10, 1.0,
            {"deadline_after_ms": 100, "expected_output_tokens": 10},
        )
        item.num_matched_prefix_tokens = None
        self.assertEqual(order_requests([item], now_s=1.0), [item])

    def test_deadline_without_output_estimate_preserves_native_order(self):
        first = request(
            "first", 1, 1.0,
            {"release_after_ms": 0, "deadline_after_ms": 100},
        )
        second = request(
            "second", 2, 2.0,
            {
                "release_after_ms": 0,
                "deadline_after_ms": 100,
                "expected_output_tokens": 10,
            },
        )
        queue = [first, second]
        apply_to_waiting_queue(queue, now_s=1.1)
        self.assertEqual([item.request_id for item in queue], ["first", "second"])

    def test_horizon_violation_preserves_native_order_without_partial_effect(self):
        queue = [
            request("later", 10, 1.0, {"release_after_ms": 5_000}),
            request("ready", 1, 1.0, {"release_after_ms": 0}),
        ]
        apply_to_waiting_queue(queue, now_s=1.1, horizon_ms=2_000)
        self.assertEqual([item.request_id for item in queue], ["later", "ready"])

    def test_horizon_uses_remaining_release_time(self):
        ready_soon = request("soon", 10, 0.1, {"release_after_ms": 1_000})
        ordered = order_requests(
            [ready_soon], now_s=1.0, horizon_ms=500
        )
        self.assertEqual(ordered, [ready_soon])

    def test_missing_prompt_length_preserves_native_order(self):
        item = request("missing-prompt", 10, 1.0, {"release_after_ms": 0})
        item.origin_input_ids = None
        queue = [item]
        apply_to_waiting_queue(queue, now_s=1.1)
        self.assertEqual(queue, [item])

    def test_deadline_uses_scheduler_receive_age(self):
        old = request(
            "old", 100, 0.1,
            {"deadline_after_ms": 1000, "expected_output_tokens": 10},
        )
        new = request(
            "new", 100, 0.9,
            {"deadline_after_ms": 1200, "expected_output_tokens": 10},
        )
        ordered = order_requests(
            [new, old], now_s=1.0, prefill_tokens_per_s=1000, decode_tokens_per_s=1000
        )
        self.assertEqual([item.request_id for item in ordered], ["old", "new"])

    def test_deadline_accepts_compatibility_receive_time(self):
        first = request(
            "first", 100, 1.0,
            {"deadline_after_ms": 1000, "expected_output_tokens": 10},
        )
        second = request(
            "second", 100, 1.0,
            {"deadline_after_ms": 1000, "expected_output_tokens": 10},
        )
        first.time_stats = None
        second.time_stats = None
        first._ekv_scheduler_recv_time = 1.0
        second._ekv_scheduler_recv_time = 0.9
        ordered = order_requests(
            [first, second], now_s=1.1, prefill_tokens_per_s=1000, decode_tokens_per_s=1000
        )
        self.assertEqual([item.request_id for item in ordered], ["second", "first"])

    def test_native_priority_tier_is_never_crossed(self):
        high = request(
            "high", 100, 1.0,
            {"deadline_after_ms": 5000, "expected_output_tokens": 10},
        )
        low = request(
            "low", 10, 1.0,
            {"deadline_after_ms": 0, "expected_output_tokens": 10},
        )
        high.priority = 10
        low.priority = 0
        ordered = order_requests([high, low], now_s=1.1, priority_sign=-1)
        self.assertEqual([item.request_id for item in ordered], ["high", "low"])

    def test_native_priority_tier_reorders_inside_equal_priority(self):
        slow = request("slow", 1000, 1.0, {"expected_output_tokens": 7000})
        urgent = request(
            "urgent",
            100,
            1.0,
            {"expected_output_tokens": 400, "deadline_after_ms": 500},
        )
        slow.priority = 4
        urgent.priority = 4
        ordered = order_requests(
            [slow, urgent],
            now_s=1.01,
            prefill_tokens_per_s=1000,
            decode_tokens_per_s=2000,
            priority_sign=-1,
        )
        self.assertEqual([item.request_id for item in ordered], ["urgent", "slow"])

    def test_off_policy_does_not_import_sglang(self):
        with patch.dict("os.environ", {"EKV_FUTURE_POLICY": "off"}), patch.dict(
            sys.modules, {}, clear=False
        ):
            sys.modules.pop("sglang", None)
            hook._installed = False
            hook.install()
            self.assertFalse(hook._installed)
            self.assertNotIn("sglang", sys.modules)
        hook._installed = False

    def test_installer_wraps_native_policy_after_it_runs(self):
        calls = []

        class SchedulePolicy:
            def calc_priority(self, waiting_queue, *args, **kwargs):
                calls.append("native")
                waiting_queue.reverse()

        modules = {
            "sglang": SimpleNamespace(__path__=[]),
            "sglang.srt": SimpleNamespace(__path__=[]),
            "sglang.srt.managers": SimpleNamespace(__path__=[]),
            "sglang.srt.managers.schedule_policy": SimpleNamespace(
                SchedulePolicy=SchedulePolicy
            ),
        }
        long = request("long", 1000, 1.0, {"expected_output_tokens": 7000})
        short = request(
            "short",
            100,
            1.0,
            {"expected_output_tokens": 400, "deadline_after_ms": 500},
        )
        with patch.dict(sys.modules, modules), patch(
            "adaptive_sparse.kv_hints_compat.install"
        ), patch.dict("os.environ", {"EKV_FUTURE_POLICY": "deadline"}):
            hook._installed = False
            hook.install()
            queue = [short, long]
            SchedulePolicy().calc_priority(queue)
        hook._installed = False
        self.assertEqual(calls, ["native"])
        self.assertEqual([item.request_id for item in queue], ["short", "long"])

    def test_installer_preserves_native_kv_hints_transport(self):
        class SchedulePolicy:
            def calc_priority(self, waiting_queue, *args, **kwargs):
                return None

        modules = {
            "sglang": SimpleNamespace(__path__=[]),
            "sglang.srt": SimpleNamespace(__path__=[]),
            "sglang.srt.managers": SimpleNamespace(__path__=[]),
            "sglang.srt.managers.kv_hints": SimpleNamespace(),
            "sglang.srt.managers.schedule_policy": SimpleNamespace(
                SchedulePolicy=SchedulePolicy
            ),
        }
        with patch.dict(sys.modules, modules), patch(
            "adaptive_sparse.kv_hints_compat.install"
        ) as compat_install, patch.dict(
            "os.environ", {"EKV_FUTURE_POLICY": "deadline"}
        ):
            hook._installed = False
            hook.install()
        hook._installed = False
        compat_install.assert_not_called()

    def test_installer_observes_real_prefill_batch_composition(self):
        class SchedulePolicy:
            def calc_priority(self, waiting_queue, *args, **kwargs):
                return None

        first = request(
            "first",
            10,
            1.0,
            {"cohort_id": "batch-0", "cohort_size": 2},
            version="4",
        )
        second = request(
            "second",
            10,
            1.0,
            {"cohort_id": "batch-0", "cohort_size": 2},
            version="4",
        )

        class Scheduler:
            def __init__(self):
                self.waiting_queue = [first, second]
                self.policy = SchedulePolicy()

            def get_new_batch_prefill(self, *args, **kwargs):
                self.policy.calc_priority(self.waiting_queue)
                return SimpleNamespace(reqs=[first, second])

        modules = {
            "sglang": SimpleNamespace(__path__=[]),
            "sglang.srt": SimpleNamespace(__path__=[]),
            "sglang.srt.managers": SimpleNamespace(__path__=[]),
            "sglang.srt.managers.kv_hints": SimpleNamespace(),
            "sglang.srt.managers.schedule_policy": SimpleNamespace(
                SchedulePolicy=SchedulePolicy
            ),
            "sglang.srt.managers.scheduler": SimpleNamespace(
                Scheduler=Scheduler
            ),
        }
        before = snapshot_stats()
        with patch.dict(sys.modules, modules), patch.dict(
            "os.environ", {"EKV_FUTURE_POLICY": "deadline"}
        ):
            hook._installed = False
            hook.install()
            batch = Scheduler().get_new_batch_prefill()
        hook._installed = False

        self.assertEqual(
            [item.request_id for item in batch.reqs], ["first", "second"]
        )
        after = snapshot_stats()
        self.assertEqual(
            after["complete_cohort_batches"]
            - before["complete_cohort_batches"],
            1,
        )
        self.assertTrue(after["last_cohort_batch"]["complete"])

    def test_installer_observes_dynamic_decode_batch_composition(self):
        class SchedulePolicy:
            def calc_priority(self, waiting_queue, *args, **kwargs):
                return None

        class DecodeMode:
            def is_decode(self):
                return True

        first = request("first", 10, 1.0, {})
        second = request("second", 10, 1.0, {})

        class Scheduler:
            def get_next_batch_to_run(self, *args, **kwargs):
                return SimpleNamespace(
                    batch_to_run=SimpleNamespace(
                        reqs=[first, second], forward_mode=DecodeMode()
                    ),
                    running_batch=object(),
                )

        modules = {
            "sglang": SimpleNamespace(__path__=[]),
            "sglang.srt": SimpleNamespace(__path__=[]),
            "sglang.srt.managers": SimpleNamespace(__path__=[]),
            "sglang.srt.managers.kv_hints": SimpleNamespace(),
            "sglang.srt.managers.schedule_policy": SimpleNamespace(
                SchedulePolicy=SchedulePolicy
            ),
            "sglang.srt.managers.scheduler": SimpleNamespace(
                Scheduler=Scheduler
            ),
        }
        before = snapshot_stats()
        with patch.dict(sys.modules, modules), patch.dict(
            "os.environ", {"EKV_FUTURE_POLICY": "deadline"}
        ):
            hook._installed = False
            hook.install()
            plan = Scheduler().get_next_batch_to_run()
        hook._installed = False

        self.assertEqual(len(plan.batch_to_run.reqs), 2)
        after = snapshot_stats()
        self.assertEqual(after["decode_batches"] - before["decode_batches"], 1)
        self.assertEqual(after["last_decode_batch"]["batch_size"], 2)

    def test_installer_accepts_new_scheduler_plan_and_restores_held_cohort(self):
        class SchedulePolicy:
            def calc_priority(self, waiting_queue, *args, **kwargs):
                return None

        partial = request(
            "partial",
            10,
            1.0,
            {"cohort_id": "batch-0", "cohort_size": 2},
            version="4",
        )
        other = request("other", 10, 1.0, {})

        class Scheduler:
            def __init__(self):
                self.waiting_queue = [partial, other]
                self.observed_queue = None
                self.policy = SchedulePolicy()

            def get_new_batch_prefill(self, running_batch):
                self.policy.calc_priority(self.waiting_queue, running_batch)
                self.observed_queue = list(self.waiting_queue)
                self.waiting_queue.remove(other)
                return SimpleNamespace(
                    batch_to_run=SimpleNamespace(reqs=[other]),
                    running_batch=running_batch,
                )

        modules = {
            "sglang": SimpleNamespace(__path__=[]),
            "sglang.srt": SimpleNamespace(__path__=[]),
            "sglang.srt.managers": SimpleNamespace(__path__=[]),
            "sglang.srt.managers.kv_hints": SimpleNamespace(),
            "sglang.srt.managers.schedule_policy": SimpleNamespace(
                SchedulePolicy=SchedulePolicy
            ),
            "sglang.srt.managers.scheduler": SimpleNamespace(
                Scheduler=Scheduler
            ),
        }
        with patch.dict(sys.modules, modules), patch.dict(
            "os.environ", {"EKV_FUTURE_POLICY": "deadline"}
        ):
            hook._installed = False
            hook.install()
            scheduler = Scheduler()
            plan = scheduler.get_new_batch_prefill("running")
        hook._installed = False

        self.assertEqual(scheduler.observed_queue, [other])
        self.assertEqual(
            [item.request_id for item in plan.batch_to_run.reqs], ["other"]
        )
        self.assertEqual(
            [item.request_id for item in scheduler.waiting_queue], ["partial"]
        )

    def test_installer_hard_gates_preannounced_request_before_native_prefill(self):
        class SchedulePolicy:
            def calc_priority(self, waiting_queue, *args, **kwargs):
                return None

        release_at_ms = math.ceil(time.perf_counter() * 1000.0) + 60_000
        future = request(
            "future",
            10,
            1.0,
            {
                "release_after_ms": 60_000,
                "release_clock": "perf_counter",
                "release_absolute_ms": release_at_ms,
            },
        )
        ready = request("ready", 10, 1.0, {})

        class Scheduler:
            def __init__(self):
                self.waiting_queue = [future, ready]
                self.native_queue = None
                self.policy = SchedulePolicy()

            def get_new_batch_prefill(self, *args, **kwargs):
                self.policy.calc_priority(self.waiting_queue)
                self.native_queue = list(self.waiting_queue)
                self.waiting_queue.remove(ready)
                return SimpleNamespace(reqs=[ready])

        modules = {
            "sglang": SimpleNamespace(__path__=[]),
            "sglang.srt": SimpleNamespace(__path__=[]),
            "sglang.srt.managers": SimpleNamespace(__path__=[]),
            "sglang.srt.managers.kv_hints": SimpleNamespace(),
            "sglang.srt.managers.schedule_policy": SimpleNamespace(
                SchedulePolicy=SchedulePolicy
            ),
            "sglang.srt.managers.scheduler": SimpleNamespace(Scheduler=Scheduler),
        }
        before = snapshot_stats()["hard_release_members_held"]
        with patch.dict(sys.modules, modules), patch.dict(
            "os.environ",
            {
                "EKV_FUTURE_POLICY": "deadline",
                hook.HARD_RELEASE_GATE_ENV: "1",
            },
        ):
            hook._installed = False
            hook.install()
            scheduler = Scheduler()
            scheduler.get_new_batch_prefill()
        hook._installed = False

        self.assertEqual(scheduler.native_queue, [ready])
        self.assertEqual(scheduler.waiting_queue, [future])
        self.assertEqual(
            snapshot_stats()["hard_release_members_held"] - before,
            1,
        )

    def test_gate_sees_grammar_request_added_before_policy(self):
        first = request(
            "first",
            10,
            1.0,
            {"cohort_id": "batch-0", "cohort_size": 2},
            version="4",
        )
        second = request(
            "second",
            10,
            1.0,
            {"cohort_id": "batch-0", "cohort_size": 2},
            version="4",
        )

        class SchedulePolicy:
            def calc_priority(self, waiting_queue, *args, **kwargs):
                return None

        class Scheduler:
            def __init__(self):
                self.waiting_queue = [first]
                self.policy = SchedulePolicy()
                self.observed_queue = None

            def get_new_batch_prefill(self, *args, **kwargs):
                self.waiting_queue.append(second)
                self.policy.calc_priority(self.waiting_queue)
                self.observed_queue = list(self.waiting_queue)
                return SimpleNamespace(reqs=[first, second])

        modules = {
            "sglang": SimpleNamespace(__path__=[]),
            "sglang.srt": SimpleNamespace(__path__=[]),
            "sglang.srt.managers": SimpleNamespace(__path__=[]),
            "sglang.srt.managers.kv_hints": SimpleNamespace(),
            "sglang.srt.managers.schedule_policy": SimpleNamespace(
                SchedulePolicy=SchedulePolicy
            ),
            "sglang.srt.managers.scheduler": SimpleNamespace(
                Scheduler=Scheduler
            ),
        }
        with patch.dict(sys.modules, modules), patch.dict(
            "os.environ", {"EKV_FUTURE_POLICY": "deadline"}
        ):
            hook._installed = False
            hook.install()
            scheduler = Scheduler()
            scheduler.get_new_batch_prefill()
        hook._installed = False

        self.assertEqual(scheduler.observed_queue, [first, second])

    def test_restoring_held_cohort_preserves_original_fcfs_order(self):
        partial = request(
            "partial",
            10,
            1.0,
            {"cohort_id": "batch-0", "cohort_size": 2},
            version="4",
        )
        other = request("other", 10, 1.0, {})

        class SchedulePolicy:
            def calc_priority(self, waiting_queue, *args, **kwargs):
                return None

        class Scheduler:
            def __init__(self):
                self.waiting_queue = [partial, other]
                self.policy = SchedulePolicy()
                self.native_queue = None

            def get_new_batch_prefill(self, *args, **kwargs):
                self.policy.calc_priority(self.waiting_queue)
                self.native_queue = list(self.waiting_queue)

        modules = {
            "sglang": SimpleNamespace(__path__=[]),
            "sglang.srt": SimpleNamespace(__path__=[]),
            "sglang.srt.managers": SimpleNamespace(__path__=[]),
            "sglang.srt.managers.kv_hints": SimpleNamespace(),
            "sglang.srt.managers.schedule_policy": SimpleNamespace(
                SchedulePolicy=SchedulePolicy
            ),
            "sglang.srt.managers.scheduler": SimpleNamespace(
                Scheduler=Scheduler
            ),
        }
        with patch.dict(sys.modules, modules), patch.dict(
            "os.environ", {"EKV_FUTURE_POLICY": "deadline"}
        ):
            hook._installed = False
            hook.install()
            scheduler = Scheduler()
            scheduler.get_new_batch_prefill()
        hook._installed = False

        self.assertEqual(scheduler.native_queue, [other])
        self.assertEqual(scheduler.waiting_queue, [partial, other])

    def test_installer_cohort_gate_error_falls_back_to_native_prefill(self):
        class SchedulePolicy:
            def calc_priority(self, waiting_queue, *args, **kwargs):
                return None

        waiting = request(
            "waiting",
            10,
            1.0,
            {"cohort_id": "batch-0", "cohort_size": 2},
            version="4",
        )

        class Scheduler:
            def __init__(self):
                self.waiting_queue = [waiting]
                self.called = False
                self.policy = SchedulePolicy()

            def get_new_batch_prefill(self, *args, **kwargs):
                self.policy.calc_priority(self.waiting_queue)
                self.called = True

        modules = {
            "sglang": SimpleNamespace(__path__=[]),
            "sglang.srt": SimpleNamespace(__path__=[]),
            "sglang.srt.managers": SimpleNamespace(__path__=[]),
            "sglang.srt.managers.kv_hints": SimpleNamespace(),
            "sglang.srt.managers.schedule_policy": SimpleNamespace(
                SchedulePolicy=SchedulePolicy
            ),
            "sglang.srt.managers.scheduler": SimpleNamespace(
                Scheduler=Scheduler
            ),
        }
        before = snapshot_stats()["cohort_gate_errors"]
        with patch.dict(sys.modules, modules), patch.dict(
            "os.environ",
            {
                "EKV_FUTURE_POLICY": "deadline",
                hook.COHORT_WAIT_TIMEOUT_ENV: "invalid",
            },
        ):
            hook._installed = False
            hook.install()
            scheduler = Scheduler()
            self.assertIsNone(scheduler.get_new_batch_prefill())
        hook._installed = False

        self.assertTrue(scheduler.called)
        self.assertEqual(scheduler.waiting_queue, [waiting])
        self.assertEqual(snapshot_stats()["cohort_gate_errors"] - before, 1)

    def test_stats_are_published_atomically(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future.json"
            with patch.dict("os.environ", {"EKV_FUTURE_STATS_PATH": str(path)}):
                hook._publish_stats(force=True, now_s=10.0)
            self.assertEqual(json.loads(path.read_text()), snapshot_stats())
            self.assertFalse(path.with_name(f"{path.name}.{os.getpid()}.tmp").exists())

    def test_ranked_stats_use_discoverable_rank_shards(self):
        import tempfile
        from pathlib import Path

        hook._stats_identity.update(
            {
                "tp_rank": 3,
                "pp_rank": 1,
                "dp_rank": 2,
                "attn_tp_rank": 3,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future.json"
            with patch.dict("os.environ", {"EKV_FUTURE_STATS_PATH": str(path)}):
                hook._publish_stats(force=True, now_s=10.0)

            paths = hook.find_stats_paths(path)
            self.assertEqual(len(paths), 1)
            self.assertNotEqual(paths[0], path)
            self.assertIn("tp_rank-3", paths[0].name)
            self.assertIn("dp_rank-2", paths[0].name)
            loaded = hook.load_stats(path)
            self.assertEqual(loaded["shard_count"], 1)
            self.assertEqual(
                loaded["stats_shards"][0]["stats"]["stats_identity"]["tp_rank"],
                3,
            )

    def test_load_stats_keeps_rank_documents_and_sums_errors(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "future.json"
            first = base.with_name("future.rank-a.json")
            second = base.with_name("future.rank-b.json")
            first.write_text(
                json.dumps(
                    {
                        "ticks": 8,
                        "prefill_batches": 2,
                        "complete_cohort_batches": 1,
                        "telemetry_errors": 2,
                        "cohort_gate_errors": 1,
                    }
                ),
                encoding="utf-8",
            )
            second.write_text(
                json.dumps(
                    {
                        "ticks": 7,
                        "prefill_batches": 1,
                        "complete_cohort_batches": 0,
                        "telemetry_errors": 3,
                        "cohort_gate_errors": 2,
                    }
                ),
                encoding="utf-8",
            )

            loaded = hook.load_stats(base)

        self.assertEqual(loaded["ticks"], 8)
        self.assertEqual(loaded["telemetry_errors"], 5)
        self.assertEqual(loaded["cohort_gate_errors"], 3)
        self.assertEqual(loaded["shard_count"], 2)
        self.assertEqual(len(loaded["stats_shards"]), 2)


if __name__ == "__main__":
    unittest.main()
