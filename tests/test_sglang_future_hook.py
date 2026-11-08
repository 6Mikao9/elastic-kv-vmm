from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from adaptive_sparse import sglang_future_hook as hook
from adaptive_sparse.sglang_future_hook import apply_to_waiting_queue, order_requests, snapshot_stats


def request(name, prompt, created, payload):
    action = SimpleNamespace(
        action_type="elastic.future",
        action_version="1",
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
        ready = request("ready", 10, 1.0, {"deadline_after_ms": 0})
        future = request("future", 10, 1.0, {"release_after_ms": 2000})
        ordered = order_requests([future, ready], now_s=1.1)
        self.assertEqual([item.request_id for item in ordered], ["ready", "future"])

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
        first = request("first", 10, 1.0, {"deadline_after_ms": 100})
        second = request("second", 10, 1.0, {"deadline_after_ms": 100})
        ordered = order_requests([first, second], now_s=1.0)
        self.assertEqual([item.request_id for item in ordered], ["first", "second"])

    def test_missing_matched_prefix_count_is_treated_as_zero(self):
        item = request("fresh", 10, 1.0, {"deadline_after_ms": 100})
        item.num_matched_prefix_tokens = None
        self.assertEqual(order_requests([item], now_s=1.0), [item])

    def test_deadline_uses_scheduler_receive_age(self):
        old = request("old", 100, 0.1, {"deadline_after_ms": 1000})
        new = request("new", 100, 0.9, {"deadline_after_ms": 1200})
        ordered = order_requests(
            [new, old], now_s=1.0, prefill_tokens_per_s=1000, decode_tokens_per_s=1000
        )
        self.assertEqual([item.request_id for item in ordered], ["old", "new"])

    def test_native_priority_tier_is_never_crossed(self):
        high = request("high", 100, 1.0, {"deadline_after_ms": 5000})
        low = request("low", 10, 1.0, {"deadline_after_ms": 0})
        high.priority = 10
        low.priority = 0
        ordered = order_requests([high, low], now_s=1.1, priority_sign=-1)
        self.assertEqual([item.request_id for item in ordered], ["high", "low"])

    def test_off_policy_does_not_import_sglang(self):
        with patch.dict("os.environ", {"EKV_FUTURE_POLICY": "off"}), patch.dict(
            sys.modules, {}, clear=False
        ):
            sys.modules.pop("sglang", None)
            hook._installed = False
            hook.install()
        self.assertFalse(hook._installed)
        self.assertNotIn("sglang", sys.modules)

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
            "sglang.srt.managers.kv_hints": SimpleNamespace(),
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
        with patch.dict(sys.modules, modules), patch.dict(
            "os.environ", {"EKV_FUTURE_POLICY": "deadline"}
        ):
            hook._installed = False
            hook.install()
            queue = [short, long]
            SchedulePolicy().calc_priority(queue)
        hook._installed = False
        self.assertEqual(calls, ["native"])
        self.assertEqual([item.request_id for item in queue], ["short", "long"])

    def test_stats_are_published_atomically(self):
        import tempfile
        from pathlib import Path
        import json

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future.json"
            with patch.dict("os.environ", {"EKV_FUTURE_STATS_PATH": str(path)}):
                hook._publish_stats(force=True, now_s=10.0)
            self.assertEqual(json.loads(path.read_text()), snapshot_stats())
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
