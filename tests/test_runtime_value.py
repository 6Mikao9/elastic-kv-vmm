import unittest
from types import SimpleNamespace as NS

from adaptive_sparse.runtime_value import (
    PrefixReuseDemand,
    RuntimeRequestValue,
    attach_prefix_reuse,
    forecast_prefix_reuse,
    merge_demotion_values,
    rank_demotions,
    runtime_values,
)
from adaptive_sparse.sglang_future_hook import ACTION_TYPE, ACTION_VERSION


def hinted_request(request_id, *, payload, received=9.5, current=64, prompt=32):
    action = NS(
        action_type=ACTION_TYPE,
        action_version=ACTION_VERSION,
        payload=payload,
    )
    return NS(
        rid=request_id,
        kv_hints=NS(actions=[action]),
        time_stats=NS(scheduler_recv_time=received),
        origin_input_ids=list(range(prompt)),
        kv=NS(kv_committed_len=current),
    )


class RuntimeValueTest(unittest.TestCase):
    def test_absolute_deadline_does_not_require_scheduler_receive_time(self):
        request = hinted_request(
            "absolute",
            payload={
                "deadline_after_ms": 1000,
                "deadline_clock": "perf_counter",
                "deadline_absolute_ms": 10300,
                "weight": 4,
            },
        )
        request.time_stats = NS()

        values, stats = runtime_values([request], now_s=10.0)

        self.assertEqual(values["absolute"].deadline_remaining_ms, 300)
        self.assertEqual(stats, {"hinted": 1, "invalid": 0})

    def test_extracts_deadline_weight_kind_prefix_and_remaining_work(self):
        request = hinted_request(
            "r",
            payload={
                "deadline_after_ms": 1200,
                "expected_output_tokens": 48,
                "weight": 4,
                "kind": "interactive",
                "prefix_group": "system",
                "shared_prefix_tokens": 24,
                "residence": "resident",
            },
        )
        values, stats = runtime_values([request], now_s=10.0)
        value = values["r"]
        self.assertEqual(value.deadline_remaining_ms, 700)
        self.assertEqual(value.remaining_output_tokens, 16)
        self.assertEqual((value.kind, value.prefix_group), ("interactive", "system"))
        self.assertEqual(value.shared_prefix_tokens, 24)
        self.assertEqual(stats, {"hinted": 1, "invalid": 0})

    def test_invalid_or_unhinted_requests_receive_neutral_values(self):
        invalid = hinted_request("bad", payload={"weight": 0})
        unhinted = NS(rid="plain", kv=NS(kv_committed_len=8))
        values, stats = runtime_values([invalid, unhinted], now_s=10.0)
        self.assertEqual(stats, {"hinted": 0, "invalid": 1})
        self.assertEqual(set(values), {"bad", "plain"})
        self.assertTrue(all(value.retention_score() == 1.0 for value in values.values()))

    def test_deadline_without_scheduler_timestamp_is_neutral_and_invalid(self):
        request = hinted_request(
            "missing-clock",
            payload={"deadline_after_ms": 1000, "weight": 4},
        )
        request.time_stats = NS()

        values, stats = runtime_values([request], now_s=10.0)

        self.assertEqual(stats, {"hinted": 0, "invalid": 1})
        self.assertEqual(values["missing-clock"].retention_score(), 1.0)

    def test_ranking_prefers_low_value_work_and_moves_deadline_last(self):
        rank_candidates = [{
            "normal": (10, 10),
            "weighted": (10, 10),
            "urgent": (100, 1),
        }]
        rank_values = [{
            "normal": merge_demotion_values([RuntimeRequestValue("normal")]),
            "weighted": merge_demotion_values([
                RuntimeRequestValue("weighted", weight=16, remaining_output_tokens=512)
            ]),
            "urgent": merge_demotion_values([
                RuntimeRequestValue("urgent", deadline_remaining_ms=20)
            ]),
        }]
        self.assertEqual(
            rank_demotions(rank_candidates, rank_values, guard_ms=500),
            ["normal", "weighted", "urgent"],
        )

    def test_group_is_guarded_when_any_member_is_guarded(self):
        value = merge_demotion_values([
            RuntimeRequestValue("ordinary"),
            RuntimeRequestValue("deadline", deadline_remaining_ms=25),
        ], guard_ms=500)
        self.assertTrue(value.deadline_guarded)
        self.assertEqual(
            {item.request_id for item in value.request_values},
            {"ordinary", "deadline"},
        )

    def test_retention_score_is_bounded_and_monotone(self):
        ordinary = RuntimeRequestValue("ordinary")
        weighted = RuntimeRequestValue(
            "weighted", weight=16, remaining_output_tokens=512,
            shared_prefix_tokens=4096, active_reuse_consumers=1,
        )
        deadline = RuntimeRequestValue("deadline", deadline_remaining_ms=25)
        self.assertEqual(ordinary.retention_score(), 1.0)
        self.assertGreater(weighted.retention_score(), ordinary.retention_score())
        self.assertGreater(deadline.retention_score(), ordinary.retention_score())
        self.assertLess(weighted.retention_score(), 1_000_000)

    def test_prefix_without_a_consumer_is_not_retained_by_default(self):
        unused = RuntimeRequestValue("unused", shared_prefix_tokens=4096)
        self.assertEqual(unused.retention_score(), 1.0)

    def test_verified_or_forecast_consumers_increase_donor_value(self):
        value = RuntimeRequestValue("donor", shared_prefix_tokens=4096)
        active = attach_prefix_reuse(
            value, PrefixReuseDemand(4096, active_consumers=1)
        )
        future = attach_prefix_reuse(
            value, PrefixReuseDemand(4096, future_consumers=3)
        )
        self.assertGreater(active.retention_score(), value.retention_score())
        self.assertGreater(future.retention_score(), active.retention_score())
        self.assertEqual(active.active_reuse_consumers, 1)
        self.assertEqual(future.future_reuse_consumers, 3)

    def test_forecast_matches_group_and_length_but_excludes_resident_hint(self):
        requests = [
            hinted_request("host", payload={
                "prefix_group": "system", "shared_prefix_tokens": 64,
                "residence": "host_backed",
            }),
            hinted_request("auto", payload={
                "prefix_group": "system", "shared_prefix_tokens": 64,
                "residence": "auto",
            }),
            hinted_request("resident", payload={
                "prefix_group": "system", "shared_prefix_tokens": 64,
                "residence": "resident",
            }),
            hinted_request("other-length", payload={
                "prefix_group": "system", "shared_prefix_tokens": 32,
                "residence": "host_backed",
            }),
        ]
        self.assertEqual(
            forecast_prefix_reuse(requests),
            {("system", 64): 2, ("system", 32): 1},
        )


if __name__ == "__main__":
    unittest.main()
