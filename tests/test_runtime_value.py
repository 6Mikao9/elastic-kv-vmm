import unittest
from types import SimpleNamespace as NS

from adaptive_sparse.runtime_value import (
    RuntimeRequestValue,
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
            shared_prefix_tokens=4096,
        )
        deadline = RuntimeRequestValue("deadline", deadline_remaining_ms=25)
        self.assertEqual(ordinary.retention_score(), 1.0)
        self.assertGreater(weighted.retention_score(), ordinary.retention_score())
        self.assertGreater(deadline.retention_score(), ordinary.retention_score())
        self.assertLess(weighted.retention_score(), 1_000_000)


if __name__ == "__main__":
    unittest.main()
