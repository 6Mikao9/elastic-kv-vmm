import unittest
from dataclasses import replace
from types import SimpleNamespace as NS

from adaptive_sparse.future_prefill import (
    FuturePrefillConfig,
    effective_prefill_headroom,
    plan_future_prefill,
    reconcile_tp_plan,
)


def request(
    rid,
    *,
    received=1.0,
    prompt=10,
    release_after_ms=0,
    expected_output_tokens=5,
    residence="resident",
    kind="generic",
    prefill_tokens_per_s=None,
    decode_tokens_per_s=None,
    prefix_group=None,
    shared_prefix_tokens=0,
):
    payload = {
        "release_after_ms": release_after_ms,
        "residence": residence,
        "kind": kind,
    }
    if expected_output_tokens is not None:
        payload["expected_output_tokens"] = expected_output_tokens
    if prefill_tokens_per_s is not None:
        payload["prefill_tokens_per_s"] = prefill_tokens_per_s
    if decode_tokens_per_s is not None:
        payload["decode_tokens_per_s"] = decode_tokens_per_s
    if prefix_group is not None:
        payload["prefix_group"] = prefix_group
        payload["shared_prefix_tokens"] = shared_prefix_tokens
    action = NS(
        action_type="elastic.future",
        action_version=(
            "3"
            if prefill_tokens_per_s is not None or decode_tokens_per_s is not None
            else "1"
        ),
        payload=payload,
    )
    return NS(
        rid=rid,
        kv_hints=NS(actions=[action]),
        origin_input_ids=list(range(prompt)),
        num_matched_prefix_tokens=0,
        time_stats=NS(scheduler_recv_time=received, wait_queue_entry_time=0.0),
    )


class FuturePrefillPlanTest(unittest.TestCase):
    def config(self, **changes):
        values = dict(
            mode="dry_run",
            horizon_ms=2_000,
            max_reserve_bytes=1_000_000,
            activation_bytes_per_token=2,
            auto_resident_output_tokens=20,
            sparse_hot_tokens=8,
        )
        values.update(changes)
        return FuturePrefillConfig(**values)

    def test_resident_plan_counts_full_sequence_and_prefill_activation(self):
        plan = plan_future_prefill(
            [request("resident", prompt=10, expected_output_tokens=5)],
            now_s=1.1,
            kv_bytes_per_token=10,
            config=self.config(),
        )
        entry = plan.entries[0]
        self.assertEqual(entry.residence, "resident")
        self.assertEqual(entry.kv_bytes, 150)
        self.assertEqual(entry.activation_bytes, 20)
        self.assertEqual(plan.reserve_bytes, 170)

    def test_kind_specific_activation_overrides_the_global_rate(self):
        config = self.config(
            activation_bytes_per_token=2,
            kind_activation_bytes_per_token=(("tool", 7), ("chat", 3)),
        )
        plan = plan_future_prefill(
            [
                request("tool", prompt=10, kind="tool"),
                request("chat", prompt=10, kind="chat"),
                request("other", prompt=10, kind="other"),
            ],
            now_s=1.0,
            kv_bytes_per_token=10,
            config=config,
        )
        by_id = {entry.request_id: entry for entry in plan.entries}
        self.assertEqual(by_id["tool"].activation_bytes, 70)
        self.assertEqual(by_id["chat"].activation_bytes, 30)
        self.assertEqual(by_id["other"].activation_bytes, 20)

    def test_host_backed_plan_is_bounded_by_sparse_hot_tokens(self):
        plan = plan_future_prefill(
            [request("sparse", prompt=10, expected_output_tokens=99,
                     residence="host_backed")],
            now_s=1.0,
            kv_bytes_per_token=10,
            config=self.config(),
        )
        self.assertEqual(plan.entries[0].kv_bytes, 80)
        self.assertEqual(plan.entries[0].activation_bytes, 20)

    def test_auto_uses_output_threshold(self):
        plan = plan_future_prefill(
            [
                request("short", prompt=10, expected_output_tokens=19,
                        residence="auto"),
                request("long", prompt=10, expected_output_tokens=20,
                        residence="auto"),
            ],
            now_s=1.0,
            kv_bytes_per_token=10,
            config=self.config(),
        )
        self.assertEqual(
            [entry.residence for entry in plan.entries],
            ["host_backed", "resident"],
        )
        self.assertEqual(
            [entry.kv_bytes for entry in plan.entries], [80, 300]
        )

    def test_horizon_and_release_order_are_deterministic(self):
        plan = plan_future_prefill(
            [
                request("later", received=1.0, release_after_ms=2_500),
                request("ready", received=1.0, release_after_ms=500),
                request("waiting", received=1.0, release_after_ms=1_500),
            ],
            now_s=1.2,
            kv_bytes_per_token=1,
            config=self.config(),
        )
        self.assertEqual(
            [entry.request_id for entry in plan.entries], ["ready", "waiting"]
        )
        self.assertEqual(plan.deferred_count, 1)

    def test_nonoverlapping_rate_windows_do_not_add_to_peak(self):
        plan = plan_future_prefill(
            [
                request(
                    "first",
                    release_after_ms=0,
                    prefill_tokens_per_s=1_000,
                    decode_tokens_per_s=1_000,
                ),
                request(
                    "second",
                    release_after_ms=100,
                    prefill_tokens_per_s=1_000,
                    decode_tokens_per_s=1_000,
                ),
            ],
            now_s=1.0,
            kv_bytes_per_token=10,
            config=self.config(),
        )

        self.assertEqual(sum(entry.kv_bytes for entry in plan.entries), 300)
        self.assertEqual(plan.kv_bytes, 150)
        self.assertEqual(plan.activation_bytes, 20)
        self.assertEqual(plan.reserve_bytes, 170)
        self.assertEqual(
            plan.overlap_policy, "rate-aware-or-conservative"
        )

    def test_overlapping_rate_windows_use_concurrent_peak(self):
        plan = plan_future_prefill(
            [
                request(
                    "first",
                    release_after_ms=0,
                    prefill_tokens_per_s=1_000,
                    decode_tokens_per_s=1_000,
                ),
                request(
                    "second",
                    release_after_ms=5,
                    prefill_tokens_per_s=1_000,
                    decode_tokens_per_s=1_000,
                ),
            ],
            now_s=1.0,
            kv_bytes_per_token=10,
            config=self.config(),
        )

        self.assertEqual(plan.kv_bytes, 300)
        self.assertEqual(plan.activation_bytes, 40)

    def test_shared_prefix_is_counted_once_while_members_overlap(self):
        plan = plan_future_prefill(
            [
                request(
                    request_id,
                    prompt=10,
                    prefill_tokens_per_s=1_000,
                    decode_tokens_per_s=1_000,
                    prefix_group="group",
                    shared_prefix_tokens=4,
                )
                for request_id in ("a", "b", "c")
            ],
            now_s=1.0,
            kv_bytes_per_token=10,
            config=self.config(),
        )

        self.assertEqual(sum(entry.kv_bytes for entry in plan.entries), 450)
        self.assertEqual(plan.kv_bytes, 370)
        self.assertEqual(
            [entry.shared_prefix_bytes for entry in plan.entries],
            [40, 40, 40],
        )

    def test_shared_prefix_credit_uses_shortest_common_group_prefix(self):
        plan = plan_future_prefill(
            [
                request(
                    request_id,
                    prompt=10,
                    prefill_tokens_per_s=1_000,
                    decode_tokens_per_s=1_000,
                    prefix_group="group",
                    shared_prefix_tokens=shared,
                )
                for request_id, shared in (("a", 1), ("b", 2), ("c", 6))
            ],
            now_s=1.0,
            kv_bytes_per_token=10,
            config=self.config(),
        )

        self.assertEqual(sum(entry.kv_bytes for entry in plan.entries), 450)
        self.assertEqual(plan.kv_bytes, 430)

    def test_host_backed_members_do_not_receive_shared_prefix_credit(self):
        plan = plan_future_prefill(
            [
                request(
                    request_id,
                    prompt=10,
                    expected_output_tokens=20,
                    residence="host_backed",
                    prefill_tokens_per_s=1_000,
                    decode_tokens_per_s=1_000,
                    prefix_group="group",
                    shared_prefix_tokens=4,
                )
                for request_id in ("a", "b")
            ],
            now_s=1.0,
            kv_bytes_per_token=10,
            config=self.config(),
        )

        self.assertEqual([entry.kv_bytes for entry in plan.entries], [80, 80])
        self.assertEqual(plan.kv_bytes, 160)

    def test_shared_prefix_credit_only_counts_resident_members(self):
        plan = plan_future_prefill(
            [
                request(
                    "resident-a",
                    prompt=10,
                    expected_output_tokens=5,
                    residence="resident",
                    prefill_tokens_per_s=1_000,
                    decode_tokens_per_s=1_000,
                    prefix_group="group",
                    shared_prefix_tokens=4,
                ),
                request(
                    "resident-b",
                    prompt=10,
                    expected_output_tokens=5,
                    residence="resident",
                    prefill_tokens_per_s=1_000,
                    decode_tokens_per_s=1_000,
                    prefix_group="group",
                    shared_prefix_tokens=4,
                ),
                request(
                    "host-backed",
                    prompt=10,
                    expected_output_tokens=20,
                    residence="host_backed",
                    prefill_tokens_per_s=1_000,
                    decode_tokens_per_s=1_000,
                    prefix_group="group",
                    shared_prefix_tokens=4,
                ),
            ],
            now_s=1.0,
            kv_bytes_per_token=10,
            config=self.config(),
        )

        self.assertEqual(
            [entry.kv_bytes for entry in plan.entries], [150, 150, 80]
        )
        self.assertEqual(plan.kv_bytes, 340)

    def test_missing_and_invalid_hints_do_not_change_reservation(self):
        no_hint = request("none")
        no_hint.kv_hints = NS(actions=[])
        invalid = request("invalid")
        invalid.kv_hints.actions[0].payload["unknown"] = 1
        unestimated = request(
            "unestimated", expected_output_tokens=None, residence="resident"
        )
        plan = plan_future_prefill(
            [no_hint, invalid, unestimated],
            now_s=1.0,
            kv_bytes_per_token=1,
            config=self.config(),
        )
        self.assertFalse(plan.entries)
        self.assertEqual(plan.missing_hint_count, 1)
        self.assertEqual(plan.invalid_hint_count, 1)
        self.assertEqual(plan.unestimated_count, 1)

    def test_request_without_stable_id_is_not_reserved(self):
        item = request("temporary")
        del item.rid
        plan = plan_future_prefill(
            [item], now_s=1.0, kv_bytes_per_token=1, config=self.config()
        )
        self.assertFalse(plan.entries)
        self.assertEqual(plan.invalid_hint_count, 1)

    def test_explicit_host_backed_without_output_reserves_prompt_hot_set(self):
        plan = plan_future_prefill(
            [request("bounded", prompt=20, expected_output_tokens=None,
                     residence="host_backed")],
            now_s=1.0,
            kv_bytes_per_token=10,
            config=self.config(),
        )
        self.assertEqual(plan.entries[0].expected_output_tokens, 0)
        self.assertEqual(plan.entries[0].kv_bytes, 80)
        self.assertEqual(plan.entries[0].activation_bytes, 40)

    def test_byte_cap_skips_requests_that_cannot_fit_completely(self):
        plan = plan_future_prefill(
            [
                request("small", prompt=1, expected_output_tokens=1),
                request("large", prompt=500, expected_output_tokens=500),
                request("next", prompt=1, expected_output_tokens=1),
            ],
            now_s=1.0,
            kv_bytes_per_token=10,
            config=self.config(max_reserve_bytes=200),
        )
        self.assertEqual(
            [entry.request_id for entry in plan.entries], ["small", "next"]
        )
        self.assertEqual(plan.capped_count, 1)
        self.assertLessEqual(plan.reserve_bytes, 200)

    def test_off_mode_is_a_no_op_even_with_invalid_inputs(self):
        plan = plan_future_prefill(
            [object()], now_s=1.0, kv_bytes_per_token=1,
            config=self.config(mode="off"),
        )
        self.assertEqual(plan.entries, ())
        self.assertEqual(plan.waiting_count, 0)

    def test_configuration_rejects_invalid_bounds(self):
        with self.assertRaisesRegex(ValueError, "unknown future prefill mode"):
            self.config(mode="predict")
        with self.assertRaisesRegex(ValueError, "max_reserve_bytes"):
            self.config(max_reserve_bytes=-1)
        with self.assertRaisesRegex(ValueError, "auto_resident_output_tokens"):
            self.config(auto_resident_output_tokens=0)
        with self.assertRaisesRegex(ValueError, "kind activation"):
            self.config(kind_activation_bytes_per_token=(("tool", -1),))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.config(
                kind_activation_bytes_per_token=(("tool", 1), ("tool", 2))
            )
        with self.assertRaisesRegex(ValueError, "tuple of pairs"):
            self.config(kind_activation_bytes_per_token={"tool": 1})

    def test_tp_plan_mismatch_falls_back_atomically(self):
        plan = plan_future_prefill(
            [request("resident")], now_s=1.0, kv_bytes_per_token=10,
            config=self.config(),
        )
        accepted, fallback = reconcile_tp_plan(
            plan, [plan.signature, plan.signature]
        )
        self.assertIs(accepted, plan)
        self.assertFalse(fallback)

        rejected, fallback = reconcile_tp_plan(
            plan, [plan.signature, ("different", 1, ())]
        )
        self.assertEqual(rejected.mode, "dry_run")
        self.assertEqual(rejected.reserve_bytes, 0)
        self.assertTrue(fallback)

    def test_tp_signature_distinguishes_kv_and_activation_split(self):
        plan = plan_future_prefill(
            [request("resident")], now_s=1.0, kv_bytes_per_token=10,
            config=self.config(),
        )
        entry = plan.entries[0]
        kv_only = replace(
            plan,
            entries=(replace(entry, kv_bytes=170, activation_bytes=0),),
        )
        activation_only = replace(
            plan,
            entries=(replace(entry, kv_bytes=0, activation_bytes=170),),
        )

        self.assertEqual(kv_only.reserve_bytes, activation_only.reserve_bytes)
        rejected, fallback = reconcile_tp_plan(
            kv_only, [activation_only.signature]
        )
        self.assertTrue(fallback)
        self.assertEqual(rejected.reserve_bytes, 0)

    def test_effective_headroom_adds_kv_and_uses_one_activation_estimate(self):
        plan = plan_future_prefill(
            [request("resident", prompt=10, expected_output_tokens=5)],
            now_s=1.0,
            kv_bytes_per_token=10,
            config=self.config(activation_bytes_per_token=3),
        )
        self.assertEqual(plan.kv_bytes, 150)
        self.assertEqual(plan.activation_bytes, 30)
        self.assertEqual(effective_prefill_headroom(plan, 20, "enforce"), 180)
        self.assertEqual(effective_prefill_headroom(plan, 50, "enforce"), 200)
        self.assertEqual(effective_prefill_headroom(plan, 50, "dry_run"), 0)


if __name__ == "__main__":
    unittest.main()
