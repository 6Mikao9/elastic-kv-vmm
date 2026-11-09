import unittest
from unittest import mock
from types import SimpleNamespace

import torch

from adaptive_sparse.shared_prefix import (
    SharedPrefixRegistry,
    complete_shared_blocks,
    make_prefix_claim,
    resolve_shared_block_locations,
    resolve_shared_block_locations_batched,
)
from adaptive_sparse.request_residency import (
    _swap_shared_prefix_rows,
    install_request_residency,
)


def request(tokens, **values):
    base = {
        "origin_input_ids": tokens,
        "extra_key": None,
        "cache_salt": None,
        "lora_id": None,
        "lora_path": None,
        "adapter_id": None,
        "multimodal_inputs": None,
        "input_embeds": None,
        "hisparse_staging": False,
    }
    base.update(values)
    return SimpleNamespace(**base)


class SharedPrefixTest(unittest.TestCase):
    def test_native_swap_masks_shared_lanes_and_preserves_other_outputs(self):
        registry = SharedPrefixRegistry()
        donor = request([1, 2, 3, 4, 5, 6])
        claim = make_prefix_claim(donor, group="system", shared_tokens=6)
        registry.register_source(0, donor, claim)
        for row in (1, 2, 3):
            registry.register_claim(row, claim)

        def native_swap(*, req_pool_indices, seq_lens, top_k_blocks, layer_id,
                        sparse_block_size):
            self.assertEqual(layer_id, 7)
            self.assertEqual(sparse_block_size, 2)
            self.assertEqual(req_pool_indices.tolist(), [4, 2, 3])
            self.assertEqual(top_k_blocks.tolist(), [[0, 2], [-1, 2], [3, -1]])
            positions = top_k_blocks[..., None] * 2 + torch.arange(2)
            valid = (top_k_blocks[..., None] >= 0) & (positions < seq_lens[:, None, None])
            return (100 * req_pool_indices[:, None, None] + positions).masked_fill(
                ~valid, -1
            ).flatten(1).to(torch.int32)

        ctrl = SimpleNamespace(
            _ekv_shared_prefixes=registry,
            _ekv_residency_stats={},
            swap_in_selected_blocks=mock.Mock(side_effect=native_swap),
        )
        result = _swap_shared_prefix_rows(
            ctrl, torch.arange(1, 7).reshape(1, 6), torch.arange(7) + 10,
            torch.arange(1, 5), torch.full((4,), 6),
            torch.tensor([[0, 1], [0, 2], [3, -1], [0, 2]], dtype=torch.int32),
            [1, 2, 3, 4], 7, 2,
        )
        ctrl.swap_in_selected_blocks.assert_called_once()
        self.assertEqual(result.tolist(), [
            [11, 12, 13, 14],       # all-direct, no native swap
            [11, 12, 204, 205],     # shared lane plus native newest-token block
            [-1, -1, -1, -1],       # invalid selections remain native
            [400, 401, 404, 405],   # no donor, unmodified native inputs
        ])
        stats = ctrl._ekv_residency_stats
        self.assertEqual(stats["shared_prefix_direct_blocks"], 3)
        self.assertEqual(stats["shared_prefix_native_blocks"], 5)
        self.assertEqual(stats["shared_prefix_native_requests"], 3)
        self.assertEqual(stats["shared_prefix_partial_requests"], 1)
        self.assertEqual(stats["shared_prefix_masked_native_blocks"], 1)

    def test_direct_mode_requires_adaptive_hints(self):
        with mock.patch.dict(
            "os.environ",
            {
                "EKV_RESIDENT_LIMIT": "1",
                "EKV_RESIDENT_FREE_RESERVE": "1",
                "EKV_SHARED_PREFIX_REUSE": "direct",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "adaptive resident hints"):
                install_request_residency()

    def test_claim_changes_when_tokens_or_adapter_change(self):
        base = make_prefix_claim(
            request([1, 2, 3, 4]), group="system", shared_tokens=3
        )
        same = make_prefix_claim(
            request([1, 2, 3, 99]), group="system", shared_tokens=3
        )
        different_token = make_prefix_claim(
            request([1, 2, 8, 4]), group="system", shared_tokens=3
        )
        different_adapter = make_prefix_claim(
            request([1, 2, 3, 4], lora_id="adapter-b"),
            group="system",
            shared_tokens=3,
        )

        self.assertEqual(base.fingerprint, same.fingerprint)
        self.assertNotEqual(base.fingerprint, different_token.fingerprint)
        self.assertNotEqual(base.fingerprint, different_adapter.fingerprint)

    def test_claim_rejects_ambiguous_or_invalid_inputs(self):
        self.assertIsNone(
            make_prefix_claim(request([1, 2]), group="system", shared_tokens=0)
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            make_prefix_claim(request([1]), group="system", shared_tokens=2)
        with self.assertRaisesRegex(ValueError, "integers"):
            make_prefix_claim(request([1, True]), group="system", shared_tokens=2)
        self.assertIsNone(
            make_prefix_claim(
                request([1, 2], multimodal_inputs=object()),
                group="system",
                shared_tokens=2,
            )
        )
        self.assertIsNone(
            make_prefix_claim(
                request([1, 2], input_embeds=object()),
                group="system",
                shared_tokens=2,
            )
        )

    def test_only_complete_prefix_blocks_are_shareable(self):
        self.assertEqual(
            complete_shared_blocks(
                [0, 1, 2, -1], seq_len=300, shared_tokens=256, block_size=128
            ),
            (True, True, False, False),
        )

    def test_block_locations_exclude_partial_and_unmapped_blocks(self):
        request_table = torch.tensor([
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22, 23, 24, 25],
        ])
        mapping = torch.arange(30, dtype=torch.long)
        mapping[21] = 0

        valid, physical = resolve_shared_block_locations(
            torch.tensor([0, 1, 2]),
            source_row=0,
            seq_len=torch.tensor(6),
            shared_tokens=4,
            block_size=2,
            request_table=request_table,
            device_mapping=mapping,
        )
        self.assertEqual(valid.tolist(), [True, True, False])
        self.assertEqual(physical[valid].tolist(), [[10, 11], [12, 13]])

        valid, _ = resolve_shared_block_locations(
            torch.tensor([0, 1]),
            source_row=1,
            seq_len=torch.tensor(6),
            shared_tokens=4,
            block_size=2,
            request_table=request_table,
            device_mapping=mapping,
        )
        self.assertEqual(valid.tolist(), [False, True])

        valid, _ = resolve_shared_block_locations(
            torch.tensor([0, 99]),
            source_row=0,
            seq_len=torch.tensor(6),
            shared_tokens=4,
            block_size=2,
            request_table=request_table,
            device_mapping=mapping,
        )
        self.assertEqual(valid.tolist(), [True, False])

    def test_batched_block_locations_match_scalar_resolution(self):
        request_table = torch.tensor([
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22, 23, 24, 25],
            [30, 31, 32, 33, 34, 35],
        ])
        mapping = torch.arange(40, dtype=torch.long)
        mapping[21] = 0
        block_ids = torch.tensor([[0, 1, 2], [1, 2, 3], [0, 1, 9]])
        source_rows = [0, 1, 2]
        prefixes = [4, 4, 2]
        lengths = torch.tensor([6, 4, 6])

        valid, physical = resolve_shared_block_locations_batched(
            block_ids,
            source_rows=source_rows,
            seq_len=lengths,
            shared_tokens=prefixes,
            block_size=2,
            request_table=request_table,
            device_mapping=mapping,
        )
        expected_valid = []
        expected_physical = []
        for row, prefix, length, blocks in zip(
            source_rows, prefixes, lengths, block_ids
        ):
            scalar_valid, scalar_physical = resolve_shared_block_locations(
                blocks,
                source_row=row,
                seq_len=length,
                shared_tokens=prefix,
                block_size=2,
                request_table=request_table,
                device_mapping=mapping,
            )
            expected_valid.append(scalar_valid)
            expected_physical.append(scalar_physical)
        self.assertTrue(torch.equal(valid, torch.stack(expected_valid)))
        self.assertTrue(torch.equal(physical, torch.stack(expected_physical)))

    def test_batched_block_locations_reject_mismatched_metadata(self):
        request_table = torch.arange(12).reshape(2, 6)
        mapping = torch.arange(12)
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            resolve_shared_block_locations_batched(
                torch.tensor([[0, 1]]),
                source_rows=[0],
                seq_len=torch.tensor([6]),
                shared_tokens=[2, 4],
                block_size=2,
                request_table=request_table,
                device_mapping=mapping,
            )

    def test_registry_requires_same_length_and_fingerprint(self):
        registry = SharedPrefixRegistry()
        source = make_prefix_claim(
            request([1, 2, 3, 4]), group="system", shared_tokens=3
        )
        consumer = make_prefix_claim(
            request([1, 2, 3, 8]), group="system", shared_tokens=3
        )
        mismatch = make_prefix_claim(
            request([1, 2, 7, 8]), group="system", shared_tokens=3
        )
        shorter = make_prefix_claim(
            request([1, 2, 3]), group="system", shared_tokens=2
        )

        registry.register_source(0, request([1, 2, 3, 4]), source)
        registry.register_claim(1, consumer)
        self.assertEqual(registry.claim_for(1), consumer)
        self.assertEqual(registry.resolve(1).row, 0)

        registry.register_claim(2, mismatch)
        self.assertIsNone(registry.resolve(2))
        registry.register_claim(3, shorter)
        self.assertIsNone(registry.resolve(3))
        self.assertEqual(registry.snapshot()["inconsistent_group"], 2)

    def test_staging_or_removed_source_falls_back(self):
        registry = SharedPrefixRegistry()
        token_ids = [1, 2, 3, 4]
        source_request = request(token_ids)
        claim = make_prefix_claim(
            source_request, group="system", shared_tokens=3
        )
        registry.register_source(0, source_request, claim)
        registry.register_claim(1, claim)

        source_request.hisparse_staging = True
        self.assertIsNone(registry.resolve(1))
        source_request.hisparse_staging = False
        self.assertEqual(registry.resolve(1).row, 0)

        registry.remove_source(0)
        self.assertIsNone(registry.resolve(1))
        self.assertEqual(registry.snapshot()["active_claims"], 2)

    def test_source_replacement_updates_both_lookup_indexes(self):
        registry = SharedPrefixRegistry()
        source_request = request([1, 2, 3, 4])
        original = make_prefix_claim(
            source_request, group="system", shared_tokens=3
        )
        registry.register_source(0, source_request, original)
        registry.register_claim(1, original)
        self.assertEqual(registry.resolve(1).row, 0)
        self.assertIsNone(registry.resolve(0))
        self.assertEqual(registry.snapshot()["missing_sources"], 1)
        self.assertTrue(
            registry.prefers_geometry_donor(0, group="system", tokens=3)
        )

        replacement = make_prefix_claim(
            request([9, 8, 7, 6]), group="system", shared_tokens=3
        )
        registry.register_source(0, request([9, 8, 7, 6]), replacement)
        self.assertIsNone(registry.resolve(1))
        self.assertEqual(
            registry.snapshot()["inconsistent_group"], 1
        )
        self.assertTrue(
            registry.prefers_geometry_donor(0, group="system", tokens=3)
        )

        registry.register_claim(2, replacement)
        self.assertEqual(registry.resolve(2).row, 0)
        registry.remove_source(0)
        self.assertIsNone(registry.resolve(2))
        self.assertFalse(
            registry.prefers_geometry_donor(0, group="system", tokens=3)
        )

    def test_only_exact_live_host_backed_consumers_count_for_donor(self):
        registry = SharedPrefixRegistry()
        source_request = request([1, 2, 3, 4])
        source = make_prefix_claim(
            source_request, group="system", shared_tokens=3
        )
        consumer = make_prefix_claim(
            request([1, 2, 3, 8]), group="system", shared_tokens=3
        )
        peer = make_prefix_claim(
            request([1, 2, 3, 9]), group="system", shared_tokens=3
        )
        mismatch = make_prefix_claim(
            request([1, 2, 8, 9]), group="system", shared_tokens=3
        )
        shorter = make_prefix_claim(
            request([1, 2, 3]), group="system", shared_tokens=2
        )
        peer_request = request([1, 2, 3, 9])
        registry.register_source(2, peer_request, peer)
        registry.register_source(0, source_request, source)
        registry.register_claim(1, consumer)
        registry.register_claim(3, mismatch)
        registry.register_claim(4, shorter)

        self.assertEqual(registry.active_donor_consumers(0), 1)
        self.assertEqual(registry.active_donor_consumers(2), 0)
        self.assertEqual(registry.resolve(1).row, 0)
        self.assertEqual(registry.active_donor_consumers(3), 0)
        self.assertEqual(registry.snapshot()["donor_consumers"], 1)
        self.assertTrue(
            registry.prefers_geometry_donor(0, group="system", tokens=3)
        )
        self.assertFalse(
            registry.prefers_geometry_donor(2, group="system", tokens=3)
        )

        source_request.hisparse_staging = True
        self.assertEqual(registry.active_donor_consumers(0), 0)
        self.assertEqual(registry.active_donor_consumers(2), 1)
        self.assertTrue(
            registry.prefers_geometry_donor(2, group="system", tokens=3)
        )


if __name__ == "__main__":
    unittest.main()
