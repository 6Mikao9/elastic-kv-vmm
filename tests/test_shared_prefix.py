import unittest
from unittest import mock
from types import SimpleNamespace

import torch

from adaptive_sparse.shared_prefix import (
    SharedPrefixRegistry,
    complete_shared_blocks,
    make_prefix_claim,
    resolve_shared_block_locations,
)
from adaptive_sparse.request_residency import install_request_residency


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


if __name__ == "__main__":
    unittest.main()
