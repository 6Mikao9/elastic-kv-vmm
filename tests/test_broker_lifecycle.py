"""CPU-only regression tests for promotion and broker lifecycle boundaries."""

import unittest
from types import SimpleNamespace as B
from unittest.mock import Mock

import numpy as np
import torch

from adaptive_sparse.migration import commit_demote, commit_promote
from adaptive_sparse.residency_broker import remaining_output_tokens


class BrokerLifecycleTest(unittest.TestCase):
    def test_demotion_removes_published_prefix_source(self):
        request = B()
        remove_source = Mock()
        allocator = B(free_hisparse_indices=Mock())

        class Mapping:
            def __setitem__(self, key, value):
                self.value = value

        coordinator = B(
            padded_buffer_size=2,
            mem_pool_device=B(
                page_size=1,
                full_to_hisparse_device_index_mapping=Mapping(),
            ),
            req_to_device_buffer=torch.zeros((1, 2), dtype=torch.int32),
            req_device_buffer_size=torch.zeros(1, dtype=torch.int32),
            req_device_buffer_tokens=torch.zeros((1, 1, 2), dtype=torch.int32),
            req_device_buffer_token_locs=torch.zeros((1, 1, 2), dtype=torch.int32),
            lru_slots=torch.zeros((1, 1, 2), dtype=torch.int32),
            _lru_init=torch.zeros(2, dtype=torch.int32),
            spec_swap=B(reset=Mock()),
            _skip_first_backup={},
            token_to_kv_pool_allocator=allocator,
            _ekv_remove_demoted_source=remove_source,
        )
        prepared = B(
            status="prepared",
            coordinator=coordinator,
            request=request,
            row=0,
            logical=np.array([7, 8, 9], dtype=np.int64),
            physical=torch.tensor([1, 2, 3, 4], dtype=torch.long),
            committed_len=4,
            retained_rows=(0, 1),
            released_rows=(2, 3),
        )
        publish = Mock()

        released = commit_demote(
            prepared, publish, Mock(), Mock(return_value=0)
        )

        remove_source.assert_called_once_with(request)
        publish.assert_called_once_with(0)
        allocator.free_hisparse_indices.assert_called_once()
        self.assertEqual(released, 0)
        self.assertEqual(prepared.status, "committed")

    def test_promotion_cleanup_uses_nonterminal_hook(self):
        request = B()
        cleanup = Mock(return_value="cleaned")
        register_source = Mock()
        tracker = B(backing=B(backed_bytes=5))
        tracker.trim_quiescent = Mock(return_value=0)

        class Mapping:
            def __init__(self):
                self.values = {}

            def __setitem__(self, key, value):
                self.values[tuple(key.tolist())] = value

            def __getitem__(self, key):
                return self.values[tuple(key.tolist())]

        mapping = Mapping()
        coordinator = B(
            mem_pool_device=B(
                full_to_hisparse_device_index_mapping=mapping,
                adaptive_vmm=tracker,
            ),
            _ekv_cleanup_promotion_source=cleanup,
            _ekv_register_promoted_source=register_source,
            request_finished=Mock(side_effect=AssertionError("terminal cleanup")),
        )
        prepared = B(
            status="prepared",
            coordinator=coordinator,
            request=request,
            logical=np.array([7, 8], dtype=np.int64),
            destination=np.array([41, 42, 43], dtype=np.int64),
            backed_before=5,
            row=1,
        )
        publish = Mock()
        synchronize = Mock()

        growth = commit_promote(prepared, publish, synchronize)

        cleanup.assert_called_once_with(request)
        register_source.assert_called_once_with(request)
        coordinator.request_finished.assert_not_called()
        key = np.array([7, 8], dtype=np.int64)
        self.assertEqual(mapping[key].tolist(), [41, 42])
        publish.assert_called_once_with(1)
        synchronize.assert_called_once_with()
        tracker.trim_quiescent.assert_called_once_with()
        self.assertEqual(growth, 0)
        self.assertEqual(prepared.status, "committed")

    def test_remaining_prediction_is_stable_across_two_promotions(self):
        baseline = 256
        self.assertEqual(remaining_output_tokens(
            predicted_total=512, current_tokens=456, baseline_tokens=baseline,
        ), 312)
        # A demote followed by another promote keeps the staging baseline.
        self.assertEqual(remaining_output_tokens(
            predicted_total=512, current_tokens=556, baseline_tokens=baseline,
        ), 212)
        # Only final completion discards the baseline, after which no later
        # promotion can occur for this request.


if __name__ == "__main__":
    unittest.main()
