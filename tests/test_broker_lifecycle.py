"""CPU-only regression tests for promotion and broker lifecycle boundaries."""

import unittest
from types import SimpleNamespace as B
from unittest.mock import Mock

import numpy as np

from adaptive_sparse.migration import commit_promote
from adaptive_sparse.residency_broker import remaining_output_tokens


class BrokerLifecycleTest(unittest.TestCase):
    def test_promotion_cleanup_uses_nonterminal_hook(self):
        request = B()
        cleanup = Mock(return_value="cleaned")
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
