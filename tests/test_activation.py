from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from elastic_kv import activation


class ActivationTest(unittest.TestCase):
    def test_prefill_protects_live_and_prospective_chunks_and_resamples(self):
        class Arena:
            mapped_chunks = {0, 1, 2, 3}
            planes = 2

            @property
            def backed_bytes(self):
                return len(self.mapped_chunks) * 20

            def reclaim(self, indices, *, synchronize):
                self.reclaimed = indices
                self.synchronize = synchronize
                self.mapped_chunks -= indices

        arena = Arena()
        cuda = Mock()
        cuda.mem_get_info.side_effect = [(5, 1000), (25, 1000), (32, 1000)]
        cuda.device.return_value = nullcontext()
        ctrl = SimpleNamespace(
            pool=SimpleNamespace(device="cuda:0"), arenas=[arena], backed=80,
            live=SimpleNamespace(nonzero=lambda: ([0, 1],)),
            chunks_for=lambda ids: ids, slot_bytes=10, chunk_bytes=10, counters={},
        )
        with patch.multiple(activation, _POLICY="both", _HEADROOM=20, _CUDA=cuda), patch.object(
            activation, "_CONTEXT", SimpleNamespace(prefill=True)
        ):
            activation.before_admit(ctrl, [(2, 3), (4, 5)])
        self.assertEqual(arena.reclaimed, {3})
        self.assertFalse(arena.synchronize)
        cuda.synchronize.assert_called_once_with("cuda:0")
        cuda.empty_cache.assert_called_once()
        self.assertEqual(ctrl.backed, 60)
        self.assertEqual(ctrl.counters["activation_trim_bytes"], 20)
        self.assertEqual(ctrl.counters["activation_flush_freed_bytes"], 7)
        self.assertEqual(ctrl.counters["activation_pressure_shortfall_bytes"], 8)

    def test_decode_recycles_only_on_missing_backing(self):
        arena = SimpleNamespace(mapped_chunks={0})
        cuda = Mock()
        cuda.device.return_value = nullcontext()
        cuda.mem_get_info.side_effect = [(1, 100), (12, 100)]
        ctrl = SimpleNamespace(
            pool=SimpleNamespace(device="cuda:0"), arenas=[arena],
            slot_bytes=5, chunk_bytes=10, counters={},
        )
        with patch.multiple(activation, _POLICY="both", _HEADROOM=99, _CUDA=cuda):
            activation.before_admit(ctrl, [(1, 2)])
            cuda.mem_get_info.assert_not_called()
            activation.before_admit(ctrl, [(2, 3)])
        cuda.synchronize.assert_not_called()
        cuda.empty_cache.assert_called_once()
        self.assertEqual(ctrl.counters["activation_flush_freed_bytes"], 11)
        self.assertNotIn("activation_pressure_unresolved_calls", ctrl.counters)
