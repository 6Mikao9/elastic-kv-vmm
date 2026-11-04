from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from experimental.resident_arena import VmmResidentArena
from experimental.virtual_layout import VirtualKVLayout


class VirtualLayoutTest(unittest.TestCase):
    def test_alignment_and_crossing_rows(self):
        layout = VirtualKVLayout(2, 3, 3, chunk_bytes=8)
        self.assertEqual(layout.capacity_bytes, 56)
        self.assertEqual(layout.request_offset(1), 32)
        self.assertEqual(layout.token_offset(0, 1), 14)
        self.assertEqual(layout.prefix_chunks(0, 0), set())
        self.assertEqual(layout.selected_chunks(0, []), set())
        self.assertEqual(layout.selected_chunks(0, [1, 1]), {1, 2})
        self.assertEqual(layout.prefix_chunks(0, 3), {1, 2, 3})
        self.assertEqual(layout.prefix_chunks(1, 3), {4, 5, 6})

    def test_rejects_invalid_fields_and_indices(self):
        fields = dict(max_requests=2, max_tokens=3, row_elements=3, itemsize=2, chunk_bytes=8)
        for field in fields:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    VirtualKVLayout(**(fields | {field: 0}))
                with self.assertRaises(TypeError):
                    VirtualKVLayout(**(fields | {field: True}))
        layout = VirtualKVLayout(**fields)
        for index in (-1, 2):
            with self.assertRaises(ValueError):
                layout.selected_chunks(index, [])
        for position in (-1, 3):
            with self.assertRaises(ValueError):
                layout.selected_chunks(0, [position])
        for length in (-1, 4):
            with self.assertRaises(ValueError):
                layout.prefix_chunks(0, length)


class ResidentArenaTest(unittest.TestCase):
    def setUp(self):
        self.calls = Mock()
        self.backing = self.calls.factory.return_value
        self.layout = VirtualKVLayout(2, 5, 3, chunk_bytes=8)
        self.arena = VmmResidentArena(self.layout, backing_factory=self.calls.factory,
                                      synchronize=self.calls.sync)

    def test_lifecycle_preserves_live_tail_chunk(self):
        from unittest.mock import call

        self.arena.reserve(0, 3)
        self.arena.reserve(0, 1)
        self.arena.reserve(1, 1)
        self.arena.resize_quiescent(0, 2)
        self.arena.close()
        self.arena.close()
        self.assertEqual(self.calls.mock_calls, [
            call.factory(72, chunk_bytes=8, device=0), call.factory().ensure({0}),
            call.factory().ensure({1, 2, 3}), call.factory().ensure({5}),
            call.sync(), call.factory().reclaim({3}, synchronize=False),
            call.sync(), call.factory().close(),
        ])
        with self.assertRaises(RuntimeError):
            self.arena.reserve(0, 1)

    def test_view_checks_extent_and_dtype(self):
        dtype = SimpleNamespace(itemsize=2)
        self.arena.reserve(1, 2)
        self.arena.row_view(1, 2, dtype)
        self.backing.raw.narrow.assert_called_once_with(0, 40, 12)
        self.backing.raw.narrow.return_value.view.assert_called_once_with(dtype)
        self.backing.raw.narrow.return_value.view.return_value.reshape.assert_called_once_with(2, 3)
        with self.assertRaises(ValueError):
            self.arena.row_view(1, 3, dtype)
        with self.assertRaises(ValueError):
            self.arena.row_view(1, 2, SimpleNamespace(itemsize=4))

    def test_backing_failure_stops_use_and_close_still_syncs(self):
        self.backing.ensure.side_effect = RuntimeError("allocation failed")
        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            self.arena.reserve(0, 2)
        self.assertEqual(self.arena._lengths[0], 0)
        with self.assertRaises(RuntimeError):
            self.arena.row_view(0, 0, SimpleNamespace(itemsize=2))
        self.arena.close()
        self.calls.sync.assert_called_once()
        self.backing.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
