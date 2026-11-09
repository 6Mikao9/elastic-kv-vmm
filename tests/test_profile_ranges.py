from contextlib import contextmanager
import os
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from adaptive_sparse.profile_ranges import install_native_profile_ranges, profile_range


class ProfileRangesTests(unittest.TestCase):
    def test_disabled_preserves_identity_without_imports(self):
        def original():
            return 7

        with patch.dict(os.environ, {"EKV_PROFILE_RANGES": "0"}):
            with patch("builtins.__import__", side_effect=AssertionError("unexpected import")):
                wrapped = profile_range("test")(original)
                install_native_profile_ranges()

        self.assertIs(wrapped, original)
        self.assertEqual(wrapped(), 7)

    def test_nested_ranges_propagate_exception(self):
        events = []
        expected = ValueError("bad input")

        with self.profile_enabled(events):
            @profile_range("test/inner")
            def inner(*, fail):
                if fail:
                    raise expected
                return 7

            @profile_range("test/outer")
            def outer(fail):
                return inner(fail=fail)

            self.assertEqual(outer(False), 7)
            with self.assertRaises(ValueError) as caught:
                outer(True)

        self.assertIs(caught.exception, expected)
        self.assertEqual(inner.__name__, "inner")
        self.assertEqual(events, [
            ("enter", "ekv/test/outer"),
            ("enter", "ekv/test/inner"),
            ("exit", "ekv/test/inner"),
            ("exit", "ekv/test/outer"),
        ] * 2)

    def test_native_installation_marks_bound_calls_once(self):
        class Coordinator:
            def map_last_loc_to_buffer(self, row):
                return row

            def swap_in_selected_blocks(self, *, row):
                return row

        name = "sglang.srt.managers.hisparse_coordinator"
        module = ModuleType(name)
        module.HiSparseCoordinator = Coordinator
        events = []
        with self.profile_enabled(events), patch.dict(sys.modules, {name: module}):
            install_native_profile_ranges()
            first_map = Coordinator.map_last_loc_to_buffer
            first_swap = Coordinator.swap_in_selected_blocks
            install_native_profile_ranges()
            self.assertIs(Coordinator.map_last_loc_to_buffer, first_map)
            self.assertIs(Coordinator.swap_in_selected_blocks, first_swap)
            coordinator = Coordinator()
            self.assertEqual(coordinator.map_last_loc_to_buffer(5), 5)
            self.assertEqual(coordinator.swap_in_selected_blocks(row=6), 6)

        self.assertEqual(events, [
            ("enter", "ekv/native/map_latest"),
            ("exit", "ekv/native/map_latest"),
            ("enter", "ekv/native/swap"),
            ("exit", "ekv/native/swap"),
        ])

    @contextmanager
    def profile_enabled(self, events):
        @contextmanager
        def record_function(name):
            events.append(("enter", name))
            try:
                yield
            finally:
                events.append(("exit", name))

        profiler = ModuleType("torch.profiler")
        profiler.record_function = record_function
        with patch.dict(os.environ, {"EKV_PROFILE_RANGES": "1"}):
            with patch.dict(sys.modules, {"torch.profiler": profiler}):
                yield
