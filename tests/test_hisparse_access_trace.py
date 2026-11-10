import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from adaptive_sparse import hisparse_access_trace
from adaptive_sparse.hisparse_access_trace import TraceLimits, _patch_coordinator, _validate_state
from adaptive_sparse.minimax_runtime import install_minimax_runtime


class _Tensor:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            result = self.data
            for part in key:
                result = result[part]
        else:
            result = self.data[key]
        return _Tensor(result)

    def cpu(self):
        return self

    def tolist(self):
        return self.data.copy()

    def item(self):
        return self.data


def _fake_coordinator(directory, limits, rank=0):
    class Coordinator:
        def __init__(self):
            self.is_m3_hisparse = True
            self.spec_swap = SimpleNamespace(enabled=False)
            self.skip_io = False
            self.compress_ratio = 1
            self.device_buffer_size = 4
            self.mem_pool_device = SimpleNamespace(start_layer=5)
            self.num_real_reqs = _Tensor(1)
            self.req_device_buffer_tokens = _Tensor([[[2, 1, -1, 6, 99]]])

        def alloc_device_buffer(self, req):
            return "allocated"

        def request_finished(self, req):
            return "finished"

        def swap_in_selected_blocks(self, slots, seq_lens, blocks, layer, size):
            self.req_device_buffer_tokens = _Tensor([[[3, 4, 5, 0, 99]]])
            return "swapped"

        def destroy(self):
            return "destroyed"

    _patch_coordinator(Coordinator, directory, limits, lambda: None, lambda: rank)
    return Coordinator()


def _request(rid):
    return SimpleNamespace(rid=rid, kv=SimpleNamespace(req_pool_idx=0))


def _swap(coordinator, seq_len, blocks, layer=5):
    return coordinator.swap_in_selected_blocks(
        _Tensor([0, 999]), _Tensor([seq_len]), _Tensor([blocks]), layer, 2
    )


class HiSparseAccessTraceTest(unittest.TestCase):
    def test_installed_preflight_uses_resolved_runtime_before_loading_weights(self):
        load_calls = []

        class ModelRunner:
            def load_model(self, model):
                load_calls.append(model)
                return "loaded"

        class Coordinator:
            pass

        legacy_args = SimpleNamespace(device=None, disable_radix_cache=None, cuda_graph_config=None)
        graph_config = {"decode": "disabled", "prefill": "disabled"}
        device = SimpleNamespace(device="cuda:0")
        memory = SimpleNamespace(disable_radix_cache=True)
        disagg = SimpleNamespace(disaggregation_mode="null")
        graph = SimpleNamespace(cuda_graph_config=graph_config, fully_disabled=True)
        coordinator = SimpleNamespace(__file__=__file__, HiSparseCoordinator=Coordinator)
        modules = {
            "torch": SimpleNamespace(
                device=lambda value: SimpleNamespace(type=value.split(":")[0]),
                distributed=SimpleNamespace(get_rank=lambda: 0),
            ),
            "sglang.srt.managers": SimpleNamespace(hisparse_coordinator=coordinator),
            "sglang.srt.model_executor.cuda_graph_config": SimpleNamespace(
                cuda_graph_fully_disabled=lambda: graph.fully_disabled,
            ),
            "sglang.srt.model_executor.model_runner": SimpleNamespace(ModelRunner=ModelRunner),
            "sglang.srt.runtime_context": SimpleNamespace(
                get_server_args=lambda: legacy_args,
                get_device=lambda: device,
                get_memory=lambda: memory,
                get_exec=lambda: SimpleNamespace(graph=graph),
                get_disagg=lambda: disagg,
                get_schedule=lambda: SimpleNamespace(disable_overlap_schedule=True),
                get_spec=lambda: SimpleNamespace(speculative_algorithm=None),
            ),
        }
        source = b"audited source"
        digest = hashlib.sha256(source).hexdigest()
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {}, clear=True),
            patch.object(Path, "read_bytes", return_value=source),
            patch.multiple(
                hisparse_access_trace, COORDINATOR_SHA256=digest, KERNEL_SHA256=digest,
                WRAPPER_SHA256=digest,
            ),
            patch.object(hisparse_access_trace, "_patch_coordinator"),
        ):
            hisparse_access_trace.install_hisparse_access_trace("/trace")
            runner = ModelRunner()
            self.assertEqual(runner.load_model("valid"), "loaded")
            for config, field, value, diagnostic in (
                (memory, "disable_radix_cache", False, "disable_radix_cache=False"),
                (graph, "fully_disabled", False, "cuda_graph_fully_disabled=False"),
                (device, "device", "cpu", "device_type='cpu'"),
                (disagg, "disaggregation_mode", "decode", "disaggregation_mode='decode'"),
            ):
                with self.subTest(field=field), patch.object(config, field, value):
                    with self.assertRaisesRegex(RuntimeError, "before ModelRunner.load_model") as error:
                        runner.load_model("invalid")
                    self.assertIn(diagnostic, str(error.exception))
                    self.assertIn(f"cuda_graph_config={graph_config!r}", str(error.exception))
            self.assertEqual(load_calls, ["valid"])

    def test_guard_reports_resolved_state_before_loading_weights(self):
        state = dict(
            phase="before ModelRunner.load_model", graph_disabled=True,
            overlap_disabled=True, radix_disabled=True, speculative_algorithm=None,
            device="cuda:0", device_type="cuda", disaggregation_mode="null",
            graph_config={"decode": "disabled", "prefill": "disabled"},
        )
        _validate_state(**state)
        for field, value in (
            ("graph_disabled", False),
            ("overlap_disabled", False),
            ("radix_disabled", False),
            ("speculative_algorithm", "EAGLE"),
            ("device_type", "cpu"),
            ("disaggregation_mode", "decode"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(RuntimeError, "before ModelRunner.load_model") as error:
                    _validate_state(**(state | {field: value}))
                self.assertIn("cuda_graph_config=", str(error.exception))
                self.assertIn("device='cuda:0'", str(error.exception))

    def test_prestate_padding_latest_short_path_and_reused_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            coordinator = _fake_coordinator(
                directory, TraceLimits(layers=frozenset({5}), max_steps=1)
            )
            first = _request("first")
            second = _request("second")
            self.assertEqual(coordinator.alloc_device_buffer(first), "allocated")
            self.assertEqual(_swap(coordinator, 7, [1, -1, 3, 1, 4]), "swapped")
            self.assertEqual(_swap(coordinator, 8, [0]), "swapped")
            self.assertEqual(coordinator.request_finished(first), "finished")
            coordinator.alloc_device_buffer(second)
            _swap(coordinator, 4, [1, 0, 2])
            self.assertEqual(coordinator.destroy(), "destroyed")
            path, = directory.glob("*.jsonl")
            events = [json.loads(line) for line in path.read_text().splitlines()]

        header, long, truncation, short, summary = events
        self.assertFalse(header["performance_evidence"])
        self.assertEqual(long["selected_tokens"], [2, 3, 6, 2, 3])
        self.assertEqual(long["prestate_hit_tokens"], [2, 6])
        self.assertFalse(long["short_path"])
        self.assertEqual(short["selected_tokens"], [2, 3, 0, 1])
        self.assertEqual(short["prestate_hit_tokens"], [0, 1, 2, 3])
        self.assertTrue(short["short_path"])
        self.assertEqual([row["rid"] for row in (long, short)], ["first", "second"])
        self.assertEqual([row["generation"] for row in (long, short)], [1, 2])
        self.assertEqual([row["step"] for row in (long, short)], [0, 0])
        self.assertEqual(truncation, {"type": "truncation", "reason": "max_steps"})
        self.assertEqual(summary, {"type": "summary", "rows": 2, "dropped": {"max_steps": 1}})

    def test_untracked_slot_and_mismatched_owner_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = _fake_coordinator(Path(directory), TraceLimits())
            try:
                with self.assertRaisesRegex(RuntimeError, "cannot resolve unique slot"):
                    _swap(coordinator, 7, [0])
                coordinator.alloc_device_buffer(_request("real"))
                with self.assertRaisesRegex(RuntimeError, "mismatched owner"):
                    coordinator.request_finished(_request("different"))
            finally:
                coordinator.destroy()

    def test_rank_exclusion_runs_native_without_opening_file(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            coordinator = _fake_coordinator(directory, TraceLimits(), rank=2)
            self.assertEqual(_swap(coordinator, 7, [0]), "swapped")
            coordinator.destroy()
            self.assertEqual(list(directory.iterdir()), [])

    def test_row_and_request_limits_report_omitted_rows(self):
        for limits, reason in (
            (TraceLimits(max_rows=1), "max_rows"),
            (TraceLimits(max_requests=1), "max_requests"),
        ):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                coordinator = _fake_coordinator(directory, limits)
                for rid in ("one", "two"):
                    request = _request(rid)
                    coordinator.alloc_device_buffer(request)
                    self.assertEqual(_swap(coordinator, 7, [0]), "swapped")
                    coordinator.request_finished(request)
                coordinator.destroy()
                path, = directory.glob("*.jsonl")
                events = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(
                    [event["type"] for event in events],
                    ["metadata", "access", "truncation", "summary"],
                )
                self.assertEqual(events[-1]["dropped"], {reason: 1})

    def test_runtime_gate(self):
        calls = []
        modules = {"adaptive_sparse.hisparse_access_trace": SimpleNamespace(
            install_hisparse_access_trace=calls.append,
        )}
        with patch.dict(sys.modules, modules), patch.dict(
            os.environ, {"EKV_HISPARSE_TRACE_DIR": "/trace"}, clear=True,
        ):
            self.assertEqual(install_minimax_runtime(), ("hisparse_access_trace",))
        self.assertEqual(calls, ["/trace"])
