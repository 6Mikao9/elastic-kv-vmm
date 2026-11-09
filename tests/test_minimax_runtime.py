import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from adaptive_sparse.minimax_runtime import install_minimax_runtime


class MiniMaxRuntimeTest(unittest.TestCase):
    def test_installs_enabled_components_in_audited_order(self):
        calls = []

        def installer(name):
            def install(*args):
                calls.append((name, *args))

            return install

        modules = {
            "adaptive_sparse.minimax_sm80": SimpleNamespace(
                install=installer("sm80")
            ),
            "adaptive_sparse.minimax_diagnostic_fallbacks": SimpleNamespace(
                install_minimax_diagnostic_fallbacks=installer("diagnostic")
            ),
            "adaptive_sparse.layer_trace": SimpleNamespace(
                install_layer_trace=installer("layer_trace")
            ),
            "adaptive_sparse.minimax_attention_capture": SimpleNamespace(
                install_minimax_attention_capture=installer("capture")
            ),
            "adaptive_sparse.vmm_runtime": SimpleNamespace(
                install_vmm_runtime=installer("vmm")
            ),
            "adaptive_sparse.sglang_future_hook": SimpleNamespace(
                install=installer("future")
            ),
            "adaptive_sparse.request_residency": SimpleNamespace(
                install_request_residency=installer("residency")
            ),
            "adaptive_sparse.scheduler_budget": SimpleNamespace(
                install_scheduler_budget=installer("scheduler")
            ),
            "adaptive_sparse.logical_shadow": SimpleNamespace(
                install_logical_shadow=installer("shadow")
            ),
            "adaptive_sparse.memory_telemetry": SimpleNamespace(
                install_memory_telemetry=installer("telemetry")
            ),
        }
        environment = {
            "EKV_MINIMAX_SM80_BF16": "1",
            "EKV_DISABLE_FUSED_QKNORM": "1",
            "EKV_LAYER_TRACE_DIR": "/trace",
            "EKV_ATTN_CAPTURE_DIR": "/capture",
            "EKV_VMM_STATS_DIR": "/vmm",
            "EKV_FUTURE_POLICY": "deadline",
            "EKV_RESIDENT_LIMIT": "2",
            "EKV_ADAPTIVE_BUDGET_BYTES": "1024",
            "EKV_LOGICAL_SHADOW_DIR": "/shadow",
            "EKV_ALLOCATOR_STATS_DIR": "/allocator",
        }

        with patch.dict(os.environ, environment, clear=True), patch.dict(
            sys.modules, modules
        ):
            installed = install_minimax_runtime()

        self.assertEqual(
            calls,
            [
                ("sm80",),
                ("diagnostic",),
                ("layer_trace", "/trace"),
                ("capture", "/capture"),
                ("vmm", "/vmm"),
                ("future",),
                ("residency",),
                ("scheduler",),
                ("shadow", "/shadow"),
                ("telemetry", "/allocator"),
            ],
        )
        self.assertEqual(
            installed,
            (
                "sm80_bf16_projections",
                "diagnostic_fallbacks",
                "layer_trace",
                "attention_capture",
                "vmm_runtime",
                "future_policy",
                "request_residency",
                "scheduler_budget",
                "logical_shadow",
                "memory_telemetry",
            ),
        )

    def test_disabled_runtime_imports_no_sglang_components(self):
        with patch.dict(os.environ, {}, clear=True), patch.dict(sys.modules, {}):
            self.assertEqual(install_minimax_runtime(), ())

    def test_native_scopes_only_install_in_native_diagnostic(self):
        calls = []
        modules = {
            "adaptive_sparse.profile_ranges": SimpleNamespace(
                install_native_profile_ranges=lambda: calls.append("native")),
            "adaptive_sparse.request_residency": SimpleNamespace(
                install_request_residency=lambda: calls.append("resident")),
        }
        with patch.dict(sys.modules, modules), patch.dict(
                os.environ, {"EKV_PROFILE_RANGES": "1"}, clear=True):
            self.assertEqual(install_minimax_runtime(), ("native_profile_ranges",))
            self.assertEqual(calls, ["native"])
            calls.clear()
            os.environ["EKV_RESIDENT_LIMIT"] = "2"
            self.assertEqual(install_minimax_runtime(), ("request_residency",))
            self.assertEqual(calls, ["resident"])


if __name__ == "__main__":
    unittest.main()
