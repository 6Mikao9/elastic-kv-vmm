import hashlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from adaptive_sparse import scheduler_budget


def _restore_class_attributes(cls, attributes):
    for name, value in attributes.items():
        if value is _MISSING:
            delattr(cls, name)
        else:
            setattr(cls, name, value)


_MISSING = object()


class RealSGLangSchedulerBudgetTest(unittest.TestCase):
    def test_prewarm_exception_falls_back_through_installed_scheduler_wrapper(self):
        try:
            import sglang
            from sglang.srt.managers import scheduler as scheduler_module
            from sglang.srt import runtime_context
            from sglang.srt.model_executor import cuda_graph_config
        except ImportError as exc:
            self.skipTest(f"SGLang scheduler runtime is unavailable: {exc}")

        if sglang.__version__ != "0.5.17":
            self.skipTest(f"unsupported real SGLang version: {sglang.__version__}")

        scheduler_class = scheduler_module.Scheduler
        names = (
            "get_new_batch_prefill",
            "update_running_batch",
            "on_idle",
            "_ekv_adaptive_budget_installed",
        )
        originals = {name: getattr(scheduler_class, name, _MISSING) for name in names}
        self.addCleanup(_restore_class_attributes, scheduler_class, originals)

        native_calls = []

        def native_prefill(self, running_batch):
            native_calls.append("prefill")
            return "native-prefill-result"

        def native_decode(self, batch):
            native_calls.append("decode")
            return "native-decode-result"

        def native_idle(self):
            native_calls.append("idle")
            return "native-idle-result"

        scheduler_class.get_new_batch_prefill = native_prefill
        scheduler_class.update_running_batch = native_decode
        scheduler_class.on_idle = native_idle
        scheduler_class._ekv_adaptive_budget_installed = False

        controller = SimpleNamespace(
            steps=0,
            stats={"predictive_prewarm_hook_errors": 0},
            observe_future_prefill=lambda *args, **kwargs: None,
            boundary=lambda *args, **kwargs: None,
        )
        invalidations = []

        class Tracker:
            def expire_predictive_prewarm(self):
                return 0

            def predictive_prewarm(self, demand_bytes):
                self.demand_bytes = demand_bytes
                raise RuntimeError("injected VMM prewarm failure")

            def invalidate_predictive_prewarm(self, reason):
                invalidations.append(reason)

        tracker = Tracker()
        pool = SimpleNamespace(
            adaptive_vmm=tracker,
            layer_num=1,
            bytes_per_token_k=1,
            bytes_per_token_v=1,
            device=object(),
        )
        coordinator = SimpleNamespace(
            is_m3_hisparse=True,
            mem_pool_device=pool,
            tp_world_size=1,
            device_buffer_size=128,
        )
        scheduler = scheduler_class.__new__(scheduler_class)
        scheduler.hisparse_coordinator = coordinator
        scheduler.waiting_queue = [object()]
        scheduler.chunked_req = None
        scheduler._ekv_adaptive_budget = controller

        source_sha256 = hashlib.sha256(Path(scheduler_module.__file__).read_bytes()).hexdigest()
        reservation = SimpleNamespace(signature=("future",), kv_bytes=4096)
        env = {
            "EKV_ADAPTIVE_BUDGET_BYTES": "8192",
            "EKV_FUTURE_PREFILL_MODE": "enforce",
            "EKV_FUTURE_PREFILL_MAX_BYTES": "4096",
        }
        with patch.dict("os.environ", env, clear=False), patch.object(
            scheduler_budget, "SCHEDULER_SHA256S", frozenset({source_sha256})
        ), patch.object(
            scheduler_budget, "_should_plan_future_waiting", return_value=True
        ), patch.object(
            scheduler_budget, "plan_future_prefill", return_value=reservation
        ), patch.object(
            scheduler_budget, "effective_prefill_headroom", return_value=0
        ), patch.object(
            cuda_graph_config, "cuda_graph_fully_disabled", return_value=True
        ), patch.object(
            runtime_context,
            "get_schedule",
            return_value=SimpleNamespace(disable_overlap_schedule=True),
        ), patch.object(
            runtime_context,
            "get_server_args",
            return_value=SimpleNamespace(disable_radix_cache=True),
        ), patch.object(
            runtime_context,
            "get_disagg",
            return_value=SimpleNamespace(disaggregation_mode="null"),
        ), patch.object(
            runtime_context,
            "get_spec",
            return_value=SimpleNamespace(speculative_algorithm=None),
        ):
            scheduler_budget.install_scheduler_budget()
            self.assertIsNot(
                scheduler_class.get_new_batch_prefill,
                native_prefill,
                "test did not exercise the installed wrapper",
            )
            result = scheduler.get_new_batch_prefill("running-batch")

        self.assertEqual(result, "native-prefill-result")
        self.assertEqual(native_calls, ["prefill"])
        self.assertEqual(tracker.demand_bytes, 4096)
        self.assertEqual(controller.stats["predictive_prewarm_hook_errors"], 1)
        self.assertEqual(len(invalidations), 1)
        self.assertIn("scheduler-prewarm:RuntimeError", invalidations[0])
        self.assertIn("injected VMM prewarm failure", invalidations[0])


if __name__ == "__main__":
    unittest.main()
