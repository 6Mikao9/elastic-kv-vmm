from contextlib import nullcontext
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from elastic_kv import activation


class ActivationTest(unittest.TestCase):
    def tearDown(self):
        # install_activation_hooks() changes process-wide SGLang hooks and
        # module state. Prevent a telemetry test from enabling pressure probes
        # in unrelated residency tests that immediately follow it.
        activation._POLICY = "off"
        activation._HEADROOM = 0
        activation._CUDA = None
        activation._PREFILL_WORKSPACE_RATE = 0
        activation._PREFILL_WORKSPACE_CAP = 0
        activation._PREFILL_HEADROOM_SCOPED = False
        activation._CONTEXT = activation.local()

    def test_installs_on_newer_forward_method(self):
        cuda = Mock()
        torch = ModuleType("torch")
        torch.cuda = cuda

        class ScheduleBatch:
            def prepare_for_extend(self):
                return "prepared"

        class ModelRunner:
            def forward(self, *args, **kwargs):
                return "forwarded"

        modules = {
            "torch": torch,
            "sglang": ModuleType("sglang"),
            "sglang.srt": ModuleType("sglang.srt"),
            "sglang.srt.managers": ModuleType("sglang.srt.managers"),
            "sglang.srt.managers.schedule_batch": SimpleNamespace(
                ScheduleBatch=ScheduleBatch
            ),
            "sglang.srt.model_executor": ModuleType(
                "sglang.srt.model_executor"
            ),
            "sglang.srt.model_executor.model_runner": SimpleNamespace(
                ModelRunner=ModelRunner
            ),
        }
        original_forward = ModelRunner.forward
        with patch.dict(sys.modules, modules), patch.dict(
            "os.environ", {"EKV_ACTIVATION_POLICY": "telemetry"}
        ):
            activation.install_activation_hooks()

        self.assertIsNot(ModelRunner.forward, original_forward)
        self.assertIs(ModelRunner.forward.__wrapped__, original_forward)
        self.assertFalse(hasattr(ModelRunner, "forward_extend"))
        self.assertTrue(ModelRunner._ekv_activation_hooks)

    def test_estimates_remaining_prompt_tokens_and_caps_workspace(self):
        requests = [
            SimpleNamespace(origin_input_ids=range(20), num_matched_prefix_tokens=5),
            SimpleNamespace(origin_input_ids=range(10)),
            SimpleNamespace(origin_input_ids=[]),
        ]

        estimate = activation.estimate_prefill_workspace(
            requests, bytes_per_token=8, max_bytes=100
        )

        self.assertEqual(estimate.prompt_tokens, 25)
        self.assertEqual(estimate.bytes, 100)
        self.assertEqual(estimate.requests, 2)
        self.assertEqual(estimate.unestimated, 1)

    def test_scoped_headroom_is_visible_only_inside_prefill_admission(self):
        with patch.multiple(
            activation,
            _PREFILL_HEADROOM_SCOPED=True,
            _CONTEXT=activation.local(),
        ):
            self.assertEqual(activation.current_prefill_headroom(20), 0)
            with activation.prefill_admission_scope(80):
                self.assertEqual(activation.current_prefill_headroom(20), 80)
            self.assertEqual(activation.current_prefill_headroom(20), 0)

    def test_scheduler_hook_scopes_estimated_workspace(self):
        cuda = Mock()
        torch = ModuleType("torch")
        torch.cuda = cuda

        class ScheduleBatch:
            def prepare_for_extend(self):
                return None

        class ModelRunner:
            def forward(self, *args, **kwargs):
                return None

        class Scheduler:
            def __init__(self):
                self.waiting_queue = [
                    SimpleNamespace(origin_input_ids=range(10)),
                    SimpleNamespace(
                        origin_input_ids=range(8), num_matched_prefix_tokens=3
                    ),
                ]
                self.chunked_req = None
                self.controller = SimpleNamespace(ctrl=SimpleNamespace(counters={}))
                self.token_to_kv_pool_allocator = self.controller

            def get_new_batch_prefill(self, running_batch):
                return activation.current_prefill_headroom(5)

        modules = {
            "torch": torch,
            "sglang": ModuleType("sglang"),
            "sglang.srt": ModuleType("sglang.srt"),
            "sglang.srt.managers": ModuleType("sglang.srt.managers"),
            "sglang.srt.managers.schedule_batch": SimpleNamespace(
                ScheduleBatch=ScheduleBatch
            ),
            "sglang.srt.managers.scheduler": SimpleNamespace(Scheduler=Scheduler),
            "sglang.srt.model_executor": ModuleType(
                "sglang.srt.model_executor"
            ),
            "sglang.srt.model_executor.model_runner": SimpleNamespace(
                ModelRunner=ModelRunner
            ),
        }
        with patch.dict(sys.modules, modules), patch.dict(
            "os.environ",
            {
                "EKV_ACTIVATION_POLICY": "off",
                "EKV_PREFILL_HEADROOM_SCOPED": "1",
                "EKV_PREFILL_WORKSPACE_BYTES_PER_TOKEN": "2",
                "EKV_PREFILL_WORKSPACE_MAX_MIB": "1",
            },
        ):
            activation.install_activation_hooks()
            scheduler = Scheduler()

        self.assertEqual(scheduler.get_new_batch_prefill(None), 30)
        self.assertEqual(
            scheduler.controller.ctrl.counters[
                "activation_dynamic_prefill_prompt_tokens"
            ],
            15,
        )
        self.assertEqual(
            activation.current_prefill_headroom(5),
            0,
        )
        self.assertFalse(getattr(ModelRunner, "_ekv_activation_hooks", False))

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
