import sys
import unittest
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from elastic_kv import activation


def _sglang_modules(*, ScheduleBatch, ModelRunner, Scheduler=None):
    modules = {
        "torch": ModuleType("torch"),
        "sglang": ModuleType("sglang"),
        "sglang.srt": ModuleType("sglang.srt"),
        "sglang.srt.managers": ModuleType("sglang.srt.managers"),
        "sglang.srt.managers.schedule_batch": SimpleNamespace(
            ScheduleBatch=ScheduleBatch
        ),
        "sglang.srt.model_executor": ModuleType("sglang.srt.model_executor"),
        "sglang.srt.model_executor.model_runner": SimpleNamespace(
            ModelRunner=ModelRunner
        ),
    }
    modules["torch"].cuda = Mock()
    if Scheduler is not None:
        modules["sglang.srt.managers.scheduler"] = SimpleNamespace(
            Scheduler=Scheduler
        )
    return modules


class ActivationTest(unittest.TestCase):
    def test_records_conservative_activation_workspace_rates(self):
        counters = {}
        activation._record_activation_workspace_sample(
            counters, growth_bytes=10, batch_tokens=4, batch_size=2
        )
        activation._record_activation_workspace_sample(
            counters, growth_bytes=7, batch_tokens=3, batch_size=1
        )
        activation._record_activation_workspace_sample(
            counters, growth_bytes=1, batch_tokens=0, batch_size=0
        )

        self.assertEqual(
            counters["activation_peak_tokens_4_batch_2_bytes"], 10
        )
        self.assertEqual(
            counters["activation_peak_tokens_3_batch_1_bytes"], 7
        )
        self.assertEqual(
            counters["activation_prefill_bytes_per_token_max_observed"], 3
        )
        self.assertEqual(
            counters["activation_prefill_bytes_per_token_samples"], 2
        )

    def test_skips_nonpositive_activation_workspace_samples(self):
        counters = {}
        for growth_bytes, batch_tokens, batch_size in (
            (0, 4, 2),
            (8, 0, 2),
            (8, 4, 0),
        ):
            activation._record_activation_workspace_sample(
                counters,
                growth_bytes=growth_bytes,
                batch_tokens=batch_tokens,
                batch_size=batch_size,
            )

        self.assertEqual(
            counters["activation_prefill_bytes_per_token_samples_skipped"], 3
        )
        self.assertNotIn(
            "activation_prefill_bytes_per_token_samples", counters
        )

    def test_activation_workspace_observation_is_best_effort(self):
        class BrokenBatch:
            batch_size = 2

            @property
            def input_ids(self):
                raise RuntimeError("telemetry unavailable")

        counters = {
            "activation_prefill_peak_growth_bytes_last": 8,
        }
        activation._observe_activation_workspace_sample(
            counters, batch=BrokenBatch()
        )

        self.assertEqual(
            counters["activation_prefill_bytes_per_token_sample_errors"], 1
        )
        self.assertNotIn(
            "activation_prefill_bytes_per_token_samples", counters
        )

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
        activation._GLOBAL_PREFILL_MODE = "off"
        activation._PENDING_PREFILL_CLAIM = None
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
            SimpleNamespace(
                origin_input_ids=range(20),
                prefix_indices=range(13),
                kv=SimpleNamespace(kv_committed_len=12),
                num_matched_prefix_tokens=5,
            ),
            SimpleNamespace(origin_input_ids=[]),
        ]

        estimate = activation.estimate_prefill_workspace(
            requests, bytes_per_token=8, max_bytes=100
        )

        self.assertEqual(estimate.prompt_tokens, 32)
        self.assertEqual(estimate.bytes, 100)
        self.assertEqual(estimate.requests, 3)
        self.assertEqual(estimate.unestimated, 1)

    def test_empty_batch_detection_supports_newer_sglang_method(self):
        class Batches:
            def __init__(self, size):
                self.size = size

            def batch_size(self):
                return self.size

        self.assertTrue(activation._batch_is_empty(Batches(0)))
        self.assertFalse(activation._batch_is_empty(Batches(1)))

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

    def test_global_prefill_claim_wraps_scheduler_and_forward(self):
        cuda = Mock()
        torch = ModuleType("torch")
        torch.cuda = cuda

        class Backing:
            has_global_broker = True

            def __init__(self):
                self.claims = []
                self.shrinks = []
                self.releases = []

            def claim_broker_capacity(self, key, byte_count):
                self.claims.append((key, byte_count))
                return True

            def release_broker_claim(self, key):
                self.releases.append(key)
                return 8

            def shrink_broker_claim(self, key, maximum_bytes):
                self.shrinks.append((key, maximum_bytes))
                return min(8, maximum_bytes)

        class ScheduleBatch:
            def prepare_for_extend(self):
                return None

        class ModelRunner:
            def forward(self, *args, **kwargs):
                return "forwarded"

        backing = Backing()
        counters = {}
        ctrl = SimpleNamespace(
            arenas=[SimpleNamespace(backing=backing)], counters=counters
        )

        class Scheduler:
            def __init__(self):
                self.waiting_queue = [SimpleNamespace(origin_input_ids=range(4))]
                self.chunked_req = None
                self.token_to_kv_pool_allocator = SimpleNamespace(ctrl=ctrl)

            def get_new_batch_prefill(self, running_batch):
                self.claim_seen_by_scheduler = bool(backing.claims)
                return SimpleNamespace(
                    batch_size=1,
                    input_ids=SimpleNamespace(numel=lambda: 3),
                )

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
                "EKV_PREFILL_WORKSPACE_BYTES_PER_TOKEN": "2",
                "EKV_PREFILL_WORKSPACE_MAX_MIB": "1",
                "EKV_GLOBAL_BROKER_PREFILL_MODE": "enforce",
            },
        ):
            activation.install_activation_hooks()
            scheduler = Scheduler()
            batch = scheduler.get_new_batch_prefill(None)
            model_runner = ModelRunner()
            model_runner.token_to_kv_pool = SimpleNamespace(residency=ctrl)
            result = model_runner.forward(batch)

        self.assertEqual(result, "forwarded")
        self.assertTrue(scheduler.claim_seen_by_scheduler)
        self.assertEqual(backing.claims[0][1], 8)
        self.assertEqual(backing.shrinks, [(backing.claims[0][0], 6)])
        self.assertEqual(backing.releases, [backing.claims[0][0]])
        self.assertEqual(counters["activation_global_prefill_claim_admissions"], 1)
        self.assertEqual(
            counters["activation_global_prefill_claim_released_forward"], 1
        )
        self.assertEqual(counters["activation_global_prefill_claim_shrunk_bytes"], 2)

    def test_global_prefill_claim_releases_when_scheduler_returns_empty(self):
        cuda = Mock()
        torch = ModuleType("torch")
        torch.cuda = cuda

        class Backing:
            has_global_broker = True

            def __init__(self):
                self.releases = []

            def claim_broker_capacity(self, key, byte_count):
                return True

            def release_broker_claim(self, key):
                self.releases.append(key)
                return 4

        class ScheduleBatch:
            def prepare_for_extend(self):
                return None

        class ModelRunner:
            def forward(self, *args, **kwargs):
                return None

        backing = Backing()
        counters = {}
        ctrl = SimpleNamespace(
            arenas=[SimpleNamespace(backing=backing)], counters=counters
        )

        class Scheduler:
            waiting_queue = [SimpleNamespace(origin_input_ids=range(2))]
            chunked_req = None
            token_to_kv_pool_allocator = SimpleNamespace(ctrl=ctrl)

            def get_new_batch_prefill(self, running_batch):
                return None

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
                "EKV_PREFILL_WORKSPACE_BYTES_PER_TOKEN": "2",
                "EKV_PREFILL_WORKSPACE_MAX_MIB": "1",
                "EKV_GLOBAL_BROKER_PREFILL_MODE": "enforce",
            },
        ):
            activation.install_activation_hooks()
            self.assertIsNone(Scheduler().get_new_batch_prefill(None))

        self.assertEqual(len(backing.releases), 1)
        self.assertEqual(
            counters["activation_global_prefill_claim_released_empty_batch"], 1
        )

    def test_prefill_batch_workspace_uses_selected_tokens_and_cap(self):
        batch = SimpleNamespace(
            batch_size=2,
            input_ids=SimpleNamespace(numel=lambda: 5),
        )
        estimate = activation.estimate_prefill_batch_workspace(
            batch, bytes_per_token=3, max_bytes=12
        )
        self.assertEqual(estimate.bytes, 12)
        self.assertEqual(estimate.prompt_tokens, 5)
        self.assertIsNone(
            activation.estimate_prefill_batch_workspace(
                SimpleNamespace(batch_size=1), bytes_per_token=3, max_bytes=12
            )
        )

    def test_prefill_batch_workspace_uses_newer_staged_cpu_inputs(self):
        batch = SimpleNamespace(
            batch_size=2,
            input_ids=None,
            prefill_input_ids_cpu=SimpleNamespace(numel=lambda: 4),
        )
        estimate = activation.estimate_prefill_batch_workspace(
            batch, bytes_per_token=3, max_bytes=20
        )
        self.assertEqual(estimate.bytes, 12)
        self.assertEqual(estimate.prompt_tokens, 4)

    def test_global_prefill_claim_survives_forward_error_until_teardown(self):
        cuda = Mock()
        torch = ModuleType("torch")
        torch.cuda = cuda

        class Backing:
            has_global_broker = True

            def __init__(self):
                self.releases = []

            def claim_broker_capacity(self, key, byte_count):
                return True

            def release_broker_claim(self, key):
                self.releases.append(key)
                return 2

        class ScheduleBatch:
            def prepare_for_extend(self):
                return None

        class ModelRunner:
            def forward(self, *args, **kwargs):
                raise RuntimeError("prefill failed")

        backing = Backing()
        counters = {}
        ctrl = SimpleNamespace(
            arenas=[SimpleNamespace(backing=backing)], counters=counters
        )

        class Scheduler:
            waiting_queue = [SimpleNamespace(origin_input_ids=range(1))]
            chunked_req = None
            token_to_kv_pool_allocator = SimpleNamespace(ctrl=ctrl)

            def get_new_batch_prefill(self, running_batch):
                return SimpleNamespace(batch_size=1)

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
                "EKV_PREFILL_WORKSPACE_BYTES_PER_TOKEN": "2",
                "EKV_PREFILL_WORKSPACE_MAX_MIB": "1",
                "EKV_GLOBAL_BROKER_PREFILL_MODE": "enforce",
            },
        ):
            activation.install_activation_hooks()
            batch = Scheduler().get_new_batch_prefill(None)
            model_runner = ModelRunner()
            model_runner.token_to_kv_pool = SimpleNamespace(residency=ctrl)
            with self.assertRaisesRegex(RuntimeError, "prefill failed"):
                model_runner.forward(batch)
            self.assertIsNone(activation._PENDING_PREFILL_CLAIM)
            with self.assertRaisesRegex(RuntimeError, "prefill failed"):
                model_runner.forward(None)

        self.assertEqual(backing.releases, [])
        self.assertEqual(
            counters["activation_global_prefill_claim_abandoned_forward_error"], 1
        )

    def test_global_prefill_claim_survives_scheduler_error(self):
        class Backing:
            has_global_broker = True

            def __init__(self):
                self.releases = []

            def claim_broker_capacity(self, key, byte_count):
                return True

            def release_broker_claim(self, key):
                self.releases.append(key)
                return 2

        class ScheduleBatch:
            def prepare_for_extend(self):
                return None

        class ModelRunner:
            def forward(self, *args, **kwargs):
                return None

        backing = Backing()
        counters = {}
        ctrl = SimpleNamespace(
            arenas=[SimpleNamespace(backing=backing)], counters=counters
        )

        class Scheduler:
            waiting_queue = [SimpleNamespace(origin_input_ids=range(1))]
            chunked_req = None
            token_to_kv_pool_allocator = SimpleNamespace(ctrl=ctrl)

            def get_new_batch_prefill(self, running_batch):
                raise RuntimeError("scheduler failed")

        with patch.dict(
            sys.modules,
            _sglang_modules(
                ScheduleBatch=ScheduleBatch,
                ModelRunner=ModelRunner,
                Scheduler=Scheduler,
            ),
        ), patch.dict(
            "os.environ",
            {
                "EKV_PREFILL_WORKSPACE_BYTES_PER_TOKEN": "2",
                "EKV_PREFILL_WORKSPACE_MAX_MIB": "1",
                "EKV_GLOBAL_BROKER_PREFILL_MODE": "enforce",
            },
        ):
            activation.install_activation_hooks()
            with self.assertRaisesRegex(RuntimeError, "scheduler failed"):
                Scheduler().get_new_batch_prefill(None)
            self.assertIsNone(activation._PENDING_PREFILL_CLAIM)

        self.assertEqual(backing.releases, [])
        self.assertEqual(
            counters["activation_global_prefill_claim_abandoned_scheduler_error"],
            1,
        )

    def test_observe_mode_checks_capacity_without_inserting_a_claim(self):
        class Backing:
            has_global_broker = True
            broker_snapshot = {
                "available_bytes": 12,
                "fair_mode": "enforce",
                "fairness_pressure": False,
            }

            def claim_broker_capacity(self, key, byte_count):
                raise AssertionError("observe mode must not mutate the broker")

        backing = Backing()
        counters = {}
        ctrl = SimpleNamespace(
            arenas=[SimpleNamespace(backing=backing)], counters=counters
        )
        estimate = activation.PrefillWorkspaceEstimate(bytes=8)

        with patch.multiple(
            activation,
            _GLOBAL_PREFILL_MODE="observe",
            _PENDING_PREFILL_CLAIM=None,
            _CONTEXT=activation.local(),
        ):
            claim, disposition = activation._reserve_global_prefill_claim(
                ctrl, estimate, counters
            )

        self.assertIsNone(claim)
        self.assertEqual(disposition, "observed")
        self.assertEqual(
            counters["activation_global_prefill_observe_would_admit"], 1
        )

    def test_enforce_mode_fails_closed_on_denial_or_missing_broker(self):
        class DenyingBacking:
            has_global_broker = True

            def claim_broker_capacity(self, key, byte_count):
                return False

        counters = {}
        ctrl = SimpleNamespace(
            arenas=[SimpleNamespace(backing=DenyingBacking())], counters=counters
        )
        estimate = activation.PrefillWorkspaceEstimate(bytes=8)
        with patch.multiple(
            activation,
            _GLOBAL_PREFILL_MODE="enforce",
            _PENDING_PREFILL_CLAIM=None,
        ):
            claim, disposition = activation._reserve_global_prefill_claim(
                ctrl, estimate, counters
            )
            self.assertIsNone(claim)
            self.assertEqual(disposition, "denied")
            self.assertIsNone(activation._PENDING_PREFILL_CLAIM)

            missing_claim, missing_disposition = (
                activation._reserve_global_prefill_claim(
                    SimpleNamespace(arenas=[], counters={}),
                    estimate,
                    {},
                )
            )
            self.assertIsNone(missing_claim)
            self.assertEqual(missing_disposition, "missing-broker")

        self.assertEqual(counters["activation_global_prefill_claim_denials"], 1)

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
