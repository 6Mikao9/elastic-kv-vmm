import sys
import unittest
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from elastic_kv import activation, adapter


def _module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


class AdapterCompatibilityTest(unittest.TestCase):
    def setUp(self):
        adapter.MODE = "v5"
        adapter.ALLOW_OVERLAP_SCHEDULE = False
        adapter.OVERLAP_SCHEDULE_ACTIVE = False
        adapter.REQUEST_AFFINE_MODE = "off"
        adapter._installed = False
        adapter.CONTROLLERS.clear()

    def tearDown(self):
        adapter._installed = False
        adapter.MODE = "baseline"
        adapter.ALLOW_OVERLAP_SCHEDULE = False
        adapter.OVERLAP_SCHEDULE_ACTIVE = False
        adapter.REQUEST_AFFINE_MODE = "off"
        adapter.CONTROLLERS.clear()

    def test_installs_on_newer_mem_cache_api(self):
        class LegacyMHA:
            pass

        class LegacyAllocator:
            def __init__(self):
                self._kvcache = SimpleNamespace()
                self.size = 0
                self.need_sort = False
                self.free_group = None

        calls = []

        class ModelRunner:
            pass

        class Configurator:
            def configure(self, *args, **kwargs):
                calls.append((args, kwargs))
                modules = sys.modules
                pool_type = modules["sglang.srt.mem_cache.memory_pool"].MHATokenToKVPool
                allocator_type = modules[
                    "sglang.srt.mem_cache.allocator.token"
                ].TokenToKVPoolAllocator
                return SimpleNamespace(
                    marker="configured",
                    token_to_kv_pool=object.__new__(pool_type),
                    token_to_kv_pool_allocator=object.__new__(allocator_type),
                )

        modules = {
            "sglang": _module("sglang", __version__="dev"),
            "sglang.srt": _module("sglang.srt"),
            "sglang.srt.model_executor": _module("sglang.srt.model_executor"),
            "sglang.srt.model_executor.model_runner": _module(
                "sglang.srt.model_executor.model_runner", ModelRunner=ModelRunner
            ),
            "sglang.srt.mem_cache": _module("sglang.srt.mem_cache"),
            "sglang.srt.mem_cache.memory_pool": _module(
                "sglang.srt.mem_cache.memory_pool", MHATokenToKVPool=LegacyMHA
            ),
            "sglang.srt.mem_cache.kv_cache_configurator": _module(
                "sglang.srt.mem_cache.kv_cache_configurator",
                KVCacheConfigurator=Configurator,
                MHATokenToKVPool=LegacyMHA,
                TokenToKVPoolAllocator=LegacyAllocator,
                _KV_POOL_BASE_FOR_KIND={"mha": LegacyMHA},
            ),
            "sglang.srt.mem_cache.allocator": _module(
                "sglang.srt.mem_cache.allocator",
                TokenToKVPoolAllocator=LegacyAllocator,
            ),
            "sglang.srt.mem_cache.allocator.token": _module(
                "sglang.srt.mem_cache.allocator.token",
                TokenToKVPoolAllocator=LegacyAllocator,
            ),
        }
        with patch.dict(sys.modules, modules):
            original_configure = Configurator.configure
            adapter.install()

            memory_pool = sys.modules["sglang.srt.mem_cache.memory_pool"]
            configurator = sys.modules[
                "sglang.srt.mem_cache.kv_cache_configurator"
            ]
            allocator_module = sys.modules[
                "sglang.srt.mem_cache.allocator.token"
            ]

            self.assertNotEqual(memory_pool.MHATokenToKVPool, LegacyMHA)
            self.assertEqual(
                configurator._KV_POOL_BASE_FOR_KIND["mha"],
                memory_pool.MHATokenToKVPool,
            )
            self.assertNotEqual(
                allocator_module.TokenToKVPoolAllocator, LegacyAllocator
            )
            self.assertIsNot(Configurator.configure, original_configure)

            # The VMM planes are NHD. Pin that metadata before any allocation
            # so an HND environment default cannot make kernels interpret them
            # as a different physical layout.
            class StopAtProjection(Exception):
                pass

            pool_type = memory_pool.MHATokenToKVPool
            pool_probe = SimpleNamespace(
                use_hnd=True,
                kv_cache_layout="hnd",
                size=32,
                page_size=1,
                head_num=2,
                head_dim=4,
                layer_num=1,
                store_dtype=SimpleNamespace(itemsize=2),
                device="cpu",
            )
            with patch.object(
                adapter, "PlaneProjection", side_effect=StopAtProjection
            ), self.assertRaises(StopAtProjection):
                pool_type._create_buffers(pool_probe)
            self.assertFalse(pool_probe.use_hnd)
            self.assertEqual(pool_probe.kv_cache_layout, "nhd")

            # install() replaces the class method. Resolve it from the class to
            # exercise the new configure guard with a representative runner.
            guarded = Configurator.configure
            server_args = SimpleNamespace(
                attention_backend="triton",
                prefill_attention_backend=None,
                decode_attention_backend=None,
                page_size=1,
                disable_cuda_graph=False,
                disable_overlap_schedule=True,
                disable_radix_cache=True,
                tp_size=1,
                pp_size=1,
                dp_size=1,
                enable_two_batch_overlap=False,
                disaggregation_mode="null",
                enable_memory_saver=False,
                speculative_algorithm=None,
                enable_hierarchical_cache=False,
            )
            configurator_instance = SimpleNamespace(
                server_args=server_args,
                is_hybrid_swa=False,
                mambaish_config=None,
                hybrid_gdn_config=None,
                hybrid_kda_config=None,
                use_mla_backend=False,
                kv_cache_dtype=adapter.torch.bfloat16,
                is_draft_worker=False,
                kv_shard_rank=None,
                post_capture_kv_active=False,
                model_config=SimpleNamespace(head_dim=128, v_head_dim=128),
            )
            self.assertEqual(
                guarded(configurator_instance, pre_model_load_memory=1).marker,
                "configured",
            )
            self.assertEqual(calls, [((), {"pre_model_load_memory": 1})])

            configurator_instance.model_config.hf_config = SimpleNamespace(
                model_type="minimax_m3", architectures=["MiniMaxM3SparseForCausalLM"]
            )
            with self.assertRaisesRegex(RuntimeError, "MiniMax sparse"):
                guarded(configurator_instance, pre_model_load_memory=1)
            configurator_instance.model_config.hf_config = None

            server_args.enable_hisparse = True
            with self.assertRaisesRegex(RuntimeError, "HiSparse"):
                guarded(configurator_instance, pre_model_load_memory=1)
            server_args.enable_hisparse = False

            configurator_instance.is_hybrid_swa_compress = True
            with self.assertRaisesRegex(RuntimeError, "Require dense MHA"):
                guarded(configurator_instance, pre_model_load_memory=1)
            configurator_instance.is_hybrid_swa_compress = False

            elastic_pool = memory_pool.MHATokenToKVPool
            memory_pool.MHATokenToKVPool = LegacyMHA
            with self.assertRaisesRegex(RuntimeError, "non-elastic token KV pool"):
                guarded(configurator_instance, pre_model_load_memory=1)
            memory_pool.MHATokenToKVPool = elastic_pool

            elastic_allocator = allocator_module.TokenToKVPoolAllocator
            allocator_module.TokenToKVPoolAllocator = LegacyAllocator
            with self.assertRaisesRegex(RuntimeError, "not observed"):
                guarded(configurator_instance, pre_model_load_memory=1)
            allocator_module.TokenToKVPoolAllocator = elastic_allocator

            server_args.attention_backend = "flashinfer"
            with self.assertRaisesRegex(RuntimeError, "Require dense MHA"):
                guarded(configurator_instance, pre_model_load_memory=1)
            server_args.attention_backend = "triton"

            server_args.enable_page_major_kv_layout = True
            with self.assertRaisesRegex(RuntimeError, "Require dense MHA"):
                guarded(configurator_instance, pre_model_load_memory=1)
            server_args.enable_page_major_kv_layout = False

            configurator_instance.model_config.v_head_dim = 64
            with self.assertRaisesRegex(RuntimeError, "symmetric K/V"):
                guarded(configurator_instance, pre_model_load_memory=1)
            configurator_instance.model_config.v_head_dim = 128

            configurator_instance.kv_cache_dtype = adapter.torch.float16
            with self.assertRaisesRegex(RuntimeError, "requires BF16"):
                guarded(configurator_instance, pre_model_load_memory=1)
            configurator_instance.kv_cache_dtype = adapter.torch.bfloat16

            configurator_instance.is_draft_worker = True
            with self.assertRaisesRegex(RuntimeError, "draft workers"):
                guarded(configurator_instance, pre_model_load_memory=1)
            configurator_instance.is_draft_worker = False

            configurator_instance.kv_shard_rank = 0
            with self.assertRaisesRegex(RuntimeError, "KV sharding"):
                guarded(configurator_instance, pre_model_load_memory=1)
            configurator_instance.kv_shard_rank = None

            configurator_instance.post_capture_kv_active = True
            with self.assertRaisesRegex(RuntimeError, "post-capture"):
                guarded(configurator_instance, pre_model_load_memory=1)
            configurator_instance.post_capture_kv_active = False

            # Baseline remains a stock-primary control and must not be rejected
            # merely because the model uses a specialized upstream KV pool.
            configurator_instance.model_config.hf_config = SimpleNamespace(
                model_type="minimax_m3"
            )
            adapter.MODE = "baseline"
            self.assertEqual(
                guarded(configurator_instance, pre_model_load_memory=1).marker,
                "configured",
            )

    def test_installs_on_0_5_17_without_kind_registry(self):
        class LegacyMHA:
            pass

        class LegacyAllocator:
            pass

        class ModelRunner:
            pass

        class Configurator:
            def configure(self, *args, **kwargs):
                return "configured"

        modules = {
            "sglang": _module("sglang", __version__="0.5.17"),
            "sglang.srt": _module("sglang.srt"),
            "sglang.srt.model_executor": _module("sglang.srt.model_executor"),
            "sglang.srt.model_executor.model_runner": _module(
                "sglang.srt.model_executor.model_runner", ModelRunner=ModelRunner
            ),
            "sglang.srt.mem_cache": _module("sglang.srt.mem_cache"),
            "sglang.srt.mem_cache.memory_pool": _module(
                "sglang.srt.mem_cache.memory_pool", MHATokenToKVPool=LegacyMHA
            ),
            "sglang.srt.mem_cache.kv_cache_configurator": _module(
                "sglang.srt.mem_cache.kv_cache_configurator",
                KVCacheConfigurator=Configurator,
                MHATokenToKVPool=LegacyMHA,
                TokenToKVPoolAllocator=LegacyAllocator,
            ),
            "sglang.srt.mem_cache.allocator": _module(
                "sglang.srt.mem_cache.allocator",
                TokenToKVPoolAllocator=LegacyAllocator,
            ),
            "sglang.srt.mem_cache.allocator.token": _module(
                "sglang.srt.mem_cache.allocator.token",
                TokenToKVPoolAllocator=LegacyAllocator,
            ),
        }
        with patch.dict(sys.modules, modules):
            adapter.install()
            memory_pool = sys.modules["sglang.srt.mem_cache.memory_pool"]
            configurator = sys.modules[
                "sglang.srt.mem_cache.kv_cache_configurator"
            ]
            self.assertNotEqual(memory_pool.MHATokenToKVPool, LegacyMHA)
            self.assertNotEqual(configurator.MHATokenToKVPool, LegacyMHA)
            self.assertNotEqual(
                configurator.TokenToKVPoolAllocator, LegacyAllocator
            )

    def test_request_affine_wraps_extend_and_decode_with_request_ids(self):
        adapter.REQUEST_AFFINE_MODE = "dry_run"
        observed = []

        def original_extend(batch):
            observed.append(("extend", batch.allocator.demands))
            return "extend-result"

        def original_decode(batch, token_per_req):
            observed.append(("decode", batch.allocator.demands, token_per_req))
            return "decode-result"

        class LegacyMHA:
            pass

        class LegacyAllocator:
            pass

        class ModelRunner:
            pass

        class Configurator:
            def configure(self, *args, **kwargs):
                return "configured"

        modules = {
            "sglang": _module("sglang", __version__="0.5.17"),
            "sglang.srt": _module("sglang.srt"),
            "sglang.srt.model_executor": _module("sglang.srt.model_executor"),
            "sglang.srt.model_executor.model_runner": _module(
                "sglang.srt.model_executor.model_runner", ModelRunner=ModelRunner
            ),
            "sglang.srt.mem_cache": _module("sglang.srt.mem_cache"),
            "sglang.srt.mem_cache.memory_pool": _module(
                "sglang.srt.mem_cache.memory_pool", MHATokenToKVPool=LegacyMHA
            ),
            "sglang.srt.mem_cache.kv_cache_configurator": _module(
                "sglang.srt.mem_cache.kv_cache_configurator",
                KVCacheConfigurator=Configurator,
                MHATokenToKVPool=LegacyMHA,
                TokenToKVPoolAllocator=LegacyAllocator,
            ),
            "sglang.srt.mem_cache.allocator": _module(
                "sglang.srt.mem_cache.allocator",
                TokenToKVPoolAllocator=LegacyAllocator,
            ),
            "sglang.srt.mem_cache.allocator.token": _module(
                "sglang.srt.mem_cache.allocator.token",
                TokenToKVPoolAllocator=LegacyAllocator,
            ),
            "sglang.srt.managers": _module("sglang.srt.managers"),
            "sglang.srt.managers.schedule_batch": _module(
                "sglang.srt.managers.schedule_batch",
                alloc_for_extend=original_extend,
                alloc_for_decode=original_decode,
            ),
        }
        with patch.dict(sys.modules, modules):
            adapter.install()
            schedule_batch = sys.modules[
                "sglang.srt.managers.schedule_batch"
            ]

            class Allocator:
                def __init__(self):
                    self.demands = None

                def set_request_affine_demands(self, demands):
                    self.demands = tuple(demands)
                    return True

                def clear_request_affine_demands(self):
                    self.demands = None

            allocator = Allocator()
            req = SimpleNamespace(
                rid="request-a",
                sampling_params=SimpleNamespace(max_new_tokens=90),
            )
            batch = SimpleNamespace(
                reqs=[req],
                seq_lens_cpu=[20],
                extend_lens=[10],
                tree_cache=SimpleNamespace(
                    token_to_kv_pool_allocator=allocator
                ),
                allocator=allocator,
            )
            self.assertEqual(schedule_batch.alloc_for_extend(batch), "extend-result")
            self.assertEqual(schedule_batch.alloc_for_decode(batch, 1), "decode-result")
            self.assertEqual([item[0] for item in observed], ["extend", "decode"])
            self.assertEqual(observed[0][1][0].request_id, "request-a")
            self.assertEqual(observed[0][1][0].slots, 10)
            self.assertEqual(observed[1][2], 1)
            self.assertIsNone(allocator.demands)

    def test_request_affine_wraps_sglang_052_alloc_token_slots(self):
        adapter.REQUEST_AFFINE_MODE = "dry_run"
        observed = []

        class LegacyMHA:
            pass

        class LegacyAllocator:
            pass

        class ModelRunner:
            def init_memory_pool(self, *args, **kwargs):
                return None

        class ScheduleBatch:
            def alloc_token_slots(self, num_tokens, backup_state=False):
                observed.append(
                    (
                        self.token_to_kv_pool_allocator.demands,
                        num_tokens,
                        backup_state,
                    )
                )
                return "tokens-result"

        schedule_batch = _module(
            "sglang.srt.managers.schedule_batch",
            ScheduleBatch=ScheduleBatch,
        )
        modules = {
            "sglang": _module("sglang", __version__="0.5.2"),
            "sglang.srt": _module("sglang.srt"),
            "sglang.srt.model_executor": _module("sglang.srt.model_executor"),
            "sglang.srt.model_executor.model_runner": _module(
                "sglang.srt.model_executor.model_runner",
                ModelRunner=ModelRunner,
                MHATokenToKVPool=LegacyMHA,
                TokenToKVPoolAllocator=LegacyAllocator,
            ),
            "sglang.srt.managers": _module("sglang.srt.managers"),
            "sglang.srt.managers.schedule_batch": schedule_batch,
        }
        with patch.dict(sys.modules, modules):
            adapter.install()

            class Allocator:
                def __init__(self):
                    self.demands = None

                def set_request_affine_demands(self, demands):
                    self.demands = tuple(demands)
                    return True

                def clear_request_affine_demands(self):
                    self.demands = None

            allocator = Allocator()
            req = SimpleNamespace(
                rid="legacy-request",
                sampling_params=SimpleNamespace(max_new_tokens=90),
                fill_ids=list(range(20)),
                extend_input_len=10,
                output_ids=(),
            )
            batch = object.__new__(ScheduleBatch)
            batch.reqs = [req]
            batch.token_to_kv_pool_allocator = allocator
            batch.forward_mode = SimpleNamespace(is_decode=lambda: False)
            self.assertEqual(
                schedule_batch.ScheduleBatch.alloc_token_slots(batch, 10),
                "tokens-result",
            )
            demand = observed[0][0][0]
            self.assertEqual(demand.request_id, "legacy-request")
            self.assertEqual(demand.slots, 10)
            self.assertEqual(demand.projected_tokens, 110)
            self.assertIsNone(allocator.demands)

    def test_physical_preflight_counts_private_chunks(self):
        residency = object.__new__(adapter.Residency)
        residency.admission_policy = "physical"
        residency.chunk_bytes = 64
        residency.prefill_headroom_bytes = 0
        residency.precommitted = set()
        residency.counters = {}
        residency.pool = SimpleNamespace(device=object())
        residency.arenas = [
            SimpleNamespace(
                planes=1,
                mapped_chunks={0},
                backing=None,
            )
        ]
        with patch.object(
            adapter.torch.cuda,
            "mem_get_info",
            return_value=(64, 0),
        ), patch.object(
            activation,
            "current_prefill_headroom",
            return_value=0,
        ):
            self.assertFalse(residency.can_commit_chunks({1, 2}))
            self.assertTrue(residency.can_commit_chunks({1}))
        self.assertEqual(residency.counters["request_affine_physical_fallbacks"], 1)

    def test_installs_on_legacy_model_runner_api(self):
        calls = []

        class LegacyMHA:
            pass

        class LegacyAllocator:
            pass

        class ModelRunner:
            def init_memory_pool(self, *args, **kwargs):
                calls.append((args, kwargs))
                self.token_to_kv_pool = object.__new__(model_runner.MHATokenToKVPool)
                self.token_to_kv_pool_allocator = object.__new__(
                    model_runner.TokenToKVPoolAllocator
                )
                return "initialized"

        model_runner = _module(
            "sglang.srt.model_executor.model_runner",
            ModelRunner=ModelRunner,
            MHATokenToKVPool=LegacyMHA,
            TokenToKVPoolAllocator=LegacyAllocator,
        )
        modules = {
            "sglang": _module("sglang", __version__="0.5.2"),
            "sglang.srt": _module("sglang.srt"),
            "sglang.srt.model_executor": _module("sglang.srt.model_executor"),
            "sglang.srt.model_executor.model_runner": model_runner,
        }
        with patch.dict(sys.modules, modules):
            adapter.install()

            self.assertNotEqual(model_runner.MHATokenToKVPool, LegacyMHA)
            self.assertNotEqual(
                model_runner.TokenToKVPoolAllocator, LegacyAllocator
            )
            self.assertNotEqual(ModelRunner.init_memory_pool.__name__, "init_memory_pool")

            observed_allocator = model_runner.TokenToKVPoolAllocator
            allocator_probe = SimpleNamespace(
                free_slots=SimpleNamespace(count=40, ranges=[(1, 41)]),
                ctrl=SimpleNamespace(
                    physical_available_tokens=lambda logical, ranges: min(logical, 17)
                ),
            )
            self.assertEqual(observed_allocator.available_size(allocator_probe), 17)

            runner = ModelRunner()
            runner.server_args = SimpleNamespace(
                attention_backend="triton",
                prefill_attention_backend=None,
                decode_attention_backend=None,
                page_size=1,
                disable_cuda_graph=False,
                disable_overlap_schedule=True,
                disable_radix_cache=True,
                tp_size=1,
                pp_size=1,
                dp_size=1,
                enable_two_batch_overlap=False,
                disaggregation_mode="null",
                enable_memory_saver=False,
                speculative_algorithm=None,
                enable_hierarchical_cache=False,
            )
            runner.is_hybrid = False
            runner.is_hybrid_gdn = False
            runner.use_mla_backend = False
            runner.model_config = SimpleNamespace(head_dim=128, v_head_dim=128)
            self.assertIsNone(runner.init_memory_pool(marker=True))
            self.assertEqual(calls, [((), {"marker": True})])

    def test_free_group_normalizes_old_and_new_allocator_apis(self):
        self.assertTrue(adapter._free_group_active(SimpleNamespace(free_group=[])))
        self.assertFalse(adapter._free_group_active(SimpleNamespace(free_group=None)))
        self.assertTrue(
            adapter._free_group_active(
                SimpleNamespace(is_not_in_free_group=False)
            )
        )
        self.assertFalse(
            adapter._free_group_active(
                SimpleNamespace(is_not_in_free_group=True)
            )
        )
        self.assertFalse(
            adapter._free_group_active(
                SimpleNamespace(is_not_in_free_group=True, free_group=[])
            )
        )

    def test_overlap_support_requires_v5_opt_in_without_tbo(self):
        server_args = SimpleNamespace(
            disable_overlap_schedule=True,
            enable_two_batch_overlap=False,
        )
        self.assertTrue(
            adapter._overlap_schedule_supported(
                server_args, mode="baseline", allow=False
            )
        )

        server_args.disable_overlap_schedule = False
        self.assertTrue(
            adapter._overlap_schedule_supported(
                server_args, mode="v5", allow=True
            )
        )
        self.assertFalse(
            adapter._overlap_schedule_supported(
                server_args, mode="v4", allow=True
            )
        )
        self.assertFalse(
            adapter._overlap_schedule_supported(
                server_args, mode="v5", allow=False
            )
        )
        server_args.enable_two_batch_overlap = True
        self.assertFalse(
            adapter._overlap_schedule_supported(
                server_args, mode="v5", allow=True
            )
        )

    def test_v5_reclaim_waits_for_scheduler_work_before_unmapping(self):
        events = []
        ctrl = object.__new__(adapter.Residency)
        ctrl.pool = SimpleNamespace(size=2)
        ctrl.slot_bytes = 1
        ctrl.chunk_bytes = 1
        ctrl.live = adapter.np.array([True, True, False], dtype=adapter.np.bool_)
        ctrl.counts = adapter.np.array([1, 1, 0], dtype=adapter.np.int64)
        ctrl.counters = {"free_calls": 0}
        ctrl.publish = lambda **_kwargs: None
        arena = SimpleNamespace(
            mapped_chunks={1},
            reclaim=lambda indices, synchronize: events.append(
                ("reclaim", set(indices), synchronize)
            ),
        )
        ctrl.arenas = [arena]

        with patch.object(
            adapter.torch.cuda,
            "synchronize",
            side_effect=lambda: events.append(("synchronize",)),
        ):
            ctrl.release(adapter.np.array([1], dtype=adapter.np.int64))

        self.assertEqual(events[0], ("synchronize",))
        self.assertEqual(events[1], ("reclaim", {1}, False))

    def test_deferred_free_takes_owned_copy(self):
        source = object()
        copied = object()

        class Tensor:
            def clone(self):
                return copied

        allocator = SimpleNamespace(
            free_group=[],
            _copy_for_free_group=lambda value: value.clone(),
        )
        adapter._queue_free_group(allocator, Tensor())
        self.assertEqual(allocator.free_group, [copied])

        allocator = SimpleNamespace(free_group=[])
        adapter._queue_free_group(allocator, source)
        self.assertEqual(allocator.free_group, [source])

    def test_physical_admission_reserves_prefill_headroom_before_token_budget(self):
        ctrl = object.__new__(adapter.Residency)
        ctrl.admission_policy = "physical"
        ctrl.pool = SimpleNamespace(device="cuda:0")
        ctrl.arenas = [SimpleNamespace(planes=2, mapped_chunks={0, 1})]
        ctrl.chunk_bytes = 1_000
        ctrl.slot_bytes = 100
        ctrl.prefill_headroom_bytes = 500
        ctrl.retain_chunks = 2
        ctrl.counts = [0] * 100
        ctrl.live = adapter.np.zeros(100, dtype=adapter.np.bool_)
        ctrl.live[0] = True
        ctrl.precommitted = {2}
        ctrl.counters = {}

        with patch.object(
            adapter.torch.cuda, "mem_get_info", return_value=(5_500, 10_000)
        ):
            admitted = ctrl.physical_available_tokens(100, [(1, 101)])

        # Chunks 0-2 are already mapped or precommitted. Free chunks 3-4 fit
        # the remaining two-chunk budget. Slot 0 is the sink, so the first
        # three chunks contain 9 + 10 + 10 usable slots and the next two add
        # 10 each.
        self.assertEqual(admitted, 49)
        self.assertEqual(ctrl.counters["physical_chunk_cost_bytes"], 2_000)
        self.assertEqual(ctrl.counters["physical_headroom_bytes"], 500)
        self.assertEqual(ctrl.counters["physical_limited_calls"], 1)

    def test_physical_admission_is_capped_by_global_broker_remaining_bytes(self):
        ctrl = object.__new__(adapter.Residency)
        ctrl.admission_policy = "physical"
        ctrl.pool = SimpleNamespace(device="cuda:0")
        ctrl.arenas = [SimpleNamespace(
            planes=1,
            mapped_chunks={0},
            backing=SimpleNamespace(broker_available_bytes=1_000),
        )]
        ctrl.chunk_bytes = 1_000
        ctrl.slot_bytes = 100
        ctrl.prefill_headroom_bytes = 0
        ctrl.counters = {}
        ctrl.precommitted = set()

        with patch.object(
            adapter.torch.cuda, "mem_get_info", return_value=(10_000, 10_000)
        ):
            admitted = ctrl.physical_available_tokens(30, [(1, 31)])

        self.assertEqual(admitted, 19)
        self.assertEqual(
            ctrl.counters["physical_global_broker_available_bytes"], 1_000
        )
        self.assertEqual(ctrl.counters["physical_global_broker_limited_calls"], 1)

    def test_physical_admission_stops_growth_under_enforced_fair_pressure(self):
        ctrl = object.__new__(adapter.Residency)
        ctrl.admission_policy = "physical"
        ctrl.pool = SimpleNamespace(device="cuda:0")
        ctrl.arenas = [SimpleNamespace(
            planes=1,
            mapped_chunks={0},
            backing=SimpleNamespace(
                broker_available_bytes=10_000,
                broker_snapshot={
                    "capacity_bytes": 10_000,
                    "outstanding_bytes": 2_000,
                    "available_bytes": 8_000,
                    "fair_mode": "enforce",
                    "fair_target_bytes": 2_500,
                    "tenant_usage_bytes": 2_000,
                    "fairness_pressure": True,
                },
            ),
        )]
        ctrl.chunk_bytes = 1_000
        ctrl.slot_bytes = 100
        ctrl.prefill_headroom_bytes = 0
        ctrl.counters = {}
        ctrl.precommitted = set()

        with patch.object(
            adapter.torch.cuda, "mem_get_info", return_value=(10_000, 10_000)
        ):
            admitted = ctrl.physical_available_tokens(30, [(1, 31)])

        self.assertEqual(admitted, 9)
        self.assertEqual(
            ctrl.counters["physical_global_broker_fair_available_bytes"], 0
        )
        self.assertEqual(
            ctrl.counters["physical_global_broker_fair_limited_calls"], 1
        )

    def test_fair_mode_still_allows_borrowing_without_a_starved_tenant(self):
        ctrl = object.__new__(adapter.Residency)
        ctrl.admission_policy = "physical"
        ctrl.pool = SimpleNamespace(device="cuda:0")
        ctrl.arenas = [SimpleNamespace(
            planes=1,
            mapped_chunks={0},
            backing=SimpleNamespace(
                broker_available_bytes=1_000,
                broker_snapshot={
                    "capacity_bytes": 10_000,
                    "outstanding_bytes": 9_000,
                    "available_bytes": 1_000,
                    "fair_mode": "enforce",
                    "fair_target_bytes": 5_000,
                    "tenant_usage_bytes": 9_000,
                    "fairness_pressure": False,
                },
            ),
        )]
        ctrl.chunk_bytes = 1_000
        ctrl.slot_bytes = 100
        ctrl.prefill_headroom_bytes = 0
        ctrl.counters = {}
        ctrl.precommitted = set()

        with patch.object(
            adapter.torch.cuda, "mem_get_info", return_value=(10_000, 10_000)
        ):
            admitted = ctrl.physical_available_tokens(30, [(1, 31)])

        self.assertEqual(admitted, 19)
        self.assertNotIn("physical_global_broker_fair_limited_calls", ctrl.counters)

    def test_logical_admission_does_not_query_cuda(self):
        ctrl = object.__new__(adapter.Residency)
        ctrl.admission_policy = "logical"
        with patch.object(adapter.torch.cuda, "mem_get_info") as mem_info:
            self.assertEqual(ctrl.physical_available_tokens(7, None), 7)
        mem_info.assert_not_called()

    def test_physical_admission_credits_mapped_holes_before_new_chunks(self):
        ctrl = object.__new__(adapter.Residency)
        ctrl.admission_policy = "physical"
        ctrl.pool = SimpleNamespace(device="cuda:0")
        ctrl.arenas = [SimpleNamespace(planes=1, mapped_chunks={0, 1, 2})]
        ctrl.chunk_bytes = 1_000
        ctrl.slot_bytes = 100
        ctrl.prefill_headroom_bytes = 0
        ctrl.counters = {}
        ctrl.precommitted = set()

        with patch.object(
            adapter.torch.cuda, "mem_get_info", return_value=(0, 10_000)
        ), patch.object(
            adapter.torch.cuda, "device", return_value=nullcontext()
        ), patch.object(adapter.torch.cuda, "empty_cache"):
            admitted = ctrl.physical_available_tokens(
                10, [(1, 6), (11, 16)]
            )

        self.assertEqual(admitted, 10)
        self.assertNotIn("physical_limited_calls", ctrl.counters)

    def test_phase_scoped_workspace_replaces_static_decode_headroom(self):
        ctrl = object.__new__(adapter.Residency)
        ctrl.admission_policy = "physical"
        ctrl.pool = SimpleNamespace(device="cuda:0")
        ctrl.arenas = [SimpleNamespace(planes=1, mapped_chunks={0, 1, 2})]
        ctrl.chunk_bytes = 1_000
        ctrl.slot_bytes = 100
        ctrl.prefill_headroom_bytes = 500
        ctrl.counters = {}
        ctrl.precommitted = set()

        with patch.multiple(
            activation,
            _PREFILL_HEADROOM_SCOPED=True,
            _CONTEXT=activation.local(),
        ), patch.object(
            adapter.torch.cuda, "mem_get_info", return_value=(5_500, 10_000)
        ):
            decode_tokens = ctrl.physical_available_tokens(100, [(3, 103)])
            decode_headroom = ctrl.counters["physical_headroom_bytes"]
            with activation.prefill_admission_scope(750):
                prefill_tokens = ctrl.physical_available_tokens(
                    100, [(3, 103)]
                )
            prefill_headroom = ctrl.counters["physical_headroom_bytes"]

        self.assertGreater(decode_tokens, prefill_tokens)
        self.assertEqual(decode_headroom, 0)
        self.assertEqual(prefill_headroom, 750)


if __name__ == "__main__":
    unittest.main()
