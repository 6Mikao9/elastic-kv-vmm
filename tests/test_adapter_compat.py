import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from elastic_kv import adapter


def _module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


class AdapterCompatibilityTest(unittest.TestCase):
    def setUp(self):
        adapter.MODE = "v5"
        adapter._installed = False
        adapter.CONTROLLERS.clear()

    def tearDown(self):
        adapter._installed = False
        adapter.MODE = "baseline"
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
                return "configured"

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
                guarded(configurator_instance, pre_model_load_memory=1),
                "configured",
            )
            self.assertEqual(calls, [((), {"pre_model_load_memory": 1})])

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

    def test_installs_on_legacy_model_runner_api(self):
        calls = []

        class LegacyMHA:
            pass

        class LegacyAllocator:
            pass

        class ModelRunner:
            def init_memory_pool(self, *args, **kwargs):
                calls.append((args, kwargs))
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


if __name__ == "__main__":
    unittest.main()
