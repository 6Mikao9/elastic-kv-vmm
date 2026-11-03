import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from elastic_kv.graph_support import install_graph_hooks


def test_padding_uses_unallocatable_sink():
    class ReqPool:
        def __init__(self, size, max_context_len, device, enable_memory_saver):
            self.size = size
            self.req_to_token = [None] * size
            self.clear()

        def clear(self):
            self.free_slots = list(range(self.size))

    class Indices:
        def __init__(self):
            self.values = [0] * 4

        def __getitem__(self, key):
            def fill(value):
                self.values[key] = [value] * len(self.values[key])

            return SimpleNamespace(fill_=fill)

    class Runner:
        disable_padding = False
        graphs = {4: object()}

        def capture(self):
            pass

        def replay_prepare(self, batch, pp_proxy_tensors=None):
            self.req_pool_indices.values[:batch.batch_size] = batch.rows

        def replay(self, batch, skip_attn_backend_init=False, pp_proxy_tensors=None):
            self.replay_prepare(batch, pp_proxy_tensors)
            return list(self.req_pool_indices.values)

    modules = {}
    for name in (
        "sglang", "sglang.srt", "sglang.srt.model_executor",
        "sglang.srt.model_executor.cuda_graph_runner",
        "sglang.srt.model_executor.model_runner", "elastic_kv.adapter",
    ):
        modules[name] = ModuleType(name)
        parent, _, child = name.rpartition(".")
        if parent in modules:
            setattr(modules[parent], child, modules[name])
    graph_module = modules["sglang.srt.model_executor.cuda_graph_runner"]
    graph_module.CudaGraphRunner = Runner
    model_runner = modules["sglang.srt.model_executor.model_runner"]
    model_runner.ReqToTokenPool = ReqPool
    ctrl = SimpleNamespace(pool=object(), counters={})
    modules["elastic_kv.adapter"].CONTROLLERS = [ctrl]

    with patch.dict(sys.modules, modules):
        install_graph_hooks()
        install_graph_hooks()
        pool = model_runner.ReqToTokenPool(4, 16, "cpu", False)
        assert len(pool.req_to_token) == 5
        assert pool.free_slots == [0, 1, 2, 3]
        pool.free_slots.clear()
        pool.clear()
        assert pool.free_slots == [0, 1, 2, 3]

        runner = Runner()
        runner.model_runner = SimpleNamespace(token_to_kv_pool=ctrl.pool, req_to_token_pool=pool)
        runner.req_pool_indices = Indices()
        for rows, expected in (
            ([1, 2, 3], [1, 2, 3, 4]),
            ([0, 3, 1, 2], [0, 3, 1, 2]),
            ([2], [2, 4, 4, 4]),
            ([0, 3], [0, 3, 4, 4]),
        ):
            assert runner.replay(SimpleNamespace(batch_size=len(rows), rows=rows)) == expected
        assert ctrl.counters["actual_graph_replay_calls"] == 4
        assert ctrl.counters["padded_graph_replay_calls"] == 3


def load_tests(loader, tests, pattern):
    tests.addTest(unittest.FunctionTestCase(test_padding_uses_unallocatable_sink))
    return tests


if __name__ == "__main__":
    unittest.main()
