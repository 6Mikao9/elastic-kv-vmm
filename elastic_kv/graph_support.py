"""Decode CUDA Graph hooks for the SGLang 0.5.2 prototype."""
from functools import wraps


def install_graph_hooks():
    try:
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
    except ImportError:
        # Newer SGLang moved decode graphs under model_executor.runner and its
        # graph buffer registry already uses token/request slot zero for padded
        # rows.  The permanent backing for slot zero is handled by Residency, so
        # no extra request row or replay hook is needed here.
        return

    from sglang.srt.model_executor import model_runner

    from .adapter import CONTROLLERS

    if getattr(CudaGraphRunner, "_ekv_graph_hooks", False):
        return
    class SinkReqToTokenPool(model_runner.ReqToTokenPool):
        def __init__(self, size, max_context_len, device, enable_memory_saver):
            super().__init__(size + 1, max_context_len, device, enable_memory_saver)
            self.size = size
            self.clear()
            self._ekv_sink_row = size

    original_prepare = CudaGraphRunner.replay_prepare
    original_capture = CudaGraphRunner.capture
    original_replay = CudaGraphRunner.replay

    def controller(runner):
        pool = runner.model_runner.token_to_kv_pool
        return next((ctrl for ctrl in CONTROLLERS if ctrl.pool is pool), None)

    @wraps(original_capture)
    def capture(self):
        ctrl = controller(self)
        if ctrl is None:
            return original_capture(self)
        requests = self.model_runner.req_to_token_pool
        if ctrl.live[1:].any() or requests.available_size() != requests.size:
            raise RuntimeError("Elastic KV CUDA Graph capture requires an idle request pool")
        # Triton captures with sequence length one.
        sink = requests._ekv_sink_row
        requests.req_to_token[sink, 0].zero_()
        self.req_pool_indices.fill_(sink)
        self.out_cache_loc.zero_()
        result = original_capture(self)
        ctrl.counters["graph_capture_calls"] = ctrl.counters.get("graph_capture_calls", 0) + 1
        ctrl.publish(force=True)
        return result

    @wraps(original_prepare)
    def replay_prepare(self, forward_batch, pp_proxy_tensors=None):
        if controller(self) is not None and not self.disable_padding:
            sink = self.model_runner.req_to_token_pool._ekv_sink_row
            self.req_pool_indices[forward_batch.batch_size:].fill_(sink)
        return original_prepare(self, forward_batch, pp_proxy_tensors)

    @wraps(original_replay)
    def replay(self, forward_batch, skip_attn_backend_init=False, pp_proxy_tensors=None):
        ctrl = controller(self)
        if ctrl is not None:
            if skip_attn_backend_init or (
                self.disable_padding and forward_batch.batch_size not in self.graphs
            ):
                raise RuntimeError("Elastic KV CUDA Graph replay requires an exact captured batch")
        padded = ctrl is not None and not self.disable_padding and forward_batch.batch_size not in self.graphs
        result = original_replay(self, forward_batch, skip_attn_backend_init, pp_proxy_tensors)
        if ctrl is not None:
            ctrl.counters["actual_graph_replay_calls"] = (
                ctrl.counters.get("actual_graph_replay_calls", 0) + 1
            )
            if padded:
                ctrl.counters["padded_graph_replay_calls"] = (
                    ctrl.counters.get("padded_graph_replay_calls", 0) + 1
                )
        return result

    model_runner.ReqToTokenPool = SinkReqToTokenPool
    CudaGraphRunner.replay_prepare = replay_prepare
    CudaGraphRunner.capture = capture
    CudaGraphRunner.replay = replay
    CudaGraphRunner._ekv_graph_hooks = True
