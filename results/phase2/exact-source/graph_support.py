"""Decode CUDA Graph hooks for the no-padding SGLang 0.5.2 prototype."""
from functools import wraps


def install_graph_hooks():
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

    from .adapter import CONTROLLERS

    if getattr(CudaGraphRunner, "_ekv_graph_hooks", False):
        return
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
        # Triton captures with sequence length one. Row zero is free during
        # capture, but may still contain a KV index from a finished request.
        requests.req_to_token[0, 0].zero_()
        self.req_pool_indices.zero_()
        self.out_cache_loc.zero_()
        result = original_capture(self)
        ctrl.counters["graph_capture_calls"] = ctrl.counters.get("graph_capture_calls", 0) + 1
        ctrl.publish(force=True)
        return result

    @wraps(original_replay)
    def replay(self, forward_batch, skip_attn_backend_init=False, pp_proxy_tensors=None):
        ctrl = controller(self)
        if ctrl is not None:
            if skip_attn_backend_init or forward_batch.batch_size not in self.graphs:
                raise RuntimeError("Elastic KV CUDA Graph replay requires an exact captured batch")
        result = original_replay(self, forward_batch, skip_attn_backend_init, pp_proxy_tensors)
        if ctrl is not None:
            ctrl.counters["actual_graph_replay_calls"] = (
                ctrl.counters.get("actual_graph_replay_calls", 0) + 1
            )
        return result

    CudaGraphRunner.capture = capture
    CudaGraphRunner.replay = replay
    CudaGraphRunner._ekv_graph_hooks = True
