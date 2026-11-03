"""Optional CUDA memory pressure instrumentation for V5.

The configured headroom is an experimental target. Remaining pressure is
recorded and normal allocation continues; scheduler admission is unchanged.
"""

import os
from functools import wraps
from threading import local
from time import perf_counter_ns


_CONTEXT = local()
_POLICY = "off"
_HEADROOM = 0
_CUDA = None


def _add(counters, key, value=1):
    counters[key] = counters.get(key, 0) + int(value)


def _minimum(counters, key, value):
    counters[key] = min(counters.get(key, value), value)


def before_admit(ctrl, ranges):
    """Run after range validation and before mapping the required chunks."""
    if _POLICY == "off" or not ranges:
        return
    prefill = getattr(_CONTEXT, "prefill", False)
    needed = set()
    for start, stop in ranges:
        first = start * ctrl.slot_bytes // ctrl.chunk_bytes
        last = (stop * ctrl.slot_bytes - 1) // ctrl.chunk_bytes
        needed.update(range(first, last + 1))
    pending = sum(len(needed - arena.mapped_chunks) * getattr(arena, "planes", 1)
                  for arena in ctrl.arenas) * ctrl.chunk_bytes
    if not prefill and not pending:
        return
    counters = ctrl.counters
    device = ctrl.pool.device
    headroom = _HEADROOM if prefill else 0
    target = pending + headroom
    free, _ = _CUDA.mem_get_info(device)
    _add(counters, "activation_admission_samples")
    counters["activation_prefill_headroom_bytes"] = _HEADROOM
    _minimum(counters, "activation_admission_free_low_bytes", free)
    if free >= target:
        return
    _add(counters, "activation_pressure_calls")

    if prefill and _POLICY in ("trim", "both"):
        protected = needed | {int(index) for index in ctrl.chunks_for(ctrl.live.nonzero()[0])}
        candidates = [(arena, arena.mapped_chunks - protected) for arena in ctrl.arenas]
        if any(indices for _, indices in candidates):
            start = perf_counter_ns()
            backed_before = ctrl.backed
            _CUDA.synchronize(device)
            for arena, indices in candidates:
                if indices:
                    arena.reclaim(indices, synchronize=False)
                    _add(counters, "activation_trim_chunks", len(indices))
            ctrl.backed = sum(arena.backed_bytes for arena in ctrl.arenas)
            _add(counters, "activation_trim_calls")
            _add(counters, "activation_trim_bytes", backed_before - ctrl.backed)
            _add(counters, "activation_trim_ns", perf_counter_ns() - start)
            free, _ = _CUDA.mem_get_info(device)

    if free < target and _POLICY in ("recycle", "both"):
        start = perf_counter_ns()
        with _CUDA.device(device):
            _CUDA.empty_cache()
        after, _ = _CUDA.mem_get_info(device)
        _add(counters, "activation_flush_calls")
        _add(counters, "activation_flush_freed_bytes", max(0, after - free))
        _add(counters, "activation_flush_ns", perf_counter_ns() - start)
        free = after

    if free < target:
        _add(counters, "activation_pressure_unresolved_calls")
        _add(counters, "activation_pressure_shortfall_bytes", target - free)


def install_activation_hooks():
    global _POLICY, _HEADROOM, _CUDA
    policy = os.environ.get("EKV_ACTIVATION_POLICY", "off")
    if policy not in ("off", "telemetry", "trim", "recycle", "both"):
        raise ValueError(f"Unknown EKV_ACTIVATION_POLICY={policy}")
    if policy == "off":
        return
    headroom = int(os.environ.get("EKV_PREFILL_HEADROOM_MIB", "0"))
    if headroom < 0:
        raise ValueError("EKV_PREFILL_HEADROOM_MIB must be nonnegative")

    from torch import cuda
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

    if getattr(ModelRunner, "_ekv_activation_hooks", False):
        return
    _POLICY, _HEADROOM, _CUDA = policy, headroom << 20, cuda
    original_prepare = ScheduleBatch.prepare_for_extend
    original_forward = ModelRunner.forward_extend

    @wraps(original_prepare)
    def prepare(self, *args, **kwargs):
        previous = getattr(_CONTEXT, "prefill", False)
        _CONTEXT.prefill = True
        try:
            return original_prepare(self, *args, **kwargs)
        finally:
            _CONTEXT.prefill = previous

    @wraps(original_forward)
    def forward(self, *args, **kwargs):
        ctrl = getattr(self.token_to_kv_pool, "residency", None)
        if ctrl is None:
            return original_forward(self, *args, **kwargs)
        device = ctrl.pool.device
        counters = ctrl.counters
        allocated_before = cuda.memory_allocated(device)
        counters["activation_prefill_allocated_before_bytes"] = allocated_before
        counters["activation_prefill_reserved_before_bytes"] = cuda.memory_reserved(device)
        counters["activation_prefill_free_before_bytes"] = cuda.mem_get_info(device)[0]
        cuda.reset_peak_memory_stats(device)
        result = original_forward(self, *args, **kwargs)
        peak = cuda.max_memory_allocated(device)
        counters["activation_prefill_allocated_after_bytes"] = cuda.memory_allocated(device)
        counters["activation_prefill_reserved_after_bytes"] = cuda.memory_reserved(device)
        counters["activation_prefill_free_after_bytes"] = cuda.mem_get_info(device)[0]
        counters["activation_prefill_peak_allocated_bytes"] = max(
            counters.get("activation_prefill_peak_allocated_bytes", 0), peak)
        counters["activation_prefill_peak_growth_bytes"] = max(
            counters.get("activation_prefill_peak_growth_bytes", 0), peak - allocated_before)
        batch = args[0] if args else kwargs.get("forward_batch")
        if batch is not None:
            key = f"activation_peak_tokens_{batch.input_ids.numel()}_batch_{batch.batch_size}_bytes"
            counters[key] = max(counters.get(key, 0), peak - allocated_before)
        _add(counters, "activation_prefill_calls")
        return result

    ScheduleBatch.prepare_for_extend = prepare
    ModelRunner.forward_extend = forward
    ModelRunner._ekv_activation_hooks = True
