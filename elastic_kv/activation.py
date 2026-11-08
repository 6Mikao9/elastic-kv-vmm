"""Optional CUDA memory pressure instrumentation for V5.

The configured headroom is an experimental target. Remaining pressure is
recorded and normal allocation continues; scheduler admission is unchanged.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import os
from functools import wraps
from threading import local
from time import perf_counter_ns


_CONTEXT = local()
_POLICY = "off"
_HEADROOM = 0
_CUDA = None
_PREFILL_WORKSPACE_RATE = 0
_PREFILL_WORKSPACE_CAP = 0
_PREFILL_HEADROOM_SCOPED = False


@dataclass(frozen=True, slots=True)
class PrefillWorkspaceEstimate:
    bytes: int
    prompt_tokens: int = 0
    requests: int = 0
    unestimated: int = 0


def _add(counters, key, value=1):
    counters[key] = counters.get(key, 0) + int(value)


def _minimum(counters, key, value):
    counters[key] = min(counters.get(key, value), value)


def _request_prompt_tokens(request):
    ids = getattr(request, "origin_input_ids", None)
    if ids is None:
        ids = getattr(request, "input_ids", None)
    if ids is None:
        fill = getattr(request, "get_fill_ids", None)
        if callable(fill):
            ids = fill()
    if ids is None:
        return None
    try:
        total = len(ids)
    except TypeError:
        return None
    if total < 1:
        return None
    matched = getattr(request, "num_matched_prefix_tokens", 0) or 0
    if isinstance(matched, bool) or not isinstance(matched, int) or matched < 0:
        return None
    return max(1, total - min(matched, total))


def estimate_prefill_workspace(waiting_queue, *, bytes_per_token, max_bytes):
    if bytes_per_token < 0 or max_bytes < 0:
        raise ValueError("prefill workspace limits must be nonnegative")
    prompt_tokens = 0
    requests = 0
    unestimated = 0
    for request in waiting_queue:
        tokens = _request_prompt_tokens(request)
        if tokens is None:
            unestimated += 1
            continue
        prompt_tokens += tokens
        requests += 1
    return PrefillWorkspaceEstimate(
        bytes=min(max_bytes, prompt_tokens * bytes_per_token),
        prompt_tokens=prompt_tokens,
        requests=requests,
        unestimated=unestimated,
    )


def current_prefill_headroom(fallback):
    if isinstance(fallback, bool) or not isinstance(fallback, int) or fallback < 0:
        raise ValueError("fallback prefill headroom must be a nonnegative integer")
    active = bool(getattr(_CONTEXT, "admission_prefill", False))
    dynamic = int(getattr(_CONTEXT, "admission_workspace_bytes", 0))
    if _PREFILL_HEADROOM_SCOPED and not active:
        return 0
    return max(fallback, dynamic if active else 0)


@contextmanager
def prefill_admission_scope(workspace_bytes):
    previous_active = getattr(_CONTEXT, "admission_prefill", False)
    previous_bytes = getattr(_CONTEXT, "admission_workspace_bytes", 0)
    _CONTEXT.admission_prefill = True
    _CONTEXT.admission_workspace_bytes = int(workspace_bytes)
    try:
        yield
    finally:
        _CONTEXT.admission_prefill = previous_active
        _CONTEXT.admission_workspace_bytes = previous_bytes


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
    headroom = current_prefill_headroom(_HEADROOM) if prefill else 0
    target = pending + headroom
    free, _ = _CUDA.mem_get_info(device)
    _add(counters, "activation_admission_samples")
    counters["activation_prefill_headroom_bytes"] = headroom
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
    global _PREFILL_WORKSPACE_RATE, _PREFILL_WORKSPACE_CAP
    global _PREFILL_HEADROOM_SCOPED
    policy = os.environ.get("EKV_ACTIVATION_POLICY", "off")
    if policy not in ("off", "telemetry", "trim", "recycle", "both"):
        raise ValueError(f"Unknown EKV_ACTIVATION_POLICY={policy}")
    headroom = int(os.environ.get("EKV_PREFILL_HEADROOM_MIB", "0"))
    if headroom < 0:
        raise ValueError("EKV_PREFILL_HEADROOM_MIB must be nonnegative")
    workspace_rate = int(
        os.environ.get("EKV_PREFILL_WORKSPACE_BYTES_PER_TOKEN", "0")
    )
    if workspace_rate < 0:
        raise ValueError(
            "EKV_PREFILL_WORKSPACE_BYTES_PER_TOKEN must be nonnegative"
        )
    workspace_cap = int(os.environ.get("EKV_PREFILL_WORKSPACE_MAX_MIB", "256")) << 20
    if workspace_cap < 0:
        raise ValueError("EKV_PREFILL_WORKSPACE_MAX_MIB must be nonnegative")
    scoped = os.environ.get("EKV_PREFILL_HEADROOM_SCOPED", "0").lower() in (
        "1", "true", "yes", "on"
    )
    _POLICY = policy
    _HEADROOM = headroom << 20
    _PREFILL_WORKSPACE_RATE = workspace_rate
    _PREFILL_WORKSPACE_CAP = workspace_cap
    _PREFILL_HEADROOM_SCOPED = scoped
    _CUDA = None
    needs_activation_hooks = policy != "off"
    needs_scheduler_hook = workspace_rate > 0 or scoped
    if not needs_activation_hooks and not needs_scheduler_hook:
        return

    from torch import cuda
    _CUDA = cuda
    if needs_activation_hooks:
        from sglang.srt.managers.schedule_batch import ScheduleBatch
        from sglang.srt.model_executor.model_runner import ModelRunner

        if not getattr(ModelRunner, "_ekv_activation_hooks", False):
            original_prepare = ScheduleBatch.prepare_for_extend
            forward_name = (
                "forward_extend"
                if hasattr(ModelRunner, "forward_extend")
                else "forward"
            )
            original_forward = getattr(ModelRunner, forward_name)

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
                    counters.get("activation_prefill_peak_growth_bytes", 0),
                    peak - allocated_before,
                )
                batch = args[0] if args else kwargs.get("forward_batch")
                if batch is not None:
                    key = f"activation_peak_tokens_{batch.input_ids.numel()}_batch_{batch.batch_size}_bytes"
                    counters[key] = max(counters.get(key, 0), peak - allocated_before)
                _add(counters, "activation_prefill_calls")
                return result

            ScheduleBatch.prepare_for_extend = prepare
            setattr(ModelRunner, forward_name, forward)
            ModelRunner._ekv_activation_hooks = True

    if needs_scheduler_hook:
        from sglang.srt.managers.scheduler import Scheduler

        if not getattr(
            Scheduler.get_new_batch_prefill,
            "_elastic_activation_prefill_hook",
            False,
        ):
            original_prefill = Scheduler.get_new_batch_prefill

            @wraps(original_prefill)
            def prefill_with_workspace(self, *args, **kwargs):
                waiting = list(getattr(self, "waiting_queue", ()))
                chunked = getattr(self, "chunked_req", None)
                if chunked is not None and all(
                    item is not chunked for item in waiting
                ):
                    waiting.append(chunked)
                estimate = estimate_prefill_workspace(
                    waiting,
                    bytes_per_token=_PREFILL_WORKSPACE_RATE,
                    max_bytes=_PREFILL_WORKSPACE_CAP,
                )
                controller = getattr(
                    getattr(self, "token_to_kv_pool_allocator", None),
                    "ctrl",
                    None,
                )
                if controller is None:
                    controller = getattr(
                        getattr(self, "token_to_kv_pool", None),
                        "residency",
                        None,
                    )
                if controller is not None:
                    counters = controller.counters
                    _add(counters, "activation_dynamic_prefill_reservations")
                    counters["activation_dynamic_prefill_bytes_last"] = (
                        estimate.bytes
                    )
                    counters["activation_dynamic_prefill_bytes_max"] = max(
                        counters.get("activation_dynamic_prefill_bytes_max", 0),
                        estimate.bytes,
                    )
                    counters["activation_dynamic_prefill_prompt_tokens"] = (
                        estimate.prompt_tokens
                    )
                    counters["activation_dynamic_prefill_requests"] = (
                        estimate.requests
                    )
                    counters["activation_dynamic_prefill_unestimated"] = (
                        estimate.unestimated
                    )
                with prefill_admission_scope(estimate.bytes):
                    return original_prefill(self, *args, **kwargs)

            prefill_with_workspace._elastic_activation_prefill_hook = True
            Scheduler.get_new_batch_prefill = prefill_with_workspace
