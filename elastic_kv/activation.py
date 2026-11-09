"""Optional CUDA memory pressure instrumentation for V5.

The configured headroom is an experimental target. Remaining pressure is
recorded and normal allocation continues; scheduler admission is unchanged.
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import wraps
from threading import local
from time import monotonic_ns, perf_counter_ns

_CONTEXT = local()
_POLICY = "off"
_HEADROOM = 0
_CUDA = None
_PREFILL_WORKSPACE_RATE = 0
_PREFILL_WORKSPACE_CAP = 0
_PREFILL_HEADROOM_SCOPED = False
_GLOBAL_PREFILL_MODE = "off"
_PENDING_PREFILL_CLAIM = None


@dataclass(frozen=True, slots=True)
class PrefillWorkspaceEstimate:
    bytes: int
    prompt_tokens: int = 0
    requests: int = 0
    unestimated: int = 0


@dataclass(frozen=True, slots=True)
class PrefillCapacityClaim:
    key: str
    bytes: int
    backing: object
    counters: dict


def _add(counters, key, value=1):
    counters[key] = counters.get(key, 0) + int(value)


def _minimum(counters, key, value):
    counters[key] = min(counters.get(key, value), value)


def _record_activation_workspace_sample(
    counters, *, growth_bytes, batch_tokens, batch_size
):
    """Record a conservative per-token activation estimate.

    The exact token/batch key preserves shape-specific observations. The
    global maximum can feed future-prefill admission without under-reserving a
    previously observed shape.
    """

    for value, name in (
        (growth_bytes, "growth_bytes"),
        (batch_tokens, "batch_tokens"),
        (batch_size, "batch_size"),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if batch_size < 1 or batch_tokens < 1 or growth_bytes < 1:
        _add(counters, "activation_prefill_bytes_per_token_samples_skipped")
        return
    key = f"activation_peak_tokens_{batch_tokens}_batch_{batch_size}_bytes"
    counters[key] = max(counters.get(key, 0), growth_bytes)
    rate = -(-growth_bytes // batch_tokens)
    counters["activation_prefill_bytes_per_token_max_observed"] = max(
        counters.get("activation_prefill_bytes_per_token_max_observed", 0),
        rate,
    )
    _add(counters, "activation_prefill_bytes_per_token_samples")


def _observe_activation_workspace_sample(counters, *, batch):
    """Best-effort calibration; observability must not fail inference."""

    try:
        tokens = _batch_prefill_tokens(batch)
        growth_bytes = counters.get("activation_prefill_peak_growth_bytes_last")
        batch_size = getattr(batch, "batch_size", None)
        if tokens is None or growth_bytes is None or batch_size is None:
            _add(counters, "activation_prefill_bytes_per_token_samples_skipped")
            return
        _record_activation_workspace_sample(
            counters,
            growth_bytes=growth_bytes,
            batch_tokens=tokens,
            batch_size=batch_size,
        )
    except Exception:
        _add(counters, "activation_prefill_bytes_per_token_sample_errors")


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
    matched_candidates = []
    kv = getattr(request, "kv", None)
    committed = getattr(kv, "kv_committed_len", None)
    if isinstance(committed, int) and not isinstance(committed, bool) and committed >= 0:
        matched_candidates.append(committed)
    prefix_indices = getattr(request, "prefix_indices", None)
    if prefix_indices is not None:
        try:
            matched_candidates.append(len(prefix_indices))
        except TypeError:
            pass
    matched = getattr(request, "num_matched_prefix_tokens", 0) or 0
    if isinstance(matched, bool) or not isinstance(matched, int) or matched < 0:
        return None
    matched_candidates.append(matched)
    matched = max(matched_candidates, default=0)
    return max(1, total - min(matched, total))


def _batch_prefill_tokens(batch):
    """Return selected extend tokens across supported SGLang staging APIs."""

    for attribute in ("input_ids", "prefill_input_ids_cpu"):
        ids = getattr(batch, attribute, None)
        if ids is None:
            continue
        numel = getattr(ids, "numel", None)
        try:
            tokens = numel() if callable(numel) else len(ids)
            tokens = int(tokens)
        except Exception:
            continue
        if tokens >= 0:
            return tokens
    return None


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


def estimate_prefill_batch_workspace(batch, *, bytes_per_token, max_bytes):
    """Estimate the workspace of the concrete batch selected by SGLang."""

    if bytes_per_token < 0 or max_bytes < 0:
        raise ValueError("prefill workspace limits must be nonnegative")
    if _batch_is_empty(batch):
        return PrefillWorkspaceEstimate(bytes=0)
    # Newer SGLang stages extend tokens in ``prefill_input_ids_cpu`` and leaves
    # ``input_ids`` as None until forward resolves the pinned CPU tensor on the
    # compute stream. 0.5.2 keeps the flattened extend tensor in ``input_ids``.
    tokens = _batch_prefill_tokens(batch)
    if tokens is not None:
        return PrefillWorkspaceEstimate(
            bytes=min(max_bytes, tokens * bytes_per_token),
            prompt_tokens=tokens,
            requests=1 if tokens else 0,
        )
    return None


def _find_broker_backing(controller):
    """Return the capacity owner shared by all arenas on this device."""

    for arena in getattr(controller, "arenas", ()):
        backing = getattr(arena, "backing", None)
        if backing is not None and getattr(backing, "has_global_broker", False):
            return backing
    return None


def _observe_global_prefill_claim(backing, byte_count, counters):
    snapshot = getattr(backing, "broker_snapshot", None)
    if not isinstance(snapshot, dict):
        return False
    available = snapshot.get("available_bytes")
    would_admit = (
        isinstance(available, int)
        and available >= byte_count
        and not (
            snapshot.get("fair_mode") == "enforce"
            and snapshot.get("fairness_pressure")
        )
    )
    _add(
        counters,
        "activation_global_prefill_observe_would_admit"
        if would_admit
        else "activation_global_prefill_observe_would_deny",
    )
    counters["activation_global_prefill_observe_available_bytes"] = available
    return would_admit


def _reserve_global_prefill_claim(controller, estimate, counters):
    global _PENDING_PREFILL_CLAIM

    byte_count = int(estimate.bytes)
    if byte_count <= 0:
        return None, "empty"
    backing = _find_broker_backing(controller)
    if backing is None:
        _add(counters, "activation_global_prefill_missing_broker")
        return None, "missing-broker"
    if _GLOBAL_PREFILL_MODE == "observe":
        _observe_global_prefill_claim(backing, byte_count, counters)
        return None, "observed"

    key = f"prefill-{os.getpid()}-{monotonic_ns()}"
    _add(counters, "activation_global_prefill_claim_checks")
    counters["activation_global_prefill_claim_bytes_last"] = byte_count
    counters["activation_global_prefill_claim_bytes_max"] = max(
        counters.get("activation_global_prefill_claim_bytes_max", 0), byte_count
    )
    admitted = backing.claim_broker_capacity(key, byte_count)
    if admitted is not True:
        _add(counters, "activation_global_prefill_claim_denials")
        return None, "denied"

    claim = PrefillCapacityClaim(key, byte_count, backing, counters)
    _PENDING_PREFILL_CLAIM = claim
    _add(counters, "activation_global_prefill_claim_admissions")
    return claim, "claimed"


def _consume_pending_prefill_claim():
    global _PENDING_PREFILL_CLAIM
    claim = _PENDING_PREFILL_CLAIM
    _PENDING_PREFILL_CLAIM = None
    return claim


def _release_prefill_claim(claim, reason):
    if claim is None:
        return 0
    released = claim.backing.release_broker_claim(claim.key)
    if released:
        _add(claim.counters, f"activation_global_prefill_claim_released_{reason}")
    else:
        _add(claim.counters, f"activation_global_prefill_claim_release_missing_{reason}")
    return released


def _shrink_prefill_claim(claim, maximum_bytes):
    global _PENDING_PREFILL_CLAIM

    if claim is None or maximum_bytes < 1 or maximum_bytes >= claim.bytes:
        return claim
    revised = claim.backing.shrink_broker_claim(claim.key, maximum_bytes)
    if revised is None or revised >= claim.bytes:
        return claim
    updated = replace(claim, bytes=int(revised))
    if _PENDING_PREFILL_CLAIM is claim:
        _PENDING_PREFILL_CLAIM = updated
    _add(claim.counters, "activation_global_prefill_claim_shrinks")
    _add(
        claim.counters,
        "activation_global_prefill_claim_shrunk_bytes",
        claim.bytes - updated.bytes,
    )
    claim.counters["activation_global_prefill_claim_bytes_after_select"] = (
        updated.bytes
    )
    return updated


def _abandon_prefill_claim(claim, reason):
    if claim is not None:
        _add(
            claim.counters,
            f"activation_global_prefill_claim_abandoned_{reason}",
        )


def _batch_is_empty(batch):
    if batch is None:
        return True
    is_empty = getattr(batch, "is_empty", None)
    if callable(is_empty):
        return bool(is_empty())
    batch_size = getattr(batch, "batch_size", None)
    if callable(batch_size):
        batch_size = batch_size()
    try:
        return batch_size is not None and int(batch_size) == 0
    except (TypeError, ValueError):
        return False


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
    global _GLOBAL_PREFILL_MODE, _PENDING_PREFILL_CLAIM
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
    global_prefill_mode = os.environ.get(
        "EKV_GLOBAL_BROKER_PREFILL_MODE", "off"
    ).strip().lower()
    if global_prefill_mode not in ("off", "observe", "enforce"):
        raise ValueError(
            f"Unknown EKV_GLOBAL_BROKER_PREFILL_MODE={global_prefill_mode}"
        )
    _POLICY = policy
    _HEADROOM = headroom << 20
    _PREFILL_WORKSPACE_RATE = workspace_rate
    _PREFILL_WORKSPACE_CAP = workspace_cap
    _PREFILL_HEADROOM_SCOPED = scoped
    _GLOBAL_PREFILL_MODE = global_prefill_mode
    _PENDING_PREFILL_CLAIM = None
    _CUDA = None
    needs_activation_hooks = policy != "off" or global_prefill_mode != "off"
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
                claim = _consume_pending_prefill_claim()
                if ctrl is None:
                    try:
                        result = original_forward(self, *args, **kwargs)
                    except BaseException:
                        _abandon_prefill_claim(claim, "forward_error")
                        raise
                    _release_prefill_claim(claim, "forward")
                    return result
                try:
                    if _POLICY == "off":
                        result = original_forward(self, *args, **kwargs)
                    else:
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
                        counters["activation_prefill_peak_growth_bytes_last"] = (
                            peak - allocated_before
                        )
                        batch = args[0] if args else kwargs.get("forward_batch")
                        if batch is not None:
                            _observe_activation_workspace_sample(
                                counters, batch=batch
                            )
                        _add(counters, "activation_prefill_calls")
                except BaseException:
                    _abandon_prefill_claim(claim, "forward_error")
                    raise
                _release_prefill_claim(claim, "forward")
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
                else:
                    counters = {}

                claim = None
                if _GLOBAL_PREFILL_MODE != "off":
                    if controller is None:
                        _add(
                            counters,
                            "activation_global_prefill_missing_controller",
                        )
                        if _GLOBAL_PREFILL_MODE == "enforce":
                            return None
                    else:
                        claim, disposition = _reserve_global_prefill_claim(
                            controller, estimate, counters
                        )
                        _add(
                            counters,
                            f"activation_global_prefill_{disposition}",
                        )
                        if _GLOBAL_PREFILL_MODE == "enforce" and claim is None:
                            return None

                try:
                    with prefill_admission_scope(estimate.bytes):
                        batch = original_prefill(self, *args, **kwargs)
                except BaseException:
                    _consume_pending_prefill_claim()
                    _abandon_prefill_claim(claim, "scheduler_error")
                    raise

                if claim is not None and _batch_is_empty(batch):
                    _consume_pending_prefill_claim()
                    _release_prefill_claim(claim, "empty_batch")
                elif claim is not None:
                    selected = estimate_prefill_batch_workspace(
                        batch,
                        bytes_per_token=_PREFILL_WORKSPACE_RATE,
                        max_bytes=_PREFILL_WORKSPACE_CAP,
                    )
                    if selected is not None:
                        _add(
                            counters,
                            "activation_global_prefill_batch_estimates",
                        )
                        counters[
                            "activation_global_prefill_batch_tokens_last"
                        ] = selected.prompt_tokens
                        counters[
                            "activation_global_prefill_batch_bytes_last"
                        ] = selected.bytes
                        claim = _shrink_prefill_claim(claim, selected.bytes)
                    else:
                        _add(
                            counters,
                            "activation_global_prefill_batch_estimates_unknown",
                        )
                return batch

            prefill_with_workspace._elastic_activation_prefill_hook = True
            Scheduler.get_new_batch_prefill = prefill_with_workspace
