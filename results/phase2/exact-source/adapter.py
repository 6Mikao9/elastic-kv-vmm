"""Opt-in dense-MHA adapter; no attention kernel or model weights are changed.

Supported: SGLang 0.5.2, TP=1, page_size=1, Triton attention, eager,
no overlap, no radix cache, no speculative decoding or disaggregation.
"""
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

from .backing import Backing
from .slots import FreeSlots

MODE = os.environ.get("EKV_MODE", "baseline")
VALID_MODES = ("baseline", "v0", "v1", "v2", "v3", "v4", "v5")
HOST_FREE_MODES = ("v2", "v3", "v4")
CONTROLLERS = []
_installed = False


class PlaneProjection:
    """Project logical chunks onto equally padded contiguous K/V planes."""

    def __init__(self, plane_bytes, planes, chunk_bytes, device):
        self.plane_chunks = (plane_bytes + chunk_bytes - 1) // chunk_bytes
        self.plane_stride = self.plane_chunks * chunk_bytes
        self.plane_bytes = plane_bytes
        self.planes = planes
        self.backing = Backing(self.plane_stride * planes, chunk_bytes, device)
        self.mapped_chunks = set()

    def _physical_chunks(self, indices):
        return [plane * self.plane_chunks + index
                for plane in range(self.planes) for index in indices]

    def ensure(self, indices):
        pending = set(indices) - self.mapped_chunks
        if pending:
            self.backing.ensure(self._physical_chunks(pending))
            self.mapped_chunks.update(pending)

    def reclaim(self, indices, *, synchronize=True):
        pending = self.mapped_chunks.intersection(indices)
        if pending:
            self.backing.reclaim(self._physical_chunks(pending), synchronize=synchronize)
            self.mapped_chunks.difference_update(pending)

    @property
    def backed_bytes(self):
        return self.backing.backed_bytes

    @property
    def metrics(self):
        return self.backing.metrics

    def views(self, dtype, shape):
        raw = self.backing.raw
        return [raw.narrow(0, plane * self.plane_stride, self.plane_bytes).view(dtype).view(shape)
                for plane in range(self.planes)]


class Residency:
    def __init__(self, pool, arenas, slot_bytes, layout):
        self.pool = pool
        self.arenas = arenas
        self.slot_bytes = slot_bytes
        self.layout = layout
        self.chunk_bytes = int(os.environ.get("EKV_CHUNK_MIB", "2")) << 20
        self.live = np.zeros(pool.size + pool.page_size, dtype=np.bool_)
        self.live[0] = True
        self.counts = np.zeros((len(self.live) * slot_bytes + self.chunk_bytes - 1)
                               // self.chunk_bytes, dtype=np.int64)
        self.counts[0] = 1  # permanent dummy/sink slot
        self.counters = dict(alloc_calls=0, free_calls=0, index_d2h_ns=0, hook_ns=0)
        self.peak = 0
        if MODE == "v5":
            self.live_count = 0
            self.backed = sum(arena.backed_bytes for arena in self.arenas)
        self.last_publish = 0.0
        self.stats_path = os.environ.get("EKV_STATS_PATH")
        self.logical_bytes = (pool.size + pool.page_size) * pool.head_num * pool.head_dim * pool.dtype.itemsize * pool.layer_num * 2
        CONTROLLERS.append(self)

    def chunks_for(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        first = ids * self.slot_bytes // self.chunk_bytes
        last = ((ids + 1) * self.slot_bytes - 1) // self.chunk_bytes
        # These test models fit a token entry in at most two chunks.
        if self.slot_bytes > self.chunk_bytes:
            raise ValueError("slot exceeds VMM chunk size")
        return np.concatenate((first, last[last != first]))

    def admit(self, ids):
        if len(ids) == 0:
            return
        if np.any(self.live[ids]) or np.any(ids < 1) or np.any(ids > self.pool.size):
            raise RuntimeError("invalid or already-live token slot")
        chunks = self.chunks_for(ids)
        needed = np.unique(chunks).tolist()
        if MODE == "v2":
            # Growth headroom: group adjacent chunks in pairs (4 MiB).
            limit = len(self.counts)
            needed = sorted({j for c in needed for j in (c // 2 * 2, c // 2 * 2 + 1) if j < limit})
        for arena in self.arenas:
            arena.ensure(needed)
        np.add.at(self.counts, chunks, 1)
        self.live[ids] = True
        self.counters["alloc_calls"] += 1
        self.publish()

    def admit_ranges(self, ranges):
        if not ranges:
            return
        if self.slot_bytes > self.chunk_bytes:
            raise ValueError("slot exceeds VMM chunk size")
        needed = set()
        previous_stop = 1
        for start, stop in ranges:
            if start < previous_stop or stop > len(self.live) or start >= stop or self.live[start:stop].any():
                raise RuntimeError("invalid or already-live token slot")
            previous_stop = stop
            first = start * self.slot_bytes // self.chunk_bytes
            last = (stop * self.slot_bytes - 1) // self.chunk_bytes
            needed.update(range(first, last + 1))
        for arena in self.arenas:
            if not needed.issubset(arena.mapped_chunks):
                arena.ensure(needed)
                self.backed = sum(a.backed_bytes for a in self.arenas)
        for start, stop in ranges:
            self.live[start:stop] = True
            self.live_count += stop - start
        self.counters["alloc_calls"] += 1

    def release_idle_only(self, ids):
        if ids[0] < 1 or ids[-1] > self.pool.size or not self.live[ids].all():
            raise RuntimeError("free of a non-live token slot")
        if (ids[1:] == ids[:-1]).any():
            raise RuntimeError("duplicate token free")
        self.live[ids] = False
        self.live_count -= len(ids)
        self.counters["free_calls"] += 1

    def reclaim_idle(self):
        if self.live_count:
            return
        candidates = [(arena, arena.mapped_chunks - {0}) for arena in self.arenas]
        if any(indices for _, indices in candidates):
            start = time.perf_counter_ns()
            torch.cuda.synchronize()
            self.counters.setdefault("boundary_sync_ns", 0)
            self.counters["boundary_sync_ns"] += time.perf_counter_ns() - start
            for arena, indices in candidates:
                arena.reclaim(indices, synchronize=False)
            self.backed = sum(arena.backed_bytes for arena in self.arenas)

    def release(self, ids):
        if len(ids) == 0:
            return
        if np.any(ids < 1) or np.any(ids > self.pool.size) or not np.all(self.live[ids]):
            raise RuntimeError("free of a non-live token slot")
        if len(np.unique(ids)) != len(ids):
            raise RuntimeError("duplicate token free")
        self.live[ids] = False
        np.add.at(self.counts, self.chunks_for(ids), -1)
        if np.any(self.counts < 0):
            raise RuntimeError("negative VMM chunk reference count")
        self.counters["free_calls"] += 1
        if MODE not in HOST_FREE_MODES or not self.live[1:].any():
            keep = {0, 1} if MODE == "v2" else {0}
            dead = {int(i) for i in np.flatnonzero(self.counts == 0)} - keep
            candidates = [(arena, dead & arena.mapped_chunks) for arena in self.arenas]
            if any(indices for _, indices in candidates):
                start = time.perf_counter_ns()
                torch.cuda.synchronize()
                self.counters.setdefault("boundary_sync_ns", 0)
                self.counters["boundary_sync_ns"] += time.perf_counter_ns() - start
                for arena, indices in candidates:
                    arena.reclaim(indices, synchronize=False)
        self.publish(force=not self.live[1:].any())

    def publish(self, force=False):
        if MODE == "v5":
            backed = self.backed
        else:
            backed = sum(arena.backed_bytes for arena in self.arenas) if self.arenas else self.logical_bytes
        self.peak = max(self.peak, backed)
        now = time.monotonic()
        if not self.stats_path or (not force and now - self.last_publish < .25):
            return
        metrics = {}
        for arena in self.arenas:
            for key, value in arena.metrics.items():
                metrics[key] = metrics.get(key, 0) + value
        result = {**metrics, **self.counters, "mode": MODE, "layout": self.layout,
                  "chunk_bytes": self.chunk_bytes, "logical_capacity_bytes": self.logical_bytes,
                  "backed_bytes": backed, "peak_backed_bytes": self.peak,
                  "live_tokens": self.live_count if MODE == "v5" else int(np.count_nonzero(self.live)) - 1}
        target = Path(self.stats_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(result), encoding="utf-8")
        temp.replace(target)
        self.last_publish = now


def install():
    global _installed
    if _installed:
        return
    if MODE not in VALID_MODES:
        raise ValueError(f"unknown EKV_MODE={MODE}")
    import sglang
    if sglang.__version__ != "0.5.2":
        raise RuntimeError("This adapter is pinned to SGLang 0.5.2")
    import sglang.srt.model_executor.model_runner as runner
    original_pool = runner.MHATokenToKVPool
    original_allocator = runner.TokenToKVPoolAllocator
    original_init = runner.ModelRunner.init_memory_pool

    class ElasticPool(original_pool):
        def _create_buffers(self):
            chunk = int(os.environ.get("EKV_CHUNK_MIB", "2")) << 20
            slots = self.size + self.page_size
            elements = self.head_num * self.head_dim
            itemsize = self.store_dtype.itemsize
            if MODE == "v0":
                arenas = [Backing(slots * elements * itemsize, chunk, self.device)
                          for _ in range(2 * self.layer_num)]
                for arena in arenas:
                    arena.ensure([0])
                views = [arena.raw.view(self.store_dtype).view(slots, self.head_num, self.head_dim)
                         for arena in arenas]
                self.k_buffer, self.v_buffer = views[:self.layer_num], views[self.layer_num:]
                stride = elements * itemsize
                layout = "layer_contiguous"
            elif MODE in ("v3", "v4", "v5"):
                stride = elements * itemsize
                projection = PlaneProjection(slots * stride, 2 * self.layer_num, chunk, self.device)
                projection.ensure([0])
                arenas = [projection]
                views = projection.views(self.store_dtype, (slots, self.head_num, self.head_dim))
                self.k_buffer, self.v_buffer = views[:self.layer_num], views[self.layer_num:]
                layout = "layer_contiguous_shared_arena"
            else:
                stride = elements * itemsize * self.layer_num * 2
                arena = Backing(slots * stride, chunk, self.device)
                arena.ensure([0])
                arenas = [arena]
                packed = arena.raw.view(self.store_dtype).view(slots, self.layer_num, 2, self.head_num, self.head_dim)
                self.k_buffer = [packed[:, layer, 0] for layer in range(self.layer_num)]
                self.v_buffer = [packed[:, layer, 1] for layer in range(self.layer_num)]
                layout = "token_major"
            self.k_data_ptrs = torch.tensor([x.data_ptr() for x in self.k_buffer], dtype=torch.uint64, device=self.device)
            self.v_data_ptrs = torch.tensor([x.data_ptr() for x in self.v_buffer], dtype=torch.uint64, device=self.device)
            self.data_ptrs = torch.cat([self.k_data_ptrs, self.v_data_ptrs])
            self.data_strides = torch.tensor([x.stride(0) * itemsize for x in self.k_buffer + self.v_buffer], device=self.device)
            self.residency = Residency(self, arenas, stride, layout)
            self.residency.publish(force=True)

        def _finalize_allocation_log(self, num_tokens):
            self.mem_usage = sum(a.backed_bytes for a in self.residency.arenas) / (1 << 30)
            logging.getLogger(__name__).info("Elastic KV: logical slots=%s, initial physical GiB=%.4f", num_tokens, self.mem_usage)

        def get_contiguous_buf_infos(self):
            raise RuntimeError("PD/HiCache registration is not supported by this prototype")

    class ObservedAllocator(original_allocator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ctrl = getattr(self._kvcache, "residency", None)
            if self.ctrl is None:
                self.ctrl = Residency(self._kvcache, [], 1, "stock")
                self.ctrl.publish(force=True)
            if MODE in HOST_FREE_MODES:
                self.host_free = np.ones(self.size + 1, dtype=np.bool_)
                self.host_free[0] = False
            if MODE in ("v4", "v5"):
                self.ctrl.counters.update(contiguous_id_view_calls=0, id_upload_calls=0)
                if MODE == "v5":
                    self.ctrl.counters.update(interval_alloc_calls=0, fragmented_alloc_calls=0)
                self.ctrl.publish(force=True)

        def available_size(self):
            if MODE == "v5":
                return self.free_slots.count
            if MODE in HOST_FREE_MODES and hasattr(self, "host_free"):
                return int(np.count_nonzero(self.host_free))
            return super().available_size()

        def clear(self):
            if hasattr(self, "ctrl") and MODE != "baseline":
                ids = np.flatnonzero(self.ctrl.live[1:]) + 1
                if len(ids):
                    if MODE == "v5":
                        self.ctrl.release_idle_only(ids)
                    else:
                        self.ctrl.release(ids)
            super().clear()
            if MODE == "v5":
                self.free_slots = FreeSlots(self.size)
            if MODE in ("v4", "v5"):
                self.gpu_slot_ids = self.free_pages
            if MODE in HOST_FREE_MODES and hasattr(self, "host_free"):
                self.host_free[:] = True
                self.host_free[0] = False

            if MODE == "v5" and hasattr(self, "ctrl"):
                self.ctrl.reclaim_idle()
                self.ctrl.publish(force=True)

        def alloc(self, need_size):
            if MODE == "baseline":
                return super().alloc(need_size)
            start = time.perf_counter_ns()
            if MODE == "v5":
                ranges = self.free_slots.peek(need_size)
                if ranges is None:
                    return None
                if len(ranges) <= 1:
                    start_slot = ranges[0][0] - 1 if ranges else 0
                    result = self.gpu_slot_ids[start_slot:start_slot + need_size]
                else:
                    ids = np.concatenate([np.arange(lo, hi, dtype=np.int64) for lo, hi in ranges])
                    result = torch.from_numpy(ids).to(self.device)
                self.ctrl.admit_ranges(ranges)
                self.free_slots.claim(ranges)
                self.ctrl.counters["interval_alloc_calls"] += 1
                if len(ranges) <= 1:
                    self.ctrl.counters["contiguous_id_view_calls"] += 1
                else:
                    self.ctrl.counters["id_upload_calls"] += 1
                    self.ctrl.counters["fragmented_alloc_calls"] += 1
                self.ctrl.counters["hook_ns"] += time.perf_counter_ns() - start
                self.ctrl.publish()
                return result
            if MODE in HOST_FREE_MODES:
                ids = np.flatnonzero(self.host_free)[:need_size]
                if len(ids) < need_size:
                    return None
                self.ctrl.admit(ids)
                self.host_free[ids] = False
                if MODE == "v4" and (len(ids) == 0 or ids[-1] - ids[0] + 1 == len(ids)):
                    start_slot = int(ids[0]) - 1 if len(ids) else 0
                    result = self.gpu_slot_ids[start_slot:start_slot + need_size]
                    self.ctrl.counters["contiguous_id_view_calls"] += 1
                else:
                    result = torch.from_numpy(ids.astype(np.int64, copy=False)).to(self.device)
                    if MODE == "v4":
                        self.ctrl.counters["id_upload_calls"] += 1
            else:
                if self.need_sort and need_size > len(self.free_pages):
                    self.merge_and_sort_free()
                if need_size > len(self.free_pages):
                    return None
                result = self.free_pages[:need_size]
                copy_start = time.perf_counter_ns()
                ids = result.cpu().numpy()
                self.ctrl.counters["index_d2h_ns"] += time.perf_counter_ns() - copy_start
                self.ctrl.admit(ids)
                self.free_pages = self.free_pages[need_size:]
            self.ctrl.counters["hook_ns"] += time.perf_counter_ns() - start
            return result

        def free(self, free_index):
            if MODE == "baseline" or not free_index.numel():
                return super().free(free_index)
            if not self.is_not_in_free_group:
                self.free_group.append(free_index)
                return
            start = time.perf_counter_ns()
            copy_start = time.perf_counter_ns()
            ids = free_index.cpu().numpy()
            self.ctrl.counters["index_d2h_ns"] += time.perf_counter_ns() - copy_start
            if MODE == "v5":
                ids = np.sort(ids)
                self.ctrl.release_idle_only(ids)
                self.free_slots.release(ids)
                self.ctrl.reclaim_idle()
                self.ctrl.counters["hook_ns"] += time.perf_counter_ns() - start
                self.ctrl.publish(force=not self.ctrl.live_count)
                return
            self.ctrl.release(ids)
            if MODE in HOST_FREE_MODES:
                self.host_free[ids] = True
            else:
                super().free(free_index)
            self.ctrl.counters["hook_ns"] += time.perf_counter_ns() - start

        def backup_state(self):
            if MODE != "baseline":
                raise RuntimeError("allocator snapshots/speculative decoding are not supported")
            return super().backup_state()

        def restore_state(self, state):
            if MODE != "baseline":
                raise RuntimeError("allocator snapshots/speculative decoding are not supported")
            return super().restore_state(state)

    def guarded_init(self, *args, **kwargs):
        a = self.server_args
        graph_safe = a.disable_cuda_graph or (
            MODE in ("baseline", "v4", "v5") and a.disable_cuda_graph_padding
        )
        required = (a.page_size == 1, a.attention_backend == "triton", graph_safe,
                    a.disable_overlap_schedule, a.disable_radix_cache, a.tp_size == 1,
                    a.pp_size == 1, a.dp_size == 1, not a.enable_two_batch_overlap,
                    a.prefill_attention_backend in (None, "triton"),
                    a.decode_attention_backend in (None, "triton"),
                    a.disaggregation_mode == "null", not a.enable_memory_saver,
                    not a.speculative_algorithm, not a.enable_hierarchical_cache)
        if not all(required) or self.is_hybrid or self.is_hybrid_gdn or self.use_mla_backend:
            raise RuntimeError("Require dense MHA, TP1, Triton, eager or exact-batch Graph, no overlap/cache/spec/PD")
        original_init(self, *args, **kwargs)

    runner.MHATokenToKVPool = original_pool if MODE == "baseline" else ElasticPool
    runner.TokenToKVPoolAllocator = ObservedAllocator
    runner.ModelRunner.init_memory_pool = guarded_init
    from .graph_support import install_graph_hooks
    install_graph_hooks()
    _installed = True
