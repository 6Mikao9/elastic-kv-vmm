"""Opt-in dense-MHA adapter; no attention kernel or model weights are changed.

Supports the SGLang 0.5.2 compatibility surface and the newer memory-pool API
where the pool/allocator live below ``sglang.srt.mem_cache`` and pool creation is
owned by ``KVCacheConfigurator``.  Scope remains TP=1, page_size=1, Triton
attention, dense MHA, no overlap/radix/speculation/PD.
"""
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import activation
from .backing import Backing
from .slots import FreeSlots

MODE = os.environ.get("EKV_MODE", "baseline")
ALLOW_OVERLAP_SCHEDULE = os.environ.get(
    "EKV_ALLOW_OVERLAP_SCHEDULE", "0"
).lower() in ("1", "true", "yes", "on")
OVERLAP_SCHEDULE_ACTIVE = False
VALID_MODES = ("baseline", "v0", "v1", "v2", "v3", "v4", "v5")
HOST_FREE_MODES = ("v2", "v3", "v4")
PREDICTIVE_PREWARM_MODES = ("off", "dry_run", "enforce")
CONTROLLERS = []
_installed = False


def _free_group_active(allocator):
    """Normalize the old boolean and new ``None/list`` free-group APIs."""
    if hasattr(allocator, "is_not_in_free_group"):
        return not allocator.is_not_in_free_group
    if hasattr(allocator, "free_group"):
        return allocator.free_group is not None
    raise AttributeError("allocator has no free-group state")


def _queue_free_group(allocator, free_index):
    """Defer a free using the allocator's owned-copy convention when present."""
    copy = getattr(allocator, "_copy_for_free_group", None)
    allocator.free_group.append(copy(free_index) if copy else free_index)


def _replace_if_present(module, name, replacement):
    if hasattr(module, name):
        setattr(module, name, replacement)


def _overlap_schedule_supported(server_args, *, mode, allow):
    """Permit the V5 lifecycle's explicit reclaim fence to back overlap opt-in."""

    return bool(
        server_args.disable_overlap_schedule
        or (mode == "v5" and allow and not server_args.enable_two_batch_overlap)
    )


class PlaneProjection:
    """Project logical chunks onto equally padded contiguous K/V planes."""

    def __init__(self, plane_bytes, planes, chunk_bytes, device):
        self.plane_chunks = (plane_bytes + chunk_bytes - 1) // chunk_bytes
        self.plane_stride = self.plane_chunks * chunk_bytes
        self.plane_bytes = plane_bytes
        self.planes = planes
        self.backing = Backing(self.plane_stride * planes, chunk_bytes, device)
        self.mapped_chunks = set()
        self.precommitted_chunks = set()

    def _physical_chunks(self, indices):
        return [plane * self.plane_chunks + index
                for plane in range(self.planes) for index in indices]

    def ensure(self, indices):
        pending = set(indices) - self.mapped_chunks
        if pending or self.precommitted_chunks.intersection(indices):
            self.backing.ensure(self._physical_chunks(indices))
            self.mapped_chunks.update(indices)
            self.precommitted_chunks.difference_update(indices)

    def precommit(self, indices):
        pending = set(indices) - self.mapped_chunks - self.precommitted_chunks
        if pending:
            self.backing.precommit(self._physical_chunks(pending))
            self.precommitted_chunks.update(pending)
            return len(pending)
        return 0

    def reclaim(self, indices, *, synchronize=True):
        pending = (self.mapped_chunks | self.precommitted_chunks).intersection(indices)
        if pending:
            self.backing.reclaim(self._physical_chunks(pending), synchronize=synchronize)
            self.mapped_chunks.difference_update(pending)
            self.precommitted_chunks.difference_update(pending)

    @property
    def allocated_chunks(self):
        return self.mapped_chunks | self.precommitted_chunks

    @property
    def backed_bytes(self):
        return self.backing.backed_bytes

    @property
    def metrics(self):
        return {
            **self.backing.metrics,
            "precommitted_chunks": len(self.precommitted_chunks),
        }

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
        self.retain_chunks = int(os.environ.get("EKV_RETAIN_CHUNKS", "1"))
        if self.retain_chunks < 1:
            raise ValueError("EKV_RETAIN_CHUNKS must retain at least the sink chunk")
        self.precommit_lookahead = int(os.environ.get("EKV_PRECOMMIT_CHUNKS", "0"))
        if not 0 <= self.precommit_lookahead <= 4:
            raise ValueError("EKV_PRECOMMIT_CHUNKS must be in [0, 4]")
        self.precommit_limit = int(
            os.environ.get(
                "EKV_PRECOMMIT_MAX_CHUNKS", str(max(1, self.precommit_lookahead))
            )
        )
        if not 0 <= self.precommit_limit <= 16:
            raise ValueError("EKV_PRECOMMIT_MAX_CHUNKS must be in [0, 16]")
        self.predictive_prewarm_mode = os.environ.get(
            "EKV_PREDICTIVE_PREWARM_MODE", "off"
        ).strip().lower()
        if self.predictive_prewarm_mode not in PREDICTIVE_PREWARM_MODES:
            raise ValueError(
                "EKV_PREDICTIVE_PREWARM_MODE must be off, dry_run or enforce"
            )
        if self.predictive_prewarm_mode != "off" and MODE != "v5":
            raise ValueError("predictive prewarm requires EKV_MODE=v5")
        self.predictive_prewarm_ttl_ms = int(os.environ.get(
            "EKV_PREDICTIVE_PREWARM_TTL_MS", "2000"
        ))
        self.predictive_prewarm_max_chunks = int(os.environ.get(
            "EKV_PREDICTIVE_PREWARM_MAX_CHUNKS", "8"
        ))
        if self.predictive_prewarm_ttl_ms < 0:
            raise ValueError(
                "EKV_PREDICTIVE_PREWARM_TTL_MS must be nonnegative"
            )
        if not 0 <= self.predictive_prewarm_max_chunks <= 64:
            raise ValueError(
                "EKV_PREDICTIVE_PREWARM_MAX_CHUNKS must be in [0, 64]"
            )
        if self.predictive_prewarm_mode != "off" and (
            self.predictive_prewarm_ttl_ms < 1
            or self.predictive_prewarm_max_chunks < 1
        ):
            raise ValueError(
                "predictive prewarm requires positive TTL and max chunks"
            )
        self.admission_policy = os.environ.get(
            "EKV_ADMISSION_POLICY", "logical"
        )
        if self.admission_policy not in ("logical", "physical"):
            raise ValueError("EKV_ADMISSION_POLICY must be logical or physical")
        if self.admission_policy != "logical" and MODE != "v5":
            raise ValueError("physical admission requires EKV_MODE=v5")
        self.prefill_headroom_bytes = int(
            os.environ.get("EKV_PREFILL_HEADROOM_MIB", "0")
        ) << 20
        if self.prefill_headroom_bytes < 0:
            raise ValueError("EKV_PREFILL_HEADROOM_MIB must be nonnegative")
        self.precommitted = set()
        self._predictive_precommitted = set()
        self._predictive_prewarm_deadlines = {}
        self.live = np.zeros(pool.size + pool.page_size, dtype=np.bool_)
        self.live[0] = True
        self.counts = np.zeros((len(self.live) * slot_bytes + self.chunk_bytes - 1)
                               // self.chunk_bytes, dtype=np.int64)
        self.counts[0] = 1  # permanent dummy/sink slot
        self.counters = dict(
            alloc_calls=0,
            free_calls=0,
            index_d2h_ns=0,
            hook_ns=0,
            predictive_prewarm_ticks=0,
            predictive_prewarm_demand_bytes=0,
            predictive_prewarm_candidate_chunks=0,
            predictive_prewarm_mapped_chunks=0,
            predictive_prewarm_dry_run_chunks=0,
            predictive_prewarm_hit_chunks=0,
            predictive_prewarm_hit_bytes=0,
            predictive_prewarm_expired_chunks=0,
            predictive_prewarm_expired_bytes=0,
            predictive_prewarm_active_chunks=0,
        )
        self.counters["overlap_schedule_enabled"] = int(
            OVERLAP_SCHEDULE_ACTIVE
        )
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

    def _predictive_prewarm_chunk_bytes(self):
        return self.chunk_bytes * sum(
            getattr(arena, "planes", 1) for arena in self.arenas
        )

    def _record_predictive_prewarm_hits(self, consumed):
        predictive = getattr(self, "_predictive_precommitted", set())
        hits = set(consumed) & predictive
        if not hits:
            return
        predictive.difference_update(hits)
        for chunk in hits:
            self._predictive_prewarm_deadlines.pop(chunk, None)
        self.counters["predictive_prewarm_hit_chunks"] += len(hits)
        self.counters["predictive_prewarm_hit_bytes"] += (
            len(hits) * self._predictive_prewarm_chunk_bytes()
        )
        self.counters["predictive_prewarm_active_chunks"] = len(
            predictive
        )

    def expire_predictive_prewarm(self, *, now_s=None):
        """Reclaim idle predictive mappings whose bounded TTL has elapsed."""

        predictive = getattr(self, "_predictive_precommitted", set())
        if (
            getattr(self, "predictive_prewarm_mode", "off") != "enforce"
            or not predictive
        ):
            return 0
        now = time.monotonic() if now_s is None else float(now_s)
        expired = {
            chunk
            for chunk in predictive
            if self._predictive_prewarm_deadlines.get(chunk, now + 1) <= now
            and self.counts[chunk] == 0
        }
        if not expired:
            return 0
        start = time.perf_counter_ns()
        torch.cuda.synchronize(self.pool.device)
        for arena in self.arenas:
            indices = expired & arena.allocated_chunks
            if indices:
                arena.reclaim(indices, synchronize=False)
        self.precommitted.difference_update(expired)
        predictive.difference_update(expired)
        for chunk in expired:
            self._predictive_prewarm_deadlines.pop(chunk, None)
        self.counters.setdefault("predictive_prewarm_expire_ns", 0)
        self.counters["predictive_prewarm_expire_ns"] += (
            time.perf_counter_ns() - start
        )
        self.counters["predictive_prewarm_expired_chunks"] += len(expired)
        self.counters["predictive_prewarm_expired_bytes"] += (
            len(expired) * self._predictive_prewarm_chunk_bytes()
        )
        self.counters["predictive_prewarm_active_chunks"] = len(
            predictive
        )
        self.backed = sum(arena.backed_bytes for arena in self.arenas)
        self.publish(force=True)
        return len(expired)

    def predictive_prewarm(self, demand_bytes, *, now_s=None):
        """Pre-map a bounded prefix of currently free slots for future prefill.

        This method only changes physical backing for slots that SGLang already
        exposes as free. It never claims a logical slot, changes queue order, or
        acts as a replacement for ``PrefillAdder``. New leases use a fixed TTL;
        repeated plans never extend an existing lease.
        """

        if (
            isinstance(demand_bytes, bool)
            or not isinstance(demand_bytes, int)
            or demand_bytes < 0
        ):
            raise ValueError("predictive prewarm demand must be nonnegative")
        demand = demand_bytes
        now = time.monotonic() if now_s is None else float(now_s)
        if not math.isfinite(now):
            raise ValueError("predictive prewarm time must be finite")
        self.counters["predictive_prewarm_ticks"] += 1
        self.counters["predictive_prewarm_demand_bytes"] = demand
        if self.predictive_prewarm_mode == "off" or demand == 0:
            return 0
        if self.predictive_prewarm_mode == "enforce":
            self.expire_predictive_prewarm(now_s=now)

        # The cap is a global outstanding-lease budget. Repeated plans must not
        # walk past existing speculative mappings and accumulate another full
        # per-call allowance on every scheduler tick.
        candidate_capacity = (
            self.predictive_prewarm_max_chunks
            - len(self._predictive_precommitted)
        )
        if candidate_capacity <= 0:
            self.counters["predictive_prewarm_candidate_chunks"] = 0
            self.counters["predictive_prewarm_active_chunks"] = len(
                self._predictive_precommitted
            )
            return 0

        max_bytes = candidate_capacity * self.chunk_bytes
        target_bytes = min(demand, max_bytes)
        target_slots = min(
            self.free_slots.count,
            max(1, (target_bytes + self.slot_bytes - 1) // self.slot_bytes),
        )
        ranges = self.free_slots.peek(target_slots)
        if not ranges:
            return 0
        arena_chunks = [set(arena.allocated_chunks) for arena in self.arenas]
        fully_allocated = (
            set.intersection(*arena_chunks) if arena_chunks else set()
        )
        candidates = []
        seen = set()
        for start, stop in ranges:
            for slot in range(start, stop):
                for chunk in self.chunks_for(np.asarray([slot], dtype=np.int64)):
                    chunk = int(chunk)
                    if (
                        chunk < 1
                        or chunk in seen
                        or self.counts[chunk] != 0
                        or chunk in self.precommitted
                        or chunk in fully_allocated
                    ):
                        continue
                    seen.add(chunk)
                    candidates.append(chunk)
                    if len(candidates) >= candidate_capacity:
                        break
                if len(candidates) >= candidate_capacity:
                    break
            if len(candidates) >= candidate_capacity:
                break
        self.counters["predictive_prewarm_candidate_chunks"] = len(candidates)
        if not candidates:
            return 0
        if self.predictive_prewarm_mode == "dry_run":
            self.counters["predictive_prewarm_dry_run_chunks"] += len(candidates)
            self.publish()
            return len(candidates)

        for arena in self.arenas:
            missing = set(candidates) - arena.allocated_chunks
            if missing:
                arena.precommit(missing)
        self.precommitted.update(candidates)
        self._predictive_precommitted.update(candidates)
        deadline = now + self.predictive_prewarm_ttl_ms / 1000.0
        for chunk in candidates:
            self._predictive_prewarm_deadlines[chunk] = deadline
        self.counters["predictive_prewarm_mapped_chunks"] += len(candidates)
        self.counters["predictive_prewarm_active_chunks"] = len(
            self._predictive_precommitted
        )
        self.backed = sum(arena.backed_bytes for arena in self.arenas)
        self.publish()
        return len(candidates)

    def physical_available_tokens(self, logical_free_tokens, free_ranges):
        """Bound scheduler admission by the bytes VMM can actually commit.

        The scheduler allocates slots in ``free_ranges`` order. Slots whose
        chunks are already mapped consume no new backing and are credited;
        only the first touch of another chunk spends physical budget. The
        configured prefill headroom is charged here, before
        ``prepare_for_extend`` allocates activations, so KV backing and the
        prefill working set share one device budget at admission time.
        """

        if logical_free_tokens < 0:
            raise ValueError("logical_free_tokens must be nonnegative")
        if self.admission_policy == "logical":
            return logical_free_tokens
        if logical_free_tokens == 0:
            return 0
        if self.slot_bytes <= 0 or self.slot_bytes > self.chunk_bytes:
            raise ValueError("physical admission requires 0 < slot_bytes <= chunk_bytes")
        ranges = tuple(free_ranges)
        if any(start < 1 or stop <= start for start, stop in ranges):
            raise ValueError("free_ranges must contain positive half-open ranges")
        if sum(stop - start for start, stop in ranges) != logical_free_tokens:
            raise ValueError("free_ranges length must match logical_free_tokens")
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
            raise ValueError("free_ranges must be sorted and disjoint")

        chunk_cost = sum(
            self.chunk_bytes * getattr(arena, "planes", 1)
            for arena in self.arenas
        )
        if chunk_cost <= 0:
            return logical_free_tokens

        self.counters["physical_available_calls"] = (
            self.counters.get("physical_available_calls", 0) + 1
        )
        start = time.perf_counter_ns()
        free, _ = torch.cuda.mem_get_info(self.pool.device)
        prefill_headroom = activation.current_prefill_headroom(
            self.prefill_headroom_bytes
        )
        required_before_commit = prefill_headroom + chunk_cost
        if free < required_before_commit:
            start = time.perf_counter_ns()
            with torch.cuda.device(self.pool.device):
                torch.cuda.empty_cache()
            after, _ = torch.cuda.mem_get_info(self.pool.device)
            self.counters["physical_empty_cache_calls"] = (
                self.counters.get("physical_empty_cache_calls", 0) + 1
            )
            self.counters["physical_empty_cache_freed_bytes"] = (
                self.counters.get("physical_empty_cache_freed_bytes", 0)
                + max(0, after - free)
            )
            self.counters["physical_empty_cache_ns"] = (
                self.counters.get("physical_empty_cache_ns", 0)
                + time.perf_counter_ns()
                - start
            )
            free = after

        usable = max(0, free - prefill_headroom)
        broker_available = []
        broker_snapshots = []
        for arena in self.arenas:
            backing = getattr(arena, "backing", None)
            snapshot = getattr(backing, "broker_snapshot", None)
            if callable(snapshot):
                snapshot = snapshot()
            if isinstance(snapshot, dict):
                broker_snapshots.append(snapshot)
                value = snapshot.get("available_bytes")
            else:
                value = getattr(backing, "broker_available_bytes", None)
            if value is not None:
                broker_available.append(value)
        if broker_available:
            global_available = min(broker_available)
            self.counters["physical_global_broker_available_bytes"] = (
                global_available
            )
            if global_available < usable:
                self.counters["physical_global_broker_limited_calls"] = (
                    self.counters.get("physical_global_broker_limited_calls", 0) + 1
                )
            usable = min(usable, global_available)
        fair_available = []
        for snapshot in broker_snapshots:
            if snapshot.get("fair_mode") != "enforce":
                continue
            if snapshot.get("fairness_pressure"):
                fair_available.append(0)
        if fair_available:
            tenant_available = min(fair_available)
            self.counters["physical_global_broker_fair_available_bytes"] = (
                tenant_available
            )
            if tenant_available < usable:
                self.counters["physical_global_broker_fair_limited_calls"] = (
                    self.counters.get(
                        "physical_global_broker_fair_limited_calls", 0
                    )
                    + 1
                )
            usable = min(usable, tenant_available)
        chunk_budget = usable // chunk_cost
        backed_chunks = set(self.precommitted)
        if self.arenas:
            backed_chunks.update(
                set.intersection(*(set(arena.mapped_chunks) for arena in self.arenas))
            )
        admitted = 0
        for start, stop in ranges:
            remaining = logical_free_tokens - admitted
            if remaining <= 0:
                break
            stop = min(stop, start + remaining)
            position = start
            while position < stop:
                first = position * self.slot_bytes // self.chunk_bytes
                last = ((position + 1) * self.slot_bytes - 1) // self.chunk_bytes
                missing = {
                    chunk
                    for chunk in range(first, last + 1)
                    if chunk not in backed_chunks
                }
                if len(missing) > chunk_budget:
                    break
                chunk_budget -= len(missing)
                backed_chunks.update(missing)
                # Every slot before next_position uses only chunks through
                # ``last`` and is therefore fully backed by this decision.
                next_position = max(
                    position + 1,
                    ((last + 1) * self.chunk_bytes) // self.slot_bytes,
                )
                segment_end = min(stop, next_position)
                admitted += segment_end - position
                position = segment_end
            if position < stop:
                break
        self.counters["physical_configured_headroom_bytes"] = (
            self.prefill_headroom_bytes
        )
        self.counters["physical_headroom_bytes"] = prefill_headroom
        self.counters["physical_chunk_cost_bytes"] = chunk_cost
        self.counters["physical_available_tokens_min"] = min(
            self.counters.get("physical_available_tokens_min", admitted), admitted
        )
        self.counters["physical_available_tokens_last"] = admitted
        if admitted < logical_free_tokens:
            self.counters["physical_limited_calls"] = (
                self.counters.get("physical_limited_calls", 0) + 1
            )
        self.counters["physical_available_ns"] = (
            self.counters.get("physical_available_ns", 0)
            + time.perf_counter_ns()
            - start
        )
        return admitted

    def admit(self, ids):
        if len(ids) == 0:
            return
        self.expire_predictive_prewarm()
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
        self._record_predictive_prewarm_hits(needed)
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
        self.expire_predictive_prewarm()
        activation.before_admit(self, ranges)
        for arena in self.arenas:
            if not needed.issubset(arena.mapped_chunks):
                arena.ensure(needed)
                self.backed = sum(a.backed_bytes for a in self.arenas)
        self._record_predictive_prewarm_hits(needed)
        self.precommitted.difference_update(needed)
        self._precommit_after(ranges)
        for start, stop in ranges:
            self.live[start:stop] = True
            self.live_count += stop - start
        self.counters["alloc_calls"] += 1

    def _precommit_after(self, ranges):
        if not self.precommit_lookahead:
            return
        candidates = set()
        chunk_count = len(self.counts)
        for _, stop in ranges:
            first = stop * self.slot_bytes // self.chunk_bytes
            for chunk in range(first, min(chunk_count, first + self.precommit_lookahead)):
                if self.counts[chunk] == 0 and chunk not in self.precommitted:
                    candidates.add(chunk)
        added_count = 0
        for chunk in sorted(candidates):
            if len(self.precommitted) >= self.precommit_limit:
                break
            added = sum(arena.precommit([chunk]) for arena in self.arenas)
            if added:
                self.precommitted.add(chunk)
                added_count += 1
        if added_count:
            self.counters["precommit_batches"] = (
                self.counters.get("precommit_batches", 0) + 1
            )
            self.counters["precommit_chunks"] = (
                self.counters.get("precommit_chunks", 0) + added_count
            )
            self.backed = sum(a.backed_bytes for a in self.arenas)

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
        keep_prewarm = set(getattr(self, "_predictive_precommitted", set()))
        candidates = [
            (
                arena,
                {
                    i
                    for i in arena.allocated_chunks
                    if i >= self.retain_chunks and i not in keep_prewarm
                },
            )
            for arena in self.arenas
        ]
        if any(indices for _, indices in candidates):
            start = time.perf_counter_ns()
            torch.cuda.synchronize()
            self.counters.setdefault("boundary_sync_ns", 0)
            self.counters["boundary_sync_ns"] += time.perf_counter_ns() - start
            for arena, indices in candidates:
                arena.reclaim(indices, synchronize=False)
            self.backed = sum(arena.backed_bytes for arena in self.arenas)
            reclaimed = set().union(*(indices for _, indices in candidates))
            self.precommitted.difference_update(reclaimed)

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
            dead.difference_update(
                getattr(self, "_predictive_precommitted", set())
            )
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
                  "admission_policy": self.admission_policy,
                  "chunk_bytes": self.chunk_bytes, "logical_capacity_bytes": self.logical_bytes,
                  "retain_chunks": self.retain_chunks if MODE == "v5" else None,
                  "precommit_lookahead": self.precommit_lookahead,
                  "precommit_limit": self.precommit_limit,
                  "precommitted_chunks": len(self.precommitted),
                  "predictive_prewarm_mode": self.predictive_prewarm_mode,
                  "predictive_prewarm_ttl_ms": self.predictive_prewarm_ttl_ms,
                  "predictive_prewarm_limit_chunks": self.predictive_prewarm_max_chunks,
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
    import sglang.srt.model_executor.model_runner as runner

    # The current SGLang layout imports these classes in mem_cache and builds
    # pools through KVCacheConfigurator.  Resolve by capability instead of an
    # exact package version so development builds and future releases with the
    # same API can use the adapter.
    modern = not hasattr(runner.ModelRunner, "init_memory_pool")
    configurator_module = None
    if modern:
        try:
            from sglang.srt.mem_cache import (
                kv_cache_configurator as configurator_module,
            )
            from sglang.srt.mem_cache import memory_pool
            from sglang.srt.mem_cache.allocator import token as token_allocator_module
        except ImportError as exc:
            raise RuntimeError(
                "Unsupported SGLang memory-pool layout: expected either the "
                "0.5.2 model_runner symbols or the newer KVCacheConfigurator API"
            ) from exc
        if not hasattr(memory_pool, "MHATokenToKVPool") or not hasattr(
            token_allocator_module, "TokenToKVPoolAllocator"
        ):
            raise RuntimeError(
                "Unsupported newer SGLang API: MHA pool or token allocator is missing"
            )
        original_pool = memory_pool.MHATokenToKVPool
        original_allocator = token_allocator_module.TokenToKVPoolAllocator
        original_init = None
    else:
        if not hasattr(runner, "MHATokenToKVPool") or not hasattr(
            runner, "TokenToKVPoolAllocator"
        ):
            raise RuntimeError(
                "Unsupported SGLang 0.5.2 API: model_runner memory-pool symbols are missing"
            )
        original_pool = runner.MHATokenToKVPool
        original_allocator = runner.TokenToKVPoolAllocator
        original_init = runner.ModelRunner.init_memory_pool

    class ElasticPool(original_pool):
        def _create_buffers(self):
            # VMM planes below expose NHD slot rows. Newer SGLang revisions can
            # enable HND from an environment knob that is not represented in
            # ServerArgs, so pin the buffer metadata to the layout we build.
            self.use_hnd = False
            self.kv_cache_layout = "nhd"
            self.k_scale_buffer = None
            self.v_scale_buffer = None
            self.dq_k_buffer = None
            self.dq_v_buffer = None
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
            if hasattr(self, "_build_kv_buffer_descs"):
                self._kv_buffer_descs = self._build_kv_buffer_descs()
                self._init_data_ptrs_and_strides()
            else:
                self.k_data_ptrs = torch.tensor(
                    [x.data_ptr() for x in self.k_buffer],
                    dtype=torch.uint64,
                    device=self.device,
                )
                self.v_data_ptrs = torch.tensor(
                    [x.data_ptr() for x in self.v_buffer],
                    dtype=torch.uint64,
                    device=self.device,
                )
                self.data_ptrs = torch.cat(
                    [self.k_data_ptrs, self.v_data_ptrs]
                )
                self.data_strides = torch.tensor(
                    [x.stride(0) * itemsize for x in self.k_buffer + self.v_buffer],
                    device=self.device,
                )
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
                return self.ctrl.physical_available_tokens(
                    self.free_slots.count, self.free_slots.ranges
                )
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
            if MODE == "baseline":
                result = super().free(free_index)
                if not _free_group_active(self):
                    self.ctrl.publish(force=True)
                return result
            if not free_index.numel():
                return super().free(free_index)
            if _free_group_active(self):
                _queue_free_group(self, free_index)
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

    def validate_server_args(a, *, is_hybrid=False, use_mla_backend=False):
        global OVERLAP_SCHEDULE_ACTIVE
        graph_safe = a.disable_cuda_graph or MODE in ("baseline", "v4", "v5")
        backends = (a.attention_backend, a.prefill_attention_backend,
                    a.decode_attention_backend)
        required = (a.page_size == 1, a.attention_backend == "triton", graph_safe,
                    _overlap_schedule_supported(
                        a, mode=MODE, allow=ALLOW_OVERLAP_SCHEDULE
                    ),
                    a.disable_radix_cache, a.tp_size == 1,
                    a.pp_size == 1, a.dp_size == 1, not a.enable_two_batch_overlap,
                    all(backend in (None, "triton") for backend in backends),
                    a.disaggregation_mode == "null", not a.enable_memory_saver,
                    not a.speculative_algorithm, not a.enable_hierarchical_cache,
                    not getattr(a, "enable_page_major_kv_layout", False),
                    getattr(a, "dcp_size", 1) == 1)
        if not all(required) or is_hybrid or use_mla_backend:
            raise RuntimeError(
                "Require dense MHA, TP1, Triton, supported Graph/overlap mode, "
                "no radix/cache/spec/PD"
            )
        OVERLAP_SCHEDULE_ACTIVE = not a.disable_overlap_schedule

    def validate_mha_geometry(model_config, extra_pairs=()):
        head_dim = getattr(model_config, "head_dim", None)
        v_head_dim = getattr(model_config, "v_head_dim", head_dim)
        if head_dim is not None and v_head_dim != head_dim:
            raise RuntimeError(
                "Elastic KV requires symmetric K/V head dimensions in this prototype"
            )
        for extra_head, extra_value_head in extra_pairs:
            if extra_head is not None and extra_value_head != extra_head:
                raise RuntimeError(
                    "Elastic KV requires symmetric K/V head dimensions in this prototype"
                )

    def is_minimax_sparse_model(configurator):
        model_config = getattr(configurator, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        checker = getattr(configurator_module, "is_minimax_sparse", None)
        if checker is None and hf_config is not None:
            try:
                from sglang.srt.configs.model_config import is_minimax_sparse
            except (ImportError, AttributeError):
                checker = None
            else:
                checker = is_minimax_sparse
        if checker is not None and hf_config is not None:
            try:
                if checker(hf_config):
                    return True
            except (AttributeError, TypeError, ValueError):
                pass
        if hf_config is None:
            return False
        model_type = str(getattr(hf_config, "model_type", "")).lower()
        architectures = " ".join(
            str(value) for value in (getattr(hf_config, "architectures", None) or ())
        ).lower()
        return "minimax" in model_type or "minimax" in architectures

    def require_elastic_components(pool=None, allocator=None, *, owner=None):
        if MODE == "baseline":
            return
        if pool is None and owner is not None:
            pool = getattr(owner, "token_to_kv_pool", None)
        if allocator is None and owner is not None:
            allocator = getattr(owner, "token_to_kv_pool_allocator", None)
        if not isinstance(pool, ElasticPool):
            raise RuntimeError(
                "Elastic KV requested a non-elastic token KV pool; refusing a silent stock fallback"
            )
        if not isinstance(allocator, ObservedAllocator):
            raise RuntimeError(
                "Elastic KV token KV allocator is not observed; refusing a silent stock fallback"
            )

    def guarded_init(self, *args, **kwargs):
        validate_server_args(
            self.server_args,
            is_hybrid=self.is_hybrid or self.is_hybrid_gdn,
            use_mla_backend=self.use_mla_backend,
        )
        validate_mha_geometry(self.model_config)
        original_init(self, *args, **kwargs)
        require_elastic_components(owner=self)

    if modern:
        original_configure = configurator_module.KVCacheConfigurator.configure

        def guarded_configure(self, *args, **kwargs):
            if MODE != "baseline" and (is_minimax_sparse_model(self) or bool(
                getattr(self.server_args, "enable_hisparse", False)
                or getattr(getattr(self, "memory_pool_config", None), "enable_hisparse", False)
            )):
                raise RuntimeError(
                    "Elastic KV does not support MiniMax sparse or HiSparse pools; use the pinned MiniMax adapter"
                )
            validate_server_args(
                self.server_args,
                is_hybrid=(
                    getattr(self, "is_hybrid_swa", False)
                    or getattr(self, "is_hybrid_swa_compress", False)
                    or getattr(self, "mambaish_config", None) is not None
                    or getattr(self, "hybrid_gdn_config", None) is not None
                    or getattr(self, "hybrid_kda_config", None) is not None
                ),
                use_mla_backend=self.use_mla_backend,
            )
            validate_mha_geometry(
                self.model_config,
                extra_pairs=(
                    (
                        getattr(self, "swa_head_dim", None),
                        getattr(self, "swa_v_head_dim", getattr(self, "swa_head_dim", None)),
                    ),
                ),
            )
            bf16 = getattr(torch, "bfloat16", None)
            if bf16 is not None and getattr(self, "kv_cache_dtype", None) != bf16:
                raise RuntimeError("Elastic KV currently requires BF16 KV cache")
            if getattr(self, "is_draft_worker", False):
                raise RuntimeError("Elastic KV does not support draft workers")
            if getattr(self, "kv_shard_rank", None) is not None:
                raise RuntimeError("Elastic KV does not support KV sharding")
            if getattr(self, "post_capture_kv_active", False):
                raise RuntimeError(
                    "Elastic KV cannot be combined with SGLang post-capture KV sizing"
                )
            result = original_configure(self, *args, **kwargs)
            require_elastic_components(
                getattr(result, "token_to_kv_pool", None),
                getattr(result, "token_to_kv_pool_allocator", None),
            )
            return result

        pool_type = original_pool if MODE == "baseline" else ElasticPool
        _replace_if_present(memory_pool, "MHATokenToKVPool", pool_type)
        _replace_if_present(
            configurator_module, "MHATokenToKVPool", pool_type
        )
        if hasattr(configurator_module, "_KV_POOL_BASE_FOR_KIND"):
            configurator_module._KV_POOL_BASE_FOR_KIND["mha"] = pool_type

        import sglang.srt.mem_cache.allocator as allocator_package
        _replace_if_present(
            token_allocator_module, "TokenToKVPoolAllocator", ObservedAllocator
        )
        _replace_if_present(
            allocator_package, "TokenToKVPoolAllocator", ObservedAllocator
        )
        _replace_if_present(
            configurator_module, "TokenToKVPoolAllocator", ObservedAllocator
        )
        configurator_module.KVCacheConfigurator.configure = guarded_configure
    else:
        runner.MHATokenToKVPool = original_pool if MODE == "baseline" else ElasticPool
        runner.TokenToKVPoolAllocator = ObservedAllocator
        runner.ModelRunner.init_memory_pool = guarded_init
    from .graph_support import install_graph_hooks
    install_graph_hooks()
    activation_requested = (
        os.environ.get("EKV_ACTIVATION_POLICY", "off") != "off"
        or os.environ.get("EKV_GLOBAL_BROKER_PREFILL_MODE", "off") != "off"
    )
    if MODE != "v5" and activation_requested:
        raise RuntimeError("Activation and prefill-claim policies currently require V5")
    activation.install_activation_hooks()
    _installed = True
