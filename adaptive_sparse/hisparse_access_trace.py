"""Bounded eager-only diagnostics for native MiniMax HiSparse block selections."""

from __future__ import annotations

import atexit
from collections import Counter
from dataclasses import dataclass, field
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


SCHEMA = "minimax-hisparse-access-v1"
COORDINATOR_SHA256 = "cfe012ce100ee86dcc5080ca5168d51fd4abd83e8c62ee72d3606b5feafc282e"
KERNEL_SHA256 = "9daae4e368a8561b08f358193be270e512d659936bf13a812219be04e846cf0a"
WRAPPER_SHA256 = "f922d4a52f7d24d252d21e4b7f9a70f9264bb751710626c8610eb2fc662f7e99"


@dataclass(frozen=True)
class TraceLimits:
    layers: frozenset[int] = frozenset({5, 15, 25})
    ranks: frozenset[int] = frozenset({0, 1})
    max_rows: int = 36_864
    max_requests: int = 24
    max_steps: int = 512

    def __post_init__(self):
        if not self.layers or min(self.layers) < 0:
            raise ValueError("HiSparse access trace requires nonnegative layer IDs")
        if not self.ranks or min(self.ranks) < 0:
            raise ValueError("HiSparse access trace requires nonnegative rank IDs")
        if min(self.max_rows, self.max_requests, self.max_steps) < 1:
            raise ValueError("HiSparse access trace limits must be positive")

    @classmethod
    def from_env(cls):
        return cls(
            layers=frozenset(map(int, os.environ.get(
                "EKV_HISPARSE_TRACE_LAYERS", "5,15,25").split(","))),
            ranks=frozenset(map(int, os.environ.get(
                "EKV_HISPARSE_TRACE_RANKS", "0,1").split(","))),
            max_rows=int(os.environ.get("EKV_HISPARSE_TRACE_MAX_ROWS", "36864")),
            max_requests=int(os.environ.get("EKV_HISPARSE_TRACE_MAX_REQUESTS", "24")),
            max_steps=int(os.environ.get("EKV_HISPARSE_TRACE_MAX_STEPS", "512")),
        )


@dataclass
class _Request:
    rid: str
    owner: int
    generation: int
    steps: dict[int, int] = field(default_factory=dict)


def _access_tokens(blocks, block_size, seq_len, hot_capacity, prestate):
    selected = [
        token
        for block in blocks if block >= 0
        for token in range(block * block_size, min((block + 1) * block_size, seq_len))
    ]
    if seq_len <= hot_capacity:
        hits = set(selected)
    else:
        resident = set(prestate)
        resident.add(seq_len - 1)
        hits = resident.intersection(selected)
    return selected, sorted(hits)


class _Trace:
    def __init__(self, directory, limits, rank):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = directory / f"hisparse-rank{rank}-pid{os.getpid()}-{uuid4().hex}.jsonl"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.file = os.fdopen(fd, "w", encoding="utf-8", buffering=1)
        self.limits = limits
        self.rank = rank
        self.rows = 0
        self.generations = 0
        self.active = {}
        self.dropped = Counter()
        self.write(
            type="metadata", schema=SCHEMA, rank=rank, diagnostic_only=True,
            prestate_hit_semantics="derived_native_membership_including_short_and_latest_bypasses",
            performance_evidence=False, coordinator_sha256=COORDINATOR_SHA256,
            kernel_sha256=KERNEL_SHA256, wrapper_sha256=WRAPPER_SHA256,
            layers=sorted(limits.layers), max_rows=limits.max_rows,
            max_requests=limits.max_requests, max_steps=limits.max_steps,
        )
        atexit.register(self.close)

    def write(self, **event):
        self.file.write(json.dumps(event, separators=(",", ":")) + "\n")

    def drop(self, reason, count=1):
        if not count:
            return
        if not self.dropped[reason]:
            self.write(type="truncation", reason=reason)
        self.dropped[reason] += count

    def close(self):
        if not self.file.closed:
            self.write(type="summary", rows=self.rows, dropped=dict(self.dropped))
            self.file.close()
            atexit.unregister(self.close)

    def allocated(self, slot, req):
        if slot in self.active:
            raise RuntimeError(f"HiSparse access trace detected overlapping slot {slot}")
        self.generations += 1
        self.active[slot] = _Request(str(req.rid), id(req), self.generations)

    def finished(self, slot, req):
        state = self.active.get(slot)
        if state is not None:
            if (state.owner, state.rid) != (id(req), str(req.rid)):
                raise RuntimeError(f"HiSparse access trace detected mismatched owner for slot {slot}")
            del self.active[slot]

    def snapshot(self, coordinator, slots, seq_lens, blocks, layer, block_size):
        if layer not in self.limits.layers:
            return []
        count = int(coordinator.num_real_reqs.item())
        if not 0 <= count <= len(slots):
            raise RuntimeError("HiSparse access trace received an invalid real batch size")
        if self.rows >= self.limits.max_rows:
            self.drop("max_rows", count)
            return []
        hot_capacity = coordinator.device_buffer_size
        host_layer = layer - coordinator.mem_pool_device.start_layer
        rows = []
        seen = set()
        for row, slot in enumerate(slots[:count].cpu().tolist()):
            state = self.active.get(slot)
            if state is None or slot in seen:
                raise RuntimeError(f"HiSparse access trace cannot resolve unique slot {slot}")
            seen.add(slot)
            step = state.steps.get(layer, 0)
            if state.generation > self.limits.max_requests:
                self.drop("max_requests")
                continue
            if step >= self.limits.max_steps:
                self.drop("max_steps")
                continue
            if self.rows + len(rows) >= self.limits.max_rows:
                self.drop("max_rows")
                continue
            seq_len = int(seq_lens[row].item())
            selected_blocks = blocks[row].cpu().tolist()
            prestate = (
                coordinator.req_device_buffer_tokens[host_layer, slot, :hot_capacity].cpu().tolist()
                if seq_len > hot_capacity else ()
            )
            selected, hits = _access_tokens(
                selected_blocks, block_size, seq_len, hot_capacity, prestate
            )
            rows.append(dict(
                type="access", rank=self.rank, rid=state.rid, slot=slot,
                generation=state.generation, layer=layer, step=step, seq_len=seq_len,
                block_size=block_size, hot_capacity=hot_capacity,
                selected_blocks=selected_blocks, selected_tokens=selected,
                prestate_hit_tokens=hits, short_path=seq_len <= hot_capacity,
                latest_token=seq_len - 1,
            ))
        return rows

    def commit(self, rows):
        for row in rows:
            self.write(**row)
            self.active[row["slot"]].steps[row["layer"]] = row["step"] + 1
            self.rows += 1


def _patch_coordinator(cls, directory, limits, validate_server, rank):
    init = cls.__init__
    alloc = cls.alloc_device_buffer
    finish = cls.request_finished
    swap = cls.swap_in_selected_blocks
    destroy = cls.destroy

    @wraps(init)
    def traced_init(self, *args, **kwargs):
        validate_server()
        init(self, *args, **kwargs)
        if (not self.is_m3_hisparse or self.spec_swap.enabled or self.skip_io
                or self.compress_ratio != 1):
            raise RuntimeError("HiSparse access trace requires native MiniMax without spec or skip-IO")
        worker_rank = rank()
        self._ekv_hisparse_trace = (
            _Trace(directory, limits, worker_rank) if worker_rank in limits.ranks else None
        )

    @wraps(alloc)
    def traced_alloc(self, req):
        result = alloc(self, req)
        if trace := self._ekv_hisparse_trace:
            trace.allocated(req.kv.req_pool_idx, req)
        return result

    @wraps(finish)
    def traced_finish(self, req):
        slot = req.kv.req_pool_idx
        result = finish(self, req)
        if trace := self._ekv_hisparse_trace:
            trace.finished(slot, req)
        return result

    @wraps(swap)
    def traced_swap(self, req_pool_indices, seq_lens, top_k_blocks, layer_id, sparse_block_size):
        trace = self._ekv_hisparse_trace
        rows = trace.snapshot(
            self, req_pool_indices, seq_lens, top_k_blocks, layer_id, sparse_block_size
        ) if trace else []
        result = swap(self, req_pool_indices, seq_lens, top_k_blocks, layer_id, sparse_block_size)
        if rows:
            trace.commit(rows)
        return result

    @wraps(destroy)
    def traced_destroy(self):
        result = destroy(self)
        if trace := self._ekv_hisparse_trace:
            trace.close()
        return result

    cls.__init__ = traced_init
    cls.alloc_device_buffer = traced_alloc
    cls.request_finished = traced_finish
    cls.swap_in_selected_blocks = traced_swap
    cls.destroy = traced_destroy


def _validate_state(*, phase, graph_disabled, overlap_disabled, radix_disabled,
                    speculative_algorithm, device, device_type, disaggregation_mode,
                    graph_config):
    if (graph_disabled and overlap_disabled and radix_disabled
            and speculative_algorithm in (None, "NONE")
            and device_type == "cuda" and disaggregation_mode == "null"):
        return
    raise RuntimeError(
        f"HiSparse access trace requires eager colocated CUDA without overlap, radix or spec "
        f"({phase}): cuda_graph_fully_disabled={graph_disabled!r}, "
        f"disable_overlap_schedule={overlap_disabled!r}, disable_radix_cache={radix_disabled!r}, "
        f"speculative_algorithm={speculative_algorithm!r}, device={device!r}, "
        f"device_type={device_type!r}, disaggregation_mode={disaggregation_mode!r}, "
        f"cuda_graph_config={graph_config!r}"
    )


def install_hisparse_access_trace(directory):
    if (int(os.environ.get("EKV_RESIDENT_LIMIT", "0"))
            or os.environ.get("EKV_FUTURE_POLICY", "off") != "off"
            or os.environ.get("EKV_VMM_STATS_DIR")):
        raise RuntimeError("HiSparse access trace requires the native residency path")

    import torch
    from sglang.srt.managers import hisparse_coordinator
    from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.runtime_context import (
        get_device, get_disagg, get_exec, get_memory, get_schedule, get_spec,
    )

    cls = hisparse_coordinator.HiSparseCoordinator
    if getattr(cls, "_ekv_access_trace_installed", False):
        return
    source = Path(hisparse_coordinator.__file__).resolve()
    kernels = source.parents[2] / "kernels"
    for path, digest in (
        (source, COORDINATOR_SHA256),
        (kernels / "jit/csrc/kvcacheio/hisparse.cuh", KERNEL_SHA256),
        (kernels / "ops/kvcache/hisparse.py", WRAPPER_SHA256),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Unaudited HiSparse access trace source: {path}")

    def validate_server(phase="HiSparseCoordinator initialization"):
        device = get_device().device
        _validate_state(
            phase=phase,
            graph_disabled=cuda_graph_fully_disabled(),
            overlap_disabled=get_schedule().disable_overlap_schedule,
            radix_disabled=get_memory().disable_radix_cache,
            speculative_algorithm=get_spec().speculative_algorithm,
            device=device,
            device_type=torch.device(device).type,
            disaggregation_mode=get_disagg().disaggregation_mode,
            graph_config=get_exec().graph.cuda_graph_config,
        )

    limits = TraceLimits.from_env()
    load_model = ModelRunner.load_model

    @wraps(load_model)
    def traced_load_model(self, *args, **kwargs):
        validate_server("before ModelRunner.load_model")
        return load_model(self, *args, **kwargs)

    _patch_coordinator(
        cls, Path(directory), limits, validate_server, torch.distributed.get_rank
    )
    ModelRunner.load_model = traced_load_model
    cls._ekv_access_trace_installed = True
