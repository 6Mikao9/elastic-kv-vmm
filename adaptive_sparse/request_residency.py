"""Static mixed residency for the pinned MiniMax eager HiSparse runtime."""
import hashlib
import os
from pathlib import Path

import torch


def install_request_residency():
    limit = int(os.environ.get("EKV_RESIDENT_LIMIT", "0"))
    if not limit:
        return
    resolver = os.environ.get("EKV_RESIDENT_RESOLVER", "torch")
    if resolver not in ("torch", "triton"):
        raise ValueError("EKV_RESIDENT_RESOLVER must be torch or triton")
    full_batch_swap = os.environ.get("EKV_RESIDENT_FULL_BATCH_SWAP", "0")
    if full_batch_swap not in ("0", "1") or (full_batch_swap == "1" and resolver != "triton"):
        raise ValueError("EKV_RESIDENT_FULL_BATCH_SWAP requires Triton resolver when enabled")
    full_batch_swap = full_batch_swap == "1"
    if resolver == "triton":
        from .resident_lookup import resolve_resident
    reserve = int(os.environ.get("EKV_RESIDENT_FREE_RESERVE", "0"))
    if limit < 0 or reserve <= 0:
        raise ValueError("Static residency requires a positive physical slot reserve")

    from sglang.srt.layers.attention import minimax_sparse_backend as backend
    from sglang.srt.managers import hisparse_coordinator as coordinator
    from sglang.srt.mem_cache.allocator import hisparse as allocator
    from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
    from sglang.srt.runtime_context import get_disagg, get_schedule, get_spec

    expected = {
        coordinator: "cfe012ce100ee86dcc5080ca5168d51fd4abd83e8c62ee72d3606b5feafc282e",
        backend: "7f8e954d3dea6aa4a76c549a7854f3743b67be49289d578e83524b762623f042",
        allocator: "f3f15a599bcb09131606bd5bf93511a4bf39f4acec5835e73975967b5697f732",
    }
    for module, digest in expected.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Static residency source mismatch: {module.__name__}")

    if full_batch_swap:
        package = Path(coordinator.__file__).resolve().parents[2]
        kernel = package / "kernels/jit/csrc/kvcacheio/hisparse.cuh"
        digest = "9daae4e368a8561b08f358193be270e512d659936bf13a812219be04e846cf0a"
        if hashlib.sha256(kernel.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Unaudited HiSparse zero-length swap kernel")

    cls = coordinator.HiSparseCoordinator
    if getattr(cls, "_ekv_static_residency", False):
        return
    original_init = cls.__init__
    original_stage = cls.admit_request_into_staging
    original_buffer = cls.alloc_device_buffer
    original_map = cls.map_last_loc_to_buffer
    original_finish = cls.request_finished
    original_abort = cls.abort_staging_request
    attn = backend.MiniMaxSparseAttnBackend
    original_swap = attn._hisparse_swap_in_blocks

    def initialize(self, *args, **kwargs):
        if (not cuda_graph_fully_disabled() or not get_schedule().disable_overlap_schedule
                or get_disagg().disaggregation_mode != "null"
                or get_spec().speculative_algorithm not in (None, "NONE")):
            raise RuntimeError("Static residency requires eager colocated execution without overlap or speculation")
        original_init(self, *args, **kwargs)
        if not self.is_m3_hisparse or self.spec_swap.enabled:
            raise RuntimeError("Static residency requires MiniMax HiSparse without speculation")
        self._ekv_resident = set()
        self._ekv_decode_rows = None
        self._ekv_residency_stats = {"resident_admissions": 0, "native_admissions": 0,
                                     "resident_decode_rows": 0, "native_decode_rows": 0,
                                     "mixed_decode_batches": 0, "resident_layer_rows": 0}

    def stage(self, req):
        if (len(self._ekv_resident) < limit
                and req.kv.kv_allocated_len > self.device_buffer_size):
            available = self.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size()
            # Match the decision on every TP worker before changing staging order.
            slots = torch.tensor(available, dtype=torch.int64, device="cpu")
            if self.tp_world_size > 1:
                torch.distributed.all_reduce(slots, op=torch.distributed.ReduceOp.MIN,
                                             group=self.tp_group)
            if slots.item() >= reserve:
                event = torch.cuda.Event()
                event.record()
                self._ekv_resident.add(req.kv.req_pool_idx)
                self._ekv_residency_stats["resident_admissions"] += 1
                req.hisparse_staging = True
                # Preserve native queue order, TP readiness and scheduler admission.
                self.ack_staging_queue.append(coordinator.HiSparseAct(event, event, req))
                return
        self._ekv_residency_stats["native_admissions"] += 1
        original_stage(self, req)

    def allocate_buffer(self, req):
        if req.kv.req_pool_idx not in self._ekv_resident:
            return original_buffer(self, req)

    def map_latest(self, lengths, locations, rows, lengths_cpu, rows_cpu):
        resident_rows = [i for i, row in enumerate(rows_cpu.tolist())
                         if row in self._ekv_resident]
        resident_set = set(resident_rows)
        sparse_rows = [i for i in range(len(rows_cpu)) if i not in resident_set]
        self._ekv_residency_stats["resident_decode_rows"] += len(resident_rows)
        self._ekv_residency_stats["native_decode_rows"] += len(sparse_rows)
        self._ekv_residency_stats["mixed_decode_batches"] += bool(resident_rows and sparse_rows)
        if not resident_rows:
            self._ekv_decode_rows = None
            return original_map(self, lengths, locations, rows, lengths_cpu, rows_cpu)

        resident = torch.tensor(resident_rows, device=rows.device, dtype=torch.long)
        sparse = torch.tensor(sparse_rows, device=rows.device, dtype=torch.long)
        native_lengths = None
        if full_batch_swap and sparse_rows:
            # The pinned block swap kernel's zero-length path writes -1 to
            # this row without reading host/device cache entries or LRU.
            native_lengths = lengths.clone().index_fill_(0, resident, 0)
        self._ekv_decode_rows = resident, sparse, native_lengths
        request_table = self.req_to_token_pool.req_to_token
        mapping = self.mem_pool_device.full_to_hisparse_device_index_mapping
        previous = mapping[request_table[rows[resident], lengths[resident] - 2]]
        physical = self.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc_decode(
            lengths[resident], lengths_cpu[resident_rows], previous)
        if physical is None:
            raise RuntimeError("Static resident decode exceeded the physical slot budget")
        mapping[locations[resident]] = physical
        if sparse_rows:
            original_map(self, lengths[sparse], locations[sparse], rows[sparse],
                         lengths_cpu[sparse_rows], rows_cpu[sparse_rows])

    def swap(self, forward_batch, topk_idx, layer_id):
        ctrl = self.hisparse_coordinator
        selection = ctrl._ekv_decode_rows
        if selection is None:
            return original_swap(self, forward_batch, topk_idx, layer_id)
        resident, sparse, native_lengths = selection
        ctrl._ekv_residency_stats["resident_layer_rows"] += len(resident)
        rows, lengths = forward_batch.req_pool_indices, forward_batch.seq_lens
        if topk_idx.shape[:2] != (1, len(rows)) or len(rows) != len(resident) + len(sparse):
            raise RuntimeError("Static residency requires unpadded eager decode")
        block_size = self.block_size_k
        mapping = ctrl.mem_pool_device.full_to_hisparse_device_index_mapping
        if native_lengths is not None:
            result = ctrl.swap_in_selected_blocks(
                req_pool_indices=rows, seq_lens=native_lengths,
                top_k_blocks=topk_idx[0], layer_id=layer_id,
                sparse_block_size=block_size,
            ).unsqueeze(0)
            resolve_resident(topk_idx, self.req_to_token, mapping, rows, lengths,
                             resident, result, block_size)
            return result
        result = torch.empty((len(rows), topk_idx.shape[2] * block_size),
                             device=rows.device, dtype=torch.int32)
        if resolver == "triton":
            resolve_resident(topk_idx, self.req_to_token, mapping, rows, lengths,
                             resident, result.unsqueeze(0), block_size)
        else:
            blocks = topk_idx[0, resident].long()
            positions = blocks.unsqueeze(-1) * block_size + torch.arange(
                block_size, device=rows.device)
            positions = positions.flatten(1)
            valid = (positions >= 0) & (positions < lengths[resident, None])
            logical = self.req_to_token[rows[resident, None], positions.masked_fill(~valid, 0)]
            result[resident] = mapping[logical].masked_fill(~valid, -1).to(torch.int32)
        if sparse.numel():
            result[sparse] = ctrl.swap_in_selected_blocks(
                req_pool_indices=rows[sparse], seq_lens=lengths[sparse],
                top_k_blocks=topk_idx[0, sparse], layer_id=layer_id,
                sparse_block_size=block_size)
        return result.unsqueeze(0)

    def finish(self, req):
        self._ekv_resident.discard(req.kv.req_pool_idx)
        return original_finish(self, req)

    def abort(self, req):
        self._ekv_resident.discard(req.kv.req_pool_idx)
        return original_abort(self, req)

    cls.__init__ = initialize
    cls.admit_request_into_staging = stage
    cls.alloc_device_buffer = allocate_buffer
    cls.map_last_loc_to_buffer = map_latest
    cls.request_finished = finish
    cls.abort_staging_request = abort
    attn._hisparse_swap_in_blocks = swap
    cls._ekv_static_residency = True
