"""Static mixed residency for the pinned MiniMax eager HiSparse runtime."""
from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import time

import torch

from .future_residency import decide_resident_claim
from .residency_broker import (
    ResidencyBroker,
    page_aligned_growth_slots,
    page_aligned_slots,
    remaining_output_tokens,
)
from .shared_prefix import (
    SharedPrefixRegistry,
    make_prefix_claim,
    resolve_shared_block_locations,
)


_REQUEST_DECISION_LIMIT = 4096


def remember_request_decision(
    stats: dict[str, object],
    request_id: object,
    *,
    actual_residence: str,
    reason: str,
    requested_residence: str | None,
    expected_output_tokens: int | None,
    sequence_tokens: int,
    native_buffer_tokens: int,
    slots_available: bool,
    tp_world_size: int,
    weight: float | None = None,
    deadline_after_ms: int | None = None,
    kind: str | None = None,
    prefix_group: str | None = None,
    shared_prefix_tokens: int = 0,
) -> None:
    """Keep bounded, rank-local evidence for the executor to reconcile."""

    if not isinstance(request_id, str) or not request_id:
        return
    decisions = stats.get("requests")
    if not isinstance(decisions, OrderedDict):
        decisions = OrderedDict()
        stats["requests"] = decisions
    decisions.pop(request_id, None)
    decisions[request_id] = {
        "actual_residence": actual_residence,
        "reason": reason,
        "requested_residence": requested_residence,
        "expected_output_tokens": expected_output_tokens,
        "sequence_tokens": sequence_tokens,
        "native_buffer_tokens": native_buffer_tokens,
        "slots_available": slots_available,
        "tp_world_size": tp_world_size,
        "weight": weight,
        "deadline_after_ms": deadline_after_ms,
        "kind": kind,
        "prefix_group": prefix_group,
        "shared_prefix_tokens": shared_prefix_tokens,
        "timestamp_monotonic_ns": time.monotonic_ns(),
    }
    while len(decisions) > _REQUEST_DECISION_LIMIT:
        decisions.popitem(last=False)


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
    shared_prefix_mode = os.environ.get("EKV_SHARED_PREFIX_REUSE", "off")
    if shared_prefix_mode not in ("off", "direct"):
        raise ValueError("EKV_SHARED_PREFIX_REUSE must be off or direct")
    if resolver == "triton":
        from .resident_lookup import resolve_resident
    reserve = int(os.environ.get("EKV_RESIDENT_FREE_RESERVE", "0"))
    if limit < 0 or reserve <= 0:
        raise ValueError("Static residency requires a positive physical slot reserve")
    hint_policy = os.environ.get("EKV_RESIDENT_HINT_POLICY", "off")
    if hint_policy not in ("off", "adaptive"):
        raise ValueError("EKV_RESIDENT_HINT_POLICY must be off or adaptive")
    if hint_policy == "adaptive" and os.environ.get("EKV_FUTURE_POLICY", "off") == "off":
        raise ValueError("adaptive resident hints require the future scheduler hook")
    if shared_prefix_mode == "direct" and hint_policy == "off":
        raise ValueError("direct shared-prefix reuse requires adaptive resident hints")
    if shared_prefix_mode == "direct" and full_batch_swap:
        raise ValueError("direct shared-prefix reuse does not support full-batch swap")
    minimum_hint_tokens = int(
        os.environ.get("EKV_RESIDENT_MIN_HINT_OUTPUT_TOKENS", "256")
    )
    if minimum_hint_tokens < 1:
        raise ValueError("resident hint output threshold must be positive")
    broker_mode = os.environ.get("EKV_RESIDENT_BROKER", "off")
    if broker_mode not in ("off", "dry_run", "enforce"):
        raise ValueError("EKV_RESIDENT_BROKER must be off, dry_run or enforce")
    broker_lookahead_tokens = int(
        os.environ.get("EKV_RESIDENT_BROKER_LOOKAHEAD_TOKENS", "256")
    )
    if broker_lookahead_tokens < 1:
        raise ValueError("resident broker lookahead must be positive")

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
    if hint_policy == "adaptive":
        from .sglang_future_hook import _extract_hint

    def hinted_output_tokens(req):
        if hint_policy != "adaptive":
            return None
        try:
            hint = _extract_hint(req)
        except (AttributeError, TypeError, ValueError):
            return None
        return None if hint is None else hint.expected_output_tokens

    def growth_estimate(req, current_tokens, pool, expected_override=None):
        predicted_tokens = (
            expected_override
            if expected_override is not None
            else hinted_output_tokens(req)
        )
        if predicted_tokens is None:
            predicted_tokens = broker_lookahead_tokens
        growth_slots = page_aligned_growth_slots(
            current_tokens=current_tokens,
            target_tokens=current_tokens + predicted_tokens,
            page_size=pool.page_size,
        )
        logical_token_bytes = pool.layer_num * (
            pool.bytes_per_token_k + pool.bytes_per_token_v
        )
        return growth_slots, growth_slots * logical_token_bytes, logical_token_bytes

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
        self._ekv_residency_broker = ResidencyBroker()
        self._ekv_broker_baseline_tokens = {}
        self._ekv_residency_stats = {"resident_admissions": 0, "native_admissions": 0,
                                     "resident_decode_rows": 0, "native_decode_rows": 0,
                                     "mixed_decode_batches": 0, "resident_layer_rows": 0,
                                     "shared_prefix_direct_blocks": 0,
                                     "shared_prefix_native_blocks": 0,
                                     "shared_prefix_native_requests": 0,
                                     "shared_prefix_partial_requests": 0,
                                     "future_hint_claim_attempts": 0, "future_hint_declines": 0,
                                     "future_hint_missing": 0, "future_hint_invalid": 0,
                                     "future_decision_reasons": {},
                                     "broker_dry_run_declines": 0,
                                     "resident_broker_mode": broker_mode,
                                     "requests": OrderedDict()}
        self._ekv_shared_prefixes = (
            SharedPrefixRegistry() if shared_prefix_mode == "direct" else None
        )

    def publish_broker_stats(self):
        snapshot = self._ekv_residency_broker.snapshot()
        stats = self._ekv_residency_stats
        stats["resident_broker"] = snapshot
        return snapshot

    def promotion_plan(coordinator, req):
        """Capture a promotion's budget before native cleanup mutates request state."""
        pool = coordinator.mem_pool_device
        current_tokens = req.kv.kv_committed_len
        predicted_total = hinted_output_tokens(req)
        baseline = coordinator._ekv_broker_baseline_tokens.get(req.kv.req_pool_idx)
        if predicted_total is not None and baseline is not None:
            predicted_total = remaining_output_tokens(
                predicted_total=predicted_total,
                current_tokens=current_tokens,
                baseline_tokens=baseline,
            )
        growth_slots, growth_bytes, token_bytes = growth_estimate(
            req, current_tokens, pool, predicted_total
        )
        return current_tokens, growth_slots, growth_bytes, token_bytes

    def broker_can_promote(self, req, available_tokens):
        if broker_mode == "off":
            return True
        pool = self.mem_pool_device
        current_tokens, growth_slots, _, token_bytes = promotion_plan(self, req)
        required_slots = page_aligned_slots(current_tokens, pool.page_size) + growth_slots
        decision = self._ekv_residency_broker.assess(
            req.kv.req_pool_idx,
            available_bytes=available_tokens * token_bytes,
            fixed_reserve_bytes=reserve * token_bytes,
            predicted_growth_bytes=required_slots * token_bytes,
        )
        self._ekv_residency_stats["resident_broker_promotion_checks"] = (
            self._ekv_residency_stats.get("resident_broker_promotion_checks", 0) + 1
        )
        publish_broker_stats(self)
        return broker_mode == "dry_run" or decision.admit

    def broker_promotion_plan(self, req):
        if broker_mode == "off":
            return None
        return promotion_plan(self, req)

    def broker_commit_promotion(self, req, growth_bytes):
        if broker_mode == "off":
            return
        current_tokens = req.kv.kv_committed_len
        key = req.kv.req_pool_idx
        self._ekv_residency_broker.reserve(key, predicted_growth_bytes=growth_bytes)
        self._ekv_residency_broker.set_last_length(key, current_tokens)
        if self._ekv_shared_prefixes is not None:
            claim = self._ekv_shared_prefixes.claim_for(key)
            if claim is not None:
                self._ekv_shared_prefixes.register_source(key, req, claim)
        publish_broker_stats(self)

    def broker_release_request(self, req):
        self._ekv_residency_broker.release(req.kv.req_pool_idx)
        publish_broker_stats(self)

    def broker_finish_request(self, req):
        self._ekv_residency_broker.release(req.kv.req_pool_idx)
        self._ekv_broker_baseline_tokens.pop(req.kv.req_pool_idx, None)
        publish_broker_stats(self)

    def stage(self, req):
        stats = self._ekv_residency_stats
        if broker_mode != "off":
            self._ekv_broker_baseline_tokens.setdefault(
                req.kv.req_pool_idx, req.kv.kv_committed_len
            )

        residence = expected = None
        hint_weight = hint_deadline = hint_kind = hint_prefix = None
        hint_shared_prefix_tokens = 0
        claim = True
        shared_claim = None
        decline_reason = None
        claim_reason = None
        slots_available = len(self._ekv_resident) < limit

        def record_decision(actual_residence, reason):
            remember_request_decision(
                stats,
                getattr(req, "rid", None),
                actual_residence=actual_residence,
                reason=reason,
                requested_residence=residence,
                expected_output_tokens=expected,
                sequence_tokens=req.kv.kv_allocated_len,
                native_buffer_tokens=self.device_buffer_size,
                slots_available=slots_available,
                tp_world_size=self.tp_world_size,
                weight=hint_weight,
                deadline_after_ms=hint_deadline,
                kind=hint_kind,
                prefix_group=hint_prefix,
                shared_prefix_tokens=hint_shared_prefix_tokens,
            )

        def adopt_native(reason=None):
            stats["native_admissions"] += 1
            if reason is not None:
                reasons = stats["future_decision_reasons"]
                reasons[reason] = reasons.get(reason, 0) + 1
            record_decision("host_backed", reason or "native")
            publish_broker_stats(self)
            original_stage(self, req)
        if hint_policy == "adaptive":
            try:
                hint = _extract_hint(req)
            except (AttributeError, TypeError, ValueError):
                hint = None
                stats["future_hint_invalid"] += 1
            if hint is None:
                stats["future_hint_missing"] += 1
                residence = expected = None
            else:
                residence = hint.residence
                expected = hint.expected_output_tokens
                hint_weight = hint.weight
                hint_deadline = hint.deadline_after_ms
                hint_kind = hint.kind
                hint_prefix = hint.prefix_group
                hint_shared_prefix_tokens = hint.shared_prefix_tokens
                if self._ekv_shared_prefixes is not None:
                    try:
                        shared_claim = make_prefix_claim(
                            req,
                            group=hint.prefix_group,
                            shared_tokens=hint.shared_prefix_tokens,
                        )
                    except (AttributeError, TypeError, ValueError):
                        stats["shared_prefix_invalid"] = (
                            stats.get("shared_prefix_invalid", 0) + 1
                        )
                    if shared_claim is not None:
                        self._ekv_shared_prefixes.register_claim(
                            req.kv.req_pool_idx, shared_claim
                        )
            decision = decide_resident_claim(
                sequence_tokens=req.kv.kv_allocated_len,
                native_buffer_tokens=self.device_buffer_size,
                slots_available=slots_available,
                hint_residence=residence,
                expected_output_tokens=expected,
                minimum_hint_output_tokens=minimum_hint_tokens,
            )
            claim = decision.claim
            claim_reason = decision.reason
            if not claim:
                decline_reason = decision.reason
        elif not slots_available or req.kv.kv_allocated_len <= self.device_buffer_size:
            claim = False

        # Every TP worker must take the collective path before any rank can
        # switch to the native staging queue. A local disagreement therefore
        # becomes a common conservative fallback instead of a collective hang.
        intent = torch.tensor(int(claim), dtype=torch.int64, device="cpu")
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(intent, op=torch.distributed.ReduceOp.MIN,
                                         group=self.tp_group)
        if not intent.item():
            if hint_policy == "adaptive":
                stats["future_hint_declines"] += 1
                decline_reason = decline_reason or "peer-declined"
            adopt_native(decline_reason)
            return
        if hint_policy == "adaptive":
            stats["future_hint_claim_attempts"] += 1

        available = self.token_to_kv_pool_allocator.hisparse_attn_allocator.available_size()
        # Match the decision on every TP worker before changing staging order.
        slots = torch.tensor(available, dtype=torch.int64, device="cpu")
        if self.tp_world_size > 1:
            torch.distributed.all_reduce(slots, op=torch.distributed.ReduceOp.MIN,
                                         group=self.tp_group)
        if slots.item() < reserve:
            adopt_native("insufficient-physical-reserve")
            return
        if broker_mode != "off":
            pool = self.mem_pool_device
            _, predicted_growth, logical_token_bytes = growth_estimate(
                req, req.kv.kv_allocated_len, pool, expected
            )
            key = req.kv.req_pool_idx
            # Assess without mutating local ledger state, then require every TP
            # rank to agree before any rank commits the reservation and takes
            # the resident staging path.  Calling try_admit independently made
            # an otherwise synchronized request vulnerable to path divergence
            # if one rank's ledger observed a different capacity.
            decision = self._ekv_residency_broker.assess(
                key,
                available_bytes=slots.item() * logical_token_bytes,
                fixed_reserve_bytes=reserve * logical_token_bytes,
                predicted_growth_bytes=predicted_growth,
            )
            broker_admit = torch.tensor(
                int(decision.admit), dtype=torch.int64, device="cpu"
            )
            if self.tp_world_size > 1:
                torch.distributed.all_reduce(
                    broker_admit,
                    op=torch.distributed.ReduceOp.MIN,
                    group=self.tp_group,
                )
            if not broker_admit.item():
                if broker_mode == "enforce":
                    decline_reason = "broker-insufficient-growth-headroom"
                    adopt_native("broker-insufficient-growth-headroom")
                    return
                stats["broker_dry_run_declines"] += 1
            self._ekv_residency_broker.reserve(
                key, predicted_growth_bytes=predicted_growth
            )
            self._ekv_residency_broker.set_last_length(key, req.kv.kv_allocated_len)
        event = torch.cuda.Event()
        event.record()
        self._ekv_resident.add(req.kv.req_pool_idx)
        if shared_claim is not None:
            self._ekv_shared_prefixes.register_source(
                req.kv.req_pool_idx, req, shared_claim
            )
        stats["resident_admissions"] += 1
        record_decision(
            "resident",
            claim_reason or ("resident-slot" if hint_policy == "off" else "resident-claim"),
        )
        req.hisparse_staging = True
        # Preserve native queue order, TP readiness and scheduler admission.
        self.ack_staging_queue.append(coordinator.HiSparseAct(event, event, req))

    def allocate_buffer(self, req):
        if req.kv.req_pool_idx not in self._ekv_resident:
            return original_buffer(self, req)

    def map_latest(self, lengths, locations, rows, lengths_cpu, rows_cpu):
        resident_rows = [i for i, row in enumerate(rows_cpu.tolist())
                         if row in self._ekv_resident]
        if broker_mode != "off" and resident_rows:
            pool = self.mem_pool_device
            bytes_per_token = pool.layer_num * (
                pool.bytes_per_token_k + pool.bytes_per_token_v
            )
            request_indices = rows_cpu.tolist()
            current_lengths = lengths_cpu.tolist()
            for row in resident_rows:
                self._ekv_residency_broker.consume_new_page_slots(
                    request_indices[row],
                    length=int(current_lengths[row]),
                    bytes_per_token=bytes_per_token,
                    page_size=pool.page_size,
                )
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
        request_rows_cpu = rows_cpu.tolist()
        native_lengths = None
        if full_batch_swap and sparse_rows:
            # The pinned block swap kernel's zero-length path writes -1 to
            # this row without reading host/device cache entries or LRU.
            native_lengths = lengths.clone().index_fill_(0, resident, 0)
        self._ekv_decode_rows = (
            resident,
            sparse,
            native_lengths,
            request_rows_cpu,
            sparse_rows,
        )
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
        resident, sparse, native_lengths, request_rows_cpu, sparse_rows = selection
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
            registry = ctrl._ekv_shared_prefixes
            if registry is None:
                result[sparse] = ctrl.swap_in_selected_blocks(
                    req_pool_indices=rows[sparse], seq_lens=lengths[sparse],
                    top_k_blocks=topk_idx[0, sparse], layer_id=layer_id,
                    sparse_block_size=block_size)
            else:
                sparse_blocks = topk_idx[0, sparse]
                sparse_row_ids = [request_rows_cpu[i] for i in sparse_rows]
                native_local = []
                partial = []
                direct_blocks = 0
                native_blocks = 0
                for local, row_id in enumerate(sparse_row_ids):
                    source = registry.resolve(int(row_id))
                    if source is None:
                        native_local.append(local)
                        native_blocks += int(sparse_blocks[local].numel())
                        continue
                    valid, physical = resolve_shared_block_locations(
                        sparse_blocks[local],
                        source_row=source.row,
                        seq_len=lengths[sparse[local]],
                        shared_tokens=source.claim.tokens,
                        block_size=block_size,
                        request_table=self.req_to_token,
                        device_mapping=mapping,
                    )
                    if not bool(valid.any()):
                        native_local.append(local)
                        native_blocks += int(sparse_blocks[local].numel())
                        continue
                    result[sparse[local]].view(-1, block_size)[valid] = physical[valid]
                    direct_blocks += int(valid.sum().item())
                    if bool((~valid).any()):
                        partial.append((local, ~valid))
                        native_blocks += int((~valid).sum().item())
                if direct_blocks:
                    ctrl._ekv_residency_stats["shared_prefix_direct_blocks"] = (
                        ctrl._ekv_residency_stats.get("shared_prefix_direct_blocks", 0)
                        + direct_blocks
                    )
                if native_blocks:
                    ctrl._ekv_residency_stats["shared_prefix_native_blocks"] = (
                        ctrl._ekv_residency_stats.get("shared_prefix_native_blocks", 0)
                        + native_blocks
                    )
                if native_local:
                    ctrl._ekv_residency_stats["shared_prefix_native_requests"] = (
                        ctrl._ekv_residency_stats.get(
                            "shared_prefix_native_requests", 0
                        )
                        + len(native_local)
                    )
                if partial:
                    ctrl._ekv_residency_stats["shared_prefix_partial_requests"] = (
                        ctrl._ekv_residency_stats.get(
                            "shared_prefix_partial_requests", 0
                        )
                        + len(partial)
                    )
                if native_local:
                    chosen = torch.tensor(native_local, device=rows.device, dtype=torch.long)
                    result[sparse[chosen]] = ctrl.swap_in_selected_blocks(
                        req_pool_indices=rows[sparse[chosen]], seq_lens=lengths[sparse[chosen]],
                        top_k_blocks=sparse_blocks[chosen], layer_id=layer_id,
                        sparse_block_size=block_size)
                for local, missing in partial:
                    native = ctrl.swap_in_selected_blocks(
                        req_pool_indices=rows[sparse[local]].reshape(1),
                        seq_lens=lengths[sparse[local]].reshape(1),
                        top_k_blocks=sparse_blocks[local][missing].reshape(1, -1),
                        layer_id=layer_id,
                        sparse_block_size=block_size,
                    )
                    result[sparse[local]].view(-1, block_size)[missing] = (
                        native.reshape(-1, block_size)
                    )
                ctrl._ekv_residency_stats["shared_prefix"] = registry.snapshot()
        return result.unsqueeze(0)

    def finish(self, req):
        broker_finish_request(self, req)
        self._ekv_resident.discard(req.kv.req_pool_idx)
        if self._ekv_shared_prefixes is not None:
            self._ekv_shared_prefixes.remove_request(req.kv.req_pool_idx)
        return original_finish(self, req)

    def cleanup_promotion_source(self, req):
        # Promotion releases the host-owned hot buffer, but the request keeps
        # generating.  Preserve its staging baseline for later demote/promote
        # cycles; only final completion or abort may discard it.
        broker_release_request(self, req)
        self._ekv_resident.discard(req.kv.req_pool_idx)
        if self._ekv_shared_prefixes is not None:
            self._ekv_shared_prefixes.remove_source(req.kv.req_pool_idx)
        return original_finish(self, req)

    def register_promoted_source(self, req):
        if self._ekv_shared_prefixes is None:
            return
        row = req.kv.req_pool_idx
        claim = self._ekv_shared_prefixes.claim_for(row)
        if claim is not None:
            self._ekv_shared_prefixes.register_source(row, req, claim)

    def remove_demoted_source(self, req):
        if self._ekv_shared_prefixes is not None:
            self._ekv_shared_prefixes.remove_source(req.kv.req_pool_idx)

    def abort(self, req):
        broker_finish_request(self, req)
        self._ekv_resident.discard(req.kv.req_pool_idx)
        if self._ekv_shared_prefixes is not None:
            self._ekv_shared_prefixes.remove_request(req.kv.req_pool_idx)
        return original_abort(self, req)

    cls.__init__ = initialize
    cls.admit_request_into_staging = stage
    cls.alloc_device_buffer = allocate_buffer
    cls.map_last_loc_to_buffer = map_latest
    cls.request_finished = finish
    cls.abort_staging_request = abort
    if broker_mode != "off" or shared_prefix_mode == "direct":
        cls._ekv_cleanup_promotion_source = cleanup_promotion_source
    if broker_mode != "off":
        cls._ekv_broker_can_promote = broker_can_promote
        cls._ekv_broker_promotion_plan = broker_promotion_plan
        cls._ekv_broker_commit_promotion = broker_commit_promotion
        cls._ekv_broker_release_request = broker_release_request
    if shared_prefix_mode == "direct":
        cls._ekv_register_promoted_source = register_promoted_source
        cls._ekv_remove_demoted_source = remove_demoted_source
    attn._hisparse_swap_in_blocks = swap
    cls._ekv_static_residency = True
