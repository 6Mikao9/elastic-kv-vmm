"""Diagnostic-only bitwise oracle for selected MiniMax sparse decode KV."""
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
from pathlib import Path

import torch


_ACTIVE = ContextVar("ekv_shadow_decode", default=None)


class LogicalShadow:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.layers = {}
        self.counts = torch.zeros(7, device=device, dtype=torch.int64)
        self.dirty = False
        self.migration_counts = dict.fromkeys((
            "migration_checked_tokens", "migration_checked_bytes",
            "migration_invalid_tokens", "migration_mismatched_tokens",
        ), 0)

    def reset(self, locations):
        for _, _, valid in self.layers.values():
            valid[locations] = False

    def write(self, layer, locations, key, value, scales):
        if (key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16
                or any(scale not in (None, 1.0) for scale in scales)):
            raise RuntimeError("Logical shadow requires unit-scale BF16 producers")
        if layer not in self.layers:
            self.layers[layer] = (
                key.new_empty((self.capacity, *key.shape[1:])),
                value.new_empty((self.capacity, *value.shape[1:])),
                torch.zeros(self.capacity, device=self.device, dtype=torch.bool),
            )
        k, v, valid = self.layers[layer]
        k[locations] = key
        v[locations] = value
        valid[locations] = True

    def check(self, layer, k_cache, v_cache, table, lengths, rows, block_size,
              blocks, physical, live_pages, page_size):
        if layer not in self.layers or blocks.shape[0] != 1:
            raise RuntimeError("Logical shadow requires one KV head and recorded producer writes")
        k, v, initialized = self.layers[layer]
        block_limit = (lengths[:, None] + block_size - 1) // block_size
        valid_blocks = (blocks >= 0) & (blocks < block_limit)
        prefix = valid_blocks.to(torch.int32).cumprod(-1).bool()
        malformed = ((blocks != -1) & ~valid_blocks) | (valid_blocks & ~prefix)
        clean_blocks = blocks.masked_fill(~prefix, -1)
        blocks = clean_blocks[0].long()
        positions = (blocks[..., None] * block_size
                     + torch.arange(block_size, device=self.device)).flatten(1)
        selected = (positions >= 0) & (positions < lengths[:, None])
        bounded = selected & (positions < table.shape[1])
        logical = table[rows[:, None], positions.masked_fill(~bounded, 0)].long()
        logical_valid = bounded & (logical >= page_size) & (logical < self.capacity)
        logical = logical.masked_fill(~logical_valid, 0)
        source_valid = logical_valid & initialized[logical]

        physical = logical if physical is None else physical[0].long()
        in_range = (physical >= page_size) & (physical < k_cache.shape[0])
        safe = physical.masked_fill(~in_range, 0)
        backed = in_range if live_pages is None else in_range & live_pages[safe // page_size]
        safe = safe.masked_fill(~backed, 0)
        # Slot zero is the permanently mapped sentinel. Invalid physical IDs
        # are counted without touching potentially unmapped storage.
        same_k = (k_cache[safe].reshape(*safe.shape, -1).view(torch.int16)
                  == k[logical].reshape(*safe.shape, -1).view(torch.int16)).all(-1)
        same_v = (v_cache[safe].reshape(*safe.shape, -1).view(torch.int16)
                  == v[logical].reshape(*safe.shape, -1).view(torch.int16)).all(-1)
        checked = selected & source_valid & backed
        tokens = checked.sum()
        bytes_per_token = (k[0].numel() + v[0].numel()) * 2
        self.counts.add_(torch.stack((tokens, tokens * bytes_per_token,
                                     (selected & ~source_valid).sum(),
                                     (selected & ~backed).sum(),
                                     (checked & ~(same_k & same_v)).sum(),
                                     tokens.new_ones(()), malformed.sum())))
        self.dirty = True
        return (clean_blocks, physical.masked_fill(selected & ~backed, 0)
                .to(torch.int32).unsqueeze(0))

    def check_complete(self, pool, logical, physical, boundary):
        """Compare every committed token in the migrated sparse main layers."""
        if logical.ndim != 1 or physical.shape != logical.shape or not logical.numel():
            raise ValueError("Invalid migration shadow slot vectors")
        page = pool.page_size
        live = torch.zeros(pool.size // page + 1, device=self.device, dtype=torch.bool)
        pages = torch.tensor(sorted(pool.adaptive_vmm.pages), device=self.device, dtype=torch.long)
        live[pages] = True
        logical_valid = (logical >= page) & (logical < self.capacity)
        physical_valid = (physical >= page) & (physical < pool.size + page)
        safe_logical = logical.masked_fill(~logical_valid, 0)
        safe_physical = physical.masked_fill(~physical_valid, 0)
        physical_valid &= live[safe_physical // page]
        safe_physical = safe_physical.masked_fill(~physical_valid, 0)
        counts = torch.zeros(4, device=self.device, dtype=torch.int64)
        for layer in range(pool.start_layer, pool.end_layer):
            if layer not in self.layers:
                raise RuntimeError(f"Missing migration shadow writes for layer {layer}")
            key, value, initialized = self.layers[layer]
            valid = logical_valid & physical_valid & initialized[safe_logical]
            actual_key = pool.get_key_buffer(layer)[safe_physical].flatten(1).view(torch.int16)
            actual_value = pool.get_value_buffer(layer)[safe_physical].flatten(1).view(torch.int16)
            expected_key = key[safe_logical].flatten(1).view(torch.int16)
            expected_value = value[safe_logical].flatten(1).view(torch.int16)
            equal = (actual_key == expected_key).all(1) & (actual_value == expected_value).all(1)
            checked = valid.sum()
            counts.add_(torch.stack((
                checked, checked * 2 * (expected_key.shape[1] + expected_value.shape[1]),
                (~valid).sum(), (valid & ~equal).sum(),
            )))
        report = dict(zip(self.migration_counts, counts.tolist(), strict=True))
        for name, count in report.items():
            self.migration_counts[name] += count
        self.dirty = True
        if report["migration_invalid_tokens"] or report["migration_mismatched_tokens"]:
            raise RuntimeError(f"Logical KV migration shadow mismatch at {boundary}: {report}")

    def publish(self, directory, rank):
        if not self.dirty:
            return
        keys = ("checked_tokens", "checked_bytes", "uninitialized_logical_tokens",
                "invalid_physical_tokens", "mismatched_tokens", "checked_calls",
                "malformed_topk_blocks")
        stats = dict(zip(keys, self.counts.tolist(), strict=True))
        stats.update(self.migration_counts)
        stats.update(tp_rank=rank, logical_capacity=self.capacity, boundary="scheduler_idle")
        path = directory / f"shadow-tp-{rank}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        self.dirty = False
        if any(stats[key] for key in (*keys[2:5], keys[6])):
            raise RuntimeError(f"Logical KV shadow mismatch: {stats}")


def install_logical_shadow(stats_dir):
    from sglang.srt.layers.attention import minimax_sparse_backend as backend
    from sglang.srt.layers.attention.minimax_sparse_ops import minimax_sparse as ops
    from sglang.srt.managers import scheduler
    from sglang.srt.mem_cache import memory_pool
    from sglang.srt.mem_cache.allocator import hisparse as allocator
    from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
    from sglang.srt.runtime_context import get_disagg, get_schedule, get_spec

    digests = {
        memory_pool: "e3d57e41beb4d5f77c2c91e6bf9bb392ef1b5ad9e853c4521b3dfcf1e12a20e1",
        allocator: "f3f15a599bcb09131606bd5bf93511a4bf39f4acec5835e73975967b5697f732",
        backend: "7f8e954d3dea6aa4a76c549a7854f3743b67be49289d578e83524b762623f042",
        ops: "ed3386669a46a6a3a9478c37a8ff73841f60540fb970540f491a63347419a65f",
        scheduler: "fc96ed4b680fdf55621a5397f844cfd4cbd30213f44eec8a888b7cce75cafe60",
    }
    for module, digest in digests.items():
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Logical shadow source mismatch: {module.__name__}")
    directory = Path(stats_dir)
    directory.mkdir(parents=True, exist_ok=True)
    cls = allocator.HiSparseTokenToKVPoolAllocator
    if getattr(cls, "_ekv_shadow_installed", False):
        return
    pool_cls = memory_pool.MiniMaxSparseKVPool
    attn = backend.MiniMaxSparseAttnBackend

    init = cls.__init__
    store = pool_cls.set_kv_buffer
    fused_store = pool_cls.set_fused_kv_index_buffer
    forward = attn.forward_decode
    attend = ops.flash_decode_with_gqa_share_sparse
    idle = scheduler.Scheduler.on_idle

    @wraps(init)
    def initialize(self, *args, **kwargs):
        init(self, *args, **kwargs)
        pool = self._kvcache
        if type(pool) is not pool_cls:
            raise RuntimeError("Logical shadow requires native MiniMax HiSparse")
        main = pool.main_pool
        if (torch.cuda.get_device_capability(main.device) != (8, 0)
                or main.dtype != torch.bfloat16 or main.store_dtype != torch.bfloat16
                or main.kv_cache_layout != "nhd" or not cuda_graph_fully_disabled()
                or not get_schedule().disable_overlap_schedule
                or get_disagg().disaggregation_mode != "null"
                or get_spec().speculative_algorithm not in (None, "NONE")):
            raise RuntimeError("Logical shadow requires SM80 eager BF16 NHD, no PD/overlap/spec")
        shadow = pool._ekv_shadow = LogicalShadow(self.size_full + self.page_size, main.device)
        logical = self.logical_attn_allocator

        def reset_on_alloc(method):
            @wraps(method)
            def allocate(*args, **kwargs):
                locations = method(*args, **kwargs)
                if locations is not None:
                    shadow.reset(locations)
                return locations
            return allocate

        for name in ("alloc", "alloc_extend", "alloc_decode"):
            setattr(logical, name, reset_on_alloc(getattr(logical, name)))

    @wraps(store)
    def store_main(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None):
        shadow = getattr(self, "_ekv_shadow", None)
        if shadow is not None:
            logical, _, _ = memory_pool.unwrap_write_loc(loc)
            shadow.write(layer.layer_id, logical, cache_k, cache_v, (k_scale, v_scale))
        return store(self, layer, loc, cache_k, cache_v, k_scale, v_scale)

    @wraps(fused_store)
    def store_both(self, layer, loc, cache_k, cache_v, cache_idx_k, cache_idx_v,
                   k_scale=None, v_scale=None, idx_k_scale=None, idx_v_scale=None):
        # Native MiniMax HiSparse always uses the separate main store here.
        if getattr(self, "_ekv_shadow", None) is not None and not self._enable_hisparse:
            raise RuntimeError("Logical shadow cannot track a fused producer bypass")
        return fused_store(self, layer, loc, cache_k, cache_v, cache_idx_k, cache_idx_v,
                           k_scale, v_scale, idx_k_scale, idx_v_scale)

    @wraps(forward)
    def forward_decode(self, q, k, v, layer, forward_batch, *args, **kwargs):
        pool = self.kv_pool
        shadow = getattr(pool, "_ekv_shadow", None)
        if shadow is None:
            return forward(self, q, k, v, layer, forward_batch, *args, **kwargs)
        main = pool.main_pool
        tracker = getattr(main, "adaptive_vmm", None)
        pages = None
        if tracker is not None:
            pages = torch.zeros(main.size // main.page_size + 1, dtype=torch.bool,
                                device=main.device)
            pages[torch.tensor(sorted(tracker.pages), device=main.device)] = True
        token = _ACTIVE.set((shadow, layer.layer_id, pages, main.page_size))
        try:
            return forward(self, q, k, v, layer, forward_batch, *args, **kwargs)
        finally:
            _ACTIVE.reset(token)

    @wraps(attend)
    def checked_attention(q, sink, k_cache, v_cache, req_to_token, seq_lens,
                          slot_ids, block_size, topk_idx, sm_scale=None, use_tma=True,
                          page_size=0, q_scale=None, k_scale=None, v_scale=None,
                          hisparse_slots=None):
        state = _ACTIVE.get()
        if state is not None:
            shadow, layer, pages, physical_page = state
            if hisparse_slots is None:
                raise RuntimeError("Logical shadow requires explicit MiniMax HiSparse slots")
            topk_idx, hisparse_slots = shadow.check(layer, k_cache, v_cache, req_to_token, seq_lens, slot_ids,
                         block_size, topk_idx, hisparse_slots, pages, physical_page)
        return attend(q, sink, k_cache, v_cache, req_to_token, seq_lens, slot_ids,
                      block_size, topk_idx, sm_scale, use_tma, page_size, q_scale,
                      k_scale, v_scale, hisparse_slots)

    @wraps(idle)
    def on_idle(self):
        pool = self.token_to_kv_pool_allocator.get_kvcache()
        shadow = getattr(pool, "_ekv_shadow", None)
        if shadow is not None and self.is_fully_idle():
            shadow.publish(directory, scheduler.get_parallel().tp_rank)
        return idle(self)

    cls.__init__ = initialize
    pool_cls.set_kv_buffer = store_main
    pool_cls.set_fused_kv_index_buffer = store_both
    attn.forward_decode = forward_decode
    ops.flash_decode_with_gqa_share_sparse = checked_attention
    scheduler.Scheduler.on_idle = on_idle
    cls._ekv_shadow_installed = True
