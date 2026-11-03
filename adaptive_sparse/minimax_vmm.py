"""Opt-in stable-VA BF16 backing for the audited MiniMax HiSparse main pool."""

import hashlib
import os
from functools import wraps
from math import prod
from pathlib import Path
from time import perf_counter_ns


CHUNK_BYTES = 2 << 20
_SOURCE_HASHES = {
    "mem_cache/memory_pool.py": "e3d57e41beb4d5f77c2c91e6bf9bb392ef1b5ad9e853c4521b3dfcf1e12a20e1",
    "mem_cache/hisparse_memory_pool.py": "74b46cc95e8336407348e7251ad08c730f1ab6a676f9a44632007d9ae3b72480",
    "mem_cache/allocator/hisparse.py": "f3f15a599bcb09131606bd5bf93511a4bf39f4acec5835e73975967b5697f732",
    "mem_cache/allocator/paged.py": "c72f28d9c5b953a05fcc0a0451307ecdbfbfb756eca3f0073f7be3eb737d38f6",
}


class PhysicalPages:
    """Physical page ownership projected into packed, aligned K/V planes."""

    def __init__(self, backing, page_size, page_count, planes, synchronize,
                 chunk_bytes=CHUNK_BYTES, fast_decode=False):
        self.backing = backing
        self.page_size = page_size
        self.page_count = page_count
        self.planes = planes
        self.synchronize = synchronize
        self.chunk_bytes = chunk_bytes
        self.trim_sync_ns = 0
        self.fast_decode = fast_decode
        self.fast_no_new_page = 0
        self.decode_new_page = 0
        self._chunk_cache = {} if fast_decode else None
        self.pages = {0}
        self.refs = dict.fromkeys(self.chunks(0), 1)
        self.backing.ensure(self.refs)

    def chunks(self, page):
        if self._chunk_cache is not None and page in self._chunk_cache:
            return self._chunk_cache[page]
        chunks = set()
        for plane_chunk, row_bytes in self.planes:
            begin = page * self.page_size * row_bytes
            end = begin + self.page_size * row_bytes - 1
            chunks.update(range(plane_chunk + begin // self.chunk_bytes,
                                plane_chunk + end // self.chunk_bytes + 1))
        chunks = frozenset(chunks)
        if self._chunk_cache is not None:
            self._chunk_cache[page] = chunks
        return chunks

    def acquire(self, pages):
        pages = set(pages)
        if any(page <= 0 or page > self.page_count for page in pages):
            raise ValueError("invalid physical page")
        fresh = pages - self.pages
        projected = [self.chunks(page) for page in fresh]
        if projected:
            self.backing.ensure(set().union(*projected))
        for chunks in projected:
            for chunk in chunks:
                self.refs[chunk] = self.refs.get(chunk, 0) + 1
        self.pages.update(fresh)

    def validate_release(self, pages):
        if 0 in pages or not pages <= self.pages:
            raise RuntimeError("invalid or duplicate physical free")

    def release(self, pages):
        self.validate_release(pages)
        for page in pages:
            for chunk in self.chunks(page):
                self.refs[chunk] -= 1
                if not self.refs[chunk]:
                    del self.refs[chunk]
        self.pages.difference_update(pages)

    def clear(self):
        self.release(self.pages - {0})

    def stats(self):
        return {"backed_bytes": self.backing.backed_bytes,
                "reclaimable_bytes": len(self.backing.mapped_chunks - self.refs.keys())
                * self.chunk_bytes, "owned_pages": len(self.pages) - 1, "trim_sync_ns": self.trim_sync_ns,
                "fast_no_new_page": self.fast_no_new_page,
                "decode_new_page": self.decode_new_page}

    def trim_quiescent(self):
        """Call only with scheduling paused and all users on the local device."""
        pending = self.backing.mapped_chunks - self.refs.keys()
        if not pending:
            return 0
        started = perf_counter_ns()
        self.synchronize()
        self.trim_sync_ns += perf_counter_ns() - started
        return self.backing.reclaim(pending, synchronize=False) * self.chunk_bytes


def _attach_allocator(allocator, tracker):
    """Observe only the physical allocator, including grouped releases."""
    for name in ("alloc", "alloc_extend", "alloc_decode"):
        original = getattr(allocator, name)

        def wrap(method):
            @wraps(method)
            def allocate(*args, **kwargs):
                saved = (allocator.free_pages, allocator.staged_pages.copy(),
                         allocator.num_staged_pages)
                result = method(*args, **kwargs)
                if result is not None and result.numel():
                    try:
                        pages = {int(slot) // tracker.page_size for slot in result.tolist()}
                        tracker.acquire(pages)
                    except Exception:
                        (allocator.free_pages, allocator.staged_pages,
                         allocator.num_staged_pages) = saved
                        # Native errors poison Backing. The outer allocator can
                        # already own logical IDs, so propagate and fail stop.
                        raise
                return result
            return allocate

        setattr(allocator, name, wrap(original))

    if tracker.fast_decode:
        from sglang.srt.mem_cache.allocator.paged import get_num_new_pages

        tracked_decode = allocator.alloc_decode
        native_decode = tracked_decode.__wrapped__

        @wraps(native_decode)
        def decode(seq_lens, seq_lens_cpu, last_loc):
            if seq_lens_cpu.device.type != "cpu":
                raise RuntimeError("Fast VMM decode requires host sequence lengths")
            new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu, page_size=tracker.page_size, decode=True)
            if not new_pages:
                result = native_decode(seq_lens, seq_lens_cpu, last_loc)
                tracker.fast_no_new_page += 1
                return result
            result = tracked_decode(seq_lens, seq_lens_cpu, last_loc)
            if result is not None:
                tracker.decode_new_page += 1
            return result

        allocator.alloc_decode = decode

    release = allocator._release_page_ids
    clear = allocator.clear

    @wraps(release)
    def release_pages(*page_groups):
        pages = [int(page) for group in page_groups for page in group.tolist()]
        unique = set(pages)
        if len(unique) != len(pages):
            raise RuntimeError("duplicate pages in physical free batch")
        tracker.validate_release(unique)
        release(*page_groups)
        tracker.release(unique)

    @wraps(clear)
    def clear_pages():
        clear()
        tracker.clear()

    allocator._release_page_ids = release_pages
    allocator.clear = clear_pages


def install_minimax_vmm():
    """Install before constructing the audited eager MiniMax HiSparse runner."""
    import torch
    from sglang.srt.mem_cache import hisparse_memory_pool
    from sglang.srt.mem_cache.allocator.hisparse import HiSparseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
    from sglang.srt.runtime_context import get_server_args

    from elastic_kv.backing import Backing

    pool_cls = hisparse_memory_pool.HiSparseMHAMainPool
    if getattr(pool_cls, "_adaptive_vmm_installed", False):
        return
    root = Path(hisparse_memory_pool.__file__).resolve().parents[1]
    for relative, expected in _SOURCE_HASHES.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Unaudited MiniMax VMM source: {relative}")

    original_pool_init = pool_cls.__init__
    original_allocator_init = HiSparseTokenToKVPoolAllocator.__init__
    original_wrapper_export = MiniMaxSparseKVPool.get_contiguous_buf_infos

    @wraps(original_pool_init)
    def pool_init(self, *args, **kwargs):
        server = get_server_args()
        if (not server.disable_cuda_graph or not server.disable_overlap_schedule
                or not server.disable_radix_cache):
            raise RuntimeError("MiniMax VMM requires eager, no overlap and no radix cache")
        if not (server.disable_prefill_cuda_graph
                or server.cuda_graph_backend_prefill == "disabled"):
            raise RuntimeError("MiniMax VMM requires prefill CUDA Graph to be disabled")
        if server.disaggregation_mode != "null" or server.enable_hierarchical_cache:
            raise RuntimeError("MiniMax VMM does not support PD or HiCache")
        original_pool_init(self, *args, **kwargs)
        if not hasattr(self, "adaptive_vmm"):
            raise RuntimeError("MiniMax VMM normal backing hook was not used")

    def create_buffers(self):
        if (self.dtype != torch.bfloat16 or self.store_dtype != torch.bfloat16
                or self.kv_cache_layout != "nhd" or self.is_quantized_kv_cache
                or self.enable_custom_mem_pool or self.post_capture_active
                or self.size % self.page_size):
            raise RuntimeError("MiniMax VMM requires aligned, ordinary BF16 NHD planes")
        device = torch.device(self.device)
        if device.type != "cuda":
            raise RuntimeError("MiniMax VMM requires CUDA")
        k_shape, v_shape = self._kv_buffer_shapes()
        shapes = [tuple(k_shape)] * self.layer_num + [tuple(v_shape)] * self.layer_num
        if any(len(shape) != 3 or shape[0] != self.size + self.page_size for shape in shapes):
            raise RuntimeError("Unexpected MiniMax BF16 plane shape")
        planes = []
        offset = 0
        for shape in shapes:
            row = prod(shape[1:]) * self.store_dtype.itemsize
            planes.append((offset // CHUNK_BYTES, row))
            length = shape[0] * row
            offset += (length + CHUNK_BYTES - 1) // CHUNK_BYTES * CHUNK_BYTES
        backing = Backing(offset, CHUNK_BYTES, self.device)
        self.adaptive_vmm = PhysicalPages(
            backing, self.page_size, self.size // self.page_size, planes,
            lambda: torch.cuda.synchronize(self.device),
            fast_decode=os.environ.get("EKV_VMM_FAST_DECODE") == "1",
        )
        raw = backing.raw
        views = [
            raw.narrow(0, start * CHUNK_BYTES, shape[0] * row)
            .view(self.store_dtype).view(shape)
            for (start, row), shape in zip(planes, shapes)
        ]
        self.k_buffer = views[:self.layer_num]
        self.v_buffer = views[self.layer_num:]

    @wraps(original_allocator_init)
    def allocator_init(self, *args, **kwargs):
        original_allocator_init(self, *args, **kwargs)
        tracker = getattr(getattr(self._kvcache, "main_pool", None), "adaptive_vmm", None)
        if tracker is not None:
            physical = self.hisparse_attn_allocator
            if type(physical) is not PagedTokenToKVPoolAllocator:
                raise RuntimeError("Unsupported MiniMax physical allocator")
            _attach_allocator(physical, tracker)

    def reject_export(*args, **kwargs):
        raise RuntimeError("Export of partially backed MiniMax VMM planes is unsupported")

    @wraps(original_wrapper_export)
    def wrapper_export(self, *args, **kwargs):
        if hasattr(self.main_pool, "adaptive_vmm"):
            return reject_export()
        return original_wrapper_export(self, *args, **kwargs)

    pool_cls.__init__ = pool_init
    pool_cls._create_buffers_normal = create_buffers
    pool_cls.get_contiguous_buf_infos = reject_export
    MiniMaxSparseKVPool.get_contiguous_buf_infos = wrapper_export
    HiSparseTokenToKVPoolAllocator.__init__ = allocator_init
    pool_cls._adaptive_vmm_installed = True
