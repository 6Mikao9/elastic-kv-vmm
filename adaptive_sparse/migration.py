"""Synchronous long-request demotion primitives for native MiniMax HiSparse.

The scheduler must pause decode and agree on prepare success across ranks
before committing. A commit failure is fail-stop across the TP group.
"""

from collections import Counter
from dataclasses import dataclass
import os


@dataclass
class PreparedDemotion:
    coordinator: object
    row: int
    logical: object
    physical: object
    committed_len: int
    retained_rows: tuple
    released_rows: tuple
    status: str = "prepared"


def plan_demote_pages(tracker, page_ids, keep_pages):
    """Choose hot pages while minimizing retention of request-exclusive chunks."""
    pages = tuple(page_ids)
    if not 0 < keep_pages <= len(pages) or len(set(pages)) != len(pages):
        raise ValueError("Invalid demotion page ownership")
    if os.environ.get("EKV_CHUNK_AWARE_RETENTION") != "1":
        return pages[:keep_pages], pages[keep_pages:]
    if tracker is None:
        raise RuntimeError("Chunk-aware retention requires a physical page tracker")
    projected = {page: tracker.chunks(page) for page in pages}
    owned = Counter(chunk for chunks in projected.values() for chunk in chunks)
    exclusive = {chunk for chunk, refs in owned.items()
                 if refs == tracker.refs.get(chunk)}
    protected = set()
    remaining = set(pages)
    retained = []
    for _ in range(keep_pages):
        # Prefer a signature with enough remaining pages to reuse its
        # protected chunks. Page ID is the final stable tie break.
        marginal = {page: frozenset(projected[page] & exclusive - protected)
                    for page in remaining}
        supply = Counter(marginal.values())
        page = min(remaining, key=lambda candidate: (
            len(marginal[candidate]), -supply[marginal[candidate]], candidate))
        retained.append(page)
        protected.update(projected[page] & exclusive)
        remaining.remove(page)
    return tuple(retained), tuple(page for page in pages if page in remaining)


def _release_host(coordinator, row):
    indices = coordinator.mem_pool_host.allocated_host_indices(
        coordinator.req_to_host_pool, row, coordinator.req_to_host_pool_allocated_len[row]
    )
    if indices.numel():
        coordinator.mem_pool_host.free(indices)
    coordinator.req_to_host_pool[row].fill_(-1)
    coordinator.req_to_host_pool_allocated_len[row] = 0


def prepare_demote(coordinator, req, synchronize):
    """Stage committed sparse main K/V without changing attention ownership."""
    if not coordinator.is_m3_hisparse or coordinator.spec_swap.enabled:
        raise RuntimeError("Demotion requires native M3 HiSparse without speculation")
    kv = req.kv
    row, length = kv.req_pool_idx, kv.kv_committed_len
    if (row is None or req.hisparse_staging or length != kv.kv_allocated_len
            or length < coordinator.padded_buffer_size):
        raise ValueError("Demotion requires a fully committed long resident request")
    if (int(coordinator.req_device_buffer_size[row])
            or int(coordinator.req_to_host_pool_allocated_len[row])):
        raise RuntimeError("Resident request unexpectedly owns HiSparse cache state")
    if coordinator.padded_buffer_size % coordinator.mem_pool_device.page_size:
        raise RuntimeError("Hot buffer must be page-aligned")
    logical = coordinator.req_to_token_pool.req_to_token[row, :length].clone()
    physical = coordinator.mem_pool_device.translate_loc_from_full_to_hisparse_device(logical)
    if not bool((physical > 0).all()):
        raise RuntimeError("Resident request has missing physical locations")
    page_size = coordinator.mem_pool_device.page_size
    page_starts = physical[::page_size].tolist()
    if any(slot % page_size for slot in page_starts):
        raise RuntimeError("Resident request page starts are misaligned")
    page_ids = [slot // page_size for slot in page_starts]
    if len(page_ids) != len(set(page_ids)):
        raise RuntimeError("Resident request reuses a physical page")
    tracker = getattr(coordinator.mem_pool_device, "adaptive_vmm", None)
    retained, released = plan_demote_pages(
        tracker, page_ids, coordinator.padded_buffer_size // page_size)
    row_by_page = {page: row for row, page in enumerate(page_ids)}
    retained_rows = tuple(row_by_page[page] for page in retained)
    released_rows = tuple(row_by_page[page] for page in released)
    coordinator.wait_for_pending_backup()
    synchronize()
    shadow = getattr(coordinator.token_to_kv_pool_allocator._kvcache, "_ekv_shadow", None)
    if shadow is not None:
        shadow.check_complete(coordinator.mem_pool_device, logical, physical, "demote_source")
    try:
        host = coordinator.mem_pool_host.alloc_paged_token_slots(
            coordinator.req_to_host_pool, coordinator.req_to_host_pool_allocated_len,
            row, 0, length,
        )
        coordinator.mem_pool_host.backup_from_device_all_layer(
            coordinator.mem_pool_device, host, physical, io_backend="kernel",
        )
        synchronize()
    except Exception:
        # If synchronization itself fails, device work may still reference
        # the storage. Propagate to the fail-stop path without freeing it.
        synchronize()
        _release_host(coordinator, row)
        raise
    return PreparedDemotion(coordinator, row, logical, physical, length,
                             retained_rows, released_rows)


def abort_demote(prepared):
    """Undo a prepared transaction after global prepare rejection."""
    if prepared.status != "prepared":
        raise RuntimeError("Only a prepared demotion can be aborted")
    _release_host(prepared.coordinator, prepared.row)
    prepared.status = "aborted"


def commit_demote(prepared, publish_host_backed, synchronize, trim_quiescent=None):
    """Transfer reused pages to the fixed hot buffer and release the remainder.

    publish_host_backed removes the row from resident dispatch. The caller
    must keep scheduling paused until every TP rank confirms completion.
    """
    if prepared.status != "prepared":
        raise RuntimeError("Demotion has already been resolved")
    prepared.status = "committing"
    coordinator = prepared.coordinator
    row = prepared.row
    capacity = coordinator.padded_buffer_size
    import torch

    physical = prepared.physical
    page_size = coordinator.mem_pool_device.page_size
    page_starts = physical[::page_size]
    retained_rows = torch.tensor(prepared.retained_rows, device=physical.device,
                                 dtype=torch.long)
    retained = (page_starts[retained_rows, None]
                + torch.arange(page_size, device=physical.device)).flatten()
    retained = retained.to(dtype=coordinator.req_to_device_buffer.dtype)
    coordinator.req_to_device_buffer[row, :capacity] = retained
    coordinator.req_device_buffer_size[row] = capacity
    # Every long-request lookup starts as a miss. The newly copied host
    # image is authoritative and includes the latest committed token.
    coordinator.req_device_buffer_tokens[:, row, :].fill_(-1)
    coordinator.req_device_buffer_token_locs[:, row, :capacity] = retained
    coordinator.lru_slots[:, row, :].copy_(coordinator._lru_init)
    coordinator.spec_swap.reset(row)
    coordinator._skip_first_backup[row] = True
    mapping = coordinator.mem_pool_device.full_to_hisparse_device_index_mapping
    mapping[prepared.logical] = 0
    synchronize()
    released_rows = torch.tensor(prepared.released_rows, device=physical.device,
                                 dtype=torch.long)
    coordinator.token_to_kv_pool_allocator.free_hisparse_indices(page_starts[released_rows])
    synchronize()
    released_bytes = trim_quiescent() if trim_quiescent is not None else 0
    publish_host_backed(row)
    prepared.status = "committed"
    return released_bytes


@dataclass
class PreparedPromotion:
    coordinator: object
    request: object
    row: int
    logical: object
    destination: object
    backed_before: int
    status: str = "prepared"


def prepare_promote(coordinator, req, synchronize):
    """Copy host-backed history into new physical pages without publishing it."""
    if (not coordinator.is_m3_hisparse or coordinator.spec_swap.enabled
            or coordinator.compress_ratio != 1):
        raise RuntimeError("Promotion requires native M3 HiSparse without speculation")
    kv = req.kv
    row, length = kv.req_pool_idx, kv.kv_committed_len
    if (row is None or req.hisparse_staging or length != kv.kv_allocated_len
            or length <= coordinator.device_buffer_size):
        raise ValueError("Promotion requires a fully committed long host-backed request")
    pool = coordinator.mem_pool_device
    page = pool.page_size
    tracker = getattr(pool, "adaptive_vmm", None)
    if tracker is None:
        raise RuntimeError("Promotion requires the tracked MiniMax VMM main pool")
    if int(coordinator.req_device_buffer_size[row]) != coordinator.padded_buffer_size:
        raise RuntimeError("Unexpected source hot-buffer capacity")
    coordinator.wait_for_pending_backup()
    synchronize()
    logical = coordinator.req_to_token_pool.req_to_token[row, :length].clone()
    host_authoritative = bool(coordinator._skip_first_backup[row])
    host_len = length if host_authoritative else length - 1
    host = coordinator.req_to_host_pool[row, :host_len]
    if (int(coordinator.req_to_host_pool_allocated_len[row]) < host_len
            or not bool((host >= 0).all())):
        raise RuntimeError("Promotion source is missing committed host KV")
    latest = None
    if not host_authoritative:
        latest = coordinator.req_to_device_buffer[row, coordinator.device_buffer_size:][:1]
        if not bool((latest > 0).all()):
            raise RuntimeError("Promotion source is missing the latest device token")

    allocator = coordinator.token_to_kv_pool_allocator
    backed_before = tracker.backing.backed_bytes
    destination = allocator.hisparse_attn_allocator.alloc(-(-length // page) * page)
    if destination is None:
        raise MemoryError("No physical pages available for promotion")
    try:
        for layer in range(pool.start_layer, pool.end_layer):
            coordinator.mem_pool_host.load_to_device_per_layer(
                pool, host, destination[:host_len], layer, io_backend="kernel",
            )
            if latest is not None:
                for buffer in (pool.get_key_buffer(layer), pool.get_value_buffer(layer)):
                    buffer[destination[host_len:host_len + 1]] = buffer[latest]
        synchronize()
        shadow = getattr(allocator._kvcache, "_ekv_shadow", None)
        if shadow is not None:
            shadow.check_complete(pool, logical, destination[:length], "promote_destination")
    except Exception:
        synchronize()
        allocator.free_hisparse_indices(destination)
        tracker.trim_quiescent()
        raise
    return PreparedPromotion(coordinator, req, row, logical, destination, backed_before)


def abort_promote(prepared):
    """Release only the new destination after a rejected prepare vote."""
    if prepared.status != "prepared":
        raise RuntimeError("Only a prepared promotion can be aborted")
    coordinator = prepared.coordinator
    coordinator.token_to_kv_pool_allocator.free_hisparse_indices(prepared.destination)
    coordinator.mem_pool_device.adaptive_vmm.trim_quiescent()
    prepared.status = "aborted"


def commit_promote(prepared, publish_gpu_resident, synchronize):
    """Release the source and publish its prepopulated full-resident mapping."""
    if prepared.status != "prepared":
        raise RuntimeError("Promotion has already been resolved")
    prepared.status = "committing"
    coordinator = prepared.coordinator
    pool = coordinator.mem_pool_device
    # Native cleanup releases host and hot ownership and clears old mappings.
    # It must run before publishing new physical locations.
    coordinator.request_finished(prepared.request)
    pool.full_to_hisparse_device_index_mapping[prepared.logical] = (
        prepared.destination[:len(prepared.logical)])
    synchronize()
    tracker = pool.adaptive_vmm
    tracker.trim_quiescent()
    publish_gpu_resident(prepared.row)
    prepared.status = "committed"
    return max(0, tracker.backing.backed_bytes - prepared.backed_before)
