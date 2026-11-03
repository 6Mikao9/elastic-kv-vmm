"""Exercise actual MiniMax BF16 pool views, CUDA backing and paged allocation."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

from adaptive_sparse.minimax_vmm import CHUNK_BYTES, _attach_allocator, install_minimax_vmm


def main():
    from sglang.srt import runtime_context
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseMHAMainPool

    page = 128
    dtype = torch.bfloat16
    device = "cuda:0"
    server = SimpleNamespace(
        disable_cuda_graph=True, disable_overlap_schedule=True,
        disable_radix_cache=True, disaggregation_mode="null",
        enable_hierarchical_cache=False,
    )
    with patch.object(runtime_context, "get_server_args", return_value=server):
        install_minimax_vmm()
        pool = HiSparseMHAMainPool(
            size=page * 66, page_size=page, dtype=dtype, head_num=1,
            head_dim=128, layer_num=2, device=device, enable_memory_saver=False,
        )
    allocator = PagedTokenToKVPoolAllocator(page * 66, page, dtype, device, pool, False)
    tracker = pool.adaptive_vmm
    _attach_allocator(allocator, tracker)
    planes = (*pool.k_buffer, *pool.v_buffer)
    pointers = [tensor.data_ptr() for tensor in planes]
    sink_bytes = len(planes) * CHUNK_BYTES
    assert tracker.backing.backed_bytes == sink_bytes
    assert all(torch.count_nonzero(tensor[:page]).item() == 0 for tensor in planes)
    slots = allocator.alloc(page * 65)
    assert slots.numel() == page * 65 and int(slots[0]) == page
    assert tracker.backing.backed_bytes == 2 * sink_bytes
    for index, tensor in enumerate(planes):
        tensor[0].fill_(index + 1)
        tensor[slots] = index + 11

    tail = allocator.alloc_decode(
        torch.tensor([2], device=device), torch.tensor([2]), slots[:1],
    )
    assert tail.tolist() == [page + 1]
    assert tracker.stats()["owned_pages"] == 65

    allocator.free_group_begin()
    allocator.free(slots[page:])
    assert tracker.stats()["owned_pages"] == 65
    allocator.free_group_end()
    assert tracker.stats()["owned_pages"] == 1
    assert tracker.trim_quiescent() == sink_bytes
    assert tracker.backing.backed_bytes == sink_bytes
    for index, tensor in enumerate(planes):
        assert torch.all(tensor[0] == index + 1)
        assert torch.all(tensor[slots[:page]] == index + 11)

    reused = allocator.alloc(page * 64)
    assert torch.equal(reused, slots[page:])
    assert tracker.backing.backed_bytes == 2 * sink_bytes
    for index, tensor in enumerate(planes):
        tensor[reused] = index + 21
        assert torch.all(tensor[reused] == index + 21)
        assert torch.all(tensor[slots[:page]] == index + 11)
    assert [tensor.data_ptr() for tensor in planes] == pointers
    allocator.free(reused)
    allocator.free(slots[:page])
    assert tracker.trim_quiescent() == sink_bytes
    assert tracker.pages == {0}
    print(json.dumps({
        "status": "complete", "planes": len(planes), "chunk_bytes": CHUNK_BYTES,
        "sink_backed_bytes": sink_bytes, "peak_backed_bytes": 2 * sink_bytes,
        "reclaimed_bytes_per_trim": sink_bytes,
    }))


if __name__ == "__main__":
    main()
