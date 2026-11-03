"""Check the native MHA swap kernel against standalone and VMM tensor planes."""

import json

import torch

from elastic_kv.backing import Backing
from sglang.kernels.ops.kvcache.hisparse import load_blocks_to_device_buffer_mha


DEVICE = "cuda:0"
CHUNK = 2 << 20
BLOCK = 128
HOT = 256
WIDTH = 128


def check(planes):
    results = []
    for layer in range(2):
        device_k, device_v = planes[layer], planes[layer + 2]
        # Distinct token, dimension, layer and K/V values represented exactly in BF16.
        host = [(torch.arange(3 * BLOCK)[:, None, None] % 43
                 + torch.arange(WIDTH)[None, None, :] % 13
                 + layer * 64 + plane * 128).to(torch.bfloat16).pin_memory()
                for plane in range(2)]
        device = (device_k, device_v)
        locs = torch.arange(1, HOT + 2, dtype=torch.int32, device=DEVICE).view(1, -1)
        # Force an old-block miss. Seed hot storage with a different old block.
        tokens = torch.full((1, HOT + 1), -1, dtype=torch.int32, device=DEVICE)
        tokens[0, :BLOCK] = torch.arange(BLOCK, 2 * BLOCK, device=DEVICE)
        for dst, src in zip(device, host):
            dst[1:BLOCK + 1].copy_(src[BLOCK:2 * BLOCK], non_blocking=True)
        lru = torch.arange(HOT, dtype=torch.int16, device=DEVICE).view(1, -1)
        out = torch.empty((1, 2 * BLOCK), dtype=torch.int32, device=DEVICE)
        for newest in range(2 * BLOCK, 2 * BLOCK + 3):
            for dst, src in zip(device, host):
                if newest > 2 * BLOCK:
                    # Back up the previous newest before reusing its dedicated slot.
                    src[newest - 1].copy_(dst[HOT + 1], non_blocking=True)
                dst[HOT + 1].copy_(src[newest], non_blocking=True)
            torch.cuda.synchronize()
            load_blocks_to_device_buffer_mha(
                top_k_blocks=torch.tensor([[0, 2]], dtype=torch.int32, device=DEVICE),
                device_buffer_tokens=tokens,
                host_cache_locs=torch.arange(3 * BLOCK, dtype=torch.int64,
                                             device=DEVICE).view(1, -1),
                device_buffer_locs=locs, host_cache_k=host[0], host_cache_v=host[1],
                device_buffer_k=device_k, device_buffer_v=device_v,
                top_k_device_locs=out,
                req_pool_indices=torch.tensor([0], dtype=torch.int64, device=DEVICE),
                seq_lens=torch.tensor([newest + 1], dtype=torch.int32, device=DEVICE),
                lru_slots=lru, item_size_bytes=WIDTH * 2, hot_buffer_size=HOT,
                sparse_block_size=BLOCK,
                num_real_reqs=torch.tensor([1], dtype=torch.int32, device=DEVICE),
            )
            torch.cuda.synchronize()
            valid = BLOCK + newest - 2 * BLOCK + 1
            selected = out[0, :valid].long()
            assert (selected > 0).all().item()
            assert selected[-1].item() == HOT + 1
            assert (out[0, valid:] == -1).all().item()
            for dst, src in zip(device, host):
                expected = torch.cat((src[:BLOCK], src[2 * BLOCK:newest + 1]))
                assert torch.equal(dst[selected].cpu(), expected), (layer, newest)
            results.append(selected.tolist())
    return results


def main():
    shape = (HOT + 2, 1, WIDTH)
    stock = [torch.zeros(shape, dtype=torch.bfloat16, device=DEVICE) for _ in range(4)]
    backing = Backing(4 * CHUNK, CHUNK, DEVICE)
    backing.ensure(range(4))
    raw = backing.raw
    length = (HOT + 2) * WIDTH * 2
    vmm = [raw.narrow(0, plane * CHUNK, length).view(torch.bfloat16).view(shape)
           for plane in range(4)]
    assert check(stock) == check(vmm)
    print(json.dumps({"status": "complete", "layouts": ["stock", "vmm"],
                      "layers": 2, "append_steps": 2, "vmm_backed_bytes": backing.backed_bytes}))


if __name__ == "__main__":
    main()
