"""Read-only SM80 projection check on a real MiniMax checkpoint slice."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn import functional as F

from adaptive_sparse.minimax_sm80 import MiniMaxM3Nvfp4Config, Sm80Projection, install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--rank", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    install()
    weights = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "language_model.model.layers.0.self_attn.index_q_proj."
    loaded = []
    for name in ("weight", "weight_scale_inv"):
        key = prefix + name
        with safe_open(args.model / weights[key], framework="pt", device="cpu") as shard:
            loaded.append(shard.get_slice(key)[args.rank * 128:(args.rank + 1) * 128])
    weight, scale = (value.cuda() for value in loaded)
    assert weight.shape == (128, 6144) and scale.shape == (128, 192)

    method = Sm80Projection(MiniMaxM3Nvfp4Config())
    layer = torch.nn.Module()
    layer.tp_group = type("TP", (), {"world_size": 4})()
    with torch.device("cuda"):
        method.create_weights(layer, 6144, [128], 6144, 512, torch.bfloat16)
    layer.weight.data.copy_(weight)
    layer.weight_scale_inv.data.copy_(scale)
    method.process_weights_after_loading(layer)
    dequantized = weight.float() * torch.exp2(scale.float() - 127).repeat_interleave(32, dim=1)
    torch.testing.assert_close(layer.weight, dequantized.bfloat16(), atol=0, rtol=0)

    generator = torch.Generator(device="cuda").manual_seed(29)
    x = torch.randn((2, 4, 6144), device="cuda", dtype=torch.bfloat16, generator=generator)
    actual = method.apply(layer, x)
    reference = F.linear(x.float(), dequantized.bfloat16().float()).bfloat16()
    torch.testing.assert_close(actual.float(), reference.float(), atol=0.015625, rtol=0.008)
    assert all(torch.equal(actual, method.apply(layer, x)) for _ in range(3))
    print(json.dumps({"rank": args.rank, "status": "passed", "shape": list(weight.shape),
                      "max_abs_error": (actual.float() - reference.float()).abs().max().item()}))


if __name__ == "__main__":
    main()
