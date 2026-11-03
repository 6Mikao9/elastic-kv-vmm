"""SM80 test of the real projection adapter with TP-local weight shapes."""
import json

import torch
from torch.nn import functional as F

from adaptive_sparse.minimax_sm80 import MiniMaxM3Nvfp4Config, Sm80Projection, install


def main():
    install()
    config = MiniMaxM3Nvfp4Config()
    assert config.get_min_capability() == 80
    method = Sm80Projection(config)
    assert not method.use_mxfp8  # Native MiniMax QKV/index fusion requires this flag.
    layer = torch.nn.Module()
    layer.tp_group = type("TP", (), {"world_size": 4})()
    with torch.device("cuda"):
        method.create_weights(layer, 128, [16, 16], 512, 32, torch.bfloat16)
    device = layer.weight.device
    generator = torch.Generator(device=device).manual_seed(29)
    fp8 = torch.randn((32, 128), device=device, generator=generator).to(torch.float8_e4m3fn)
    scales = torch.randint(124, 130, (32, 4), device=device, dtype=torch.uint8,
                           generator=generator)
    layer.weight.data.copy_(fp8)
    layer.weight_scale_inv.data.copy_(scales)
    reference = fp8.float() * (2.0 ** (scales.float() - 127)).repeat_interleave(32, dim=1)
    method.process_weights_after_loading(layer)
    assert layer.weight_scale_inv is None
    torch.testing.assert_close(layer.weight.float(), reference, atol=0, rtol=0)

    errors = []
    for shape, zero in (((128,), False), ((3, 2, 128), False), ((2, 128), True)):
        x = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
        if zero:
            x.zero_()
        bias = torch.randn((32,), device=device, dtype=torch.bfloat16, generator=generator)
        actual = method.apply(layer, x, bias)
        expected = F.linear(x.float(), reference, bias.float())
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.01)
        errors.append((actual.float() - expected).abs().max().item())
    print(json.dumps({"status": "passed", "capability": torch.cuda.get_device_capability(),
                      "max_abs_errors": errors}))


if __name__ == "__main__":
    main()
