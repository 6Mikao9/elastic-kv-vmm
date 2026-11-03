"""Opt-in BF16 projection fallback for MiniMax M3.1 NVFP4 on SM80."""
import os

import torch
from torch.nn import functional as F
from sglang.srt.layers.quantization import fp8
from sglang.srt.layers.quantization.minimax_m3_nvfp4 import MiniMaxM3Nvfp4Config
from sglang.srt.layers.quantization.mxfp8_block_convert import dequant_mxfp8_2d_to_bf16


class Sm80Projection(fp8.Fp8LinearMethod):
    def __init__(self, config):
        if not (config.is_checkpoint_fp8_serialized and config.use_mxfp8
                and config.weight_block_size == [1, 32]):
            raise ValueError("SM80 fallback requires serialized 1x32 MXFP8")
        self.quant_config = config
        # This method loads MXFP8 storage but executes BF16. In particular,
        # MiniMax must not build its native fused MXFP8 QKV/index projection.
        self.use_mxfp8 = False

    def create_weights(
        self, layer, input_size_per_partition, output_partition_sizes,
        input_size, output_size, params_dtype, skip_block_quant_check=False,
        **extra_weight_attrs,
    ):
        self.create_fp8_weight_(
            layer,
            block_quant=True,
            quant_config=self.quant_config,
            use_mxfp8=True,
            output_size_per_partition=sum(output_partition_sizes),
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            weight_loader=extra_weight_attrs.get("weight_loader"),
            is_checkpoint_fp8_serialized=True,
            skip_block_quant_check=skip_block_quant_check,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
        )

    def process_weights_after_loading(self, layer):
        if layer.weight.dtype == torch.bfloat16:
            return
        scale = getattr(layer, "weight_scale_inv", None)
        if (layer.weight.dtype != torch.float8_e4m3fn or scale is None
                or scale.dtype != torch.uint8
                or layer.weight.shape[1] % 32
                or scale.shape != (layer.weight.shape[0], layer.weight.shape[1] // 32)):
            raise ValueError("Invalid serialized MXFP8 weight/scale pair")
        weight = dequant_mxfp8_2d_to_bf16(layer.weight.detach(), scale.detach())
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.register_parameter("weight_scale_inv", None)

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)


def install():
    if os.environ.get("EKV_MINIMAX_SM80_BF16") != "1":
        raise RuntimeError("Set EKV_MINIMAX_SM80_BF16=1 to enable the experiment")
    if getattr(MiniMaxM3Nvfp4Config, "_ekv_sm80_patch", False):
        return
    from sglang.srt.layers.linear import LinearBase

    original_method = MiniMaxM3Nvfp4Config.get_quant_method
    original_capability = MiniMaxM3Nvfp4Config.get_min_capability

    def is_sm80():
        # The parent and spawned workers inherit the same visible CUDA devices.
        return torch.cuda.get_device_capability() == (8, 0)

    def get_quant_method(self, layer, prefix):
        if is_sm80() and isinstance(layer, LinearBase):
            if fp8.is_layer_skipped(prefix, self.ignored_layers,
                                    fused_mapping=self.packed_modules_mapping):
                return fp8.UnquantizedLinearMethod()
            return Sm80Projection(self)
        return original_method(self, layer, prefix)

    def get_min_capability(self):
        return 80 if is_sm80() else original_capability(self)

    MiniMaxM3Nvfp4Config.get_quant_method = get_quant_method
    MiniMaxM3Nvfp4Config.get_min_capability = get_min_capability
    MiniMaxM3Nvfp4Config._ekv_sm80_patch = True
