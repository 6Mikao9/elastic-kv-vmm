"""Opt-in ablation of MiniMax CUDA QK norm/RoPE fusion."""
import hashlib
import os
from functools import wraps
from pathlib import Path

from torch import cuda


def _contiguous_store(original):
    @wraps(original)
    def store(k, v, k_cache, v_cache, idx_k, idx_k_cache, idx_v, idx_v_cache,
              indices, *, num_kv_heads, head_bytes):
        return original(k.contiguous(), v.contiguous(), k_cache, v_cache,
                        idx_k.contiguous(), idx_k_cache,
                        None if idx_v is None else idx_v.contiguous(),
                        idx_v_cache, indices, num_kv_heads=num_kv_heads,
                        head_bytes=head_bytes)

    return store


def install_minimax_diagnostic_fallbacks():
    enabled = os.environ.get("EKV_DISABLE_FUSED_QKNORM", "0")
    if enabled == "0":
        return
    if enabled != "1":
        raise ValueError("EKV_DISABLE_FUSED_QKNORM must be 0 or 1")

    from sglang.kernels.ops.kvcache import minimax_store_kv_index
    from sglang.srt.models import minimax_m3

    digest = hashlib.sha256(Path(minimax_m3.__file__).read_bytes()).hexdigest()
    if digest != "a60deed8241d8dcf45148af331d61ccdcbf3c1aa1676a2cd4ce9b9d8b9d342fd":
        raise RuntimeError("MiniMax diagnostic source mismatch")
    cls = minimax_m3.MiniMaxM3Attention
    if getattr(cls, "_ekv_qknorm_ablation", False):
        return
    original = cls.__init__

    @wraps(original)
    def initialize(self, *args, **kwargs):
        if cuda.get_device_capability() != (8, 0):
            raise RuntimeError("MiniMax QK norm ablation requires SM80")
        original(self, *args, **kwargs)
        self._use_fused_qknorm_rope = False
        self._combined_qknorm_ok = False
        self._fp8_q_epilogue_ok = False

    store_digest = hashlib.sha256(
        Path(minimax_store_kv_index.__file__).read_bytes()
    ).hexdigest()
    if store_digest != "a78a77be03fd8a809e42950a9fc893ade81c670d4c752654c2158ba199689cc9":
        raise RuntimeError("MiniMax diagnostic cache store source mismatch")
    minimax_store_kv_index.store_kv_index = _contiguous_store(
        minimax_store_kv_index.store_kv_index)
    cls.__init__ = initialize
    cls._ekv_qknorm_ablation = True
