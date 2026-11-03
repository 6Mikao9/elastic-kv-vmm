"""Opt-in exact-image capture of inputs and outputs of sparse attention."""

import hashlib
import inspect
from functools import wraps
from pathlib import Path
from threading import local


def install_minimax_attention_capture(output_dir, max_records=32):
    """Install in eager resident workers; GPU snapshots are saved at idle."""
    import torch
    from sglang.srt.layers.attention import minimax_sparse_backend as backend
    from sglang.srt.layers.attention.minimax_sparse_ops import minimax_sparse as ops
    from sglang.srt.managers import scheduler

    for module, expected in (
        (ops, "ed3386669a46a6a3a9478c37a8ff73841f60540fb970540f491a63347419a65f"),
        (backend, "7f8e954d3dea6aa4a76c549a7854f3743b67be49289d578e83524b762623f042"),
        (scheduler, "fc96ed4b680fdf55621a5397f844cfd4cbd30213f44eec8a888b7cce75cafe60"),
    ):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Unaudited MiniMax attention capture source")
    if getattr(ops.flash_decode_with_gqa_share_sparse, "_adaptive_capture", False):
        return
    if max_records <= 0:
        raise ValueError("max_records must be positive")
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    context = local()
    pending = []
    captured = 0
    saved = 0
    original_idle = scheduler.Scheduler.on_idle
    original_forward = backend.MiniMaxSparseAttnBackend.forward_decode
    original_kernel = ops.flash_decode_with_gqa_share_sparse
    signature = inspect.signature(original_kernel)

    @wraps(original_forward)
    def forward(self, q, k, v, layer, forward_batch, *args, **kwargs):
        old = getattr(context, "layer_id", None)
        context.layer_id = layer.layer_id
        try:
            return original_forward(self, q, k, v, layer, forward_batch, *args, **kwargs)
        finally:
            context.layer_id = old

    @wraps(original_kernel)
    def kernel(*args, **kwargs):
        nonlocal captured
        layer_id = getattr(context, "layer_id", None)
        if layer_id not in (4, 5) or captured >= max_records:
            return original_kernel(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        data = bound.arguments
        q = data["q"]
        # This fixture is isolated, no HiSparse and one request at a time.
        if q.shape[0] != 1 or data["hisparse_slots"] is not None:
            return original_kernel(*args, **kwargs)
        # Filter using scheduler's host sequence metadata, when available.
        lengths = getattr(context, "lengths", None)
        if lengths is None or lengths not in (129, 8193):
            return original_kernel(*args, **kwargs)

        topk = data["topk_idx"]
        width = data["block_size"]
        table = data["req_to_token"]
        lengths_gpu = data["seq_lens"]
        rows = data["slot_ids"]
        k_cache, v_cache = data["k_cache"], data["v_cache"]
        if k_cache.ndim != 3 or v_cache.ndim != 3:
            raise RuntimeError("MiniMax attention capture expects resident NHD cache")
        offsets = torch.arange(width, device=topk.device)
        positions = topk.to(torch.int64).unsqueeze(-1) * width + offsets
        logical_valid = (positions >= 0) & (positions < lengths_gpu.reshape(1, -1, 1, 1))
        logical_valid &= positions < table.shape[1]
        safe_rows = rows.to(torch.int64).clamp(0, table.shape[0] - 1)
        slots = table[safe_rows.reshape(1, -1, 1, 1), positions.clamp(0, table.shape[1] - 1)]
        page_size = data["page_size"]
        if page_size >= width and page_size % width == 0:
            block_start = (positions[..., :1]).clamp(0, table.shape[1] - 1)
            slots = table[safe_rows.reshape(1, -1, 1, 1), block_start] + offsets
        physical_valid = (slots > 0) & (slots < min(k_cache.shape[0], v_cache.shape[0]))
        safe_slots = torch.where(logical_valid & physical_valid, slots, 0).to(torch.int64)
        heads = torch.arange(topk.shape[0], device=topk.device).reshape(-1, 1, 1, 1)
        record = {
            "layer_id": layer_id, "capture_index": captured, "host_seq_len": lengths,
            "block_size": width, "page_size": data["page_size"],
            "q_scale": data["q_scale"], "k_scale": data["k_scale"], "v_scale": data["v_scale"],
            "sm_scale": data["sm_scale"],
            "q": q.detach().clone(), "topk": topk.detach().clone(),
            "sink": data["sink"].detach().clone() if data["sink"] is not None else None,
            "seq_lens": lengths_gpu.detach().clone(), "rows": rows.detach().clone(),
            "slots": slots, "logical_valid": logical_valid, "physical_valid": physical_valid,
            "selected_k": k_cache[safe_slots, heads], "selected_v": v_cache[safe_slots, heads],
        }
        output = original_kernel(*args, **kwargs)
        record["output"] = output.detach().clone()
        pending.append(record)
        captured += 1
        return output

    @wraps(original_idle)
    def on_idle(self):
        nonlocal saved
        if pending and self.is_fully_idle():
            rank = scheduler.get_parallel().tp_rank
            while pending:
                record = pending[0]
                cpu_record = {key: value.cpu() if isinstance(value, torch.Tensor) else value
                              for key, value in record.items()}
                path = target_dir / f"tp-{rank}-{saved:04d}.pt"
                tmp = path.with_suffix(".tmp")
                torch.save(cpu_record, tmp)
                tmp.replace(path)
                pending.pop(0)
                saved += 1
        return original_idle(self)

    @wraps(original_forward)
    def capture_forward(self, q, k, v, layer, forward_batch, *args, **kwargs):
        old = getattr(context, "lengths", None)
        lengths = forward_batch.seq_lens_cpu
        context.lengths = int(lengths[0]) if lengths is not None and len(lengths) == 1 else None
        try:
            return forward(self, q, k, v, layer, forward_batch, *args, **kwargs)
        finally:
            context.lengths = old

    kernel._adaptive_capture = True
    ops.flash_decode_with_gqa_share_sparse = kernel
    backend.MiniMaxSparseAttnBackend.forward_decode = capture_forward
    scheduler.Scheduler.on_idle = on_idle
