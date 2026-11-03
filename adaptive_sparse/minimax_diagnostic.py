"""Opt-in eager Top-K contract capture for the audited MiniMax SM80 path."""

import hashlib
import inspect
import json
from functools import wraps
from pathlib import Path
from threading import local


def install_minimax_diagnostic(output_dir, prompt_lengths=(128, 8192), repair_tail=False):
    """Install in each worker. D2H synchronization changes execution timing."""
    from sglang.srt.layers.attention import minimax_sparse_backend as backend
    from sglang.srt.layers.attention.minimax_sparse_ops import minimax_sparse as ops
    from sglang.srt.runtime_context import get_parallel

    for module, expected in (
        (ops, "ed3386669a46a6a3a9478c37a8ff73841f60540fb970540f491a63347419a65f"),
        (backend, "7f8e954d3dea6aa4a76c549a7854f3743b67be49289d578e83524b762623f042"),
    ):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Unaudited MiniMax diagnostic source")
    original_forward = backend.MiniMaxSparseAttnBackend.forward_decode
    original_kernel = ops.flash_decode_with_gqa_share_sparse
    if getattr(original_kernel, "_adaptive_diagnostic", False):
        return
    signature = inspect.signature(original_kernel)
    wanted = {length + step for length in prompt_lengths for step in range(1, 4)}
    context = local()
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    @wraps(original_forward)
    def forward(self, q, k, v, layer, forward_batch, *args, **kwargs):
        previous = getattr(context, "layer_id", None)
        context.layer_id = layer.layer_id
        try:
            return original_forward(self, q, k, v, layer, forward_batch, *args, **kwargs)
        finally:
            context.layer_id = previous

    @wraps(original_kernel)
    def kernel(*args, **kwargs):
        if getattr(context, "layer_id", None) is None:
            return original_kernel(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        seq_lens = values["seq_lens"].tolist()
        if not wanted.intersection(seq_lens):
            return original_kernel(*args, **kwargs)
        rows = values["slot_ids"].tolist()
        topk = values["topk_idx"]
        blocks = topk.tolist()
        width = values["block_size"]
        cache_slots = values["k_cache"].shape[0]
        table = values["req_to_token"]
        hisparse = values["hisparse_slots"]
        physical = hisparse.tolist() if hisparse is not None else None
        repairs = []
        records = []
        for request, (row, length) in enumerate(zip(rows, seq_lens)):
            if length not in wanted:
                continue
            row_valid = 0 <= row < table.shape[0] and 0 < length <= table.shape[1]
            mapping = table[row, :length].tolist() if row_valid and physical is None else None
            num_blocks = (length + width - 1) // width
            for head, head_blocks in enumerate(blocks):
                selected = head_blocks[request]
                violations = [] if row_valid else ["invalid_request_row_or_length"]
                tail = False
                valid_slots = []
                for index, block in enumerate(selected):
                    if block < 0 or block >= num_blocks:
                        if block != -1:
                            violations.append(f"invalid_block:{index}")
                        tail = True
                        if block != -1:
                            repairs.append((head, request, index))
                        continue
                    if tail:
                        violations.append(f"nonpacked_block:{index}")
                        repairs.append((head, request, index))
                        continue
                    start = block * width
                    count = min(width, length - start)
                    if physical is not None:
                        start_slot = index * width
                        slots = physical[head][request][start_slot:start_slot + count]
                    else:
                        slots = mapping[start:start + count] if mapping is not None else []
                    if len(slots) != count or any(not 0 < slot < cache_slots for slot in slots):
                        violations.append(f"invalid_kv_slot:{index}")
                    valid_slots.extend(slots)
                records.append({"layer_id": context.layer_id, "row": row, "seq_len": length,
                                "head": head, "block_ids": selected,
                                "selected_slot_min": min(valid_slots, default=None),
                                "selected_slot_max": max(valid_slots, default=None),
                                "violations": violations})
        target = directory / f"tp-{get_parallel().tp_rank}.jsonl"
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"diagnostic_d2h": True, "tail_repair_enabled": repair_tail,
                                     "hisparse": physical is not None, "records": records}) + "\n")
        if repair_tail and repairs:
            fixed = topk.clone()
            for head, request, index in repairs:
                fixed[head, request, index] = -1
            values["topk_idx"] = fixed
        return original_kernel(*bound.args, **bound.kwargs)

    kernel._adaptive_diagnostic = True
    ops.flash_decode_with_gqa_share_sparse = kernel
    backend.MiniMaxSparseAttnBackend.forward_decode = forward
