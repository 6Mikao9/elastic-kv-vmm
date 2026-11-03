"""Bounded decode boundary capture for the audited eager MiniMax runtime."""

from functools import wraps
import hashlib
import json
from pathlib import Path


MODEL_SHA256 = "a60deed8241d8dcf45148af331d61ccdcbf3c1aa1676a2cd4ce9b9d8b9d342fd"
SCHEDULER_SHA256 = "fc96ed4b680fdf55621a5397f844cfd4cbd30213f44eec8a888b7cce75cafe60"


def install_layer_trace(output_dir, max_steps=24, max_bytes=48 << 20):
    """Install before worker startup in each spawned process."""
    import torch
    from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank
    from sglang.srt.layers.layer_boundary.residual import stream as residual_stream
    from sglang.srt.managers import scheduler
    from sglang.srt.models import minimax_m3 as model

    for module, expected in ((model, MODEL_SHA256), (scheduler, SCHEDULER_SHA256),
                             (residual_stream, "03528f6343f3afdab14e6a523f6bfa304c63143104f13e43a6b7fb205f2f4e2a")):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError("unaudited MiniMax layer trace source")
    if getattr(model.MiniMaxM3Model.forward, "_ekv_trace", False):
        raise RuntimeError("layer trace already installed")
    if max_steps < 1 or max_bytes < 1:
        raise ValueError("positive trace limits required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    allowed = {length + step for length in (128, 8192) for step in (1, 2, 3)}
    original_model = model.MiniMaxM3Model.forward
    original_layer = model.MiniMaxM3DecoderLayer.forward
    original_idle = scheduler.Scheduler.on_idle
    active = None
    pending = []
    captured = 0
    buffered_bytes = 0
    episode = 0
    previous_length = None

    def snapshot(tensor):
        nonlocal buffered_bytes
        if isinstance(tensor, residual_stream.OwedOutput):
            tensor = tensor.contribution.value
        if tensor is None:
            return None
        size = tensor.numel() * tensor.element_size()
        if buffered_bytes + size > max_bytes:
            raise RuntimeError("MiniMax trace reached its GPU clone byte limit")
        buffered_bytes += size
        return tensor.detach().clone()

    @wraps(original_model)
    def model_forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        nonlocal active, captured, episode, previous_length
        lengths = forward_batch.seq_lens_cpu
        # The multimodal wrapper may pass precomputed embeddings and None here.
        # Diagnostics use the scheduler's IDs without changing model arguments.
        diagnostic_ids = input_ids if input_ids is not None else forward_batch.input_ids
        if (captured >= max_steps or not forward_batch.forward_mode.is_decode()
                or diagnostic_ids is None or positions is None or diagnostic_ids.numel() != 1
                or lengths is None or lengths.device.type != "cpu"
                or lengths.numel() != 1):
            return original_model(self, input_ids, positions, forward_batch, *args, **kwargs)
        length = int(lengths[0])
        if length in (129, 8193) and previous_length != length:
            episode += 1
        previous_length = length
        if length not in allowed:
            return original_model(self, input_ids, positions, forward_batch, *args, **kwargs)
        record = {
            "episode": episode, "sequence_length": length,
            "input_ids": snapshot(diagnostic_ids), "positions": snapshot(positions),
            "request_slot": snapshot(forward_batch.req_pool_indices), "layers": [],
        }
        active = record
        try:
            result = original_model(self, input_ids, positions, forward_batch, *args, **kwargs)
            pending.append(record)
            captured += 1
            return result
        finally:
            active = None

    @wraps(original_layer)
    def layer_forward(self, positions, hidden_states, forward_batch, *args, **kwargs):
        if active is None:
            return original_layer(self, positions, hidden_states, forward_batch, *args, **kwargs)
        stream = getattr(forward_batch, "residual_stream", None)
        record = {"hidden_in": snapshot(hidden_states),
                  "written_residual_in": snapshot(getattr(stream, "residual", None))}
        handles = []
        if self.layer_id in (4, 5):
            def attach(module, name):
                def before(_module, inputs, kwargs):
                    value = kwargs.get("hidden_states", inputs[0] if inputs else None)
                    record[f"{name}_in"] = snapshot(value)
                    if name == "experts":
                        routing = kwargs.get("topk_output", inputs[1] if len(inputs) > 1 else None)
                        for key in ("router_logits", "topk_ids", "topk_weights"):
                            record[f"experts_{key}"] = snapshot(getattr(routing, key, None))

                def after(_module, _inputs, _kwargs, value):
                    record[f"{name}_out"] = snapshot(value)

                handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(module.register_forward_hook(after, with_kwargs=True))

            attach(self.self_attn, "attention")
            attach(self.mlp, "mlp")
            experts = getattr(self.mlp, "experts", None)
            if experts is not None:
                attach(experts, "experts")
        try:
            result = original_layer(self, positions, hidden_states, forward_batch, *args, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        stream = getattr(forward_batch, "residual_stream", None)
        record.update(hidden_out=snapshot(result),
                      written_residual_out=snapshot(getattr(stream, "residual", None)))
        active["layers"].append(record)
        return result

    def export_tensor(value, cpu_record, hashes, key):
        if value is not None:
            cpu = value.cpu().contiguous()
            cpu_record[key] = cpu
            hashes[key] = {
                "sha256": hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
                "finite": bool(torch.isfinite(cpu).all()),
                "shape": list(cpu.shape), "dtype": str(cpu.dtype),
            }

    @wraps(original_idle)
    def on_idle(self, *args, **kwargs):
        nonlocal buffered_bytes, previous_length
        if pending and self.is_fully_idle():
            rank = get_tensor_model_parallel_rank()
            for index, record in enumerate(pending, captured - len(pending)):
                stem = f"rank{rank:02d}-step{index:03d}"
                meta = {key: record[key] for key in ("episode", "sequence_length")}
                tensors, hashes = {}, {}
                for key in ("input_ids", "positions", "request_slot"):
                    export_tensor(record[key], tensors, hashes, key)
                for layer, values in enumerate(record["layers"]):
                    for key, value in values.items():
                        export_tensor(value, tensors, hashes, f"layer{layer:02d}/{key}")
                with (output / (stem + ".pt")).open("xb") as handle:
                    torch.save(tensors, handle)
                with (output / (stem + ".json")).open("x", encoding="utf-8") as handle:
                    json.dump({**meta, "rank": rank, "hashes": hashes}, handle, indent=2)
            pending.clear()
            buffered_bytes = 0
            previous_length = None
        return original_idle(self, *args, **kwargs)

    model_forward._ekv_trace = True
    model.MiniMaxM3Model.forward = model_forward
    model.MiniMaxM3DecoderLayer.forward = layer_forward
    scheduler.Scheduler.on_idle = on_idle
