"""Per-rank PyTorch allocator telemetry at MiniMax scheduler idle."""
from functools import wraps
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns


_SCHEDULER_SHA256 = "fc96ed4b680fdf55621a5397f844cfd4cbd30213f44eec8a888b7cce75cafe60"


def install_memory_telemetry(stats_dir):
    """Install before scheduler construction, after VMM idle hooks."""
    import torch
    from sglang.srt.managers import scheduler

    cls = scheduler.Scheduler
    if getattr(cls, "_ekv_allocator_telemetry", False):
        return
    if hashlib.sha256(Path(scheduler.__file__).read_bytes()).hexdigest() != _SCHEDULER_SHA256:
        raise RuntimeError("MiniMax allocator telemetry scheduler source mismatch")
    output = Path(stats_dir)
    output.mkdir(parents=True, exist_ok=True)
    original_idle = cls.on_idle

    @wraps(original_idle)
    def on_idle(self):
        result = original_idle(self)
        if not self.is_fully_idle():
            return result
        parallel = scheduler.get_parallel()
        device = scheduler.get_device().gpu_id
        stats = {
            "memory_allocated_bytes": torch.cuda.memory_allocated(device),
            "memory_reserved_bytes": torch.cuda.memory_reserved(device),
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        if stats == getattr(self, "_ekv_allocator_last_stats", None):
            return result
        path = output / f"allocator-tp-{parallel.tp_rank}.json"
        temporary = path.with_suffix(".tmp")
        sample = dict(stats, tp_rank=parallel.tp_rank, tp_size=parallel.tp_size,
                      boundary="scheduler_idle", timestamp_monotonic_ns=perf_counter_ns())
        temporary.write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        self._ekv_allocator_last_stats = stats
        return result

    cls.on_idle = on_idle
    cls._ekv_allocator_telemetry = True
