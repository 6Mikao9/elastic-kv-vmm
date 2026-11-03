"""Isolated single-GPU VMM pressure fixture, with synthetic 2 MiB slots.

Run each policy in a fresh process on an otherwise idle, explicitly selected
GPU. This tests the mechanism. Model prefill headroom requires separate calibration.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


MIB = 1 << 20
GIB = 1 << 30
CHUNK = 2 * MIB
PENDING = range(512, 1536)


def run(policy, report):
    import numpy as np
    import pynvml
    import torch

    from elastic_kv import activation
    from elastic_kv.backing import Backing

    cuda = torch.cuda
    if cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one isolated GPU to this fixture")
    cuda.set_device(0)
    pynvml.nvmlInit()
    backing = ballast = raw = cached = None
    try:
        uuid = os.environ.get("BENCH_GPU_UUID") or getattr(cuda.get_device_properties(0), "uuid", None)
        if uuid is None:
            raise RuntimeError("Set BENCH_GPU_UUID to the exposed CUDA GPU UUID")
        nvml = pynvml.nvmlDeviceGetHandleByUUID(str(uuid))

        def sample(name):
            free, total = cuda.mem_get_info(0)
            report["samples"][name] = {
                "driver_free_bytes": free, "driver_total_bytes": total,
                "torch_allocated_bytes": cuda.memory_allocated(0),
                "torch_reserved_bytes": cuda.memory_reserved(0),
                "nvml_used_bytes": int(pynvml.nvmlDeviceGetMemoryInfo(nvml).used),
            }
            return free

        sample("initial")
        backing = Backing(4 * GIB, CHUNK, 0)
        backing.ensure(range(256))
        raw = backing.raw
        raw[0].fill_(17)
        raw[CHUNK].fill_(29)
        cuda.synchronize(0)
        cached = torch.empty(GIB, dtype=torch.uint8, device="cuda:0")
        cached = None
        free = sample("warm_and_cached")
        # Leave a little room for allocator rounding, with one bounded allocation.
        ballast_bytes = (free - GIB - 32 * MIB) // CHUNK * CHUNK
        if ballast_bytes <= GIB:
            raise RuntimeError("Insufficient clean VRAM to construct pressure fixture")
        ballast = torch.empty(ballast_bytes, dtype=torch.uint8, device="cuda:0")
        report["ballast_bytes"] = ballast_bytes
        free = sample("before_policy")
        cached_bytes = cuda.memory_reserved(0) - cuda.memory_allocated(0)
        report["torch_inactive_before_bytes"] = cached_bytes
        if cached_bytes < GIB or not GIB <= free < 2 * GIB:
            raise RuntimeError("Fixture did not establish the required driver and cache pressure")

        live = np.zeros(2048, dtype=np.bool_)
        live[:2] = True
        ctrl = SimpleNamespace(pool=SimpleNamespace(device=0), arenas=[backing],
                               backed=backing.backed_bytes, live=live, chunks_for=lambda ids: ids,
                               slot_bytes=CHUNK, chunk_bytes=CHUNK, counters=report["counters"])
        activation._POLICY = policy
        activation._HEADROOM = 128 * MIB
        activation._CUDA = cuda
        activation._CONTEXT.prefill = True
        activation.before_admit(ctrl, [(PENDING.start, PENDING.stop)])
        free_after = sample("after_policy")
        report["backed_after_policy_bytes"] = backing.backed_bytes
        report["sentinels_before_ensure"] = [raw[0].item(), raw[CHUNK].item()]
        if report["sentinels_before_ensure"] != [17, 29]:
            raise RuntimeError("Live VMM sentinel changed after policy")
        try:
            report["committed_chunks"] = backing.ensure(PENDING)
        except RuntimeError as exc:
            error = str(exc)
            if policy != "off" or "cuMemCreate" not in error or not (
                "CUDA_ERROR_OUT_OF_MEMORY" in error or "out of memory" in error.lower()
            ):
                raise
            report["outcome"] = "expected_native_oom"
            report["native_error"] = error
        else:
            if policy == "off":
                raise RuntimeError("Control unexpectedly mapped the pending 2 GiB")
            report["sentinels_after_ensure"] = [raw[0].item(), raw[CHUNK].item()]
            if (report["sentinels_after_ensure"] != [17, 29]
                    or report["committed_chunks"] != len(PENDING)
                    or report["counters"].get("activation_trim_bytes", 0) <= 0
                    or report["counters"].get("activation_flush_freed_bytes", 0) <= 0
                    or free_after < report["pending_bytes"]):
                raise RuntimeError("Pressure policy failed the mapping or live-data invariant")
            report["outcome"] = "mapped"
        report["backing_metrics"] = backing.metrics
        sample("after_ensure")
    finally:
        raw = cached = ballast = None
        try:
            if backing is not None:
                backing.close()
        finally:
            try:
                cuda.empty_cache()
                if "initial" in report["samples"]:
                    sample("after_cleanup")
            finally:
                pynvml.nvmlShutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, choices=("off", "both"))
    parser.add_argument("--outputpath", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    report = {"fixture": "synthetic_vmm_activation_pressure", "policy": args.policy,
              "chunk_bytes": CHUNK, "pending_bytes": len(PENDING) * CHUNK,
              "headroom_bytes": 128 * MIB, "samples": {}, "counters": {}}
    try:
        run(args.policy, report)
    except Exception as exc:
        report["outcome"] = "failed"
        report["error_type"] = type(exc).__name__
        raise
    finally:
        args.outputpath.parent.mkdir(parents=True, exist_ok=True)
        args.outputpath.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
