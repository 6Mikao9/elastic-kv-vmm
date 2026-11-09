"""Profile a model-free VMM handoff on one explicitly selected idle A100."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHUNK = 16 << 20
GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ack-idle-gpu", action="store_true", required=True)
    parser.add_argument("--profiler-backend", choices=("torch", "nvtx"), default="torch",
                        help="use Torch profiling (default) or emit NVTX ranges for an external profiler")
    args = parser.parse_args()
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if GPU_UUID.fullmatch(selected) is None:
        parser.error("CUDA_VISIBLE_DEVICES must be one full GPU UUID")
    if any(name.startswith("EKV_GLOBAL_BROKER_") for name in os.environ):
        parser.error("unset EKV_GLOBAL_BROKER_* for this isolated smoke")

    import torch

    from elastic_kv.backing import Backing

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    require("A100" in properties.name and (properties.major, properties.minor) == (8, 0),
            f"expected an A100, got {properties.name}")
    reported_uuid = getattr(properties, "uuid", None)
    if reported_uuid is not None:
        require(str(reported_uuid).lower().removeprefix("gpu-") == selected[4:].lower(),
                f"CUDA device UUID mismatch: {reported_uuid} versus {selected}")

    library = Path(os.environ.get("ELASTIC_VMM_LIBRARY", ROOT / "native/libelastic_vmm.so"))
    library = library.expanduser().resolve()
    sources = [Path(__file__).resolve(), ROOT / "elastic_kv/backing.py",
               ROOT / "native/elastic_vmm.cpp", library]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    args.output_dir.mkdir(parents=True)
    regions = {}
    states = []

    if args.profiler_backend == "torch":
        from torch.profiler import ProfilerActivity, profile, record_function

        profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        mark_region = record_function
    else:
        profiler = nullcontext()
        mark_region = torch.cuda.nvtx.range

    @contextmanager
    def region(name):
        start = time.perf_counter_ns()
        with mark_region(f"vmm_handoff/{name}"):
            yield
        regions.setdefault(name, []).append(time.perf_counter_ns() - start)

    started = time.time()
    with profiler as trace:
        with region("reserve"):
            backing = Backing(4 * CHUNK, chunk_bytes=CHUNK, device=device, library=library)
        graph = None
        try:
            raw = None
            base = backing.base

            def check_state(label, expected):
                require(backing.mapped_chunks == expected,
                        f"{label}: mapped {backing.mapped_chunks}, expected {expected}")
                backed = backing.backed_bytes
                require(backed == len(expected) * CHUNK, f"{label}: backed {backed} bytes")
                require(backing.base == base and (raw is None or raw.data_ptr() == base),
                        f"{label}: virtual address changed")
                states.append({"label": label, "mapped_chunks": sorted(expected), "backed_bytes": backed})

            check_state("reserved", set())
            with region("map"):
                require(backing.ensure([0, 1]) == 2, "initial mapping did not commit two chunks")
                raw = backing.raw
                raw[:CHUNK].fill_(17)
                raw[CHUNK:2 * CHUNK].fill_(31)
                torch.cuda.synchronize(device)
            check_state("mapped", {0, 1})
            with region("grow"):
                require(backing.ensure([2]) == 1, "growth did not commit one chunk")
                raw[2 * CHUNK:3 * CHUNK].fill_(59)
                torch.cuda.synchronize(device)
            check_state("grown", {0, 1, 2})

            indices = torch.tensor([CHUNK], device=device)
            value = torch.tensor([73], dtype=raw.dtype, device=device)
            warmup_stream = torch.cuda.Stream(device=device)
            warmup_stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(warmup_stream):
                raw.index_copy_(0, indices, value)
            torch.cuda.current_stream(device).wait_stream(warmup_stream)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with region("graph_capture"), torch.cuda.graph(graph):
                raw.index_copy_(0, indices, value)

            with region("drop"):
                require(backing.reclaim([1]) == 1, "drop did not reclaim one chunk")
            check_state("dropped", {0, 2})

            def check_retained_content():
                for chunk, expected in ((0, 17), (2, 59)):
                    start = chunk * CHUNK
                    require(bool(raw[start:start + CHUNK].eq(expected).all().item()),
                            f"retained chunk {chunk} changed")

            with region("verify_retained_after_drop"):
                check_retained_content()
            with region("remap"):
                require(backing.ensure([1]) == 1, "remap did not commit one chunk")
                raw[CHUNK:2 * CHUNK].fill_(41)
                torch.cuda.synchronize(device)
            check_state("remapped", {0, 1, 2})
            with region("graph_replay"):
                graph.replay()
                torch.cuda.synchronize(device)
            with region("verify_after_replay"):
                require(raw[CHUNK].item() == 73, "graph did not write to the remapped chunk")
                require(bool(raw[CHUNK + 1:2 * CHUNK].eq(41).all().item()),
                        "graph modified data outside its target")
                check_retained_content()
            check_state("replayed", {0, 1, 2})
            metrics = backing.metrics
            for name, expected in (("ensure_calls", 3), ("ensure_chunks", 4),
                                   ("reclaim_calls", 1), ("reclaim_chunks", 1)):
                require(metrics[name] == expected, f"{name}: {metrics[name]}, expected {expected}")
        finally:
            with region("release"):
                torch.cuda.synchronize(device)
                if graph is not None:
                    graph.reset()
                backing.close()

    result = {
        "status": "passed",
        "scope": "model-free CUDA VMM and CUDA Graph smoke",
        "profiler_backend": args.profiler_backend,
        "gpu": {"uuid": selected, "name": properties.name, "capability": [8, 0]},
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "started_unix_seconds": started,
        "chunk_bytes": CHUNK,
        "reserved_bytes": 4 * CHUNK,
        "source_sha256": hashes,
        "states": states,
        "backing_metrics": metrics,
        "profile_regions": {
            name: {"count": len(samples), "host_elapsed_ns": samples}
            for name, samples in regions.items()
        },
    }
    if args.profiler_backend == "torch":
        trace.export_chrome_trace(str(args.output_dir / "trace.json"))
        result["trace"] = "trace.json"
    else:
        result["external_trace_verified"] = False
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
