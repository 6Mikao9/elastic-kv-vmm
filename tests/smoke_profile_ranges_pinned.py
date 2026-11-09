"""Verify profiler binding against the pinned coordinator, with no GPU execution.

Only deliberately failing, pre-kernel calls are covered. This is not a native
swap correctness test or an end-to-end serving profile.
"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path


COORDINATOR_SHA256 = "cfe012ce100ee86dcc5080ca5168d51fd4abd83e8c62ee72d3606b5feafc282e"


def failure(call):
    try:
        call()
    except (AttributeError, AssertionError) as error:
        return {"type": type(error).__name__, "message": str(error)}
    raise AssertionError("Expected a pre-kernel validation failure")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("This CPU-only diagnostic requires CUDA_VISIBLE_DEVICES=''")
    os.environ["EKV_PROFILE_RANGES"] = "1"

    import torch
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from adaptive_sparse import profile_ranges

    source = Path(inspect.getfile(HiSparseCoordinator))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha != COORDINATOR_SHA256:
        raise RuntimeError("Unaudited coordinator: pre-kernel failure paths may differ")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cls = HiSparseCoordinator
    names = ("map_last_loc_to_buffer", "swap_in_selected_blocks")
    originals = {name: getattr(cls, name) for name in names}
    assert not getattr(cls, "_ekv_native_profile_ranges", False)
    indices = torch.tensor([0], dtype=torch.long, device="cpu")
    lengths = torch.tensor([1], dtype=torch.int32, device="cpu")
    blocks = torch.tensor([[0, 1]], dtype=torch.int32, device="cpu")

    def calls():
        # No coordinator __init__, CUDA pools, host staging or kernels.
        unmapped = cls.__new__(cls)
        over_capacity = cls.__new__(cls)
        over_capacity.is_m3_hisparse = True
        over_capacity.device_buffer_size = 128
        return {
            names[0]: lambda: unmapped.map_last_loc_to_buffer(
                lengths, indices, indices, lengths, indices),
            names[1]: lambda: over_capacity.swap_in_selected_blocks(
                indices, lengths, blocks, layer_id=0, sparse_block_size=128),
        }

    before = {name: failure(call) for name, call in calls().items()}
    assert before[names[0]]["type"] == "AttributeError"
    assert before[names[1]]["type"] == "AssertionError"
    assert "selected 256 tokens" in before[names[1]]["message"]
    profile_ranges.install_native_profile_ranges()
    installed = {name: getattr(cls, name) for name in names}
    assert all(installed[name].__wrapped__ is originals[name] for name in names)
    profile_ranges.install_native_profile_ranges()
    assert all(getattr(cls, name) is installed[name] for name in names)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        after = {name: failure(call) for name, call in calls().items()}
    assert after == before
    assert not torch.cuda.is_initialized()
    events = [event for event in profiler.events() if event.name.startswith("ekv/")]
    assert sorted(event.name for event in events) == [
        "ekv/native/map_latest", "ekv/native/swap"]
    assert all(event.cpu_time_total > 0 for event in events)
    trace = args.output_dir / "trace.json"
    profiler.export_chrome_trace(str(trace))
    result = {
        "status": "passed",
        "scope": "Actual pinned SGLang bound calls, profiler ranges and original pre-kernel exceptions; no GPU or successful swap path",
        "torch_version": torch.__version__,
        "coordinator_sha256": source_sha,
        "helper_sha256": hashlib.sha256(Path(profile_ranges.__file__).read_bytes()).hexdigest(),
        "smoke_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        "original_exceptions": before,
        "signatures": {name: str(inspect.signature(originals[name])) for name in names},
        "events": [{"name": event.name, "cpu_time_total_us": event.cpu_time_total}
                   for event in events],
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
