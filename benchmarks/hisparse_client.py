"""Matched MiniMax HiSparse performance and token-ID probes."""

import argparse
from pathlib import Path
import threading
import time

from benchmarks.client import (
    IDLE_SECONDS, SAMPLE_SECONDS, MemorySampler, distribution, future_hint, generate,
    run_requests, safe_error_message, write_json,
)


WORKLOADS = {
    "mixed12k_unequal": (4, [(128, 64), (12288, 64), (12288, 128), (128, 64)]),
    "short_serial": (1, [(128, 64)] * 4),
    "short_concurrent": (4, [(128, 64)] * 4),
    "long_serial": (1, [(8192, 128)] * 4),
    "long_concurrent": (4, [(8192, 128)] * 4),
    "mixed": (4, [(128, 64), (8192, 128)] * 2),
}


class RankSampler(MemorySampler):
    def __init__(self, uuids):
        import pynvml

        self.nvml = pynvml
        self.nvml.nvmlInit()
        self.samples = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error = None
        self.origin = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        try:
            self.handles = [self.nvml.nvmlDeviceGetHandleByUUID(uuid) for uuid in uuids]
        except BaseException:
            self.nvml.nvmlShutdown()
            raise

    def take(self):
        with self.lock:
            used = [int(self.nvml.nvmlDeviceGetMemoryInfo(handle).used)
                    for handle in self.handles]
            self.samples.append({
                "offset_s": time.perf_counter() - self.origin,
                "rank_used_bytes": used,
                "sum_used_bytes": sum(used),
            })


def spec(name, index, inputs, outputs):
    return {
        "request_id": f"hisparse/{name}/{index}",
        "input_tokens": inputs,
        "requested_output_tokens": outputs,
    }


def verify(record):
    if record["output_tokens"] != record["requested_output_tokens"]:
        raise RuntimeError(f"unexpected completion count for {record['request_id']}")
    return record


def run_phase(
    url,
    name,
    repeat,
    uuids,
    timeout,
    warmup=False,
    forced_replay=False,
    emit_future_hints=False,
):
    concurrency, shapes = WORKLOADS[name]
    key = f"{'warmup' if warmup else 'measured'}/{name}/{repeat}"
    specs = [spec(key, i, *shape) for i, shape in enumerate(shapes)]
    if forced_replay:
        for request in specs:
            request["forced_replay"] = True
    if emit_future_hints:
        for request in specs:
            request["future_hint"] = future_hint(
                request["request_id"], request["requested_output_tokens"]
            )
    sampler = RankSampler(uuids)
    with sampler:
        time.sleep(IDLE_SECONDS)
        sampler.take()
        before = sampler.samples[-1]
        records, started, ended = run_requests(url, specs, concurrency, timeout, stream=True)
        time.sleep(IDLE_SECONDS)
        sampler.take()
        after = sampler.samples[-1]
    begin, end = started - sampler.origin, ended - sampler.origin
    active = [sample for sample in sampler.samples if begin <= sample["offset_s"] <= end]
    if not active:
        raise RuntimeError(f"no active memory samples for {key}")
    for record in records:
        verify(record)
        record["start_offset_s"] = record.pop("_started_perf_s") - sampler.origin
        record.pop("output_ids", None)
    output_tokens = sum(record["output_tokens"] for record in records)
    return {
        "workload": name, "repeat": repeat, "warmup": warmup,
        "concurrency": concurrency, "request_count": len(records),
        "phase_wall_s": ended - started, "output_tokens": output_tokens,
        "aggregate_output_tokens_per_s": output_tokens / (ended - started),
        "request_ttft_s": distribution([record["ttft_s"] for record in records]),
        "request_tpot_s": distribution([record["tpot_s"] for record in records]),
        "request_end_to_end_s": distribution([record["end_to_end_s"] for record in records]),
        "requests": records, "memory": {
            "idle_before": before, "idle_after": after,
            "active_window_s": [begin, end], "active_sample_count": len(active),
            "rank_peak_used_bytes": [max(sample["rank_used_bytes"][rank] for sample in active)
                                     for rank in range(len(uuids))],
            "peak_sum_used_bytes": max(sample["sum_used_bytes"] for sample in active),
            "samples": sampler.samples,
        },
    }


def run_benchmark(
    url,
    output,
    gpu_uuids,
    repeats=3,
    workloads=None,
    timeout=600,
    forced_replay=False,
    warmup_once=False,
    emit_future_hints=False,
):
    """Write performance and serial correctness artifacts to fresh file paths."""
    output = Path(output)
    correctness_path = output.with_name(output.stem + ".correctness.json")
    if output.exists() or correctness_path.exists():
        raise ValueError("choose fresh performance and correctness output paths")
    names = list(WORKLOADS if workloads is None else workloads)
    if not names or len(set(names)) != len(names) or any(name not in WORKLOADS for name in names):
        raise ValueError("require unique workload names")
    if (len(gpu_uuids) != 4 or len(set(gpu_uuids)) != 4
            or any(not uuid.startswith("GPU-") for uuid in gpu_uuids)):
        raise ValueError("supply four distinct GPU UUIDs in tensor-parallel rank order")
    if repeats < 1 or timeout <= 0:
        raise ValueError("require positive repeats and timeout")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "workloads": names, "repeats": repeats, "tp_size": len(gpu_uuids),
        "sample_interval_s": SAMPLE_SECONDS, "idle_pause_s": IDLE_SECONDS,
        "rank_selection": "explicit UUIDs in launcher rank order",
        "memory_scope": "whole-device NVML usage on each selected GPU",
        "forced_replay": forced_replay,
        "future_hints": emit_future_hints,
        "warmup_policy": "once_all_shapes" if warmup_once else "each_repeat",
        "sampling_note": ("Synthetic fixed token replay with trusted native logit processor and output logprob IDs"
                          if forced_replay else "Native greedy sampling, no custom processor"),
    }
    results = {**metadata, "status": "running", "warmups": [], "phases": []}
    correctness = {"status": "pending", "probes": []}
    endpoint = url.rstrip("/")
    if not endpoint.endswith("/generate"):
        endpoint += "/generate"
    stage = "initialization"
    try:
        if warmup_once:
            for name in names:
                stage = f"warmup/{name}/0"
                results["warmups"].append(run_phase(
                    endpoint, name, 0, gpu_uuids, timeout, warmup=True,
                    forced_replay=forced_replay, emit_future_hints=emit_future_hints))
        for repeat in range(repeats):
            for name in names:
                if not warmup_once:
                    stage = f"warmup/{name}/{repeat}"
                    results["warmups"].append(run_phase(
                        endpoint, name, repeat, gpu_uuids, timeout, warmup=True,
                        forced_replay=forced_replay,
                        emit_future_hints=emit_future_hints))
                stage = f"measured/{name}/{repeat}"
                results["phases"].append(run_phase(endpoint, name, repeat, gpu_uuids, timeout,
                                                  forced_replay=forced_replay,
                                                  emit_future_hints=emit_future_hints))
                write_json(output, results)
        results["status"] = "complete"
        correctness["status"] = "running"
        for index, (inputs, outputs) in enumerate(((128, 64), (8192, 128), (8192, 64))):
            probe = spec("probe", index, inputs, outputs)
            if forced_replay:
                probe["forced_replay"] = True
            if emit_future_hints:
                probe["future_hint"] = future_hint(
                    probe["request_id"], probe["requested_output_tokens"]
                )
            stage = probe["request_id"]
            record = verify(generate(endpoint, probe, timeout, stream=False))
            record.pop("_started_perf_s")
            if not record.get("output_ids_complete"):
                raise RuntimeError(f"incomplete output token IDs for {stage}")
            correctness["probes"].append(record)
        correctness["status"] = "complete"
    except BaseException as error:
        target = correctness if results["status"] == "complete" else results
        target.update(status="failed", failed_step=stage, error=safe_error_message(error))
        raise
    finally:
        write_json(output, results)
        write_json(correctness_path, correctness)
    return results, correctness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:31080")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-uuids", nargs=4, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    run_benchmark(args.url, args.output, args.gpu_uuids, args.repeats,
                  args.workloads, args.timeout)


if __name__ == "__main__":
    main()
