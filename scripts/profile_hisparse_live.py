"""Capture and inspect real SGLang CPU and CUDA traces for one HiSparse phase."""

import gzip
import json
import logging
import math
import re
import urllib.request
from collections import Counter
from pathlib import Path


TRACE_NAME = re.compile(r".+-TP-(\d+)(?:-(?:DP|PP|EP)-\d+)*\.trace\.json\.gz")
GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
CUDA_API_CATEGORIES = ("cuda_driver", "cuda_runtime")
WAIT_CALLS = re.compile(r"Synchronize|StreamWaitEvent")
DEFAULT_OUTPUT_TOKEN_CAP = 16
MAX_OUTPUT_TOKEN_CAP = 32


def interval_union_us(intervals):
    end = -math.inf
    total = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(end, start))
        end = max(end, stop)
    return total


def inspect_trace(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        events = json.load(source)["traceEvents"]
    if not isinstance(events, list):
        raise ValueError(f"{path.name}: invalid trace event list")
    counts = Counter()
    kernels = {}
    waits = {}
    cuda_apis = {category: {} for category in CUDA_API_CATEGORIES}
    devices = set()
    intervals = []
    memcpy = {}
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        if category not in GPU_CATEGORIES and category != "cpu_op" and category not in cuda_apis:
            continue
        name = event["name"]
        start, duration = event["ts"], event["dur"]
        if (type(start) not in (int, float) or type(duration) not in (int, float)
                or not math.isfinite(start) or not math.isfinite(duration) or duration < 0):
            raise ValueError(f"{path.name}: invalid {category} interval")
        if duration > 0:
            counts[category] += 1
        if category in GPU_CATEGORIES:
            intervals.append((start, start + duration))
            device = event.get("args", {}).get("device")
            if type(device) is int and device >= 0:
                devices.add(device)
        if category == "kernel":
            accumulate_duration(kernels, name, duration)
        if category in cuda_apis:
            accumulate_duration(cuda_apis[category], name, duration)
            if WAIT_CALLS.search(name):
                accumulate_duration(waits, name, duration)
        if category == "gpu_memcpy":
            entry = memcpy.setdefault(name, {"events": 0, "events_with_bytes": 0,
                                             "reported_bytes": 0})
            entry["events"] += 1
            args = event.get("args", {})
            size = args.get("bytes", args.get("Bytes"))
            if type(size) is int and size >= 0:
                entry["events_with_bytes"] += 1
                entry["reported_bytes"] += size
    if not counts["cpu_op"] or not counts["kernel"]:
        raise ValueError(f"{path.name}: missing positive-duration CPU ops or CUDA kernels")
    if len(devices) > 1:
        raise ValueError(f"{path.name}: GPU intervals span multiple devices: {sorted(devices)}")
    return {
        "trace": path.name,
        "cpu_op_events": counts["cpu_op"],
        "cuda_kernel_events": counts["kernel"],
        "gpu_activity_union_us": interval_union_us(intervals),
        "gpu_activity_sum_us": sum(stop - start for start, stop in intervals),
        "cuda_device_from_trace": next(iter(devices), None),
        "cuda_kernels_by_name": kernels,
        "cuda_api_cpu_durations_by_category_and_name": cuda_apis,
        "cuda_wait_api_cpu_durations_by_name": waits,
        "explicit_gpu_memcpy": memcpy,
    }


def accumulate_duration(entries, name, duration):
    entry = entries.setdefault(name, {"count": 0, "sum_us": 0, "max_us": 0})
    entry["count"] += 1
    entry["sum_us"] += duration
    entry["max_us"] = max(entry["max_us"], duration)


def inspect_traces(directory):
    ranks = {}
    for path in sorted(directory.glob("*.trace.json.gz")):
        match = TRACE_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Unexpected trace filename: {path.name}")
        rank = int(match[1])
        if rank in ranks:
            raise ValueError(f"Duplicate TP rank {rank} in {directory}")
        ranks[rank] = path
    if ranks.keys() != {0, 1, 2, 3}:
        raise ValueError(f"Expected TP ranks 0, 1, 2, 3, found {sorted(ranks)}")
    return {str(rank): inspect_trace(path) for rank, path in sorted(ranks.items())}


def control_profile(opener, base_url, endpoint, payload=None):
    request = urllib.request.Request(
        base_url + "/" + endpoint,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with opener.open(request, timeout=180) as response:
        body = response.read()
    if body:
        try:
            status = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return  # The pinned API can return a successful plain-text response.
        if isinstance(status, dict) and status.get("success") is False:
            raise RuntimeError(f"{endpoint} rejected: {status.get('message', 'unknown error')}")


def capture(base_url, output, workload, opener, output_token_cap=DEFAULT_OUTPUT_TOKEN_CAP):
    from benchmarks.client import run_requests, write_json
    from benchmarks.hisparse_client import WORKLOADS, spec, verify

    trace_dir = (Path(output) / "profile_traces").resolve()
    trace_dir.mkdir()
    concurrency, shapes = WORKLOADS[workload]
    specs = [spec(f"measured/{workload}/0", index, inputs, min(outputs, output_token_cap))
             for index, (inputs, outputs) in enumerate(shapes)]

    def phase():
        records, _, _ = run_requests(base_url + "/generate", specs, concurrency,
                                     timeout=180, stream=True)
        return [{key: verify(record)[key] for key in (
            "request_id", "prompt_sha256", "input_tokens", "output_tokens",
            "requested_output_tokens",
        )} for record in records]

    warmup = phase()
    try:
        control_profile(opener, base_url, "start_profile", {
            "output_dir": str(trace_dir),
            "activities": ["CPU", "GPU"],
            "merge_profiles": False,
            "with_stack": False,
            "record_shapes": False,
        })
        captured = phase()
    except BaseException:
        # Start can time out after the server has already armed its profiler.
        try:
            control_profile(opener, base_url, "stop_profile")
        except Exception as cleanup_error:
            logging.getLogger(__name__).warning("Profiler cleanup also failed: %s", cleanup_error)
        raise
    control_profile(opener, base_url, "stop_profile")
    if warmup != captured:
        raise ValueError("Profile request shapes or prompts differ from warmup")
    report = {
        "status": "complete",
        "workload": workload,
        "concurrency": concurrency,
        "output_token_cap": output_token_cap,
        "requests": captured,
        "ranks": inspect_traces(trace_dir),
        "limitations": (
            "GPU union assumes one CUDA device per rank, checked when device metadata is present. "
            "CUDA API durations describe observed CPU calls only. An empty API category "
            "does not establish that VMM operations were absent. "
            "Wait API durations measure CPU calls, including stream-wait enqueues, and cannot "
            "measure GPU stall time. Explicit CUDA memcpy events "
            "report bytes only when provided by the trace. HiSparse kernels can read "
            "mapped host memory without a CUDA memcpy event, so these are not total H2D bytes."
        ),
    }
    write_json(Path(output) / "profile_summary.json", report)
    return report
