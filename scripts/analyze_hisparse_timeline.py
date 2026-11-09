"""Read-only timeline analysis for Kineto Chrome traces from HiSparse profiles."""

import argparse
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.profile_hisparse_live import (
    CUDA_API_CATEGORIES,
    GPU_CATEGORIES,
    TRACE_NAME,
    accumulate_duration,
    interval_union_us,
)


def merge(intervals):
    merged = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def intersection_us(left, right):
    left_index = right_index = 0
    total = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_stop = left[left_index]
        right_start, right_stop = right[right_index]
        total += max(0, min(left_stop, right_stop) - max(left_start, right_start))
        if left_stop <= right_stop:
            left_index += 1
        else:
            right_index += 1
    return total


def launch_deltas(launches, kernels):
    # Chrome GPU event PIDs identify devices, so do not join on PID.
    deltas = []
    unmatched = ambiguous = 0
    for correlation, starts in kernels.items():
        ends = launches.get(correlation, [])
        if not ends:
            unmatched += len(starts)
        elif len(starts) != 1 or len(ends) != 1:
            ambiguous += len(starts)
        else:
            deltas.append(starts[0] - ends[0])
    return {
        "matched_kernel_events": len(deltas),
        "unmatched_kernel_events_with_id": unmatched,
        "ambiguous_kernel_events": ambiguous,
        "negative_delta_events": sum(delta < 0 for delta in deltas),
        "min_us": min(deltas, default=None),
        "max_us": max(deltas, default=None),
        "mean_us": sum(deltas) / len(deltas) if deltas else None,
    }


def analyze_trace(path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        events = json.load(source)["traceEvents"]
    if not isinstance(events, list):
        raise ValueError(f"{path.name}: invalid trace event list")
    categories = defaultdict(list)
    streams = defaultdict(list)
    cpu_ops = []
    zero_events = Counter()
    blocking_sync = {}
    stream_wait = {}
    devices = set()
    gpu_without_device = 0
    launches = defaultdict(list)
    kernels = defaultdict(list)
    launch_processes = set()
    kernels_without_id = 0
    for event in events:
        category = event.get("cat")
        if event.get("ph") != "X" or category not in (
            GPU_CATEGORIES | {"cpu_op", *CUDA_API_CATEGORIES}
        ):
            continue
        start, duration = event["ts"], event["dur"]
        if (type(start) not in (int, float) or type(duration) not in (int, float)
                or not math.isfinite(start) or not math.isfinite(duration) or duration < 0):
            raise ValueError(f"{path.name}: invalid {category} interval")
        stop = start + duration
        if not math.isfinite(stop):
            raise ValueError(f"{path.name}: invalid {category} interval end")
        if not duration:
            zero_events[category] += 1
            continue
        args = event.get("args", {})
        correlation = args.get("correlation")
        has_id = type(correlation) is int and correlation > 0
        if category in GPU_CATEGORIES:
            categories[category].append((start, stop))
            device = args.get("device")
            if type(device) is int and device >= 0:
                devices.add(device)
            else:
                gpu_without_device += 1
            stream = args.get("stream")
            key = str(stream) if type(stream) is int and stream >= 0 else "unknown"
            streams[key].append((start, stop))
            if category == "kernel":
                if has_id:
                    kernels[correlation].append(start)
                else:
                    kernels_without_id += 1
        elif category == "cpu_op":
            cpu_ops.append((start, stop))
        else:
            name = event["name"]
            if "StreamWaitEvent" in name:
                accumulate_duration(stream_wait, name, duration)
            elif re.fullmatch(r"(?:cu|cuda)(?:Ctx|Device|Stream|Event|Thread)Synchronize(?:_v\d+|_ptsz)?", name):
                accumulate_duration(blocking_sync, name, duration)
            if has_id and re.match(r"^(?:cuda|cu)Launch(?:Cooperative)?Kernel(?:Ex)?(?:$|_)", name):
                launches[correlation].append(stop)
                launch_processes.add(event.get("pid"))

    if len(devices) > 1:
        raise ValueError(f"{path.name}: GPU intervals span multiple devices: {sorted(devices)}")
    gpu = merge(interval for intervals in categories.values() for interval in intervals)
    if not gpu:
        raise ValueError(f"{path.name}: missing positive-duration GPU activity")
    origin = gpu[0][0]
    gaps = [(end, next_start) for (_, end), (next_start, _) in zip(gpu, gpu[1:])]
    largest = sorted(gaps, key=lambda gap: gap[1] - gap[0], reverse=True)[:10]
    if len(launch_processes) > 1:
        correlation_report = {"status": "skipped_multiple_launch_processes"}
    else:
        correlation_report = launch_deltas(launches, kernels)
    correlation_report["kernel_events_without_id"] = kernels_without_id
    return {
        "trace": path.name,
        "cuda_device_from_trace": next(iter(devices), None),
        "gpu_events_without_device_metadata": gpu_without_device,
        "gpu_envelope_us": gpu[-1][1] - origin,
        "gpu_activity_union_us": sum(stop - start for start, stop in gpu),
        "interior_gpu_idle_us": sum(stop - start for start, stop in gaps),
        "interior_gpu_gap_count": len(gaps),
        "max_interior_gpu_gap_us": max((stop - start for start, stop in gaps), default=0),
        "largest_interior_gpu_gaps": [
            {"start_from_first_gpu_us": start - origin, "duration_us": stop - start}
            for start, stop in largest
        ],
        "gpu_activity_union_by_category_us": {
            key: interval_union_us(intervals) for key, intervals in categories.items()
        },
        "gpu_activity_union_by_stream_us": {
            key: interval_union_us(intervals) for key, intervals in streams.items()
        },
        "recorded_cpu_op_overlap_with_gpu_us": intersection_us(merge(cpu_ops), gpu),
        "blocking_sync_cpu_calls_by_name": blocking_sync,
        "stream_wait_event_cpu_enqueue_calls_by_name": stream_wait,
        "zero_duration_events_by_category": dict(zero_events),
        "correlated_kernel_start_minus_launch_end": correlation_report,
    }


def analyze_path(path, *, expected_tp_size=None):
    if expected_tp_size is not None and expected_tp_size <= 0:
        raise ValueError("expected TP size must be positive")
    if path.is_file():
        traces = [path]
    else:
        traces = sorted(path.glob("*.trace.json.gz"))
    if not traces:
        raise ValueError(f"{path}: no compressed Chrome traces found")
    ranks = {}
    for trace in traces:
        match = TRACE_NAME.fullmatch(trace.name)
        rank = match[1] if match else None
        if rank is None and len(traces) > 1:
            raise ValueError(f"Unexpected trace filename: {trace.name}")
        key = rank if rank is not None else "unknown"
        if key in ranks:
            raise ValueError(f"Duplicate TP rank {key} in {path}")
        ranks[key] = trace
    if expected_tp_size is not None:
        expected = {str(rank) for rank in range(expected_tp_size)}
        if ranks.keys() != expected:
            raise ValueError(f"Expected TP ranks 0 through {expected_tp_size - 1}, "
                             f"found {', '.join(sorted(ranks))}")
    return {
        "ranks": {rank: analyze_trace(trace) for rank, trace in ranks.items()},
        "limitations": [
            "The envelope spans first through last positive GPU activity on each rank. "
            "It gives no startup or shutdown timing.",
            "Idle and gap metrics measure absence of recorded activity inside this trace. "
            "Activity from missing events or other processes is invisible, so they cannot verify "
            "physical device idleness or wasted cycles.",
            "The GPU union assumes one device per trace. The device field and missing-metadata "
            "count refer to positive-duration GPU events.",
            "CPU-op overlap measures recorded CPU wall scopes, not CPU utilization. "
            "Synchronization and stream-wait durations measure CPU calls.",
            "Category and stream unions can overlap each other. The unknown stream "
            "bucket combines activities lacking stream metadata.",
            "Launch deltas use unique explicit IDs for direct kernel launch APIs within "
            "one trace. They are signed temporal observations, not GPU stall or critical-path costs.",
            "Compare treatment deltas only across matched workloads and capture settings.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_or_directory", type=Path)
    parser.add_argument("--expected-tp-size", type=int,
                        help="require the exact TP rank set from 0 through size minus one")
    args = parser.parse_args()
    try:
        report = analyze_path(args.trace_or_directory, expected_tp_size=args.expected_tp_size)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
