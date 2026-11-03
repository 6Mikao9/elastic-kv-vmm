#!/usr/bin/env python3
"""Real SGLang HTTP streaming benchmark; no dataset or tokenizer download.

Run on the GPU host/container so NVML and EKV_STATS_PATH describe the server.
Dependencies: requests, nvidia-ml-py (imported as pynvml).

--output stores normal-request performance. <stem>.correctness.json stores a
separate nonstreaming/logprob pass over exactly the measured prompts. Its time
and memory are never included in performance aggregates. Synthetic input IDs
require a vocabulary of at least 8192 entries; they test execution, not language
quality. Warmups have identical shapes but different prompts from measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests


WORKLOADS = {
    "short_serial": (1, [(128, 64)] * 6),
    "short_concurrent": (8, [(128, 64)] * 8),
    "long_concurrent": (4, [(4096, 128)] * 4),
    "mixed": (8, [(128, 64), (1024, 128)] * 4),
}
STATS_NUMBERS = {
    "chunk_bytes", "logical_capacity_bytes", "backed_bytes", "peak_backed_bytes",
    "live_tokens", "alloc_calls", "free_calls", "index_d2h_ns", "hook_ns",
    "ensure_calls", "ensure_chunks", "ensure_ns", "reclaim_calls",
    "reclaim_chunks", "reclaim_ns", "reclaim_sync_ns", "reserved_bytes",
    "actual_graph_replay_calls", "padded_graph_replay_calls", "graph_capture_calls", "boundary_sync_ns",
    "interval_alloc_calls", "fragmented_alloc_calls", "contiguous_id_view_calls", "id_upload_calls",
    "retain_chunks",
}
SAMPLE_SECONDS = 0.05
IDLE_SECONDS = 0.3


class BenchmarkError(RuntimeError):
    """Messages contain only client-authored descriptions and numeric counts."""


def safe_error_message(exc):
    if isinstance(exc, BenchmarkError):
        return str(exc)
    if isinstance(exc, json.JSONDecodeError):
        return f"invalid JSON at line {exc.lineno}, column {exc.colno}"
    if isinstance(exc, requests.Timeout):
        return "HTTP request timed out"
    if isinstance(exc, requests.RequestException):
        return "HTTP transport failed; inspect server/network logs"
    if isinstance(exc, OSError):
        return f"local I/O failed (errno={exc.errno}); check output and adapter stats access"
    return f"{type(exc).__name__}; inspect server and GPU runtime logs"


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_ids(key, length):
    raw = hashlib.shake_256(("elastic-kv-prompts-v1/" + key).encode()).digest(length * 2)
    return [256 + int.from_bytes(raw[i:i + 2], "little") % 7936
            for i in range(0, len(raw), 2)]


def specifications(workload, repeat, warmup=False):
    return [
        {"request_id": f"{'warmup' if warmup else 'measured'}/{workload}/{repeat}/{i}",
         "input_tokens": shape[0], "requested_output_tokens": shape[1]}
        for i, shape in enumerate(WORKLOADS[workload][1])
    ]


def payload_for(spec, stream):
    body = {
        "input_ids": prompt_ids(spec["request_id"], spec["input_tokens"]),
        "sampling_params": {
            "temperature": 0, "max_new_tokens": spec["requested_output_tokens"],
            "ignore_eos": True, "stream_interval": 1,
        },
        "stream": stream, "return_logprob": not stream,
    }
    if not stream:
        body.update(logprob_start_len=-1, top_logprobs_num=0,
                    return_text_in_logprobs=False)
    if spec.get("forced_replay"):
        from benchmarks.forced_tokens import forced_payload
        forced = forced_payload(spec["input_tokens"], spec["requested_output_tokens"])
        body["sampling_params"].update(forced.pop("sampling_params"))
        body.update(forced, return_logprob=True, logprob_start_len=-1,
                    top_logprobs_num=0, return_text_in_logprobs=False)
    return body


def sse_data(response):
    """Yield complete SSE data fields without requests' 512-byte line buffering."""
    pending = []
    for raw in response.iter_lines(chunk_size=None):
        line = raw.decode("utf-8")
        if not line:
            if pending:
                yield "\n".join(pending)
                pending = []
        elif line.startswith("data:"):
            pending.append(line[5:].lstrip(" "))
        # SSE comments, event names, IDs, and retry hints contain no payload.
    if pending:
        yield "\n".join(pending)


def token_ids_from_response(data, first_chunk=False):
    direct = data.get("output_ids")
    meta = data["meta_info"]
    # SGLang 0.5.2 sends a five-token prompt suffix in the initial
    # incremental detokenization window, before the generated IDs.
    if first_chunk and isinstance(direct, list):
        prompt_suffix_length = min(5, meta["prompt_tokens"])
        if len(direct) == meta["completion_tokens"] + prompt_suffix_length:
            direct = direct[prompt_suffix_length:]
    logprobs = meta.get("output_token_logprobs")
    logged = [row[1] for row in logprobs] if logprobs is not None else None
    if direct is not None and logged is not None and direct != logged:
        raise BenchmarkError("output_ids disagree with output_token_logprobs")
    ids = direct if direct is not None else logged
    if ids is not None and (not isinstance(ids, list) or
                            any(type(token) is not int or token < 0 for token in ids)):
        raise BenchmarkError("invalid output token IDs")
    return ids


def validate_response(data):
    if not isinstance(data, dict) or data.get("error") is not None:
        raise BenchmarkError("server returned an error or invalid response")
    meta = data.get("meta_info")
    reason = meta.get("finish_reason") if isinstance(meta, dict) else None
    if isinstance(reason, dict) and reason.get("type") in {"abort", "error"}:
        raise BenchmarkError("server aborted generation")
    if not isinstance(meta, dict) or type(meta.get("completion_tokens")) is not int:
        count_type = type(meta.get("completion_tokens")).__name__ if isinstance(meta, dict) else "absent"
        raise BenchmarkError("response lacks integer meta_info.completion_tokens "
                             f"(meta_info_type={type(meta).__name__}, completion_type={count_type})")
    if meta["completion_tokens"] < 0:
        raise BenchmarkError("negative completion token count")
    return meta


def generate(url, spec, timeout, stream=True):
    body = payload_for(spec, stream)
    started = time.perf_counter()
    first_token = None
    events, ids = [], []
    count, previous_count = 0, 0
    final, saw_done, saw_ids = None, False, False
    ids_complete, id_gaps = True, []
    # One independent HTTP session per request avoids shared mutable Session state.
    # Both modes use this same connection policy; connection setup is timed.
    with requests.Session() as session:
        with session.post(url, json=body, stream=stream, timeout=(10, timeout)) as response:
            if response.status_code != 200:
                raise BenchmarkError(f"generation HTTP status {response.status_code}")
            if stream:
                for wire in sse_data(response):
                    now = time.perf_counter()
                    if wire == "[DONE]":
                        saw_done = True
                        break
                    data = json.loads(wire)
                    meta = validate_response(data)
                    count = meta["completion_tokens"]
                    if count < previous_count:
                        raise BenchmarkError("stream completion count went backwards")
                    if count and first_token is None:
                        first_token = now
                    incoming = token_ids_from_response(data, first_chunk=final is None)
                    if incoming is not None:
                        saw_ids = True
                        if len(incoming) == count:
                            if ids_complete and ids and incoming[:len(ids)] != ids:
                                raise BenchmarkError("stream token prefix changed")
                            ids = incoming
                            ids_complete = True
                        elif len(incoming) == count - previous_count:
                            if ids_complete:
                                ids.extend(incoming)
                        elif len(incoming) < count - previous_count:
                            # SGLang 0.5.2 returns delta output_ids, but its
                            # tokenizer manager may coalesce queued responses by
                            # retaining only the last one. completion_tokens and
                            # text remain cumulative; omitted IDs cannot safely
                            # be reconstructed. Keep timing/counts and use the
                            # separate nonstream pass for complete token hashes.
                            ids_complete = False
                            ids = []
                            id_gaps.append({"completion_tokens": count,
                                            "new_tokens": count - previous_count,
                                            "received_ids": len(incoming)})
                        else:
                            raise BenchmarkError(
                                f"unrecognized streaming output_ids shape: count={count}, "
                                f"previous_count={previous_count}, ids_length={len(incoming)}")
                    elif count > previous_count:
                        ids_complete = False
                        ids = []
                    events.append({"arrival_s": now - started,
                                   "completion_tokens": count,
                                   "new_tokens": count - previous_count})
                    previous_count, final = count, data
            else:
                final = response.json()
                count = validate_response(final)["completion_tokens"]
                ids = token_ids_from_response(final, first_chunk=True)
                saw_ids = ids is not None
            ended = time.perf_counter()
    if final is None or count != spec["requested_output_tokens"]:
        raise BenchmarkError(f"unexpected output token count: received={count}, "
                             f"expected={spec['requested_output_tokens']}")
    if stream and not (saw_done or final["meta_info"].get("finish_reason")):
        raise BenchmarkError("stream ended without a completion marker")
    if saw_ids and (not stream or ids_complete) and len(ids) != count:
        raise BenchmarkError(f"incomplete output token IDs: ids_length={len(ids)}, count={count}")
    if not stream and not saw_ids:
        raise BenchmarkError("correctness response lacks output IDs/logprob token IDs")
    if stream and first_token is None:
        raise BenchmarkError("stream contained no positive token count")
    text = final.get("text")
    if text is not None and not isinstance(text, str):
        raise BenchmarkError("response text is not a string")
    elapsed = ended - started
    record = {
        **spec, "prompt_sha256": digest_ids(body["input_ids"]),
        "_started_perf_s": started,
        "output_tokens": count, "end_to_end_s": elapsed,
        "ttft_s": first_token - started if stream else None,
        "tpot_s": (ended - first_token) / (count - 1) if stream and count > 1 else None,
        "request_output_tokens_per_s": count / elapsed,
        "output_ids": ids if saw_ids and ids_complete else None,
        "output_ids_sha256": digest_ids(ids) if saw_ids and ids_complete else None,
        "output_ids_complete": bool(saw_ids and ids_complete),
        "stream_output_id_gaps": id_gaps,
        "output_text_sha256": digest_text(text) if text is not None else None,
        "token_count_events": events,
        "first_event_tokens": next((e["completion_tokens"] for e in events
                                    if e["completion_tokens"] > 0), None),
    }
    if spec.get("forced_replay"):
        from benchmarks.forced_tokens import forced_ids, verify_forced_output
        verify_forced_output(record, forced_ids(spec["input_tokens"], spec["requested_output_tokens"]))
        record["forced_ids_verified"] = True
    return record


class MemorySampler:
    """Whole-device NVML readings, not process-only or PyTorch allocator bytes."""

    def __init__(self):
        import pynvml
        self.nvml = pynvml
        pynvml.nvmlInit()
        uuid = os.environ.get("BENCH_GPU_UUID")
        self.handle = (pynvml.nvmlDeviceGetHandleByUUID(uuid) if uuid
                       else pynvml.nvmlDeviceGetHandleByIndex(0))
        self.selection = "BENCH_GPU_UUID" if uuid else "NVML index 0"
        self.stats_path = os.environ.get("EKV_STATS_PATH")
        self.samples = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.error = None
        self.origin = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def take(self):
        with self.lock:
            stats = None
            if self.stats_path:
                with open(self.stats_path, encoding="utf-8") as source:
                    raw = json.load(source)
                if not isinstance(raw, dict) or type(raw.get("backed_bytes")) is not int:
                    raise BenchmarkError("adapter stats lack integer backed_bytes")
                numeric_keys = STATS_NUMBERS | {key for key in raw if key.startswith("activation_")}
                stats = {key: raw[key] for key in numeric_keys
                         if key in raw and type(raw[key]) in (int, float)}
                # Do not copy arbitrary strings (paths, host names, or identifiers).
                for key in ("mode", "layout"):
                    value = raw.get(key)
                    if isinstance(value, str) and value.replace("_", "").replace("-", "").isalnum():
                        stats[key] = value
            used = int(self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used)
            self.samples.append({"offset_s": time.perf_counter() - self.origin,
                                 "device_used_bytes": used, "kv_stats": stats})

    def _run(self):
        try:
            while not self.stop.wait(SAMPLE_SECONDS):
                self.take()
        except Exception as exc:
            self.error = exc
            self.stop.set()

    def __enter__(self):
        self.take()  # Fail before sending requests if GPU/stats cannot be read.
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join()
        self.nvml.nvmlShutdown()
        if kind is None and self.error is not None:
            raise BenchmarkError("NVML/adapter background sampling failed: "
                                 + safe_error_message(self.error)) from self.error


def distribution(values):
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return None
    def percentile(q):
        at = (len(ordered) - 1) * q
        low, high = math.floor(at), math.ceil(at)
        return ordered[low] + (ordered[high] - ordered[low]) * (at - low)
    return {"mean": statistics.mean(ordered), "median": statistics.median(ordered),
            "p95": percentile(0.95), "min": ordered[0], "max": ordered[-1]}


def memory_summary(samples):
    used = [sample["device_used_bytes"] for sample in samples]
    backed = [sample["kv_stats"]["backed_bytes"] for sample in samples
              if sample["kv_stats"] is not None]
    return {"sample_count": len(samples),
            "device_used_bytes": distribution(used),
            "kv_backed_bytes": distribution(backed)}


def run_requests(url, specs, concurrency, timeout, stream):
    if concurrency == 1:
        start = time.perf_counter()
        records = [generate(url, spec, timeout, stream) for spec in specs]
        return records, start, time.perf_counter()
    ready, release = threading.Barrier(len(specs) + 1), threading.Event()
    def worker(spec):
        ready.wait(timeout=30)
        release.wait(timeout=30)
        return generate(url, spec, timeout, stream)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, spec) for spec in specs]
        ready.wait(timeout=30)
        start = time.perf_counter()
        release.set()
        records = [future.result() for future in futures]
        end = time.perf_counter()
    return records, start, end


def run_phase(url, workload, repeat, timeout, warmup=False, correctness=False):
    specs = specifications(workload, repeat, warmup)
    concurrency = WORKLOADS[workload][0]
    with MemorySampler() as sampler:
        time.sleep(IDLE_SECONDS)
        sampler.take()
        records, started, ended = run_requests(url, specs, concurrency, timeout,
                                               stream=not correctness)
        sampler.take()
        time.sleep(IDLE_SECONDS)
        sampler.take()
    start_offset, end_offset = started - sampler.origin, ended - sampler.origin
    for record in records:
        record["start_offset_s"] = record.pop("_started_perf_s") - sampler.origin
        record["finish_offset_s"] = record["start_offset_s"] + record["end_to_end_s"]
    for sample in sampler.samples:
        t = sample["offset_s"]
        sample["period"] = "idle_before" if t < start_offset else (
            "active" if t <= end_offset else "idle_after")
    duration = ended - started
    tokens = sum(record["output_tokens"] for record in records)
    return {
        "workload": workload, "repeat": repeat, "warmup": warmup,
        "concurrency": concurrency, "request_count": len(records),
        "phase_wall_s": duration, "output_tokens": tokens,
        "aggregate_output_tokens_per_s": tokens / duration,
        "request_end_to_end_s": distribution([r["end_to_end_s"] for r in records]),
        "request_ttft_s": distribution([r["ttft_s"] for r in records]),
        "request_tpot_s": distribution([r["tpot_s"] for r in records]),
        "memory": {period: memory_summary([s for s in sampler.samples if s["period"] == period])
                   for period in ("idle_before", "active", "idle_after")},
        "active_start_offset_s": start_offset, "active_end_offset_s": end_offset,
        "device_used_peak_bytes": max(s["device_used_bytes"] for s in sampler.samples),
        "memory_samples": sampler.samples, "requests": records,
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize_phases(phases, workloads):
    result = {}
    for name in workloads:
        selected = [phase for phase in phases if phase["workload"] == name]
        records = [record for phase in selected for record in phase["requests"]]
        result[name] = {
            "measured_repeats": len(selected), "request_count": len(records),
            "output_tokens": sum(phase["output_tokens"] for phase in selected),
            "aggregate_output_tokens_per_s": (
                sum(phase["output_tokens"] for phase in selected)
                / sum(phase["phase_wall_s"] for phase in selected)),
            "per_repeat_output_tokens_per_s": distribution(
                [phase["aggregate_output_tokens_per_s"] for phase in selected]),
            "request_ttft_s": distribution([r["ttft_s"] for r in records]),
            "request_tpot_s": distribution([r["tpot_s"] for r in records]),
            "request_end_to_end_s": distribution([r["end_to_end_s"] for r in records]),
            "sampled_device_used_peak_bytes": max(
                phase["device_used_peak_bytes"] for phase in selected),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="server base URL or /generate URL")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-label", required=True, help="public, non-path model label")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workloads", nargs="+", default=list(WORKLOADS),
                        help="workload names, separated by spaces or commas")
    parser.add_argument("--timeout", type=float, default=600, help="HTTP read timeout seconds")
    parser.add_argument("--correctness-only", action="store_true",
                        help="run fixed probes using metadata from the completed --output file")
    parser.add_argument("--padding-probes", action="store_true",
                        help="also run 3/5/3 concurrent long-request probes after performance")
    args = parser.parse_args()
    names = [name for item in args.workloads for name in item.split(",") if name]
    if not names or len(names) != len(set(names)) or any(name not in WORKLOADS for name in names):
        parser.error("--workloads must be unique names from: " + ", ".join(WORKLOADS))
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("--repeats and --timeout must be positive")
    url = args.url.rstrip("/")
    if not url.endswith("/generate"):
        url += "/generate"
    correctness_path = args.output.with_name(args.output.stem + ".correctness.json")
    common = {
        "schema_version": 2, "model_label": args.model_label,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "repeats": args.repeats, "workloads": names,
        "prompt_generator": "elastic-kv-prompts-v1/shake256/ids256..8191",
        "nvml_device_selection": "BENCH_GPU_UUID" if os.environ.get("BENCH_GPU_UUID") else "NVML index 0",
        "adapter_stats_enabled": bool(os.environ.get("EKV_STATS_PATH")),
        "sample_interval_s": SAMPLE_SECONDS, "idle_pause_s": IDLE_SECONDS,
    }
    performance = {**common, "kind": "normal_streaming_performance", "status": "running",
                   "return_logprob": False, "warmups": [], "phases": [],
                   "measurement_notes": [
                       "TTFT is first positive completion_tokens event arrival, including HTTP setup.",
                       "TPOT=(stream completion time-first positive event)/(output_tokens-1).",
                       "SSE may coalesce tokens; inspect first_event_tokens and token_count_events.",
                       "SGLang 0.5.2 may omit delta output_ids when coalescing; incomplete IDs get no hash.",
                       "Throughput is actual output tokens / phase wall time; idle pauses are excluded.",
                       "NVML measures the whole selected GPU, including other processes.",
                       "50ms samples can miss short peaks; adapter values reflect its latest write.",
                       "Adapter peak/counter fields may be cumulative; sampled backed_bytes is instantaneous.",
                       "Warmups use distinct prompts and are excluded from every measured aggregate.",
                       "Prefix caching is not reset by this client; record server cache configuration.",
                   ]}
    if args.correctness_only:
        performance = json.loads(args.output.read_text(encoding="utf-8"))
        if (performance.get("status") != "complete"
                or performance.get("kind") != "normal_streaming_performance"
                or performance.get("model_label") != args.model_label):
            parser.error("--correctness-only requires completed performance data for this model")
        common.update(repeats=performance["repeats"], workloads=performance["workloads"])
        correctness_path = args.output.with_name(args.output.stem + ".correctness-v2.json")
    if correctness_path.exists():
        parser.error(f"correctness output already exists: {correctness_path}")
    correctness = {**common, "kind": "separate_greedy_token_correctness", "status": "pending",
                   "return_logprob": True, "concurrency": 1, "probes": [],
                   "measurement_notes": [
                       "These nonstreaming/logprob requests are excluded from performance results.",
                       "Token hashes compare exact greedy output across server modes, not language quality.",
                       "Compare fixed probes by request_id plus prompt_sha256 across layouts.",
                       "A historical baseline trial saw stream/nonstream differences under unconstrained batch shapes.",
                   ]}
    failure_context = {"stage": "initialization"}
    try:
        if not args.correctness_only:
            write_json(args.output, performance)
            write_json(correctness_path, correctness)
            for repeat in range(args.repeats):
                for name in names:
                    failure_context = {"stage": "warmup", "repeat": repeat, "workload": name}
                    performance["warmups"].append(run_phase(url, name, repeat, args.timeout, warmup=True))
                    write_json(args.output, performance)
                    failure_context = {"stage": "measured", "repeat": repeat, "workload": name}
                    phase = run_phase(url, name, repeat, args.timeout)
                    performance["phases"].append(phase)
                    write_json(args.output, performance)
                    print(f"repeat={repeat} workload={name} tokens={phase['output_tokens']} "
                          f"throughput={phase['aggregate_output_tokens_per_s']:.3f} token/s", flush=True)
            performance["status"] = "complete"
            performance["summary"] = summarize_phases(performance["phases"], names)
            write_json(args.output, performance)
        correctness["status"] = "running"
        for input_tokens in (128, 1024, 4096):
            for output_tokens in (64, 128):
                for seed in range(2):
                    spec = {
                        "request_id": f"probe-v2/{input_tokens}/{output_tokens}/{seed}",
                        "input_tokens": input_tokens,
                        "requested_output_tokens": output_tokens,
                    }
                    failure_context = {"stage": "fixed_probe", "request_id": spec["request_id"]}
                    correctness["probes"].append(generate(url, spec, args.timeout, stream=False))
                    write_json(correctness_path, correctness)
        if args.padding_probes:
            correctness["padded_batches"] = []
            for round_number, batch_size in enumerate((3, 5, 3)):
                failure_context = {"stage": "padding_probe", "round": round_number}
                specs = [{"request_id": f"padding/{round_number}/{batch_size}/{index}",
                          "input_tokens": 4096, "requested_output_tokens": 128}
                         for index in range(batch_size)]
                with MemorySampler() as sampler:
                    records, started, ended = run_requests(
                        url, specs, batch_size, args.timeout, stream=False)
                    time.sleep(IDLE_SECONDS)
                    sampler.take()
                for record in records:
                    record.pop("_started_perf_s", None)
                correctness["padded_batches"].append({
                    "round": round_number, "batch_size": batch_size, "requests": records,
                    "memory_samples": sampler.samples,
                    "active_window_s": [started - sampler.origin, ended - sampler.origin],
                })
                write_json(correctness_path, correctness)
        correctness["status"] = "complete"
        write_json(correctness_path, correctness)
        return 0
    except Exception as exc:
        # Keep completed real records; never replace failed observations with estimates.
        target = performance if performance["status"] != "complete" else correctness
        target["status"] = "failed"
        target["error_type"] = type(exc).__name__
        target["error_message"] = safe_error_message(exc)
        target["failure_context"] = failure_context
        if not args.correctness_only:
            write_json(args.output, performance)
        write_json(correctness_path, correctness)
        print(f"Benchmark failed ({type(exc).__name__}): {safe_error_message(exc)}",
              file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
