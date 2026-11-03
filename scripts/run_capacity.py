#!/usr/bin/env python3
"""Measure a burst on one GPU with a second model resident and idle."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import signal
import time

import run_case
import run_dual
from run_dual import DualSampler, PORTS, generate, write_json


CAPACITY = 327680


def burst(instance, phase, count, inputs, outputs, stats_paths):
    sampler = DualSampler(stats_paths)
    with sampler, ThreadPoolExecutor(max_workers=count) as pool:
        sampler.take()
        before = sampler.samples[-1]
        started = time.perf_counter()
        pending = [
            pool.submit(generate, f"http://127.0.0.1:{PORTS[instance]}/generate", {
                "request_id": f"capacity/{phase}/{instance}/{index}",
                "instance": instance,
                "input_tokens": inputs,
                "requested_output_tokens": outputs,
            }, 600)
            for index in range(count)
        ]
        requests = [future.result() for future in pending]
        finished = time.perf_counter()
        elapsed = finished - started
        deadline = time.monotonic() + 15
        while True:
            sampler.take()
            after = sampler.samples[-1]
            if all(item["live_tokens"] == 0
                   for item in after["instances"].values()):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"KV did not drain after {phase}")
            time.sleep(0.05)
    for request in requests:
        if request["output_tokens"] != outputs:
            raise RuntimeError(f"Unexpected completion token count for {request['request_id']}")
        for key in ("_started_perf_s", "output_ids", "token_count_events"):
            request.pop(key, None)
    generated = sum(request["output_tokens"] for request in requests)
    return {
        "phase": phase, "elapsed_s": elapsed, "completed_requests": len(requests),
        "output_tokens": generated, "output_tokens_per_s": generated / elapsed,
        "before": before, "after": after, "requests": requests,
        "samples": sampler.samples,
        "active_window_s": [started - sampler.origin, finished - sampler.origin],
    }


def start(model, mode, port, stats, log, capacity):
    import os
    import subprocess
    import sys

    command = [
        sys.executable, "-m", "elastic_kv.launch", "--model-path", str(model.resolve()),
        "--host", "127.0.0.1", "--port", str(port), "--page-size", "1",
        "--attention-backend", "triton", "--disable-cuda-graph",
        "--disable-overlap-schedule", "--disable-radix-cache",
        "--max-total-tokens", str(capacity), "--context-length", "8192",
        "--chunked-prefill-size", "2048", "--mem-fraction-static", "0.95",
        "--max-running-requests", "64",
    ]
    env = {**os.environ, "EKV_MODE": mode, "EKV_STATS_PATH": str(stats),
           "PYTHONUNBUFFERED": "1"}
    with log.open("xb") as handle:
        return subprocess.Popen(command, cwd=run_case.ROOT, env=env, stdout=handle,
                                stderr=subprocess.STDOUT, start_new_session=True)


def actual_capacity(model, stats, log):
    if stats.is_file():
        config = json.loads((model / "config.json").read_text(encoding="utf-8"))
        bytes_per_slot = (4 * config["num_hidden_layers"] *
                          config["num_key_value_heads"] * config["head_dim"])
        raw = json.loads(stats.read_text(encoding="utf-8"))
        return raw["logical_capacity_bytes"] // bytes_per_slot - 1
    values = re.findall(r"\bmax_total_num_tokens\s*=\s*(\d+)",
                        log.read_text(encoding="utf-8", errors="replace"))
    return int(values[-1]) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for label in PORTS:
        parser.add_argument(f"--model-{label}", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("baseline", "v5"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--input-tokens", type=int, default=6144)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--capacity-a", type=int, default=CAPACITY)
    parser.add_argument("--capacity-b", type=int, default=CAPACITY)
    args = parser.parse_args()
    if (not 1 <= args.concurrency <= 64 or args.repeats < 1 or
            min(args.input_tokens, args.output_tokens) < 1 or
            args.input_tokens + args.output_tokens >= 8192 or
            min(args.capacity_a, args.capacity_b) < 8192):
        parser.error("require concurrency in 1..64, positive token counts and repeats, and total tokens below 8192")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = (
        "elastic_kv/adapter.py", "elastic_kv/backing.py", "elastic_kv/launch.py",
        "elastic_kv/slots.py", "elastic_kv/graph_support.py", "elastic_kv/__init__.py",
        "native/elastic_vmm.cpp", "benchmarks/client.py", "scripts/run_case.py",
        "scripts/run_dual.py", "scripts/run_capacity.py", "scripts/container_capacity.sh",
    )
    write_json(output / "source_manifest.json", {
        "mode": args.mode,
        "sha256": {name: hashlib.sha256((run_case.ROOT / name).read_bytes()).hexdigest()
                   for name in sources},
    })
    stats = {label: output / f"{label}.stats.json" for label in PORTS}
    sampled = stats if args.mode == "v5" else {}
    results = {
        "mode": args.mode,
        "requested_capacities": {"a": args.capacity_a, "b": args.capacity_b},
        "capacity_note": "Actual capacities reflect admission under resident GPU allocations. Requested capacity is matched across modes.",
        "actual_capacity": {}, "adapter_stats_available": bool(sampled),
        "illustrative_slo": {"ttft_thresholds_s": [5, 10, 20, 30],
                             "end_to_end_threshold_s": 60},
        "workload": {key: getattr(args, key) for key in
                     ("concurrency", "input_tokens", "output_tokens", "repeats")},
        "phases": [], "warmups": [], "status": "running",
    }
    servers = []
    step = "startup"
    previous_handler = signal.signal(signal.SIGTERM, run_case.on_sigterm)
    try:
        for label, port in PORTS.items():
            step = f"startup_{label}"
            model = getattr(args, f"model_{label}")
            log = output / f"{label}.server.log"
            servers.append(start(model, args.mode, port, stats[label], log,
                                 getattr(args, f"capacity_{label}")))
            run_dual.wait_healthy(servers[-1], port)
            results["actual_capacity"][label] = actual_capacity(model, stats[label], log)
            step = f"warmup_{label}"
            results["warmups"].append(burst(label, step, 1 if label == "a" else 2, 128, 16,
                                           {key: sampled[key] for key in results["actual_capacity"] if key in sampled}))
        step = "first_burst"
        results["first_burst"] = burst("b", step, args.concurrency,
                                       args.input_tokens, args.output_tokens, sampled)
        write_json(output / "results.json", results)
        for repeat in range(1, args.repeats + 1):
            step = f"burst_{repeat}"
            results["phases"].append(burst("b", step, args.concurrency,
                                           args.input_tokens, args.output_tokens, sampled))
            write_json(output / "results.json", results)
        results["status"] = "complete"
    except BaseException as error:
        results.update(status="failed", failed_step=step, error_type=type(error).__name__)
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            write_json(output / "results.json", results)
        finally:
            try:
                for server in reversed(servers):
                    run_case.stop_server(server)
            finally:
                signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
