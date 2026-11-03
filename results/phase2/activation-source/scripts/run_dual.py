#!/usr/bin/env python3
"""Measure load reversal across two isolated SGLang servers on one GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import run_case

sys.path.insert(0, str(run_case.ROOT / "benchmarks"))
from client import MemorySampler, generate, write_json


PORTS = {"a": 31080, "b": 31081}
SCENARIOS = {"a-hot": ("a",), "b-hot": ("b",), "both-active": ("a", "b")}


class DualSampler(MemorySampler):
    def __init__(self, stats_paths):
        super().__init__()
        self.stats_paths = stats_paths

    def take(self):
        with self.lock:
            stats = {}
            for instance, path in self.stats_paths.items():
                raw = json.loads(path.read_text(encoding="utf-8"))
                stats[instance] = {key: raw[key] for key in ("backed_bytes", "live_tokens")}
                if any(type(value) is not int or value < 0 for value in stats[instance].values()):
                    raise ValueError("invalid adapter backing or token count")
            self.samples.append({
                "offset_s": time.perf_counter() - self.origin,
                "device_used_bytes": int(self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used),
                "instances": stats,
            })


def wait_healthy(server, port):
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + 300
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if run_case.leader_exit(server) is not None:
            raise RuntimeError(f"server on port {port} exited during startup")
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"server on port {port} did not become healthy")


def start_server(model, mode, port, stats, log_path):
    command = [
        sys.executable, "-m", "elastic_kv.launch",
        "--model-path", str(model.resolve()), "--host", "127.0.0.1",
        "--port", str(port), "--page-size", "1", "--attention-backend", "triton",
        "--disable-cuda-graph", "--disable-overlap-schedule", "--disable-radix-cache",
        "--max-total-tokens", "32768", "--context-length", "8192",
        "--chunked-prefill-size", "2048", "--mem-fraction-static", "0.5",
    ]
    env = {**os.environ, "EKV_MODE": mode, "EKV_STATS_PATH": str(stats),
           "PYTHONUNBUFFERED": "1"}
    with log_path.open("xb") as log:
        return subprocess.Popen(command, cwd=run_case.ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)


def phase(scenario, repeat, stats_paths, warmup=False):
    jobs = [
        (instance, {
            "request_id": f"{'warmup' if warmup else 'measured'}/{scenario}/{repeat}/{instance}/{index}",
            "instance": instance, "input_tokens": 1024, "requested_output_tokens": 128,
        })
        for instance in SCENARIOS[scenario] for index in range(4)
    ]
    sampler = DualSampler(stats_paths)
    with sampler, ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        started = time.perf_counter()
        futures = [
            pool.submit(generate, f"http://127.0.0.1:{PORTS[instance]}/generate", spec, 120)
            for instance, spec in jobs
        ]
        requests = [future.result() for future in futures]
        elapsed = time.perf_counter() - started
        active_samples = len(sampler.samples)
        time.sleep(0.3)
        sampler.take()

    idle = sampler.samples[-1]
    if any(stats["live_tokens"] for stats in idle["instances"].values()):
        raise RuntimeError(f"live tokens remain after {scenario} repeat {repeat}")
    for request in requests:
        request.pop("_started_perf_s")
        request.pop("output_ids", None)
    output_tokens = sum(request["output_tokens"] for request in requests)
    return {
        "scenario": scenario, "repeat": repeat, "elapsed_s": elapsed,
        "input_tokens": sum(request["input_tokens"] for request in requests),
        "output_tokens": output_tokens, "output_tokens_per_s": output_tokens / elapsed,
        "requests": requests, "samples": sampler.samples,
        "active_sample_count": active_samples,
        "peak_device_used_bytes": max(sample["device_used_bytes"] for sample in sampler.samples),
        "idle": idle,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", required=True, type=Path, help="local 0.6B snapshot")
    parser.add_argument("--model-b", required=True, type=Path, help="local 4B snapshot")
    parser.add_argument("--mode", required=True, choices=("baseline", "v2"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    models = {"a": args.model_a, "b": args.model_b}
    if not all(model.is_dir() for model in models.values()):
        parser.error("both model snapshots must exist locally")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stats_paths = {instance: output / f"{instance}.stats.json" for instance in PORTS}
    sampled_stats = stats_paths if args.mode == "v2" else {}
    servers = []
    previous_handler = signal.signal(signal.SIGTERM, run_case.on_sigterm)
    try:
        for instance, port in PORTS.items():
            server = start_server(models[instance], args.mode, port, stats_paths[instance],
                                  output / f"{instance}.server.log")
            servers.append(server)
            wait_healthy(server, port)
        for scenario in SCENARIOS:
            phase(scenario, 0, sampled_stats, warmup=True)
        results = {
            "mode": args.mode, "models": {"a": "0.6B", "b": "4B"},
            "max_total_tokens_per_instance": 32768,
            "scope": "Two servers share one GPU. No global broker or OOM/preemption test.",
            "adapter_stats_available": bool(sampled_stats),
            "phases": [],
        }
        for repeat in range(1, 4):
            for scenario in SCENARIOS:
                results["phases"].append(phase(scenario, repeat, sampled_stats))
                write_json(output / "results.json", results)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            for server in reversed(servers):
                run_case.stop_server(server)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
