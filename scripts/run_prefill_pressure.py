"""Run a fixed Qwen3-4B prefill pressure comparison arm on one isolated GPU."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

from run_case import ROOT, URL, on_sigterm, stop_server, wait_for_health

GENERATE_URL = f"{URL}/generate"
MIB = 1 << 20
SNAPSHOT = "models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
PROBE = {"request_id": "prefill-pressure/short", "input_tokens": 128,
         "requested_output_tokens": 64}


def ballast(target, limit):
    import torch

    cuda = torch.cuda
    if cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one isolated GPU")
    cuda.set_device(0)
    bootstrap = torch.empty(1, device="cuda:0")
    del bootstrap
    cuda.empty_cache()
    before, _ = cuda.mem_get_info(0)
    allocation = (before - target) // (2 * MIB) * (2 * MIB)
    if not 0 < allocation <= limit:
        raise RuntimeError("Required ballast exceeds its explicit limit or is zero")
    held = torch.empty(allocation, dtype=torch.uint8, device="cuda:0")
    after, _ = cuda.mem_get_info(0)
    if abs(after - target) > 64 * MIB:
        raise RuntimeError("Measured driver free missed target by more than 64 MiB")
    print(json.dumps({"driver_free_before_bytes": before, "driver_free_ready_bytes": after,
                      "ballast_allocated_bytes": held.numel(),
                      "ballast_torch_reserved_bytes": cuda.memory_reserved(0)}), flush=True)
    sys.stdin.readline()


def read_idle(path):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        stats = json.loads(path.read_text(encoding="utf-8"))
        if stats["live_tokens"] == 0:
            return stats
        time.sleep(0.05)
    raise TimeoutError("Server did not publish idle KV state")


def scenario(args, report, output):
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from client import MemorySampler, generate

    stats_path = output / "adapter_stats.json"
    os.environ["EKV_STATS_PATH"] = str(stats_path)
    env = {**os.environ, "EKV_MODE": "v5", "EKV_RETAIN_CHUNKS": "8",
           "EKV_ACTIVATION_POLICY": args.policy,
           "EKV_PREFILL_HEADROOM_MIB": str(args.headroom_mib), "PYTHONUNBUFFERED": "1"}
    command = [
        sys.executable, "-m", "elastic_kv.launch",
        "--model-path", str(args.hub / SNAPSHOT), "--host", "127.0.0.1", "--port", "31080",
        "--page-size", "1", "--attention-backend", "triton", "--disable-overlap-schedule",
        "--disable-radix-cache", "--max-total-tokens", "32768", "--context-length", "8192",
        "--chunked-prefill-size", "8192", "--mem-fraction-static", "0.5",
        "--cuda-graph-bs", "1", "2", "4", "8",
    ]
    with (output / "server.log").open("xb") as log:
        server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_for_health(server)
            warmup = {"request_id": "prefill-pressure/warmup", "input_tokens": 7168,
                      "requested_output_tokens": 16}
            generate(GENERATE_URL, warmup, args.timeout, stream=False)
            report["reference"] = generate(GENERATE_URL, PROBE, args.timeout, stream=False)
            idle = read_idle(stats_path)
            if idle["retain_chunks"] != 8 or idle["backed_bytes"] != 8 * 2 * MIB * 72:
                raise RuntimeError("Expected eight retained Qwen3-4B K/V plane chunks")
            report["warm_idle_stats"] = idle
            with (output / "ballast.log").open("xb") as ballast_log:
                proc = subprocess.Popen(
                    [sys.executable, __file__, "--ballast", str(args.target_free_mib * MIB),
                     str(args.ballast_limit_mib * MIB)], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=ballast_log, text=True,
                    start_new_session=True,
                )
                try:
                    if not select.select([proc.stdout], [], [], 120)[0]:
                        raise TimeoutError("Ballast did not report readiness")
                    report["ballast"] = json.loads(proc.stdout.readline())
                    with MemorySampler() as sampler:
                        for index in range(3):
                            before = read_idle(stats_path)
                            timed = generate(GENERATE_URL, PROBE, args.timeout, stream=True)
                            if proc.poll() is not None:
                                raise RuntimeError("Ballast exited during pressure sampling")
                            after = read_idle(stats_path)
                            report["pressure_requests"].append({
                                "index": index, "timed": timed, "before": before, "after": after,
                                "activation_delta": {
                                    key: value - before.get(key, 0)
                                    for key, value in after.items()
                                    if key.startswith(("activation_trim_", "activation_flush_"))
                                },
                            })
                        # Preserve the first pressure action above. Obtain full
                        # token IDs separately under the same held ballast.
                        report["pressure_correctness"] = generate(
                            GENERATE_URL, PROBE, args.timeout, stream=False)
                        read_idle(stats_path)
                        sampler.take()
                    report["whole_gpu_nvml_samples"] = sampler.samples
                finally:
                    proc.stdin.close()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        stop_server(proc)
            reference = report["reference"]
            candidate = report["pressure_correctness"]
            if not (reference["output_ids_complete"] and candidate["output_ids_complete"]
                    and len(reference["output_ids"]) == len(candidate["output_ids"]) == 64
                    and all(reference[key] == candidate[key]
                            for key in ("prompt_sha256", "output_ids"))):
                raise RuntimeError("Pressure correctness probe does not match reference")
            report["outcome"] = "complete"
        finally:
            stop_server(server)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "--ballast":
        ballast(int(sys.argv[2]), int(sys.argv[3]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, choices=("telemetry", "both"))
    parser.add_argument("--hub", type=Path, default=Path("/hf/hub"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-free-mib", type=int, required=True)
    parser.add_argument("--headroom-mib", type=int, required=True)
    parser.add_argument("--ballast-limit-mib", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if not (0 < args.target_free_mib < args.headroom_mib and args.ballast_limit_mib > 0):
        parser.error("Require positive target < headroom and a positive explicit ballast limit")
    args.output.mkdir(parents=True, exist_ok=False)
    signal.signal(signal.SIGTERM, on_sigterm)
    report = {"model": "Qwen3-4B", "execution": "graph-pad", "mode": "v5",
              "prefill_chunk": 8192, "retain_chunks": 8, "policy": args.policy,
              "driver_free_target_bytes": args.target_free_mib * MIB,
              "prefill_headroom_bytes": args.headroom_mib * MIB, "pressure_requests": [],
              "note": "Fixed post-warmup scenario. NVML measures the whole GPU including ballast."}
    sources = (
        "scripts/run_prefill_pressure.py", "scripts/container_prefill_pressure.sh",
        "scripts/run_case.py", "benchmarks/client.py", "elastic_kv/activation.py",
        "elastic_kv/adapter.py", "elastic_kv/backing.py", "elastic_kv/graph_support.py",
        "elastic_kv/launch.py", "elastic_kv/slots.py", "native/elastic_vmm.cpp",
    )
    manifest = {
        "model": report["model"], "snapshot": SNAPSHOT, "execution": "graph-pad",
        "mode": "v5", "retain_chunks": 8, "prefill_chunk": 8192,
        "policy": args.policy, "headroom_mib": args.headroom_mib,
        "target_free_mib": args.target_free_mib, "ballast_limit_mib": args.ballast_limit_mib,
        "ballast_target_tolerance_mib": 64, "timeout_s": args.timeout,
        "attention_backend": "triton", "cuda_graph_buckets": [1, 2, 4, 8],
        "page_size": 1, "max_total_tokens": 32768, "context_length": 8192,
        "mem_fraction_static": 0.5, "overlap_schedule": False, "radix_cache": False,
        "warmup": {"input_tokens": 7168, "requested_output_tokens": 16},
        "probe": PROBE, "timed_pressure_requests": 3,
        "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                   for name in sources},
    }
    (args.output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        scenario(args, report, args.output)
    except BaseException as exc:
        report["outcome"] = "failed"
        report["error_type"] = type(exc).__name__
        raise
    finally:
        (args.output / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
