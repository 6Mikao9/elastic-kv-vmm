"""Run one matched vLLM comparison arm in its own GPU container."""

import argparse
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from scripts.run_case import ROOT, URL, leader_exit, on_sigterm, stop_server
from vllm_integration import client


def wait_healthy(server):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        status = leader_exit(server)
        if status is not None:
            raise RuntimeError(f"vLLM startup exited with status {status.si_status}")
        try:
            with opener.open(URL + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError("vLLM /health did not respond within 600 seconds")


def main():
    common = client.common
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", required=True, choices=("baseline", "vmm"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workloads", nargs="+", choices=common.WORKLOADS,
                        default=list(common.WORKLOADS))
    args = parser.parse_args()
    if (not args.model_path.is_dir() or args.repeats < 1 or
            len(set(args.workloads)) != len(args.workloads)):
        parser.error("require a local model, positive repeats and unique workloads")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted((ROOT / "vllm_integration").glob("*"))
    sources = [source for source in sources if source.is_file()]
    sources.extend(ROOT / name for name in (
        "elastic_kv/backing.py", "native/elastic_vmm.cpp",
        "benchmarks/client.py", "scripts/run_case.py",
    ))
    common.write_json(output / "source_manifest.json", {
        "engine": "vllm", "engine_version": "0.29.0", "mode": args.mode,
        "sha256": {source.relative_to(ROOT).as_posix(): hashlib.sha256(source.read_bytes()).hexdigest()
                   for source in sources},
    })

    stats = output / "adapter_stats.json"
    os.environ["EKV_VLLM_MODE"] = args.mode
    os.environ.pop("EKV_VLLM_STATS_PATH", None)
    if args.mode == "vmm":
        os.environ["EKV_VLLM_STATS_PATH"] = str(stats)
    command = [
        sys.executable, "-m", "vllm_integration.launch", str(args.model_path.resolve()),
        "--served-model-name", "bench", "--host", "127.0.0.1", "--port", "31080",
        "--block-size", "16", "--num-gpu-blocks-override", "2048",
        "--max-model-len", "8192", "--max-num-seqs", "8",
        "--max-num-batched-tokens", "2048", "--gpu-memory-utilization", "0.5",
    ]
    # Reuse the shared phase collector with the engine-specific transport and stats.
    common.generate = client.generate
    common.MemorySampler = client.MemorySampler
    metadata = {
        "schema_version": 2, "engine": "vllm", "engine_version": "0.29.0",
        "mode": args.mode, "model_label": args.label, "repeats": args.repeats,
        "workloads": args.workloads, "sample_interval_s": common.SAMPLE_SECONDS,
        "idle_pause_s": common.IDLE_SECONDS, "adapter_stats_enabled": args.mode == "vmm",
    }
    performance = {**metadata, "kind": "normal_streaming_performance",
                   "status": "running", "warmups": [], "phases": []}
    correctness = {**metadata, "kind": "separate_greedy_token_correctness",
                   "status": "pending", "concurrency": 1, "probes": []}
    perf_path, correctness_path = output / "results.json", output / "results.correctness.json"
    endpoint = URL + "/v1/completions"
    stage = "startup"
    server = None
    previous = signal.signal(signal.SIGTERM, on_sigterm)
    try:
        with (output / "server.log").open("xb") as log:
            server = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
        wait_healthy(server)
        for repeat in range(args.repeats):
            for name in args.workloads:
                stage = f"warmup/{name}/{repeat}"
                performance["warmups"].append(common.run_phase(
                    endpoint, name, repeat, 600, warmup=True))
                stage = f"measured/{name}/{repeat}"
                performance["phases"].append(common.run_phase(endpoint, name, repeat, 600))
                common.write_json(perf_path, performance)
        performance["summary"] = common.summarize_phases(performance["phases"], args.workloads)
        performance["status"] = "complete"
        common.write_json(perf_path, performance)

        correctness["status"] = "running"
        for inputs in (128, 1024, 4096):
            for outputs in (64, 128):
                for seed in range(2):
                    stage = f"probe-v2/{inputs}/{outputs}/{seed}"
                    spec = {"request_id": stage, "input_tokens": inputs,
                            "requested_output_tokens": outputs}
                    record = client.generate(endpoint, spec, 600, stream=False)
                    record.pop("_started_perf_s")
                    correctness["probes"].append(record)
        correctness["status"] = "complete"
    except BaseException as error:
        target = correctness if performance["status"] == "complete" else performance
        target.update(status="failed", failed_step=stage, error_type=type(error).__name__)
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            common.write_json(perf_path, performance)
            common.write_json(correctness_path, correctness)
        finally:
            try:
                if server is not None:
                    stop_server(server)
            finally:
                signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
