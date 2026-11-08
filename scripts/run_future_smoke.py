#!/usr/bin/env python3
"""Run a real SGLang scheduler smoke with versioned elastic.future hints."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


HOST = "127.0.0.1"
PORT = 31180
URL = f"http://{HOST}:{PORT}"
ROOT = Path(__file__).resolve().parents[1]


def hint(rid: str, *, output_tokens: int, deadline_ms: int | None = None) -> dict:
    return {
        "protocol_version": "1",
        "message_id": rid,
        "actions": [
            {
                "action_id": f"{rid}:future",
                "action_type": "elastic.future",
                "action_version": "1",
                "payload": {
                    "release_after_ms": 0,
                    "deadline_after_ms": deadline_ms,
                    "expected_output_tokens": output_tokens,
                    "residence": "auto",
                    "weight": 1.0,
                },
            }
        ],
    }


def leader_exit(server: subprocess.Popen) -> object | None:
    return os.waitid(os.P_PID, server.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)


def stop_server(server: subprocess.Popen) -> None:
    try:
        os.killpg(server.pid, 15)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 15
    while leader_exit(server) is None and time.monotonic() < deadline:
        time.sleep(0.1)
    try:
        os.killpg(server.pid, 9)
    except ProcessLookupError:
        pass
    server.wait()


def wait_for_health(server: subprocess.Popen) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 300
    while (remaining := deadline - time.monotonic()) > 0:
        status = leader_exit(server)
        if status is not None:
            raise RuntimeError(f"server exited before health check (status={status})")
        try:
            with opener.open(f"{URL}/health", timeout=min(remaining, 2)) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
    raise TimeoutError("SGLang health check timed out")


def generate(name: str, prompt: int, output: int, deadline_ms: int | None) -> dict:
    rid = f"future-smoke/{name}"
    from benchmarks.client import generate as generate_request

    record = generate_request(
        f"{URL}/generate",
        {
            "request_id": rid,
            "input_tokens": prompt,
            "requested_output_tokens": output,
            "future_hint": hint(
                rid, output_tokens=output, deadline_ms=deadline_ms
            ),
        },
        timeout=180,
        stream=True,
    )
    started = record.pop("_started_perf_s")
    elapsed = record["end_to_end_s"]
    if record["output_tokens"] != output:
        raise RuntimeError(
            f"{name}: invalid response tokens={record['output_tokens']!r}"
        )
    return {
        "name": name,
        "started_s": started,
        "finished_s": started + elapsed,
        "elapsed_s": elapsed,
        "ttft_s": record["ttft_s"],
        "tpot_s": record["tpot_s"],
        "prompt_tokens": prompt,
        "output_tokens": record["output_tokens"],
        "request_output_tokens_per_s": record[
            "request_output_tokens_per_s"
        ],
        "deadline_ms": deadline_ms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--elastic-mode",
        choices=("baseline", "v0", "v1", "v2", "v3", "v4", "v5"),
        default="baseline",
    )
    parser.add_argument("--elastic-chunk-mib", type=int, default=2)
    parser.add_argument("--elastic-retain-chunks", type=int, default=1)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument(
        "--future-policy", choices=("off", "deadline"), default="deadline"
    )
    args = parser.parse_args(argv)
    if not args.model_path.is_dir():
        parser.error("--model-path must be an existing directory")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.mkdir(parents=True)
    log_path = output / "server.log"
    stats_path = output / "future_stats.json"
    vmm_stats_path = output / "vmm_stats.json"
    result_path = output / "results.json"

    env = dict(os.environ)
    env.update(
        EKV_MODE=args.elastic_mode,
        EKV_CHUNK_MIB=str(args.elastic_chunk_mib),
        EKV_RETAIN_CHUNKS=str(args.elastic_retain_chunks),
        EKV_FUTURE_POLICY=args.future_policy,
        EKV_FUTURE_STATS_PATH=str(stats_path),
        EKV_STATS_PATH=str(vmm_stats_path),
        PYTHONUNBUFFERED="1",
    )
    server_command = [
        sys.executable,
        "-m",
        "elastic_kv.launch",
        "--elastic-mode",
        args.elastic_mode,
        "--elastic-chunk-mib",
        str(args.elastic_chunk_mib),
        "--elastic-retain-chunks",
        str(args.elastic_retain_chunks),
        "--elastic-future-policy",
        args.future_policy,
        "--model-path",
        str(args.model_path.resolve()),
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--page-size",
        "1",
        "--attention-backend",
        "triton",
        "--disable-overlap-schedule",
        "--disable-radix-cache",
        "--max-running-requests",
        "1",
        "--max-total-tokens",
        "32768",
        "--context-length",
        "8192",
        "--chunked-prefill-size",
        "2048",
        "--mem-fraction-static",
        "0.35",
        "--schedule-policy",
        "fcfs",
        "--skip-server-warmup",
    ]
    if args.disable_cuda_graph:
        server_command.append("--disable-cuda-graph")
    server = None
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as address:
            address.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            address.bind((HOST, PORT))
        with log_path.open("xb") as log:
            server = subprocess.Popen(
                server_command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_for_health(server)
            from benchmarks.client import MemorySampler, memory_summary

            with MemorySampler() as sampler:
                with ThreadPoolExecutor(max_workers=3) as pool:
                    initial = pool.submit(generate, "initial", 256, 256, None)
                    time.sleep(0.1)
                    slow = pool.submit(generate, "slow", 4096, 64, None)
                    time.sleep(0.03)
                    urgent = pool.submit(generate, "urgent", 128, 8, 3000)
                    rows = [item.result() for item in (initial, slow, urgent)]
                sampler.take()
            nvml = memory_summary(sampler.samples)
        deadline = time.monotonic() + 2
        while not stats_path.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        hook_stats = (
            json.loads(stats_path.read_text())
            if args.future_policy != "off" and stats_path.exists()
            else None
        )
        elastic_stats = (
            json.loads(vmm_stats_path.read_text())
            if args.elastic_mode != "baseline" and vmm_stats_path.exists()
            else None
        )
        by_name = {row["name"]: row for row in rows}
        run_started = min(row["started_s"] for row in rows)
        run_finished = max(row["finished_s"] for row in rows)
        checks = {"requests_completed": len(rows) == 3}
        if args.future_policy != "off":
            checks.update(
                {
                    "urgent_finishes_before_slow": by_name["urgent"]["finished_s"]
                    < by_name["slow"]["finished_s"],
                    "hook_ticks": bool(hook_stats and hook_stats.get("ticks", 0) > 0),
                    "hint_queues_planned": bool(
                        hook_stats
                        and hook_stats.get("released_requests", 0)
                        + hook_stats.get("unreleased_requests", 0)
                        > 0
                    ),
                    "fallback_safe": bool(
                        hook_stats
                        and hook_stats.get("fallback_exception", 0) == 0
                        and hook_stats.get("fallback_invalid_hint", 0) == 0
                        and hook_stats.get("telemetry_errors", 0) == 0
                    ),
                    "queue_reordered": bool(
                        hook_stats and hook_stats.get("reordered_ticks", 0) > 0
                    ),
                }
            )
        if args.elastic_mode != "baseline":
            checks["vmm_active"] = bool(
                elastic_stats
                and elastic_stats.get("mode") == args.elastic_mode
                and elastic_stats.get("backed_bytes", 0) > 0
                and elastic_stats.get("backed_bytes", 0)
                < elastic_stats.get("logical_capacity_bytes", 0)
            )
        result = {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "requests": rows,
            "efficiency": {
                "wall_s": run_finished - run_started,
                "output_tokens_per_s": sum(
                    row["output_tokens"] for row in rows
                ) / (run_finished - run_started),
                "ttft_s": {
                    row["name"]: row["ttft_s"] for row in rows
                },
                "tpot_s": {
                    row["name"]: row["tpot_s"] for row in rows
                },
            },
            "whole_gpu_nvml": nvml,
            "future_stats": hook_stats,
            "elastic_mode": args.elastic_mode,
            "elastic_stats": elastic_stats,
        }
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"status": result["status"], "checks": checks, "stats": hook_stats}))
        return 0 if result["status"] == "passed" else 1
    except Exception as exc:
        print(f"future scheduler smoke failed: {type(exc).__name__}: {exc}; log={log_path}", file=sys.stderr)
        return 1
    finally:
        if server is not None:
            stop_server(server)


if __name__ == "__main__":
    raise SystemExit(main())
