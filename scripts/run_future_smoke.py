#!/usr/bin/env python3
"""Run a real SGLang scheduler smoke with versioned elastic.future hints."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
    parser.add_argument(
        "--plan",
        type=Path,
        help="execute an existing future_plan JSON rollout",
    )
    parser.add_argument(
        "--workload",
        type=Path,
        help="plan and execute a known-future workload JSON",
    )
    parser.add_argument(
        "--planning-policy",
        choices=("fifo", "longest", "mode-aware", "future", "online", "oracle"),
        default="future",
    )
    parser.add_argument(
        "--no-hints",
        action="store_true",
        help="execute the timestamps as an unhinted control arm",
    )
    parser.add_argument(
        "--service-time-feedback",
        action="store_true",
        help="update later v3 hints from completed requests' TTFT and TPOT",
    )
    parser.add_argument("--service-time-ewma-alpha", type=float, default=0.2)
    parser.add_argument(
        "--residency-stats-dir",
        type=Path,
        help="read per-request, per-TP residency evidence after the rollout",
    )
    parser.add_argument("--require-residency-evidence", action="store_true")
    parser.add_argument("--start-tolerance-ms", type=float, default=100.0)
    parser.add_argument("--request-timeout-s", type=float, default=180.0)
    parser.add_argument("--residency-evidence-timeout-s", type=float, default=5.0)
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--max-inflight", type=int)
    parser.add_argument(
        "--queue-policy", choices=("timeline", "deadline"), default="timeline"
    )
    parser.add_argument(
        "--elastic-residency-budget",
        action="store_true",
        help="demote new requests from the planner byte budget as the live set grows",
    )
    parser.add_argument("--forbid-deadline-overcommit", action="store_true")
    parser.add_argument("--poll-interval-ms", type=float, default=10.0)
    args = parser.parse_args(argv)
    if not args.model_path.is_dir():
        parser.error("--model-path must be an existing directory")
    if args.plan is not None and args.workload is not None:
        parser.error("--plan and --workload are mutually exclusive")
    if (
        args.start_tolerance_ms < 0
        or args.request_timeout_s <= 0
        or args.residency_evidence_timeout_s < 0
    ):
        parser.error("request timeout must be positive; timing tolerances non-negative")
    if args.max_parallel is not None and args.max_parallel < 1:
        parser.error("--max-parallel must be positive")
    if args.max_inflight is not None and args.max_inflight < 1:
        parser.error("--max-inflight must be positive")
    if args.poll_interval_ms <= 0:
        parser.error("--poll-interval-ms must be positive")
    if args.require_residency_evidence and args.residency_stats_dir is None:
        parser.error("--require-residency-evidence needs --residency-stats-dir")
    if args.plan is None and args.workload is None:
        if args.max_inflight is not None:
            parser.error("--max-inflight requires --plan or --workload")
        if args.queue_policy != "timeline":
            parser.error("--queue-policy requires --plan or --workload")
        if args.elastic_residency_budget:
            parser.error("--elastic-residency-budget requires --plan or --workload")
        if args.forbid_deadline_overcommit:
            parser.error("--forbid-deadline-overcommit requires --plan or --workload")
    if (args.plan is not None or args.workload is not None) and args.no_hints and args.future_policy != "off":
        parser.error("--no-hints requires --future-policy off")
    if args.no_hints and args.elastic_residency_budget:
        parser.error("--elastic-residency-budget requires emitted future hints")
    if args.service_time_feedback and (args.plan is None and args.workload is None):
        parser.error("--service-time-feedback requires --plan or --workload")
    if args.service_time_feedback and args.no_hints:
        parser.error("--service-time-feedback requires emitted future hints")
    if not 0 < args.service_time_ewma_alpha <= 1:
        parser.error("--service-time-ewma-alpha must be in (0, 1]")
    if (args.plan is not None or args.workload is not None) and not args.no_hints and args.future_policy == "off":
        parser.error("planned hints require --future-policy deadline; use --no-hints for control")
    output = args.output_dir.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.mkdir(parents=True)
    log_path = output / "server.log"
    stats_path = output / "future_stats.json"
    vmm_stats_path = output / "vmm_stats.json"
    result_path = output / "results.json"

    planned_submissions = None
    plan_manifest = None
    plan_config = None
    effective_planning_policy = args.planning_policy
    if args.plan is not None or args.workload is not None:
        from adaptive_sparse.future_plan import (
            PlannerConfig,
            RequestProfile,
            RuntimeCapabilities,
            plan_rollout,
            validate_runtime_capabilities,
        )
        from adaptive_sparse.rollout_executor import load_plan

        if args.plan is not None:
            try:
                planned_submissions = load_plan(args.plan)
                plan_payload = json.loads(args.plan.read_text(encoding="utf-8"))
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                parser.error(f"invalid --plan: {error}")
            plan_manifest = plan_payload.get("rollout", plan_payload)
            effective_planning_policy = plan_manifest.get("policy", "external")
            raw_config = plan_manifest.get("planner_config")
            if raw_config is not None:
                try:
                    plan_config = PlannerConfig(**raw_config)
                    validate_runtime_capabilities(
                        plan_config, RuntimeCapabilities()
                    )
                except (TypeError, ValueError) as exc:
                    parser.error(f"invalid planner_config in --plan: {exc}")
        else:
            workload = json.loads(args.workload.read_text(encoding="utf-8"))
            if not isinstance(workload, dict):
                parser.error("--workload must contain a JSON object")
            try:
                config = PlannerConfig(**workload["config"])
                requests = [RequestProfile(**item) for item in workload["requests"]]
            except (KeyError, TypeError, ValueError) as exc:
                parser.error(f"invalid --workload: {exc}")
            rollout = plan_rollout(
                requests,
                config,
                args.planning_policy,
                runtime_capabilities=RuntimeCapabilities(),
            )
            planned_submissions = rollout.submissions
            plan_manifest = rollout.as_dict()
            plan_config = config
        (output / "plan.json").write_text(
            json.dumps(plan_manifest or {"submissions": []}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

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
                if planned_submissions is not None:
                    from adaptive_sparse.elastic_admission import (
                        elastic_config_from_planner,
                    )
                    from adaptive_sparse.rollout_executor import execute_rollout
                    from adaptive_sparse.service_time import ServiceTimeEstimator

                    elastic_config = None
                    if args.elastic_residency_budget:
                        if plan_config is None:
                            raise RuntimeError(
                                "elastic residency budget needs planner_config; regenerate the plan"
                            )
                        elastic_config = elastic_config_from_planner(plan_config)

                    planned_result = execute_rollout(
                        f"{URL}/generate",
                        planned_submissions,
                        timeout_s=args.request_timeout_s,
                        emit_hints=not args.no_hints,
                        max_parallel=args.max_parallel,
                        max_submit_lateness_ms=args.start_tolerance_ms,
                        residency_stats_dir=args.residency_stats_dir,
                        require_residency_evidence=args.require_residency_evidence,
                        residency_evidence_timeout_s=args.residency_evidence_timeout_s,
                        max_inflight=args.max_inflight,
                        queue_policy=args.queue_policy,
                        elastic_kv_config=elastic_config,
                        allow_deadline_overcommit=not args.forbid_deadline_overcommit,
                        poll_interval_ms=args.poll_interval_ms,
                        service_time_estimator=(
                            ServiceTimeEstimator(alpha=args.service_time_ewma_alpha)
                            if args.service_time_feedback
                            else None
                        ),
                    )
                else:
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
        if planned_submissions is not None:
            checks = {
                "requests_completed": planned_result["summary"]["completed"],
                "submit_timing_within_tolerance": (
                    planned_result["summary"]["max_submit_lateness_ms"]
                    <= args.start_tolerance_ms
                ),
            }
            if args.future_policy != "off" and not args.no_hints:
                checks.update(
                    {
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
                    }
                )
            if args.require_residency_evidence:
                checks["residency_evidence"] = bool(
                    planned_result["residency_evidence"]
                    and planned_result["residency_evidence"].get("available")
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
                "request_count": len(planned_submissions),
                "requests": planned_result["requests"],
                "plan_summary": planned_result["summary"],
                "residency_evidence": planned_result["residency_evidence"],
                "efficiency": planned_result["summary"]["efficiency"],
                "whole_gpu_nvml": nvml,
                "future_stats": hook_stats,
                "elastic_mode": args.elastic_mode,
                "elastic_stats": elastic_stats,
                "hint_enabled": not args.no_hints,
                "service_time_feedback": args.service_time_feedback,
                "service_time_ewma_alpha": args.service_time_ewma_alpha,
                "planning_policy": effective_planning_policy,
                "max_inflight": args.max_inflight,
                "queue_policy": args.queue_policy,
                "elastic_residency_budget": args.elastic_residency_budget,
            }
            result_path.write_text(
                json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "checks": checks,
                        "summary": planned_result["summary"],
                    },
                    sort_keys=True,
                )
            )
            return 0 if result["status"] == "passed" else 1

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
