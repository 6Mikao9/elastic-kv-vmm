"""Smoke-test resident or HiSparse MiniMax serving with actual weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

from scripts.run_case import HOST, PORT, ROOT, URL, leader_exit, on_sigterm, stop_server
from benchmarks.client import digest_ids, prompt_ids, write_json


def wait_healthy(server, opener):
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        status = leader_exit(server)
        if status is not None:
            raise RuntimeError(f"server exited during startup with status {status.si_status}")
        try:
            with opener.open(URL + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError("server health timeout")


def probe(opener, length):
    prompt = prompt_ids(f"hisparse-smoke/{length}", length)
    body = {
        "input_ids": prompt, "stream": False,
        "return_logprob": True, "logprob_start_len": -1,
        "top_logprobs_num": 0, "return_text_in_logprobs": False,
        "sampling_params": {"temperature": 0, "max_new_tokens": 16, "ignore_eos": True},
    }
    request = urllib.request.Request(URL + "/generate", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    started = time.perf_counter()
    with opener.open(request, timeout=600) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    counts = result["meta_info"]
    ids = result.get("output_ids")
    logged = counts.get("output_token_logprobs")
    if ids is None and logged is not None:
        ids = [row[1] for row in logged]
    if (not isinstance(ids, list) or len(ids) != 16 or
            any(type(token) is not int or token < 0 for token in ids) or
            counts["completion_tokens"] != 16 or counts["prompt_tokens"] != length):
        raise RuntimeError("incorrect prompt or completion token count")
    return {
        "input_tokens": length, "output_tokens": len(ids), "elapsed_s": elapsed,
        "prompt_sha256": digest_ids(prompt), "output_ids": ids,
        "output_ids_sha256": digest_ids(ids),
    }


def check_capacity(opener, requested, ratio):
    with opener.open(URL + "/server_info", timeout=30) as response:
        info = json.load(response)
    actual = info.get("max_total_num_tokens")
    if type(actual) is not int or actual != requested:
        raise RuntimeError(f"effective KV pool capacity {actual!r} differs from requested {requested}")
    return {"reported_pool_capacity": actual,
            "resolved_max_total_tokens": info.get("max_total_tokens"),
            "derived_logical_capacity": actual * ratio,
            "host_to_device_ratio": ratio,
            "derivation": "Audited MiniMax pool and allocator code; logical/index capacity is physical pool size times ratio."}


def validate_resident_hint_options(parser, *, policy, minimum_tokens, resident_limit, future_policy):
    if minimum_tokens < 1:
        parser.error("resident hint output threshold must be positive")
    if policy == "adaptive" and (not resident_limit or future_policy == "off"):
        parser.error("adaptive resident hints require future hints and request residency")


def validate_resident_broker_options(parser, *, mode, lookahead_tokens, resident_limit):
    if mode not in ("off", "dry_run", "enforce"):
        parser.error("resident broker mode must be off, dry_run or enforce")
    if lookahead_tokens < 1:
        parser.error("resident broker lookahead must be positive")
    if mode != "off" and not resident_limit:
        parser.error("resident broker requires enabled request residency")


def resident_hint_environment(policy, minimum_tokens, resident_limit):
    if not resident_limit:
        return {}
    return {
        "EKV_RESIDENT_HINT_POLICY": policy,
        "EKV_RESIDENT_MIN_HINT_OUTPUT_TOKENS": str(minimum_tokens),
    }


def resident_hint_manifest(policy, minimum_tokens):
    return {
        "resident_hint_policy": policy,
        "resident_min_hint_output_tokens": minimum_tokens,
    }


def resident_broker_environment(mode, lookahead_tokens, resident_limit):
    if not resident_limit or mode == "off":
        return {}
    return {
        "EKV_RESIDENT_BROKER": mode,
        "EKV_RESIDENT_BROKER_LOOKAHEAD_TOKENS": str(lookahead_tokens),
    }


def resident_broker_manifest(mode, lookahead_tokens):
    return {
        "resident_broker_mode": mode,
        "resident_broker_lookahead_tokens": lookahead_tokens,
    }


def resident_hint_sources(resident_limit):
    if not resident_limit:
        return ()
    return (
        "adaptive_sparse/request_residency.py",
        "adaptive_sparse/future_residency.py",
        "adaptive_sparse/residency_broker.py",
        "adaptive_sparse/resident_lookup.py",
    )


def load_future_submissions(plan_path, workload_path, policy):
    """Load or construct one executable rollout for the MiniMax runner."""

    from adaptive_sparse.future_plan import (
        PlannerConfig,
        RequestProfile,
        plan_rollout,
    )
    from adaptive_sparse.rollout_executor import load_plan

    if plan_path is not None:
        submissions = load_plan(plan_path)
        payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        manifest = payload.get("rollout", payload)
        return submissions, manifest, manifest.get("policy", "external")
    workload = json.loads(Path(workload_path).read_text(encoding="utf-8"))
    config = PlannerConfig(**workload["config"])
    requests = [RequestProfile(**item) for item in workload["requests"]]
    rollout = plan_rollout(requests, config, policy)
    return rollout.submissions, rollout.as_dict(), policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--attention-backend", default="triton")
    parser.add_argument("--mode", choices=("native", "resident"), required=True)
    parser.add_argument("--sm80-bf16-projections", action="store_true",
                        help="Use the experimental MiniMax SM80 projection fallback")
    parser.add_argument("--benchmark-repeats", type=int, default=0)
    parser.add_argument("--forced-replay", action="store_true",
                        help="Controlled benchmark with synthetic fixed generated token IDs")
    parser.add_argument("--warmup-once", action="store_true")
    parser.add_argument("--semantic-probes", action="store_true")
    parser.add_argument("--workloads", nargs="+")
    parser.add_argument("--logical-capacity", type=int, default=65536)
    parser.add_argument("--host-ratio", type=int, default=2)
    parser.add_argument("--mem-fraction-static", type=float, default=0.92)
    parser.add_argument("--vmm-backing", action="store_true")
    parser.add_argument("--vmm-fast-decode", action="store_true")
    parser.add_argument("--vmm-idle-grace-ms", type=int, default=0)
    parser.add_argument("--resident-limit", type=int, default=0)
    parser.add_argument("--resident-free-reserve", type=int, default=16384)
    parser.add_argument("--diagnostic-captures", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--launch-blocking", action="store_true",
                        help="Diagnostic only: serialize CUDA launches; do not use for performance claims")
    parser.add_argument("--trace-layers", action="store_true")
    parser.add_argument("--disable-fused-qknorm", action="store_true")
    parser.add_argument("--capture-attention", action="store_true")
    parser.add_argument("--logical-shadow", action="store_true",
                        help="Diagnostic only: compare selected KV bytes to independent logical writes")
    parser.add_argument("--validation-rounds", type=int, default=0)
    parser.add_argument("--validation-long-tokens", type=int, default=8192)
    parser.add_argument("--validation-new-tokens", type=int, default=16)
    parser.add_argument("--validation-tail-new-tokens", type=int, default=0)
    parser.add_argument("--resident-resolver", choices=("torch", "triton"), default="torch")
    parser.add_argument("--resident-full-batch-swap", action="store_true")
    parser.add_argument("--resident-hint-policy", choices=("off", "adaptive"), default="off")
    parser.add_argument("--resident-min-hint-output-tokens", type=int, default=128)
    parser.add_argument("--resident-broker", choices=("off", "dry_run", "enforce"), default="off")
    parser.add_argument("--resident-broker-lookahead-tokens", type=int, default=256)
    parser.add_argument("--adaptive-budget-mib", type=int, default=0)
    parser.add_argument("--adaptive-low-mib", type=int, default=0)
    parser.add_argument("--adaptive-prefill-headroom-mib", type=int, default=0)
    parser.add_argument("--adaptive-force-demote-step", type=int, default=0)
    parser.add_argument("--adaptive-force-promote-step", type=int, default=0)
    parser.add_argument("--adaptive-dwell-seconds", type=float, default=1)
    parser.add_argument("--adaptive-low-stable-seconds", type=float, default=2)
    parser.add_argument("--adaptive-observe-steps", type=int, default=16)
    parser.add_argument("--adaptive-migration-cap-mib", type=int, default=512)
    parser.add_argument("--adaptive-max-group", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--adaptive-group-search-limit", type=int, default=256)
    parser.add_argument("--adaptive-deadline-guard-ms", type=int, default=500)
    parser.add_argument("--adaptive-plan-capture", action="store_true")
    parser.add_argument("--adaptive-trim-low-water", action="store_true")
    parser.add_argument("--chunk-aware-retention", action="store_true")
    parser.add_argument("--future-policy", choices=("off", "deadline"), default="off")
    parser.add_argument("--future-plan", type=Path,
                        help="execute a known-future rollout plan")
    parser.add_argument("--future-workload", type=Path,
                        help="plan and execute a known-future workload")
    parser.add_argument("--future-planning-policy",
                        choices=("fifo", "longest", "mode-aware", "future", "oracle"),
                        default="future")
    parser.add_argument("--future-no-hints", action="store_true")
    parser.add_argument("--future-start-tolerance-ms", type=float, default=100.0)
    parser.add_argument("--future-request-timeout-s", type=float, default=600.0)
    parser.add_argument("--future-residency-evidence-timeout-s", type=float, default=10.0)
    parser.add_argument("--future-max-parallel", type=int)
    parser.add_argument("--future-max-inflight", type=int)
    parser.add_argument(
        "--future-queue-policy", choices=("timeline", "deadline"), default="timeline"
    )
    parser.add_argument("--future-elastic-residency-budget", action="store_true")
    parser.add_argument("--future-forbid-deadline-overcommit", action="store_true")
    parser.add_argument("--future-poll-interval-ms", type=float, default=10.0)
    args = parser.parse_args()
    if not (args.model_path / "config.json").is_file() or args.tp_size < 4:
        parser.error("require a local MiniMax snapshot and TP >= 4")
    if args.future_plan is not None and args.future_workload is not None:
        parser.error("future plan and workload are mutually exclusive")
    if args.future_start_tolerance_ms < 0 or args.future_request_timeout_s <= 0:
        parser.error("future timing tolerance must be nonnegative and timeout positive")
    if args.future_residency_evidence_timeout_s < 0:
        parser.error("future residency evidence timeout must be nonnegative")
    if args.future_max_parallel is not None and args.future_max_parallel < 1:
        parser.error("future max parallelism must be positive")
    if args.future_max_inflight is not None and args.future_max_inflight < 1:
        parser.error("future max in-flight requests must be positive")
    if args.future_poll_interval_ms <= 0:
        parser.error("future poll interval must be positive")
    if (args.future_plan is not None or args.future_workload is not None) and args.future_no_hints and args.future_policy != "off":
        parser.error("future no-hints requires future policy off")
    if args.future_no_hints and args.future_elastic_residency_budget:
        parser.error("future elastic residency budget requires emitted future hints")
    if args.future_plan is None and args.future_workload is None:
        if args.future_max_inflight is not None:
            parser.error("future max-inflight requires a future plan or workload")
        if args.future_queue_policy != "timeline":
            parser.error("future queue-policy requires a future plan or workload")
        if args.future_elastic_residency_budget:
            parser.error("future elastic residency budget requires a future plan or workload")
        if args.future_forbid_deadline_overcommit:
            parser.error("future forbid-deadline-overcommit requires a future plan or workload")
    if ((args.future_plan is not None or args.future_workload is not None)
            and not args.future_no_hints and args.future_policy == "off"):
        parser.error("future plan hints require future policy deadline; use no-hints for control")
    if (args.host_ratio < 1 or args.logical_capacity <= 0
            or args.logical_capacity % (args.host_ratio * 128)
            or not 0 < args.mem_fraction_static < 1):
        parser.error("require aligned logical capacity, positive ratio and memory fraction in (0,1)")
    if (not 0 <= args.resident_limit <= 4
            or (args.resident_limit and args.resident_free_reserve <= 0)):
        parser.error("require a resident limit between 0 and 4 and positive slot reserve")
    if args.diagnostic_captures and args.diagnostic_captures < 2:
        parser.error("diagnostic requires at least two captures")
    if (args.trace_layers or args.disable_fused_qknorm or args.capture_attention) and not args.sm80_bf16_projections:
        parser.error("diagnostic adapters require the SM80 launcher")
    if (args.trace_layers or args.launch_blocking or args.capture_attention) and args.benchmark_repeats:
        parser.error("timing with intrusive diagnostic capture is not a performance benchmark")
    if args.logical_shadow and (args.mode != "native" or not args.sm80_bf16_projections
                                or args.benchmark_repeats):
        parser.error("logical shadow requires native HiSparse SM80 and forbids performance benchmarks")
    if args.validation_rounds < 0 or (args.validation_rounds and not args.logical_shadow):
        parser.error("validation rounds require logical shadow and a nonnegative count")
    if (not 4096 < args.validation_long_tokens < 16384 or
            not 2 <= args.validation_new_tokens <= 128 or
            args.validation_long_tokens + args.validation_new_tokens > 16384):
        parser.error("validation requests must fit the context and exercise long-request decode")
    tail_tokens = args.validation_tail_new_tokens or args.validation_new_tokens
    if not 2 <= tail_tokens <= 128 or args.validation_long_tokens + tail_tokens > 16384:
        parser.error("validation tail request must fit the context")
    if args.forced_replay and not args.benchmark_repeats:
        parser.error("forced replay requires performance benchmark enabled")
    adaptive = bool(args.adaptive_budget_mib or args.adaptive_force_demote_step)
    if args.vmm_fast_decode and not args.vmm_backing:
        parser.error("fast decode accounting requires VMM backing")
    if args.vmm_idle_grace_ms < 0 or (args.vmm_idle_grace_ms and not args.vmm_backing):
        parser.error("nonnegative idle grace requires VMM backing")
    if adaptive and not (args.vmm_backing and args.resident_limit):
        parser.error("adaptive scheduler requires VMM and enabled request residency")
    if args.chunk_aware_retention and not adaptive:
        parser.error("chunk-aware retention requires adaptive migrations")
    if args.resident_full_batch_swap and (not args.resident_limit or args.resident_resolver != "triton"):
        parser.error("full-batch swap requires resident requests and Triton resolver")
    validate_resident_hint_options(
        parser,
        policy=args.resident_hint_policy,
        minimum_tokens=args.resident_min_hint_output_tokens,
        resident_limit=args.resident_limit,
        future_policy=args.future_policy,
    )
    validate_resident_broker_options(
        parser,
        mode=args.resident_broker,
        lookahead_tokens=args.resident_broker_lookahead_tokens,
        resident_limit=args.resident_limit,
    )
    if args.adaptive_plan_capture and not args.logical_shadow:
        parser.error("allocation plan capture is diagnostic and requires logical shadow")
    if (args.adaptive_force_demote_step or args.adaptive_force_promote_step) and not args.logical_shadow:
        parser.error("forced migrations are diagnostic and require logical shadow")
    adaptive_settings = {
        "EKV_ADAPTIVE_BUDGET_BYTES": args.adaptive_budget_mib * (1 << 20),
        "EKV_ADAPTIVE_LOW_BYTES": args.adaptive_low_mib * (1 << 20),
        "EKV_ADAPTIVE_PREFILL_HEADROOM_BYTES": args.adaptive_prefill_headroom_mib * (1 << 20),
        "EKV_ADAPTIVE_FORCE_DECODE_STEP": args.adaptive_force_demote_step,
        "EKV_ADAPTIVE_FORCE_PROMOTE_STEP": args.adaptive_force_promote_step,
        "EKV_ADAPTIVE_DWELL_SECONDS": args.adaptive_dwell_seconds,
        "EKV_ADAPTIVE_LOW_STABLE_SECONDS": args.adaptive_low_stable_seconds,
        "EKV_ADAPTIVE_OBSERVE_STEPS": args.adaptive_observe_steps,
        "EKV_ADAPTIVE_MIGRATION_BYTE_CAP": args.adaptive_migration_cap_mib * (1 << 20),
        "EKV_ADAPTIVE_MAX_GROUP": args.adaptive_max_group,
        "EKV_ADAPTIVE_GROUP_SEARCH_LIMIT": args.adaptive_group_search_limit,
        "EKV_ADAPTIVE_DEADLINE_GUARD_MS": args.adaptive_deadline_guard_ms,
        "EKV_ADAPTIVE_PLAN_CAPTURE": int(args.adaptive_plan_capture),
        "EKV_ADAPTIVE_TRIM_LOW_WATER": int(args.adaptive_trim_low_water),
    }
    if min(adaptive_settings.values()) < 0 or args.adaptive_observe_steps < 1:
        parser.error("adaptive settings must be nonnegative and observation interval positive")
    if (args.vmm_backing or args.resident_limit) and (
            args.mode != "native" or not args.sm80_bf16_projections):
        parser.error("this VMM/residency adapter requires the native HiSparse SM80 launcher")
    library = ROOT / "native/libelastic_vmm.so"
    if args.vmm_backing and not library.is_file():
        parser.error("compile native/libelastic_vmm.so before using VMM")
    gpu_uuids = os.environ.get("EKV_BENCH_GPU_UUIDS", "").split(",")
    if args.benchmark_repeats < 0 or (args.benchmark_repeats and (
            args.tp_size != 4 or len(gpu_uuids) != 4 or len(set(gpu_uuids)) != 4
            or any(not uuid.startswith("GPU-") for uuid in gpu_uuids))):
        parser.error("benchmark requires TP4 and four launcher-provided GPU UUIDs")
    if args.workloads:
        from benchmarks.hisparse_client import WORKLOADS
        if not args.benchmark_repeats or any(name not in WORKLOADS for name in args.workloads):
            parser.error("workloads require enabled benchmark and known names")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    future_plan_mode = args.future_plan is not None or args.future_workload is not None
    planned_submissions = None
    future_plan_manifest = None
    future_plan_config = None
    effective_future_policy = args.future_planning_policy
    if future_plan_mode:
        try:
            (
                planned_submissions,
                future_plan_manifest,
                effective_future_policy,
            ) = load_future_submissions(
                args.future_plan,
                args.future_workload,
                args.future_planning_policy,
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            parser.error(f"invalid future plan or workload: {error}")
        write_json(output / "future_plan.json", future_plan_manifest)
        raw_config = future_plan_manifest.get("planner_config")
        if raw_config is not None:
            from adaptive_sparse.future_plan import PlannerConfig

            try:
                future_plan_config = PlannerConfig(**raw_config)
            except (TypeError, ValueError) as error:
                parser.error(f"invalid future planner_config: {error}")
        if any(item.prompt_tokens + item.output_tokens > 16384 for item in planned_submissions):
            parser.error("future plan request exceeds the MiniMax context length")
    needs_planned_resident_rows = bool(
        planned_submissions
        and any(item.hint.get("residence") == "resident" for item in planned_submissions)
    )
    require_runtime_resident_rows = needs_planned_resident_rows
    config = {"top_k": 2048, "device_buffer_size": 4096,
              "host_to_device_ratio": args.host_ratio}
    capacity = (args.logical_capacity // args.host_ratio
                if args.mode == "native" else args.logical_capacity)
    server_args = [
        "--host", HOST, "--port", str(PORT), "--tp-size", str(args.tp_size),
        "--attention-backend", args.attention_backend, "--mm-attention-backend", "triton_attn",
        "--dtype", "bfloat16", "--kv-cache-dtype", "bfloat16",
        "--moe-runner-backend", "marlin", "--watchdog-timeout", "1800",
        "--random-seed", "29",
        "--page-size", "128", "--max-total-tokens", str(capacity), "--context-length", "16384",
        "--chunked-prefill-size", "2048", "--max-running-requests", "4",
        "--mem-fraction-static", str(args.mem_fraction_static),
        "--disable-cuda-graph", "--disable-prefill-cuda-graph",
        "--disable-overlap-schedule", "--disable-radix-cache",
    ]
    if args.mode == "native":
        server_args.extend(["--enable-hisparse", "--hisparse-config", json.dumps(config)])
    if args.deterministic:
        server_args.extend(["--enable-deterministic-inference", "--disable-custom-all-reduce"])
    if args.forced_replay:
        server_args.append("--enable-custom-logit-processor")
    launch = ([str(ROOT / "adaptive_sparse/launch_minimax_sm80.py")]
              if args.sm80_bf16_projections else ["-m", "sglang.launch_server"])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT), environment.get("PYTHONPATH"))))
    if args.launch_blocking:
        environment["CUDA_LAUNCH_BLOCKING"] = "1"
    else:
        environment.pop("CUDA_LAUNCH_BLOCKING", None)
    if args.sm80_bf16_projections:
        environment["EKV_MINIMAX_SM80_BF16"] = "1"
        environment["EKV_ALLOCATOR_STATS_DIR"] = str(output / "allocator_stats")
    else:
        environment.pop("EKV_MINIMAX_SM80_BF16", None)
        environment.pop("EKV_ALLOCATOR_STATS_DIR", None)
    for name in ("EKV_VMM_STATS_DIR", "EKV_RESIDENT_LIMIT", "EKV_RESIDENT_FREE_RESERVE",
                 "EKV_RESIDENT_HINT_POLICY", "EKV_RESIDENT_MIN_HINT_OUTPUT_TOKENS",
                 "EKV_RESIDENT_BROKER", "EKV_RESIDENT_BROKER_LOOKAHEAD_TOKENS",
                 "EKV_LAYER_TRACE_DIR", "EKV_DISABLE_FUSED_QKNORM", "EKV_RESIDENT_RESOLVER"):
        environment.pop(name, None)
    environment.pop("EKV_FUTURE_POLICY", None)
    environment.pop("EKV_FUTURE_STATS_PATH", None)
    environment.pop("EKV_RESIDENT_FULL_BATCH_SWAP", None)
    if args.resident_full_batch_swap:
        environment["EKV_RESIDENT_FULL_BATCH_SWAP"] = "1"
    environment.pop("EKV_ATTN_CAPTURE_DIR", None)
    environment.pop("EKV_LOGICAL_SHADOW_DIR", None)
    environment.pop("EKV_VMM_FAST_DECODE", None)
    environment.pop("EKV_VMM_IDLE_GRACE_MS", None)
    if args.vmm_backing:
        environment["EKV_VMM_IDLE_GRACE_MS"] = str(args.vmm_idle_grace_ms)
    environment.pop("EKV_CHUNK_AWARE_RETENTION", None)
    if args.chunk_aware_retention:
        environment["EKV_CHUNK_AWARE_RETENTION"] = "1"
    if args.vmm_fast_decode:
        environment["EKV_VMM_FAST_DECODE"] = "1"
    for name in (*adaptive_settings, "EKV_ADAPTIVE_STATS_DIR"):
        environment.pop(name, None)
    if adaptive:
        environment.update({key: str(value) for key, value in adaptive_settings.items()})
        environment["EKV_ADAPTIVE_STATS_DIR"] = str(output / "adaptive_stats")
    if args.logical_shadow:
        environment["EKV_LOGICAL_SHADOW_DIR"] = str(output / "shadow_stats")
    if args.capture_attention:
        environment["EKV_ATTN_CAPTURE_DIR"] = str(output / "private_trace" / "attention")
    if args.trace_layers:
        environment["EKV_LAYER_TRACE_DIR"] = str(output / "private_trace")
    if args.disable_fused_qknorm:
        environment["EKV_DISABLE_FUSED_QKNORM"] = "1"
    if args.vmm_backing:
        environment["EKV_VMM_STATS_DIR"] = str(output / "vmm_stats")
        environment["ELASTIC_VMM_LIBRARY"] = str(library)
    if args.future_policy != "off":
        environment["EKV_FUTURE_POLICY"] = args.future_policy
        environment["EKV_FUTURE_STATS_PATH"] = str(output / "future_stats.json")
    if args.resident_limit:
        environment["EKV_RESIDENT_LIMIT"] = str(args.resident_limit)
        environment["EKV_RESIDENT_FREE_RESERVE"] = str(args.resident_free_reserve)
        environment["EKV_RESIDENT_RESOLVER"] = args.resident_resolver
        environment.update(resident_hint_environment(
            args.resident_hint_policy, args.resident_min_hint_output_tokens,
            args.resident_limit,
        ))
        environment.update(resident_broker_environment(
            args.resident_broker, args.resident_broker_lookahead_tokens,
            args.resident_limit,
        ))
    command = [sys.executable, "-u", *launch, "--model-path",
               str(args.model_path.resolve()), *server_args]
    sources = ("scripts/run_case.py", "benchmarks/client.py", "scripts/run_hisparse_smoke.py",
               "scripts/container_hisparse_smoke.sh")
    if args.sm80_bf16_projections:
        sources += ("adaptive_sparse/minimax_sm80.py", "adaptive_sparse/launch_minimax_sm80.py",
                    "adaptive_sparse/memory_telemetry.py")
    if args.benchmark_repeats:
        sources += ("benchmarks/hisparse_client.py",)
    if args.forced_replay:
        sources += ("benchmarks/forced_tokens.py",)
    if args.semantic_probes:
        sources += ("benchmarks/hisparse_semantic.py",)
    if args.vmm_backing:
        sources += ("adaptive_sparse/minimax_vmm.py", "adaptive_sparse/vmm_runtime.py",
                    "elastic_kv/backing.py", "native/elastic_vmm.cpp", "native/libelastic_vmm.so")
    if args.future_policy != "off":
        sources += ("adaptive_sparse/sglang_future_hook.py", "adaptive_sparse/kv_hints_compat.py")
    if future_plan_mode:
        sources += (
            "adaptive_sparse/future_plan.py",
            "adaptive_sparse/rollout_executor.py",
            "adaptive_sparse/elastic_admission.py",
        )
    if args.resident_limit:
        sources += resident_hint_sources(args.resident_limit)
    if args.diagnostic_captures:
        sources += ("benchmarks/hisparse_correctness.py",)
    if args.trace_layers:
        sources += ("adaptive_sparse/layer_trace.py",)
    if args.disable_fused_qknorm:
        sources += ("adaptive_sparse/minimax_diagnostic_fallbacks.py",)
    if args.capture_attention:
        sources += ("adaptive_sparse/minimax_attention_capture.py",)
    if args.logical_shadow:
        sources += ("adaptive_sparse/logical_shadow.py",)
    if adaptive:
        sources += ("adaptive_sparse/scheduler_budget.py", "adaptive_sparse/migration.py",
                    "adaptive_sparse/adaptive_runtime.py", "adaptive_sparse/policy.py",
                    "adaptive_sparse/future_prefill.py", "adaptive_sparse/runtime_value.py")
    with zipfile.ZipFile(output / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in dict.fromkeys(sources):
            if Path(name).suffix in (".py", ".cpp", ".sh"):
                archive.write(ROOT / name, name)
    write_json(output / "source_manifest.json", {
        "mode": args.mode, "server_args": server_args,
        "sm80_bf16_projections": args.sm80_bf16_projections,
        "benchmark_repeats": args.benchmark_repeats, "workloads": args.workloads,
        "forced_replay": args.forced_replay,
        "warmup_once": args.warmup_once,
        "semantic_probes": args.semantic_probes,
        "requested_logical_capacity": args.logical_capacity,
        "requested_pool_capacity": capacity,
        "capacity_note": "Verify post-profile effective capacities in runtime logs; flags are upper bounds.",
        "vmm_backing": args.vmm_backing, "resident_limit": args.resident_limit,
        "vmm_fast_decode": args.vmm_fast_decode,
        "vmm_idle_grace_ms": args.vmm_idle_grace_ms,
        "resident_free_reserve": args.resident_free_reserve,
        "compiler_cache_directory_preexisted": os.environ.get("EKV_COMPILER_CACHE_PREEXISTED") == "1",
        "diagnostic_captures": args.diagnostic_captures,
        "cuda_launch_blocking_diagnostic": args.launch_blocking,
        "layer_trace_diagnostic": args.trace_layers,
        "qknorm_fusion_disabled": args.disable_fused_qknorm,
        "attention_capture_diagnostic": args.capture_attention,
        "logical_shadow_diagnostic": args.logical_shadow,
        "validation_rounds": args.validation_rounds,
        "validation_long_tokens": args.validation_long_tokens,
        "validation_new_tokens": args.validation_new_tokens,
        "validation_tail_new_tokens": tail_tokens,
        "resident_resolver": args.resident_resolver if args.resident_limit else None,
        "resident_full_batch_swap": args.resident_full_batch_swap,
        **resident_hint_manifest(
            args.resident_hint_policy, args.resident_min_hint_output_tokens
        ),
        **resident_broker_manifest(
            args.resident_broker, args.resident_broker_lookahead_tokens
        ),
        "adaptive_settings": adaptive_settings if adaptive else None,
        "chunk_aware_retention": args.chunk_aware_retention,
        "future_policy": args.future_policy,
        "future_plan": future_plan_manifest,
        "future_planning_policy": effective_future_policy if future_plan_mode else None,
        "future_hints_emitted": future_plan_mode and not args.future_no_hints,
        "future_max_inflight": args.future_max_inflight,
        "future_queue_policy": args.future_queue_policy,
        "future_elastic_residency_budget": args.future_elastic_residency_budget,
        "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
    })
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    results = {"mode": args.mode, "tp_size": args.tp_size,
               "sm80_bf16_projections": args.sm80_bf16_projections,
               "vmm_backing": args.vmm_backing, "resident_limit": args.resident_limit,
               **resident_hint_manifest(
                   args.resident_hint_policy, args.resident_min_hint_output_tokens
               ),
               **resident_broker_manifest(
                   args.resident_broker, args.resident_broker_lookahead_tokens
               ),
               "future_policy": args.future_policy,
               "future_planning_policy": effective_future_policy if future_plan_mode else None,
               "future_hints_emitted": future_plan_mode and not args.future_no_hints,
               "future_max_inflight": args.future_max_inflight,
               "future_queue_policy": args.future_queue_policy,
               "future_elastic_residency_budget": args.future_elastic_residency_budget,
               "status": "running", "probes": []}
    server = None
    step = "startup"
    previous = signal.signal(signal.SIGTERM, on_sigterm)
    try:
        with (output / "server.log").open("xb") as log:
            server = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True, env=environment)
        wait_healthy(server, opener)
        step = "capacity_validation"
        results["capacity"] = check_capacity(
            opener, capacity, args.host_ratio if args.mode == "native" else 1)
        write_json(output / "results.json", results)
        for length in (128, 8192):
            step = f"probe_{length}"
            results["probes"].append(probe(opener, length))
            write_json(output / "results.json", results)
        if future_plan_mode:
            from adaptive_sparse.rollout_executor import execute_rollout
            from adaptive_sparse.elastic_admission import elastic_config_from_planner
            from benchmarks.client import MemorySampler, memory_summary

            elastic_config = None
            if args.future_elastic_residency_budget:
                if future_plan_config is None:
                    raise RuntimeError(
                        "future elastic residency budget needs planner_config; regenerate the plan"
                    )
                elastic_config = elastic_config_from_planner(future_plan_config)

            step = "future_plan_execution"
            previous_gpu_uuid = os.environ.get("BENCH_GPU_UUID")
            if gpu_uuids:
                os.environ["BENCH_GPU_UUID"] = gpu_uuids[0]
            try:
                with MemorySampler() as sampler:
                    planned_result = execute_rollout(
                        URL + "/generate",
                        planned_submissions,
                        timeout_s=args.future_request_timeout_s,
                        emit_hints=not args.future_no_hints,
                        max_parallel=args.future_max_parallel,
                        max_submit_lateness_ms=args.future_start_tolerance_ms,
                        residency_stats_dir=(
                            output / "vmm_stats"
                            if args.vmm_backing and args.resident_limit
                            else None
                        ),
                        require_residency_evidence=bool(
                            args.vmm_backing and args.resident_limit
                        ),
                        residency_evidence_timeout_s=(
                            args.future_residency_evidence_timeout_s
                        ),
                        max_inflight=args.future_max_inflight,
                        queue_policy=args.future_queue_policy,
                        elastic_kv_config=elastic_config,
                        allow_deadline_overcommit=(
                            not args.future_forbid_deadline_overcommit
                        ),
                        poll_interval_ms=args.future_poll_interval_ms,
                    )
                    sampler.take()
            finally:
                if previous_gpu_uuid is None:
                    os.environ.pop("BENCH_GPU_UUID", None)
                else:
                    os.environ["BENCH_GPU_UUID"] = previous_gpu_uuid
            write_json(output / "future_execution.json", planned_result)
            require_runtime_resident_rows = any(
                row.get("runtime_residence") == "resident"
                for row in planned_result["requests"]
            )
            results["future_execution"] = {
                "summary": planned_result["summary"],
                "residency_evidence": planned_result["residency_evidence"],
                "rank0_whole_gpu_nvml": memory_summary(sampler.samples),
            }
            results["future_requests"] = planned_result["requests"]
            write_json(output / "results.json", results)
        if args.diagnostic_captures:
            from benchmarks.hisparse_correctness import main as diagnose
            step = "determinism_diagnostic"
            diagnose(["--output", str(output / "determinism.json"),
                      "--captures", str(args.diagnostic_captures)])
        if args.benchmark_repeats:
            from benchmarks.hisparse_client import run_benchmark
            step = "benchmark"
            run_benchmark(URL, output / "performance.json", gpu_uuids,
                          repeats=args.benchmark_repeats, workloads=args.workloads,
                          forced_replay=args.forced_replay, warmup_once=args.warmup_once,
                          emit_future_hints=args.future_policy != "off")
        if args.logical_shadow:
            step = "logical_shadow_validation"
            if args.validation_rounds:
                from benchmarks.client import run_requests
                results["validation_rounds"] = []
                for repeat in range(args.validation_rounds):
                    specifications = [
                        {"request_id": f"shadow/mixed/{repeat}/{index}",
                         "input_tokens": length,
                         "requested_output_tokens": tail_tokens if index == 2 else args.validation_new_tokens}
                        for index, length in enumerate((128, args.validation_long_tokens,
                                                        args.validation_long_tokens, 128))
                    ]
                    records, _, _ = run_requests(URL + "/generate", specifications, 4, 600, stream=True)
                    if any(item["output_tokens"] != item["requested_output_tokens"] for item in records):
                        raise RuntimeError("incomplete concurrent shadow validation outputs")
                    for item in records:
                        item.pop("_started_perf_s", None)
                    results["validation_rounds"].append(records)
                    write_json(output / "results.json", results)
            deadline = time.monotonic() + 60
            # Each 16-token probe executes 15 decode steps on 60 sparse layers;
            # require the long probe too, not merely an earlier idle snapshot.
            minimum_tokens = 60 * 15 * (128 + 2048 - 127)
            minimum_calls = 60 * (15 * 2 + (max(args.validation_new_tokens, tail_tokens) - 1) * args.validation_rounds)
            while True:
                if leader_exit(server) is not None:
                    raise RuntimeError("server exited during logical shadow validation")
                snapshots = []
                for rank in range(args.tp_size):
                    path = output / "shadow_stats" / f"shadow-tp-{rank}.json"
                    if path.is_file():
                        snapshots.append(json.loads(path.read_text()))
                if (len(snapshots) == args.tp_size and
                        all(item["checked_tokens"] >= minimum_tokens and
                            item["checked_calls"] >= minimum_calls for item in snapshots)):
                    for item in snapshots:
                        for key, value in item.items():
                            if (key.startswith(("invalid_", "uninitialized_", "mismatched_", "malformed_"))
                                    and value != 0):
                                raise RuntimeError(f"Logical shadow failed: {item}")
                        if any(item.get(name, 0) for name in (
                                "migration_invalid_tokens", "migration_mismatched_tokens")):
                            raise RuntimeError(f"Complete migration shadow failed: {item}")
                    results["logical_shadow"] = snapshots
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("incomplete per-rank logical shadow coverage")
                time.sleep(0.2)
        if args.vmm_backing:
            step = "vmm_idle_validation"
            deadline = time.monotonic() + 60
            while True:
                paths = [output / "vmm_stats" / f"tp-{rank}.json" for rank in range(args.tp_size)]
                snapshots = [json.loads(path.read_text()) for path in paths if path.is_file()]
                ready = len(snapshots) == args.tp_size and all(
                    item["owned_pages"] == 0 and item["reclaimable_bytes"] == 0 for item in snapshots)
                if args.resident_limit:
                    require_resident_rows = (
                        require_runtime_resident_rows if future_plan_mode else True
                    )
                    if require_resident_rows:
                        ready = ready and all(item.get("request_residency", {}).get(
                            "resident_layer_rows", 0) > 0 for item in snapshots)
                    if args.validation_rounds:
                        ready = ready and all(item.get("request_residency", {}).get(
                            "mixed_decode_batches", 0) > 0 for item in snapshots)
                if ready:
                    results["vmm_idle"] = snapshots
                    break
                if leader_exit(server) is not None or time.monotonic() >= deadline:
                    raise RuntimeError("VMM idle cleanup or residency coverage incomplete")
                time.sleep(0.2)
        if adaptive:
            step = "adaptive_validation"
            paths = [output / "adaptive_stats" / f"rank{rank:02d}.json" for rank in range(args.tp_size)]
            results["adaptive"] = [json.loads(path.read_text()) for path in paths]
            for item in results["adaptive"]:
                if ((args.adaptive_force_demote_step and item["demotions"] < 1)
                        or (args.adaptive_force_promote_step and item["promotions"] < 1)):
                    raise RuntimeError("forced migration did not execute")
            if args.logical_shadow and any(item["demotions"] or item["promotions"] for item in results["adaptive"]):
                if any(item.get("migration_checked_tokens", 0) <= 0 for item in results["logical_shadow"]):
                    raise RuntimeError("missing complete migration KV validation")
        if args.sm80_bf16_projections:
            step = "allocator_telemetry_validation"
            deadline = time.monotonic() + 60
            paths = [output / "allocator_stats" / f"allocator-tp-{rank}.json" for rank in range(args.tp_size)]
            while not all(path.is_file() for path in paths):
                if leader_exit(server) is not None or time.monotonic() >= deadline:
                    raise RuntimeError("missing per-rank allocator telemetry")
                time.sleep(0.2)
            results["allocator_idle"] = [json.loads(path.read_text()) for path in paths]
        if args.semantic_probes:
            from benchmarks.hisparse_semantic import smoke as semantic_smoke
            step = "semantic_smoke"
            with opener.open(URL + "/v1/models", timeout=30) as response:
                served_model = json.load(response)["data"][0]["id"]
            semantic = semantic_smoke(URL, served_model, 180)
            write_json(output / "semantic_smoke.json", semantic)
            results["semantic_smoke_passed"] = semantic["passed"]
        if args.future_policy != "off":
            stats_path = output / "future_stats.json"
            if not stats_path.is_file():
                raise RuntimeError("future scheduler produced no telemetry")
            future_stats = json.loads(stats_path.read_text())
            if (future_stats.get("ticks", 0) < 1
                    or future_stats.get("fallback_exception", 0)
                    or future_stats.get("fallback_invalid_hint", 0)
                    or future_stats.get("telemetry_errors", 0)):
                raise RuntimeError("future scheduler telemetry indicates an unhealthy run")
            results["future_stats"] = future_stats
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
                if server is not None:
                    stop_server(server)
            finally:
                signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
