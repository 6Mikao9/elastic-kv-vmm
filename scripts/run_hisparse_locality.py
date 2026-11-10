"""Run a serial MiniMax HiSparse pilot on prepared real chat prompts."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import urllib.request
from uuid import uuid4

from adaptive_sparse.hisparse_access_trace import TraceLimits
from benchmarks.client import digest_ids, write_json
from scripts.run_case import HOST, PORT, ROOT, on_sigterm, stop_server
from scripts.run_hisparse_smoke import check_capacity, server_base_url, wait_healthy


TOP_K = 2048
CONTEXT = 16384
POOL_CAPACITY = 32768
HOST_RATIO = 2
WORKLOADS = ("chatbot", "agent", "long_document")
CAPTURE_RANKS = tuple(sorted(TraceLimits().ranks))
CAPTURE_LAYERS = tuple(sorted(TraceLimits().layers))


def prepare_rows(data, tokenizer, args, skipped):
    """Tokenize full conversations before allocating GPU resources."""
    seen = set()
    accepted = Counter()
    for line in data.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rid, workload = row["request_id"], row["workload"]
        if not isinstance(rid, str) or not rid or rid in seen or workload not in WORKLOADS:
            raise ValueError(f"invalid or repeated request identity {rid!r}")
        seen.add(rid)
        if accepted[workload] >= args.limit_per_workload:
            skipped[workload]["workload_limit"] += 1
            continue
        tokens = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                                add_generation_prompt=True)
        if not isinstance(tokens, list):
            tokens = tokens["input_ids"]
        if not all(type(token) is int and token >= 0 for token in tokens):
            raise ValueError(f"chat template returned invalid token IDs for {rid!r}")
        requested_cap = row.get("max_new_tokens", args.max_output_tokens)
        if type(requested_cap) is not int or requested_cap < 1:
            raise ValueError(f"invalid output limit for {rid!r}")
        output_cap = min(requested_cap, args.max_output_tokens)
        if len(tokens) < args.min_prompt_tokens:
            skipped[workload]["prompt_too_short"] += 1
        elif len(tokens) > args.max_prompt_tokens or len(tokens) + output_cap > CONTEXT:
            skipped[workload]["prompt_too_long"] += 1
        else:
            accepted[workload] += 1
            yield {key: row[key] for key in
                   ("request_id", "origin_id", "workload", "mode", "checkpoint", "source")} | {
                       "input_ids": tokens, "input_tokens": len(tokens),
                       "input_ids_sha256": digest_ids(tokens), "max_output_tokens": output_cap,
                   }


def submit(opener, base_url, row):
    body = {
        "rid": row["request_id"], "input_ids": row["input_ids"], "stream": False,
        "return_logprob": True, "logprob_start_len": -1, "top_logprobs_num": 0,
        "sampling_params": {"temperature": 0, "max_new_tokens": row["max_output_tokens"]},
    }
    request = urllib.request.Request(base_url + "/generate", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with opener.open(request, timeout=1800) as response:
        result = json.load(response)
    info = result["meta_info"]
    ids = result.get("output_ids")
    if ids is None and info.get("output_token_logprobs") is not None:
        ids = [entry[1] for entry in info["output_token_logprobs"]]
    if (info.get("id") != row["request_id"] or info["prompt_tokens"] != row["input_tokens"]
            or not isinstance(ids, list) or len(ids) != info["completion_tokens"]
            or len(ids) > row["max_output_tokens"]
            or any(type(token) is not int or token < 0 for token in ids)):
        raise RuntimeError(f"response identity or token count differs for {row['request_id']!r}")
    return {"output_ids": ids, "output_ids_sha256": digest_ids(ids),
            "output_tokens": len(ids), "finish_reason": info["finish_reason"]}


def capture_evidence(directory, completed):
    streams = {}
    files = []
    ranks_seen = set()
    truncations = Counter()
    for path in sorted(directory.glob("hisparse-rank*.jsonl")):
        with path.open(encoding="utf-8") as trace:
            header = json.loads(next(trace))
            rank = header["rank"]
            if (header["schema"] != "minimax-hisparse-access-v1"
                    or rank not in CAPTURE_RANKS or rank in ranks_seen
                    or tuple(header["layers"]) != CAPTURE_LAYERS):
                raise ValueError(f"unexpected capture header in {path.name}")
            ranks_seen.add(rank)
            files.append(path.name)
            for line in trace:
                event = json.loads(line)
                if event["type"] == "access":
                    rid, layer = event["rid"], event["layer"]
                    if (event["rank"] != rank or rid not in completed
                            or layer not in CAPTURE_LAYERS):
                        raise ValueError(f"unmatched capture access in {path.name}")
                    streams.setdefault((rid, rank, layer), []).append(event["step"])
                elif event["type"] == "truncation":
                    truncations[event["reason"]] += 1
    no_decode = [rid for rid, tokens in completed.items() if tokens <= 1]
    incomplete = []
    for rid, tokens in completed.items():
        expected = list(range(max(0, tokens - 1)))
        for rank in CAPTURE_RANKS:
            for layer in CAPTURE_LAYERS:
                observed = streams.get((rid, rank, layer), [])
                if observed != expected:
                    incomplete.append({
                        "request_id": rid, "rank": rank, "layer": layer,
                        "expected_steps": len(expected), "observed_steps": len(observed),
                    })
    return {
        "files": files, "access_rows": sum(map(len, streams.values())),
        "no_decode_expected": no_decode, "incomplete_request_rank_layers": incomplete,
        "truncation_markers": truncations,
        "request_rank_coverage_ok": bool(streams) and not incomplete and not truncations,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hot-buffer-tokens", type=int, choices=(2048, 4096, 8192), default=4096)
    parser.add_argument("--min-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-prompt-tokens", type=int, default=12288)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--limit-per-workload", type=int, default=8)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="tokenize locally and save prepared rows without starting the server")
    parser.add_argument("--sm80-bf16-projections", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.min_prompt_tokens <= args.max_prompt_tokens < CONTEXT:
        parser.error("invalid prompt token range")
    if not 1 <= args.max_output_tokens <= 512 or not 1 <= args.limit_per_workload <= 8:
        parser.error("output cap must be 1–512 and workload limit must be 1–8")

    if args.prepare_only and args.capture:
        parser.error("--prepare-only and --capture are mutually exclusive")
    output_root = args.output_root.resolve()
    private_root = ROOT / ".container-work" / "locality-study" / "runs"
    if output_root != Path("/output").resolve() and not output_root.is_relative_to(private_root):
        parser.error("output must be under .container-work/locality-study/runs")
    os.umask(0o077)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output_root / f"{timestamp}-{uuid4().hex[:8]}"
    output.mkdir(mode=0o700, parents=True)
    print(output, flush=True)

    model = args.model_path.resolve()
    dataset = args.dataset.resolve()
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("EKV_", "_ELASTIC_KV_"))}
    environment["PYTHONPATH"] = str(ROOT)
    environment.pop("CUDA_LAUNCH_BLOCKING", None)
    if args.sm80_bf16_projections:
        environment["EKV_MINIMAX_SM80_BF16"] = "1"
    if args.capture:
        environment.update(EKV_HISPARSE_TRACE_DIR=str(output / "access"),
                           EKV_HISPARSE_TRACE_RANKS="0,1", EKV_HISPARSE_TRACE_MAX_STEPS="512")

    config = {"top_k": TOP_K, "device_buffer_size": args.hot_buffer_tokens,
              "host_to_device_ratio": HOST_RATIO}
    command = [
        sys.executable, "-u", "-m", "elastic_kv.launch",
        "--elastic-runtime-profile", "minimax-hisparse", "--model-path", str(model),
        "--host", HOST, "--port", str(PORT), "--tp-size", "4",
        "--attention-backend", "triton", "--mm-attention-backend", "triton_attn",
        "--dtype", "bfloat16", "--kv-cache-dtype", "bfloat16", "--moe-runner-backend", "marlin",
        "--watchdog-timeout", "1800", "--random-seed", "29", "--page-size", "128",
        "--skip-server-warmup",
        "--max-total-tokens", str(POOL_CAPACITY), "--context-length", str(CONTEXT),
        "--chunked-prefill-size", "2048", "--max-running-requests", "4",
        "--mem-fraction-static", "0.92", "--disable-cuda-graph", "--disable-prefill-cuda-graph",
        "--disable-overlap-schedule", "--disable-radix-cache",
        "--enable-hisparse", "--hisparse-config", json.dumps(config),
    ]
    results = {"status": "running", "prepare_only": args.prepare_only,
               "diagnostic_capture": args.capture, "hisparse": config,
               "command": command, "gpu_uuids": os.environ.get("EKV_BENCH_GPU_UUIDS"),
               "shared_cache_reused": os.environ.get("EKV_LOCALITY_SHARED_CACHE_REUSED") == "1",
               "container_name": os.environ.get("EKV_LOCALITY_CONTAINER_NAME"),
               "prompt_token_range": [args.min_prompt_tokens, args.max_prompt_tokens],
               "max_output_tokens": args.max_output_tokens,
               "limit_per_workload": args.limit_per_workload,
               "requests": []}
    skipped = {name: Counter() for name in WORKLOADS}
    server = None
    step = "prepare"
    previous = signal.signal(signal.SIGTERM, on_sigterm)
    try:
        data = dataset.read_bytes()
        source_paths = [
            "scripts/run_hisparse_locality.py", "scripts/container_hisparse_locality.sh",
            "scripts/run_hisparse_smoke.py", "scripts/run_case.py", "benchmarks/client.py",
            "elastic_kv/launch.py", "elastic_kv/config.py", "adaptive_sparse/minimax_runtime.py",
        ]
        if args.capture:
            source_paths.append("adaptive_sparse/hisparse_access_trace.py")
        if args.sm80_bf16_projections:
            source_paths.append("adaptive_sparse/minimax_sm80.py")
        write_json(output / "source_manifest.json", {
            "dataset": str(dataset), "dataset_sha256": hashlib.sha256(data).hexdigest(),
            "model": str(model), "model_metadata_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in model.iterdir()
                if path.is_file() and (path.name.endswith((".json", ".jinja"))
                                      or path.name.startswith("tokenizer"))
            },
            "runner_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                              for name in source_paths},
        })
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        rows = list(prepare_rows(data.decode("utf-8"), tokenizer, args, skipped))
        prepared_counts = Counter(row["workload"] for row in rows)
        results["prepared_by_workload"] = prepared_counts
        missing_workloads = [name for name in WORKLOADS if not prepared_counts[name]]
        results["missing_workloads"] = missing_workloads
        if missing_workloads:
            print(f"Workload coverage incomplete. Missing {', '.join(missing_workloads)}.", file=sys.stderr)
        if args.prepare_only:
            with (output / "prepared.jsonl").open("x", encoding="utf-8") as prepared:
                for row in rows:
                    prepared.write(json.dumps(row) + "\n")
            results["status"] = "prepared_partial" if missing_workloads else "prepared"
            return
        if not rows:
            raise RuntimeError("no prompts passed the configured length filter")

        step = "startup"
        base_url = server_base_url(HOST, PORT)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with (output / "server.log").open("xb") as log:
            server = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True, env=environment)
        wait_healthy(server, opener, base_url)
        step = "capacity"
        results["capacity"] = check_capacity(opener, POOL_CAPACITY, HOST_RATIO, base_url)
        completed = {}
        with (output / "requests.jsonl").open("x", encoding="utf-8", buffering=1) as records:
            for row in rows:
                step = row["request_id"]
                response = submit(opener, base_url, row)
                records.write(json.dumps(row | response) + "\n")
                completed[step] = response["output_tokens"]
                results["requests"].append(step)
        results["status"] = "partial" if missing_workloads else "ok"
        if args.capture:
            step = "capture_verification"
            evidence = capture_evidence(output / "access", completed)
            results["capture_verification"] = evidence
            if not evidence["request_rank_coverage_ok"]:
                results["status"] = "partial"
                print("Capture coverage incomplete. See results.json.", file=sys.stderr)
    except BaseException as error:
        results.update(status="failed", failed_step=step,
                       error_type=type(error).__name__, error=str(error))
        raise
    finally:
        results["skipped_by_workload"] = skipped
        try:
            if server is not None:
                stop_server(server)
        finally:
            signal.signal(signal.SIGTERM, previous)
            write_json(output / "results.json", results)


if __name__ == "__main__":
    main()
