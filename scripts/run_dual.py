#!/usr/bin/env python3
"""Measure load reversal across two isolated SGLang servers on one GPU."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts import run_case
except ImportError:  # Direct execution from the scripts directory.
    import run_case

sys.path.insert(0, str(run_case.ROOT / "benchmarks"))
from client import (
    STATS_NUMBERS,
    MemorySampler,
    distribution,
    generate,
    safe_error_message,
    write_json,
)

PORTS = {"a": 31080, "b": 31081}
SCENARIOS = {"a-hot": ("a",), "b-hot": ("b",), "both-active": ("a", "b")}
FAIR_MODES = ("off", "observe", "enforce")
SOURCE_ROOTS = ("elastic_kv", "adaptive_sparse", "native", "benchmarks", "scripts")
SOURCE_SUFFIXES = {".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".cuh", ".sh"}
SOURCE_INPUTS = ("pyproject.toml", "requirements.txt")
STATS_PREFIXES = (
    "activation_",
    "global_broker_",
    "physical_global_broker_",
    "precommit_",
)
MAX_BROKER_FAILURE_RATIO = 0.25
PROBE_CASES = tuple(
    (input_tokens, output_tokens, seed)
    for input_tokens, output_tokens in ((32, 16), (1024, 16), (2048, 8))
    for seed in (0, 1)
)


def source_hashes():
    """Hash every serving source that can affect a dual-instance result."""
    paths = set()
    for root_name in SOURCE_ROOTS:
        root = run_case.ROOT / root_name
        if not root.exists():
            continue
        paths.update(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
        )
    paths.update(
        path for name in SOURCE_INPUTS
        if (path := run_case.ROOT / name).is_file()
    )
    hashes = {
        path.relative_to(run_case.ROOT).as_posix():
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    required = {"elastic_kv/adapter.py", "elastic_kv/config.py"}
    missing = sorted(required - set(hashes))
    if missing:
        raise FileNotFoundError(
            "required serving source is missing: " + ", ".join(missing)
        )
    return hashes


def _command_bytes(command, cwd=None):
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    return result.stdout


def _command_output(command, cwd=None):
    output = _command_bytes(command, cwd)
    return None if output is None else output.decode("utf-8", errors="replace").strip()


def git_provenance(root=None):
    """Identify the exact checkout including its tracked and untracked diffs."""
    root = run_case.ROOT if root is None else Path(root)
    commit = _command_output(("git", "rev-parse", "HEAD"), root)
    status_bytes = _command_bytes(
        ("git", "-c", "core.quotepath=false", "status", "--porcelain=v1"),
        root,
    )
    diff_bytes = _command_bytes(("git", "diff", "--binary", "HEAD"), root)
    status = (
        None if status_bytes is None
        else status_bytes.decode("utf-8", errors="replace").strip()
    )
    untracked = None
    if status:
        names = [
            line[3:] for line in status.splitlines()
            if line.startswith("?? ") and line[3:]
        ]
        digest = hashlib.sha256()
        for name in sorted(names):
            path = root / name
            if path.is_file():
                digest.update(name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(path.read_bytes())
        untracked = digest.hexdigest() if names else None
    return {
        "available": (
            commit is not None and status_bytes is not None and diff_bytes is not None
        ),
        "commit": commit,
        "dirty": None if status_bytes is None else bool(status),
        "status": status or "",
        "diff_available": diff_bytes is not None,
        "diff_sha256": (
            None if diff_bytes is None
            else hashlib.sha256(diff_bytes).hexdigest()
        ),
        "untracked_sha256": untracked,
    }


def source_manifest():
    files = source_hashes()
    manifest = git_provenance()
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    manifest["tree_sha256"] = digest.hexdigest()
    manifest["files"] = files
    return manifest


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def model_kv_geometry(path):
    """Return dense KV bytes per token for TP1 from a local config snapshot."""
    config_path = Path(path) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        layers = int(config["num_hidden_layers"])
        hidden_size = int(config["hidden_size"])
        attention_heads = int(config["num_attention_heads"])
        kv_heads = int(config.get("num_key_value_heads", attention_heads))
        head_dim = int(config.get("head_dim", hidden_size // attention_heads))
        dtype_name = str(
            config.get("torch_dtype", config.get("dtype", "bfloat16"))
        ).lower()
        dtype_bytes = {
            "float16": 2, "bfloat16": 2, "float32": 4,
        }.get(dtype_name)
        if dtype_bytes is None:
            raise ValueError(f"unsupported cache dtype: {dtype_name}")
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"error_type": type(exc).__name__}
    return {
        "bytes_per_token": 2 * layers * kv_heads * head_dim * dtype_bytes,
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "dtype": dtype_name,
        "dtype_bytes": dtype_bytes,
        "tensor_parallel_size": 1,
        "formula": "2 * layers * kv_heads * head_dim * dtype_bytes / TP",
    }


def _torch_runtime():
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return None
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    return {
        "version": getattr(torch, "__version__", None),
        "cuda_runtime": getattr(getattr(torch, "version", None), "cuda", None),
        "cudnn": cudnn.version() if cudnn is not None else None,
    }


def model_identity(path):
    """Record a local snapshot without hashing multi-gigabyte weight files."""
    resolved = path.resolve()
    revision = resolved.name
    if resolved.parent.name != "snapshots":
        revision = os.environ.get("HF_MODEL_REVISION")
    files = {}
    for child in sorted(resolved.iterdir()):
        if child.is_file():
            stat = child.stat()
            files[child.name] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    config_hashes = {}
    for name in ("config.json", "generation_config.json", "tokenizer_config.json"):
        candidate = resolved / name
        if candidate.is_file():
            config_hashes[name] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    return {
        "path": str(resolved),
        "revision": revision,
        "files": files,
        "config_sha256": config_hashes,
        "kv_geometry": model_kv_geometry(resolved),
    }


def runtime_provenance(models):
    gpu_query = _command_output(
        (
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        )
    )
    gpu_uuid = os.environ.get("BENCH_GPU_UUID")
    gpu_rows = []
    if gpu_query:
        for row in gpu_query.splitlines():
            fields = [field.strip() for field in row.split(",")]
            if len(fields) == 4 and (gpu_uuid is None or fields[0] == gpu_uuid):
                gpu_rows.append({
                    "uuid": fields[0],
                    "name": fields[1],
                    "driver_version": fields[2],
                    "memory_total_mib": fields[3],
                })
    native_path = Path(
        os.environ.get(
            "ELASTIC_VMM_LIBRARY",
            run_case.ROOT / "native" / "libelastic_vmm.so",
        )
    ).expanduser()
    native = {"path": str(native_path.resolve())}
    if native_path.is_file():
        native["sha256"] = hashlib.sha256(native_path.read_bytes()).hexdigest()
        native["size"] = native_path.stat().st_size
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "container": {
            "image": os.environ.get("SGLANG_IMAGE") or os.environ.get("EKV_CONTAINER_IMAGE"),
            "image_id": os.environ.get("EKV_IMAGE_ID") or os.environ.get("EKV_CONTAINER_IMAGE_ID"),
            "image_digest": (
                os.environ.get("EKV_IMAGE_DIGEST")
                or os.environ.get("EKV_CONTAINER_IMAGE_DIGEST")
            ),
        },
        "packages": {
            name: _package_version(name)
            for name in (
                "sglang", "torch", "triton", "nvidia-ml-py",
                "cupy-cuda12x", "nvidia-cuda-runtime-cu12",
            )
        },
        "torch": _torch_runtime(),
        "gpu": gpu_rows,
        "native_library": native,
        "models": {instance: model_identity(path) for instance, path in models.items()},
    }


def sqlite_state(path):
    """Describe SQLite journaling without mutating the broker database."""
    if path is None or not Path(path).exists():
        return {"path": None if path is None else str(path), "exists": False}
    database = Path(path)
    connection = sqlite3.connect(
        database.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.2
    )
    try:
        journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0])
    finally:
        connection.close()
    sidecars = {}
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(str(database) + suffix)
        if sidecar.exists():
            sidecar_stat = sidecar.stat()
            sidecars[sidecar.name] = {
                "size": sidecar_stat.st_size,
                "mtime_ns": sidecar_stat.st_mtime_ns,
            }
    return {
        "path": str(database.resolve()),
        "exists": True,
        "journal_mode": journal_mode,
        "sidecars": sidecars,
    }


def broker_environment(
    path,
    mib,
    fair_mode,
    tenant,
    tenant_weight,
    stale_seconds,
    demand_ttl_seconds,
    wait_ms,
):
    """Return one process's explicit global-broker configuration."""
    if path is None:
        return {}
    result = {
        "EKV_GLOBAL_BROKER_PATH": str(path),
        "EKV_GLOBAL_BROKER_MIB": str(mib),
        "EKV_GLOBAL_BROKER_FAIR_MODE": fair_mode,
        "EKV_GLOBAL_BROKER_TENANT_WEIGHT": str(tenant_weight),
        "EKV_GLOBAL_BROKER_STALE_SECONDS": str(stale_seconds),
        "EKV_GLOBAL_BROKER_DEMAND_TTL_SECONDS": str(demand_ttl_seconds),
        "EKV_GLOBAL_BROKER_WAIT_MS": str(wait_ms),
    }
    if tenant is not None:
        result["EKV_GLOBAL_BROKER_TENANT"] = tenant
    return result


def validate_broker_options(
    parser,
    *,
    mode,
    path,
    mib,
    fair_mode,
    tenants,
    weights,
    stale_seconds,
    demand_ttl_seconds,
    wait_ms,
):
    if mib < 0:
        parser.error("global broker capacity must not be negative")
    if path is None and mib != 0:
        parser.error("global broker capacity requires an explicit state path")
    if path is not None and mib <= 0:
        parser.error("global broker requires positive capacity")
    if path is not None and mode != "v2":
        parser.error("global broker requires the VMM-backed v2 mode")
    if fair_mode not in FAIR_MODES:
        parser.error("global broker fair mode must be off, observe or enforce")
    if fair_mode != "off" and path is None:
        parser.error("fair global broker requires an enabled broker")
    if any(value is not None for value in tenants) and not all(
        value is not None for value in tenants
    ):
        parser.error("set both --broker-tenant-a and --broker-tenant-b")
    if any(
        value is not None and (not value.strip() or len(value) > 128)
        for value in tenants
    ):
        parser.error("broker tenant ids must be nonempty and at most 128 characters")
    if fair_mode != "off":
        if any(value is None for value in tenants) or tenants[0] == tenants[1]:
            parser.error("fair global broker requires two distinct tenants")
    if (stale_seconds <= 0 or demand_ttl_seconds <= 0
            or not 0 <= wait_ms <= 5000):
        parser.error("broker timing values must be positive and bounded")
    if any(
        not math.isfinite(float(weight)) or not 0 < weight <= 1e6
        for weight in weights
    ):
        parser.error("broker tenant weights must be finite and in (0, 1e6]")


def weighted_targets(capacity_bytes, weights):
    """Mirror the broker's integer target split for read-only telemetry."""
    if not weights:
        return {}
    total_weight = math.fsum(weights.values())
    tenants = sorted(weights)
    targets = {
        tenant: math.floor(capacity_bytes * weights[tenant] / total_weight)
        for tenant in tenants
    }
    remainder = capacity_bytes - sum(targets.values())
    for tenant in tenants[:remainder]:
        targets[tenant] += 1
    for tenant in tenants:
        if targets[tenant] == 0 and capacity_bytes >= len(tenants):
            donor = max(tenants, key=lambda item: targets[item])
            if targets[donor] > 1:
                targets[donor] -= 1
                targets[tenant] = 1
    return targets


def read_broker_snapshot(path):
    """Read broker state without creating an owner or taking a write lock."""
    if path is None:
        return None
    if not Path(path).exists():
        return {"status": "uninitialized"}
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    connection = None
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=0.1)
        connection.execute("BEGIN")
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        now = time.time()
        capacity = int(metadata["capacity_bytes"])
        usage = {
            str(tenant): int(bytes_used)
            for tenant, bytes_used in connection.execute(
                "SELECT owners.tenant_id, SUM(leases.bytes) "
                "FROM leases JOIN owners ON owners.owner=leases.owner "
                "WHERE owners.tenant_id IS NOT NULL GROUP BY owners.tenant_id"
            )
        }
        demands = {
            str(tenant): int(requested)
            for tenant, requested in connection.execute(
                "SELECT tenant_id, SUM(requested_bytes) FROM demands "
                "WHERE expires_at > ? GROUP BY tenant_id", (now,)
            )
        }
        active = set(usage) | set(demands)
        weights = {
            str(tenant): float(weight)
            for tenant, weight in connection.execute(
                "SELECT tenant_id, weight FROM tenants"
            )
            if tenant in active
        }
        missing_weights = sorted(active - set(weights))
        if missing_weights:
            return {
                "status": "configuration_error",
                "missing_tenant_weights": missing_weights,
            }
        targets = weighted_targets(capacity, weights)
        starved = sorted(
            tenant for tenant in demands
            if targets.get(tenant, 0) > usage.get(tenant, 0)
        )
        yielding = sorted(
            tenant for tenant in targets
            if usage.get(tenant, 0) >= targets[tenant]
            and any(other != tenant for other in starved)
        )
        outstanding = int(
            connection.execute(
                "SELECT COALESCE(SUM(bytes), 0) FROM leases"
            ).fetchone()[0]
        )
        owners = {
            str(owner): {
                "pid": int(pid),
                "tenant_id": tenant,
                "last_seen": float(last_seen),
                "leased_bytes": int(bytes_used),
                "lease_count": int(lease_count),
            }
            for owner, pid, tenant, last_seen, bytes_used, lease_count in connection.execute(
                "SELECT owners.owner, owners.pid, owners.tenant_id, owners.last_seen, "
                "COALESCE(SUM(leases.bytes), 0), COUNT(leases.lease_key) "
                "FROM owners LEFT JOIN leases ON leases.owner=owners.owner "
                "GROUP BY owners.owner ORDER BY owners.owner"
            )
        }
        return {
            "status": "ready",
            "capacity_bytes": capacity,
            "outstanding_bytes": outstanding,
            "available_bytes": capacity - outstanding,
            "fair_mode": metadata.get("fair_mode", "off"),
            "tenant_usage_bytes": usage,
            "tenant_demand_bytes": demands,
            "fair_target_bytes": targets,
            "tenant_deficit_bytes": {
                tenant: max(0, target - usage.get(tenant, 0))
                for tenant, target in targets.items()
            },
            "starved_tenants": starved,
            "tenants_over_target_with_demand": yielding,
            "overcommitted": outstanding > capacity,
            "owners": owners,
        }
    except (FileNotFoundError, sqlite3.Error, KeyError, TypeError, ValueError) as exc:
        # Record a stable error class. Snapshot sampling must not fabricate zero
        # usage and must not leak paths or SQLite text into the result manifest.
        return {"status": "snapshot_error", "error_type": type(exc).__name__}
    finally:
        if connection is not None:
            connection.close()


def adapter_stats(raw):
    """Keep every numeric adapter and admission counter safe to serialize."""
    if not isinstance(raw, dict):
        raise ValueError("adapter stats must contain a JSON object")
    if type(raw.get("backed_bytes")) is not int or raw["backed_bytes"] < 0:
        raise ValueError("adapter stats lack a nonnegative integer backed_bytes")
    result = {}
    for key, value in raw.items():
        if key in STATS_NUMBERS or key.startswith(STATS_PREFIXES):
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                result[key] = value
    for key in ("mode", "layout", "admission_policy"):
        value = raw.get(key)
        if isinstance(value, str) and value.replace("_", "").replace("-", "").isalnum():
            result[key] = value
    broker = raw.get("global_broker")
    if isinstance(broker, dict):
        result["global_broker"] = {
            key: value for key, value in broker.items()
            if isinstance(key, str)
            and (
                type(value) in (int, float, bool, str)
                or (
                    isinstance(value, (list, tuple))
                    and all(isinstance(item, str) for item in value)
                )
            )
        }
    if type(result.get("live_tokens")) is not int or result["live_tokens"] < 0:
        raise ValueError("adapter stats lack a nonnegative integer live_tokens")
    return result


class DualSampler(MemorySampler):
    def __init__(self, stats_paths, broker_path=None):
        super().__init__()
        self.stats_paths = stats_paths
        self.broker_path = broker_path
        self.broker_sample_count = 0
        self.broker_failure_count = 0

    def take(self):
        with self.lock:
            stats = {}
            for instance, path in self.stats_paths.items():
                raw = json.loads(path.read_text(encoding="utf-8"))
                stats[instance] = adapter_stats(raw)
            broker = read_broker_snapshot(self.broker_path)
            if self.broker_path is not None:
                self.broker_sample_count += 1
                if not broker or broker.get("status") != "ready":
                    self.broker_failure_count += 1
            self.samples.append({
                "offset_s": time.perf_counter() - self.origin,
                "device_used_bytes": int(self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used),
                "instances": stats,
                "broker": broker,
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


def read_adapter_stats(path):
    return adapter_stats(json.loads(path.read_text(encoding="utf-8")))


def server_launch_spec(model, mode, port, stats, extra_environment=None):
    command = [
        sys.executable, "-m", "elastic_kv.launch",
        "--model-path", str(model.resolve()), "--host", "127.0.0.1",
        "--port", str(port), "--page-size", "1", "--attention-backend", "triton",
        "--disable-cuda-graph", "--disable-overlap-schedule", "--disable-radix-cache",
        "--max-total-tokens", "32768", "--context-length", "8192",
        "--chunked-prefill-size", "2048", "--mem-fraction-static", "0.5",
    ]
    environment = {
        "EKV_MODE": mode,
        "EKV_STATS_PATH": str(stats),
        "PYTHONUNBUFFERED": "1",
        **(extra_environment or {}),
    }
    return {
        "command": command,
        "environment": environment,
        "cwd": str(run_case.ROOT),
    }


def start_server(launch_spec, log_path):
    with log_path.open("xb") as log:
        return subprocess.Popen(
            launch_spec["command"],
            cwd=launch_spec["cwd"],
            env={**os.environ, **launch_spec["environment"]},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def scenario_order(repeat):
    order = ("a-hot", "b-hot", "both-active")
    return order if repeat % 2 else ("b-hot", "a-hot", "both-active")


def phase_stem(scenario, repeat, warmup=False):
    prefix = "warmup" if warmup else f"repeat-{repeat:02d}"
    return f"{prefix}-{scenario}"


def build_phase_jobs(scenario, repeat, workload, warmup=False):
    request_count = min(2, workload["requests_per_instance"]) if warmup else (
        workload["requests_per_instance"]
    )
    return [
        (instance, {
            "request_id": (
                f"{'warmup' if warmup else 'measured'}/{scenario}/{repeat}/"
                f"{instance}/{index}"
            ),
            "instance": instance,
            "input_tokens": workload["input_tokens"],
            "requested_output_tokens": workload["output_tokens"],
        })
        for instance in SCENARIOS[scenario]
        for index in range(request_count)
    ]


def estimate_phase_kv_bytes(scenario, workload, geometry):
    instances = {}
    for instance in SCENARIOS[scenario]:
        value = geometry[instance].get("bytes_per_token")
        if value is None:
            raise ValueError(f"model {instance} has no dense KV geometry")
        instances[instance] = (
            workload["requests_per_instance"]
            * (workload["input_tokens"] + workload["output_tokens"])
            * value
        )
    return instances


def wait_for_drain(stats_paths, timeout=30):
    """Wait for KV lifecycle completion instead of treating sleep as a drain."""
    started = time.perf_counter()
    if not stats_paths:
        time.sleep(0.3)
        return 0.3
    deadline = started + timeout
    while time.perf_counter() < deadline:
        stats = {
            instance: read_adapter_stats(path)
            for instance, path in stats_paths.items()
        }
        if not any(value["live_tokens"] for value in stats.values()):
            return time.perf_counter() - started
        time.sleep(0.05)
    raise TimeoutError(f"KV lifecycle did not drain within {timeout} seconds")


def wait_for_broker_owners(path, required_tenants=(), timeout=30):
    if path is None:
        return None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = read_broker_snapshot(path)
        if snapshot and snapshot.get("status") == "ready":
            owners = snapshot.get("owners") or {}
            tenant_ids = {
                owner.get("tenant_id") for owner in owners.values()
                if owner.get("tenant_id") is not None
            }
            required = {tenant for tenant in required_tenants if tenant is not None}
            if owners and (not required or required <= tenant_ids):
                return snapshot
        time.sleep(0.1)
    raise RuntimeError("broker has no complete set of live tenant owners after warmup")


def run_synchronized(jobs, worker, timeout=30, delay_ms=None):
    """Release every request dispatch point only after all workers are ready."""
    barrier = threading.Barrier(len(jobs))
    starts = []
    starts_lock = threading.Lock()

    def dispatch(job):
        barrier.wait(timeout=timeout)
        delay = 0 if delay_ms is None else delay_ms(job)
        if delay:
            time.sleep(delay / 1000.0)
        with starts_lock:
            starts.append(time.perf_counter())
        return worker(job)

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(dispatch, job) for job in jobs]
        records = [future.result() for future in futures]
    return records, starts


def phase(
    scenario,
    repeat,
    stats_paths,
    workload,
    kv_geometry,
    warmup=False,
    broker_path=None,
    broker_capacity_bytes=None,
):
    jobs = build_phase_jobs(scenario, repeat, workload, warmup=warmup)
    sampler = DualSampler(stats_paths, broker_path)
    with sampler:
        requests, starts = run_synchronized(
            jobs,
            lambda job: generate(
                f"http://127.0.0.1:{PORTS[job[0]]}/generate", job[1], 120
            ),
            delay_ms=lambda job: workload["arrival_delay_ms"].get(job[0], 0),
        )
        finished = time.perf_counter()
        started = min(starts)
        dispatch_skew_s = max(starts) - started
        elapsed = finished - started
        active_start = started - sampler.origin
        active_end = finished - sampler.origin
        drain_s = wait_for_drain(stats_paths)
        time.sleep(0.05)
        sampler.take()

    idle = sampler.samples[-1]
    active_samples = [sample for sample in sampler.samples
                      if active_start <= sample["offset_s"] <= active_end]
    if not active_samples:
        raise RuntimeError("no memory samples in the active request window")
    broker_statuses = Counter(
        sample["broker"].get("status", "missing")
        for sample in active_samples
        if sample["broker"] is not None
    )
    broker_failures = sum(
        count for status, count in broker_statuses.items() if status != "ready"
    )
    if broker_path is not None:
        if broker_statuses["ready"] == 0:
            raise RuntimeError("no initialized broker snapshot in the active request window")
        if broker_failures / len(active_samples) > MAX_BROKER_FAILURE_RATIO:
            raise RuntimeError("broker snapshot failure rate exceeded 25%")
    if any(
        sample["broker"]
        and sample["broker"].get("status") == "ready"
        and sample["broker"].get("overcommitted")
        for sample in active_samples
    ):
        raise RuntimeError("broker exceeded its configured capacity")
    if any(stats["live_tokens"] for stats in idle["instances"].values()):
        raise RuntimeError(f"live tokens remain after {scenario} repeat {repeat}")
    for request in requests:
        request.pop("_started_perf_s")
        request.pop("output_ids", None)
    output_tokens = sum(request["output_tokens"] for request in requests)
    estimated_kv_bytes = estimate_phase_kv_bytes(scenario, workload, kv_geometry)
    estimated_kv_bytes_total = sum(estimated_kv_bytes.values())
    return {
        "status": "ok", "scenario": scenario, "repeat": repeat,
        "warmup": warmup, "elapsed_s": elapsed, "drain_s": drain_s,
        "dispatch_skew_s": dispatch_skew_s,
        "input_tokens": sum(request["input_tokens"] for request in requests),
        "output_tokens": output_tokens, "output_tokens_per_s": output_tokens / elapsed,
        "estimated_kv_bytes": estimated_kv_bytes,
        "estimated_kv_bytes_total": estimated_kv_bytes_total,
        "estimated_broker_capacity_bytes": broker_capacity_bytes,
        "estimated_phase_to_broker_capacity_ratio": (
            estimated_kv_bytes_total / broker_capacity_bytes
            if broker_capacity_bytes else None
        ),
        "requests": requests, "samples": sampler.samples,
        "active_window_s": [active_start, active_end],
        "active_sample_count": len(active_samples),
        "peak_device_used_bytes": max(sample["device_used_bytes"] for sample in active_samples),
        "broker_snapshot_status": dict(sorted(broker_statuses.items())),
        "broker_failure_ratio": (
            broker_failures / len(active_samples) if broker_path is not None else None
        ),
        "idle": idle,
    }


def summarize_phase(result):
    active_start, active_end = result["active_window_s"]
    active_samples = [
        sample for sample in result["samples"]
        if active_start <= sample["offset_s"] <= active_end
    ]
    instances = {}
    for instance in PORTS:
        values = [
            sample["instances"][instance]
            for sample in active_samples
            if instance in sample["instances"]
        ]
        if values:
            instances[instance] = {
                "peak_backed_bytes": max(value["backed_bytes"] for value in values),
                "peak_live_tokens": max(value["live_tokens"] for value in values),
                "final_counters": values[-1],
            }
    requests = result["requests"]
    return {
        "name": result["scenario"],
        "scenario": result["scenario"],
        "status": result["status"],
        "repeat": result["repeat"],
        "active_window_s": result["active_window_s"],
        "idle": result["idle"],
        "summary": {
            "elapsed_s": result["elapsed_s"],
            "drain_s": result["drain_s"],
            "dispatch_skew_s": result["dispatch_skew_s"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "output_tokens_per_s": result["output_tokens_per_s"],
            "estimated_kv_bytes": result["estimated_kv_bytes"],
            "estimated_kv_bytes_total": result["estimated_kv_bytes_total"],
            "estimated_broker_capacity_bytes": result["estimated_broker_capacity_bytes"],
            "estimated_phase_to_broker_capacity_ratio": (
                result["estimated_phase_to_broker_capacity_ratio"]
            ),
            "ttft_s": distribution([item["ttft_s"] for item in requests]),
            "tpot_s": distribution([item["tpot_s"] for item in requests]),
            "end_to_end_s": distribution([item["end_to_end_s"] for item in requests]),
            "active_sample_count": result["active_sample_count"],
            "peak_device_used_bytes": result["peak_device_used_bytes"],
            "broker_snapshot_status": result["broker_snapshot_status"],
            "broker_failure_ratio": result["broker_failure_ratio"],
            "instances": instances,
        },
    }


def write_phase_artifacts(output, result):
    stem = phase_stem(result["scenario"], result["repeat"], result["warmup"])
    write_json(output / f"{stem}.requests.json", result["requests"])
    write_json(output / f"{stem}.samples.json", result["samples"])
    return summarize_phase(result)


def run_correctness_probes(output):
    """Run a fixed non-streaming matrix outside the timed serving phases."""
    jobs = []
    for instance in PORTS:
        for input_tokens, output_tokens, seed in PROBE_CASES:
            jobs.append((instance, {
                "request_id": (
                    f"correctness/{instance}/{input_tokens}/{output_tokens}/seed-{seed}"
                ),
                "instance": instance,
                "input_tokens": input_tokens,
                "requested_output_tokens": output_tokens,
            }))

    def probe(job):
        instance, spec = job
        record = generate(
            f"http://127.0.0.1:{PORTS[instance]}/generate",
            spec,
            120,
            stream=False,
        )
        if not record["output_ids_complete"] or not record["output_ids_sha256"]:
            raise RuntimeError("non-streaming correctness probe lacks complete output IDs")
        record.pop("_started_perf_s", None)
        return record

    requests, starts = run_synchronized(jobs, probe)
    payload = {
        "status": "complete",
        "dispatch_skew_s": max(starts) - min(starts),
        "requests": requests,
    }
    write_json(output / "correctness.json", payload)
    instances = {}
    for instance in PORTS:
        selected = [item for item in requests if item["instance"] == instance]
        instances[instance] = {
            "probe_count": len(selected),
            "output_ids_sha256": {
                item["request_id"]: item["output_ids_sha256"] for item in selected
            },
        }
    return {
        "status": "complete",
        "path": "correctness.json",
        "dispatch_skew_s": payload["dispatch_skew_s"],
        "instances": instances,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", required=True, type=Path, help="local 0.6B snapshot")
    parser.add_argument("--model-b", required=True, type=Path, help="local 4B snapshot")
    parser.add_argument("--mode", required=True, choices=("baseline", "v2"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--broker-path", type=Path,
                        help="fresh SQLite state file shared by both servers")
    parser.add_argument("--broker-mib", type=int, default=0)
    parser.add_argument("--broker-fair-mode", choices=FAIR_MODES, default="off")
    parser.add_argument("--broker-tenant-a")
    parser.add_argument("--broker-tenant-b")
    parser.add_argument("--broker-weight-a", type=float, default=1.0)
    parser.add_argument("--broker-weight-b", type=float, default=1.0)
    parser.add_argument("--broker-stale-seconds", type=int, default=120)
    parser.add_argument("--broker-demand-ttl-seconds", type=float, default=10.0)
    parser.add_argument("--broker-wait-ms", type=int, default=0)
    parser.add_argument("--requests-per-instance", type=int, default=4)
    parser.add_argument("--input-tokens", type=int, default=1024)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--arrival-delay-a-ms", type=int, default=0)
    parser.add_argument("--arrival-delay-b-ms", type=int, default=0)
    parser.add_argument(
        "--skip-correctness-probe",
        action="store_true",
        help="skip the fixed non-streaming output-ID matrix (debug runs only)",
    )
    args = parser.parse_args()
    validate_broker_options(
        parser,
        mode=args.mode,
        path=args.broker_path,
        mib=args.broker_mib,
        fair_mode=args.broker_fair_mode,
        tenants=(args.broker_tenant_a, args.broker_tenant_b),
        weights=(args.broker_weight_a, args.broker_weight_b),
        stale_seconds=args.broker_stale_seconds,
        demand_ttl_seconds=args.broker_demand_ttl_seconds,
        wait_ms=args.broker_wait_ms,
    )
    models = {"a": args.model_a, "b": args.model_b}
    if not all(model.is_dir() for model in models.values()):
        parser.error("both model snapshots must exist locally")
    if not 1 <= args.requests_per_instance <= 32:
        parser.error("--requests-per-instance must be in [1, 32]")
    if not 1 <= args.input_tokens <= 8192:
        parser.error("--input-tokens must be in [1, 8192]")
    if not 1 <= args.output_tokens <= 8191:
        parser.error("--output-tokens must be in [1, 8191]")
    if args.input_tokens + args.output_tokens > 8192:
        parser.error("input plus output tokens must fit the 8192-token context")
    if args.requests_per_instance * (
        args.input_tokens + args.output_tokens
    ) > 32768:
        parser.error("configured requests exceed --max-total-tokens=32768")
    if not 0 <= args.arrival_delay_a_ms <= 60000:
        parser.error("--arrival-delay-a-ms must be in [0, 60000]")
    if not 0 <= args.arrival_delay_b_ms <= 60000:
        parser.error("--arrival-delay-b-ms must be in [0, 60000]")
    output = args.output_dir.resolve()
    broker_path = None if args.broker_path is None else args.broker_path.resolve()
    if broker_path is not None and broker_path.exists():
        parser.error("broker state must not exist; use a fresh output directory")

    manifest = source_manifest()
    if not manifest["available"]:
        parser.error("Git source identity is required for a reproducible dual run")
    runtime = runtime_provenance(models)
    kv_geometry = {
        instance: metadata["kv_geometry"]
        for instance, metadata in runtime["models"].items()
    }
    invalid_geometry = sorted(
        instance for instance, value in kv_geometry.items()
        if value.get("bytes_per_token") is None
    )
    if invalid_geometry:
        parser.error(
            "dense KV geometry is unavailable for model(s): "
            + ", ".join(invalid_geometry)
        )
    workload = {
        "requests_per_instance": args.requests_per_instance,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "arrival_delay_ms": {
            "a": args.arrival_delay_a_ms,
            "b": args.arrival_delay_b_ms,
        },
    }
    broker_capacity_bytes = (
        args.broker_mib * 1024 * 1024 if broker_path is not None else None
    )
    output.mkdir(parents=True, exist_ok=False)
    stats_paths = {instance: output / f"{instance}.stats.json" for instance in PORTS}
    sampled_stats = stats_paths if args.mode == "v2" else {}
    launches = {}
    for instance, port in PORTS.items():
        tenant = args.broker_tenant_a if instance == "a" else args.broker_tenant_b
        weight = args.broker_weight_a if instance == "a" else args.broker_weight_b
        launches[instance] = server_launch_spec(
            models[instance],
            args.mode,
            port,
            stats_paths[instance],
            broker_environment(
                broker_path,
                args.broker_mib,
                args.broker_fair_mode,
                tenant,
                weight,
                args.broker_stale_seconds,
                args.broker_demand_ttl_seconds,
                args.broker_wait_ms,
            ),
        )
    results = {
        "status": "running",
        "arm": args.mode,
        "configuration": {
            "mode": args.mode,
            "server_ports": dict(PORTS),
            "max_total_tokens_per_instance": 32768,
            "workload": workload,
            "kv_geometry": kv_geometry,
            "adapter_stats_available": bool(sampled_stats),
            "scope": (
                "Two SGLang servers share one GPU with an optional cooperative "
                "VMM broker; the broker does not preempt another live process."
            ),
            "broker": {
                "enabled": broker_path is not None,
                "path": None if broker_path is None else str(broker_path),
                "capacity_mib": args.broker_mib,
                "fair_mode": args.broker_fair_mode,
                "tenants": {
                    "a": args.broker_tenant_a,
                    "b": args.broker_tenant_b,
                },
                "weights": {
                    "a": args.broker_weight_a,
                    "b": args.broker_weight_b,
                },
                "sqlite_initial": sqlite_state(broker_path),
            },
            "launches": launches,
        },
        "phases": [],
        "warmups": [],
        "violations": [],
        "correctness": None,
        "source_manifest": manifest,
        "runtime": runtime,
        # Keep the earlier flat keys for existing plotting utilities.
        "mode": args.mode,
        "models": {"a": "0.6B", "b": "4B"},
        "server_ports": dict(PORTS),
        "server_pids": {},
        "max_total_tokens_per_instance": 32768,
        "workload": workload,
        "adapter_stats_available": bool(sampled_stats),
        "source_sha256": manifest["tree_sha256"],
        "broker": {
            "enabled": broker_path is not None,
            "path": None if broker_path is None else str(broker_path),
            "capacity_mib": args.broker_mib,
            "fair_mode": args.broker_fair_mode,
        },
    }
    result_path = output / "results.json"
    write_json(result_path, results)
    servers = []
    previous_handler = signal.signal(signal.SIGTERM, run_case.on_sigterm)
    try:
        for instance, port in PORTS.items():
            server = start_server(launches[instance], output / f"{instance}.server.log")
            servers.append(server)
            results["server_pids"][instance] = server.pid
            wait_healthy(server, port)
            write_json(result_path, results)
        for scenario in SCENARIOS:
            warmup = phase(
                scenario,
                0,
                sampled_stats,
                workload,
                kv_geometry,
                warmup=True,
                broker_path=broker_path,
                broker_capacity_bytes=broker_capacity_bytes,
            )
            results["warmups"].append(write_phase_artifacts(output, warmup))
            write_json(result_path, results)
        if args.skip_correctness_probe:
            results["correctness"] = {"status": "skipped", "reason": "cli"}
        else:
            results["correctness"] = run_correctness_probes(output)
        write_json(result_path, results)
        wait_for_broker_owners(
            broker_path,
            (args.broker_tenant_a, args.broker_tenant_b),
        )
        for repeat in range(1, 4):
            for scenario in scenario_order(repeat):
                measured = phase(
                    scenario,
                    repeat,
                    sampled_stats,
                    workload,
                    kv_geometry,
                    broker_path=broker_path,
                    broker_capacity_bytes=broker_capacity_bytes,
                )
                results["phases"].append(write_phase_artifacts(output, measured))
                write_json(result_path, results)
        results["status"] = "complete"
        results["configuration"]["broker"]["sqlite_final"] = sqlite_state(
            broker_path
        )
        write_json(result_path, results)
    except Exception as exc:
        results["status"] = "failed"
        results["violations"].append({
            "kind": "phase_or_runtime_failure",
            "error_type": type(exc).__name__,
            "message": safe_error_message(exc),
        })
        results["configuration"]["broker"]["sqlite_final"] = sqlite_state(
            broker_path
        )
        write_json(result_path, results)
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            for server in reversed(servers):
                run_case.stop_server(server)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
