#!/usr/bin/env python3
"""Measure load reversal across two isolated SGLang servers on one GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

try:
    from scripts import run_case
except ImportError:  # Direct execution from the scripts directory.
    import run_case

sys.path.insert(0, str(run_case.ROOT / "benchmarks"))
from client import MemorySampler, generate, write_json


PORTS = {"a": 31080, "b": 31081}
SCENARIOS = {"a-hot": ("a",), "b-hot": ("b",), "both-active": ("a", "b")}
FAIR_MODES = ("off", "observe", "enforce")
SOURCE_FILES = (
    "scripts/run_dual.py",
    "scripts/run_case.py",
    "elastic_kv/backing.py",
    "elastic_kv/global_broker.py",
    "elastic_kv/launch.py",
    "adaptive_sparse/sglang_future_hook.py",
    "native/elastic_vmm.cpp",
    "benchmarks/client.py",
)


def source_hashes():
    return {
        name: hashlib.sha256((run_case.ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_FILES
        if (run_case.ROOT / name).is_file()
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
    if bool(path) != (mib > 0):
        parser.error("global broker requires both a path and positive capacity")
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
    except (FileNotFoundError, sqlite3.Error, KeyError, TypeError, ValueError):
        # The first requests may race broker initialization. Preserve that
        # observation as uninitialized instead of fabricating zero usage.
        return {"status": "uninitialized"}
    finally:
        if connection is not None:
            connection.close()


class DualSampler(MemorySampler):
    def __init__(self, stats_paths, broker_path=None):
        super().__init__()
        self.stats_paths = stats_paths
        self.broker_path = broker_path

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
                "broker": read_broker_snapshot(self.broker_path),
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


def start_server(model, mode, port, stats, log_path, extra_environment=None):
    command = [
        sys.executable, "-m", "elastic_kv.launch",
        "--model-path", str(model.resolve()), "--host", "127.0.0.1",
        "--port", str(port), "--page-size", "1", "--attention-backend", "triton",
        "--disable-cuda-graph", "--disable-overlap-schedule", "--disable-radix-cache",
        "--max-total-tokens", "32768", "--context-length", "8192",
        "--chunked-prefill-size", "2048", "--mem-fraction-static", "0.5",
    ]
    env = {**os.environ, "EKV_MODE": mode, "EKV_STATS_PATH": str(stats),
           "PYTHONUNBUFFERED": "1", **(extra_environment or {})}
    with log_path.open("xb") as log:
        return subprocess.Popen(command, cwd=run_case.ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)


def phase(scenario, repeat, stats_paths, warmup=False, broker_path=None):
    jobs = [
        (instance, {
            "request_id": f"{'warmup' if warmup else 'measured'}/{scenario}/{repeat}/{instance}/{index}",
            "instance": instance, "input_tokens": 1024, "requested_output_tokens": 128,
        })
        for instance in SCENARIOS[scenario] for index in range(4)
    ]
    sampler = DualSampler(stats_paths, broker_path)
    with sampler, ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        started = time.perf_counter()
        futures = [
            pool.submit(generate, f"http://127.0.0.1:{PORTS[instance]}/generate", spec, 120)
            for instance, spec in jobs
        ]
        requests = [future.result() for future in futures]
        finished = time.perf_counter()
        elapsed = finished - started
        active_start = started - sampler.origin
        active_end = finished - sampler.origin
        time.sleep(0.3)
        sampler.take()

    idle = sampler.samples[-1]
    active_samples = [sample for sample in sampler.samples
                      if active_start <= sample["offset_s"] <= active_end]
    if not active_samples:
        raise RuntimeError("no memory samples in the active request window")
    if broker_path is not None and not any(
        sample["broker"] and sample["broker"].get("status") == "ready"
        for sample in active_samples
    ):
        raise RuntimeError("no initialized broker snapshot in the active request window")
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
        "active_window_s": [active_start, active_end],
        "active_sample_count": len(active_samples),
        "peak_device_used_bytes": max(sample["device_used_bytes"] for sample in active_samples),
        "idle": idle,
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
    output = args.output_dir.resolve()
    broker_path = None if args.broker_path is None else args.broker_path.resolve()
    if broker_path is not None and broker_path.exists():
        parser.error("broker state must not exist; use a fresh output directory")
    output.mkdir(parents=True, exist_ok=False)
    stats_paths = {instance: output / f"{instance}.stats.json" for instance in PORTS}
    sampled_stats = stats_paths if args.mode == "v2" else {}
    servers = []
    previous_handler = signal.signal(signal.SIGTERM, run_case.on_sigterm)
    try:
        for instance, port in PORTS.items():
            tenant = (
                args.broker_tenant_a if instance == "a" else args.broker_tenant_b
            )
            weight = (
                args.broker_weight_a if instance == "a" else args.broker_weight_b
            )
            server = start_server(models[instance], args.mode, port, stats_paths[instance],
                                  output / f"{instance}.server.log",
                                  broker_environment(
                                      broker_path,
                                      args.broker_mib,
                                      args.broker_fair_mode,
                                      tenant,
                                      weight,
                                      args.broker_stale_seconds,
                                      args.broker_demand_ttl_seconds,
                                      args.broker_wait_ms,
                                  ))
            servers.append(server)
            wait_healthy(server, port)
        for scenario in SCENARIOS:
            phase(scenario, 0, sampled_stats, warmup=True, broker_path=broker_path)
        results = {
            "mode": args.mode, "models": {"a": "0.6B", "b": "4B"},
            "server_ports": dict(PORTS),
            "server_pids": {
                instance: server.pid for instance, server in zip(PORTS, servers)
            },
            "max_total_tokens_per_instance": 32768,
            "scope": (
                "Two servers share one GPU with an optional cooperative VMM broker; "
                "the broker does not preempt another live process."
            ),
            "adapter_stats_available": bool(sampled_stats),
            "source_sha256": source_hashes(),
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
            },
            "phases": [],
        }
        for repeat in range(1, 4):
            for scenario in SCENARIOS:
                results["phases"].append(
                    phase(scenario, repeat, sampled_stats, broker_path=broker_path)
                )
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
