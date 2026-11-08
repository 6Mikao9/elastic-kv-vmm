"""Measure SQLite lease-control overhead without CUDA or model execution."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from elastic_kv.global_broker import GlobalAdmissionBroker


def _percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    at = (len(ordered) - 1) * fraction
    low = int(at)
    high = min(len(ordered) - 1, low + 1)
    weight = at - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _run_mode(mode, iterations, directory, synchronous="NORMAL"):
    capacity = 1 << 30
    chunk = 64 << 10
    path = Path(directory) / f"broker-{mode}.sqlite"
    kwargs = {
        "fair_mode": mode,
        "tenant_id": "tenant-a" if mode != "off" else None,
    }
    first = GlobalAdmissionBroker(
        path, capacity, owner_id=f"first-{mode}",
        heartbeat_seconds=None, **kwargs,
    )
    second_kwargs = dict(kwargs)
    if mode != "off":
        second_kwargs["tenant_id"] = "tenant-b"
    second = GlobalAdmissionBroker(
        path, capacity, owner_id=f"second-{mode}",
        heartbeat_seconds=None, **second_kwargs,
    )
    owners = (first, second)
    for owner in owners:
        owner._connect().execute(f"PRAGMA synchronous={synchronous.upper()}")
    samples_us = []
    claim_samples_us = []
    claim_shrink_samples_us = []
    snapshot_samples_us = []
    try:
        first.reserve({"first-anchor": chunk})
        second.reserve({"second-anchor": chunk})
        for index in range(iterations + 50):
            owner = owners[index % 2]
            key = f"{owner.owner_id}/chunk-{index}"
            started = time.perf_counter_ns()
            decision = owner.reserve({key: chunk})
            if not decision.admit or owner.release([key]) != chunk:
                raise RuntimeError("broker benchmark lease did not round trip")
            elapsed_us = (time.perf_counter_ns() - started) / 1_000.0
            if index >= 50:
                samples_us.append(elapsed_us)

            claim_key = f"{owner.owner_id}/claim-{index}"
            started = time.perf_counter_ns()
            decision = owner.claim(claim_key, chunk)
            if not decision.admit or owner.release_claim(claim_key) != chunk:
                raise RuntimeError("broker benchmark claim did not round trip")
            claim_elapsed_us = (time.perf_counter_ns() - started) / 1_000.0
            if index >= 50:
                claim_samples_us.append(claim_elapsed_us)

            claim_key = f"{owner.owner_id}/shrink-claim-{index}"
            started = time.perf_counter_ns()
            decision = owner.claim(claim_key, chunk)
            revised = owner.shrink_claim(claim_key, chunk // 2)
            released = owner.release_claim(claim_key)
            if not decision.admit or revised != chunk // 2 or released != chunk // 2:
                raise RuntimeError("broker benchmark claim shrink did not round trip")
            claim_shrink_elapsed_us = (
                time.perf_counter_ns() - started
            ) / 1_000.0
            if index >= 50:
                claim_shrink_samples_us.append(claim_shrink_elapsed_us)

            started = time.perf_counter_ns()
            owner.snapshot()
            snapshot_elapsed_us = (time.perf_counter_ns() - started) / 1_000.0
            if index >= 50:
                snapshot_samples_us.append(snapshot_elapsed_us)
    finally:
        first.close()
        second.close()
    return {
        "fair_mode": mode,
        "synchronous": synchronous.upper(),
        "iterations": iterations,
        "median_reserve_release_us": statistics.median(samples_us),
        "p95_reserve_release_us": _percentile(samples_us, 0.95),
        "max_reserve_release_us": max(samples_us),
        "median_claim_release_us": statistics.median(claim_samples_us),
        "p95_claim_release_us": _percentile(claim_samples_us, 0.95),
        "max_claim_release_us": max(claim_samples_us),
        "median_claim_shrink_release_us": statistics.median(
            claim_shrink_samples_us
        ),
        "p95_claim_shrink_release_us": _percentile(
            claim_shrink_samples_us, 0.95
        ),
        "max_claim_shrink_release_us": max(claim_shrink_samples_us),
        "median_snapshot_us": statistics.median(snapshot_samples_us),
        "p95_snapshot_us": _percentile(snapshot_samples_us, 0.95),
        "max_snapshot_us": max(snapshot_samples_us),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--synchronous",
        choices=("normal", "full"),
        default="normal",
        help="SQLite synchronous level used by the benchmark connections",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    with tempfile.TemporaryDirectory() as directory:
        payload = {
            mode: _run_mode(mode, args.iterations, directory, args.synchronous)
            for mode in ("off", "observe", "enforce")
        }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
