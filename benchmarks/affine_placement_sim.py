"""Compare fixed-stride and packed request-affine KV placement geometry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from random import Random

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experimental.affine_placement import (
    AffineConfig,
    AffineRequest,
    SimulationResult,
    simulate_affine_placement,
)

SCENARIOS = ("mixed", "short-heavy", "equal-length")


def generate_workload(
    scenario: str, request_count: int, seed: int
) -> list[AffineRequest]:
    if request_count < 1:
        raise ValueError("request_count must be positive")
    rng = Random(seed)
    requests = []
    arrival = 0.0
    for index in range(request_count):
        if scenario == "mixed":
            draw = rng.random()
            if draw < 0.35:
                total = rng.randint(128, 1024)
            elif draw < 0.75:
                total = rng.randint(1024, 4096)
            else:
                total = rng.randint(4096, 16384)
        elif scenario == "short-heavy":
            total = (
                rng.randint(8192, 16384)
                if rng.random() < 0.08
                else rng.randint(64, 512)
            )
        elif scenario == "equal-length":
            total = 4096
        else:
            raise ValueError(f"unknown scenario: {scenario!r}")
        prompt = max(1, min(total, int(total * rng.uniform(0.55, 0.9))))
        requests.append(
            AffineRequest(
                request_id=f"{scenario}-{index:03d}",
                arrival_ms=arrival,
                prompt_tokens=prompt,
                output_tokens=total - prompt,
            )
        )
        arrival += rng.expovariate(1 / 2.5)
    return requests


def _compact(result: SimulationResult) -> dict[str, object]:
    payload = result.as_dict()
    payload.pop("admissions")
    return payload


def run_scenario(
    scenario: str,
    request_count: int,
    seed: int,
    config: AffineConfig,
) -> dict[str, object]:
    workload = generate_workload(scenario, request_count, seed)
    fixed = simulate_affine_placement(workload, config, "fixed-affine")
    private = simulate_affine_placement(workload, config, "private-affine")
    hybrid = simulate_affine_placement(workload, config, "hybrid-affine")
    packed = simulate_affine_placement(workload, config, "packed-affine")
    fixed_peak = fixed.peak.mapped_bytes
    private_peak = private.peak.mapped_bytes
    hybrid_peak = hybrid.peak.mapped_bytes
    packed_peak = packed.peak.mapped_bytes
    return {
        "scenario": scenario,
        "seed": seed,
        "request_count": request_count,
        "total_tokens": sum(item.total_tokens for item in workload),
        "min_total_tokens": min(item.total_tokens for item in workload),
        "max_total_tokens": max(item.total_tokens for item in workload),
        "fixed_affine": _compact(fixed),
        "private_affine": _compact(private),
        "hybrid_affine": _compact(hybrid),
        "packed_affine": _compact(packed),
        "comparison": {
            "fixed_peak_backing_bytes": fixed_peak,
            "private_peak_backing_bytes": private_peak,
            "hybrid_peak_backing_bytes": hybrid_peak,
            "packed_peak_backing_bytes": packed_peak,
            "private_peak_backing_delta_vs_fixed_bytes": private_peak - fixed_peak,
            "peak_backing_delta_bytes": packed_peak - fixed_peak,
            "peak_backing_reduction_fraction": (
                (fixed_peak - packed_peak) / fixed_peak if fixed_peak else 0.0
            ),
            "packed_shared_savings_at_peak_bytes": packed.peak.shared_savings_bytes,
            "packed_shared_chunks_at_peak": packed.peak.shared_chunks,
            "private_shared_chunks_at_peak": private.peak.shared_chunks,
            "hybrid_shared_chunks_at_peak": hybrid.peak.shared_chunks,
            "fixed_virtual_reservation_at_peak_bytes": fixed.peak.reserved_bytes,
            "private_virtual_reservation_at_peak_bytes": private.peak.reserved_bytes,
            "hybrid_virtual_reservation_at_peak_bytes": hybrid.peak.reserved_bytes,
            "packed_virtual_reservation_at_peak_bytes": packed.peak.reserved_bytes,
            "packed_extra_active_requests_at_peak": (
                packed.peak.active_requests - fixed.peak.active_requests
            ),
            "max_start_delay_reduction_ms": (
                fixed.max_start_delay_ms - packed.max_start_delay_ms
            ),
            "makespan_reduction_ms": fixed.makespan_ms - packed.makespan_ms,
            "private_to_packed_start_delay_delta_ms": (
                private.max_start_delay_ms - packed.max_start_delay_ms
            ),
            "private_to_packed_makespan_delta_ms": (
                private.makespan_ms - packed.makespan_ms
            ),
            "hybrid_to_packed_start_delay_delta_ms": (
                hybrid.max_start_delay_ms - packed.max_start_delay_ms
            ),
            "hybrid_to_packed_makespan_delta_ms": (
                hybrid.makespan_ms - packed.makespan_ms
            ),
        },
    }


def analyze(args: argparse.Namespace) -> dict[str, object]:
    config = AffineConfig(
        arena_bytes=args.arena_mib * (1 << 20),
        chunk_bytes=args.chunk_mib * (1 << 20),
        row_bytes=args.row_bytes,
        max_active_requests=args.max_active,
        prefill_tokens_per_ms=args.prefill_tokens_per_ms,
        decode_tokens_per_ms=args.decode_tokens_per_ms,
    )
    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    return {
        "scope": (
            "CPU mapping-geometry simulation; not a GPU, model-attention, "
            "mapping-latency, or serving-throughput measurement"
        ),
        "model": {
            "arena_bytes_per_rank": config.arena_bytes,
            "chunk_bytes_per_rank": config.chunk_bytes,
            "row_bytes_per_token": config.row_bytes,
            "max_active_requests": config.max_active_requests,
            "prefill_tokens_per_ms": config.prefill_tokens_per_ms,
            "decode_tokens_per_ms": config.decode_tokens_per_ms,
        },
        "placement": {
            "fixed-affine": (
                "every request begins on a chunk-aligned slot sized for the "
                "workload's largest final sequence"
            ),
            "private-affine": (
                "every request has a chunk-aligned base and reserves its own "
                "exact final sequence rounded up to whole VMM chunks"
            ),
            "hybrid-affine": (
                "requests whose exact extent reaches one VMM chunk use private "
                "chunk-rounded slabs; smaller requests remain best-fit packed"
            ),
            "packed-affine": (
                "exact final sequence extent, best-fit in one virtual arena; "
                "adjacent requests may share VMM chunks"
            ),
        },
        "limitations": [
            "Requests must declare prompt and output lengths at admission.",
            "Request bases remain fixed, so packed allocations reserve their full final extents.",
            "No prefix sharing, offload, migration, or GPU execution is modeled.",
            "The objective is backing geometry and admission delay, not attention speed.",
        ],
        "results": [
            run_scenario(scenario, args.requests, args.seed, config)
            for scenario in scenarios
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--requests", type=int, default=96)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--max-active", type=int, default=16)
    parser.add_argument("--arena-mib", type=int, default=64)
    parser.add_argument("--chunk-mib", type=int, default=2)
    parser.add_argument("--row-bytes", type=int, default=512)
    parser.add_argument("--prefill-tokens-per-ms", type=float, default=8.0)
    parser.add_argument("--decode-tokens-per-ms", type=float, default=20.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.requests < 1 or args.max_active < 1:
        parser.error("request and active counts must be positive")
    if args.arena_mib < 1 or args.chunk_mib < 1 or args.row_bytes < 1:
        parser.error("arena, chunk, and row sizes must be positive")
    if args.arena_mib % args.chunk_mib:
        parser.error("--arena-mib must be a multiple of --chunk-mib")

    payload = analyze(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
