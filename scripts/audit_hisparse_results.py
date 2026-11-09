"""Audit complete formal HiSparse result directories without changing them."""

import argparse
import hashlib
import json
import math
from pathlib import Path

if __package__:
    from .summarize_hisparse import load, require, require_matched
else:
    from summarize_hisparse import load, require, require_matched


SEMANTIC_CASES = {"arithmetic", "capital", "translation", "sort"}
PROBE_SHAPES = ((128, 64), (8192, 128), (8192, 64))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def is_sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def audit_correctness(directory, outer, manifest):
    semantic = read_json(directory / "semantic_smoke.json")
    checks = semantic["checks"]
    require(manifest["semantic_probes"] is True and outer["semantic_smoke_passed"] is True
            and semantic["status"] == "complete" and semantic["passed"] is True
            and len(checks) == len(SEMANTIC_CASES)
            and {check["case"] for check in checks} == SEMANTIC_CASES
            and all(check["passed"] is True and check["finish_reason"] == "stop"
                    for check in checks),
            "Incomplete or failed semantic smoke")
    correctness = read_json(directory / "performance.correctness.json")
    probes = correctness["probes"]
    require(correctness["status"] == "complete" and len(probes) == len(PROBE_SHAPES),
            "Incomplete correctness probes")
    identities = []
    for index, (probe, (inputs, outputs)) in enumerate(zip(probes, PROBE_SHAPES)):
        ids = probe["output_ids"]
        require(probe["request_id"] == f"hisparse/probe/{index}"
                and probe["input_tokens"] == inputs
                and probe["requested_output_tokens"] == probe["output_tokens"] == outputs
                and is_sha256(probe["prompt_sha256"])
                and probe["output_ids_complete"] is True and isinstance(ids, list)
                and len(ids) == outputs and all(type(token) is int and token >= 0 for token in ids)
                and digest_ids(ids) == probe["output_ids_sha256"],
                f"Invalid correctness probe {index}")
        identities.append((probe["request_id"], probe["prompt_sha256"], inputs, outputs))
    return identities


def audit_vmm(directory, outer, manifest, ranks):
    enabled = manifest["vmm_backing"]
    require(type(enabled) is bool and outer["vmm_backing"] is enabled
            and outer["resident_limit"] == manifest["resident_limit"],
            "VMM or resident limit differs between manifest and result")
    if not enabled:
        return "not_enabled"
    files = set((directory / "vmm_stats").glob("tp-*.json"))
    expected = {directory / "vmm_stats" / f"tp-{rank}.json" for rank in range(ranks)}
    idle = outer["vmm_idle"]
    require(files == expected and len(idle) == ranks
            and {snapshot["tp_rank"] for snapshot in idle} == set(range(ranks)),
            "Incomplete VMM rank evidence")
    for rank, snapshot in enumerate(sorted(idle, key=lambda row: row["tp_rank"])):
        disk = read_json(directory / "vmm_stats" / f"tp-{rank}.json")
        require(snapshot["owned_pages"] == snapshot["reclaimable_bytes"] == 0,
                f"Incomplete benchmark idle cleanup for VMM rank {rank}")
        # The on-disk snapshot may have been updated during the later semantic probes.
        for row in (snapshot, disk):
            require(row["tp_rank"] == rank and row["tp_size"] == ranks
                    and row["idle"] is True
                    and all(type(row[key]) is int and row[key] >= 0 for key in (
                        "owned_pages", "reclaimable_bytes", "backed_bytes", "peak_backed_bytes",
                    ))
                    and row["reclaimable_bytes"] <= row["backed_bytes"] <= row["peak_backed_bytes"]
                    and (not manifest["resident_limit"]
                         or row["request_residency"]["resident_layer_rows"] > 0),
                    f"Invalid VMM idle or residency evidence for rank {rank}")
    return "verified"


def audit_nvml(phase, ranks, label):
    detail = phase["memory"]
    begin, end = detail["active_window_s"]
    require(all(type(value) in (int, float) and math.isfinite(value)
                for value in (begin, end, phase["phase_wall_s"]))
            and 0 <= begin < end and math.isclose(
                end - begin, phase["phase_wall_s"], rel_tol=1e-6, abs_tol=1e-6),
            f"Invalid NVML active window for {label}")
    active = []
    previous = -math.inf
    for sample in detail["samples"]:
        offset = sample["offset_s"]
        used = sample["rank_used_bytes"]
        require(type(offset) in (int, float) and math.isfinite(offset)
                and 0 <= offset and offset >= previous
                and len(used) == ranks and all(type(value) is int and value >= 0 for value in used)
                and type(sample["sum_used_bytes"]) is int and sample["sum_used_bytes"] == sum(used),
                f"Invalid NVML sample for {label}")
        previous = offset
        if begin <= offset <= end:
            active.append(sample)
    require(active and len(active) == detail["active_sample_count"]
            and detail["rank_peak_used_bytes"] == [
                max(sample["rank_used_bytes"][rank] for sample in active) for rank in range(ranks)
            ]
            and detail["peak_sum_used_bytes"] == max(sample["sum_used_bytes"] for sample in active),
            f"NVML active samples or stored peaks differ for {label}")


def phase_identity(performance):
    identity = {}
    for kind in ("warmups", "phases"):
        for phase in performance[kind]:
            label = f"{kind}/{phase['workload']}/{phase['repeat']}"
            audit_nvml(phase, performance["tp_size"], label)
            concurrency = phase["concurrency"]
            require(type(concurrency) is int and 1 <= concurrency <= len(phase["requests"]),
                    f"Invalid concurrency for {phase['workload']}/{phase['repeat']}")
            requests = []
            for request in phase["requests"]:
                ttft, tpot, elapsed = (request[key] for key in ("ttft_s", "tpot_s", "end_to_end_s"))
                require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                            for value in (ttft, tpot, elapsed))
                        and elapsed > 0 and math.isclose(
                            elapsed, ttft + (request["output_tokens"] - 1) * tpot,
                            rel_tol=1e-6, abs_tol=1e-6),
                        f"Inconsistent request latency for {request['request_id']}")
                require(is_sha256(request["prompt_sha256"])
                        and type(request["input_tokens"]) is int and request["input_tokens"] > 0,
                        f"Invalid prompt identity for {request['request_id']}")
                requests.append((request["request_id"], request["prompt_sha256"],
                                 request["input_tokens"], request["requested_output_tokens"]))
            identity[kind, phase["workload"], phase["repeat"]] = (
                concurrency, sorted(requests),
            )
    return identity


def audit(directory, expected_repeats):
    case = load(directory)
    performance = read_json(directory / "performance.json")
    outer = read_json(directory / "results.json")
    manifest = read_json(directory / "source_manifest.json")
    require(case["tp_size"] == 4 and performance["repeats"] == expected_repeats
            and case["warmup_policy"] == "once_all_shapes",
            f"Expected TP4, one warmup per workload and {expected_repeats} measured repeats")
    require(performance["sample_interval_s"] == 0.05 and performance["idle_pause_s"] == 0.3,
            "Expected a 0.05 s NVML sample interval and 0.3 s idle pause")
    probes = audit_correctness(directory, outer, manifest)
    vmm = audit_vmm(directory, outer, manifest, case["tp_size"])
    return case, phase_identity(performance), probes, {
        "directory": str(directory),
        "result_artifacts": "verified",
        "vmm_rank_evidence": vmm,
        "gpu_runtime_and_resource_evidence": "not_verified",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--expected-repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.expected_repeats < 1:
        parser.error("--expected-repeats must be positive")
    report = []
    reference = None
    for directory in args.directories:
        try:
            case, identity, probes, row = audit(directory, args.expected_repeats)
            if reference is None:
                reference = case, identity, probes
            else:
                require_matched(reference[0], case)
                require(identity == reference[1], "Measured or warmup prompt identities differ")
                require(probes == reference[2], "Correctness probe prompt identities differ")
                if reference[0]["hisparse_config"] is None and case["hisparse_config"] is not None:
                    reference = case, identity, probes
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            parser.exit(2, f"Audit rejected {directory}: {error}\n")
        report.append(row)
    print(json.dumps({
        "arms": report,
        "scope": (
            "Checks result artifact consistency. Model, image, source provenance and external "
            "GPU runtime/resource evidence are not verified. Token hashes validate recorded "
            "output IDs only and do not establish KV byte correctness. VMM cleanup is checked "
            "at the benchmark idle checkpoint. Later per-rank files can include pending "
            "reclamation after semantic probes."
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
