"""Compare complete private HiSparse captures and emit only anonymized statistics."""

import argparse
from array import array
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path


WARMUP = 16
SCHEMA = "minimax-hisparse-access-v1"
HIT_SEMANTICS = "derived_native_membership_including_short_and_latest_bypasses"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def valid_ids(ids):
    return isinstance(ids, list) and all(type(token) is int and token >= 0 for token in ids)


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def read_requests(path):
    requests = {}
    for row in read_rows(path):
        rid, prompt, output = row["request_id"], row["input_ids"], row["output_ids"]
        if not isinstance(rid, str) or not rid or rid in requests:
            raise ValueError("invalid or repeated request identity")
        if not valid_ids(prompt) or not valid_ids(output) or not prompt or not output:
            raise ValueError("invalid input or output token sequence")
        for name, ids in (("input", prompt), ("output", output)):
            if row[f"{name}_tokens"] != len(ids) or row[f"{name}_ids_sha256"] != digest_ids(ids):
                raise ValueError("token sequence disagrees with its recorded length or digest")
        reason = row["finish_reason"]
        reason_type = reason.get("type") if isinstance(reason, dict) else reason
        requests[rid] = {
            "input_sha256": row["input_ids_sha256"],
            "input_tokens": len(prompt),
            "max_output_tokens": row["max_output_tokens"],
            "output": output,
            "finish_reason": reason_type if reason_type in ("stop", "length", "abort") else "other",
        }
    if not requests:
        raise ValueError("no completed requests")
    return requests


def read_trace(directory, requests, hot, capture):
    headers, rows = {}, {}
    steps = {}
    for path in sorted(directory.glob("hisparse-rank*.jsonl")):
        events = iter(read_rows(path))
        header = next(events, {})
        rank = header.get("rank")
        layers = header.get("layers")
        if (header.get("type") != "metadata" or header.get("schema") != SCHEMA
                or header.get("diagnostic_only") is not True
                or header.get("performance_evidence") is not False
                or header.get("prestate_hit_semantics") != HIT_SEMANTICS
                or type(rank) is not int or rank < 0 or rank in headers
                or not valid_ids(layers) or not layers or len(set(layers)) != len(layers)
                or any(not header.get(key) for key in
                       ("coordinator_sha256", "kernel_sha256", "wrapper_sha256"))):
            raise ValueError("unexpected or repeated trace metadata")
        headers[rank] = header
        file_rows = 0
        closed = False
        for event in events:
            if closed:
                raise ValueError("trace event after footer")
            if event["type"] == "summary":
                if event["rows"] != file_rows or event["dropped"]:
                    raise ValueError("trace footer reports missing or dropped rows")
                closed = True
                continue
            if event["type"] != "access":
                raise ValueError("unexpected event or trace truncation")
            rid, layer, step = event["rid"], event["layer"], event["step"]
            if (rid not in requests or layer not in layers or event["rank"] != rank
                    or event["hot_capacity"] != hot or type(step) is not int or step < 0):
                raise ValueError("trace row disagrees with run configuration")
            if event["seq_len"] != requests[rid]["input_tokens"] + step + 1:
                raise ValueError("trace sequence length disagrees with decode input history")
            stream = rid, rank, layer
            if step != steps.get(stream, 0):
                raise ValueError("trace decode steps are missing, repeated or out of order")
            steps[stream] = step + 1
            selected_ids, hit_ids = event["selected_tokens"], event["prestate_hit_tokens"]
            if not valid_ids(selected_ids) or not valid_ids(hit_ids):
                raise ValueError("invalid selected or hit token positions")
            selected, hits = set(selected_ids), set(hit_ids)
            if not hits <= selected or any(token >= event["seq_len"] for token in selected):
                raise ValueError("trace contains out-of-range selection or hit positions")
            selection = array("I", sorted(selected)).tobytes()
            rows[(*stream, step)] = selection, len(selected), len(selected) - len(hits)
            file_rows += 1
    if not headers or sum(steps.values()) != capture["access_rows"]:
        raise ValueError("trace coverage disagrees with results")
    for rid, request in requests.items():
        for rank, header in headers.items():
            for layer in header["layers"]:
                if steps.get((rid, rank, layer), 0) != len(request["output"]) - 1:
                    raise ValueError("trace coverage does not match output length")
    return headers, rows


def read_run(directory):
    manifest = read_json(directory / "source_manifest.json")
    identity = {name: manifest[name] for name in
                ("dataset_sha256", "model_metadata_sha256", "runner_sha256")}
    if not all(identity.values()):
        raise ValueError("source manifest has missing content identities")
    state = read_json(directory / "results.json")
    capture = state["capture_verification"]
    config = state["hisparse"]
    hot = config["device_buffer_size"]
    if (state["status"] != "ok" or state["diagnostic_capture"] is not True
            or capture["request_rank_coverage_ok"] is not True or capture["truncation_markers"]
            or config["top_k"] != 2048 or type(hot) is not int or hot <= 0):
        raise ValueError("run is not a complete native diagnostic capture")
    requests = read_requests(directory / "requests.jsonl")
    if state["requests"] != list(requests):
        raise ValueError("recorded request order disagrees with results")
    headers, rows = read_trace(directory / "access", requests, hot, capture)
    configuration = state["capacity"], {
        key: value for key, value in config.items() if key != "device_buffer_size"
    }
    return identity, configuration, hot, headers, requests, rows


def common_prefix(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def count_stats(counts):
    return {
        "rows": counts["rows"],
        "selected_tokens": counts["selected"],
        "native_prestate_misses": counts["misses"],
        "native_prestate_miss_rate": counts["misses"] / counts["selected"] if counts["selected"] else None,
    }


def paired_counts(left_rows, right_rows, rid, prefix, ranks):
    all_rows = {rank: Counter() for rank in ranks}
    after_warmup = {rank: Counter() for rank in ranks}
    for key, (left_set, selected, left_misses) in left_rows.items():
        request, rank, _, step = key
        if request != rid or step >= prefix or key not in right_rows:
            continue
        right_set, _, right_misses = right_rows[key]
        counts = Counter(same_history_rows=1)
        if left_set == right_set:
            counts.update(same_selection_rows=1, selected_tokens=selected,
                          left_native_prestate_misses=left_misses,
                          right_native_prestate_misses=right_misses)
        all_rows[rank].update(counts)
        if step >= WARMUP:
            after_warmup[rank].update(counts)
    fields = ("same_history_rows", "same_selection_rows", "selected_tokens",
              "left_native_prestate_misses", "right_native_prestate_misses")
    return [
        {str(rank): {key: counts[key] for key in fields}
         for rank, counts in sorted(period.items())}
        for period in (all_rows, after_warmup)
    ]


def compare(directories):
    if len(directories) < 2:
        raise ValueError("at least two run directories are required")
    runs = [read_run(directory) for directory in directories]
    base_identity, base_config, _, base_headers, base_requests, _ = runs[0]
    for identity, config, _, headers, requests, _ in runs[1:]:
        if identity != base_identity or config != base_config or headers != base_headers:
            raise ValueError("source, non-hot capacity configuration or trace metadata differs")
        if requests.keys() != base_requests.keys():
            raise ValueError("request identities differ")
        for rid, row in requests.items():
            if any(row[key] != base_requests[rid][key] for key in
                   ("input_sha256", "input_tokens", "max_output_tokens")):
                raise ValueError("matched requests have different prompts or generation limits")

    result = []
    for number, rid in enumerate(base_requests, 1):
        outputs = [requests[rid]["output"] for _, _, _, _, requests, _ in runs]
        run_stats = []
        for run_id, (_, _, hot, headers, requests, rows) in enumerate(runs, 1):
            by_rank = {rank: Counter() for rank in headers}
            for (request, rank, _, step), (_, selected, misses) in rows.items():
                if request == rid and step >= WARMUP:
                    by_rank[rank].update(rows=1, selected=selected, misses=misses)
            run_stats.append({
                "run": run_id, "native_hot_capacity": hot, "output_tokens": len(outputs[run_id - 1]),
                "finish_reason": requests[rid]["finish_reason"],
                "native_after_first_16": {str(rank): count_stats(counts)
                                          for rank, counts in sorted(by_rank.items())},
            })

        pairs = []
        for left, right in combinations(range(len(runs)), 2):
            prefix = common_prefix(outputs[left], outputs[right])
            all_rows, after_warmup = paired_counts(
                runs[left][-1], runs[right][-1], rid, prefix, base_headers
            )
            pairs.append({
                "left_run": left + 1, "right_run": right + 1,
                "output_ids_exact": outputs[left] == outputs[right],
                "output_common_prefix_tokens": prefix,
                "paired_all": all_rows, "paired_after_first_16": after_warmup,
            })
        result.append({"request": f"R{number:02d}", "runs": run_stats, "pairs": pairs})
    return {
        "warmup_decode_steps": WARMUP,
        "decode_pairing": (
            "Rows align by request, rank, layer and step. Decode step s consumes output token s. "
            "A common output prefix of c tokens supports observed steps s < c."
        ),
        "interpretation": (
            "Paired miss counts cover identical current selected sets under the same input history. "
            "Earlier selections and cache state can differ between independent free-generation runs. "
            "Counts describe native prestate membership and bypasses. They do not measure transfers "
            "or isolate a causal capacity effect."
        ),
        "requests": result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="private completed run directories")
    args = parser.parse_args(argv)
    try:
        result = compare(args.runs)
    except (KeyError, TypeError, OverflowError, OSError, ValueError):
        parser.exit(2, "Comparison refused because the run inputs are invalid or incompatible.\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
