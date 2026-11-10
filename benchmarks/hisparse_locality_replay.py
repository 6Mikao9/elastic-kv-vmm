"""Replay ordered MiniMax decode selections in a simulated batch-token LRU.

JSONL input rows contain request_id (string), layer_id and step (nonnegative
integers), and selected (a list of unique, nonnegative logical KV positions).
Rows can interleave streams but steps must strictly increase within each
(request_id, layer_id). Each stream and each new segment after a gap start
cold. Windows are consecutive trace rows within a segment. Cache capacity
is in token entries. This model counts absent entries, including first use
of newly appended KV, and does not model native HiSparse prefill, latest-token
reservation, or transfers.

Native capture JSONL files are also accepted. Capture files, ranks, and
request slot generations are kept in separate replay streams.
"""

import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path


DEFAULT_CAPACITIES = (2048, 4096, 8192)


def _nonnegative_int(value):
    return type(value) is int and value >= 0


def replay(rows, capacities=DEFAULT_CAPACITIES, window_size=16):
    """Return simulated misses and selection turnover by stream and window."""
    capacities = tuple(capacities)
    if (not capacities or any(not _nonnegative_int(c) or not c for c in capacities)
            or len(set(capacities)) != len(capacities)):
        raise ValueError("capacities must be distinct positive integers")
    if not _nonnegative_int(window_size) or not window_size:
        raise ValueError("window_size must be a positive integer")

    streams = {}
    windows = []
    for row in rows:
        request_id, layer_id, step, selected = (
            row["request_id"], row["layer_id"], row["step"], row["selected"]
        )
        if (not isinstance(request_id, str) or not request_id
                or not _nonnegative_int(layer_id) or not _nonnegative_int(step)):
            raise ValueError("invalid request_id, layer_id or step")
        if (not isinstance(selected, list) or any(not _nonnegative_int(t) for t in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("selected must contain unique nonnegative integer KV positions")
        if len(selected) > min(capacities):
            raise ValueError("selection exceeds a replay capacity")

        identity = {name: row[name] for name in ("capture_id", "rank", "generation")
                    if name in row}
        key = (request_id, layer_id, identity.get("capture_id"),
               identity.get("rank"), identity.get("generation"))
        state = streams.get(key)
        if state is not None and step <= state["step"]:
            raise ValueError(f"out-of-order or duplicate step for stream {key!r}")
        if state is None or step != state["step"] + 1:
            state = {
                "caches": {capacity: OrderedDict() for capacity in capacities},
                "previous": None,
                "window": None,
                "segment_start_step": step,
            }
            streams[key] = state

        window = state["window"]
        if window is None or window["steps"] == window_size:
            window = {
                "request_id": request_id, "layer_id": layer_id,
                **identity,
                "segment_start_step": state["segment_start_step"],
                "first_step": step, "steps": 0, "selected": 0,
                "turnover_selected": 0, "turnover_new": 0,
                "capacities": {str(capacity): {"misses": 0} for capacity in capacities},
            }
            windows.append(window)
            state["window"] = window

        selection = set(selected)
        window["last_step"] = step
        window["steps"] += 1
        window["selected"] += len(selected)
        if state["previous"] is not None:
            window["turnover_selected"] += len(selected)
            window["turnover_new"] += len(selection - state["previous"])

        for capacity, cache in state["caches"].items():
            # Count misses at entry and touch all existing hits before eviction.
            misses = [token for token in selected if token not in cache]
            for token in selected:
                if token in cache:
                    cache.move_to_end(token)
            while len(cache) + len(misses) > capacity:
                cache.popitem(last=False)
            cache.update((token, None) for token in misses)
            window["capacities"][str(capacity)]["misses"] += len(misses)

        state["step"] = step
        state["previous"] = selection

    for window in windows:
        denominator = window["turnover_selected"]
        window["turnover_rate"] = window["turnover_new"] / denominator if denominator else None
        for counts in window["capacities"].values():
            denominator = window["selected"]
            counts["miss_rate"] = counts["misses"] / denominator if denominator else None

    return {"policy": "simulated_batch_token_lru", "initial_state": "cold_each_segment",
            "capacity_unit": "token_entries", "windows": windows}


def trace_rows(paths, capture_stats=None):
    for path in paths:
        with path.open(encoding="utf-8") as trace:
            metadata = None
            for line in trace:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("type") == "metadata":
                    if row.get("schema") != "minimax-hisparse-access-v1" or metadata is not None:
                        raise ValueError(f"invalid capture metadata in {path}")
                    metadata = row
                elif metadata is None:
                    yield row
                elif row.get("type") == "access":
                    if row["rank"] != metadata["rank"]:
                        raise ValueError(f"capture rank differs from metadata in {path}")
                    selected = list(dict.fromkeys(row["selected_tokens"]))
                    if capture_stats is not None:
                        capture_stats["duplicate_token_occurrences"] += (
                            len(row["selected_tokens"]) - len(selected)
                        )
                    yield {
                        "request_id": row["rid"], "layer_id": row["layer"], "step": row["step"],
                        "selected": selected, "capture_id": str(path.resolve()),
                        "rank": row["rank"], "generation": row["generation"],
                    }
                elif row.get("type") == "truncation" and capture_stats is not None:
                    capture_stats["truncation_markers"][row["reason"]] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, nargs="+", help="ordered JSONL selection trace files")
    parser.add_argument("--capacities", nargs="+", type=int, default=DEFAULT_CAPACITIES)
    parser.add_argument("--window-size", type=int, default=16)
    args = parser.parse_args()
    capture_stats = {"duplicate_token_occurrences": 0, "truncation_markers": Counter()}
    result = replay(trace_rows(args.trace, capture_stats), args.capacities, args.window_size)
    result["capture_input"] = capture_stats
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
