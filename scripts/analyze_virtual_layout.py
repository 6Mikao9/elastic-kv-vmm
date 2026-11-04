"""CPU allocation geometry, not measured GPU or LouisKV performance."""

import argparse
import json
from pathlib import Path
from random import Random

from experimental.virtual_layout import VirtualKVLayout


def support_sets(length=65536, selected=2048, seed=137):
    if not 0 < selected <= length:
        raise ValueError("selected must be positive and no larger than length")
    rng = Random(seed)
    # 'clustered' is a synthetic locality pattern, NOT a LouisKV capture.
    clustered = set()
    run = max(1, selected // 8)
    while len(clustered) < selected:
        begin = rng.randrange(length)
        for step in range(min(run, selected - len(clustered))):
            clustered.add((begin + step) % length)
    return {
        "contiguous": list(range(selected)),
        "uniform_random": rng.sample(range(length), selected),
        "synthetic_clustered": sorted(clustered),
    }


def analyze(length=65536, selected=2048, width=256, chunk_bytes=2 << 20):
    layout = VirtualKVLayout(1, length, width, chunk_bytes=chunk_bytes)
    cases = []
    for name, positions in support_sets(length, selected).items():
        payload = len(set(positions)) * layout.row_bytes
        chunks = layout.selected_chunks(0, positions)
        mapped = len(chunks) * chunk_bytes
        compact = ((payload + chunk_bytes - 1) // chunk_bytes) * chunk_bytes
        cases.append({"pattern": name, "unique_tokens": len(set(positions)),
                      "selected_payload_bytes": payload, "fixed_position_chunks": len(chunks),
                      "fixed_position_backing_bytes": mapped,
                      "compact_payload_lower_bound_bytes": compact,
                      "backing_over_payload": mapped / payload})
    return {"scope": "CPU geometry on synthetic support sets; no GPU or model measurement",
            "length": length, "selected": selected, "row_elements": width,
            "itemsize": layout.itemsize, "chunk_bytes": chunk_bytes,
            "sink_bytes_excluded_from_cases": chunk_bytes,
            "full_request_backing_bytes": len(layout.prefix_chunks(0, length)) * chunk_bytes,
            "limitations": ["No mapping latency or transfer timing", "No LouisKV trace or quality test",
                            "Compact lower bound excludes indices, allocator slack and packing cost",
                            "VMM commitment does not imply every byte must be transferred"],
            "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(analyze(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
