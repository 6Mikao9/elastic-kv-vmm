#!/usr/bin/env python3
"""Generate conservative known-future profiles from SGLang rollout evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_sparse.profile_calibration import calibrate_profiles


def _load(path: Path) -> object:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PlannerConfig prefill/decode profiles from a future smoke "
            "result or separate request and hook-stat files."
        )
    )
    parser.add_argument(
        "--result",
        type=Path,
        help="run_future_smoke result containing requests and future_stats",
    )
    parser.add_argument(
        "--rows",
        type=Path,
        help="JSON object/list containing rollout request rows",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        help="SGLang future-hook stats JSON",
    )
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-available", action="store_true")
    args = parser.parse_args(argv)

    if args.result is not None:
        if args.rows is not None or args.stats is not None:
            parser.error("--result cannot be combined with --rows/--stats")
        payload = _load(args.result)
        if not isinstance(payload, dict):
            parser.error("--result must contain a JSON object")
        rows = payload.get("requests")
        stats = payload.get("future_stats")
    else:
        if args.rows is None or args.stats is None:
            parser.error("provide --result or both --rows and --stats")
        rows_payload = _load(args.rows)
        rows = (
            rows_payload.get("requests")
            if isinstance(rows_payload, dict)
            else rows_payload
        )
        stats = _load(args.stats)
    if not isinstance(rows, list):
        parser.error("rollout rows must be a JSON list")
    if not isinstance(stats, dict):
        parser.error("future stats must be a JSON object")

    result = calibrate_profiles(
        rows,
        stats,
        min_samples=args.min_samples,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        target = args.output.expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if args.require_available and not result["available"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
