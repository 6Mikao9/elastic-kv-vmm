#!/usr/bin/env python3
"""Compare admission policies on a known, controlled request horizon."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_sparse.future_plan import main


if __name__ == "__main__":
    raise SystemExit(main())
