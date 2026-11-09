"""Build the CUDA driver library from the source carried by the wheel."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path

DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / "native" / "elastic_vmm.cpp"


def default_cuda_home() -> Path:
    configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    return Path(configured).expanduser() if configured else Path("/usr/local/cuda")


def build_command(
    source: Path,
    output: Path,
    *,
    cxx: str,
    cuda_home: Path,
) -> list[str]:
    return [
        cxx,
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-shared",
        "-fPIC",
        "-I",
        str(cuda_home / "include"),
        str(source),
        "-lcuda",
        "-o",
        str(output),
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile libelastic_vmm.so from the CUDA VMM source included in "
            "the elastic-kv-vmm package"
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="native/elastic_vmm.cpp source path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output library; defaults beside the packaged source",
    )
    parser.add_argument(
        "--cxx",
        default=os.environ.get("CXX", "c++"),
        help="C++17 compiler (default: CXX or c++)",
    )
    parser.add_argument(
        "--cuda-home",
        type=Path,
        default=default_cuda_home(),
        help="CUDA toolkit root containing include/cuda.h",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    source = args.source.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else source.with_name("libelastic_vmm.so")
    )
    cuda_home = args.cuda_home.expanduser().resolve()
    if not source.is_file():
        parser.error(f"VMM source is missing: {source}")
    header = cuda_home / "include" / "cuda.h"
    if not header.is_file():
        parser.error(
            f"CUDA header is missing: {header}; pass --cuda-home or set CUDA_HOME"
        )
    command = build_command(
        source,
        output,
        cxx=args.cxx,
        cuda_home=cuda_home,
    )
    if args.dry_run:
        print(shlex.join(command))
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        parser.error(f"C++ compiler was not found: {args.cxx}")
    if not output.is_file():
        parser.error(f"compiler completed without producing {output}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
