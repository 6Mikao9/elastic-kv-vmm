"""Spawn-safe entry point for the elastic SGLang inference server."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from .config import ACTIVATION_POLICIES, MODES, ElasticConfig


SPAWN_INSTALL_ENV = "_ELASTIC_KV_INSTALL_ON_IMPORT"


def _install_adapter() -> None:
    from .adapter import install

    install()


def _install_spawned_worker() -> None:
    """Install in a multiprocessing-spawned worker reconstructing this module.

    The parent sets ``SPAWN_INSTALL_ENV`` only after validating configuration.
    A normal import used by tests, tooling, and ``--elastic-dry-run`` never
    imports SGLang or touches CUDA.
    """

    if os.environ.get(SPAWN_INSTALL_ENV) != "1":
        return
    ElasticConfig.from_env().apply()
    _install_adapter()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elastic-sglang",
        description=(
            "Launch SGLang with optional demand-backed CUDA VMM KV storage. "
            "Unrecognized arguments are passed to SGLang unchanged."
        ),
    )
    parser.add_argument("--elastic-mode", choices=MODES)
    parser.add_argument("--elastic-chunk-mib", type=int)
    parser.add_argument("--elastic-retain-chunks", type=int)
    parser.add_argument("--elastic-activation-policy", choices=ACTIVATION_POLICIES)
    parser.add_argument("--elastic-prefill-headroom-mib", type=int)
    parser.add_argument("--elastic-stats-path")
    parser.add_argument(
        "--elastic-dry-run",
        action="store_true",
        help="validate and print the resolved backend settings without importing SGLang",
    )
    return parser


def resolve_config(argv: Sequence[str]) -> tuple[ElasticConfig, list[str], bool]:
    """Split elastic options from the upstream SGLang argument vector."""
    args, sglang_argv = _parser().parse_known_args(list(argv))
    config = ElasticConfig.from_env().with_overrides(
        mode=args.elastic_mode,
        chunk_mib=args.elastic_chunk_mib,
        retain_chunks=args.elastic_retain_chunks,
        activation_policy=args.elastic_activation_policy,
        prefill_headroom_mib=args.elastic_prefill_headroom_mib,
        stats_path=args.elastic_stats_path,
    )
    return config, sglang_argv, args.elastic_dry_run


def main(argv: Sequence[str] | None = None) -> int:
    config, sglang_argv, dry_run = resolve_config(sys.argv[1:] if argv is None else argv)
    config.apply()
    print(f"elastic-sglang configuration: {json.dumps(config.as_dict(), sort_keys=True)}", flush=True)
    if dry_run:
        return 0

    # Install only after configuration is authoritative. The adapter captures
    # these environment values at import time. The sentinel makes a spawned
    # worker install the same adapter when Python reconstructs this module.
    os.environ[SPAWN_INSTALL_ENV] = "1"
    _install_adapter()
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args

    launch_server(prepare_server_args(sglang_argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# A multiprocessing child re-imports this module without calling ``main``.
# It inherits the sentinel and no longer sees a half-configured server.
_install_spawned_worker()
