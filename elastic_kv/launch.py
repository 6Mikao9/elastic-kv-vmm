"""Spawn-safe entry point for the elastic SGLang inference server."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from .config import (
    ACTIVATION_POLICIES,
    ADMISSION_POLICIES,
    FUTURE_POLICIES,
    GLOBAL_BROKER_FAIR_MODES,
    GLOBAL_BROKER_PREFILL_MODES,
    MODES,
    PREDICTIVE_PREWARM_MODES,
    REQUEST_AFFINE_MODES,
    ElasticConfig,
)

SPAWN_INSTALL_ENV = "_ELASTIC_KV_INSTALL_ON_IMPORT"


def _install_adapter() -> None:
    config = ElasticConfig.from_env()
    if config.mode != "baseline":
        from .adapter import install

        install()
    if config.future_policy == "deadline":
        from adaptive_sparse.sglang_future_hook import install as install_future_policy

        install_future_policy()


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
    parser.add_argument("--elastic-admission-policy", choices=ADMISSION_POLICIES)
    parser.add_argument("--elastic-prefill-headroom-mib", type=int)
    parser.add_argument(
        "--elastic-prefill-headroom-scoped", action="store_true", default=None
    )
    parser.add_argument(
        "--elastic-allow-overlap-schedule", action="store_true", default=None
    )
    parser.add_argument(
        "--elastic-prefill-workspace-bytes-per-token", type=int
    )
    parser.add_argument("--elastic-prefill-workspace-max-mib", type=int)
    parser.add_argument("--elastic-precommit-chunks", type=int, choices=range(5))
    parser.add_argument("--elastic-precommit-max-chunks", type=int)
    parser.add_argument(
        "--elastic-predictive-prewarm-mode", choices=PREDICTIVE_PREWARM_MODES
    )
    parser.add_argument("--elastic-predictive-prewarm-ttl-ms", type=int)
    parser.add_argument("--elastic-predictive-prewarm-max-chunks", type=int)
    parser.add_argument(
        "--elastic-request-affine-mode", choices=REQUEST_AFFINE_MODES
    )
    parser.add_argument("--elastic-future-policy", choices=FUTURE_POLICIES)
    parser.add_argument("--elastic-future-ordering-horizon-ms", type=int)
    parser.add_argument("--elastic-cohort-wait-timeout-ms", type=int)
    parser.add_argument(
        "--elastic-hard-release-gate",
        action="store_true",
        default=None,
        help="hold hinted requests out of prefill until release_after_ms",
    )
    parser.add_argument("--elastic-stats-path")
    parser.add_argument("--elastic-future-stats-path")
    parser.add_argument("--elastic-global-broker-path")
    parser.add_argument("--elastic-global-broker-mib", type=int)
    parser.add_argument("--elastic-global-broker-stale-seconds", type=int)
    parser.add_argument(
        "--elastic-global-broker-fair-mode", choices=GLOBAL_BROKER_FAIR_MODES
    )
    parser.add_argument("--elastic-global-broker-tenant")
    parser.add_argument("--elastic-global-broker-tenant-weight", type=float)
    parser.add_argument(
        "--elastic-global-broker-demand-ttl-seconds", type=float
    )
    parser.add_argument("--elastic-global-broker-wait-ms", type=int)
    parser.add_argument(
        "--elastic-global-broker-prefill-mode",
        choices=GLOBAL_BROKER_PREFILL_MODES,
    )
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
        admission_policy=args.elastic_admission_policy,
        prefill_headroom_mib=args.elastic_prefill_headroom_mib,
        precommit_chunks=args.elastic_precommit_chunks,
        precommit_max_chunks=args.elastic_precommit_max_chunks,
        predictive_prewarm_mode=args.elastic_predictive_prewarm_mode,
        predictive_prewarm_ttl_ms=args.elastic_predictive_prewarm_ttl_ms,
        predictive_prewarm_max_chunks=(
            args.elastic_predictive_prewarm_max_chunks
        ),
        request_affine_mode=args.elastic_request_affine_mode,
        future_policy=args.elastic_future_policy,
        future_ordering_horizon_ms=args.elastic_future_ordering_horizon_ms,
        cohort_wait_timeout_ms=args.elastic_cohort_wait_timeout_ms,
        hard_release_gate=args.elastic_hard_release_gate,
        stats_path=args.elastic_stats_path,
        future_stats_path=args.elastic_future_stats_path,
        prefill_headroom_scoped=args.elastic_prefill_headroom_scoped,
        allow_overlap_schedule=args.elastic_allow_overlap_schedule,
        prefill_workspace_bytes_per_token=(
            args.elastic_prefill_workspace_bytes_per_token
        ),
        prefill_workspace_max_mib=args.elastic_prefill_workspace_max_mib,
        global_broker_path=args.elastic_global_broker_path,
        global_broker_mib=args.elastic_global_broker_mib,
        global_broker_stale_seconds=args.elastic_global_broker_stale_seconds,
        global_broker_fair_mode=args.elastic_global_broker_fair_mode,
        global_broker_tenant=args.elastic_global_broker_tenant,
        global_broker_tenant_weight=args.elastic_global_broker_tenant_weight,
        global_broker_demand_ttl_seconds=(
            args.elastic_global_broker_demand_ttl_seconds
        ),
        global_broker_wait_ms=args.elastic_global_broker_wait_ms,
        global_broker_prefill_mode=args.elastic_global_broker_prefill_mode,
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
