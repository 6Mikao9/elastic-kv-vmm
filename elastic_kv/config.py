"""Validated configuration for the elastic SGLang server.

Keep configuration parsing independent from SGLang and CUDA. The launch entry
point can therefore reject invalid settings before a worker allocates VMM
address space or loads model weights.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from typing import Mapping


MODES = ("baseline", "v0", "v1", "v2", "v3", "v4", "v5")
ACTIVATION_POLICIES = ("off", "telemetry", "trim", "recycle", "both")
ADMISSION_POLICIES = ("logical", "physical")
FUTURE_POLICIES = ("off", "deadline")
DEFAULT_MODE = "baseline"
DEFAULT_CHUNK_MIB = 2
DEFAULT_RETAIN_CHUNKS = 1
DEFAULT_ACTIVATION_POLICY = "off"
DEFAULT_ADMISSION_POLICY = "logical"
DEFAULT_PREFILL_HEADROOM_MIB = 0
DEFAULT_PREFILL_WORKSPACE_MAX_MIB = 256
DEFAULT_FUTURE_POLICY = "off"
DEFAULT_PRECOMMIT_CHUNKS = 0
DEFAULT_FUTURE_ORDERING_HORIZON_MS = 0
DEFAULT_GLOBAL_BROKER_STALE_SECONDS = 120
GLOBAL_BROKER_FAIR_MODES = ("off", "observe", "enforce")
GLOBAL_BROKER_PREFILL_MODES = ("off", "observe", "enforce")


def _integer(name: str, value: object, *, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _boolean(name: str, value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean")


def _number(name: str, value: object, *, minimum: float, inclusive: bool) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(parsed) or parsed < minimum or (
        not inclusive and parsed == minimum
    ):
        relation = "at least" if inclusive else "greater than"
        raise ValueError(f"{name} must be {relation} {minimum}")
    return parsed


@dataclass(frozen=True, slots=True)
class ElasticConfig:
    """Runtime settings shared by the launcher and the SGLang adapter."""

    mode: str = DEFAULT_MODE
    chunk_mib: int = DEFAULT_CHUNK_MIB
    retain_chunks: int = DEFAULT_RETAIN_CHUNKS
    activation_policy: str = DEFAULT_ACTIVATION_POLICY
    admission_policy: str = DEFAULT_ADMISSION_POLICY
    prefill_headroom_mib: int = DEFAULT_PREFILL_HEADROOM_MIB
    precommit_chunks: int = DEFAULT_PRECOMMIT_CHUNKS
    precommit_max_chunks: int = 1
    future_policy: str = DEFAULT_FUTURE_POLICY
    future_ordering_horizon_ms: int = DEFAULT_FUTURE_ORDERING_HORIZON_MS
    stats_path: str | None = None
    future_stats_path: str | None = None
    prefill_headroom_scoped: bool = False
    allow_overlap_schedule: bool = False
    prefill_workspace_bytes_per_token: int = 0
    prefill_workspace_max_mib: int = DEFAULT_PREFILL_WORKSPACE_MAX_MIB
    global_broker_path: str | None = None
    global_broker_mib: int = 0
    global_broker_stale_seconds: int = DEFAULT_GLOBAL_BROKER_STALE_SECONDS
    global_broker_fair_mode: str = "off"
    global_broker_tenant: str | None = None
    global_broker_tenant_weight: float = 1.0
    global_broker_demand_ttl_seconds: float = 10.0
    global_broker_wait_ms: int = 0
    global_broker_prefill_mode: str = "off"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown elastic mode: {self.mode!r}")
        if self.activation_policy not in ACTIVATION_POLICIES:
            raise ValueError(
                f"unknown activation policy: {self.activation_policy!r}"
            )
        if self.admission_policy not in ADMISSION_POLICIES:
            raise ValueError(
                f"unknown admission policy: {self.admission_policy!r}"
            )
        if self.chunk_mib < 1:
            raise ValueError("chunk_mib must be positive")
        if self.retain_chunks < 1:
            raise ValueError("retain_chunks must retain at least the sink chunk")
        if self.prefill_headroom_mib < 0:
            raise ValueError("prefill_headroom_mib must be nonnegative")
        if self.allow_overlap_schedule and self.mode != "v5":
            raise ValueError("overlap scheduling requires elastic mode v5")
        if not 0 <= self.precommit_chunks <= 4:
            raise ValueError("precommit_chunks must be between 0 and 4")
        if self.precommit_max_chunks < max(1, self.precommit_chunks):
            raise ValueError(
                "precommit_max_chunks must cover precommit_chunks"
            )
        if self.future_policy not in FUTURE_POLICIES:
            raise ValueError(f"unknown future policy: {self.future_policy!r}")
        if self.future_ordering_horizon_ms < 0:
            raise ValueError("future_ordering_horizon_ms must be nonnegative")
        if self.prefill_workspace_bytes_per_token < 0:
            raise ValueError("prefill_workspace_bytes_per_token must be nonnegative")
        if self.prefill_workspace_max_mib < 1:
            raise ValueError("prefill_workspace_max_mib must be positive")
        if self.global_broker_mib < 0:
            raise ValueError("global_broker_mib must be nonnegative")
        if self.global_broker_stale_seconds < 1:
            raise ValueError("global_broker_stale_seconds must be positive")
        if self.global_broker_fair_mode not in GLOBAL_BROKER_FAIR_MODES:
            raise ValueError("unknown global broker fair mode")
        if self.global_broker_fair_mode != "off" and not self.global_broker_tenant:
            raise ValueError("fair global broker requires a tenant name")
        if (
            self.global_broker_tenant is not None
            and (
                not self.global_broker_tenant.strip()
                or len(self.global_broker_tenant) > 128
            )
        ):
            raise ValueError("global broker tenant must be a short nonempty string")
        if (
            not math.isfinite(self.global_broker_tenant_weight)
            or not 0 < self.global_broker_tenant_weight <= 1e6
        ):
            raise ValueError("global broker tenant weight must be in (0, 1e6]")
        if (
            not math.isfinite(self.global_broker_demand_ttl_seconds)
            or self.global_broker_demand_ttl_seconds <= 0
        ):
            raise ValueError("global broker demand TTL must be positive")
        if not 0 <= self.global_broker_wait_ms <= 5_000:
            raise ValueError("global broker wait must be between 0 and 5000 ms")
        if self.global_broker_prefill_mode not in GLOBAL_BROKER_PREFILL_MODES:
            raise ValueError("unknown global broker prefill mode")
        if bool(self.global_broker_path) != (self.global_broker_mib > 0):
            raise ValueError(
                "global broker requires both a path and a positive capacity"
            )
        if self.global_broker_fair_mode != "off" and not self.global_broker_path:
            raise ValueError(
                "fair global broker requires an enabled broker path and capacity"
            )
        if self.global_broker_prefill_mode != "off":
            if not self.global_broker_path:
                raise ValueError(
                    "global broker prefill claims require an enabled broker path and capacity"
                )
            if self.mode != "v5" or self.admission_policy != "physical":
                raise ValueError(
                    "global broker prefill claims require v5 physical admission"
                )
            if self.prefill_workspace_bytes_per_token <= 0:
                raise ValueError(
                    "global broker prefill claims require a positive workspace estimate"
                )
        if self.activation_policy != "off" and self.mode != "v5":
            raise ValueError("activation policies require elastic mode v5")
        if self.admission_policy != "logical" and self.mode != "v5":
            raise ValueError("physical admission requires elastic mode v5")
        if (
            self.prefill_headroom_scoped
            or self.prefill_workspace_bytes_per_token > 0
        ):
            if self.mode != "v5":
                raise ValueError("dynamic prefill workspace requires elastic mode v5")
            if self.admission_policy != "physical":
                raise ValueError(
                    "dynamic prefill workspace requires physical admission"
                )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ElasticConfig":
        env = os.environ if environ is None else environ
        precommit_chunks = _integer(
            "EKV_PRECOMMIT_CHUNKS",
            env.get("EKV_PRECOMMIT_CHUNKS", DEFAULT_PRECOMMIT_CHUNKS),
            minimum=0,
        )
        return cls(
            mode=env.get("EKV_MODE", DEFAULT_MODE),
            chunk_mib=_integer(
                "EKV_CHUNK_MIB",
                env.get("EKV_CHUNK_MIB", DEFAULT_CHUNK_MIB),
                minimum=1,
            ),
            retain_chunks=_integer(
                "EKV_RETAIN_CHUNKS",
                env.get("EKV_RETAIN_CHUNKS", DEFAULT_RETAIN_CHUNKS),
                minimum=1,
            ),
            activation_policy=env.get(
                "EKV_ACTIVATION_POLICY", DEFAULT_ACTIVATION_POLICY
            ),
            admission_policy=env.get(
                "EKV_ADMISSION_POLICY", DEFAULT_ADMISSION_POLICY
            ),
            prefill_headroom_mib=_integer(
                "EKV_PREFILL_HEADROOM_MIB",
                env.get(
                    "EKV_PREFILL_HEADROOM_MIB", DEFAULT_PREFILL_HEADROOM_MIB
                ),
                minimum=0,
            ),
            precommit_chunks=precommit_chunks,
            precommit_max_chunks=_integer(
                "EKV_PRECOMMIT_MAX_CHUNKS",
                env.get(
                    "EKV_PRECOMMIT_MAX_CHUNKS", str(max(1, precommit_chunks))
                ),
                minimum=1,
            ),
            future_policy=env.get("EKV_FUTURE_POLICY", DEFAULT_FUTURE_POLICY),
            future_ordering_horizon_ms=_integer(
                "EKV_FUTURE_ORDERING_HORIZON_MS",
                env.get(
                    "EKV_FUTURE_ORDERING_HORIZON_MS",
                    DEFAULT_FUTURE_ORDERING_HORIZON_MS,
                ),
                minimum=0,
            ),
            stats_path=env.get("EKV_STATS_PATH") or None,
            future_stats_path=env.get("EKV_FUTURE_STATS_PATH") or None,
            prefill_headroom_scoped=_boolean(
                "EKV_PREFILL_HEADROOM_SCOPED",
                env.get("EKV_PREFILL_HEADROOM_SCOPED", "0"),
            ),
            allow_overlap_schedule=_boolean(
                "EKV_ALLOW_OVERLAP_SCHEDULE",
                env.get("EKV_ALLOW_OVERLAP_SCHEDULE", "0"),
            ),
            prefill_workspace_bytes_per_token=_integer(
                "EKV_PREFILL_WORKSPACE_BYTES_PER_TOKEN",
                env.get("EKV_PREFILL_WORKSPACE_BYTES_PER_TOKEN", "0"),
                minimum=0,
            ),
            prefill_workspace_max_mib=_integer(
                "EKV_PREFILL_WORKSPACE_MAX_MIB",
                env.get(
                    "EKV_PREFILL_WORKSPACE_MAX_MIB",
                    DEFAULT_PREFILL_WORKSPACE_MAX_MIB,
                ),
                minimum=1,
            ),
            global_broker_path=env.get("EKV_GLOBAL_BROKER_PATH") or None,
            global_broker_mib=_integer(
                "EKV_GLOBAL_BROKER_MIB",
                env.get("EKV_GLOBAL_BROKER_MIB", "0"),
                minimum=0,
            ),
            global_broker_stale_seconds=_integer(
                "EKV_GLOBAL_BROKER_STALE_SECONDS",
                env.get(
                    "EKV_GLOBAL_BROKER_STALE_SECONDS",
                    DEFAULT_GLOBAL_BROKER_STALE_SECONDS,
                ),
                minimum=1,
            ),
            global_broker_fair_mode=env.get(
                "EKV_GLOBAL_BROKER_FAIR_MODE", "off"
            ).strip().lower(),
            global_broker_tenant=env.get("EKV_GLOBAL_BROKER_TENANT") or None,
            global_broker_tenant_weight=_number(
                "EKV_GLOBAL_BROKER_TENANT_WEIGHT",
                env.get("EKV_GLOBAL_BROKER_TENANT_WEIGHT", "1"),
                minimum=0,
                inclusive=False,
            ),
            global_broker_demand_ttl_seconds=_number(
                "EKV_GLOBAL_BROKER_DEMAND_TTL_SECONDS",
                env.get("EKV_GLOBAL_BROKER_DEMAND_TTL_SECONDS", "10"),
                minimum=0,
                inclusive=False,
            ),
            global_broker_wait_ms=_integer(
                "EKV_GLOBAL_BROKER_WAIT_MS",
                env.get("EKV_GLOBAL_BROKER_WAIT_MS", "0"),
                minimum=0,
            ),
            global_broker_prefill_mode=env.get(
                "EKV_GLOBAL_BROKER_PREFILL_MODE", "off"
            ).strip().lower(),
        )

    def with_overrides(
        self,
        *,
        mode: str | None = None,
        chunk_mib: int | None = None,
        retain_chunks: int | None = None,
        activation_policy: str | None = None,
        admission_policy: str | None = None,
        prefill_headroom_mib: int | None = None,
        precommit_chunks: int | None = None,
        precommit_max_chunks: int | None = None,
        future_policy: str | None = None,
        future_ordering_horizon_ms: int | None = None,
        stats_path: str | None = None,
        future_stats_path: str | None = None,
        prefill_headroom_scoped: bool | None = None,
        allow_overlap_schedule: bool | None = None,
        prefill_workspace_bytes_per_token: int | None = None,
        prefill_workspace_max_mib: int | None = None,
        global_broker_path: str | None = None,
        global_broker_mib: int | None = None,
        global_broker_stale_seconds: int | None = None,
        global_broker_fair_mode: str | None = None,
        global_broker_tenant: str | None = None,
        global_broker_tenant_weight: float | None = None,
        global_broker_demand_ttl_seconds: float | None = None,
        global_broker_wait_ms: int | None = None,
        global_broker_prefill_mode: str | None = None,
    ) -> "ElasticConfig":
        changes = {
            key: value
            for key, value in (
                ("mode", mode),
                ("chunk_mib", chunk_mib),
                ("retain_chunks", retain_chunks),
                ("activation_policy", activation_policy),
                ("admission_policy", admission_policy),
                ("prefill_headroom_mib", prefill_headroom_mib),
                ("precommit_chunks", precommit_chunks),
                ("precommit_max_chunks", precommit_max_chunks),
                ("future_policy", future_policy),
                ("future_ordering_horizon_ms", future_ordering_horizon_ms),
                ("stats_path", stats_path),
                ("future_stats_path", future_stats_path),
                ("prefill_headroom_scoped", prefill_headroom_scoped),
                ("allow_overlap_schedule", allow_overlap_schedule),
                (
                    "prefill_workspace_bytes_per_token",
                    prefill_workspace_bytes_per_token,
                ),
                ("prefill_workspace_max_mib", prefill_workspace_max_mib),
                ("global_broker_path", global_broker_path),
                ("global_broker_mib", global_broker_mib),
                ("global_broker_stale_seconds", global_broker_stale_seconds),
                ("global_broker_fair_mode", global_broker_fair_mode),
                ("global_broker_tenant", global_broker_tenant),
                ("global_broker_tenant_weight", global_broker_tenant_weight),
                (
                    "global_broker_demand_ttl_seconds",
                    global_broker_demand_ttl_seconds,
                ),
                ("global_broker_wait_ms", global_broker_wait_ms),
                ("global_broker_prefill_mode", global_broker_prefill_mode),
            )
            if value is not None
        }
        return replace(self, **changes)

    def as_env(self) -> dict[str, str]:
        result = {
            "EKV_MODE": self.mode,
            "EKV_CHUNK_MIB": str(self.chunk_mib),
            "EKV_RETAIN_CHUNKS": str(self.retain_chunks),
            "EKV_ACTIVATION_POLICY": self.activation_policy,
            "EKV_ADMISSION_POLICY": self.admission_policy,
            "EKV_PREFILL_HEADROOM_MIB": str(self.prefill_headroom_mib),
            "EKV_PRECOMMIT_CHUNKS": str(self.precommit_chunks),
            "EKV_PRECOMMIT_MAX_CHUNKS": str(self.precommit_max_chunks),
            "EKV_FUTURE_POLICY": self.future_policy,
            "EKV_FUTURE_ORDERING_HORIZON_MS": str(
                self.future_ordering_horizon_ms
            ),
            "EKV_PREFILL_HEADROOM_SCOPED": (
                "1" if self.prefill_headroom_scoped else "0"
            ),
            "EKV_ALLOW_OVERLAP_SCHEDULE": (
                "1" if self.allow_overlap_schedule else "0"
            ),
            "EKV_PREFILL_WORKSPACE_BYTES_PER_TOKEN": str(
                self.prefill_workspace_bytes_per_token
            ),
            "EKV_PREFILL_WORKSPACE_MAX_MIB": str(
                self.prefill_workspace_max_mib
            ),
            "EKV_GLOBAL_BROKER_MIB": str(self.global_broker_mib),
            "EKV_GLOBAL_BROKER_STALE_SECONDS": str(
                self.global_broker_stale_seconds
            ),
            "EKV_GLOBAL_BROKER_FAIR_MODE": self.global_broker_fair_mode,
            "EKV_GLOBAL_BROKER_TENANT_WEIGHT": str(
                self.global_broker_tenant_weight
            ),
            "EKV_GLOBAL_BROKER_DEMAND_TTL_SECONDS": str(
                self.global_broker_demand_ttl_seconds
            ),
            "EKV_GLOBAL_BROKER_WAIT_MS": str(self.global_broker_wait_ms),
            "EKV_GLOBAL_BROKER_PREFILL_MODE": self.global_broker_prefill_mode,
        }
        if self.stats_path is not None:
            result["EKV_STATS_PATH"] = self.stats_path
        if self.future_stats_path is not None:
            result["EKV_FUTURE_STATS_PATH"] = self.future_stats_path
        if self.global_broker_path is not None:
            result["EKV_GLOBAL_BROKER_PATH"] = self.global_broker_path
        if self.global_broker_tenant is not None:
            result["EKV_GLOBAL_BROKER_TENANT"] = self.global_broker_tenant
        return result

    def apply(self, environ: dict[str, str] | None = None) -> None:
        """Publish settings before the adapter module is imported."""
        target = os.environ if environ is None else environ
        target.update(self.as_env())
        if self.stats_path is None:
            target.pop("EKV_STATS_PATH", None)
        if self.future_stats_path is None:
            target.pop("EKV_FUTURE_STATS_PATH", None)
        if self.global_broker_path is None:
            target.pop("EKV_GLOBAL_BROKER_PATH", None)
        if self.global_broker_tenant is None:
            target.pop("EKV_GLOBAL_BROKER_TENANT", None)

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "chunk_mib": self.chunk_mib,
            "retain_chunks": self.retain_chunks,
            "activation_policy": self.activation_policy,
            "admission_policy": self.admission_policy,
            "prefill_headroom_mib": self.prefill_headroom_mib,
            "precommit_chunks": self.precommit_chunks,
            "precommit_max_chunks": self.precommit_max_chunks,
            "future_policy": self.future_policy,
            "future_ordering_horizon_ms": self.future_ordering_horizon_ms,
            "stats_path": self.stats_path,
            "future_stats_path": self.future_stats_path,
            "prefill_headroom_scoped": self.prefill_headroom_scoped,
            "allow_overlap_schedule": self.allow_overlap_schedule,
            "prefill_workspace_bytes_per_token": (
                self.prefill_workspace_bytes_per_token
            ),
            "prefill_workspace_max_mib": self.prefill_workspace_max_mib,
            "global_broker_path": self.global_broker_path,
            "global_broker_mib": self.global_broker_mib,
            "global_broker_stale_seconds": self.global_broker_stale_seconds,
            "global_broker_fair_mode": self.global_broker_fair_mode,
            "global_broker_tenant": self.global_broker_tenant,
            "global_broker_tenant_weight": self.global_broker_tenant_weight,
            "global_broker_demand_ttl_seconds": (
                self.global_broker_demand_ttl_seconds
            ),
            "global_broker_wait_ms": self.global_broker_wait_ms,
            "global_broker_prefill_mode": self.global_broker_prefill_mode,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True)
