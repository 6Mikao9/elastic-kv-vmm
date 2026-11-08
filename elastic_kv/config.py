"""Validated configuration for the elastic SGLang server.

Keep configuration parsing independent from SGLang and CUDA. The launch entry
point can therefore reject invalid settings before a worker allocates VMM
address space or loads model weights.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from typing import Mapping


MODES = ("baseline", "v0", "v1", "v2", "v3", "v4", "v5")
ACTIVATION_POLICIES = ("off", "telemetry", "trim", "recycle", "both")
DEFAULT_MODE = "baseline"
DEFAULT_CHUNK_MIB = 2
DEFAULT_RETAIN_CHUNKS = 1
DEFAULT_ACTIVATION_POLICY = "off"
DEFAULT_PREFILL_HEADROOM_MIB = 0


def _integer(name: str, value: object, *, minimum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


@dataclass(frozen=True, slots=True)
class ElasticConfig:
    """Runtime settings shared by the launcher and the SGLang adapter."""

    mode: str = DEFAULT_MODE
    chunk_mib: int = DEFAULT_CHUNK_MIB
    retain_chunks: int = DEFAULT_RETAIN_CHUNKS
    activation_policy: str = DEFAULT_ACTIVATION_POLICY
    prefill_headroom_mib: int = DEFAULT_PREFILL_HEADROOM_MIB
    stats_path: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown elastic mode: {self.mode!r}")
        if self.activation_policy not in ACTIVATION_POLICIES:
            raise ValueError(
                f"unknown activation policy: {self.activation_policy!r}"
            )
        if self.chunk_mib < 1:
            raise ValueError("chunk_mib must be positive")
        if self.retain_chunks < 1:
            raise ValueError("retain_chunks must retain at least the sink chunk")
        if self.prefill_headroom_mib < 0:
            raise ValueError("prefill_headroom_mib must be nonnegative")
        if self.activation_policy != "off" and self.mode != "v5":
            raise ValueError("activation policies require elastic mode v5")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ElasticConfig":
        env = os.environ if environ is None else environ
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
            prefill_headroom_mib=_integer(
                "EKV_PREFILL_HEADROOM_MIB",
                env.get(
                    "EKV_PREFILL_HEADROOM_MIB", DEFAULT_PREFILL_HEADROOM_MIB
                ),
                minimum=0,
            ),
            stats_path=env.get("EKV_STATS_PATH") or None,
        )

    def with_overrides(
        self,
        *,
        mode: str | None = None,
        chunk_mib: int | None = None,
        retain_chunks: int | None = None,
        activation_policy: str | None = None,
        prefill_headroom_mib: int | None = None,
        stats_path: str | None = None,
    ) -> "ElasticConfig":
        changes = {
            key: value
            for key, value in (
                ("mode", mode),
                ("chunk_mib", chunk_mib),
                ("retain_chunks", retain_chunks),
                ("activation_policy", activation_policy),
                ("prefill_headroom_mib", prefill_headroom_mib),
                ("stats_path", stats_path),
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
            "EKV_PREFILL_HEADROOM_MIB": str(self.prefill_headroom_mib),
        }
        if self.stats_path is not None:
            result["EKV_STATS_PATH"] = self.stats_path
        return result

    def apply(self, environ: dict[str, str] | None = None) -> None:
        """Publish settings before the adapter module is imported."""
        target = os.environ if environ is None else environ
        target.update(self.as_env())
        if self.stats_path is None:
            target.pop("EKV_STATS_PATH", None)

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "chunk_mib": self.chunk_mib,
            "retain_chunks": self.retain_chunks,
            "activation_policy": self.activation_policy,
            "prefill_headroom_mib": self.prefill_headroom_mib,
            "stats_path": self.stats_path,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True)
