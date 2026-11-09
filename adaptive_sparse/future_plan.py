"""Offline, future-aware admission and residency planning.

This module is deliberately independent of CUDA and SGLang.  It models the
budget decision that sits above the VMM/HiSparse runtime so policies can be
compared on deterministic workloads before they affect a live server.

The planner assumes that an admitted request keeps one residency mode until it
finishes.  That is intentional: a planner that appears better by switching a
request every decoding step is not deployable under migration and jitter
costs.  Live integration must still be performed at a quiescent scheduler
boundary and must validate the resulting reservation against physical backing.
"""

from __future__ import annotations

import argparse
import json
import math
import secrets
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from itertools import product
from pathlib import Path

from .future_residency import RESIDENCE_HINTS

POLICIES = (
    "fifo",
    "longest",
    "mode-aware",
    "future",
    "batched",
    "online",
    "oracle",
)
RUNTIME_HINT_MAX_MS = 86_400_000


class ResidencyMode(str, Enum):
    """A request's memory policy for its whole lifetime."""

    RESIDENT = "resident"
    HI_SPARSE = "hi-sparse"


@dataclass(frozen=True, slots=True)
class RequestProfile:
    """Known request metadata available in a controlled rollout.

    ``priority`` is the legacy JSON field name for a positive soft business
    weight used by the offline objective. ``plan_rollout`` emits it only as
    SGLang's ``FutureHint.weight`` and never assigns ``request.priority``;
    native SGLang hard-priority tiers therefore remain owned by SGLang.
    ``prefix_group`` and ``shared_prefix_tokens`` describe reusable content.
    Resident requests in the same group share one physical-prefix charge in the
    planner. Runtime placement still uses the live VMM owner index; the planner
    only supplies a conservative capacity model.
    """

    request_id: str
    arrival_ms: int
    prompt_tokens: int
    output_tokens: int
    kind: str = "generic"
    priority: float = 1.0
    deadline_ms: int | None = None
    prefix_group: str | None = None
    shared_prefix_tokens: int = 0
    residence_hint: str = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        prefix = f"{self.request_id}: " if self.request_id else ""
        _bounded_int(
            self.arrival_ms,
            "arrival_ms",
            maximum=RUNTIME_HINT_MAX_MS,
            prefix=prefix,
        )
        _bounded_int(self.prompt_tokens, "prompt_tokens", minimum=1, prefix=prefix)
        _bounded_int(
            self.output_tokens,
            "output_tokens",
            minimum=1,
            maximum=RUNTIME_HINT_MAX_MS,
            prefix=prefix,
        )
        if (
            isinstance(self.priority, bool)
            or not isinstance(self.priority, (int, float))
            or not math.isfinite(self.priority)
            or not 0 < self.priority <= 1e6
        ):
            raise ValueError(
                f"{prefix}priority must be finite number in (0, 1e6]"
            )
        if not isinstance(self.kind, str) or not self.kind or len(self.kind) > 64:
            raise ValueError(
                f"{prefix}kind must be a short non-empty string"
            )
        if self.deadline_ms is not None:
            _bounded_int(
                self.deadline_ms,
                "deadline_ms",
                maximum=RUNTIME_HINT_MAX_MS,
                prefix=prefix,
            )
            if self.deadline_ms < self.arrival_ms:
                raise ValueError(f"{prefix}deadline precedes arrival")
        _bounded_int(self.shared_prefix_tokens, "shared_prefix_tokens", prefix=prefix)
        if self.shared_prefix_tokens > self.prompt_tokens:
            raise ValueError(f"{prefix}invalid shared_prefix_tokens")
        if self.prefix_group is not None and (
            not isinstance(self.prefix_group, str)
            or not self.prefix_group
            or len(self.prefix_group) > 64
        ):
            raise ValueError(f"{prefix}prefix_group must be a short non-empty string")
        if self.shared_prefix_tokens and not self.prefix_group:
            raise ValueError(f"{prefix}prefix_group is required for reuse")
        if self.residence_hint not in RESIDENCE_HINTS:
            raise ValueError(f"{prefix}invalid residence_hint")

    @property
    def sequence_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class KindServiceProfile:
    """Optional resource and speed overrides for one known request kind.

    Missing values inherit :class:`PlannerConfig`.  A controlled rollout may
    know that a tool call, chat turn and long generation have different prefill,
    decode, hot-set and workspace costs even when they use the same model.
    Keeping the overrides keyed by the existing ``kind`` hint avoids growing the
    request transport with deployment-specific schema fields.
    """

    prefill_tokens_per_ms: float | None = None
    resident_decode_tokens_per_ms: float | None = None
    sparse_decode_tokens_per_ms: float | None = None
    sparse_hot_tokens: int | None = None
    prefill_workspace_bytes_per_token: int | None = None
    auto_resident_output_threshold: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "prefill_tokens_per_ms",
            "resident_decode_tokens_per_ms",
            "sparse_decode_tokens_per_ms",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number or None")
        for name, minimum in (
            ("sparse_hot_tokens", 1),
            ("prefill_workspace_bytes_per_token", 0),
            ("auto_resident_output_threshold", 1),
        ):
            value = getattr(self, name)
            if value is None:
                continue
            _bounded_int(value, name, minimum=minimum)

    def as_dict(self) -> dict[str, object]:
        return {
            key: value
            for key, value in (
                ("prefill_tokens_per_ms", self.prefill_tokens_per_ms),
                (
                    "resident_decode_tokens_per_ms",
                    self.resident_decode_tokens_per_ms,
                ),
                (
                    "sparse_decode_tokens_per_ms",
                    self.sparse_decode_tokens_per_ms,
                ),
                ("sparse_hot_tokens", self.sparse_hot_tokens),
                (
                    "prefill_workspace_bytes_per_token",
                    self.prefill_workspace_bytes_per_token,
                ),
                (
                    "auto_resident_output_threshold",
                    self.auto_resident_output_threshold,
                ),
            )
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class KindDecodeBatchProfile:
    """Measured aggregate decode rate for one kind at one batch size."""

    resident_tokens_per_ms: float | None = None
    sparse_tokens_per_ms: float | None = None

    def __post_init__(self) -> None:
        for name in ("resident_tokens_per_ms", "sparse_tokens_per_ms"):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number or None")
        if self.resident_tokens_per_ms is None and self.sparse_tokens_per_ms is None:
            raise ValueError(
                "kind decode batch profile requires resident or sparse throughput"
            )

    def rate(self, mode: ResidencyMode) -> float | None:
        if mode is ResidencyMode.RESIDENT:
            return self.resident_tokens_per_ms
        return self.sparse_tokens_per_ms

    def as_dict(self) -> dict[str, object]:
        return {
            key: value
            for key, value in (
                ("resident_tokens_per_ms", self.resident_tokens_per_ms),
                ("sparse_tokens_per_ms", self.sparse_tokens_per_ms),
            )
            if value is not None
        }


def normalize_kind_decode_batch_profiles(
    value: object,
) -> tuple[tuple[str, KindDecodeBatchProfile], ...]:
    """Validate per-kind overrides nested under one batch profile."""

    raw_profiles: Iterable[tuple[object, object]]
    if isinstance(value, Mapping):
        raw_profiles = value.items()
    elif isinstance(value, (tuple, list)):
        raw_profiles = value
    else:
        raise ValueError(
            "kind_profiles in a decode batch profile must be a mapping or pair sequence"
        )
    profiles: dict[str, KindDecodeBatchProfile] = {}
    for item in raw_profiles:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(
                "kind decode batch profiles must be kind/profile pairs"
            )
        kind, raw_profile = item
        if not isinstance(kind, str) or not kind or len(kind) > 64:
            raise ValueError(
                "kind decode batch profile keys must be short non-empty strings"
            )
        if kind in profiles:
            raise ValueError(f"duplicate kind decode batch profile: {kind!r}")
        if isinstance(raw_profile, KindDecodeBatchProfile):
            profile = raw_profile
        elif isinstance(raw_profile, Mapping):
            profile = KindDecodeBatchProfile(**dict(raw_profile))
        else:
            raise ValueError(
                f"decode batch profile for kind {kind!r} must be an object"
            )
        profiles[kind] = profile
    return tuple(sorted(profiles.items()))


@dataclass(frozen=True, slots=True)
class DecodeBatchProfile:
    """Measured aggregate decode throughput at one co-active batch size.

    The values describe all streams in the batch, not one request. The planner
    divides by the measured batch size before estimating an individual decode
    duration. A missing mode inherits the ordinary per-kind rate, so operators
    can publish only the measurements they actually have.
    """

    resident_tokens_per_ms: float | None = None
    sparse_tokens_per_ms: float | None = None
    kind_profiles: (
        tuple[tuple[str, KindDecodeBatchProfile], ...]
        | Mapping[str, KindDecodeBatchProfile | Mapping[str, object]]
    ) = ()

    def __post_init__(self) -> None:
        for name in ("resident_tokens_per_ms", "sparse_tokens_per_ms"):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number or None")
        if self.resident_tokens_per_ms is None and self.sparse_tokens_per_ms is None:
            raise ValueError(
                "decode batch profile requires resident or sparse throughput"
            )
        object.__setattr__(
            self,
            "kind_profiles",
            normalize_kind_decode_batch_profiles(self.kind_profiles),
        )

    def rate(self, mode: ResidencyMode, kind: str | None = None) -> float | None:
        if kind is not None:
            for candidate, profile in self.kind_profiles:
                if candidate == kind:
                    measured = profile.rate(mode)
                    if measured is not None:
                        return measured
                    break
        if mode is ResidencyMode.RESIDENT:
            return self.resident_tokens_per_ms
        return self.sparse_tokens_per_ms

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            key: value
            for key, value in (
                ("resident_tokens_per_ms", self.resident_tokens_per_ms),
                ("sparse_tokens_per_ms", self.sparse_tokens_per_ms),
            )
            if value is not None
        }
        if self.kind_profiles:
            result["kind_profiles"] = {
                kind: profile.as_dict()
                for kind, profile in self.kind_profiles
            }
        return result


def normalize_kind_profiles(
    value: object,
) -> tuple[tuple[str, KindServiceProfile], ...]:
    """Validate and canonicalize JSON or Python request-kind profiles."""

    raw_profiles: Iterable[tuple[str, object]]
    if isinstance(value, Mapping):
        raw_profiles = value.items()
    elif isinstance(value, (tuple, list)):
        raw_profiles = value
    else:
        raise ValueError("kind_profiles must be a mapping or pair sequence")
    profiles: dict[str, KindServiceProfile] = {}
    for item in raw_profiles:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("kind_profiles entries must be key/profile pairs")
        kind, raw_profile = item
        if not isinstance(kind, str) or not kind or len(kind) > 64:
            raise ValueError("kind_profiles keys must be short non-empty strings")
        if kind in profiles:
            raise ValueError(f"duplicate kind profile: {kind!r}")
        if isinstance(raw_profile, KindServiceProfile):
            profile = raw_profile
        elif isinstance(raw_profile, Mapping):
            profile = KindServiceProfile(**dict(raw_profile))
        else:
            raise ValueError(f"kind profile for {kind!r} must be an object")
        profiles[kind] = profile
    return tuple(sorted(profiles.items()))


def normalize_decode_batch_profiles(
    value: object,
) -> tuple[tuple[int, DecodeBatchProfile], ...]:
    """Validate and canonicalize measured batch-size throughput profiles."""

    raw_profiles: Iterable[tuple[object, object]]
    if isinstance(value, Mapping):
        raw_profiles = value.items()
    elif isinstance(value, (tuple, list)):
        raw_profiles = value
    else:
        raise ValueError(
            "decode_batch_profiles must be a mapping or pair sequence"
        )
    profiles: dict[int, DecodeBatchProfile] = {}
    for item in raw_profiles:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(
                "decode_batch_profiles entries must be batch/profile pairs"
            )
        raw_batch_size, raw_profile = item
        if isinstance(raw_batch_size, str) and raw_batch_size.isdecimal():
            batch_size = int(raw_batch_size)
        elif isinstance(raw_batch_size, bool):
            raise ValueError("decode batch size must be an integer")
        else:
            batch_size = raw_batch_size
        _bounded_int(batch_size, "decode batch size", minimum=1)
        if batch_size in profiles:
            raise ValueError(f"duplicate decode batch profile: {batch_size}")
        if isinstance(raw_profile, DecodeBatchProfile):
            profile = raw_profile
        elif isinstance(raw_profile, Mapping):
            profile = DecodeBatchProfile(**dict(raw_profile))
        else:
            raise ValueError(
                f"decode batch profile for {batch_size} must be an object"
            )
        profiles[batch_size] = profile
    return tuple(sorted(profiles.items()))


def normalize_prefill_batch_profiles(
    value: object,
) -> tuple[tuple[int, float], ...]:
    """Validate measured aggregate prefill throughput by batch size."""

    raw_profiles: Iterable[tuple[object, object]]
    if isinstance(value, Mapping):
        raw_profiles = value.items()
    elif isinstance(value, (tuple, list)):
        raw_profiles = value
    else:
        raise ValueError(
            "prefill_batch_profiles must be a mapping or pair sequence"
        )
    profiles: dict[int, float] = {}
    for item in raw_profiles:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(
                "prefill_batch_profiles entries must be batch/rate pairs"
            )
        raw_batch_size, raw_rate = item
        if isinstance(raw_batch_size, str) and raw_batch_size.isdecimal():
            batch_size = int(raw_batch_size)
        elif isinstance(raw_batch_size, bool):
            raise ValueError("prefill batch size must be an integer")
        else:
            batch_size = raw_batch_size
        _bounded_int(batch_size, "prefill batch size", minimum=1)
        if batch_size in profiles:
            raise ValueError(f"duplicate prefill batch profile: {batch_size}")
        if (
            isinstance(raw_rate, bool)
            or not isinstance(raw_rate, (int, float))
            or not math.isfinite(raw_rate)
            or raw_rate <= 0
        ):
            raise ValueError(
                "prefill batch throughput must be a finite positive number"
            )
        profiles[batch_size] = raw_rate
    return tuple(sorted(profiles.items()))


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    """Conservative per-instance resource model.

    Rates are measured in tokens per millisecond.  A HiSparse request reserves
    at most ``sparse_hot_tokens`` logical positions rather than its full K/V
    sequence.  ``lookahead_ms`` is the knowledge horizon available to the
    oracle policy; it is not a claim that online serving has this information.
    """

    memory_budget_bytes: int
    kv_bytes_per_token: int
    prefill_workspace_bytes_per_token: int
    sparse_hot_tokens: int = 2048
    prefill_tokens_per_ms: float = 8.0
    resident_decode_tokens_per_ms: float = 20.0
    sparse_decode_tokens_per_ms: float = 16.0
    max_active_requests: int = 8
    lookahead_ms: int = 0
    batch_window_ms: int = 0
    max_batch_requests: int = 1
    auto_resident_output_threshold: int = 1
    active_prefix_prefill_reuse: bool = False
    decode_batch_profiles: (
        tuple[tuple[int, DecodeBatchProfile], ...]
        | Mapping[int | str, DecodeBatchProfile | Mapping[str, object]]
    ) = ()
    prefill_batch_profiles: (
        tuple[tuple[int, float], ...]
        | Mapping[int | str, float]
    ) = ()
    oracle_max_expansions: int = 100_000
    online_max_rollouts: int = 2_000
    kind_profiles: (
        tuple[tuple[str, KindServiceProfile], ...]
        | Mapping[str, KindServiceProfile | Mapping[str, object]]
    ) = ()

    def __post_init__(self) -> None:
        positive_ints = {
            "memory_budget_bytes": self.memory_budget_bytes,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "sparse_hot_tokens": self.sparse_hot_tokens,
            "max_active_requests": self.max_active_requests,
            "max_batch_requests": self.max_batch_requests,
            "auto_resident_output_threshold": self.auto_resident_output_threshold,
            "oracle_max_expansions": self.oracle_max_expansions,
            "online_max_rollouts": self.online_max_rollouts,
        }
        for name, value in positive_ints.items():
            _bounded_int(value, name, minimum=1)
        _bounded_int(
            self.prefill_workspace_bytes_per_token,
            "prefill_workspace_bytes_per_token",
        )
        for name, value in (
            ("prefill_tokens_per_ms", self.prefill_tokens_per_ms),
            ("resident_decode_tokens_per_ms", self.resident_decode_tokens_per_ms),
            ("sparse_decode_tokens_per_ms", self.sparse_decode_tokens_per_ms),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")
        _bounded_int(
            self.lookahead_ms,
            "lookahead_ms",
            maximum=RUNTIME_HINT_MAX_MS,
        )
        _bounded_int(
            self.batch_window_ms,
            "batch_window_ms",
            maximum=RUNTIME_HINT_MAX_MS,
        )
        if self.max_batch_requests > self.max_active_requests:
            raise ValueError(
                "max_batch_requests must be <= max_active_requests"
            )
        if not isinstance(self.active_prefix_prefill_reuse, bool):
            raise ValueError("active_prefix_prefill_reuse must be a boolean")
        object.__setattr__(
            self, "kind_profiles", normalize_kind_profiles(self.kind_profiles)
        )
        object.__setattr__(
            self,
            "decode_batch_profiles",
            normalize_decode_batch_profiles(self.decode_batch_profiles),
        )
        object.__setattr__(
            self,
            "prefill_batch_profiles",
            normalize_prefill_batch_profiles(self.prefill_batch_profiles),
        )

    def profile_for(self, kind: str) -> KindServiceProfile:
        """Return explicit kind overrides, or an all-inherited profile."""

        for candidate, profile in self.kind_profiles:
            if candidate == kind:
                return profile
        return KindServiceProfile()

    def prefill_rate(self, kind: str) -> float:
        override = self.profile_for(kind).prefill_tokens_per_ms
        return self.prefill_tokens_per_ms if override is None else override

    def decode_rate(self, kind: str, mode: ResidencyMode) -> float:
        profile = self.profile_for(kind)
        if mode is ResidencyMode.RESIDENT:
            override = profile.resident_decode_tokens_per_ms
            return self.resident_decode_tokens_per_ms if override is None else override
        override = profile.sparse_decode_tokens_per_ms
        return self.sparse_decode_tokens_per_ms if override is None else override

    def sparse_hot_tokens_for(self, kind: str) -> int:
        override = self.profile_for(kind).sparse_hot_tokens
        return self.sparse_hot_tokens if override is None else override

    def decode_batch_profile(self, batch_size: int) -> DecodeBatchProfile | None:
        """Return an exact measured profile, or ``None`` for linear fallback."""

        _bounded_int(batch_size, "batch_size", minimum=1)
        for candidate, profile in self.decode_batch_profiles:
            if candidate == batch_size:
                return profile
        return None

    def prefill_rate_for_batch(self, kind: str, batch_size: int) -> float:
        """Return estimated per-stream prefill throughput at ``batch_size``.

        A measured prefill profile contains aggregate tokens per millisecond,
        so it is divided by the measured batch size. A missing measurement
        falls back to the existing per-kind rate and never interpolates.
        """

        _bounded_int(batch_size, "batch_size", minimum=1)
        for candidate, aggregate_rate in self.prefill_batch_profiles:
            if candidate == batch_size:
                return aggregate_rate / batch_size
        return self.prefill_rate(kind)

    def decode_rate_for_batch(
        self,
        kind: str,
        mode: ResidencyMode,
        batch_size: int,
    ) -> float:
        """Return estimated per-stream decode throughput.

        Measured profiles contain aggregate batch throughput. Requests without
        an exact profile or mode measurement keep the ordinary per-kind rate;
        the planner never invents interpolation between measurements.
        """

        profile = self.decode_batch_profile(batch_size)
        if profile is None:
            return self.decode_rate(kind, mode)
        aggregate_rate = profile.rate(mode, kind)
        if aggregate_rate is None:
            return self.decode_rate(kind, mode)
        return aggregate_rate / batch_size

    def prefill_workspace_rate(self, kind: str) -> int:
        override = self.profile_for(kind).prefill_workspace_bytes_per_token
        if override is None:
            return self.prefill_workspace_bytes_per_token
        return override

    def auto_resident_threshold(self, kind: str) -> int:
        override = self.profile_for(kind).auto_resident_output_threshold
        if override is None:
            return self.auto_resident_output_threshold
        return override

    def as_dict(self) -> dict[str, object]:
        return {
            "memory_budget_bytes": self.memory_budget_bytes,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "prefill_workspace_bytes_per_token": self.prefill_workspace_bytes_per_token,
            "sparse_hot_tokens": self.sparse_hot_tokens,
            "prefill_tokens_per_ms": self.prefill_tokens_per_ms,
            "resident_decode_tokens_per_ms": self.resident_decode_tokens_per_ms,
            "sparse_decode_tokens_per_ms": self.sparse_decode_tokens_per_ms,
            "max_active_requests": self.max_active_requests,
            "lookahead_ms": self.lookahead_ms,
            "batch_window_ms": self.batch_window_ms,
            "max_batch_requests": self.max_batch_requests,
            "auto_resident_output_threshold": self.auto_resident_output_threshold,
            "active_prefix_prefill_reuse": self.active_prefix_prefill_reuse,
            "decode_batch_profiles": {
                str(batch_size): profile.as_dict()
                for batch_size, profile in self.decode_batch_profiles
            },
            "prefill_batch_profiles": {
                str(batch_size): rate
                for batch_size, rate in self.prefill_batch_profiles
            },
            "oracle_max_expansions": self.oracle_max_expansions,
            "online_max_rollouts": self.online_max_rollouts,
            "kind_profiles": {
                kind: profile.as_dict()
                for kind, profile in self.kind_profiles
            },
        }


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    """Capabilities that a rollout executor must verify before execution."""

    radix_cache_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.radix_cache_enabled, bool):
            raise ValueError("radix_cache_enabled must be a boolean")

    def as_dict(self) -> dict[str, object]:
        return {"radix_cache_enabled": self.radix_cache_enabled}


def validate_runtime_capabilities(
    config: PlannerConfig,
    capabilities: RuntimeCapabilities,
) -> None:
    """Reject a plan whose timing claims the live executor cannot provide."""

    if (
        config.active_prefix_prefill_reuse
        and not capabilities.radix_cache_enabled
    ):
        raise ValueError(
            "active_prefix_prefill_reuse requires the execution device's "
            "SGLang radix cache to be enabled"
        )


def _bounded_int(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
    prefix: str = "",
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{prefix}{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{prefix}{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(
            f"{prefix}{name} must be between {minimum} and {maximum}"
        )
    return value


@dataclass(frozen=True, slots=True)
class Admission:
    request_id: str
    arrival_ms: int
    start_ms: int
    finish_ms: int
    mode: ResidencyMode
    kind: str
    resident_kv_bytes: int
    prefill_workspace_bytes: int
    reason: str
    decode_batch_size: int = 1
    prefill_batch_size: int = 1
    cohort_id: str | None = None

    @property
    def startup_delay_ms(self) -> int:
        return self.start_ms - self.arrival_ms


@dataclass(frozen=True, slots=True)
class PlanResult:
    policy: str
    admissions: tuple[Admission, ...]
    peak_memory_bytes: int
    peak_backed_bytes: int
    makespan_ms: int
    weighted_completion_ms: float
    deadline_misses: int
    sparse_requests: int
    mode_switches: int
    search_expansions: int = 0
    search_truncated: bool = False
    fallback_policy: str | None = None
    search_preparation_rollouts: int = 0
    peak_shared_prefix_credit_bytes: int = 0

    @property
    def search_exact(self) -> bool:
        """Whether a bounded oracle proved its returned schedule optimal."""

        return self.policy == "oracle" and not self.search_truncated

    @property
    def throughput_requests_per_second(self) -> float:
        return 0.0 if self.makespan_ms == 0 else 1000.0 * len(self.admissions) / self.makespan_ms

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "peak_memory_bytes": self.peak_memory_bytes,
            "peak_backed_bytes": self.peak_backed_bytes,
            "peak_shared_prefix_credit_bytes": self.peak_shared_prefix_credit_bytes,
            "makespan_ms": self.makespan_ms,
            "weighted_completion_ms": self.weighted_completion_ms,
            "deadline_misses": self.deadline_misses,
            "sparse_requests": self.sparse_requests,
            "mode_switches": self.mode_switches,
            "search_expansions": self.search_expansions,
            "search_truncated": self.search_truncated,
            "search_exact": self.search_exact,
            "fallback_policy": self.fallback_policy,
            "search_preparation_rollouts": self.search_preparation_rollouts,
            "throughput_requests_per_second": self.throughput_requests_per_second,
            "admissions": [
                {
                    "request_id": item.request_id,
                    "arrival_ms": item.arrival_ms,
                    "start_ms": item.start_ms,
                    "finish_ms": item.finish_ms,
                    "mode": item.mode.value,
                    "kind": item.kind,
                    "resident_kv_bytes": item.resident_kv_bytes,
                    "prefill_workspace_bytes": item.prefill_workspace_bytes,
                    "reason": item.reason,
                    "decode_batch_size": item.decode_batch_size,
                    "prefill_batch_size": item.prefill_batch_size,
                    "cohort_id": item.cohort_id,
                }
                for item in self.admissions
            ],
        }


@dataclass(frozen=True, slots=True)
class RolloutSubmission:
    """One controller action that reproduces a simulated admission."""

    request_id: str
    planned_arrival_ms: int
    submit_at_ms: int
    estimated_finish_ms: int
    kind: str
    prompt_tokens: int
    output_tokens: int
    hint: dict[str, object]
    deadline_ms: int | None = None
    decode_batch_size: int = 1
    prefill_batch_size: int = 1
    cohort_id: str | None = None


@dataclass(frozen=True, slots=True)
class RolloutPlan:
    """A simulation plus executable ``elastic.future`` hints.

    The controller delays each request until ``submit_at_ms``.  The emitted
    hint therefore uses ``release_after_ms=0`` and converts absolute deadlines
    to scheduler-relative deadlines.  SGLang still performs every native
    admission check; this object only carries scheduling metadata.
    """

    simulation: PlanResult
    submissions: tuple[RolloutSubmission, ...]
    planner_config: PlannerConfig | None = None
    runtime_capabilities: RuntimeCapabilities | None = None
    cohort_namespace: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "policy": self.simulation.policy,
            "simulation": self.simulation.as_dict(),
            "submissions": [
                {
                    "request_id": item.request_id,
                    "planned_arrival_ms": item.planned_arrival_ms,
                    "submit_at_ms": item.submit_at_ms,
                    "estimated_finish_ms": item.estimated_finish_ms,
                    "kind": item.kind,
                    "prompt_tokens": item.prompt_tokens,
                    "output_tokens": item.output_tokens,
                    "hint": item.hint,
                    "deadline_ms": item.deadline_ms,
                    "decode_batch_size": item.decode_batch_size,
                    "prefill_batch_size": item.prefill_batch_size,
                    "cohort_id": item.cohort_id,
                }
                for item in self.submissions
            ],
        }
        if self.planner_config is not None:
            payload["planner_config"] = self.planner_config.as_dict()
        if self.runtime_capabilities is not None:
            payload["runtime_capabilities"] = self.runtime_capabilities.as_dict()
        if self.cohort_namespace is not None:
            payload["cohort_namespace"] = self.cohort_namespace
        return payload


@dataclass(slots=True)
class _Active:
    request: RequestProfile
    admission: Admission


def _ceil_duration(tokens: int, rate: float) -> int:
    return max(1, math.ceil(tokens / rate))


def _decode_batch_size(
    active: Sequence[_Active],
    extra_placements: Sequence[tuple[RequestProfile, ResidencyMode]],
    config: PlannerConfig,
) -> int:
    return min(
        config.max_active_requests,
        len(active) + len(extra_placements) + 1,
    )


def _hold_bytes(request: RequestProfile, mode: ResidencyMode, config: PlannerConfig) -> int:
    tokens = request.sequence_tokens
    if mode is ResidencyMode.HI_SPARSE:
        tokens = min(tokens, config.sparse_hot_tokens_for(request.kind))
    return tokens * config.kv_bytes_per_token


def _placement_memory(
    placements: Sequence[tuple[RequestProfile, ResidencyMode]],
    config: PlannerConfig,
) -> tuple[int, int]:
    """Return aggregate KV and shared-prefix credit for active placements.

    Only resident requests can share a complete immutable prefix. HiSparse
    requests remain fully charged by their bounded hot set because its Top-K
    selection is not guaranteed to retain the shared prefix. This is a
    conservative planner model, not a claim that the runtime lookup exists.
    """

    total = 0
    resident_prefixes: dict[str, list[int]] = {}
    for request, mode in placements:
        total += _hold_bytes(request, mode, config)
        if (
            mode is ResidencyMode.RESIDENT
            and request.prefix_group is not None
            and request.shared_prefix_tokens > 0
        ):
            resident_prefixes.setdefault(request.prefix_group, []).append(
                request.shared_prefix_tokens
            )
    credit_tokens = sum(
        (len(lengths) - 1) * min(lengths)
        for lengths in resident_prefixes.values()
        if len(lengths) > 1
    )
    credit = credit_tokens * config.kv_bytes_per_token
    return total - credit, credit


def _active_placements(
    active: Sequence[_Active],
) -> list[tuple[RequestProfile, ResidencyMode]]:
    return [
        (item.request, item.admission.mode)
        for item in active
    ]


def _prefill_tokens(
    request: RequestProfile,
    active: Sequence[_Active],
    config: PlannerConfig,
    *,
    extra_placements: Sequence[tuple[RequestProfile, ResidencyMode]] = (),
) -> int:
    """Return prompt tokens sent to prefill under the selected model.

    A resident request is a conservative proxy for a live SGLang radix-cache
    entry. HiSparse requests do not receive this credit because their bounded
    Top-K hot set is not guaranteed to retain the complete prefix.
    """

    if (
        not config.active_prefix_prefill_reuse
        or request.prefix_group is None
        or request.shared_prefix_tokens <= 0
    ):
        return request.prompt_tokens
    placements = (*_active_placements(active), *extra_placements)
    if any(
        mode is ResidencyMode.RESIDENT
        and candidate.prefix_group == request.prefix_group
        and candidate.shared_prefix_tokens >= request.shared_prefix_tokens
        for candidate, mode in placements
    ):
        return max(1, request.prompt_tokens - request.shared_prefix_tokens)
    return request.prompt_tokens


def _prefill_duration(
    request: RequestProfile,
    active: Sequence[_Active],
    config: PlannerConfig,
    *,
    extra_placements: Sequence[tuple[RequestProfile, ResidencyMode]] = (),
) -> int:
    return _ceil_duration(
        _prefill_tokens(
            request,
            active,
            config,
            extra_placements=extra_placements,
        ),
        config.prefill_rate(request.kind),
    )


def _duration_ms(
    request: RequestProfile,
    mode: ResidencyMode,
    config: PlannerConfig,
    active: Sequence[_Active] = (),
    *,
    extra_placements: Sequence[tuple[RequestProfile, ResidencyMode]] = (),
) -> int:
    prefill = _prefill_duration(
        request,
        active,
        config,
        extra_placements=extra_placements,
    )
    batch_size = _decode_batch_size(active, extra_placements, config)
    decode = _ceil_duration(
        request.output_tokens,
        config.decode_rate_for_batch(request.kind, mode, batch_size),
    )
    return prefill + decode


def _urgency(
    request: RequestProfile,
    active: Sequence[_Active],
    now_ms: int,
    config: PlannerConfig,
) -> float:
    if request.deadline_ms is None:
        return 0.0
    slack = (
        request.deadline_ms
        - now_ms
        - _duration_ms(request, ResidencyMode.RESIDENT, config, active)
    )
    horizon = max(1, config.lookahead_ms or 1)
    if slack < 0:
        return 1.0 + min(1.0, -slack / horizon)
    return 1.0 / (1.0 + slack / horizon)


def _future_contention(
    request: RequestProfile,
    mode: ResidencyMode,
    active: Sequence[_Active],
    pending: Sequence[RequestProfile],
    future: Sequence[RequestProfile],
    now_ms: int,
    config: PlannerConfig,
) -> int:
    """Count imminent valuable requests displaced by this admission.

    The comparison gives every future request its bounded HiSparse footprint.
    It is deliberately conservative: one displaced request is enough to make
    an otherwise faster resident placement less attractive.
    """

    if (not pending and not future) or mode is ResidencyMode.HI_SPARSE:
        return 0
    finish = now_ms + _duration_ms(request, mode, config, active)
    spend, _ = _placement_memory(
        (*_active_placements(active), (request, mode)), config
    )
    displaced = 0
    future_need = 0
    for item in (*pending, *future):
        if item.request_id == request.request_id or item.arrival_ms > finish:
            continue
        valuable = item.priority >= request.priority or (
            item.deadline_ms is not None and item.deadline_ms <= finish
        )
        if not valuable:
            continue
        required = _hold_bytes(item, ResidencyMode.HI_SPARSE, config)
        required += (
            item.prompt_tokens
            * config.prefill_workspace_rate(item.kind)
        )
        future_need += required
        if spend + future_need > config.memory_budget_bytes:
            displaced += 1
    return displaced


def _feasible_mode(
    request: RequestProfile,
    active: Sequence[_Active],
    config: PlannerConfig,
    preferred: ResidencyMode = ResidencyMode.RESIDENT,
    *,
    extra_placements: Sequence[tuple[RequestProfile, ResidencyMode]] = (),
) -> tuple[ResidencyMode, str] | None:
    if len(active) + len(extra_placements) > config.max_active_requests:
        return None
    workspace = request.prompt_tokens * config.prefill_workspace_rate(
        request.kind
    )
    native_hold = _hold_bytes(request, ResidencyMode.HI_SPARSE, config)
    placements = (*_active_placements(active), *extra_placements)
    if request.sequence_tokens <= config.sparse_hot_tokens_for(request.kind):
        active_bytes, _ = _placement_memory(placements, config)
        if active_bytes + workspace + native_hold <= config.memory_budget_bytes:
            return ResidencyMode.HI_SPARSE, "fits-native-hot-set"
        return None
    if request.residence_hint == "resident":
        order = (ResidencyMode.RESIDENT, ResidencyMode.HI_SPARSE)
    elif request.residence_hint == "host_backed":
        order = (ResidencyMode.HI_SPARSE,)
    elif request.output_tokens < config.auto_resident_threshold(request.kind):
        order = (ResidencyMode.HI_SPARSE,)
    else:
        order = (preferred, ResidencyMode.HI_SPARSE if preferred is ResidencyMode.RESIDENT else ResidencyMode.RESIDENT)
    for mode in order:
        aggregate, _ = _placement_memory(
            (*placements, (request, mode)), config
        )
        if aggregate + workspace <= config.memory_budget_bytes:
            if request.residence_hint == "resident":
                reason = (
                    "hint-resident"
                    if mode is ResidencyMode.RESIDENT
                    else "hint-resident-falls-back"
                )
            elif request.residence_hint == "host_backed":
                reason = "hint-requires-bounded-hot-set"
            elif request.output_tokens < config.auto_resident_threshold(request.kind):
                reason = "hint-short-auto"
            else:
                reason = "fits-resident" if mode is ResidencyMode.RESIDENT else "fits-bounded-hot-set"
            return mode, reason
    return None


def _defer_for_imminent_work(
    choice: RequestProfile,
    mode: ResidencyMode,
    active: Sequence[_Active],
    future: Sequence[RequestProfile],
    now_ms: int,
    config: PlannerConfig,
) -> bool:
    """Return whether starting ``choice`` now blocks more valuable work.

    This helper models the controlled-rollout case where the caller knows a
    request is about to arrive and may intentionally leave the prefill engine
    idle for a short interval.  It only defers when the future request fits the
    current live set, the current prefill would delay it, and the current
    choice can tolerate being placed behind that work.
    """

    current_prefill = _prefill_duration(choice, active, config)
    choice_duration = _duration_ms(choice, mode, config, active)
    for item in future:
        if item.arrival_ms > now_ms + config.lookahead_ms:
            break
        if item.deadline_ms is None:
            continue
        standalone = _feasible_mode(item, active, config)
        if standalone is None:
            continue
        future_mode = standalone[0]
        future_duration = _duration_ms(
            item,
            future_mode,
            config,
            active,
            extra_placements=((choice, mode),),
        )
        future_prefill = _prefill_duration(
            item,
            active,
            config,
            extra_placements=((choice, mode),),
        )
        blocked_start = max(item.arrival_ms, now_ms + current_prefill)
        delayed_finish = blocked_start + future_duration

        with_choice = _feasible_mode(
            item,
            active,
            config,
            extra_placements=((choice, mode),),
        )
        memory_blocked = with_choice is None
        deadline_blocked = (
            delayed_finish > item.deadline_ms
            and item.arrival_ms + future_duration <= item.deadline_ms
        )
        # Idle only to protect a hard latency objective.  Merely having a
        # higher-weight request in the horizon should not leave otherwise
        # useful prefill capacity idle: native priority handles that case when
        # the request actually arrives.
        if not (memory_blocked or deadline_blocked):
            continue

        if choice.deadline_ms is not None:
            # If both requests cannot coexist, the choice cannot restart until
            # the future request has released its decode residency.  A pure
            # prefill-only estimate would incorrectly claim that the current
            # request can still meet its deadline after idling.
            delayed_start = item.arrival_ms + (
                future_duration if memory_blocked else future_prefill
            )
            delayed_choice_finish = delayed_start + choice_duration
            if delayed_choice_finish > choice.deadline_ms:
                continue
            if item.deadline_ms > choice.deadline_ms:
                continue
        return True
    return False


def _defer_batch_for_imminent_work(
    requests: Sequence[RequestProfile],
    modes: Sequence[ResidencyMode],
    active: Sequence[_Active],
    future: Sequence[RequestProfile],
    now_ms: int,
    config: PlannerConfig,
) -> bool:
    """Protect a known deadline from an entire ready prefill cohort."""

    placements = tuple(zip(requests, modes, strict=True))
    current_prefill = _prefill_batch_duration(
        requests, modes, active, config
    )
    for item in future:
        if item.arrival_ms > now_ms + config.lookahead_ms:
            break
        if item.deadline_ms is None:
            continue
        standalone = _feasible_mode(
            item,
            active,
            config,
            extra_placements=placements,
        )
        future_mode = (
            standalone[0]
            if standalone is not None
            else ResidencyMode.HI_SPARSE
        )
        future_duration = _duration_ms(
            item,
            future_mode,
            config,
            active,
            extra_placements=placements,
        )
        future_prefill = _prefill_duration(
            item,
            active,
            config,
            extra_placements=placements,
        )
        blocked_start = max(item.arrival_ms, now_ms + current_prefill)
        deadline_blocked = (
            blocked_start + future_duration > item.deadline_ms
            and item.arrival_ms + future_duration <= item.deadline_ms
        )
        if not (standalone is None or deadline_blocked):
            continue

        protected_tier = False
        for request, mode in placements:
            if request.deadline_ms is None:
                continue
            delayed_start = item.arrival_ms + (
                future_duration if standalone is None else future_prefill
            )
            delayed_finish = delayed_start + _duration_ms(
                request, mode, config, active
            )
            if delayed_finish > request.deadline_ms:
                protected_tier = True
                break
            if item.deadline_ms > request.deadline_ms:
                protected_tier = True
                break
        if protected_tier:
            continue
        return True
    return False


def _select(
    policy: str,
    unstarted: dict[str, RequestProfile],
    active: Sequence[_Active],
    now_ms: int,
    config: PlannerConfig,
) -> tuple[RequestProfile, ResidencyMode, str] | None:
    pending = sorted(
        (item for item in unstarted.values() if item.arrival_ms <= now_ms),
        key=lambda item: (item.arrival_ms, item.request_id),
    )
    if not pending:
        return None
    if policy == "fifo":
        choice = pending[0]
        selected = _feasible_mode(choice, active, config)
        return None if selected is None else (choice, *selected)

    if policy == "longest":
        for choice in sorted(pending, key=lambda item: (-item.sequence_tokens, item.request_id)):
            selected = _feasible_mode(choice, active, config)
            if selected is not None:
                return choice, *selected
        return None

    if policy == "mode-aware":
        ranked = sorted(
            pending,
            key=lambda item: (
                -_urgency(item, active, now_ms, config),
                -item.priority,
                _hold_bytes(item, ResidencyMode.RESIDENT, config),
                item.request_id,
            ),
        )
        for choice in ranked:
            selected = _feasible_mode(choice, active, config)
            if selected is not None:
                return choice, *selected
        return None

    if policy != "future":
        raise ValueError(f"unknown planning policy: {policy!r}")

    future = sorted(
        (
            item
            for item in unstarted.values()
            if now_ms < item.arrival_ms <= now_ms + config.lookahead_ms
        ),
        key=lambda item: (item.arrival_ms, item.request_id),
    )
    options: list[tuple[float, str, str, RequestProfile, ResidencyMode, str]] = []
    for choice in pending:
        first = _feasible_mode(choice, active, config)
        if first is None:
            continue
        modes: list[tuple[ResidencyMode, str]] = [first]
        workspace = choice.prompt_tokens * config.prefill_workspace_rate(
            choice.kind
        )
        sparse_memory, _ = _placement_memory(
            (
                *_active_placements(active),
                (choice, ResidencyMode.HI_SPARSE),
            ),
            config,
        )
        if (choice.residence_hint == "auto"
                and choice.output_tokens
                >= config.auto_resident_threshold(choice.kind)
                and first[0] is ResidencyMode.RESIDENT
                and sparse_memory + workspace <= config.memory_budget_bytes):
            modes.append((ResidencyMode.HI_SPARSE, "lookahead-protects-imminent-demand"))
        for mode, reason in modes:
            displaced = _future_contention(
                choice, mode, active, pending, future, now_ms, config
            )
            urgency = _urgency(choice, active, now_ms, config)
            # Priority and deadlines decide whether work is worth reserving.
            # Sparse placement is cheaper for following requests, while its
            # decode slowdown remains an explicit cost in the simulator.
            score = 100.0 * choice.priority + 60.0 * urgency - 20.0 * displaced
            if mode is ResidencyMode.HI_SPARSE:
                slowdown = (
                    config.decode_rate(choice.kind, ResidencyMode.RESIDENT)
                    / config.decode_rate(choice.kind, ResidencyMode.HI_SPARSE)
                )
                score -= 25.0 * max(0.0, slowdown - 1.0)
            # Prefer waiting when even a bounded placement would displace a
            # sufficiently more valuable request that is about to arrive.
            if future and mode is ResidencyMode.HI_SPARSE and displaced:
                best_future = max(item.priority for item in future)
                if best_future > choice.priority and choice.deadline_ms is None:
                    score -= 150.0
            options.append((score, choice.request_id, mode.value, choice, mode, reason))
    if not options:
        return None
    _, _, _, choice, mode, reason = min(
        options,
        key=lambda item: (
            -item[0],
            item[3].arrival_ms,
            item[1],
            0 if item[4] is ResidencyMode.RESIDENT else 1,
        ),
    )
    if _defer_for_imminent_work(
        choice, mode, active, future, now_ms, config
    ):
        return None
    return choice, mode, reason


@dataclass(frozen=True, slots=True)
class _OracleState:
    """One immutable node in the bounded known-future search."""

    now_ms: int
    prefill_busy_until: int
    remaining_ids: tuple[str, ...]
    active: tuple[_Active, ...]
    admissions: tuple[Admission, ...]
    peak_memory_bytes: int
    peak_shared_prefix_credit_bytes: int


def _feasible_modes(
    request: RequestProfile,
    active: Sequence[_Active],
    config: PlannerConfig,
) -> tuple[tuple[ResidencyMode, str], ...]:
    """Return every policy-compatible placement for an oracle branch."""

    modes: list[tuple[ResidencyMode, str]] = []
    seen: set[ResidencyMode] = set()
    for preferred in (ResidencyMode.RESIDENT, ResidencyMode.HI_SPARSE):
        selected = _feasible_mode(request, active, config, preferred=preferred)
        if selected is None or selected[0] in seen:
            continue
        modes.append(selected)
        seen.add(selected[0])
    return tuple(modes)


def _normalize_active(
    active: Sequence[_Active], now_ms: int
) -> tuple[_Active, ...]:
    return tuple(item for item in active if item.admission.finish_ms > now_ms)


def _admission_child(
    state: _OracleState,
    request: RequestProfile,
    mode: ResidencyMode,
    reason: str,
    config: PlannerConfig,
) -> _OracleState:
    active = _normalize_active(state.active, state.now_ms)
    hold = _hold_bytes(request, mode, config)
    workspace = request.prompt_tokens * config.prefill_workspace_rate(
        request.kind
    )
    prefill = _prefill_duration(request, active, config)
    base_prefill = _ceil_duration(
        request.prompt_tokens, config.prefill_rate(request.kind)
    )
    if prefill < base_prefill:
        reason = f"{reason}+active-prefix-reuse"
    aggregate, shared_credit = _placement_memory(
        (
            *_active_placements(active),
            (request, mode),
        ),
        config,
    )
    admission = Admission(
        request_id=request.request_id,
        arrival_ms=request.arrival_ms,
        start_ms=state.now_ms,
        finish_ms=state.now_ms + _duration_ms(request, mode, config, active),
        mode=mode,
        kind=request.kind,
        resident_kv_bytes=hold,
        prefill_workspace_bytes=workspace,
        reason=reason,
        decode_batch_size=_decode_batch_size(active, (), config),
    )
    return _OracleState(
        now_ms=state.now_ms + prefill,
        prefill_busy_until=state.now_ms + prefill,
        remaining_ids=tuple(
            item for item in state.remaining_ids if item != request.request_id
        ),
        active=(*active, _Active(request, admission)),
        admissions=(*state.admissions, admission),
        peak_memory_bytes=max(state.peak_memory_bytes, aggregate + workspace),
        peak_shared_prefix_credit_bytes=max(
            state.peak_shared_prefix_credit_bytes, shared_credit
        ),
    )


def _oracle_idle_child(
    state: _OracleState,
    requests_by_id: dict[str, RequestProfile],
) -> _OracleState | None:
    active = _normalize_active(state.active, state.now_ms)
    wake_times = [
        requests_by_id[request_id].arrival_ms
        for request_id in state.remaining_ids
        if requests_by_id[request_id].arrival_ms > state.now_ms
    ]
    wake_times.extend(
        item.admission.finish_ms
        for item in active
        if item.admission.finish_ms > state.now_ms
    )
    if not wake_times:
        return None
    return replace(
        state,
        now_ms=min(wake_times),
        active=active,
    )


def _assign_batch_modes(
    requests: Sequence[RequestProfile],
    active: Sequence[_Active],
    config: PlannerConfig,
) -> tuple[tuple[ResidencyMode, ...], tuple[str, ...]] | None:
    """Choose a joint feasible placement for one prefill cohort.

    The search is bounded by the already limited scheduler batch. Prefix credit
    depends on the complete resident set, so assigning modes greedily can miss
    a feasible mixed resident/HiSparse cohort.
    """

    if (
        not requests
        or len(requests) > config.max_batch_requests
        or len(active) + len(requests) > config.max_active_requests
    ):
        return None
    choices = tuple(_feasible_modes(item, active, config) for item in requests)
    if any(not item for item in choices):
        return None
    workspace = sum(
        item.prompt_tokens * config.prefill_workspace_rate(item.kind)
        for item in requests
    )
    best: tuple[
        tuple[int, tuple[int, ...], tuple[str, ...]],
        tuple[ResidencyMode, ...],
        tuple[str, ...],
    ] | None = None
    for candidate in product(*choices):
        modes = tuple(item[0] for item in candidate)
        reasons = tuple(item[1] for item in candidate)
        placements = (
            *_active_placements(active),
            *zip(requests, modes, strict=True),
        )
        aggregate, _ = _placement_memory(placements, config)
        if aggregate + workspace > config.memory_budget_bytes:
            continue
        preference_penalty = tuple(
            next(
                index
                for index, (mode, _) in enumerate(request_choices)
                if mode is selected
            )
            for request_choices, selected in zip(choices, modes, strict=True)
        )
        sparse_count = sum(mode is ResidencyMode.HI_SPARSE for mode in modes)
        key = (
            sparse_count,
            preference_penalty,
            tuple(mode.value for mode in modes),
        )
        if best is None or key < best[0]:
            best = (key, modes, reasons)
    if best is None:
        return None
    return best[1], best[2]


def _prefill_batch_tokens(
    requests: Sequence[RequestProfile],
    modes: Sequence[ResidencyMode],
    active: Sequence[_Active],
    config: PlannerConfig,
) -> tuple[int, int]:
    """Return aggregate tokens and no-profile serial prefill duration."""

    total_tokens = 0
    serial_duration = 0
    cohort_placements: list[tuple[RequestProfile, ResidencyMode]] = []
    for request, mode in zip(requests, modes, strict=True):
        tokens = _prefill_tokens(
            request,
            active,
            config,
            extra_placements=cohort_placements,
        )
        total_tokens += tokens
        serial_duration += _ceil_duration(
            tokens, config.prefill_rate(request.kind)
        )
        cohort_placements.append((request, mode))
    return total_tokens, serial_duration


def _prefill_batch_duration(
    requests: Sequence[RequestProfile],
    modes: Sequence[ResidencyMode],
    active: Sequence[_Active],
    config: PlannerConfig,
) -> int:
    """Return measured aggregate prefill time or the serial fallback."""

    total_tokens, serial_duration = _prefill_batch_tokens(
        requests, modes, active, config
    )
    for batch_size, aggregate_rate in config.prefill_batch_profiles:
        if batch_size == len(requests):
            return _ceil_duration(total_tokens, aggregate_rate)
    return serial_duration


def _batch_decode_improves(
    requests: Sequence[RequestProfile],
    modes: Sequence[ResidencyMode],
    base_batch_size: int,
    combined_batch_size: int,
    config: PlannerConfig,
) -> bool:
    profile = config.decode_batch_profile(combined_batch_size)
    if profile is None:
        return False
    for request, mode in zip(requests, modes, strict=True):
        measured = profile.rate(mode)
        if measured is None:
            continue
        if measured / combined_batch_size > config.decode_rate_for_batch(
            request.kind, mode, base_batch_size
        ):
            return True
    return False


def _wait_for_batch_improves(
    base: Sequence[RequestProfile],
    future: RequestProfile,
    active: Sequence[_Active],
    now_ms: int,
    config: PlannerConfig,
) -> bool:
    """Return whether a measured profile justifies waiting for one arrival."""

    combined = (*base, future)
    base_assignment = _assign_batch_modes(base, active, config)
    combined_assignment = _assign_batch_modes(combined, active, config)
    if base_assignment is None or combined_assignment is None:
        return False
    base_modes, _ = base_assignment
    combined_modes, _ = combined_assignment

    base_prefill = _prefill_batch_duration(
        base, base_modes, active, config
    )
    _, base_serial = _prefill_batch_tokens(base, base_modes, active, config)
    future_prefill, _ = _prefill_batch_tokens(
        (future,),
        (combined_modes[-1],),
        active,
        config,
    )
    combined_prefill = _prefill_batch_duration(
        combined, combined_modes, active, config
    )
    has_prefill_profile = any(
        batch_size == len(combined)
        for batch_size, _ in config.prefill_batch_profiles
    )
    prefill_gain = (
        has_prefill_profile
        and combined_prefill < base_serial + future_prefill
    )

    base_size = min(
        config.max_active_requests, len(active) + len(base)
    )
    combined_size = min(
        config.max_active_requests, len(active) + len(combined)
    )
    decode_gain = _batch_decode_improves(
        combined,
        combined_modes,
        base_size,
        combined_size,
        config,
    )
    if not (prefill_gain or decode_gain):
        return False

    start_ms = max(now_ms, future.arrival_ms)
    base_finishes = [
        now_ms
        + base_prefill
        + _ceil_duration(
            request.output_tokens,
            config.decode_rate_for_batch(request.kind, mode, base_size),
        )
        for request, mode in zip(base, base_modes, strict=True)
    ]
    future_finish_without_wait = (
        future.arrival_ms
        + _prefill_duration(
            future,
            active,
            config,
            extra_placements=tuple(zip(base, base_modes, strict=True)),
        )
        + _ceil_duration(
            future.output_tokens,
            config.decode_rate_for_batch(
                future.kind, combined_modes[-1], combined_size
            ),
        )
    )
    combined_finishes = [
        start_ms
        + combined_prefill
        + _ceil_duration(
            request.output_tokens,
            config.decode_rate_for_batch(request.kind, mode, combined_size),
        )
        for request, mode in zip(combined, combined_modes, strict=True)
    ]
    without_wait_weight = math.fsum(item.priority for item in combined)
    without_wait_completion = math.fsum(
        [
            request.priority * finish
            for request, finish in zip(base, base_finishes, strict=True)
        ]
        + [future.priority * future_finish_without_wait]
    )
    wait_completion = math.fsum(
        request.priority * finish
        for request, finish in zip(combined, combined_finishes, strict=True)
    )
    if without_wait_weight <= 0:
        return False
    without_wait_weighted = without_wait_completion / without_wait_weight
    wait_weighted = wait_completion / without_wait_weight
    without_wait_makespan = max(*base_finishes, future_finish_without_wait)
    wait_makespan = max(combined_finishes)
    if wait_weighted > without_wait_weighted or wait_makespan > without_wait_makespan:
        return False
    if (
        wait_weighted == without_wait_weighted
        and wait_makespan == without_wait_makespan
    ):
        return False

    for index, (request, mode) in enumerate(
        zip(base, base_modes, strict=True)
    ):
        if request.deadline_ms is None:
            continue
        current_finish = base_finishes[index]
        wait_finish = start_ms + combined_prefill + _ceil_duration(
            request.output_tokens,
            config.decode_rate_for_batch(
                request.kind, mode, combined_size
            ),
        )
        if current_finish <= request.deadline_ms < wait_finish:
            return False

    future_mode = combined_modes[-1]
    if future.deadline_ms is not None:
        future_finish = start_ms + combined_prefill + _ceil_duration(
            future.output_tokens,
            config.decode_rate_for_batch(
                future.kind, future_mode, combined_size
            ),
        )
        if future_finish > future.deadline_ms:
            return False
    return True


def _has_measured_batch_signal(
    requests: Sequence[RequestProfile],
    modes: Sequence[ResidencyMode],
    active: Sequence[_Active],
    config: PlannerConfig,
) -> bool:
    """Return whether an exact measured profile supports this cohort."""

    if len(requests) <= 1:
        return True
    if any(
        batch_size == len(requests)
        for batch_size, _ in config.prefill_batch_profiles
    ):
        return True
    total_batch_size = min(
        config.max_active_requests, len(active) + len(requests)
    )
    profile = config.decode_batch_profile(total_batch_size)
    return profile is not None and any(
        profile.rate(mode) is not None for mode in modes
    )


def _ready_batch(
    pending: Sequence[RequestProfile],
    active: Sequence[_Active],
    now_ms: int,
    config: PlannerConfig,
) -> tuple[tuple[RequestProfile, ...], tuple[ResidencyMode, ...], tuple[str, ...]]:
    """Build the largest joint-feasible cohort without creating a miss."""

    selected: list[RequestProfile] = []
    selected_assignment = None
    for request in pending:
        if len(selected) >= config.max_batch_requests:
            break
        candidate = (*selected, request)
        assignment = _assign_batch_modes(candidate, active, config)
        if assignment is None:
            continue
        modes = assignment[0]
        prefill = _prefill_batch_duration(candidate, modes, active, config)
        decode_batch_size = min(
            config.max_active_requests, len(active) + len(candidate)
        )
        deadline_safe = True
        for item, mode in zip(candidate, modes, strict=True):
            if item.deadline_ms is None:
                continue
            combined_finish = now_ms + prefill + _ceil_duration(
                item.output_tokens,
                config.decode_rate_for_batch(
                    item.kind, mode, decode_batch_size
                ),
            )
            if combined_finish <= item.deadline_ms:
                continue
            standalone_prefill = _prefill_batch_duration(
                (item,), (mode,), active, config
            )
            standalone_finish = now_ms + standalone_prefill + _ceil_duration(
                item.output_tokens,
                config.decode_rate_for_batch(
                    item.kind,
                    mode,
                    min(config.max_active_requests, len(active) + 1),
                ),
            )
            if standalone_finish <= item.deadline_ms:
                deadline_safe = False
                break
        if not deadline_safe:
            continue
        selected.append(request)
        selected_assignment = assignment
    if selected_assignment is None:
        return (), (), ()

    # Serialization is the conservative model when no exact cohort profile
    # exists.  Forming a larger cohort would delay each member's decode until
    # every earlier member has prefetched without any measured throughput gain.
    # Keep any smaller prefix that does have an exact profile; otherwise
    # execute one request and let the event loop add newly-ready work later.
    if len(selected) > 1:
        for subset_size in range(len(selected), 1, -1):
            candidate = tuple(selected[:subset_size])
            assignment = _assign_batch_modes(candidate, active, config)
            if assignment is None:
                continue
            candidate_modes, candidate_reasons = assignment
            if not _has_measured_batch_signal(
                candidate, candidate_modes, active, config
            ):
                continue
            selected = list(candidate)
            selected_assignment = (candidate_modes, candidate_reasons)
            break
        else:
            selected = selected[:1]
            selected_assignment = _assign_batch_modes(
                tuple(selected), active, config
            )
            if selected_assignment is None:
                return (), (), ()
    return tuple(selected), *selected_assignment


def _batched_order(request: RequestProfile) -> tuple[object, ...]:
    """Order arrivals by deadlines, then value, then deterministic FIFO."""

    return (
        0 if request.deadline_ms is not None else 1,
        request.deadline_ms if request.deadline_ms is not None else math.inf,
        -request.priority,
        request.arrival_ms,
        request.request_id,
    )


def _simulate_batched(
    workload: Sequence[RequestProfile],
    config: PlannerConfig,
) -> PlanResult:
    """Form measured prefill/decode cohorts from known near-future arrivals."""

    requests_by_id = {item.request_id: item for item in workload}
    remaining_ids = set(requests_by_id)
    active: tuple[_Active, ...] = ()
    admissions: list[Admission] = []
    now_ms = min(item.arrival_ms for item in workload)
    prefill_busy_until = -1
    peak_memory_bytes = 0
    peak_shared_prefix_credit_bytes = 0
    cohort_index = 0

    while remaining_ids or active:
        active = _normalize_active(active, now_ms)
        if not remaining_ids and not active:
            break
        if not remaining_ids:
            now_ms = min(item.admission.finish_ms for item in active)
            continue
        if prefill_busy_until > now_ms:
            wake_times = [prefill_busy_until]
            wake_times.extend(
                requests_by_id[request_id].arrival_ms
                for request_id in remaining_ids
                if requests_by_id[request_id].arrival_ms > now_ms
            )
            wake_times.extend(
                item.admission.finish_ms
                for item in active
                if item.admission.finish_ms > now_ms
            )
            now_ms = min(wake_times)
            continue

        pending = sorted(
            (
                requests_by_id[request_id]
                for request_id in remaining_ids
                if requests_by_id[request_id].arrival_ms <= now_ms
            ),
            key=_batched_order,
        )
        if not pending:
            wake_times = [
                requests_by_id[request_id].arrival_ms
                for request_id in remaining_ids
                if requests_by_id[request_id].arrival_ms > now_ms
            ]
            wake_times.extend(
                item.admission.finish_ms
                for item in active
                if item.admission.finish_ms > now_ms
            )
            if not wake_times:
                raise RuntimeError(f"batched planner deadlock at {now_ms} ms")
            now_ms = min(wake_times)
            continue

        base, modes, reasons = _ready_batch(pending, active, now_ms, config)
        if not base:
            wake_times = [
                item.admission.finish_ms
                for item in active
                if item.admission.finish_ms > now_ms
            ]
            wake_times.extend(
                requests_by_id[request_id].arrival_ms
                for request_id in remaining_ids
                if requests_by_id[request_id].arrival_ms > now_ms
            )
            if not wake_times:
                raise RuntimeError(
                    f"batched planner cannot place ready work at {now_ms} ms"
                )
            now_ms = min(wake_times)
            continue

        if config.batch_window_ms > 0:
            horizon = now_ms + config.batch_window_ms
            near_future = sorted(
                (
                    requests_by_id[request_id]
                    for request_id in remaining_ids
                    if now_ms < requests_by_id[request_id].arrival_ms <= horizon
                ),
                key=_batched_order,
            )
            batching_candidate = next(
                (
                    candidate
                    for candidate in near_future
                    if len(base) < config.max_batch_requests
                    and _wait_for_batch_improves(
                        base, candidate, active, now_ms, config
                    )
                ),
                None,
            )
            if batching_candidate is not None:
                now_ms = batching_candidate.arrival_ms
                continue

            if near_future and _defer_batch_for_imminent_work(
                base,
                modes,
                active,
                near_future,
                now_ms,
                config,
            ):
                # A long ready request can make the whole cohort temporarily
                # unsafe for an imminent hard deadline.  Waiting is correct
                # when even the highest-priority ready request would block the
                # future work.  Otherwise, advance a safe subset now and keep
                # the harmful members queued for a later cohort.
                safe_subset = None
                for subset_size in range(len(base) - 1, 0, -1):
                    candidate = base[:subset_size]
                    candidate_assignment = _assign_batch_modes(
                        candidate, active, config
                    )
                    if candidate_assignment is None:
                        continue
                    candidate_modes, candidate_reasons = candidate_assignment
                    if not _has_measured_batch_signal(
                        candidate, candidate_modes, active, config
                    ):
                        continue
                    if _defer_batch_for_imminent_work(
                        candidate,
                        candidate_modes,
                        active,
                        near_future,
                        now_ms,
                        config,
                    ):
                        continue
                    safe_subset = (
                        candidate,
                        candidate_modes,
                        candidate_reasons,
                    )
                    break
                if safe_subset is None:
                    now_ms = near_future[0].arrival_ms
                    continue
                base, modes, reasons = safe_subset

        prefill = _prefill_batch_duration(base, modes, active, config)
        decode_batch_size = min(
            config.max_active_requests,
            len(active) + len(base),
        )
        cohort_id = (
            f"cohort-{cohort_index:04d}-{now_ms}"
            if len(base) > 1
            else None
        )
        if cohort_id is not None:
            cohort_index += 1
        aggregate, shared_credit = _placement_memory(
            (
                *_active_placements(active),
                *zip(base, modes, strict=True),
            ),
            config,
        )
        workspace = sum(
            item.prompt_tokens * config.prefill_workspace_rate(item.kind)
            for item in base
        )
        peak_memory_bytes = max(
            peak_memory_bytes, aggregate + workspace
        )
        peak_shared_prefix_credit_bytes = max(
            peak_shared_prefix_credit_bytes, shared_credit
        )
        new_active: list[_Active] = list(active)
        for request, mode, reason in zip(
            base, modes, reasons, strict=True
        ):
            finish_ms = now_ms + prefill + _ceil_duration(
                request.output_tokens,
                config.decode_rate_for_batch(
                    request.kind, mode, decode_batch_size
                ),
            )
            admission = Admission(
                request_id=request.request_id,
                arrival_ms=request.arrival_ms,
                start_ms=now_ms,
                finish_ms=finish_ms,
                mode=mode,
                kind=request.kind,
                resident_kv_bytes=_hold_bytes(request, mode, config),
                prefill_workspace_bytes=(
                    request.prompt_tokens
                    * config.prefill_workspace_rate(request.kind)
                ),
                reason=f"batched:{reason}",
                decode_batch_size=decode_batch_size,
                prefill_batch_size=len(base),
                cohort_id=cohort_id,
            )
            admissions.append(admission)
            new_active.append(_Active(request, admission))
            remaining_ids.remove(request.request_id)
        active = tuple(new_active)
        prefill_busy_until = now_ms + prefill
        now_ms = prefill_busy_until

    if len(admissions) != len(workload):
        raise RuntimeError("batched planner did not admit every request")
    return _result_from_admissions(
        "batched",
        workload,
        admissions,
        peak_memory_bytes,
        peak_shared_prefix_credit_bytes=peak_shared_prefix_credit_bytes,
    )


def _oracle_objective(result: PlanResult) -> tuple[object, ...]:
    # The order matches the research contract: hard deadlines first, then user
    # and batch completion, followed by lower-cost sparse placement.  The last
    # component makes equal-cost plans deterministic across Python versions.
    signature = tuple(
        (item.request_id, item.start_ms, item.mode.value)
        for item in sorted(result.admissions, key=lambda item: item.request_id)
    )
    return (
        result.deadline_misses,
        result.weighted_completion_ms,
        result.makespan_ms,
        result.sparse_requests,
        signature,
    )


def _oracle_symmetry_key(request: RequestProfile) -> tuple[object, ...]:
    """Identify requests whose schedules are interchangeable.

    The final objective sorts equal plans by request ID.  Once pending
    requests have identical resource and timing fields, admitting a later ID
    before the lexicographically first one can only worsen that tie-break.
    Collapsing those permutations is exact and prevents factorial blow-up in
    rollout batches containing repeated tool or evaluation prompts.
    """

    return (
        request.arrival_ms,
        request.prompt_tokens,
        request.output_tokens,
        request.kind,
        request.priority,
        request.deadline_ms,
        request.prefix_group,
        request.shared_prefix_tokens,
        request.residence_hint,
    )


def _result_from_admissions(
    policy: str,
    workload: Sequence[RequestProfile],
    admissions: Sequence[Admission],
    peak_memory_bytes: int,
    *,
    peak_shared_prefix_credit_bytes: int = 0,
    search_expansions: int = 0,
    search_truncated: bool = False,
    fallback_policy: str | None = None,
    search_preparation_rollouts: int = 0,
) -> PlanResult:
    admission_by_id = {item.request_id: item for item in admissions}
    weight = sum(item.priority for item in workload)
    weighted_completion = sum(
        item.priority * admission_by_id[item.request_id].finish_ms
        for item in workload
    ) / weight
    misses = sum(
        item.deadline_ms is not None
        and admission_by_id[item.request_id].finish_ms > item.deadline_ms
        for item in workload
    )
    ordered = tuple(
        sorted(admissions, key=lambda item: (item.start_ms, item.request_id))
    )
    return PlanResult(
        policy=policy,
        admissions=ordered,
        peak_memory_bytes=peak_memory_bytes,
        peak_backed_bytes=peak_memory_bytes,
        makespan_ms=max(item.finish_ms for item in admissions),
        weighted_completion_ms=weighted_completion,
        deadline_misses=misses,
        sparse_requests=sum(
            item.mode is ResidencyMode.HI_SPARSE for item in admissions
        ),
        mode_switches=0,
        search_expansions=search_expansions,
        search_truncated=search_truncated,
        fallback_policy=fallback_policy,
        search_preparation_rollouts=search_preparation_rollouts,
        peak_shared_prefix_credit_bytes=peak_shared_prefix_credit_bytes,
    )


def _simulate_oracle(
    workload: Sequence[RequestProfile],
    config: PlannerConfig,
) -> PlanResult:
    """Search known-future schedules under a deterministic expansion cap.

    Every node may admit any currently eligible request in either feasible
    residency mode or deliberately advance to the next arrival/completion.  A
    cap makes this usable as a research oracle and guarantees a deployable
    heuristic fallback for traces that are too large to enumerate.
    """

    requests_by_id = {item.request_id: item for item in workload}
    initial = _OracleState(
        now_ms=min(item.arrival_ms for item in workload),
        prefill_busy_until=-1,
        remaining_ids=tuple(item.request_id for item in workload),
        active=(),
        admissions=(),
        peak_memory_bytes=0,
        peak_shared_prefix_credit_bytes=0,
    )
    stack = [initial]
    future = _simulate_greedy(workload, config, "future")
    online = _simulate_online(workload, config)
    incumbent = min((future, online), key=_oracle_objective)
    best = replace(
        incumbent,
        policy="oracle",
        fallback_policy=incumbent.policy,
    )
    best_from_search = False
    expansions = 0
    truncated = False

    while stack:
        if expansions >= config.oracle_max_expansions:
            truncated = True
            break
        state = stack.pop()
        expansions += 1
        active = _normalize_active(state.active, state.now_ms)
        state = replace(state, active=active)

        if not state.remaining_ids:
            result = _result_from_admissions(
                "oracle",
                workload,
                state.admissions,
                state.peak_memory_bytes,
                peak_shared_prefix_credit_bytes=(
                    state.peak_shared_prefix_credit_bytes
                ),
            )
            if _oracle_objective(result) < _oracle_objective(best):
                best = result
                best_from_search = True
            continue

        if state.prefill_busy_until > state.now_ms:
            wake_times = [state.prefill_busy_until]
            wake_times.extend(
                requests_by_id[request_id].arrival_ms
                for request_id in state.remaining_ids
                if requests_by_id[request_id].arrival_ms > state.now_ms
            )
            wake_times.extend(
                item.admission.finish_ms
                for item in active
                if item.admission.finish_ms > state.now_ms
            )
            stack.append(replace(state, now_ms=min(wake_times)))
            continue

        pending = sorted(
            (
                requests_by_id[request_id]
                for request_id in state.remaining_ids
                if requests_by_id[request_id].arrival_ms <= state.now_ms
            ),
            key=lambda item: (item.arrival_ms, item.request_id),
        )
        children: list[_OracleState] = []
        seen_symmetry: set[tuple[object, ...]] = set()
        for request in pending:
            symmetry = _oracle_symmetry_key(request)
            if symmetry in seen_symmetry:
                continue
            seen_symmetry.add(symmetry)
            for mode, reason in _feasible_modes(request, active, config):
                children.append(
                    _admission_child(
                        state,
                        request,
                        mode,
                        f"oracle:{reason}",
                        config,
                    )
                )

        # Explore useful work before spending the expansion budget on an idle
        # branch.  Exhaustive runs still compare both, while truncated runs
        # retain a concrete admission policy.
        children.sort(
            key=lambda item: (
                -item.admissions[-1].resident_kv_bytes,
                item.admissions[-1].request_id,
                item.admissions[-1].mode.value,
            )
        )
        idle = _oracle_idle_child(state, requests_by_id)
        if idle is not None:
            stack.append(idle)
        stack.extend(reversed(children))

    return replace(
        best,
        policy="oracle",
        search_expansions=expansions,
        search_truncated=truncated,
        fallback_policy=None if best_from_search else incumbent.policy,
        search_preparation_rollouts=online.search_expansions,
    )


def simulate(
    requests: Iterable[RequestProfile],
    config: PlannerConfig,
    policy: str = "future",
) -> PlanResult:
    """Run a deterministic event simulation for one admission policy.

    Requests reserve their full resident KV or bounded HiSparse hot set while
    prefilling and decoding.  A one-slot prefill engine is modeled separately,
    while already-prefilled decode work can overlap.  These assumptions are a
    policy harness, not measured SGLang timing.
    """

    workload = tuple(requests)
    _validate_workload(workload, config, policy)
    if not workload:
        return PlanResult(policy, (), 0, 0, 0, 0.0, 0, 0, 0)
    if policy == "oracle":
        return _simulate_oracle(workload, config)
    if policy == "online":
        return _simulate_online(workload, config)
    if policy == "batched":
        return _simulate_batched(workload, config)
    return _simulate_greedy(workload, config, policy)


def _validate_workload(
    workload: Sequence[RequestProfile],
    config: PlannerConfig,
    policy: str,
) -> None:
    if policy not in POLICIES:
        raise ValueError(f"unknown planning policy: {policy!r}")
    ids = [item.request_id for item in workload]
    if len(set(ids)) != len(ids):
        raise ValueError("request IDs must be unique")
    prefix_lengths: dict[str, set[int]] = {}
    for item in workload:
        if item.prefix_group is not None:
            prefix_lengths.setdefault(item.prefix_group, set()).add(
                item.shared_prefix_tokens
            )
        sparse = _feasible_mode(item, (), config, preferred=ResidencyMode.HI_SPARSE)
        if sparse is None:
            raise ValueError(
                f"{item.request_id}: no residency mode fits the configured budget"
            )
    inconsistent = next(
        (group for group, lengths in prefix_lengths.items() if len(lengths) > 1),
        None,
    )
    if inconsistent is not None:
        raise ValueError(
            f"prefix_group {inconsistent!r} has inconsistent shared prefix lengths"
        )


def _initial_state(workload: Sequence[RequestProfile]) -> _OracleState:
    return _OracleState(
        now_ms=min(item.arrival_ms for item in workload),
        prefill_busy_until=-1,
        remaining_ids=tuple(item.request_id for item in workload),
        active=(),
        admissions=(),
        peak_memory_bytes=0,
        peak_shared_prefix_credit_bytes=0,
    )


def _simulate_heuristic_from_state(
    workload: tuple[RequestProfile, ...],
    config: PlannerConfig,
    policy: str,
    state: _OracleState,
) -> PlanResult:
    """Complete a rollout from ``state`` with a deterministic heuristic."""

    if policy not in ("fifo", "longest", "mode-aware", "future"):
        raise ValueError(f"unknown greedy planning policy: {policy!r}")

    requests_by_id = {item.request_id: item for item in workload}
    while state.remaining_ids or state.active:
        active = _normalize_active(state.active, state.now_ms)
        state = replace(state, active=active)
        if not state.remaining_ids and not active:
            break
        if not state.remaining_ids:
            state = replace(
                state,
                now_ms=min(item.admission.finish_ms for item in active),
            )
            continue

        if state.prefill_busy_until > state.now_ms:
            candidates = [state.prefill_busy_until]
            candidates.extend(
                requests_by_id[request_id].arrival_ms
                for request_id in state.remaining_ids
                if requests_by_id[request_id].arrival_ms > state.now_ms
            )
            candidates.extend(
                item.admission.finish_ms
                for item in active
                if item.admission.finish_ms > state.now_ms
            )
            state = replace(state, now_ms=min(candidates))
            continue

        unstarted = {
            request_id: requests_by_id[request_id]
            for request_id in state.remaining_ids
        }
        selected = _select(policy, unstarted, active, state.now_ms, config)
        if selected is not None:
            request, mode, reason = selected
            workspace = request.prompt_tokens * config.prefill_workspace_rate(
                request.kind
            )
            hold = _hold_bytes(request, mode, config)
            duration = _duration_ms(request, mode, config, active)
            prefill = _prefill_duration(request, active, config)
            if prefill < _ceil_duration(
                request.prompt_tokens, config.prefill_rate(request.kind)
            ):
                reason = f"{reason}+active-prefix-reuse"
            admission = Admission(
                request_id=request.request_id,
                arrival_ms=request.arrival_ms,
                start_ms=state.now_ms,
                finish_ms=state.now_ms + duration,
                mode=mode,
                kind=request.kind,
                resident_kv_bytes=hold,
                prefill_workspace_bytes=workspace,
                reason=reason,
                decode_batch_size=_decode_batch_size(active, (), config),
            )
            aggregate, shared_credit = _placement_memory(
                (
                    *_active_placements(active),
                    (request, mode),
                ),
                config,
            )
            prefill_busy_until = state.now_ms + prefill
            state = replace(
                state,
                now_ms=prefill_busy_until,
                prefill_busy_until=prefill_busy_until,
                remaining_ids=tuple(
                    request_id
                    for request_id in state.remaining_ids
                    if request_id != request.request_id
                ),
                active=(*active, _Active(request, admission)),
                admissions=(*state.admissions, admission),
                peak_memory_bytes=max(
                    state.peak_memory_bytes, aggregate + workspace
                ),
                peak_shared_prefix_credit_bytes=max(
                    state.peak_shared_prefix_credit_bytes, shared_credit
                ),
            )
            continue

        idle = _oracle_idle_child(state, requests_by_id)
        if idle is None:
            blocked = sorted(
                item.request_id
                for item in unstarted.values()
                if item.arrival_ms <= state.now_ms
            )
            raise RuntimeError(
                f"planning deadlock at {state.now_ms} ms for {blocked}"
            )
        state = idle

    if len(state.admissions) != len(workload):
        raise RuntimeError("planner did not admit every request")
    return _result_from_admissions(
        policy,
        workload,
        state.admissions,
        state.peak_memory_bytes,
        peak_shared_prefix_credit_bytes=state.peak_shared_prefix_credit_bytes,
    )


def _simulate_greedy(
    workload: tuple[RequestProfile, ...],
    config: PlannerConfig,
    policy: str,
) -> PlanResult:
    """Run the existing deterministic event simulation for online policies."""

    return _simulate_heuristic_from_state(
        workload,
        config,
        policy,
        _initial_state(workload),
    )


def _simulate_online(
    workload: tuple[RequestProfile, ...],
    config: PlannerConfig,
) -> PlanResult:
    """Use receding-horizon rollouts while committing one action at a time.

    Each candidate first action is completed by the deployable ``future``
    heuristic. Only that first action is committed; the next scheduler boundary
    searches again. This captures short-horizon interactions without claiming
    the exhaustive ``oracle``'s lookahead or running an unbounded search on the
    serving hot path.
    """

    requests_by_id = {item.request_id: item for item in workload}
    state = _initial_state(workload)
    rollouts = 0

    while state.remaining_ids or state.active:
        active = _normalize_active(state.active, state.now_ms)
        state = replace(state, active=active)
        if not state.remaining_ids:
            if not active:
                break
            state = replace(
                state,
                now_ms=min(item.admission.finish_ms for item in active),
            )
            continue

        if state.prefill_busy_until > state.now_ms:
            candidates = [state.prefill_busy_until]
            candidates.extend(
                requests_by_id[request_id].arrival_ms
                for request_id in state.remaining_ids
                if requests_by_id[request_id].arrival_ms > state.now_ms
            )
            candidates.extend(
                item.admission.finish_ms
                for item in active
                if item.admission.finish_ms > state.now_ms
            )
            state = replace(state, now_ms=min(candidates))
            continue

        pending = sorted(
            (
                requests_by_id[request_id]
                for request_id in state.remaining_ids
                if requests_by_id[request_id].arrival_ms <= state.now_ms
            ),
            key=lambda item: (item.arrival_ms, item.request_id),
        )
        if not pending:
            idle = _oracle_idle_child(state, requests_by_id)
            if idle is None:
                raise RuntimeError(f"online planner deadlock at {state.now_ms} ms")
            state = idle
            continue

        children: list[_OracleState] = []
        seen_symmetry: set[tuple[object, ...]] = set()
        for request in pending:
            symmetry = _oracle_symmetry_key(request)
            if symmetry in seen_symmetry:
                continue
            seen_symmetry.add(symmetry)
            for mode, reason in _feasible_modes(request, active, config):
                children.append(
                    _admission_child(
                        state,
                        request,
                        mode,
                        f"online:{reason}",
                        config,
                    )
                )
        idle = _oracle_idle_child(state, requests_by_id)
        if idle is not None:
            children.append(idle)
        if not children:
            raise RuntimeError(f"online planner found no action at {state.now_ms} ms")

        if rollouts + len(children) > config.online_max_rollouts:
            suffix = _simulate_heuristic_from_state(
                workload, config, "future", state
            )
            return replace(
                suffix,
                policy="online",
                search_expansions=rollouts,
                search_truncated=True,
                fallback_policy=(
                    "online_prefix_then_future"
                    if state.admissions
                    else "future"
                ),
            )

        best_child = None
        best_result = None
        for child in children:
            result = _simulate_heuristic_from_state(
                workload, config, "future", child
            )
            rollouts += 1
            if best_result is None or _oracle_objective(result) < _oracle_objective(
                best_result
            ):
                best_child = child
                best_result = result
        assert best_child is not None
        state = best_child

    return _result_from_admissions(
        "online",
        workload,
        state.admissions,
        state.peak_memory_bytes,
        peak_shared_prefix_credit_bytes=state.peak_shared_prefix_credit_bytes,
        search_expansions=rollouts,
    )


def plan_rollout(
    requests: Iterable[RequestProfile],
    config: PlannerConfig,
    policy: str = "future",
    *,
    simulation: PlanResult | None = None,
    runtime_capabilities: RuntimeCapabilities | None = None,
    cohort_namespace: str | None = None,
) -> RolloutPlan:
    """Convert a simulated policy into delayed submissions and runtime hints.

    The rollout controller should submit request ``i`` at ``submit_at_ms``.
    Its attached ``elastic.future`` action emits the selected residency mode,
    expected output length, weight, type and scheduler-relative deadline.  This
    keeps the oracle-to-runtime boundary explicit and leaves memory admission,
    routing and fallback inside SGLang.
    """

    workload = tuple(requests)
    if policy not in POLICIES:
        raise ValueError(f"unknown planning policy: {policy!r}")
    if simulation is None:
        simulation = simulate(workload, config, policy)
    elif simulation.policy != policy:
        raise ValueError(
            f"simulation policy {simulation.policy!r} does not match {policy!r}"
        )
    capabilities = runtime_capabilities or RuntimeCapabilities()
    validate_runtime_capabilities(config, capabilities)
    workload_ids = {item.request_id for item in workload}
    admission_ids = {item.request_id for item in simulation.admissions}
    if workload_ids != admission_ids or len(workload_ids) != len(simulation.admissions):
        raise ValueError("simulation requests do not match the rollout workload")
    requests_by_id = {item.request_id: item for item in workload}
    namespace = cohort_namespace or secrets.token_hex(6)
    if (
        not isinstance(namespace, str)
        or not namespace
        or len(namespace) > 32
        or any(
            not (character.isalnum() or character in "-_")
            for character in namespace
        )
    ):
        raise ValueError("cohort_namespace must be a short alphanumeric identifier")
    runtime_cohort_ids = {
        cohort_id: f"{cohort_id}-{namespace}"
        for cohort_id in {
            item.cohort_id
            for item in simulation.admissions
            if item.cohort_id is not None
        }
    }
    submissions = []
    for admission in simulation.admissions:
        request = requests_by_id[admission.request_id]
        hint: dict[str, object] = {
            "release_after_ms": 0,
            "expected_output_tokens": request.output_tokens,
            "weight": request.priority,
            "kind": request.kind,
            "residence": (
                "resident"
                if admission.mode is ResidencyMode.RESIDENT
                else "host_backed"
            ),
            "prefill_tokens_per_s": (
                config.prefill_rate_for_batch(
                    request.kind, admission.prefill_batch_size
                )
                * 1000.0
            ),
            "decode_tokens_per_s": (
                config.decode_rate_for_batch(
                    request.kind,
                    admission.mode,
                    admission.decode_batch_size,
                )
                * 1000.0
            ),
        }
        if request.deadline_ms is not None:
            hint["deadline_after_ms"] = max(
                0, request.deadline_ms - admission.start_ms
            )
        if request.prefix_group is not None:
            hint["prefix_group"] = request.prefix_group
            hint["shared_prefix_tokens"] = request.shared_prefix_tokens
        submissions.append(
            RolloutSubmission(
                request_id=request.request_id,
                planned_arrival_ms=admission.arrival_ms,
                submit_at_ms=admission.start_ms,
                estimated_finish_ms=admission.finish_ms,
                kind=request.kind,
                prompt_tokens=request.prompt_tokens,
                output_tokens=request.output_tokens,
                hint=hint,
                deadline_ms=request.deadline_ms,
                decode_batch_size=admission.decode_batch_size,
                prefill_batch_size=admission.prefill_batch_size,
                cohort_id=(
                    runtime_cohort_ids.get(admission.cohort_id)
                    if admission.cohort_id is not None
                    else None
                ),
            )
        )
    return RolloutPlan(
        simulation,
        tuple(submissions),
        config,
        capabilities,
        namespace,
    )


def _load_workload(
    request_path: str | Path,
) -> tuple[PlannerConfig, tuple[RequestProfile, ...]]:
    payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
    config = PlannerConfig(**payload["config"])
    workload = tuple(RequestProfile(**item) for item in payload["requests"])
    return config, workload


def _compare_workload(
    config: PlannerConfig,
    workload: Sequence[RequestProfile],
    policies: Sequence[str],
) -> dict[str, PlanResult]:
    requested = tuple(dict.fromkeys(policies))
    unknown = tuple(policy for policy in requested if policy not in POLICIES)
    if unknown:
        raise ValueError(f"unknown planning policy: {unknown[0]!r}")
    return {policy: simulate(workload, config, policy) for policy in requested}


def _comparison_payload(
    simulations: Mapping[str, PlanResult],
) -> dict[str, dict[str, object]]:
    oracle = simulations.get("oracle")
    payload = {policy: result.as_dict() for policy, result in simulations.items()}
    if oracle is None:
        return payload
    for policy, result in simulations.items():
        completion_gap = result.weighted_completion_ms - oracle.weighted_completion_ms
        makespan_gap = result.makespan_ms - oracle.makespan_ms
        peak_gap = result.peak_memory_bytes - oracle.peak_memory_bytes
        payload[policy]["oracle_gap"] = {
            "weighted_completion_abs_ms": completion_gap,
            "weighted_completion_pct": (
                None
                if oracle.weighted_completion_ms == 0
                else 100.0 * completion_gap / oracle.weighted_completion_ms
            ),
            "makespan_abs_ms": makespan_gap,
            "makespan_pct": (
                None
                if oracle.makespan_ms == 0
                else 100.0 * makespan_gap / oracle.makespan_ms
            ),
            "deadline_miss_delta": (
                result.deadline_misses - oracle.deadline_misses
            ),
            "peak_memory_delta_bytes": peak_gap,
            "oracle_exact": oracle.search_exact,
            "oracle_fallback_policy": oracle.fallback_policy,
            "is_oracle": policy == "oracle",
        }
    return payload


def compare(
    request_path: str | Path,
    policies: Sequence[str] = POLICIES,
) -> dict[str, dict[str, object]]:
    """Load ``{config, requests}`` JSON and simulate selected policies."""

    config, workload = _load_workload(request_path)
    return _comparison_payload(_compare_workload(config, workload, policies))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare deterministic future-aware admission policies"
    )
    parser.add_argument("workload", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--emit-policy", choices=POLICIES)
    parser.add_argument(
        "--radix-cache-enabled",
        action="store_true",
        help=(
            "declare that the rollout runtime has SGLang radix cache enabled; "
            "required by active_prefix_prefill_reuse"
        ),
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES,
        help="simulate only these policies (default: all built-in policies)",
    )
    args = parser.parse_args(argv)

    config, workload = _load_workload(args.workload)
    policies = tuple(args.policies or POLICIES)
    if args.emit_policy is not None and args.emit_policy not in policies:
        policies = (*policies, args.emit_policy)
    simulations = _compare_workload(config, workload, policies)
    payload: dict[str, object] = _comparison_payload(simulations)
    if args.emit_policy is not None:
        payload["rollout"] = plan_rollout(
            workload,
            config,
            args.emit_policy,
            simulation=simulations[args.emit_policy],
            runtime_capabilities=RuntimeCapabilities(
                radix_cache_enabled=args.radix_cache_enabled
            ),
        ).as_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

