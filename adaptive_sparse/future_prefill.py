"""Bounded prefill reservations derived from known-future request hints.

This module is deliberately CPU-only.  It estimates how much headroom a
short horizon of waiting requests may need, but never allocates memory and
never bypasses SGLang's native ``PrefillAdder``.  The scheduler integration
can therefore apply the result as advisory headroom while all native memory,
request-slot, grammar and allocator checks remain authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from .sglang_future_hook import _extract_hint, _prompt_tokens, _request_age_ms


PREFILL_RESERVATION_MODES = ("off", "dry_run", "enforce")


@dataclass(frozen=True, slots=True)
class FuturePrefillConfig:
    """Limits for a bounded, request-aggregate prefill reservation."""

    mode: str = "off"
    horizon_ms: int = 2_000
    max_reserve_bytes: int = 256 << 20
    activation_bytes_per_token: int = 0
    kind_activation_bytes_per_token: tuple[tuple[str, int], ...] = ()
    auto_resident_output_tokens: int = 128
    sparse_hot_tokens: int = 1_024

    def __post_init__(self) -> None:
        if self.mode not in PREFILL_RESERVATION_MODES:
            raise ValueError(f"unknown future prefill mode: {self.mode!r}")
        for name in (
            "horizon_ms",
            "max_reserve_bytes",
            "activation_bytes_per_token",
            "auto_resident_output_tokens",
            "sparse_hot_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.auto_resident_output_tokens < 1:
            raise ValueError("auto_resident_output_tokens must be positive")
        if self.sparse_hot_tokens < 1:
            raise ValueError("sparse_hot_tokens must be positive")
        if not isinstance(self.kind_activation_bytes_per_token, tuple):
            raise ValueError("kind activation overrides must be a tuple of pairs")
        seen = set()
        for kind, value in self.kind_activation_bytes_per_token:
            if not isinstance(kind, str) or not kind or len(kind) > 64:
                raise ValueError(
                    "kind activation keys must be short non-empty strings"
                )
            if kind in seen:
                raise ValueError(f"duplicate kind activation override: {kind!r}")
            seen.add(kind)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "kind activation overrides must be nonnegative integers"
                )


@dataclass(frozen=True, slots=True)
class FuturePrefillEntry:
    """One request's bounded contribution to the aggregate reservation."""

    request_id: str
    kind: str
    residence: str
    prompt_tokens: int
    expected_output_tokens: int
    kv_bytes: int
    activation_bytes: int
    release_in_ms: float
    native_order: int

    @property
    def total_bytes(self) -> int:
        return self.kv_bytes + self.activation_bytes


@dataclass(frozen=True, slots=True)
class FuturePrefillPlan:
    """Serializable result of one deterministic planning pass."""

    mode: str
    entries: tuple[FuturePrefillEntry, ...]
    waiting_count: int = 0
    deferred_count: int = 0
    missing_hint_count: int = 0
    invalid_hint_count: int = 0
    unestimated_count: int = 0
    capped_count: int = 0

    @property
    def kv_bytes(self) -> int:
        return sum(entry.kv_bytes for entry in self.entries)

    @property
    def activation_bytes(self) -> int:
        return sum(entry.activation_bytes for entry in self.entries)

    @property
    def reserve_bytes(self) -> int:
        return self.kv_bytes + self.activation_bytes

    @property
    def selected_count(self) -> int:
        return len(self.entries)

    @property
    def signature(self) -> tuple[Any, ...]:
        """Small value suitable for a TP ``all_gather_object`` check."""

        return (
            self.mode,
            self.reserve_bytes,
            tuple(
                (
                    entry.request_id,
                    entry.kind,
                    entry.kv_bytes,
                    entry.activation_bytes,
                    entry.residence,
                )
                for entry in self.entries
            ),
        )

    @property
    def active(self) -> bool:
        return self.mode != "off"


def empty_plan(mode: str = "off") -> FuturePrefillPlan:
    return FuturePrefillPlan(mode=mode, entries=())


def effective_prefill_headroom(
    plan: FuturePrefillPlan, base_headroom: int, mode: str
) -> int:
    """Combine KV backing demand with one activation headroom estimate.

    A configured fixed headroom and the request-level activation estimate both
    describe scratch space, so take their maximum.  Predicted KV bytes are a
    separate allocation and are always added.  Dry-run mode never changes the
    effective headroom.
    """

    if (
        isinstance(base_headroom, bool)
        or not isinstance(base_headroom, int)
        or base_headroom < 0
    ):
        raise ValueError("base_headroom must be a nonnegative integer")
    if mode != "enforce":
        return 0
    return plan.kv_bytes + max(base_headroom, plan.activation_bytes)


def reconcile_tp_plan(
    plan: FuturePrefillPlan, peer_signatures: Iterable[Any]
) -> tuple[FuturePrefillPlan, bool]:
    """Fall back on every rank when TP reservation plans disagree."""

    signatures = list(peer_signatures)
    agrees = bool(signatures) and all(
        signature == plan.signature for signature in signatures
    )
    return (plan, False) if agrees else (empty_plan(plan.mode), True)


def _request_id(request: Any) -> str:
    for name in ("rid", "request_id"):
        value = getattr(request, name, None)
        if value is not None and str(value):
            return str(value)
    raise ValueError("request id is unavailable for a TP-stable reservation")


def _residence(hint: Any, expected: int, config: FuturePrefillConfig) -> str:
    if hint.residence == "resident":
        return "resident"
    if hint.residence == "host_backed":
        return "host_backed"
    if hint.residence == "auto":
        return (
            "resident"
            if expected >= config.auto_resident_output_tokens
            else "host_backed"
        )
    # SGLang's existing request-residency hook keeps long requests resident
    # when no explicit residence preference is supplied.  Reserve the full
    # predicted footprint to match that default instead of optimistically
    # assuming the bounded HiSparse path.
    return "resident"


def _activation_bytes_per_token(hint: Any, config: FuturePrefillConfig) -> int:
    overrides = dict(config.kind_activation_bytes_per_token)
    return overrides.get(hint.kind, config.activation_bytes_per_token)


def plan_future_prefill(
    waiting_queue: Iterable[Any],
    *,
    now_s: float,
    kv_bytes_per_token: int,
    config: FuturePrefillConfig,
) -> FuturePrefillPlan:
    """Return a bounded reservation plan without mutating the waiting queue.

    ``release_after_ms`` is measured from scheduler receive time by the
    existing future-hint transport.  Requests outside the horizon, invalid
    hints, and requests whose complete footprint cannot be estimated are
    skipped.  A request that does not fit the remaining byte cap is also
    skipped rather than represented as a misleading partial row.
    """

    queue = list(waiting_queue)
    if config.mode == "off":
        return empty_plan("off")
    if (
        isinstance(kv_bytes_per_token, bool)
        or not isinstance(kv_bytes_per_token, int)
        or kv_bytes_per_token <= 0
    ):
        raise ValueError("kv_bytes_per_token must be a positive integer")
    if not math.isfinite(now_s):
        raise ValueError("now_s must be finite")

    selected: list[FuturePrefillEntry] = []
    reserve = 0
    deferred = missing = invalid = unestimated = capped = 0
    for native_order, request in enumerate(queue):
        try:
            hint = _extract_hint(request)
            if hint is None:
                missing += 1
                continue
            age_ms = _request_age_ms(request, now_s)
            release_in_ms = hint.release_after_ms - age_ms
            if release_in_ms > config.horizon_ms:
                deferred += 1
                continue
            expected = hint.expected_output_tokens
            if expected is None:
                # An explicit host-backed request has a bounded hot-set even
                # when its output length is unknown.  Other modes need the
                # output prediction for a complete-KV estimate.
                if hint.residence != "host_backed":
                    unestimated += 1
                    continue
                expected = 0
            prompt = _prompt_tokens(request)
            request_id = _request_id(request)
        except (AttributeError, TypeError, ValueError):
            invalid += 1
            continue

        residence = _residence(hint, expected, config)
        sequence_tokens = prompt + expected
        kv_tokens = (
            sequence_tokens
            if residence == "resident"
            else min(sequence_tokens, config.sparse_hot_tokens)
        )
        kv_bytes = kv_tokens * kv_bytes_per_token
        activation_bytes = prompt * _activation_bytes_per_token(hint, config)
        total = kv_bytes + activation_bytes
        if reserve + total > config.max_reserve_bytes:
            capped += 1
            continue
        reserve += total
        selected.append(
            FuturePrefillEntry(
                request_id=request_id,
                kind=hint.kind,
                residence=residence,
                prompt_tokens=prompt,
                expected_output_tokens=expected,
                kv_bytes=kv_bytes,
                activation_bytes=activation_bytes,
                release_in_ms=release_in_ms,
                native_order=native_order,
            )
        )

    return FuturePrefillPlan(
        mode=config.mode,
        entries=tuple(selected),
        waiting_count=len(queue),
        deferred_count=deferred,
        missing_hint_count=missing,
        invalid_hint_count=invalid,
        unestimated_count=unestimated,
        capped_count=capped,
    )
