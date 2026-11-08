"""Deterministic admission accounting for future GPU KV growth.

The broker is intentionally CPU-only.  It does not allocate CUDA memory or
change SGLang's native allocator.  A caller supplies the currently available
physical KV bytes and a prediction for each resident request.  The broker then
keeps the pending growth reservations in one ledger so later requests cannot
spend the same headroom again.

The API is split into ``assess`` and ``reserve`` so experiments can first run
the policy in dry-run mode.  ``try_admit`` makes the decision and reservation
one operation for enforced mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable


@dataclass(frozen=True, slots=True)
class BrokerDecision:
    admit: bool
    reason: str
    required_bytes: int
    reserved_bytes: int


def page_aligned_growth_slots(
    *, current_tokens: int, target_tokens: int, page_size: int
) -> int:
    """Return the token slots a paged allocator adds between two lengths."""

    for value, name in (
        (current_tokens, "current_tokens"),
        (target_tokens, "target_tokens"),
        (page_size, "page_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if page_size < 1:
        raise ValueError("page_size must be positive")
    current_slots = page_aligned_slots(current_tokens, page_size)
    target_slots = page_aligned_slots(max(current_tokens, target_tokens), page_size)
    return target_slots - current_slots


def page_aligned_slots(tokens: int, page_size: int) -> int:
    """Round a token count up to the allocator's page count."""

    for value, name in ((tokens, "tokens"), (page_size, "page_size")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if page_size < 1:
        raise ValueError("page_size must be positive")
    return -(-tokens // page_size) * page_size


def remaining_output_tokens(
    *, predicted_total: int, current_tokens: int, baseline_tokens: int
) -> int:
    """Discount output prediction already generated since staging."""

    for value, name in (
        (predicted_total, "predicted_total"),
        (current_tokens, "current_tokens"),
        (baseline_tokens, "baseline_tokens"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    generated = max(0, current_tokens - baseline_tokens)
    return max(0, predicted_total - generated)


class ResidencyBroker:
    """Track physical headroom promised to requests still generating tokens."""

    def __init__(self) -> None:
        self._reservations: dict[Hashable, int] = {}
        self._last_lengths: dict[Hashable, int] = {}
        self._stats = {
            "assessments": 0,
            "would_admit": 0,
            "would_decline": 0,
            "reservations": 0,
            "duplicate_reservations": 0,
            "consumed_bytes": 0,
            "released_bytes": 0,
            "decline_reasons": {},
        }

    @staticmethod
    def _nonnegative(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return value

    def assess(
        self,
        key: Hashable,
        *,
        available_bytes: int,
        fixed_reserve_bytes: int,
        predicted_growth_bytes: int,
    ) -> BrokerDecision:
        """Assess one claim without mutating reservations.

        ``available_bytes`` already reflects physical memory consumed by live
        KV.  ``_reservations`` therefore contains only growth that has been
        promised but not yet observed by the allocator.  The returned
        ``reserved_bytes`` is the total including the candidate's prediction.
        """

        available = self._nonnegative(available_bytes, "available_bytes")
        fixed = self._nonnegative(fixed_reserve_bytes, "fixed_reserve_bytes")
        predicted = self._nonnegative(predicted_growth_bytes, "predicted_growth_bytes")
        self._stats["assessments"] += 1
        if key in self._reservations:
            self._stats["would_decline"] += 1
            self._record_decline("duplicate-request")
            return BrokerDecision(False, "duplicate-request", fixed + predicted, self.reserved_bytes)

        promised = sum(self._reservations.values())
        required = fixed + promised + predicted
        self._stats["would_decline" if available < required else "would_admit"] += 1
        if available < required:
            self._record_decline("insufficient-growth-headroom")
            return BrokerDecision(False, "insufficient-growth-headroom", required, promised)
        return BrokerDecision(True, "growth-headroom-available", required, promised + predicted)

    def reserve(self, key: Hashable, *, predicted_growth_bytes: int) -> None:
        """Commit a growth prediction after a successful assessment."""

        predicted = self._nonnegative(predicted_growth_bytes, "predicted_growth_bytes")
        if key in self._reservations:
            self._stats["duplicate_reservations"] += 1
            raise ValueError("duplicate residency reservation for request")
        self._reservations[key] = predicted
        self._stats["reservations"] += 1

    def try_admit(
        self,
        key: Hashable,
        *,
        available_bytes: int,
        fixed_reserve_bytes: int,
        predicted_growth_bytes: int,
    ) -> BrokerDecision:
        """Assess and reserve atomically when the claim fits the ledger."""

        decision = self.assess(
            key,
            available_bytes=available_bytes,
            fixed_reserve_bytes=fixed_reserve_bytes,
            predicted_growth_bytes=predicted_growth_bytes,
        )
        if decision.admit:
            self.reserve(key, predicted_growth_bytes=predicted_growth_bytes)
        return decision

    def consume(self, key: Hashable, *, bytes_consumed: int) -> int:
        """Reduce a reservation as the allocator makes its promise real."""

        consumed = self._nonnegative(bytes_consumed, "bytes_consumed")
        remaining = self._reservations.get(key, 0)
        applied = min(remaining, consumed)
        if key in self._reservations:
            self._reservations[key] = remaining - applied
            self._stats["consumed_bytes"] += applied
        return applied

    def release(self, key: Hashable) -> int:
        """Release every unconsumed byte reserved for a completed request."""

        remaining = self._reservations.pop(key, 0)
        self._last_lengths.pop(key, None)
        self._stats["released_bytes"] += remaining
        return remaining

    def set_last_length(self, key: Hashable, length: int) -> None:
        """Record the token length already reflected in the KV allocator."""

        self._last_lengths[key] = self._nonnegative(length, "length")

    def consume_new_tokens(
        self,
        key: Hashable,
        *,
        length: int,
        bytes_per_token: int,
    ) -> int:
        """Consume growth between this decode length and the previous one."""

        current = self._nonnegative(length, "length")
        row_bytes = self._nonnegative(bytes_per_token, "bytes_per_token")
        previous = self._last_lengths.get(key)
        if previous is None:
            self._last_lengths[key] = current
            return 0
        delta = max(0, current - previous)
        self._last_lengths[key] = max(previous, current)
        return self.consume(key, bytes_consumed=delta * row_bytes)

    def consume_new_page_slots(
        self,
        key: Hashable,
        *,
        length: int,
        bytes_per_token: int,
        page_size: int,
    ) -> int:
        """Consume page-aligned growth since the previously seen length."""

        current = self._nonnegative(length, "length")
        row_bytes = self._nonnegative(bytes_per_token, "bytes_per_token")
        page = self._nonnegative(page_size, "page_size")
        if page < 1:
            raise ValueError("page_size must be positive")
        previous = self._last_lengths.get(key)
        if previous is None:
            self._last_lengths[key] = current
            return 0
        slots = page_aligned_growth_slots(
            current_tokens=previous, target_tokens=current, page_size=page
        )
        self._last_lengths[key] = max(previous, current)
        return self.consume(key, bytes_consumed=slots * row_bytes)

    def _record_decline(self, reason: str) -> None:
        reasons = self._stats["decline_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1

    @property
    def reserved_bytes(self) -> int:
        return sum(self._reservations.values())

    def snapshot(self) -> dict[str, object]:
        result = dict(self._stats)
        result["active_reservations"] = len(self._reservations)
        result["outstanding_bytes"] = self.reserved_bytes
        result["decline_reasons"] = dict(self._stats["decline_reasons"])
        return result
