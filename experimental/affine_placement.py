"""CPU simulation for request-affine KV placement geometries.

This module compares a fixed-stride affine layout with an exact-length packed
affine layout.  Both preserve ``base(request) + token_offset * row_bytes``
addressing.  The packed layout removes request-aligned gaps so adjacent short
requests can share VMM chunks.

The simulator models mapping geometry only.  It does not move KV bytes, launch
attention, or measure GPU mapping latency.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AffineRequest:
    """Known request geometry used by the controlled simulator."""

    request_id: str
    arrival_ms: float
    prompt_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        _finite_nonnegative(self.arrival_ms, "arrival_ms")
        _positive_int(self.prompt_tokens, "prompt_tokens")
        _positive_int(self.output_tokens, "output_tokens")

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class AffineConfig:
    """Geometry and service assumptions for one comparison.

    ``arena_bytes`` is the amount of committed backing made available to one
    rank.  Virtual reservations are confined to the same arena so visiting a
    chunk never requires backing outside the measured capacity.
    """

    arena_bytes: int
    chunk_bytes: int
    row_bytes: int
    max_active_requests: int
    prefill_tokens_per_ms: float = 8.0
    decode_tokens_per_ms: float = 20.0

    def __post_init__(self) -> None:
        for name in ("arena_bytes", "chunk_bytes", "row_bytes"):
            _positive_int(getattr(self, name), name)
        _positive_int(self.max_active_requests, "max_active_requests")
        for name in ("prefill_tokens_per_ms", "decode_tokens_per_ms"):
            _positive_finite(getattr(self, name), name)
        if self.arena_bytes % self.chunk_bytes:
            raise ValueError("arena_bytes must be chunk aligned")


@dataclass(frozen=True, slots=True)
class Placement:
    """A stable affine mapping for one request."""

    request_id: str
    start_byte: int
    reserved_bytes: int
    total_tokens: int
    row_bytes: int

    def token_offset(self, token_position: int) -> int:
        if not 0 <= token_position < self.total_tokens:
            raise ValueError("token_position outside the reserved sequence")
        return self.start_byte + token_position * self.row_bytes


class AffineArena:
    """Non-overlapping affine reservations with whole-chunk VMM mapping."""

    def __init__(
        self,
        arena_bytes: int,
        chunk_bytes: int,
        *,
        slot_bytes: int | None = None,
    ) -> None:
        _positive_int(arena_bytes, "arena_bytes")
        _positive_int(chunk_bytes, "chunk_bytes")
        if arena_bytes % chunk_bytes:
            raise ValueError("arena_bytes must be chunk aligned")
        if slot_bytes is not None:
            _positive_int(slot_bytes, "slot_bytes")
            if slot_bytes % chunk_bytes:
                raise ValueError("slot_bytes must be chunk aligned")
            if slot_bytes > arena_bytes:
                raise ValueError("slot_bytes exceeds arena_bytes")
        self.arena_bytes = arena_bytes
        self.chunk_bytes = chunk_bytes
        self.slot_bytes = slot_bytes
        self._free = [(0, arena_bytes)]
        self._placements: dict[str, Placement] = {}

    @property
    def free_bytes(self) -> int:
        return sum(size for _, size in self._free)

    @property
    def free_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._free)

    def reserve(
        self,
        request_id: str,
        total_tokens: int,
        row_bytes: int,
    ) -> Placement | None:
        """Place a full, fixed request interval using best fit.

        Packed mode rounds only the physical byte count.  Its virtual base can
        have arbitrary chunk phase.  Fixed-stride mode rounds every request to
        ``slot_bytes``.  Reserving the full sequence keeps the affine base
        stable for the request's lifetime; only touched prefix chunks are
        mapped by the caller.
        """

        if request_id in self._placements:
            raise ValueError(f"request already placed: {request_id!r}")
        _positive_int(total_tokens, "total_tokens")
        _positive_int(row_bytes, "row_bytes")
        exact_bytes = total_tokens * row_bytes
        reserved = exact_bytes if self.slot_bytes is None else self.slot_bytes
        if exact_bytes > reserved:
            raise ValueError("request sequence exceeds the configured affine slot")
        if reserved > self.arena_bytes:
            return None

        candidates = [
            (size, index, start)
            for index, (start, size) in enumerate(self._free)
            if size >= reserved
        ]
        if not candidates:
            return None
        _size, index, start = min(candidates)
        old_start, old_size = self._free[index]
        replacement: list[tuple[int, int]] = []
        if start > old_start:
            replacement.append((old_start, start - old_start))
        if start + reserved < old_start + old_size:
            replacement.append(
                (start + reserved, old_start + old_size - start - reserved)
            )
        self._free[index : index + 1] = replacement

        placement = Placement(request_id, start, reserved, total_tokens, row_bytes)
        self._placements[request_id] = placement
        return placement

    def release(self, request_id: str) -> Placement:
        try:
            placement = self._placements.pop(request_id)
        except KeyError as exc:
            raise ValueError(f"request is not placed: {request_id!r}") from exc
        self._free.append((placement.start_byte, placement.reserved_bytes))
        self._free.sort()
        merged: list[tuple[int, int]] = []
        for start, size in self._free:
            if merged and merged[-1][0] + merged[-1][1] == start:
                previous_start, previous_size = merged[-1]
                merged[-1] = (previous_start, previous_size + size)
            else:
                merged.append((start, size))
        self._free = merged
        return placement

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._placements

    def __getitem__(self, request_id: str) -> Placement:
        return self._placements[request_id]


@dataclass(frozen=True, slots=True)
class Snapshot:
    time_ms: float
    mapped_chunks: int
    mapped_bytes: int
    useful_bytes: int
    reserved_bytes: int
    independent_chunks: int
    independent_bytes: int
    shared_chunks: int
    active_requests: int

    @property
    def tail_waste_bytes(self) -> int:
        return self.mapped_bytes - self.useful_bytes

    @property
    def shared_savings_bytes(self) -> int:
        return self.independent_bytes - self.mapped_bytes


@dataclass(frozen=True, slots=True)
class Admission:
    request_id: str
    arrival_ms: float
    start_ms: float
    finish_ms: float
    start_byte: int
    reserved_bytes: int

    @property
    def start_delay_ms(self) -> float:
        return self.start_ms - self.arrival_ms


@dataclass(frozen=True, slots=True)
class SimulationResult:
    strategy: str
    requests: int
    admitted_requests: int
    waiting_requests: int
    makespan_ms: float
    max_start_delay_ms: float
    mean_start_delay_ms: float
    max_active_requests: int
    peak: Snapshot
    admissions: tuple[Admission, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "requests": self.requests,
            "admitted_requests": self.admitted_requests,
            "waiting_requests": self.waiting_requests,
            "makespan_ms": self.makespan_ms,
            "max_start_delay_ms": self.max_start_delay_ms,
            "mean_start_delay_ms": self.mean_start_delay_ms,
            "max_active_requests": self.max_active_requests,
            "peak": {
                "time_ms": self.peak.time_ms,
                "active_requests": self.peak.active_requests,
                "mapped_chunks": self.peak.mapped_chunks,
                "mapped_bytes": self.peak.mapped_bytes,
                "useful_bytes": self.peak.useful_bytes,
                "tail_waste_bytes": self.peak.tail_waste_bytes,
                "reserved_bytes": self.peak.reserved_bytes,
                "independent_chunks": self.peak.independent_chunks,
                "independent_bytes": self.peak.independent_bytes,
                "shared_chunks": self.peak.shared_chunks,
                "shared_savings_bytes": self.peak.shared_savings_bytes,
                "mapping_efficiency": (
                    self.peak.useful_bytes / self.peak.mapped_bytes
                    if self.peak.mapped_bytes
                    else 1.0
                ),
            },
            "admissions": [
                {
                    "request_id": item.request_id,
                    "arrival_ms": item.arrival_ms,
                    "start_ms": item.start_ms,
                    "finish_ms": item.finish_ms,
                    "start_byte": item.start_byte,
                    "reserved_bytes": item.reserved_bytes,
                    "start_delay_ms": item.start_delay_ms,
                }
                for item in self.admissions
            ],
        }


@dataclass(slots=True)
class _ActiveRequest:
    request: AffineRequest
    placement: Placement
    start_ms: float
    prompt_done_ms: float
    finish_ms: float
    prefill_tokens_per_ms: float
    decode_tokens_per_ms: float

    def used_tokens(self, now_ms: float) -> int:
        if now_ms <= self.start_ms:
            return 0
        return min(
            self.request.total_tokens,
            max(0, math.floor(self._token_rate(now_ms) + 1e-9)),
        )

    def _token_rate(self, now_ms: float) -> float:
        # The simulator needs only aggregate geometry.  Treat the prompt as a
        # prefix that appears at the prefill rate, then advance the generated
        # prefix at the decode rate.
        if now_ms <= self.start_ms:
            return 0.0
        if now_ms <= self.prompt_done_ms:
            elapsed = now_ms - self.start_ms
            return elapsed * self.prefill_tokens_per_ms
        decode_ms = self.finish_ms - self.prompt_done_ms
        if decode_ms == 0:
            return self.request.total_tokens
        elapsed = now_ms - self.prompt_done_ms
        return self.request.prompt_tokens + elapsed * self.decode_tokens_per_ms


def chunks_for_prefix(
    placement: Placement, used_tokens: int, chunk_bytes: int
) -> set[int]:
    """Return every chunk touched by a live request prefix."""

    _positive_int(chunk_bytes, "chunk_bytes")
    if not 0 <= used_tokens <= placement.total_tokens:
        raise ValueError("used_tokens outside the reserved sequence")
    if used_tokens == 0:
        return set()
    first = placement.start_byte // chunk_bytes
    end = placement.token_offset(used_tokens - 1) + placement.row_bytes
    last = (end - 1) // chunk_bytes
    return set(range(first, last + 1))


def simulate_affine_placement(
    requests: Iterable[AffineRequest],
    config: AffineConfig,
    strategy: str,
) -> SimulationResult:
    """Run a deterministic event simulation for one placement strategy."""

    workload = tuple(requests)
    if not workload:
        raise ValueError("requests must not be empty")
    ids = [item.request_id for item in workload]
    if len(ids) != len(set(ids)):
        raise ValueError("request IDs must be unique")
    if strategy not in {"fixed-affine", "packed-affine"}:
        raise ValueError("strategy must be fixed-affine or packed-affine")

    max_tokens = max(item.total_tokens for item in workload)
    if strategy == "fixed-affine":
        slot_bytes = _ceil_div(max_tokens * config.row_bytes, config.chunk_bytes)
        slot_bytes *= config.chunk_bytes
    else:
        slot_bytes = None
    arena = AffineArena(config.arena_bytes, config.chunk_bytes, slot_bytes=slot_bytes)

    events: list[tuple[float, int, str, str]] = []
    for index, request in enumerate(workload):
        heapq.heappush(
            events, (request.arrival_ms, 0, f"arrival-{index:08d}", request.request_id)
        )

    by_id = {item.request_id: item for item in workload}
    waiting: set[str] = set()
    active: dict[str, _ActiveRequest] = {}
    admissions: list[Admission] = []
    peak: Snapshot | None = None
    last_time = 0.0
    max_active = 0

    def snapshot(now_ms: float, finishing: frozenset[str] = frozenset()) -> Snapshot:
        refs: Counter[int] = Counter()
        useful = 0
        reserved = 0
        independent = 0
        for request_id, item in active.items():
            used = (
                item.request.total_tokens
                if request_id in finishing
                else item.used_tokens(now_ms)
            )
            chunks = chunks_for_prefix(item.placement, used, config.chunk_bytes)
            refs.update(chunks)
            useful += used * config.row_bytes
            reserved += item.placement.reserved_bytes
            independent += len(chunks) * config.chunk_bytes
        return Snapshot(
            time_ms=now_ms,
            mapped_chunks=len(refs),
            mapped_bytes=len(refs) * config.chunk_bytes,
            useful_bytes=useful,
            reserved_bytes=reserved,
            independent_chunks=independent // config.chunk_bytes,
            independent_bytes=independent,
            shared_chunks=sum(count > 1 for count in refs.values()),
            active_requests=len(active),
        )

    def update_peak(candidate: Snapshot) -> None:
        nonlocal peak
        if peak is None or (
            candidate.mapped_bytes,
            candidate.useful_bytes,
            candidate.active_requests,
        ) > (
            peak.mapped_bytes,
            peak.useful_bytes,
            peak.active_requests,
        ):
            peak = candidate

    def admit_waiting(now_ms: float) -> None:
        nonlocal max_active
        made_progress = True
        while made_progress:
            made_progress = False
            for request_id in sorted(waiting):
                if len(active) >= config.max_active_requests:
                    return
                request = by_id[request_id]
                placement = arena.reserve(
                    request_id,
                    request.total_tokens,
                    config.row_bytes,
                )
                if placement is None:
                    continue
                prompt_ms = request.prompt_tokens / config.prefill_tokens_per_ms
                decode_ms = request.output_tokens / config.decode_tokens_per_ms
                prompt_done = now_ms + prompt_ms
                finish = prompt_done + decode_ms
                entry = _ActiveRequest(
                    request,
                    placement,
                    now_ms,
                    prompt_done,
                    finish,
                    config.prefill_tokens_per_ms,
                    config.decode_tokens_per_ms,
                )
                active[request_id] = entry
                waiting.remove(request_id)
                admissions.append(
                    Admission(
                        request_id,
                        request.arrival_ms,
                        now_ms,
                        finish,
                        placement.start_byte,
                        placement.reserved_bytes,
                    )
                )
                heapq.heappush(events, (finish, 1, f"finish-{request_id}", request_id))
                made_progress = True
                break
        max_active = max(max_active, len(active))

    while events:
        now_ms = events[0][0]
        last_time = now_ms
        current: list[tuple[float, int, str, str]] = []
        while events and events[0][0] == now_ms:
            current.append(heapq.heappop(events))

        arrivals = [item for item in current if item[1] == 0]
        finishes = [item for item in current if item[1] == 1]
        for _, _, _, request_id in arrivals:
            if request_id in active:
                raise RuntimeError(f"duplicate active request: {request_id}")
            waiting.add(request_id)

        # Capture the terminal prefix before releasing the finishing slots.
        if finishes:
            update_peak(snapshot(now_ms, frozenset(item[3] for item in finishes)))
            for _, _, _, request_id in finishes:
                active.pop(request_id)
                arena.release(request_id)

        admit_waiting(now_ms)
        update_peak(snapshot(now_ms))
        max_active = max(max_active, len(active))

    if waiting:
        raise RuntimeError("simulation ended with waiting requests")
    if peak is None:
        raise RuntimeError("simulation did not produce a snapshot")

    delays = [item.start_delay_ms for item in admissions]
    return SimulationResult(
        strategy=strategy,
        requests=len(workload),
        admitted_requests=len(admissions),
        waiting_requests=len(waiting),
        makespan_ms=last_time,
        max_start_delay_ms=max(delays, default=0.0),
        mean_start_delay_ms=(sum(delays) / len(delays) if delays else 0.0),
        max_active_requests=max_active,
        peak=peak,
        admissions=tuple(admissions),
    )


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_nonnegative(value: object, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")


def _positive_finite(value: object, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number")
