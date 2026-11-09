"""Request-aware VMM slot placement without changing attention addressing.

SGLang's token allocator returns logical slot IDs.  Dense VMM maps a logical
chunk of those IDs to one physical chunk.  The standard first-fit allocator
lets tails from different requests share a physical chunk, which couples their
reclaim.  This module keeps a long request's aligned chunks private while small
requests continue to pack together.  The attention path still consumes ordinary
logical slot IDs; only the allocator's ownership policy changes.

The planner is deliberately independent from CUDA and SGLang so placement
geometry and lifecycle behavior can be tested on CPU.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import groupby

REQUEST_AFFINE_MODES = ("off", "dry_run", "enforce")
DEFAULT_REQUEST_AFFINE_MODE = "off"


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _validate_ranges(ranges: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    result = tuple((int(start), int(stop)) for start, stop in ranges)
    previous_stop = 0
    for start, stop in result:
        if start < 1 or stop <= start or start < previous_stop:
            raise ValueError("ranges must be positive, sorted, and disjoint")
        previous_stop = stop
    return result


def _ranges_for_ids(ids: Iterable[int]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(int(item) for item in ids)
    ranges = []
    for _, group in groupby(enumerate(ordered), lambda item: item[1] - item[0]):
        values = [value for _, value in group]
        ranges.append((values[0], values[-1] + 1))
    return tuple(ranges)


def _subtract_ranges(
    free_ranges: Sequence[tuple[int, int]],
    removed_ranges: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    free = _validate_ranges(free_ranges)
    removed = _validate_ranges(removed_ranges)
    if not removed:
        return free
    result = []
    remove_index = 0
    for free_start, free_stop in free:
        cursor = free_start
        while remove_index < len(removed) and removed[remove_index][1] <= cursor:
            remove_index += 1
        probe = remove_index
        while probe < len(removed) and removed[probe][0] < free_stop:
            remove_start, remove_stop = removed[probe]
            if remove_stop <= cursor:
                probe += 1
                continue
            if remove_start > cursor:
                result.append((cursor, min(remove_start, free_stop)))
            cursor = max(cursor, min(remove_stop, free_stop))
            if cursor >= free_stop:
                break
            probe += 1
        if cursor < free_stop:
            result.append((cursor, free_stop))
    return tuple(result)


def _take_ranges(
    ranges: Sequence[tuple[int, int]], count: int
) -> tuple[tuple[int, int], ...] | None:
    if count < 0:
        raise ValueError("slot count must be nonnegative")
    result = []
    remaining = count
    for start, stop in _validate_ranges(ranges):
        if remaining <= 0:
            break
        end = min(stop, start + remaining)
        result.append((start, end))
        remaining -= end - start
    return tuple(result) if remaining == 0 else None


@dataclass(frozen=True, slots=True)
class RequestDemand:
    """One request's current allocation and projected lifetime."""

    request_id: str
    slots: int
    projected_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a nonempty string")
        _positive_int(self.slots, "slots")
        _positive_int(self.projected_tokens, "projected_tokens")
        if self.projected_tokens < self.slots:
            raise ValueError("projected_tokens must cover the current allocation")


@dataclass(frozen=True, slots=True)
class PlannedRequest:
    request_id: str
    private: bool
    ranges: tuple[tuple[int, int], ...]

    @property
    def slot_ids(self) -> tuple[int, ...]:
        return tuple(slot for start, stop in self.ranges for slot in range(start, stop))


@dataclass(frozen=True, slots=True)
class RequestAffinePlan:
    requests: tuple[PlannedRequest, ...]
    new_reserved_chunks: tuple[int, ...]
    reserved_chunk_ranges: tuple[tuple[int, int], ...]
    packed_ranges: tuple[tuple[int, int], ...]

    @property
    def all_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            sorted(
                (item for request in self.requests for item in request.ranges),
                key=lambda value: value[0],
            )
        )

    @property
    def private_requests(self) -> int:
        return sum(request.private for request in self.requests)

    @property
    def packed_requests(self) -> int:
        return len(self.requests) - self.private_requests


@dataclass(frozen=True, slots=True)
class PackedObservation:
    requests: int
    private_candidates: int
    mapped_chunks: int
    shared_chunks: int
    maximum_owners: int


class RequestAffinePlanner:
    """Hybrid private/packed logical placement.

    A request that needs at least ``chunk_slots`` projected tokens reserves a
    whole logical chunk at a time.  Its unused tail is withheld from first-fit
    so another request cannot make the chunk jointly reclaimable.  Shorter
    requests remain packed to preserve small-request capacity.
    """

    def __init__(self, pool_size: int, chunk_slots: int) -> None:
        self.pool_size = _positive_int(pool_size, "pool_size")
        self.chunk_slots = _positive_int(chunk_slots, "chunk_slots")
        self._chunk_owner: dict[int, str] = {}
        self._request_chunks: dict[str, set[int]] = {}
        self._slot_owner: dict[int, str] = {}
        self._free_chunks: set[int] = set()

    @property
    def free_chunks(self) -> frozenset[int]:
        return frozenset(self._free_chunks)

    @property
    def private_request_count(self) -> int:
        return len(self._request_chunks)

    @property
    def withheld_slots(self) -> int:
        chunks = (
            set().union(*self._request_chunks.values())
            if self._request_chunks
            else set()
        )
        return len(chunks) * self.chunk_slots

    def chunk_for_slot(self, slot: int) -> int:
        if not 0 <= slot <= self.pool_size:
            raise ValueError("slot outside the pool")
        return slot // self.chunk_slots

    def is_complete_chunk(self, chunk: int) -> bool:
        """Return whether ``chunk`` has a full aligned range in the pool."""

        if chunk < 1:
            return False
        start = chunk * self.chunk_slots
        return start + self.chunk_slots - 1 <= self.pool_size

    def is_slot_withheld(self, slot: int) -> bool:
        """Return whether a slot belongs to a live private request chunk."""

        return self.chunk_for_slot(int(slot)) in self._chunk_owner

    def chunk_range(self, chunk: int) -> tuple[int, int]:
        if chunk < 1:
            raise ValueError("request-affine chunks start after the sink chunk")
        start = chunk * self.chunk_slots
        stop = start + self.chunk_slots
        if start < 1 or stop - 1 > self.pool_size:
            raise ValueError("chunk is incomplete or outside the pool")
        return start, stop

    def prime(self, is_live: Callable[[int], bool]) -> None:
        self._free_chunks.clear()
        final_chunk = (self.pool_size + 1) // self.chunk_slots - 1
        for chunk in range(1, final_chunk + 1):
            start, stop = self.chunk_range(chunk)
            if not any(is_live(slot) for slot in range(start, stop)):
                self._free_chunks.add(chunk)

    def reset(self) -> None:
        self._chunk_owner.clear()
        self._request_chunks.clear()
        self._slot_owner.clear()
        self._free_chunks.clear()

    def _fill_private_chunks(
        self,
        request: RequestDemand,
        chunks: set[int],
        candidate_chunks: set[int],
        allocated: set[int],
        is_live: Callable[[int], bool],
    ) -> tuple[list[int], set[int]] | None:
        selected = []
        new_chunks: set[int] = set()
        for chunk in sorted(chunks):
            start, stop = self.chunk_range(chunk)
            for slot in range(start, stop):
                if slot not in allocated and not is_live(slot):
                    selected.append(slot)
                    allocated.add(slot)
                    if len(selected) == request.slots:
                        return selected, new_chunks
        for chunk in sorted(candidate_chunks):
            start, stop = self.chunk_range(chunk)
            if any(is_live(slot) for slot in range(start, stop)):
                candidate_chunks.discard(chunk)
                continue
            candidate_chunks.remove(chunk)
            chunks.add(chunk)
            new_chunks.add(chunk)
            for slot in range(start, stop):
                if slot not in allocated:
                    selected.append(slot)
                    allocated.add(slot)
                    if len(selected) == request.slots:
                        return selected, new_chunks
        return None

    def plan(
        self,
        demands: Sequence[RequestDemand],
        *,
        free_ranges: Sequence[tuple[int, int]],
        is_live: Callable[[int], bool],
    ) -> RequestAffinePlan | None:
        if not demands:
            return None
        ids = [demand.request_id for demand in demands]
        if len(ids) != len(set(ids)):
            raise ValueError("request IDs in one allocation must be unique")
        free_ranges = _validate_ranges(free_ranges)
        candidate_chunks = set(self._free_chunks)
        allocated: set[int] = set()
        private_chunks = {
            request_id: set(chunks)
            for request_id, chunks in self._request_chunks.items()
        }
        new_reserved: set[int] = set()
        planned: list[PlannedRequest] = []
        packed_demands: list[RequestDemand] = []

        for demand in demands:
            private = (
                demand.request_id in private_chunks
                or demand.projected_tokens >= self.chunk_slots
            )
            if not private:
                packed_demands.append(demand)
                continue
            chunks = private_chunks.setdefault(demand.request_id, set())
            filled = self._fill_private_chunks(
                demand,
                chunks,
                candidate_chunks,
                allocated,
                is_live,
            )
            if filled is None:
                return None
            selected, added_chunks = filled
            private_chunks[demand.request_id] = chunks
            new_reserved.update(added_chunks)
            planned.append(
                PlannedRequest(
                    demand.request_id,
                    True,
                    _ranges_for_ids(selected),
                )
            )

        reserved_chunk_ranges = tuple(
            self.chunk_range(chunk) for chunk in sorted(new_reserved)
        )
        packed_available = _subtract_ranges(free_ranges, reserved_chunk_ranges)
        packed_count = sum(demand.slots for demand in packed_demands)
        packed_ranges = _take_ranges(packed_available, packed_count)
        if packed_ranges is None:
            return None
        packed_ids = [
            slot for start, stop in packed_ranges for slot in range(start, stop)
        ]
        packed_offset = 0
        for demand in packed_demands:
            request_ids = packed_ids[packed_offset : packed_offset + demand.slots]
            if len(request_ids) != demand.slots:
                raise RuntimeError("internal packed placement shortfall")
            request_ranges = _ranges_for_ids(request_ids)
            planned.append(PlannedRequest(demand.request_id, False, request_ranges))
            packed_offset += demand.slots

        order = {demand.request_id: index for index, demand in enumerate(demands)}
        planned.sort(key=lambda request: order[request.request_id])
        return RequestAffinePlan(
            tuple(planned),
            tuple(sorted(new_reserved)),
            reserved_chunk_ranges,
            packed_ranges,
        )

    def commit(self, plan: RequestAffinePlan) -> None:
        # Validate the complete plan first.  The allocator mutates both its free
        # ranges and the residency ledger after this call, so an ownership
        # conflict must not leave only part of the placement committed.
        for chunk in plan.new_reserved_chunks:
            if chunk not in self._free_chunks:
                raise RuntimeError("planned chunk was not free at commit")
        planned_owner: dict[int, str] = {}
        for request in plan.requests:
            if not request.private:
                continue
            for slot in request.slot_ids:
                chunk = self.chunk_for_slot(slot)
                owner = planned_owner.get(chunk, self._chunk_owner.get(chunk))
                if owner is not None and owner != request.request_id:
                    raise RuntimeError("private chunk owner changed at commit")
                if slot in self._slot_owner:
                    raise RuntimeError("private slot was already committed")
                planned_owner[chunk] = request.request_id

        for chunk in plan.new_reserved_chunks:
            self._free_chunks.remove(chunk)
        for request in plan.requests:
            if request.private:
                chunks = self._request_chunks.setdefault(request.request_id, set())
                for slot in request.slot_ids:
                    chunk = self.chunk_for_slot(slot)
                    self._chunk_owner[chunk] = request.request_id
                    self._slot_owner[slot] = request.request_id
                    chunks.add(chunk)
            else:
                # A packed chunk is not a safe candidate for a future private
                # request while any of its slots are live.
                for slot in request.slot_ids:
                    self._free_chunks.discard(self.chunk_for_slot(slot))

    def note_packed_slots(self, slot_ids: Sequence[int]) -> None:
        """Exclude chunks touched by a first-fit allocation from reuse."""

        for slot in slot_ids:
            self._free_chunks.discard(self.chunk_for_slot(int(slot)))

    def release_slots(
        self,
        slot_ids: Sequence[int],
        *,
        is_live: Callable[[int], bool],
    ) -> tuple[tuple[int, int], ...]:
        """Update owners after slots change from live to free.

        The caller marks SGLang's ``live`` bits before this method.  Returned
        ranges are complete private chunks that must be returned to ``FreeSlots``
        as a whole; their used slots are covered by those ranges.
        """

        touched_requests = set()
        touched_chunks = {self.chunk_for_slot(int(slot)) for slot in slot_ids}
        for slot in slot_ids:
            owner = self._slot_owner.pop(int(slot), None)
            if owner is not None:
                touched_requests.add(owner)
        # A packed slot can be the last live user in a chunk still owned by a
        # private request. Include that owner even though the packed slot has
        # no entry in ``_slot_owner``; otherwise the empty chunk leaks from the
        # private pool permanently.
        for chunk in touched_chunks:
            owner = self._chunk_owner.get(chunk)
            if owner is not None:
                touched_requests.add(owner)
        released = []
        released_chunks = set()
        for request_id in sorted(touched_requests):
            chunks = self._request_chunks.get(request_id)
            if not chunks:
                continue
            for chunk in sorted(chunks):
                start, stop = self.chunk_range(chunk)
                if any(is_live(slot) for slot in range(start, stop)):
                    continue
                chunks.remove(chunk)
                if self._chunk_owner.pop(chunk, None) != request_id:
                    raise RuntimeError("private chunk owner mismatch on release")
                self._free_chunks.add(chunk)
                released_chunks.add(chunk)
                released.append((start, stop))
            if not chunks:
                self._request_chunks.pop(request_id, None)
        for chunk in touched_chunks:
            if (
                not self.is_complete_chunk(chunk)
                or chunk in released_chunks
                or chunk in self._chunk_owner
            ):
                continue
            start, stop = self.chunk_range(chunk)
            if not any(is_live(slot) for slot in range(start, stop)):
                self._free_chunks.add(chunk)
        return tuple(released)


def observe_packed_allocation(
    demands: Sequence[RequestDemand],
    slots_by_request: Sequence[Sequence[int]],
    *,
    chunk_slots: int,
) -> PackedObservation:
    """Measure cross-request sharing in an allocation left as first-fit."""

    _positive_int(chunk_slots, "chunk_slots")
    if len(demands) != len(slots_by_request):
        raise ValueError("demand and allocation counts differ")
    owners: dict[int, set[str]] = {}
    for demand, slots in zip(demands, slots_by_request):
        if len(slots) != demand.slots:
            raise ValueError("request allocation length differs from demand")
        for slot in slots:
            chunk = int(slot) // chunk_slots
            owners.setdefault(chunk, set()).add(demand.request_id)
    return PackedObservation(
        requests=len(demands),
        private_candidates=sum(
            demand.projected_tokens >= chunk_slots for demand in demands
        ),
        mapped_chunks=len(owners),
        shared_chunks=sum(len(request_ids) > 1 for request_ids in owners.values()),
        maximum_owners=max((len(value) for value in owners.values()), default=0),
    )


def extract_request_demands(
    reqs: Sequence[object],
    current_seq_lens: Sequence[object],
    allocation_lens: Sequence[int],
    *,
    decode: bool,
) -> tuple[RequestDemand, ...] | None:
    """Extract SGLang request metadata without importing SGLang."""

    if len(reqs) != len(current_seq_lens) or len(reqs) != len(allocation_lens):
        return None
    result = []
    for req, seq_len, allocation_len in zip(reqs, current_seq_lens, allocation_lens):
        request_id = getattr(req, "rid", None)
        if not isinstance(request_id, str) or not request_id:
            return None
        if hasattr(seq_len, "item"):
            seq_len = seq_len.item()
        try:
            current = int(seq_len)
            slots = int(allocation_len)
        except (TypeError, ValueError):
            return None
        sampling = getattr(req, "sampling_params", None)
        max_new_tokens = getattr(sampling, "max_new_tokens", 0)
        try:
            max_new = max(0, int(max_new_tokens or 0))
        except (TypeError, ValueError):
            max_new = 0
        output_ids = getattr(req, "output_ids", ()) or ()
        try:
            generated = len(output_ids)
        except TypeError:
            generated = 0
        remaining_new = max(0, max_new - generated) if decode else max_new
        projected = max(current, slots) + remaining_new
        try:
            result.append(RequestDemand(request_id, slots, max(projected, slots)))
        except ValueError:
            return None
    return tuple(result)
