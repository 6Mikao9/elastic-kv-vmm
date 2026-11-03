"""Byte accounting and migration policy. GPU actions are owned by the caller."""

from dataclasses import dataclass
from enum import Enum, auto
from threading import RLock


class State(Enum):
    GPU_RESIDENT = auto()
    DEMOTING = auto()
    HOST_BACKED = auto()
    PROMOTING = auto()


@dataclass
class Request:
    state: State
    gpu_bytes: int
    since: float


@dataclass(frozen=True)
class Candidate:
    request_id: str
    reclaimable_bytes: int
    growth_bytes: int
    transfer_bytes: int
    remaining_benefit: float = 1.0
    temporary_gpu_bytes: int = 0


@dataclass(frozen=True)
class Migration:
    request_id: str
    state: State
    gpu_delta_bytes: int
    transfer_bytes: int
    temporary_gpu_bytes: int


class BudgetPolicy:
    """Serialize accounting; execute returned migrations outside the lock.

    A demotion reports actual unique device bytes released. A failed operation
    must restore its starting residency before reporting failure. Candidate and
    request charges must exclude shared bytes already charged elsewhere. A
    scalar TP budget is valid only for a conservative common per-rank bound;
    the runtime must coordinate migration completion across ranks.
    """

    def __init__(self, capacity_bytes, high_bytes, low_bytes, min_dwell_s,
                 max_migration_bytes, max_migration_requests, low_stable_s=0):
        if not (0 <= low_bytes < high_bytes <= capacity_bytes
                and min_dwell_s >= 0 and low_stable_s >= 0 and max_migration_bytes > 0
                and max_migration_requests > 0):
            raise ValueError("invalid budget policy")
        self.capacity_bytes = capacity_bytes
        self.high_bytes = high_bytes
        self.low_bytes = low_bytes
        self.min_dwell_s = min_dwell_s
        self.low_stable_s = low_stable_s
        self._low_since = None
        self.max_migration_bytes = max_migration_bytes
        self.max_migration_requests = max_migration_requests
        self._requests = {}
        self._grants = {}
        self._inflight = {}
        self._committed = 0
        self._reserved = 0
        self._migration_bytes = 0
        self._lock = RLock()

    def snapshot(self):
        with self._lock:
            return {"committed_bytes": self._committed, "reserved_bytes": self._reserved,
                    "migration_bytes": self._migration_bytes,
                    "migration_requests": len(self._inflight),
                    "states": {key: req.state for key, req in self._requests.items()}}

    def add(self, request_id, gpu_bytes, now):
        with self._lock:
            if request_id in self._requests or gpu_bytes < 0:
                raise ValueError("invalid request registration")
            if self._committed + self._reserved + gpu_bytes > self.capacity_bytes:
                return False
            self._requests[request_id] = Request(State.GPU_RESIDENT, gpu_bytes, now)
            self._committed += gpu_bytes
            return True

    def remove(self, request_id):
        """Call only after the runtime has confirmed device storage release."""
        with self._lock:
            if request_id in self._inflight:
                raise ValueError("cannot remove a migrating request")
            self._committed -= self._requests.pop(request_id).gpu_bytes

    def reserve_prefill(self, key, size):
        """Reserve a caller-supplied memory grant. Activation integration is external."""
        with self._lock:
            if key in self._grants or size < 0:
                raise ValueError("invalid prefill grant")
            if self._committed + self._reserved + size > self.capacity_bytes:
                return False
            self._grants[key] = size
            self._reserved += size
            return True

    def release_prefill(self, key):
        with self._lock:
            self._reserved -= self._grants.pop(key)

    def commit_prefill(self, key, request_id, gpu_bytes, now):
        """Convert part of an existing grant to retained GPU KV atomically."""
        with self._lock:
            grant = self._grants[key]
            if request_id in self._requests or not 0 <= gpu_bytes <= grant:
                raise ValueError("invalid prefill commitment")
            self._requests[request_id] = Request(State.GPU_RESIDENT, gpu_bytes, now)
            self._committed += gpu_bytes
            self.release_prefill(key)

    def commit_growth(self, key, request_id, gpu_bytes):
        """Convert a reservation to unique device bytes for a stable request."""
        with self._lock:
            req = self._requests[request_id]
            grant = self._grants[key]
            if request_id in self._inflight or not 0 <= gpu_bytes <= grant:
                raise ValueError("invalid request growth commitment")
            req.gpu_bytes += gpu_bytes
            self._committed += gpu_bytes
            self.release_prefill(key)

    def begin_demote(self, request_id, reclaim_bytes, transfer_bytes, now,
                     temporary_gpu_bytes=0, *, emergency=False):
        """Emergency bypasses dwell only. Admission and migration limits still apply."""
        return self._begin(request_id, State.DEMOTING, reclaim_bytes, transfer_bytes, now,
                           temporary_gpu_bytes, emergency=emergency)

    def begin_promote(self, request_id, growth_bytes, transfer_bytes, now, temporary_gpu_bytes=0):
        return self._begin(request_id, State.PROMOTING, growth_bytes, transfer_bytes, now,
                           temporary_gpu_bytes)

    def _begin(self, request_id, state, delta, transfer, now, temporary_gpu_bytes, emergency=False):
        with self._lock:
            req = self._requests[request_id]
            promoting = state is State.PROMOTING
            origin = State.HOST_BACKED if promoting else State.GPU_RESIDENT
            if (delta <= 0 or transfer < 0 or temporary_gpu_bytes < 0
                    or (not promoting and delta > req.gpu_bytes)):
                raise ValueError("invalid migration size")
            if req.state is not origin or (not emergency and now - req.since < self.min_dwell_s):
                return None
            if (len(self._inflight) >= self.max_migration_requests
                    or self._migration_bytes + transfer > self.max_migration_bytes):
                return None
            reservation = temporary_gpu_bytes + (delta if promoting else 0)
            if self._committed + self._reserved + reservation > self.capacity_bytes:
                return None
            self._reserved += reservation
            migration = Migration(request_id, state, delta, transfer, temporary_gpu_bytes)
            self._migration_bytes += transfer
            self._inflight[request_id] = migration
            req.state = state
            return migration

    def finish(self, migration, success, actual_bytes, now):
        """Confirm complete target residency, using measured net device bytes.

        The migration delta is a reservation/estimate, not a copy-progress
        counter. Partial copies must complete or roll back before this call.
        """
        with self._lock:
            request_id = migration.request_id
            if self._inflight.get(request_id) is not migration:
                raise ValueError("stale migration completion")
            if not 0 <= actual_bytes <= migration.gpu_delta_bytes or (not success and actual_bytes):
                raise ValueError("invalid migration completion bytes")
            req = self._requests[request_id]
            self._reserved -= migration.temporary_gpu_bytes
            if migration.state is State.PROMOTING:
                self._reserved -= migration.gpu_delta_bytes
                delta = actual_bytes
                req.state = State.GPU_RESIDENT if success else State.HOST_BACKED
            else:
                delta = -actual_bytes
                req.state = State.HOST_BACKED if success else State.GPU_RESIDENT
            req.gpu_bytes += delta
            req.since = now
            self._committed += delta
            self._migration_bytes -= migration.transfer_bytes
            del self._inflight[request_id]

    def plan(self, peak_extra_bytes, candidates, now):
        """Reserve migrations for peak demand beyond grants.

        Call at budget observation points. A sampled excursion to or above
        low water resets the promotion stability window.
        """
        if peak_extra_bytes < 0:
            raise ValueError("negative peak demand")
        with self._lock:
            projected = self._committed + self._reserved + peak_extra_bytes
            migrations = []
            if projected >= self.low_bytes or any(
                m.state is State.DEMOTING for m in self._inflight.values()
            ):
                self._low_since = None
            elif self._low_since is None:
                self._low_since = now
            if projected > self.high_bytes:
                # Pending demotions inform planning only. Their bytes cannot be
                # used by admissions until finish confirms release.
                projected -= sum(m.gpu_delta_bytes + m.temporary_gpu_bytes
                                 for m in self._inflight.values()
                                 if m.state is State.DEMOTING)
                ranked = sorted(candidates,
                                key=lambda c: c.reclaimable_bytes / max(1, c.transfer_bytes),
                                reverse=True)
                for candidate in ranked:
                    if projected <= self.high_bytes:
                        break
                    if candidate.reclaimable_bytes > 0:
                        req = self._requests[candidate.request_id]
                        if req.state is not State.GPU_RESIDENT:
                            continue
                        migration = self.begin_demote(candidate.request_id, candidate.reclaimable_bytes,
                                                      candidate.transfer_bytes, now,
                                                      candidate.temporary_gpu_bytes)
                        if migration:
                            migrations.append(migration)
                            projected -= migration.gpu_delta_bytes
            elif self._low_since is not None and now - self._low_since >= self.low_stable_s:
                ranked = sorted(candidates,
                                key=lambda c: c.remaining_benefit / max(1, c.transfer_bytes),
                                reverse=True)
                for candidate in ranked:
                    if (candidate.growth_bytes <= 0 or candidate.remaining_benefit <= 0
                            or projected + candidate.growth_bytes + candidate.temporary_gpu_bytes
                            > self.low_bytes):
                        continue
                    migration = self.begin_promote(candidate.request_id, candidate.growth_bytes,
                                                   candidate.transfer_bytes, now,
                                                   candidate.temporary_gpu_bytes)
                    if migration:
                        migrations.append(migration)
                        projected += migration.gpu_delta_bytes + migration.temporary_gpu_bytes
            return migrations
