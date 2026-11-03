"""Collective demotion at a caller-owned quiescent scheduler boundary."""

from .migration import abort_demote, commit_demote, prepare_demote
from .policy import State


class MigrationFatal(RuntimeError):
    """The caller must stop every TP worker before resuming scheduling."""


class AdaptiveRuntime:
    """Run identical policy replicas using a CPU TP all-gather callback.

    The scheduler must invoke this on every rank, outside forward execution,
    with scheduling paused and global request IDs and policy time agreed.
    A fatal error is terminal. The scheduler must tear down the TP group.
    """

    def __init__(self, policy, coordinator, exchange, synchronize, trim_quiescent,
                 *, eager, overlap):
        if not eager or overlap:
            raise ValueError("adaptive migration requires eager execution without overlap")
        if policy.max_migration_requests != 1:
            raise ValueError("adaptive runtime requires one migration per boundary")
        self.policy = policy
        self.coordinator = coordinator
        self.exchange = exchange
        self.synchronize = synchronize
        self.trim_quiescent = trim_quiescent
        self.poisoned = False

    def _gather(self, value):
        replies = tuple(self.exchange(value))
        if not replies:
            raise MigrationFatal("empty TP collective")
        return replies

    def _agree(self, value):
        if any(other != value for other in self._gather(value)):
            raise MigrationFatal("TP migration plan disagreement")

    def at_prefill_boundary(self, peak_extra_bytes, candidates, requests, now):
        """Attempt demotions before prefill and return actual per-rank release.

        Candidates use a conservative common budget and an upper bound on
        attributable physical release. The trim callback must release only
        this transaction's backing, or the caller must drain prior slack
        separately before planning. No overlap with cache writes is permitted.
        """
        if self.poisoned:
            raise MigrationFatal("migration runtime is poisoned")
        try:
            return self._run(peak_extra_bytes, tuple(candidates), requests, now)
        except Exception as error:
            self.poisoned = True
            if isinstance(error, MigrationFatal):
                raise
            raise MigrationFatal("migration boundary failed") from error

    def _run(self, peak, candidates, requests, now):
        unsupported = any(candidate.growth_bytes for candidate in candidates)
        self._agree(("prefill", now, peak, candidates, unsupported))
        if unsupported:
            raise MigrationFatal("promotion is unsupported")
        planned = self.policy.plan(peak, candidates, now)
        self._agree(("plan", tuple(planned)))
        if not planned:
            return []
        chosen, = planned
        if chosen.state is not State.DEMOTING:
            raise MigrationFatal("unexpected migration direction")
        return [self._demote(chosen, requests.get(chosen.request_id), now)]

    def _demote(self, migration, request, now):
        resident = self.coordinator._ekv_resident
        row = request.kv.req_pool_idx if request is not None else None
        eligible = row in resident and not request.hisparse_staging
        if not all(self._gather(eligible)):
            self.policy.finish(migration, False, 0, now)
            return {"request_id": migration.request_id, "status": "rejected"}

        prepared = None
        local_error = None
        try:
            prepared = prepare_demote(self.coordinator, request, self.synchronize)
        except Exception as error:
            local_error = error
        if not all(self._gather(prepared is not None)):
            if prepared is not None:
                abort_demote(prepared)
            raise MigrationFatal("TP prepare failed; host/KV state requires teardown") from local_error

        released = None
        try:
            released = commit_demote(
                prepared, resident.remove, self.synchronize, self.trim_quiescent,
            )
        except Exception as error:
            local_error = error
        releases = self._gather(released)
        if any(type(value) is not int or value < 0 or value > migration.gpu_delta_bytes
               for value in releases):
            raise MigrationFatal("TP commit or physical-release accounting failed") from local_error
        common_release = min(releases)
        self.policy.finish(migration, True, common_release, now)
        return {"request_id": migration.request_id, "status": "demoted",
                "released_bytes": common_release, "released_bytes_by_rank": releases}
