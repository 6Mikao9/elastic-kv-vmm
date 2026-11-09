"""MiniMax eager scheduler integration for budget-triggered demotion."""

import hashlib
import json
import logging
import math
import os
import time
from collections import Counter, deque
from dataclasses import replace
from functools import wraps
from itertools import combinations
from pathlib import Path

from .adaptive_runtime import MigrationFatal
from .budget_ownership import build_budget_ownership_snapshot
from .future_prefill import (
    FuturePrefillConfig,
    effective_prefill_headroom,
    empty_plan,
    plan_future_prefill,
    reconcile_tp_plan,
)
from .migration import (
    abort_demote,
    abort_promote,
    commit_demote,
    commit_promote,
    plan_demote_pages,
    prepare_demote,
    prepare_promote,
)
from .migration_budget import MigrationBandwidthEstimator
from .runtime_value import (
    DEFAULT_DEADLINE_GUARD_MS,
    PrefixReuseDemand,
    attach_prefix_reuse,
    forecast_prefix_reuse,
    merge_demotion_values,
    rank_demotions,
    rank_promotions,
    runtime_values,
)
from .runtime_value import evidence as runtime_value_evidence

SCHEDULER_SHA256S = frozenset({
    # Pinned MiniMax development image used by the request-level experiments.
    "fc96ed4b680fdf55621a5397f844cfd4cbd30213f44eec8a888b7cce75cafe60",
    # lmsysorg/sglang:latest with SGLang 0.5.17 (audited wrapper boundary).
    "1d4a7683e32f0862cf5b03be33e5ec23f9d112e59546d391e28fa222fcd1619b",
})
LOG = logging.getLogger(__name__)


def _should_plan_future_waiting(mode, waiting_queue, *, prefill, steps, observe):
    """Return whether this scheduler boundary should refresh future demand."""

    if observe < 1:
        raise ValueError("observe must be positive")
    return mode != "off" and bool(waiting_queue) and (
        prefill or (steps + 1) % observe == 0
    )


def _invalidate_predictive_prewarm(tracker, stage, exc):
    invalidate = getattr(tracker, "invalidate_predictive_prewarm", None)
    if callable(invalidate):
        try:
            invalidate(f"{stage}:{type(exc).__name__}:{str(exc)[:160]}")
        except Exception:
            pass
    else:
        try:
            tracker._predictive_prewarm_disabled_reason = str(exc)[:240]
        except Exception:
            pass


def _best_effort_predictive_prewarm(
    tracker, demand, controller=None, *, lead_ms=None
):
    """Run the optional VMM hint without replacing native admission."""

    prewarm = getattr(tracker, "predictive_prewarm", None)
    if not callable(prewarm):
        return 0
    try:
        due_bytes = getattr(demand, "prewarm_bytes", None)
        demand_bytes = due_bytes(lead_ms) if callable(due_bytes) else demand
        return prewarm(demand_bytes)
    except Exception as exc:
        if controller is not None:
            controller.stats["predictive_prewarm_hook_errors"] += 1
        _invalidate_predictive_prewarm(tracker, "scheduler-prewarm", exc)
        LOG.warning("predictive prewarm fell back to native admission: %s", exc)
        return 0


def _best_effort_expire_predictive_prewarm(tracker, controller=None):
    """Reclaim expired speculative mappings without breaking an idle tick."""

    expire = getattr(tracker, "expire_predictive_prewarm", None)
    if not callable(expire):
        return 0
    try:
        return expire()
    except Exception as exc:
        if controller is not None:
            controller.stats["predictive_prewarm_expire_errors"] += 1
        _invalidate_predictive_prewarm(tracker, "scheduler-expire", exc)
        LOG.warning("predictive prewarm expiry failed closed: %s", exc)
        return 0


def _broker_snapshot(tracker):
    """Return a validated aggregate VMM snapshot, or None when disabled."""

    value = getattr(tracker.backing, "broker_snapshot", None)
    if value is None:
        return None
    if callable(value):
        value = value()
    if value is None:
        return None
    try:
        capacity = int(value["capacity_bytes"])
        outstanding = int(value["outstanding_bytes"])
        available = int(value["available_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("invalid global VMM broker snapshot") from exc
    if capacity < 1 or not 0 <= outstanding <= capacity:
        raise RuntimeError("invalid global VMM broker snapshot")
    if available != capacity - outstanding:
        raise RuntimeError("inconsistent global VMM broker snapshot")
    fairness_pressure = value.get("fairness_pressure", False)
    if not isinstance(fairness_pressure, bool):
        raise RuntimeError("invalid global VMM broker fairness pressure")
    return {
        "capacity_bytes": capacity,
        "outstanding_bytes": outstanding,
        "available_bytes": available,
        "fairness_pressure": fairness_pressure,
        "lease_bytes": value.get("lease_bytes"),
        "claim_bytes": value.get("claim_bytes"),
        "claims": value.get("claims"),
        "fair_mode": value.get("fair_mode"),
        "fair_share_would_yield": value.get("fair_share_would_yield"),
    }


def _releasable(tracker, pages):
    return _releasable_counts(tracker, _chunk_release_counts(tracker, set(pages)))


def _chunk_release_counts(tracker, pages):
    return Counter(chunk for page in pages for chunk in tracker.chunks(page))


def _releasable_counts(tracker, counts):
    mapped = tracker.backing.mapped_chunks
    return {chunk for chunk, count in counts.items()
            if count == tracker.refs.get(chunk) and chunk in mapped}


def _releasable_pair(tracker, left_pages, left_counts, right_pages, right_counts):
    # Preserve union semantics if two candidates name the same physical page.
    if not left_pages.isdisjoint(right_pages):
        return _releasable(tracker, left_pages | right_pages)
    return _releasable_counts(tracker, left_counts + right_counts)


def _releasable_group(tracker, candidates):
    """Return chunks that become releasable after a bounded request group.

    Candidates normally own disjoint physical pages.  If two requests name the
    same page, additive chunk counts would count that page twice, so recompute
    from the page union for that uncommon case.
    """
    pages = set()
    counts = Counter()
    overlap = False
    for candidate_pages, candidate_counts in candidates:
        if pages & candidate_pages:
            overlap = True
        pages.update(candidate_pages)
        counts.update(candidate_counts)
    if overlap:
        return _releasable(tracker, pages)
    return _releasable_counts(tracker, counts)


def _bounded_demotion_groups(tracker, requests, cost, *, max_group,
                             byte_cap, search_limit, rotation=0,
                             cap_observer=None):
    """Enumerate small request groups with a deterministic work bound.

    The common case has only a handful of resident long requests, so exhaustive
    enumeration through ``max_group`` is cheaper and more predictable than
    maintaining a chunk-to-request inverted index at every scheduler boundary.
    The bound protects the pressure path if a dense deployment has many active
    requests; telemetry reports when that bound was reached.
    """
    groups = {}
    chunks = {}
    examined = 0
    truncated = False
    rejected_by_cap = 0
    if search_limit < 1:
        raise ValueError("search_limit must be positive")
    request_ids = sorted(requests)
    if request_ids:
        rotation %= len(request_ids)
        request_ids = request_ids[rotation:] + request_ids[:rotation]
    sizes = tuple(range(2, min(max_group, len(request_ids)) + 1))
    base_quota, extra = divmod(search_limit, max(1, len(sizes)))
    quotas = [base_quota + (index < extra) for index in range(len(sizes))]
    for size, quota in zip(sizes, quotas):
        size_examined = 0
        for group in combinations(request_ids, size):
            if size_examined >= quota:
                truncated = True
                break
            examined += 1
            size_examined += 1
            transfer = sum(cost(request_id) for request_id in group)
            if transfer > byte_cap:
                rejected_by_cap += 1
                if cap_observer is not None:
                    cap_observer(transfer)
                continue
            released = _releasable_group(
                tracker,
                (
                    (requests[request_id][1], requests[request_id][2])
                    for request_id in group
                ),
            )
            if released:
                group = tuple(sorted(group))
                groups[group] = (len(released) * tracker.chunk_bytes, transfer)
                chunks[group] = released
    return groups, chunks, examined, truncated, rejected_by_cap


class PhysicalDemandIndex:
    """Incrementally index request pages and the physical chunks they release.

    The scheduler already computes page ownership for candidates.  This index
    reuses those values across pressure ticks, so a changed request updates
    only its own page/chunk entries instead of rebuilding a chunk-to-owner map
    for every request on every scheduler boundary.
    """

    def __init__(self):
        self.records = {}
        self.page_owners = {}
        self.chunk_owners = {}

    def sync(self, requests):
        active = set(requests)
        changed = 0
        for request_id in tuple(self.records):
            if request_id not in active:
                self._remove(request_id)
                changed += 1
        for request_id, entry in requests.items():
            pages = frozenset(entry[1])
            chunks = Counter({chunk: count for chunk, count in entry[2].items() if count})
            record = (pages, chunks)
            if self.records.get(request_id) == record:
                continue
            if request_id in self.records:
                self._remove(request_id)
            self._add(request_id, pages, chunks)
            changed += 1
        return changed

    def _add(self, request_id, pages, chunks):
        self.records[request_id] = (pages, Counter(chunks))
        for page in pages:
            self.page_owners.setdefault(page, set()).add(request_id)
        for chunk, count in chunks.items():
            self.chunk_owners.setdefault(chunk, {})[request_id] = count

    def _remove(self, request_id):
        pages, chunks = self.records.pop(request_id)
        for page in pages:
            owners = self.page_owners[page]
            owners.remove(request_id)
            if not owners:
                del self.page_owners[page]
        for chunk in chunks:
            owners = self.chunk_owners[chunk]
            del owners[request_id]
            if not owners:
                del self.chunk_owners[chunk]

    @property
    def has_page_overlap(self):
        return any(len(owners) > 1 for owners in self.page_owners.values())

    def _candidate_groups(self, tracker, *, max_group, candidate_limit, rotation):
        request_ids = sorted(self.records)
        if request_ids:
            rotation %= len(request_ids)
            request_ids = request_ids[rotation:] + request_ids[:rotation]
        order = {request_id: index for index, request_id in enumerate(request_ids)}
        queue = deque((request_id,) for request_id in request_ids)
        seen = set()
        candidates = set()
        truncated = False
        while queue:
            group = tuple(sorted(queue.popleft()))
            if group in seen:
                continue
            seen.add(group)
            if len(group) >= max_group:
                if len(group) >= 2:
                    candidates.add(group)
                continue
            chunks = set().union(
                *(self.records[request_id][1] for request_id in group)
            )
            expanded_any = False
            for chunk in chunks:
                owner_counts = self.chunk_owners.get(chunk, {})
                if sum(owner_counts.values()) != tracker.refs.get(chunk):
                    continue
                expanded = tuple(sorted(set(group) | set(owner_counts)))
                if len(expanded) <= max_group and expanded != group:
                    queue.append(expanded)
                    expanded_any = True
            if len(group) >= 2:
                candidates.add(group)
            if not expanded_any and len(seen) >= candidate_limit:
                truncated = True
                break
            if len(seen) >= candidate_limit:
                truncated = True
                break
        ordered = sorted(
            (group for group in candidates if len(group) >= 2),
            key=lambda group: tuple(order[request_id] for request_id in group),
        )
        return ordered, truncated


def _indexed_demotion_groups(index, tracker, requests, cost, *, max_group,
                             byte_cap, search_limit, rotation=0,
                             cap_observer=None):
    """Search owner closures before falling back to bounded enumeration."""
    if search_limit < 1:
        raise ValueError("search_limit must be positive")
    candidate_limit = max(32, search_limit * 4)
    candidates, candidate_truncated = index._candidate_groups(
        tracker,
        max_group=max_group,
        candidate_limit=candidate_limit,
        rotation=rotation,
    )
    groups = {}
    chunks = {}
    examined = rejected_by_cap = 0
    truncated = candidate_truncated
    for group in candidates:
        if examined >= search_limit:
            truncated = True
            break
        examined += 1
        transfer = sum(cost(request_id) for request_id in group)
        if transfer > byte_cap:
            rejected_by_cap += 1
            if cap_observer is not None:
                cap_observer(transfer)
            continue
        released = _releasable_group(
            tracker,
            (
                (requests[request_id][1], requests[request_id][2])
                for request_id in group
            ),
        )
        if released:
            groups[group] = (len(released) * tracker.chunk_bytes, transfer)
            chunks[group] = released
    return groups, chunks, examined, truncated, rejected_by_cap


def _demand_index_demotion_groups(index, tracker, requests, cost, *, max_group,
                                  byte_cap, search_limit, rotation=0,
                                  cap_observer=None):
    """Use indexed owner closures and exact fallback for overlapping pages."""
    groups, chunks, indexed_examined, truncated, rejected = (
        _indexed_demotion_groups(
            index,
            tracker,
            requests,
            cost,
            max_group=max_group,
            byte_cap=byte_cap,
            search_limit=search_limit,
            rotation=rotation,
            cap_observer=cap_observer,
        )
    )
    if groups or not index.has_page_overlap:
        return (
            groups, chunks, indexed_examined, 0, truncated, False,
            rejected, False,
        )
    (groups, chunks, fallback_examined, fallback_truncated,
     fallback_rejected) = _bounded_demotion_groups(
        tracker,
        requests,
        cost,
        max_group=max_group,
        byte_cap=byte_cap,
        search_limit=search_limit,
        rotation=rotation,
        cap_observer=cap_observer,
    )
    return (
        groups,
        chunks,
        indexed_examined,
        fallback_examined,
        truncated,
        fallback_truncated,
        rejected + fallback_rejected,
        True,
    )


def _rank_demotions(
    rank_candidates,
    rank_values=None,
    *,
    deadline_guard_ms=DEFAULT_DEADLINE_GUARD_MS,
):
    """Backward-compatible adapter to the pure runtime-value ranker."""

    return rank_demotions(
        tuple(rank_candidates), tuple(rank_values or ()),
        guard_ms=deadline_guard_ms,
    )


class SchedulerBudget:
    def __init__(self, budget, headroom, dwell, force_step, low=0, stable=2,
                 observe=16, promote_step=0, stats_path=None, byte_cap=512 << 20,
                 max_group=1, capture_plans=False, trim_low_water=False,
                 group_search_limit=256, deadline_guard_ms=DEFAULT_DEADLINE_GUARD_MS,
                 migration_time_cap_ms=0,
                 initial_migration_bandwidth_bytes_per_s=1 << 30,
                 migration_bandwidth_ewma_alpha=0.25,
                 broker_headroom_bytes=0, broker_low_bytes=0):
        if max_group not in (1, 2, 3, 4):
            raise ValueError("max_group must be between 1 and 4")
        if group_search_limit < 1:
            raise ValueError("group_search_limit must be positive")
        if deadline_guard_ms < 0:
            raise ValueError("deadline_guard_ms must be nonnegative")
        if migration_time_cap_ms < 0:
            raise ValueError("migration_time_cap_ms must be nonnegative")
        if broker_headroom_bytes < 0 or broker_low_bytes < 0:
            raise ValueError("global broker watermarks must be nonnegative")
        if bool(broker_headroom_bytes) != bool(broker_low_bytes):
            raise ValueError("global broker headroom and low watermark must be paired")
        if broker_headroom_bytes and broker_low_bytes <= broker_headroom_bytes:
            raise ValueError("global broker low watermark must exceed headroom")
        self.budget = budget
        self.headroom = headroom
        self.dwell = dwell
        self.force_step = force_step
        self.low = low
        self.stable = stable
        self.observe = observe
        self.promote_step = promote_step
        self.promoted = False
        self.low_since = None
        self.stats_path = stats_path
        self.byte_cap = byte_cap
        self.max_group = max_group
        self.group_search_limit = group_search_limit
        self.deadline_guard_ms = deadline_guard_ms
        self.migration_time_cap_ms = migration_time_cap_ms
        self.broker_headroom_bytes = broker_headroom_bytes
        self.broker_low_bytes = broker_low_bytes
        self.migration_bandwidth = MigrationBandwidthEstimator(
            initial_migration_bandwidth_bytes_per_s,
            migration_bandwidth_ewma_alpha,
        )
        self.capture_plans = capture_plans
        self.trim_low_water = trim_low_water
        self.demand_index = PhysicalDemandIndex()
        self.plan_snapshots = []
        self.plan_keys = set()
        self.residency_since = {}
        self.stats = {"demotions": 0, "promotions": 0, "migration_seconds": 0,
                      "transfer_bytes": 0, "released_bytes": 0, "growth_bytes": 0,
                      "max_observed_backed_bytes": 0, "reasons": {},
                      "rejected_cap": 0, "rejected_dwell": 0,
                      "pressure_observations": 0, "no_reclaimable_candidate": 0,
                      "group_actions": 0, "group_sizes": {}, "group_released_bytes": 0,
                      "group_requests_demoted": 0,
                      "group_search_ticks": 0, "group_search_evaluated": 0,
                      "group_search_truncated": 0,
                      "group_index_updates": 0, "group_index_ticks": 0,
                      "group_index_evaluated": 0, "group_index_truncated": 0,
                      "group_index_fallbacks": 0,
                      "slack_reclaim_actions": 0, "slack_reclaim_bytes": 0,
                      "low_water_slack_actions": 0, "low_water_slack_bytes": 0,
                      "broker_rejected_promotions": 0,
                      "future_plan_ticks": 0,
                      "future_prefill_ticks": 0, "future_prefill_selected": 0,
                      "future_prefill_reserve_bytes": 0,
                      "future_prefill_effective_headroom_bytes": 0,
                      "future_prefill_deferred": 0,
                      "future_prefill_missing_hint": 0,
                      "future_prefill_invalid_hint": 0,
                      "future_prefill_unestimated": 0,
                      "future_prefill_capped": 0,
                      "future_prefill_tp_fallbacks": 0,
                      "future_prefill_pressure_observations": 0,
                      "future_decode_pressure_observations": 0,
                      "predictive_prewarm_hook_errors": 0,
                      "predictive_prewarm_expire_errors": 0,
                      "last_future_plan_boundary": None,
                      "runtime_value_hints": 0,
                      "runtime_value_invalid": 0,
                      "runtime_value_guarded": 0,
                      "prefix_reuse_donors": 0,
                      "prefix_reuse_active_consumers": 0,
                      "prefix_reuse_future_consumers": 0,
                       "promotion_candidates": 0,
                       "promotion_value_hints": 0,
                       "promotion_value_invalid": 0,
                       "deadline_guarded_promotions": 0,
                       "last_promotion_runtime_values": {},
                      "deadline_last_resort_actions": 0,
                      "migration_bandwidth_bytes_per_s": self.migration_bandwidth.snapshot(),
                      "migration_bandwidth_samples": self.migration_bandwidth.sample_counts(),
                      "migration_effective_byte_cap": {
                          direction: self.migration_byte_cap(direction)
                          for direction in MigrationBandwidthEstimator.DIRECTIONS
                      },
                      "migration_time_cap_ms": migration_time_cap_ms,
                      "time_cap_rejections": 0,
                      "time_cap_rejections_demotion": 0,
                      "time_cap_rejections_promotion": 0,
                      "global_broker_observations": 0,
                      "global_broker_pressure_observations": 0,
                      "global_broker_fair_pressure_observations": 0,
                      "global_broker_min_available_bytes": None,
                      "global_broker_max_outstanding_bytes": 0,
                      "global_broker_min_capacity_bytes": None,
                      "budget_ownership": None,
                      "budget_ownership_error": None,
                      "last_demotion_runtime_values": {},
                      "min_live_bytes": None, "max_live_bytes": 0,
                      "min_reclaimable_bytes": None, "max_reclaimable_bytes": 0}
        self.steps = 0
        self.forced = False
        self.last_change = float("-inf")
        self.broker_low_since = None

    def publish(self):
        if self.stats_path is not None:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.stats_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.stats, indent=2), encoding="utf-8")
            temporary.replace(self.stats_path)

    def observe_budget_ownership(self, coordinator, tracker, broker_snapshot):
        """Publish a read-only view of the existing budget owners."""

        try:
            request_broker = getattr(
                coordinator, "_ekv_residency_broker", None
            )
            request_snapshot = (
                request_broker.snapshot()
                if request_broker is not None
                and callable(getattr(request_broker, "snapshot", None))
                else None
            )
            residency_stats = getattr(
                coordinator, "_ekv_residency_stats", {}
            )
            request_mode = (
                residency_stats.get("resident_broker_mode", "off")
                if isinstance(residency_stats, dict)
                else "off"
            )
            live = len(tracker.refs) * tracker.chunk_bytes
            backing = {
                "backed_bytes": tracker.backing.backed_bytes,
                "live_bytes": live,
                "reclaimable_bytes": (
                    tracker.backing.backed_bytes - live
                ),
                "mapped_chunks": len(tracker.refs),
            }
            self.stats["budget_ownership"] = (
                build_budget_ownership_snapshot(
                    physical_broker_snapshot=broker_snapshot,
                    request_growth_snapshot=request_snapshot,
                    request_growth_mode=request_mode,
                    vmm_backing_snapshot=backing,
                    activation_counters=getattr(tracker, "counters", None),
                )
            )
            self.stats["budget_ownership_error"] = None
        except Exception as exc:
            self.stats["budget_ownership_error"] = type(exc).__name__

    def migration_byte_cap(self, direction="demotion"):
        return self.migration_bandwidth.effective_byte_cap(
            self.byte_cap, self.migration_time_cap_ms, direction
        )

    def _refresh_migration_telemetry(self):
        self.stats["migration_bandwidth_bytes_per_s"] = (
            self.migration_bandwidth.snapshot()
        )
        self.stats["migration_bandwidth_samples"] = (
            self.migration_bandwidth.sample_counts()
        )
        self.stats["migration_effective_byte_cap"] = {
            direction: self.migration_byte_cap(direction)
            for direction in MigrationBandwidthEstimator.DIRECTIONS
        }

    def record(self, direction, reason, started, transfer, actual):
        elapsed = time.monotonic() - started
        self.stats[direction + "s"] += 1
        self.stats["migration_seconds"] += elapsed
        self.stats["transfer_bytes"] += transfer
        self.stats["released_bytes" if direction == "demotion" else "growth_bytes"] += actual
        reasons = self.stats["reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
        self.migration_bandwidth.observe(direction, transfer, elapsed)
        self._refresh_migration_telemetry()
        self.publish()

    def observe_residency(self, batch, resident, now):
        current = {}
        for request in batch.reqs:
            state = request.kv.req_pool_idx in resident
            previous = self.residency_since.get(request.rid)
            current[request.rid] = (state, previous[1] if previous and previous[0] == state else now)
        self.residency_since = current

    def transfer_cost(self, request, pool):
        return request.kv.kv_committed_len * pool.layer_num * (
            pool.bytes_per_token_k + pool.bytes_per_token_v)

    def eligible(self, request, pool, now, forced, direction="demotion"):
        if forced:
            return True
        transfer = self.transfer_cost(request, pool)
        if transfer > self.byte_cap:
            self.stats["rejected_cap"] += 1
            return False
        effective_cap = self.migration_byte_cap(direction)
        if transfer > effective_cap:
            self.stats["rejected_cap"] += 1
            self.stats["time_cap_rejections"] += 1
            self.stats["time_cap_rejections_" + direction] += 1
            return False
        if now - self.residency_since[request.rid][1] < self.dwell:
            self.stats["rejected_dwell"] += 1
            return False
        return True

    def observe_backing(self, tracker):
        live = len(tracker.refs) * tracker.chunk_bytes
        reclaimable = tracker.backing.backed_bytes - live
        for name, value in (("live_bytes", live), ("reclaimable_bytes", reclaimable)):
            low, high = "min_" + name, "max_" + name
            previous = self.stats[low]
            self.stats[low] = value if previous is None else min(previous, value)
            self.stats[high] = max(self.stats[high], value)
        return reclaimable

    def observe_broker(self, states):
        """Record the tightest aggregate VMM state across tensor-parallel ranks."""

        if not self.broker_headroom_bytes:
            return None
        capacities = [int(item[5]) for item in states]
        available = [int(item[4]) for item in states]
        if any(capacity < 1 for capacity in capacities):
            raise RuntimeError("invalid global VMM broker capacity")
        minimum_available = min(available)
        maximum_capacity = max(capacities)
        self.stats["global_broker_observations"] += 1
        previous = self.stats["global_broker_min_available_bytes"]
        self.stats["global_broker_min_available_bytes"] = (
            minimum_available if previous is None else min(previous, minimum_available)
        )
        previous_capacity = self.stats["global_broker_min_capacity_bytes"]
        self.stats["global_broker_min_capacity_bytes"] = (
            maximum_capacity
            if previous_capacity is None
            else min(previous_capacity, maximum_capacity)
        )
        self.stats["global_broker_max_outstanding_bytes"] = max(
            self.stats["global_broker_max_outstanding_bytes"],
            max(capacity - value for capacity, value in zip(capacities, available)),
        )
        fair_pressure = any(bool(item[6]) for item in states)
        if fair_pressure:
            self.stats["global_broker_fair_pressure_observations"] += 1
        return minimum_available, fair_pressure

    def _attach_prefix_reuse_values(
        self,
        request_values,
        requests,
        coordinator,
        future_prefix_demand=None,
    ):
        """Annotate donation value with verified live and imminent consumers."""

        demand = future_prefix_demand or {}
        registry = getattr(coordinator, "_ekv_shared_prefixes", None)
        for request in requests:
            rid = getattr(request, "rid", None)
            value = request_values.get(rid)
            row = getattr(getattr(request, "kv", None), "req_pool_idx", None)
            if value is None or not isinstance(row, int) or isinstance(row, bool):
                continue
            active_consumers = (
                registry.active_donor_consumers(row) if registry is not None else 0
            )
            future_consumers = 0
            if (
                registry is not None
                and value.prefix_group is not None
                and value.shared_prefix_tokens > 0
            ):
                future_consumers = demand.get(
                    (value.prefix_group, value.shared_prefix_tokens), 0
                )
                if future_consumers and not registry.prefers_geometry_donor(
                    row,
                    group=value.prefix_group,
                    tokens=value.shared_prefix_tokens,
                ):
                    future_consumers = 0
            if not active_consumers and not future_consumers:
                continue
            request_values[rid] = attach_prefix_reuse(
                value,
                PrefixReuseDemand(
                    tokens=value.shared_prefix_tokens,
                    active_consumers=active_consumers,
                    future_consumers=future_consumers,
                ),
            )
            self.stats["prefix_reuse_donors"] += 1
            self.stats["prefix_reuse_active_consumers"] += active_consumers
            self.stats["prefix_reuse_future_consumers"] += future_consumers
        return request_values

    def observe_future_prefill(self, plan, effective_headroom=0, *, tp_fallback=False,
                               boundary="prefill"):
        """Publish the latest bounded waiting-queue reservation snapshot."""

        if boundary not in ("prefill", "decode"):
            raise ValueError("future plan boundary must be prefill or decode")
        stats = self.stats
        stats["future_plan_ticks"] += 1
        stats["last_future_plan_boundary"] = boundary
        if boundary == "prefill":
            stats["future_prefill_ticks"] += 1
        stats["future_prefill_selected"] = plan.selected_count
        stats["future_prefill_reserve_bytes"] = plan.reserve_bytes
        stats["future_prefill_effective_headroom_bytes"] = int(effective_headroom)
        stats["future_prefill_deferred"] = plan.deferred_count
        stats["future_prefill_missing_hint"] = plan.missing_hint_count
        stats["future_prefill_invalid_hint"] = plan.invalid_hint_count
        stats["future_prefill_unestimated"] = plan.unestimated_count
        stats["future_prefill_capped"] = plan.capped_count
        if tp_fallback:
            stats["future_prefill_tp_fallbacks"] += 1
        self.publish()

    def trim_slack(self, coordinator, tracker, exchange, synchronize, reclaimable,
                   *, low_water=False):
        if not any(exchange(reclaimable)):
            return
        released = None
        error = None
        try:
            before = tracker.backing.backed_bytes
            coordinator.wait_for_pending_backup()
            synchronize()
            released = tracker.trim_quiescent()
            if released != before - tracker.backing.backed_bytes:
                released = None
        except Exception as exc:
            error = exc
        if any(type(value) is not int or value < 0 for value in exchange(released)):
            raise MigrationFatal("scheduler TP slack trim failed") from error
        if released:
            self.stats["slack_reclaim_actions"] += 1
            self.stats["slack_reclaim_bytes"] += released
            if low_water:
                self.stats["low_water_slack_actions"] += 1
                self.stats["low_water_slack_bytes"] += released
            self.publish()

    def boundary(self, batch, coordinator, exchange, synchronize, *, prefill, now,
                 future_headroom=0, future_prefix_demand=None):
        if (
            isinstance(future_headroom, bool)
            or not isinstance(future_headroom, int)
            or future_headroom < 0
        ):
            raise ValueError("future_headroom must be a nonnegative integer")
        tracker = coordinator.mem_pool_device.adaptive_vmm
        self.observe_residency(batch, coordinator._ekv_resident, now)
        if not prefill and not batch.is_empty():
            self.steps += 1
        force = (not prefill and not self.forced and self.force_step > 0
                 and self.steps >= self.force_step)
        force_promote = (not prefill and self.forced and not self.promoted and
                         self.promote_step > 0 and self.steps >= self.promote_step)
        backed = tracker.backing.backed_bytes
        self.stats["max_observed_backed_bytes"] = max(self.stats["max_observed_backed_bytes"], backed)
        if not prefill and not force and not force_promote and not future_headroom and (
                not (self.budget or self.broker_headroom_bytes)
                or self.steps % self.observe):
            return
        snapshot = _broker_snapshot(tracker)
        if self.broker_headroom_bytes and snapshot is None:
            raise RuntimeError("global broker pressure requires a broker-enabled VMM backing")
        self.observe_budget_ownership(coordinator, tracker, snapshot)
        local = (
            backed, force, now - self.last_change >= self.dwell, force_promote,
            None if snapshot is None else snapshot["available_bytes"],
            None if snapshot is None else snapshot["capacity_bytes"],
            None if snapshot is None else snapshot["fairness_pressure"],
        )
        states = exchange(local)
        if self.broker_headroom_bytes and any(
                item[4] is None or item[5] is None or item[6] is None
                for item in states):
            raise RuntimeError("global broker pressure must be configured on every TP rank")
        if self.broker_headroom_bytes:
            broker_available, broker_fair_pressure = self.observe_broker(states)
        else:
            broker_available, broker_fair_pressure = 0, False
        headroom = future_headroom
        if prefill:
            headroom = max(self.headroom, headroom)
        local_pressure = (
            self.budget > 0
            and max(item[0] for item in states) + headroom > self.budget
        )
        broker_pressure = (
            self.broker_headroom_bytes > 0
            and (
                broker_available < self.broker_headroom_bytes
                or broker_fair_pressure
            )
        )
        pressure = local_pressure or broker_pressure
        force = all(item[1] for item in states)
        reclaimable = self.observe_backing(tracker)
        peak = max(item[0] for item in states) + headroom
        if pressure:
            self.stats["pressure_observations"] += 1
            if broker_pressure:
                self.stats["global_broker_pressure_observations"] += 1
            if future_headroom > 0:
                key = (
                    "future_prefill_pressure_observations"
                    if prefill else "future_decode_pressure_observations"
                )
                self.stats[key] += 1
            self.low_since = None
            self.trim_slack(coordinator, tracker, exchange, synchronize, reclaimable)
            post_snapshot = _broker_snapshot(tracker)
            self.observe_budget_ownership(
                coordinator, tracker, post_snapshot
            )
            post_local = (
                tracker.backing.backed_bytes, local[1], local[2], local[3],
                None if post_snapshot is None else post_snapshot["available_bytes"],
                None if post_snapshot is None else post_snapshot["capacity_bytes"],
                None if post_snapshot is None else post_snapshot["fairness_pressure"],
            )
            states = exchange(post_local)
            if self.broker_headroom_bytes:
                broker_available, broker_fair_pressure = self.observe_broker(states)
            else:
                broker_available, broker_fair_pressure = 0, False
            peak = max(item[0] for item in states) + headroom
            local_pressure = self.budget > 0 and peak > self.budget
            broker_pressure = (
                self.broker_headroom_bytes > 0
                and (
                    broker_available < self.broker_headroom_bytes
                    or broker_fair_pressure
                )
            )
            pressure = local_pressure or broker_pressure
        if self.trim_low_water and self.low and not pressure and not force and peak >= self.low:
            live = len(tracker.refs) * tracker.chunk_bytes
            host_long = any(
                request.kv.req_pool_idx is not None
                and request.kv.req_pool_idx not in coordinator._ekv_resident
                and not request.finished() and not request.hisparse_staging
                and request.kv.kv_committed_len > coordinator.device_buffer_size
                and request.kv.kv_committed_len == request.kv.kv_allocated_len
                for request in batch.reqs
            )
            votes = exchange((host_long, live + headroom < self.low))
            if all(host and below for host, below in votes):
                self.trim_slack(coordinator, tracker, exchange, synchronize,
                                self.observe_backing(tracker), low_water=True)
                peak = max(exchange(tracker.backing.backed_bytes)) + headroom
        if not self.low or peak >= self.low:
            self.low_since = None
        elif self.low_since is None:
            self.low_since = now
        stable = self.low_since is not None and now - self.low_since >= self.stable
        if self.broker_headroom_bytes:
            broker_relaxed = (
                broker_available >= self.broker_low_bytes
                and not broker_fair_pressure
            )
            if broker_relaxed:
                if self.broker_low_since is None:
                    self.broker_low_since = now
            else:
                self.broker_low_since = None
            broker_stable = (
                self.broker_low_since is not None
                and now - self.broker_low_since >= self.stable
            )
            recovery_stable = broker_stable
        else:
            recovery_stable = True
        forced_up = all(item[3] for item in states)
        if not pressure and not force and (
                self.low or self.promote_step or self.broker_headroom_bytes):
            can_promote = all(exchange(
                recovery_stable and local[2]
                and (stable if self.low or self.promote_step else True)
            ))
            if forced_up or can_promote:
                self.promote(
                    batch,
                    coordinator,
                    exchange,
                    synchronize,
                    forced_up,
                    now,
                    future_prefix_demand=future_prefix_demand,
                )
            return
        if not force and not (
                pressure and all(item[2] for item in states)):
            return

        capacity = coordinator.padded_buffer_size
        page_size = coordinator.mem_pool_device.page_size
        resident = coordinator._ekv_resident
        requests = {}
        plans = {}
        for request in batch.reqs:
            kv = request.kv
            if (kv.req_pool_idx not in resident or request.finished() or request.hisparse_staging
                    or kv.kv_committed_len != kv.kv_allocated_len
                    or kv.kv_committed_len <= capacity
                    or int(coordinator.req_device_buffer_size[kv.req_pool_idx])
                    or int(coordinator.req_to_host_pool_allocated_len[kv.req_pool_idx])):
                continue
            if not self.eligible(request, coordinator.mem_pool_device, now, force):
                continue
            logical = coordinator.req_to_token_pool.req_to_token[
                kv.req_pool_idx, :kv.kv_committed_len:page_size]
            physical = coordinator.mem_pool_device.translate_loc_from_full_to_hisparse_device(logical)
            slots = physical.tolist()
            pages = [slot // page_size for slot in slots]
            if any(slot <= 0 or slot % page_size for slot in slots) or len(set(pages)) != len(pages):
                continue
            retained_pages, released_pages = plan_demote_pages(tracker, pages, capacity // page_size)
            unique_pages = set(released_pages)
            counts = _chunk_release_counts(tracker, unique_pages)
            chunks = _releasable_counts(tracker, counts)
            requests[request.rid] = (request, unique_pages, counts, chunks)
            if self.capture_plans:
                plans[kv.req_pool_idx] = {"page_ids": pages, "retained": retained_pages,
                                          "released": released_pages,
                                          "predicted_chunks": sorted(chunks)}

        request_values, value_stats = runtime_values(batch.reqs, now_s=now)
        request_values = self._attach_prefix_reuse_values(
            request_values,
            batch.reqs,
            coordinator,
            future_prefix_demand,
        )
        self.stats["runtime_value_hints"] += value_stats["hinted"]
        self.stats["runtime_value_invalid"] += value_stats["invalid"]
        self.stats["runtime_value_guarded"] += sum(
            value.deadline_guarded(self.deadline_guard_ms)
            for value in request_values.values()
        )

        def cost(rid):
            return self.transfer_cost(requests[rid][0], coordinator.mem_pool_device)

        metrics = {rid: (len(chunks) * tracker.chunk_bytes, cost(rid))
                   for rid, (_, _, _, chunks) in requests.items() if chunks or force}
        ranking_values = exchange({
            rid: merge_demotion_values(
                (request_values[rid],), guard_ms=self.deadline_guard_ms
            )
            for rid in metrics
        })
        ranked = _rank_demotions(
            exchange(metrics), ranking_values,
            deadline_guard_ms=self.deadline_guard_ms,
        )
        if not ranked and not force:
            self.capture_plan(plans, tracker, page_size)
        if ranked:
            chosen = (ranked[0],)
            chunks = requests[chosen[0]][3]
        elif self.max_group > 1 and not force and len(requests) > 1:
            index_updates = self.demand_index.sync(requests)
            self.stats["group_index_updates"] += index_updates
            self.stats["group_index_ticks"] += 1
            effective_byte_cap = self.migration_byte_cap("demotion")

            def observe_group_cap(transfer):
                if transfer > self.byte_cap:
                    return
                if effective_byte_cap < self.byte_cap:
                    self.stats["time_cap_rejections"] += 1
                    self.stats["time_cap_rejections_demotion"] += 1

            (groups, group_chunks, examined, fallback_examined, index_truncated,
             fallback_truncated, rejected_by_cap,
             used_fallback) = _demand_index_demotion_groups(
                self.demand_index, tracker, requests, cost,
                max_group=self.max_group, byte_cap=effective_byte_cap,
                search_limit=self.group_search_limit,
                rotation=self.stats["group_search_ticks"],
                cap_observer=observe_group_cap,
            )
            self.stats["group_search_ticks"] += 1
            self.stats["group_search_evaluated"] += examined + fallback_examined
            self.stats["group_search_truncated"] += int(
                index_truncated or fallback_truncated
            )
            self.stats["group_index_evaluated"] += examined
            self.stats["group_index_truncated"] += int(index_truncated)
            self.stats["rejected_cap"] += rejected_by_cap
            self.stats["group_index_fallbacks"] += int(used_fallback)
            group_values = {
                group: merge_demotion_values(
                    (request_values[rid] for rid in group),
                    guard_ms=self.deadline_guard_ms,
                )
                for group in groups
            }
            ranked_groups = _rank_demotions(
                exchange(groups), exchange(group_values),
                deadline_guard_ms=self.deadline_guard_ms,
            )
            chosen = ranked_groups[0] if ranked_groups else ()
            chunks = group_chunks.get(chosen, set())
        else:
            chosen = ()
        if not chosen:
            if pressure:
                self.stats["no_reclaimable_candidate"] += 1
            return
        self.demote_group(chosen, requests, chunks, coordinator, tracker, exchange,
                          synchronize, force, prefill, sum(cost(rid) for rid in chosen),
                          runtime_values=request_values)

    def capture_plan(self, plans, tracker, page_size):
        if (not self.capture_plans or self.stats_path is None or
                len(self.plan_snapshots) >= 3 or len(plans) < 2):
            return
        key = tuple((row, tuple(plan["page_ids"])) for row, plan in sorted(plans.items()))
        if key in self.plan_keys:
            return
        self.plan_keys.add(key)
        request_pages = {page for plan in plans.values() for page in plan["page_ids"]}
        pairs = [
            {"rows": [a, b], "predicted_chunks": sorted(_releasable(
                tracker, [*plans[a]["released"], *plans[b]["released"]]))}
            for a, b in combinations(sorted(plans), 2)
        ]
        self.plan_snapshots.append({
            "chunk_bytes": tracker.chunk_bytes, "page_size": page_size,
            "planes": tracker.planes, "refs": dict(tracker.refs),
            "mapped_chunks": sorted(tracker.backing.mapped_chunks),
            "other_live_pages": sorted(tracker.pages - request_pages),
            "requests": plans, "pairs": pairs,
        })
        output = self.stats_path.with_name("plans-" + self.stats_path.name)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.plan_snapshots, indent=2), encoding="utf-8")
        temporary.replace(output)

    def demote_group(self, chosen, requests, chunks, coordinator, tracker, exchange,
                     synchronize, force, prefill, transfer, runtime_values=None):
        started = time.monotonic()
        prepared = []
        error = None
        try:
            for rid in chosen:
                prepared.append(prepare_demote(coordinator, requests[rid][0], synchronize))
        except Exception as exc:
            error = exc
        if not all(exchange(len(prepared) == len(chosen))):
            for item in reversed(prepared):
                abort_demote(item)
            raise MigrationFatal("scheduler TP demotion preparation failed") from error

        rows = []
        released = None
        try:
            before = tracker.backing.backed_bytes
            for item in prepared:
                commit_demote(item, rows.append, synchronize)
            if chunks & tracker.refs.keys():
                raise MigrationFatal("predicted released chunks remain owned")
            synchronize()
            released = tracker.backing.reclaim(chunks, synchronize=False) * tracker.chunk_bytes
            if released != before - tracker.backing.backed_bytes or (
                    not force and released <= 0):
                released = None
        except Exception as exc:
            error = exc
        releases = exchange(released)
        if any(type(value) is not int or value < 0 for value in releases):
            raise MigrationFatal("scheduler TP demotion commit failed") from error
        coordinator._ekv_resident.difference_update(rows)
        release_broker = getattr(coordinator, "_ekv_broker_release_request", None)
        if release_broker is not None:
            for rid in chosen:
                release_broker(requests[rid][0])
        self.last_change = time.monotonic()
        self.forced |= force
        self.low_since = None
        for rid in chosen:
            self.residency_since[rid] = (False, self.last_change)
        reason = "forced_decode" if force else ("prefill_pressure" if prefill else "decode_pressure")
        selected_values = {
            rid: runtime_values.get(rid) for rid in chosen
        } if runtime_values is not None else {}
        self.stats["last_demotion_runtime_values"] = {
            rid: runtime_value_evidence(value, guard_ms=self.deadline_guard_ms)
            for rid, value in selected_values.items()
            if value is not None
        }
        if any(
            value is not None and value.deadline_guarded(self.deadline_guard_ms)
            for value in selected_values.values()
        ):
            self.stats["deadline_last_resort_actions"] += 1
            reason += "_deadline_last_resort"
        if len(chosen) > 1:
            self.stats["group_actions"] += 1
            self.stats["group_requests_demoted"] += len(chosen)
            self.stats["group_released_bytes"] += released
            sizes = self.stats["group_sizes"]
            size = str(len(chosen))
            sizes[size] = sizes.get(size, 0) + 1
            self.stats["demotions"] += len(chosen) - 1
            reason += "_group"
        self.record("demotion", reason, started, transfer, released)
        LOG.info("Adaptive demotion: boundary=%s forced=%s group_size=%d released_bytes_by_rank=%s",
                 "prefill" if prefill else "decode", force, len(chosen), releases)

    def promote(
        self,
        batch,
        coordinator,
        exchange,
        synchronize,
        forced,
        now,
        *,
        future_prefix_demand=None,
    ):
        pool = coordinator.mem_pool_device
        tracker = pool.adaptive_vmm
        physical = coordinator.token_to_kv_pool_allocator.hisparse_attn_allocator
        resident = coordinator._ekv_resident
        requests = {}
        for request in batch.reqs:
            kv = request.kv
            row, length = kv.req_pool_idx, kv.kv_committed_len
            if (row is None or row in resident or request.finished() or request.hisparse_staging
                    or length != kv.kv_allocated_len or length <= coordinator.device_buffer_size
                    or int(coordinator.req_device_buffer_size[row]) != coordinator.padded_buffer_size):
                continue
            if not self.eligible(
                request, pool, now, forced, direction="promotion"
            ):
                continue
            pages = -(-length // pool.page_size)
            available_tokens = physical.available_size()
            can_promote = getattr(coordinator, "_ekv_broker_can_promote", None)
            if can_promote is not None and not can_promote(request, available_tokens):
                self.stats["broker_rejected_promotions"] += 1
                continue
            # Upper bound includes transient coexistence with the source hot buffer.
            if len(physical.free_pages) >= pages:
                available_pages = physical.free_pages[:pages]
            else:
                available_pages = physical.get_all_free_pages().sort().values[:pages]
            missing = set().union(*(tracker.chunks(page) for page in available_pages.tolist()))
            missing.difference_update(tracker.backing.mapped_chunks)
            ceiling = self.budget if forced else self.low
            if available_tokens < pages * pool.page_size or (
                    ceiling and tracker.backing.backed_bytes + len(missing) * tracker.chunk_bytes > ceiling):
                continue
            requests[request.rid] = request
        if not requests:
            return
        request_values, value_stats = runtime_values(requests.values(), now_s=now)
        request_values = self._attach_prefix_reuse_values(
            request_values,
            requests.values(),
            coordinator,
            future_prefix_demand,
        )
        costs = {
            rid: self.transfer_cost(req, pool) for rid, req in requests.items()
        }
        ranking_values = exchange({
            rid: merge_demotion_values(
                (request_values[rid],), guard_ms=self.deadline_guard_ms
            )
            for rid in requests
        })
        ranked = rank_promotions(
            exchange(costs),
            ranking_values,
            guard_ms=self.deadline_guard_ms,
        )
        if not ranked:
            return
        request = requests[ranked[0]]
        selected_value = request_values[request.rid]
        self.stats["promotion_candidates"] += len(ranked)
        self.stats["promotion_value_hints"] += value_stats["hinted"]
        self.stats["promotion_value_invalid"] += value_stats["invalid"]
        self.stats["deadline_guarded_promotions"] += int(
            selected_value.deadline_guarded(self.deadline_guard_ms)
        )
        self.stats["last_promotion_runtime_values"] = {
            request.rid: runtime_value_evidence(
                selected_value, guard_ms=self.deadline_guard_ms
            )
        }
        started = time.monotonic()
        broker_plan = None
        plan_broker = getattr(coordinator, "_ekv_broker_promotion_plan", None)
        if plan_broker is not None:
            try:
                broker_plan = plan_broker(request)
            except Exception as exc:
                raise MigrationFatal("scheduler TP broker planning failed") from exc
            plans = exchange(broker_plan)
            if not plans or any(plan != plans[0] for plan in plans):
                raise MigrationFatal("scheduler TP broker plan mismatch")
            if (not isinstance(broker_plan, tuple) or len(broker_plan) != 4
                    or any(type(value) is not int or value < 0 for value in broker_plan)):
                raise MigrationFatal("scheduler TP broker plan is malformed")
        prepared = None
        error = None
        try:
            prepared = prepare_promote(coordinator, request, synchronize)
        except Exception as exc:
            error = exc
        if not all(exchange(prepared is not None)):
            if prepared is not None:
                abort_promote(prepared)
            raise MigrationFatal("scheduler TP promotion preparation failed") from error
        growth = None
        try:
            self.stats["max_observed_backed_bytes"] = max(
                self.stats["max_observed_backed_bytes"], tracker.backing.backed_bytes)
            growth = commit_promote(prepared, resident.add, synchronize)
        except Exception as exc:
            error = exc
        if any(type(value) is not int or value < 0 for value in exchange(growth)):
            raise MigrationFatal("scheduler TP promotion commit failed") from error
        commit_broker = getattr(coordinator, "_ekv_broker_commit_promotion", None)
        if commit_broker is not None:
            if broker_plan is None:
                raise MigrationFatal("scheduler TP broker plan missing before commit")
            commit_broker(request, broker_plan[2])
        self.last_change = time.monotonic()
        self.low_since = None
        self.promoted |= forced
        self.residency_since[request.rid] = (True, self.last_change)
        self.record("promotion", "forced_decode" if forced else "low_water_stable",
                    started, request.kv.kv_committed_len * pool.layer_num *
                    (pool.bytes_per_token_k + pool.bytes_per_token_v), growth)


def install_scheduler_budget():
    """Install before scheduler construction, after the residency hook."""
    budget = int(os.environ.get("EKV_ADAPTIVE_BUDGET_BYTES", "0"))
    headroom = int(os.environ.get("EKV_ADAPTIVE_PREFILL_HEADROOM_BYTES", "0"))
    force = int(os.environ.get("EKV_ADAPTIVE_FORCE_DECODE_STEP", "0"))
    dwell = float(os.environ.get("EKV_ADAPTIVE_DWELL_SECONDS", "1"))
    low = int(os.environ.get("EKV_ADAPTIVE_LOW_BYTES", "0"))
    stable = float(os.environ.get("EKV_ADAPTIVE_LOW_STABLE_SECONDS", "2"))
    observe = int(os.environ.get("EKV_ADAPTIVE_OBSERVE_STEPS", "16"))
    promote = int(os.environ.get("EKV_ADAPTIVE_FORCE_PROMOTE_STEP", "0"))
    stats_dir = os.environ.get("EKV_ADAPTIVE_STATS_DIR")
    cap = int(os.environ.get("EKV_ADAPTIVE_MIGRATION_BYTE_CAP", str(512 << 20)))
    time_cap_ms = int(os.environ.get(
        "EKV_ADAPTIVE_MIGRATION_TIME_CAP_MS", "0"
    ))
    broker_headroom = int(os.environ.get(
        "EKV_ADAPTIVE_GLOBAL_BROKER_HEADROOM_BYTES", "0"
    ))
    broker_low = int(os.environ.get(
        "EKV_ADAPTIVE_GLOBAL_BROKER_LOW_BYTES", "0"
    ))
    initial_bandwidth_mib_s = float(os.environ.get(
        "EKV_ADAPTIVE_INITIAL_BANDWIDTH_MIB_S", "1024"
    ))
    bandwidth_ewma_alpha = float(os.environ.get(
        "EKV_ADAPTIVE_BANDWIDTH_EWMA_ALPHA", "0.25"
    ))
    max_group = int(os.environ.get("EKV_ADAPTIVE_MAX_GROUP", "1"))
    group_search_limit = int(os.environ.get(
        "EKV_ADAPTIVE_GROUP_SEARCH_LIMIT", "256"
    ))
    deadline_guard_ms = int(os.environ.get(
        "EKV_ADAPTIVE_DEADLINE_GUARD_MS", str(DEFAULT_DEADLINE_GUARD_MS)
    ))
    capture_plans = os.environ.get("EKV_ADAPTIVE_PLAN_CAPTURE") == "1"
    trim_low_water = os.environ.get("EKV_ADAPTIVE_TRIM_LOW_WATER") == "1"
    shared_prefix_direct = os.environ.get("EKV_SHARED_PREFIX_REUSE") == "direct"
    prewarm_lead_raw = os.environ.get("EKV_PREDICTIVE_PREWARM_TTL_MS")
    prewarm_lead_ms = (
        None
        if prewarm_lead_raw in (None, "")
        else int(prewarm_lead_raw)
    )
    if prewarm_lead_ms is not None and prewarm_lead_ms < 0:
        raise ValueError("predictive prewarm TTL must be nonnegative")
    future_mode = os.environ.get("EKV_FUTURE_PREFILL_MODE", "off")
    kind_activation_raw = json.loads(os.environ.get(
        "EKV_FUTURE_PREFILL_KIND_ACTIVATION_JSON", "{}"
    ))
    if not isinstance(kind_activation_raw, dict):
        raise ValueError("EKV_FUTURE_PREFILL_KIND_ACTIVATION_JSON must be an object")
    future_config = FuturePrefillConfig(
        mode=future_mode,
        horizon_ms=int(os.environ.get("EKV_FUTURE_PREFILL_HORIZON_MS", "2000")),
        max_reserve_bytes=int(os.environ.get(
            "EKV_FUTURE_PREFILL_MAX_BYTES", str(256 << 20)
        )),
        activation_bytes_per_token=int(os.environ.get(
            "EKV_FUTURE_PREFILL_ACTIVATION_BYTES_PER_TOKEN", "0"
        )),
        kind_activation_bytes_per_token=tuple(kind_activation_raw.items()),
        auto_resident_output_tokens=int(os.environ.get(
            "EKV_FUTURE_PREFILL_AUTO_RESIDENT_TOKENS", "128"
        )),
    )
    if (not budget and not broker_headroom and not force and not promote and not time_cap_ms
            and future_mode == "off"):
        return
    if (min(budget, broker_headroom, broker_low, headroom, force, dwell, low, stable, promote) < 0
            or observe < 1 or cap < 1 or group_search_limit < 1
            or deadline_guard_ms < 0
            or time_cap_ms < 0 or not math.isfinite(initial_bandwidth_mib_s)
            or initial_bandwidth_mib_s <= 0
            or not 0 < bandwidth_ewma_alpha <= 1
            or max_group not in (1, 2, 3, 4)
            or bool(broker_headroom) != bool(broker_low)
            or (broker_headroom and broker_low <= broker_headroom)
            or (low and low >= budget) or (promote and (not force or promote <= force))):
        raise ValueError("adaptive scheduler settings must be nonnegative")
    if future_mode == "enforce" and budget <= 0:
        raise ValueError("future prefill enforce mode requires a positive adaptive budget")
    import torch
    from sglang.srt.managers import scheduler
    from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
    from sglang.srt.runtime_context import (
        get_disagg,
        get_schedule,
        get_server_args,
        get_spec,
    )

    source_sha256 = hashlib.sha256(Path(scheduler.__file__).read_bytes()).hexdigest()
    if source_sha256 not in SCHEDULER_SHA256S:
        raise RuntimeError(f"unaudited scheduler budget source: {source_sha256}")
    cls = scheduler.Scheduler
    if getattr(cls, "_ekv_adaptive_budget_installed", False):
        return

    def run(self, batch, prefill):
        now = time.monotonic()
        coordinator = self.hisparse_coordinator
        tracker = (
            None
            if coordinator is None
            else getattr(coordinator.mem_pool_device, "adaptive_vmm", None)
        )
        _best_effort_expire_predictive_prewarm(tracker)
        if not hasattr(self, "_ekv_adaptive_budget"):
            if (coordinator is None or not coordinator.is_m3_hisparse
                    or not hasattr(coordinator, "_ekv_resident")
                    or not hasattr(coordinator.mem_pool_device, "adaptive_vmm")
                    or not cuda_graph_fully_disabled()
                    or not get_schedule().disable_overlap_schedule
                    or not get_server_args().disable_radix_cache
                    or get_disagg().disaggregation_mode != "null"
                    or get_spec().speculative_algorithm not in (None, "NONE")):
                raise RuntimeError("adaptive scheduler requires eager colocated MiniMax VMM residency")
            from sglang.srt.distributed.parallel_state import (
                get_tensor_model_parallel_rank,
            )

            stats_path = (Path(stats_dir) / f"rank{get_tensor_model_parallel_rank():02d}.json"
                          if stats_dir else None)
            self._ekv_adaptive_budget = SchedulerBudget(
                budget, headroom, dwell, force, low, stable, observe, promote, stats_path, cap, max_group, capture_plans,
                trim_low_water, group_search_limit, deadline_guard_ms,
                time_cap_ms, int(initial_bandwidth_mib_s * (1 << 20)),
                bandwidth_ewma_alpha, broker_headroom, broker_low)

        def exchange(value):
            if coordinator.tp_world_size == 1:
                return (value,)
            replies = [None] * coordinator.tp_world_size
            torch.distributed.all_gather_object(replies, value, group=coordinator.tp_group)
            return replies

        reservation = empty_plan(future_mode)
        tp_fallback = False
        controller = self._ekv_adaptive_budget
        should_plan_future = _should_plan_future_waiting(
            future_mode, self.waiting_queue, prefill=prefill,
            steps=controller.steps, observe=observe,
        )
        if should_plan_future:
            pool = coordinator.mem_pool_device
            local_config = replace(
                future_config, sparse_hot_tokens=coordinator.device_buffer_size
            )
            reservation = plan_future_prefill(
                self.waiting_queue,
                now_s=time.perf_counter(),
                kv_bytes_per_token=pool.layer_num * (
                    pool.bytes_per_token_k + pool.bytes_per_token_v
                ),
                config=local_config,
            )
            if coordinator.tp_world_size > 1:
                signatures = exchange(reservation.signature)
                reservation, tp_fallback = reconcile_tp_plan(reservation, signatures)
        future_headroom = effective_prefill_headroom(
            reservation, headroom, future_mode
        )
        if should_plan_future:
            controller.observe_future_prefill(
                reservation, future_headroom, tp_fallback=tp_fallback,
                boundary="prefill" if prefill else "decode",
            )
        future_prefix_demand = (
            forecast_prefix_reuse((
                *self.waiting_queue,
                *((self.chunked_req,) if self.chunked_req is not None else ()),
            ))
            if shared_prefix_direct and (self.waiting_queue or self.chunked_req is not None)
            else {}
        )
        controller.boundary(
            batch, coordinator, exchange,
            lambda: torch.cuda.synchronize(coordinator.mem_pool_device.device),
            prefill=prefill, now=now, future_headroom=future_headroom,
            future_prefix_demand=future_prefix_demand,
        )
        if should_plan_future and not tp_fallback:
            _best_effort_predictive_prewarm(
                tracker,
                reservation,
                controller,
                lead_ms=prewarm_lead_ms,
            )

    prefill = cls.get_new_batch_prefill
    decode = cls.update_running_batch

    @wraps(prefill)
    def before_prefill(self, running_batch):
        if (future_mode != "off" or budget or broker_headroom) and (
                self.waiting_queue or self.chunked_req is not None):
            run(self, running_batch, True)
        return prefill(self, running_batch)

    @wraps(decode)
    def before_decode(self, batch):
        run(self, batch, False)
        return decode(self, batch)

    idle = cls.on_idle

    @wraps(idle)
    def on_idle(self):
        controller = getattr(self, "_ekv_adaptive_budget", None)
        coordinator = getattr(self, "hisparse_coordinator", None)
        tracker = (
            None
            if coordinator is None
            else getattr(coordinator.mem_pool_device, "adaptive_vmm", None)
        )
        result = idle(self)
        if self.is_fully_idle():
            _best_effort_expire_predictive_prewarm(tracker, controller)
            if controller is not None:
                signature = tuple(value for value in controller.stats.values() if not isinstance(value, dict))
                if getattr(controller, "_published", None) != signature:
                    controller.publish()
                    controller._published = signature
        return result

    cls.on_idle = on_idle
    cls.get_new_batch_prefill = before_prefill
    cls.update_running_batch = before_decode
    cls._ekv_adaptive_budget_installed = True
