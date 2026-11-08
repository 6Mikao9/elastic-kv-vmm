"""MiniMax eager scheduler integration for budget-triggered demotion."""

from collections import Counter
from dataclasses import replace
from functools import wraps
from fractions import Fraction
import hashlib
from itertools import combinations
import logging
import json
import os
from pathlib import Path
import time

from .adaptive_runtime import MigrationFatal
from .migration import (
    abort_demote, abort_promote, commit_demote, commit_promote, plan_demote_pages, prepare_demote, prepare_promote,
)
from .future_prefill import (
    FuturePrefillConfig, effective_prefill_headroom, empty_plan,
    plan_future_prefill, reconcile_tp_plan,
)


SCHEDULER_SHA256S = frozenset({
    # Pinned MiniMax development image used by the request-level experiments.
    "fc96ed4b680fdf55621a5397f844cfd4cbd30213f44eec8a888b7cce75cafe60",
    # lmsysorg/sglang:latest with SGLang 0.5.17 (audited wrapper boundary).
    "1d4a7683e32f0862cf5b03be33e5ec23f9d112e59546d391e28fa222fcd1619b",
})
LOG = logging.getLogger(__name__)


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


def _rank_demotions(rank_candidates):
    common = set.intersection(*(set(rank) for rank in rank_candidates))
    return sorted(common, key=lambda rid: (
        -Fraction(min(rank[rid][0] for rank in rank_candidates),
                  max(1, max(rank[rid][1] for rank in rank_candidates))), rid))


class SchedulerBudget:
    def __init__(self, budget, headroom, dwell, force_step, low=0, stable=2,
                 observe=16, promote_step=0, stats_path=None, byte_cap=512 << 20, max_group=1, capture_plans=False, trim_low_water=False):
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
        self.capture_plans = capture_plans
        self.trim_low_water = trim_low_water
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
                      "slack_reclaim_actions": 0, "slack_reclaim_bytes": 0,
                      "low_water_slack_actions": 0, "low_water_slack_bytes": 0,
                      "broker_rejected_promotions": 0,
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
                      "min_live_bytes": None, "max_live_bytes": 0,
                      "min_reclaimable_bytes": None, "max_reclaimable_bytes": 0}
        self.steps = 0
        self.forced = False
        self.last_change = float("-inf")

    def publish(self):
        if self.stats_path is not None:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.stats_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.stats, indent=2), encoding="utf-8")
            temporary.replace(self.stats_path)

    def record(self, direction, reason, started, transfer, actual):
        self.stats[direction + "s"] += 1
        self.stats["migration_seconds"] += time.monotonic() - started
        self.stats["transfer_bytes"] += transfer
        self.stats["released_bytes" if direction == "demotion" else "growth_bytes"] += actual
        reasons = self.stats["reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
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

    def eligible(self, request, pool, now, forced):
        if forced:
            return True
        if self.transfer_cost(request, pool) > self.byte_cap:
            self.stats["rejected_cap"] += 1
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

    def observe_future_prefill(self, plan, effective_headroom=0, *, tp_fallback=False):
        """Publish the latest bounded waiting-queue reservation snapshot."""

        stats = self.stats
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
                 future_headroom=0):
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
        if not prefill and not force and not force_promote and (
                not self.budget or self.steps % self.observe):
            return
        local = (backed, force, now - self.last_change >= self.dwell, force_promote)
        states = exchange(local)
        headroom = max(self.headroom, int(future_headroom)) if prefill else 0
        pressure = self.budget > 0 and max(item[0] for item in states) + headroom > self.budget
        force = all(item[1] for item in states)
        reclaimable = self.observe_backing(tracker)
        peak = max(item[0] for item in states) + headroom
        if pressure:
            self.stats["pressure_observations"] += 1
            if future_headroom > 0:
                self.stats["future_prefill_pressure_observations"] += 1
            self.low_since = None
            self.trim_slack(coordinator, tracker, exchange, synchronize, reclaimable)
            peak = max(exchange(tracker.backing.backed_bytes)) + headroom
            pressure = peak > self.budget
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
        forced_up = all(item[3] for item in states)
        if not pressure and not force and (self.low or self.promote_step):
            can_promote = all(exchange(stable and local[2]))
            if forced_up or can_promote:
                self.promote(batch, coordinator, exchange, synchronize, forced_up, now)
            return
        if not force and not (
                pressure and all(item[2] for item in states)):
            return

        capacity = coordinator.padded_buffer_size
        page_size = coordinator.mem_pool_device.page_size
        resident = coordinator._ekv_resident
        requests = {}
        plans = {}
        release_counts = {}
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
            requests[request.rid] = (request, unique_pages, chunks)
            release_counts[request.rid] = counts
            if self.capture_plans:
                plans[kv.req_pool_idx] = {"page_ids": pages, "retained": retained_pages,
                                          "released": released_pages,
                                          "predicted_chunks": sorted(chunks)}

        def cost(rid):
            return self.transfer_cost(requests[rid][0], coordinator.mem_pool_device)

        metrics = {rid: (len(chunks) * tracker.chunk_bytes, cost(rid))
                   for rid, (_, _, chunks) in requests.items() if chunks or force}
        ranked = _rank_demotions(exchange(metrics))
        if not ranked and not force:
            self.capture_plan(plans, tracker, page_size)
        if ranked:
            chosen = (ranked[0],)
            chunks = requests[chosen[0]][2]
        elif self.max_group == 2 and not force:
            pair_chunks = {}
            pairs = {}
            for pair in combinations(sorted(requests), 2):
                transfer = sum(cost(rid) for rid in pair)
                if transfer > self.byte_cap:
                    self.stats["rejected_cap"] += 1
                    continue
                left, right = pair
                released = _releasable_pair(
                    tracker, requests[left][1], release_counts[left],
                    requests[right][1], release_counts[right],
                )
                if released:
                    pairs[pair] = (len(released) * tracker.chunk_bytes, transfer)
                    pair_chunks[pair] = released
            ranked_pairs = _rank_demotions(exchange(pairs))
            chosen = ranked_pairs[0] if ranked_pairs else ()
            chunks = pair_chunks.get(chosen, set())
        else:
            chosen = ()
        if not chosen:
            if pressure:
                self.stats["no_reclaimable_candidate"] += 1
            return
        self.demote_group(chosen, requests, chunks, coordinator, tracker, exchange,
                          synchronize, force, prefill, sum(cost(rid) for rid in chosen))

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
                     synchronize, force, prefill, transfer):
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

    def promote(self, batch, coordinator, exchange, synchronize, forced, now):
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
            if not self.eligible(request, pool, now, forced):
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
        candidates = set(requests)
        for rank_ids in exchange(tuple(sorted(requests))):
            candidates.intersection_update(rank_ids)
        if not candidates:
            return
        request = requests[min(candidates)]
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
    max_group = int(os.environ.get("EKV_ADAPTIVE_MAX_GROUP", "1"))
    capture_plans = os.environ.get("EKV_ADAPTIVE_PLAN_CAPTURE") == "1"
    trim_low_water = os.environ.get("EKV_ADAPTIVE_TRIM_LOW_WATER") == "1"
    future_mode = os.environ.get("EKV_FUTURE_PREFILL_MODE", "off")
    future_config = FuturePrefillConfig(
        mode=future_mode,
        horizon_ms=int(os.environ.get("EKV_FUTURE_PREFILL_HORIZON_MS", "2000")),
        max_reserve_bytes=int(os.environ.get(
            "EKV_FUTURE_PREFILL_MAX_BYTES", str(256 << 20)
        )),
        activation_bytes_per_token=int(os.environ.get(
            "EKV_FUTURE_PREFILL_ACTIVATION_BYTES_PER_TOKEN", "0"
        )),
        auto_resident_output_tokens=int(os.environ.get(
            "EKV_FUTURE_PREFILL_AUTO_RESIDENT_TOKENS", "128"
        )),
    )
    if not budget and not force and not promote and future_mode == "off":
        return
    if (min(budget, headroom, force, dwell, low, stable, promote) < 0 or observe < 1 or cap < 1 or max_group not in (1, 2)
            or (low and low >= budget) or (promote and (not force or promote <= force))):
        raise ValueError("adaptive scheduler settings must be nonnegative")
    if future_mode == "enforce" and budget <= 0:
        raise ValueError("future prefill enforce mode requires a positive adaptive budget")
    import torch
    from sglang.srt.managers import scheduler
    from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
    from sglang.srt.runtime_context import get_disagg, get_schedule, get_server_args, get_spec

    source_sha256 = hashlib.sha256(Path(scheduler.__file__).read_bytes()).hexdigest()
    if source_sha256 not in SCHEDULER_SHA256S:
        raise RuntimeError(f"unaudited scheduler budget source: {source_sha256}")
    cls = scheduler.Scheduler
    if getattr(cls, "_ekv_adaptive_budget_installed", False):
        return

    def run(self, batch, prefill):
        now = time.monotonic()
        coordinator = self.hisparse_coordinator
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
            from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank

            stats_path = (Path(stats_dir) / f"rank{get_tensor_model_parallel_rank():02d}.json"
                          if stats_dir else None)
            self._ekv_adaptive_budget = SchedulerBudget(
                budget, headroom, dwell, force, low, stable, observe, promote, stats_path, cap, max_group, capture_plans,
                trim_low_water)

        def exchange(value):
            if coordinator.tp_world_size == 1:
                return (value,)
            replies = [None] * coordinator.tp_world_size
            torch.distributed.all_gather_object(replies, value, group=coordinator.tp_group)
            return replies

        reservation = empty_plan(future_mode)
        tp_fallback = False
        if prefill and future_mode != "off":
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
        if prefill and future_mode != "off":
            self._ekv_adaptive_budget.observe_future_prefill(
                reservation, future_headroom, tp_fallback=tp_fallback
            )
        self._ekv_adaptive_budget.boundary(
            batch, coordinator, exchange,
            lambda: torch.cuda.synchronize(coordinator.mem_pool_device.device),
            prefill=prefill, now=now, future_headroom=future_headroom,
        )

    prefill = cls.get_new_batch_prefill
    decode = cls.update_running_batch

    @wraps(prefill)
    def before_prefill(self, running_batch):
        if (future_mode != "off" or budget) and (
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
        if controller is not None and self.is_fully_idle():
            signature = tuple(value for value in controller.stats.values() if not isinstance(value, dict))
            if getattr(controller, "_published", None) != signature:
                controller.publish()
                controller._published = signature
        return idle(self)

    cls.on_idle = on_idle
    cls.get_new_batch_prefill = before_prefill
    cls.update_running_batch = before_decode
    cls._ekv_adaptive_budget_installed = True
