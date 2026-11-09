"""Scheduler idle integration for the stock MiniMax HiSparse VMM arm."""

import hashlib
import json
import os
from functools import wraps
from pathlib import Path
from time import monotonic_ns, perf_counter_ns

from .budget_ownership import build_budget_ownership_snapshot
from .minimax_vmm import install_minimax_vmm


_SCHEDULER_HASH = "fc96ed4b680fdf55621a5397f844cfd4cbd30213f44eec8a888b7cce75cafe60"


def attach_budget_ownership(stats, *, tracker, tracker_stats):
    """Attach read-only ownership telemetry without failing an idle tick."""

    try:
        residency = stats.get("request_residency", {})
        stats["budget_ownership"] = build_budget_ownership_snapshot(
            physical_broker_snapshot=tracker.backing.metrics.get(
                "global_broker"
            ),
            request_growth_snapshot=residency.get("resident_broker"),
            request_growth_mode=residency.get("resident_broker_mode", "off"),
            vmm_backing_snapshot=tracker_stats,
            activation_counters=getattr(tracker, "counters", None),
        )
    except Exception as exc:
        stats["budget_ownership_error"] = type(exc).__name__
    return stats


class IdleGrace:
    """Track a single deferred trim per uninterrupted idle interval."""

    def __init__(self, milliseconds):
        self.milliseconds = milliseconds
        self.deadline_ns = None
        self.deferred_trims = 0

    def reset(self):
        self.deadline_ns = None

    def ready(self, now):
        if not self.milliseconds:
            return True
        if self.deadline_ns is None:
            self.deadline_ns = now + self.milliseconds * 1_000_000
            self.deferred_trims += 1
        return now >= self.deadline_ns


def install_vmm_runtime(stats_dir):
    """Install in each scheduler process before constructing its worker."""
    from sglang.srt.managers import scheduler

    if getattr(scheduler.Scheduler, "_adaptive_vmm_idle_installed", False):
        return
    if hashlib.sha256(Path(scheduler.__file__).read_bytes()).hexdigest() != _SCHEDULER_HASH:
        raise RuntimeError("Unaudited MiniMax VMM scheduler")
    grace_ms = int(os.environ.get("EKV_VMM_IDLE_GRACE_MS", "0"))
    if grace_ms < 0:
        raise ValueError("EKV_VMM_IDLE_GRACE_MS must be nonnegative")
    install_minimax_vmm()
    stats_dir = Path(stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)
    original_idle = scheduler.Scheduler.on_idle
    original_run_batch = scheduler.Scheduler.run_batch
    original_sleep = scheduler.Scheduler.maybe_sleep_on_idle

    @wraps(original_run_batch)
    def run_batch(self, *args, **kwargs):
        grace = getattr(self, "_adaptive_vmm_grace", None)
        if grace is not None:
            grace.reset()
        return original_run_batch(self, *args, **kwargs)

    @wraps(original_sleep)
    def maybe_sleep_on_idle(self):
        grace = getattr(self, "_adaptive_vmm_grace", None)
        sleeper = self.idle_sleeper
        if grace is not None and grace.deadline_ns is not None and sleeper is not None:
            remaining = grace.deadline_ns - monotonic_ns()
            if remaining > 0:
                timeout = min(1000, (remaining + 999_999) // 1_000_000)
                if hasattr(sleeper, "poller"):
                    sleeper.poller.poll(timeout)
                    return
                if hasattr(sleeper, "rust_server"):
                    sleeper.rust_server.wait_request(timeout)
                    return
            else:
                # Return to the scheduler loop to reconsider trim immediately.
                return
        return original_sleep(self)

    @wraps(original_idle)
    def on_idle(self):
        allocator = self.token_to_kv_pool_allocator
        pool = getattr(allocator.get_kvcache(), "main_pool", None)
        tracker = getattr(pool, "adaptive_vmm", None)
        grace = getattr(self, "_adaptive_vmm_grace", None)
        eligible = tracker is not None and self.is_fully_idle() and len(tracker.pages) == 1
        if not eligible and grace is not None:
            grace.reset()
        if eligible:
            if grace is None:
                grace = self._adaptive_vmm_grace = IdleGrace(grace_ms)
            coordinator = self.hisparse_coordinator
            if coordinator is None:
                raise RuntimeError("Missing MiniMax HiSparse coordinator")
            if not coordinator.has_ongoing_staging():
                if tracker.stats()["reclaimable_bytes"]:
                    if grace.ready(monotonic_ns()):
                        coordinator.wait_for_pending_backup()
                        tracker.trim_quiescent()
                        grace.reset()
                else:
                    grace.reset()
                parallel = scheduler.get_parallel()
                tracker_stats = tracker.stats()
                stats = {"tp_rank": parallel.tp_rank, "tp_size": parallel.tp_size,
                         "idle": True, **tracker_stats, **tracker.backing.metrics,
                         "idle_grace_ms": grace.milliseconds,
                         "deferred_idle_trims": grace.deferred_trims,
                         "timestamp_monotonic_ns": perf_counter_ns()}
                if hasattr(coordinator, "_ekv_residency_stats"):
                    stats["request_residency"] = dict(coordinator._ekv_residency_stats)
                attach_budget_ownership(
                    stats,
                    tracker=tracker,
                    tracker_stats=tracker_stats,
                )
                signature = {key: value for key, value in stats.items()
                             if key != "timestamp_monotonic_ns"}
                if signature != getattr(self, "_adaptive_vmm_last_stats", None):
                    target = stats_dir / f"tp-{parallel.tp_rank}.json"
                    temp = target.with_suffix(".tmp")
                    temp.write_text(json.dumps(stats, allow_nan=False) + "\n", encoding="utf-8")
                    temp.replace(target)
                    self._adaptive_vmm_last_stats = signature
        return original_idle(self)

    if grace_ms:
        scheduler.Scheduler.run_batch = run_batch
        scheduler.Scheduler.maybe_sleep_on_idle = maybe_sleep_on_idle
    scheduler.Scheduler.on_idle = on_idle
    scheduler.Scheduler._adaptive_vmm_idle_installed = True
