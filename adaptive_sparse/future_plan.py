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
from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from .future_residency import RESIDENCE_HINTS


POLICIES = ("fifo", "longest", "mode-aware", "future")


class ResidencyMode(str, Enum):
    """A request's memory policy for its whole lifetime."""

    RESIDENT = "resident"
    HI_SPARSE = "hi-sparse"


@dataclass(frozen=True, slots=True)
class RequestProfile:
    """Known request metadata available in a controlled rollout.

    ``prefix_group`` and ``shared_prefix_tokens`` describe reusable content.
    The initial planner reports their availability but does not yet deduct
    cross-request shared backing; that requires physical page identity from the
    live VMM tracker and is handled at runtime.
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
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.arrival_ms < 0:
            raise ValueError(f"{self.request_id}: arrival_ms must be non-negative")
        if self.prompt_tokens < 1 or self.output_tokens < 1:
            raise ValueError(f"{self.request_id}: prompt and output must be positive")
        if not math.isfinite(self.priority) or not 0 < self.priority <= 1e6:
            raise ValueError(
                f"{self.request_id}: priority must be finite and in (0, 1e6]"
            )
        if not isinstance(self.kind, str) or not self.kind or len(self.kind) > 64:
            raise ValueError(
                f"{self.request_id}: kind must be a short non-empty string"
            )
        if self.deadline_ms is not None and self.deadline_ms < self.arrival_ms:
            raise ValueError(f"{self.request_id}: deadline precedes arrival")
        if self.shared_prefix_tokens < 0 or self.shared_prefix_tokens > self.prompt_tokens:
            raise ValueError(f"{self.request_id}: invalid shared_prefix_tokens")
        if self.shared_prefix_tokens and not self.prefix_group:
            raise ValueError(f"{self.request_id}: prefix_group is required for reuse")
        if self.residence_hint not in RESIDENCE_HINTS:
            raise ValueError(f"{self.request_id}: invalid residence_hint")

    @property
    def sequence_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


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
    auto_resident_output_threshold: int = 1

    def __post_init__(self) -> None:
        positive_ints = {
            "memory_budget_bytes": self.memory_budget_bytes,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "sparse_hot_tokens": self.sparse_hot_tokens,
            "max_active_requests": self.max_active_requests,
            "auto_resident_output_threshold": self.auto_resident_output_threshold,
        }
        for name, value in positive_ints.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.prefill_workspace_bytes_per_token < 0:
            raise ValueError("prefill_workspace_bytes_per_token must be non-negative")
        for name, value in (
            ("prefill_tokens_per_ms", self.prefill_tokens_per_ms),
            ("resident_decode_tokens_per_ms", self.resident_decode_tokens_per_ms),
            ("sparse_decode_tokens_per_ms", self.sparse_decode_tokens_per_ms),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.lookahead_ms < 0:
            raise ValueError("lookahead_ms must be non-negative")


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

    @property
    def throughput_requests_per_second(self) -> float:
        return 0.0 if self.makespan_ms == 0 else 1000.0 * len(self.admissions) / self.makespan_ms

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "peak_memory_bytes": self.peak_memory_bytes,
            "peak_backed_bytes": self.peak_backed_bytes,
            "makespan_ms": self.makespan_ms,
            "weighted_completion_ms": self.weighted_completion_ms,
            "deadline_misses": self.deadline_misses,
            "sparse_requests": self.sparse_requests,
            "mode_switches": self.mode_switches,
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
    hint: dict[str, object]


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

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.simulation.policy,
            "simulation": self.simulation.as_dict(),
            "submissions": [
                {
                    "request_id": item.request_id,
                    "planned_arrival_ms": item.planned_arrival_ms,
                    "submit_at_ms": item.submit_at_ms,
                    "estimated_finish_ms": item.estimated_finish_ms,
                    "kind": item.kind,
                    "hint": item.hint,
                }
                for item in self.submissions
            ],
        }


@dataclass(slots=True)
class _Active:
    request: RequestProfile
    admission: Admission


def _ceil_duration(tokens: int, rate: float) -> int:
    return max(1, math.ceil(tokens / rate))


def _hold_bytes(request: RequestProfile, mode: ResidencyMode, config: PlannerConfig) -> int:
    tokens = request.sequence_tokens
    if mode is ResidencyMode.HI_SPARSE:
        tokens = min(tokens, config.sparse_hot_tokens)
    return tokens * config.kv_bytes_per_token


def _duration_ms(request: RequestProfile, mode: ResidencyMode, config: PlannerConfig) -> int:
    prefill = _ceil_duration(request.prompt_tokens, config.prefill_tokens_per_ms)
    decode_rate = (
        config.resident_decode_tokens_per_ms
        if mode is ResidencyMode.RESIDENT
        else config.sparse_decode_tokens_per_ms
    )
    return prefill + _ceil_duration(request.output_tokens, decode_rate)


def _urgency(request: RequestProfile, now_ms: int, config: PlannerConfig) -> float:
    if request.deadline_ms is None:
        return 0.0
    slack = request.deadline_ms - now_ms - _duration_ms(request, ResidencyMode.RESIDENT, config)
    horizon = max(1, config.lookahead_ms or 1)
    if slack < 0:
        return 1.0 + min(1.0, -slack / horizon)
    return 1.0 / (1.0 + slack / horizon)


def _future_contention(
    request: RequestProfile,
    mode: ResidencyMode,
    active_bytes: int,
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
    finish = now_ms + _duration_ms(request, mode, config)
    spend = active_bytes + _hold_bytes(request, mode, config)
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
        required += item.prompt_tokens * config.prefill_workspace_bytes_per_token
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
    extra_hold_bytes: int = 0,
    extra_active_requests: int = 0,
) -> tuple[ResidencyMode, str] | None:
    if extra_hold_bytes < 0 or extra_active_requests < 0:
        raise ValueError("extra active state must be nonnegative")
    if len(active) + extra_active_requests > config.max_active_requests:
        return None
    active_bytes = (
        sum(item.admission.resident_kv_bytes for item in active)
        + extra_hold_bytes
    )
    workspace = request.prompt_tokens * config.prefill_workspace_bytes_per_token
    native_hold = _hold_bytes(request, ResidencyMode.HI_SPARSE, config)
    if request.sequence_tokens <= config.sparse_hot_tokens:
        if active_bytes + workspace + native_hold <= config.memory_budget_bytes:
            return ResidencyMode.HI_SPARSE, "fits-native-hot-set"
        return None
    if request.residence_hint == "resident":
        order = (ResidencyMode.RESIDENT, ResidencyMode.HI_SPARSE)
    elif request.residence_hint == "host_backed":
        order = (ResidencyMode.HI_SPARSE,)
    elif request.output_tokens < config.auto_resident_output_threshold:
        order = (ResidencyMode.HI_SPARSE,)
    else:
        order = (preferred, ResidencyMode.HI_SPARSE if preferred is ResidencyMode.RESIDENT else ResidencyMode.RESIDENT)
    for mode in order:
        hold = _hold_bytes(request, mode, config)
        if active_bytes + workspace + hold <= config.memory_budget_bytes:
            if request.residence_hint == "resident":
                reason = (
                    "hint-resident"
                    if mode is ResidencyMode.RESIDENT
                    else "hint-resident-falls-back"
                )
            elif request.residence_hint == "host_backed":
                reason = "hint-requires-bounded-hot-set"
            elif request.output_tokens < config.auto_resident_output_threshold:
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

    current_prefill = _ceil_duration(choice.prompt_tokens, config.prefill_tokens_per_ms)
    held_choice = _hold_bytes(choice, mode, config)
    choice_duration = _duration_ms(choice, mode, config)
    for item in future:
        if item.arrival_ms > now_ms + config.lookahead_ms:
            break
        if item.deadline_ms is None:
            continue
        standalone = _feasible_mode(item, active, config)
        if standalone is None:
            continue
        future_mode = standalone[0]
        future_duration = _duration_ms(item, future_mode, config)
        future_prefill = _ceil_duration(item.prompt_tokens, config.prefill_tokens_per_ms)
        blocked_start = max(item.arrival_ms, now_ms + current_prefill)
        delayed_finish = blocked_start + future_duration

        with_choice = _feasible_mode(
            item,
            active,
            config,
            extra_hold_bytes=held_choice,
            extra_active_requests=1,
        )
        memory_blocked = with_choice is None
        deadline_blocked = (
            delayed_finish > item.deadline_ms
            and item.arrival_ms + future_duration <= item.deadline_ms
        )
        # Idle only to protect a hard latency objective.  Merely having a
        # higher-priority request in the horizon should not leave otherwise
        # useful prefill capacity idle: native priority handles that case when
        # the request actually arrives.
        if not (memory_blocked or deadline_blocked):
            continue

        if choice.deadline_ms is not None:
            delayed_choice_finish = (
                item.arrival_ms + future_prefill + choice_duration
            )
            if delayed_choice_finish > choice.deadline_ms:
                continue
            if item.deadline_ms > choice.deadline_ms:
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
    active_bytes = sum(item.admission.resident_kv_bytes for item in active)
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
                -_urgency(item, now_ms, config),
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
        sparse_spare = _hold_bytes(choice, ResidencyMode.HI_SPARSE, config)
        workspace = choice.prompt_tokens * config.prefill_workspace_bytes_per_token
        if (choice.residence_hint == "auto"
                and choice.output_tokens >= config.auto_resident_output_threshold
                and first[0] is ResidencyMode.RESIDENT
                and active_bytes + sparse_spare + workspace <= config.memory_budget_bytes):
            modes.append((ResidencyMode.HI_SPARSE, "lookahead-protects-imminent-demand"))
        for mode, reason in modes:
            displaced = _future_contention(choice, mode, active_bytes, pending, future, now_ms, config)
            urgency = _urgency(choice, now_ms, config)
            # Priority and deadlines decide whether work is worth reserving.
            # Sparse placement is cheaper for following requests, while its
            # decode slowdown remains an explicit cost in the simulator.
            score = 100.0 * choice.priority + 60.0 * urgency - 20.0 * displaced
            if mode is ResidencyMode.HI_SPARSE:
                slowdown = config.resident_decode_tokens_per_ms / config.sparse_decode_tokens_per_ms
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

    if policy not in POLICIES:
        raise ValueError(f"unknown planning policy: {policy!r}")
    workload = tuple(requests)
    if not workload:
        return PlanResult(policy, (), 0, 0, 0, 0.0, 0, 0, 0)
    ids = [item.request_id for item in workload]
    if len(set(ids)) != len(ids):
        raise ValueError("request IDs must be unique")
    for item in workload:
        sparse = _feasible_mode(item, (), config, preferred=ResidencyMode.HI_SPARSE)
        if sparse is None:
            raise ValueError(f"{item.request_id}: no residency mode fits the configured budget")

    unstarted = {item.request_id: item for item in workload}
    active: list[_Active] = []
    admissions: list[Admission] = []
    now_ms = min(item.arrival_ms for item in workload)
    prefill_busy_until = -1
    peak = 0

    while unstarted or active:
        active = [item for item in active if item.admission.finish_ms > now_ms]
        base = sum(item.admission.resident_kv_bytes for item in active)
        if not unstarted and not active:
            break
        if not unstarted:
            now_ms = min(item.admission.finish_ms for item in active)
            continue

        if prefill_busy_until > now_ms:
            candidates = [prefill_busy_until]
            candidates.extend(item.arrival_ms for item in unstarted.values() if item.arrival_ms > now_ms)
            candidates.extend(item.admission.finish_ms for item in active if item.admission.finish_ms > now_ms)
            now_ms = min(candidates)
            continue

        selected = _select(policy, unstarted, active, now_ms, config)
        if selected is not None:
            request, mode, reason = selected
            workspace = request.prompt_tokens * config.prefill_workspace_bytes_per_token
            hold = _hold_bytes(request, mode, config)
            duration = _duration_ms(request, mode, config)
            prefill = _ceil_duration(request.prompt_tokens, config.prefill_tokens_per_ms)
            admission = Admission(
                request_id=request.request_id,
                arrival_ms=request.arrival_ms,
                start_ms=now_ms,
                finish_ms=now_ms + duration,
                mode=mode,
                kind=request.kind,
                resident_kv_bytes=hold,
                prefill_workspace_bytes=workspace,
                reason=reason,
            )
            peak = max(peak, base + hold + workspace)
            unstarted.pop(request.request_id)
            active.append(_Active(request, admission))
            admissions.append(admission)
            prefill_busy_until = now_ms + prefill
            now_ms = prefill_busy_until
            continue

        future_times = [item.arrival_ms for item in unstarted.values() if item.arrival_ms > now_ms]
        finish_times = [item.admission.finish_ms for item in active if item.admission.finish_ms > now_ms]
        candidates = future_times + finish_times
        if not candidates:
            blocked = sorted(
                item.request_id for item in unstarted.values() if item.arrival_ms <= now_ms
            )
            raise RuntimeError(f"planning deadlock at {now_ms} ms for {blocked}")
        now_ms = min(candidates)

    if len(admissions) != len(workload):
        raise RuntimeError("planner did not admit every request")
    admission_by_id = {item.request_id: item for item in admissions}
    finish_by_id = {item.request_id: item.finish_ms for item in admissions}
    weight = sum(item.priority for item in workload)
    weighted_completion = sum(item.priority * finish_by_id[item.request_id] for item in workload) / weight
    misses = sum(
        item.deadline_ms is not None and admission_by_id[item.request_id].finish_ms > item.deadline_ms
        for item in workload
    )
    ordered = tuple(sorted(admissions, key=lambda item: (item.start_ms, item.request_id)))
    return PlanResult(
        policy=policy,
        admissions=ordered,
        peak_memory_bytes=peak,
        peak_backed_bytes=peak,
        makespan_ms=max(item.finish_ms for item in admissions),
        weighted_completion_ms=weighted_completion,
        deadline_misses=misses,
        sparse_requests=sum(item.mode is ResidencyMode.HI_SPARSE for item in admissions),
        mode_switches=0,
    )


def plan_rollout(
    requests: Iterable[RequestProfile],
    config: PlannerConfig,
    policy: str = "future",
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
    simulation = simulate(workload, config, policy)
    requests_by_id = {item.request_id: item for item in workload}
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
        }
        if request.deadline_ms is not None:
            hint["deadline_after_ms"] = max(
                0, request.deadline_ms - admission.start_ms
            )
        submissions.append(
            RolloutSubmission(
                request_id=request.request_id,
                planned_arrival_ms=admission.arrival_ms,
                submit_at_ms=admission.start_ms,
                estimated_finish_ms=admission.finish_ms,
                kind=request.kind,
                hint=hint,
            )
        )
    return RolloutPlan(simulation, tuple(submissions))


def compare(request_path: str | Path) -> dict[str, dict[str, object]]:
    """Load ``{config, requests}`` JSON and simulate every built-in policy."""

    payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
    config = PlannerConfig(**payload["config"])
    workload = [RequestProfile(**item) for item in payload["requests"]]
    return {policy: simulate(workload, config, policy).as_dict() for policy in POLICIES}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare deterministic future-aware admission policies"
    )
    parser.add_argument("workload", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--emit-policy", choices=POLICIES)
    args = parser.parse_args(argv)

    payload: dict[str, object] = compare(args.workload)
    if args.emit_policy is not None:
        raw = json.loads(args.workload.read_text(encoding="utf-8"))
        config = PlannerConfig(**raw["config"])
        workload = [RequestProfile(**item) for item in raw["requests"]]
        payload["rollout"] = plan_rollout(
            workload, config, args.emit_policy
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

