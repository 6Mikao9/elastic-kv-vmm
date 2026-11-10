"""Enumerate exact-budget capacity allocations using an anonymized locality summary."""

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import re


CANDIDATES = (2560, 3072, 3584, 4096, 6144, 8192)
BASE_CAPACITY = 4096


def request_curves(summary):
    """Sum the selected-token denominators and simulated misses across layers."""
    curves = defaultdict(lambda: defaultdict(Counter))
    seen = set()
    feasibility = summary["selection_limits"]["capacities"]
    available = tuple(cap for cap in CANDIDATES
                      if feasibility.get(str(cap), {}).get("status") == "feasible")
    for row in summary["rank_request_layer_after_first_16"]:
        rank, request, layer = row["rank"], row["request"], row["layer"]
        if type(rank) is not int or rank < 0 or not re.fullmatch(r"R[0-9]+", request):
            raise ValueError("invalid rank or anonymous request alias")
        key = rank, request, layer
        if key in seen:
            raise ValueError("duplicate request-layer summary")
        seen.add(key)
        selected = row.get("selected", 0)
        if type(selected) is not int or selected < 0:
            raise ValueError("invalid selection denominator")
        if not selected:
            continue
        curve = curves[rank][request]
        curve["selected"] += selected
        for capacity in available:
            misses = row.get(f"simulated_misses_{capacity}")
            if type(misses) is not int or not 0 <= misses <= selected:
                raise ValueError(f"invalid simulated miss count at capacity {capacity}")
            curve[capacity] += misses

    if not curves:
        raise ValueError("summary contains no post-warmup selections")
    requests = sorted(next(iter(curves.values())))
    if any(set(rank_curves) != set(requests) for rank_curves in curves.values()):
        raise ValueError("ranks have different post-warmup request sets")
    if not available:
        raise ValueError("no candidate capacity has a feasible replay")
    return curves, requests, available


def compare_allocations(summary, budgets=None):
    curves, requests, available = request_curves(summary)
    budgets = tuple(budgets) if budgets is not None else (len(requests) * BASE_CAPACITY,)
    if not budgets or any(type(budget) is not int or budget <= 0 for budget in budgets):
        raise ValueError("budgets must be positive integers")
    candidates = {budget: [] for budget in budgets}
    for allocation in product(available, repeat=len(requests)):
        budget = sum(allocation)
        if budget in candidates:
            candidates[budget].append(allocation)

    def compare_group(group, equal_request_weight=False):
        selected = sum(group[request]["selected"] for request in requests)

        def misses(allocation):
            return sum(group[request][capacity]
                       for request, capacity in zip(requests, allocation))

        def request_rates(allocation):
            return {request: Fraction(group[request][capacity], group[request]["selected"])
                    for request, capacity in zip(requests, allocation)}

        def objective(allocation):
            return (sum(request_rates(allocation).values()) / len(requests)
                    if equal_request_weight else Fraction(misses(allocation), selected))

        def describe(allocation):
            count = misses(allocation)
            rates = request_rates(allocation)
            return {"allocation": dict(zip(requests, allocation)),
                    "simulated_misses": count, "weighted_simulated_miss_rate": count / selected,
                    "mean_per_request_simulated_miss_rate": float(sum(rates.values()) / len(requests)),
                    "per_request_simulated_miss_rate": {request: float(rate)
                                                        for request, rate in rates.items()}}

        comparisons = []
        for budget, allocations in sorted(candidates.items()):
            comparison = {"total_token_budget": budget, "exact_allocations": len(allocations),
                          "uniform": None, "best": None, "worst": None,
                          "best_improvement_over_uniform_pp": None,
                          "best_objective_improvement_over_uniform_pp": None}
            if allocations:
                best = min(allocations, key=objective)
                worst = max(allocations, key=objective)
                comparison.update(best=describe(best), worst=describe(worst))
                uniform, remainder = divmod(budget, len(requests))
                if remainder == 0 and uniform in available:
                    uniform_allocation = (uniform,) * len(requests)
                    comparison["uniform"] = describe(uniform_allocation)
                    comparison["best_improvement_over_uniform_pp"] = (
                        100 * (misses(uniform_allocation) - misses(best)) / selected
                    )
                    comparison["best_objective_improvement_over_uniform_pp"] = float(
                        100 * (objective(uniform_allocation) - objective(best))
                    )
            comparisons.append(comparison)
        return {
            "objective": ("mean_per_request_simulated_miss_rate" if equal_request_weight
                          else "total_simulated_misses_over_total_selected"),
            "selected_tokens": selected,
            "request_selected_tokens": {request: group[request]["selected"] for request in requests},
            "comparisons": comparisons,
        }

    rank_results = [dict(rank=rank) | compare_group(rank_curves)
                    for rank, rank_curves in sorted(curves.items())]
    joint_curves = defaultdict(Counter)
    for rank_curves in curves.values():
        for request in requests:
            joint_curves[request].update(rank_curves[request])
    joint = {
        "sampled_ranks": sorted(curves),
        "unsampled_tp4_ranks": sorted(set(range(4)) - curves.keys()),
        "scope": "One allocation shared across sampled ranks only. This does not establish a TP4 optimum.",
        "weighted": compare_group(joint_curves),
        "mean_per_request": compare_group(joint_curves, equal_request_weight=True),
    }
    return {
        "method": "hindsight_exact_token_budget_simulation",
        "interpretation": [
            "These are simulated batch-token LRU miss counts summed across sampled layers.",
            "Serial request traces are combined as a hypothetical concurrent allocation.",
            "This is hindsight allocation, not an online policy or evidence of performance or physical memory reclaim.",
            "The separate rank oracles may choose different allocations. The joint oracle uses one shared allocation.",
            "The request-mean objective first pools each request across sampled ranks, then weights requests equally.",
        ],
        "input_capture_coverage": summary.get("coverage"),
        "warmup_steps_excluded": summary["warmup_steps"],
        "request_aliases": requests,
        "requested_candidates": CANDIDATES,
        "feasible_candidates": available,
        "rank_results": rank_results,
        "sampled_rank_joint_oracle": joint,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="anonymized locality summary.json")
    parser.add_argument("--budgets", type=int, nargs="+",
                        help="exact total token budgets, default 4096 times the eligible requests")
    args = parser.parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = compare_allocations(summary, args.budgets)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
