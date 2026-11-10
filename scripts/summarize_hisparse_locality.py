"""Publish anonymized aggregates from a MiniMax locality diagnostic run."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

from benchmarks.hisparse_locality_replay import DEFAULT_CAPACITIES, replay


PRIMARY_RANK = 0
WARMUP = 16
WORKLOADS = {"chatbot", "agent", "long_document"}
SUPPLEMENTARY_CAPACITY = 2176
COUNTS = ("steps", "selected", "native_prestate_misses", "turnover_selected", "turnover_new")


def jsonl(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def sim_field(capacity):
    return f"simulated_misses_{capacity}"


def metrics(counts, hot_capacity, capacities):
    selected = counts["selected"]
    simulated = {capacity: counts.get(sim_field(capacity)) for capacity in capacities}

    def delta(left, right):
        return None if left is None or right is None else fraction(100 * (left - right), selected)

    return dict(counts) | {
        "native_prestate_miss_rate": fraction(counts["native_prestate_misses"], selected),
        "simulated_minus_native_at_measured_capacity_pp": delta(
            simulated[hot_capacity], counts["native_prestate_misses"]),
        "turnover_rate": fraction(counts["turnover_new"], counts["turnover_selected"]),
        **{sim_field(capacity): misses for capacity, misses in simulated.items()},
        **{f"simulated_miss_rate_{capacity}": (
            None if misses is None else fraction(misses, selected))
           for capacity, misses in simulated.items()},
        **{f"simulated_miss_reduction_pp_{left}_to_{right}": delta(simulated[left], simulated[right])
           for left, right in zip(DEFAULT_CAPACITIES, DEFAULT_CAPACITIES[1:])},
    }


def supplementary_thresholds(counts, supplementary):
    """Return the smallest tested supplementary shrink capacity per hindsight bound."""
    selected = counts["selected"]
    baseline = counts.get(sim_field(4096))
    candidates = [(capacity, counts[sim_field(capacity)]) for capacity in sorted(supplementary)
                  if capacity < 4096 and sim_field(capacity) in counts]
    return {
        str(limit): next((capacity for capacity, misses in candidates
                          if selected and baseline is not None
                          and 100 * (misses - baseline) <= limit * selected), None)
        for limit in (1, 2, 5)
    }


def selection_limits(paths, capacities):
    max_selected = max_slots = max_valid_slots = access_rows = 0
    oversized = Counter()
    for path in paths:
        for row in jsonl(path):
            if row["type"] != "access":
                continue
            access_rows += 1
            selected = len(set(row["selected_tokens"]))
            blocks = row["selected_blocks"]
            max_selected = max(max_selected, selected)
            max_slots = max(max_slots, len(blocks))
            max_valid_slots = max(max_valid_slots, sum(
                0 <= block * row["block_size"] < row["seq_len"] for block in blocks))
            oversized.update(capacity for capacity in capacities if selected > capacity)
    return {
        "access_rows_all_ranks": access_rows, "max_unique_selected_tokens": max_selected,
        "max_block_slots": max_slots, "max_valid_block_slots": max_valid_slots,
        "capacities": {str(capacity): {
            "status": "unobserved" if not access_rows else
                      "infeasible" if oversized[capacity] else "feasible",
            "oversized_access_rows_all_ranks": oversized[capacity],
            "supplementary": capacity not in DEFAULT_CAPACITIES,
        } for capacity in capacities},
    }


def request_aliases(path):
    requests, origins = {}, {}
    for row in jsonl(path):
        rid = row["request_id"]
        workload = row["workload"]
        if rid in requests or workload not in WORKLOADS:
            raise ValueError("duplicate request or unknown workload in requests.jsonl")
        origin_key = workload, row["origin_id"]
        requests[rid] = {
            "request": f"R{len(requests) + 1:02d}",
            "origin_group": origins.setdefault(origin_key, f"G{len(origins) + 1:02d}"),
            "workload": workload, "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
        }
    return requests


def trace_steps(paths, requests, hot_capacity, steps, rank_keys, metadata, truncations, validation):
    previous = {}
    for path in paths:
        rows = iter(jsonl(path))
        header = next(rows)
        rank = header["rank"]
        if (header.get("type") != "metadata"
                or header.get("schema") != "minimax-hisparse-access-v1"
                or rank in metadata):
            raise ValueError("unexpected schema or duplicate capture file for a rank")
        metadata[rank] = set(header["layers"])
        for row in rows:
            if row["type"] == "truncation":
                reason = row["reason"]
                truncations[reason if reason in {"max_steps", "max_rows", "max_requests"}
                            else "other"] += 1
            if row["type"] != "access":
                continue
            request = requests[row["rid"]]
            layer, step, generation = row["layer"], row["step"], row["generation"]
            if (row["rank"] != rank or layer not in metadata[rank] or step < 0
                    or row["hot_capacity"] != hot_capacity):
                raise ValueError("capture row disagrees with rank, layer, step or capacity metadata")
            selected = list(dict.fromkeys(row["selected_tokens"]))
            hits = set(row["prestate_hit_tokens"])
            selection = set(selected)
            if (not hits <= selection
                    or any(type(token) is not int or not 0 <= token < row["seq_len"]
                           for token in selected)):
                raise ValueError("capture contains invalid selection or hit IDs")
            validation["duplicate_selection_occurrences"] += len(row["selected_tokens"]) - len(selected)
            key = request["request"], layer, step
            if key in rank_keys[rank]:
                raise ValueError("duplicate request, layer and step within a rank")
            fingerprint = hashlib.sha256(json.dumps(sorted(selection)).encode()).digest()
            rank_keys[rank][key] = fingerprint
            stream = rank, request["request"], layer, generation
            previous_step, previous_selection = previous.get(stream, (-2, set()))
            if step <= previous_step:
                raise ValueError("out-of-order capture step")
            contiguous = step == previous_step + 1
            previous[stream] = step, selection
            steps[(rank, *key)] = {
                "request": request["request"], "workload": request["workload"], "rank": rank,
                "layer": layer, "step": step, "steps": 1, "selected": len(selected),
                "native_prestate_misses": len(selected) - len(hits),
                "turnover_selected": len(selected) if contiguous else 0,
                "turnover_new": len(selection - previous_selection) if contiguous else 0,
            }
            yield {"request_id": request["request"], "rank": rank, "generation": generation,
                   "layer_id": layer, "step": step, "selected": selected}


def summarize(run, supplementary_capacities=()):
    if any(type(capacity) is not int or capacity <= 0 for capacity in supplementary_capacities):
        raise ValueError("supplementary capacities must be positive integers")
    supplementary = sorted({SUPPLEMENTARY_CAPACITY, *supplementary_capacities}
                           - set(DEFAULT_CAPACITIES))
    capacities = (*DEFAULT_CAPACITIES, *supplementary)
    requests = request_aliases(run / "requests.jsonl")
    state = json.loads((run / "results.json").read_text(encoding="utf-8"))
    hot_capacity = state["hisparse"]["device_buffer_size"]
    if state["hisparse"]["top_k"] != 2048 or hot_capacity not in DEFAULT_CAPACITIES:
        raise ValueError("unexpected configured HiSparse Top-K or capacity")
    steps, metadata = {}, {}
    rank_keys = defaultdict(dict)
    truncations, validation = Counter(), Counter()
    paths = sorted((run / "access").glob("hisparse-rank*.jsonl"))
    selection = selection_limits(paths, capacities)
    feasible = tuple(capacity for capacity in capacities
                     if selection["capacities"][str(capacity)]["status"] == "feasible")
    trace = trace_steps(paths, requests, hot_capacity, steps, rank_keys,
                        metadata, truncations, validation)
    if feasible:
        replayed = replay(trace, capacities=feasible, window_size=1)
        for row in replayed["windows"]:
            step = steps[row["rank"], row["request_id"], row["layer_id"], row["first_step"]]
            step.update({sim_field(capacity): row["capacities"][str(capacity)]["misses"]
                         for capacity in feasible})
    else:
        for _ in trace:
            pass

    phase_windows = defaultdict(Counter)
    totals = defaultdict(Counter)
    request_totals = defaultdict(Counter)
    rank_workload_totals = defaultdict(Counter)
    rank_request_layer_totals = defaultdict(Counter)
    count_fields = (*COUNTS, *(sim_field(capacity) for capacity in feasible))
    for row in steps.values():
        rank, request, layer, step = (row["rank"], row["request"], row["layer"], row["step"])
        counts = {field: row[field] for field in count_fields}
        for size in (16, 32):
            phase_windows[rank, request, row["workload"], layer, size, step // size * size].update(counts)
        layer_totals = rank_request_layer_totals[rank, request, row["workload"], layer]
        if step >= WARMUP:
            layer_totals.update(counts)
            for workload in ("all", row["workload"]):
                rank_workload_totals[rank, workload].update(counts)
        if rank == PRIMARY_RANK:
            if step >= WARMUP:
                request_totals[request].update(counts)
            periods = ("all", "first_16") if step < WARMUP else ("all", "after_first_16")
            for period in periods:
                for workload in ("all", row["workload"]):
                    totals[workload, period].update(counts)

    windows = [
        dict(rank=rank, request=request, workload=workload, layer=layer,
             window_steps=size, window_start_step=start) | metrics(counts, hot_capacity, capacities)
        for (rank, request, workload, layer, size, start), counts in sorted(phase_windows.items())
    ]
    comparisons = []
    primary = rank_keys[PRIMARY_RANK]
    for rank in sorted(set(metadata) - {PRIMARY_RANK}):
        other = rank_keys[rank]
        overlap = primary.keys() & other.keys()
        comparisons.append({
            "rank": rank, "matched_steps": len(overlap),
            "selection_mismatch_steps": sum(primary[key] != other[key] for key in overlap),
            "primary_only_steps": len(primary.keys() - other.keys()),
            "other_only_steps": len(other.keys() - primary.keys()),
        })

    verification = state.get("capture_verification", {})
    rank_coverage = []
    for rank in sorted(metadata):
        observed = defaultdict(set)
        for request, layer, step in rank_keys[rank]:
            observed[request, layer].add(step)
        for request in requests.values():
            expected = set(range(max(0, request["output_tokens"] - 1)))
            for layer in sorted(metadata[rank]):
                actual = observed[request["request"], layer]
                if expected != actual:
                    rank_coverage.append(dict(rank=rank, request=request["request"], layer=layer,
                                              missing_steps=len(expected - actual),
                                              extra_steps=len(actual - expected)))
    complete = (state.get("status") == "ok" and verification.get("request_rank_coverage_ok") is True
                and {0, 1} <= metadata.keys() and bool(primary) and not truncations
                and not verification.get("truncation_markers")
                and not rank_coverage and not any(
                    c["primary_only_steps"] or c["other_only_steps"] for c in comparisons))
    spread = []
    complete_windows = [row for row in windows if row["rank"] == PRIMARY_RANK
                        and row["window_steps"] == 16 and row["window_start_step"] >= WARMUP
                        and row["steps"] == 16 and row["selected"]]
    for workload in ("all", *sorted(WORKLOADS)):
        group = [row for row in complete_windows
                 if workload == "all" or row["workload"] == workload]
        for field in ("native_prestate_miss_rate", "turnover_rate",
                      *(f"simulated_miss_rate_{capacity}" for capacity in DEFAULT_CAPACITIES)):
            values = [row[field] for row in group if row[field] is not None]
            if values:
                spread.append(dict(workload=workload, metric=field, complete_windows=len(values),
                                   min=min(values), median=median(values), max=max(values)))
    per_request = []
    for request in requests.values():
        counts = request_totals[request["request"]]
        stats = metrics(counts, hot_capacity, capacities)
        shrink = stats["simulated_miss_reduction_pp_2048_to_4096"]
        per_request.append(request | stats | {
            "simulated_shrink_cost_pp_4096_to_2048": shrink,
            "simulated_expansion_benefit_pp_4096_to_8192":
                stats["simulated_miss_reduction_pp_4096_to_8192"],
            "offline_hindsight_shrink_cost_within_pp": {
                str(limit): (None if shrink is None else shrink <= limit) for limit in (1, 2, 5)
            },
            "offline_hindsight_min_supplementary_capacity_by_pp":
                supplementary_thresholds(counts, supplementary),
        })
    burst_windows = defaultdict(list)
    for row in complete_windows:
        burst_windows[row["request"], row["layer"]].append(row)
    bursts = []
    for (request, layer), group in sorted(burst_windows.items()):
        native_rates = [row["native_prestate_miss_rate"] for row in group]
        simulated_spread = {}
        for capacity in feasible:
            rates = [row[f"simulated_miss_rate_{capacity}"] for row in group]
            simulated_spread[str(capacity)] = {
                "min": min(rates), "max": max(rates), "range_pp": 100 * (max(rates) - min(rates)),
            }
        shrink_key = "simulated_miss_reduction_pp_2048_to_4096"
        shrink_windows = [row for row in group if row[shrink_key] is not None]
        worst = max(shrink_windows, key=lambda row: row[shrink_key]) if shrink_windows else None
        bursts.append(dict(
            request=request, layer=layer, complete_windows=len(group),
            native_prestate_miss_rate_min=min(native_rates),
            native_prestate_miss_rate_max=max(native_rates),
            native_prestate_miss_rate_range_pp=100 * (max(native_rates) - min(native_rates)),
            simulated_miss_rate_spread=simulated_spread,
            simulated_shrink_cost_max_pp=worst[shrink_key] if worst else None,
            simulated_shrink_cost_max_window_start_step=worst["window_start_step"] if worst else None,
        ))
    return {
        "coverage": "complete" if complete else "partial",
        "native_hot_capacity": hot_capacity, "configured_top_k": 2048, "primary_rank": PRIMARY_RANK,
        "warmup_steps": WARMUP, "request_counts": dict(Counter(r["workload"] for r in requests.values())),
        "requests": list(requests.values()),
        "selection_limits": selection,
        "capture": {"truncation_markers": dict(truncations), "rank_layer_coverage_issues": rank_coverage,
                    "rank_selection_comparison": comparisons,
                    "runner_request_rank_coverage_ok": verification.get("request_rank_coverage_ok"),
                    "primary_access_rows": len(primary), "validation": dict(validation)},
        "aggregates": [dict(workload=workload, period=period) | metrics(counts, hot_capacity, capacities)
                       for (workload, period), counts in sorted(totals.items())],
        "primary_phase_spread": spread,
        "primary_request_after_first_16": per_request,
        "rank_workload_after_first_16": [
            dict(rank=rank, workload=workload) | metrics(counts, hot_capacity, capacities) | {
                "offline_hindsight_min_supplementary_capacity_by_pp":
                    supplementary_thresholds(counts, supplementary),
            } for (rank, workload), counts in sorted(rank_workload_totals.items())
        ],
        "rank_request_layer_after_first_16": [
            dict(rank=rank, request=request, workload=workload, layer=layer)
            | metrics(counts, hot_capacity, capacities) | {
                "offline_hindsight_min_supplementary_capacity_by_pp":
                    supplementary_thresholds(counts, supplementary),
            } for (rank, request, workload, layer), counts in sorted(rank_request_layer_totals.items())
        ],
        "primary_request_layer_complete_16_step_bursts": bursts,
        "windows": windows,
    }


def percentage(rate):
    return "—" if rate is None else f"{100 * rate:.2f}%"


def points(value):
    return "—" if value is None else f"{value:.2f} pp"


def report(summary):
    counts = "、".join(f"{name} {summary['request_counts'].get(name, 0)} 条"
                       for name in sorted(WORKLOADS))
    limits = summary["selection_limits"]
    capacities = "、".join(
        f"{capacity} {info['status']}（超限 {info['oversized_access_rows_all_ranks']} 行）"
        for capacity, info in limits["capacities"].items())
    curve_capacities = sorted(map(int, limits["capacities"]))
    base_table_capacities = (*DEFAULT_CAPACITIES, SUPPLEMENTARY_CAPACITY)
    lines = [
        "# MiniMax HiSparse 请求内局部性",
        "",
        f"采集覆盖状态为 **{summary['coverage']}**。原生热缓冲为 "
        f"{summary['native_hot_capacity']} token，配置 Top-K 固定为 2048。"
        "每个采样层的前 16 个 decode step 作为预热，主汇总使用 rank 0 的预热后记录。",
        "",
        f"请求数为 {counts}。步数按请求和层累计。",
        "",
        f"两个 rank 共采集 {limits['access_rows_all_ranks']} 行。单步最多选中 "
        f"{limits['max_unique_selected_tokens']} 个去重位置，最多 "
        f"{limits['max_block_slots']} 个 block 槽，其中有效槽最多 "
        f"{limits['max_valid_block_slots']} 个。容量可行性为 {capacities}。",
        "2176 和显式指定的额外容量是补充模拟。任一采样行超限的容量整档标为不可行，"
        "缺失率和相应差值留空。",
        "",
        "原生缺失率由 swap 前的驻留成员关系推导。分母为去重后的选中逻辑 KV 位置，"
        "短序列与最新 token 按原生路径计入命中。模拟从每段冷启动，在一次 decode 内保护所有已选命中，"
        "并持续回放预热数据。模拟未复现原生 prefill 或最新 token 保留槽，缺失也不直接代表传输次数。",
        "容量差值保留符号。模拟策略的跨容量结果可能非单调。",
        "",
        "| 工作负载 | 计入步数 | 原生推导缺失率 | 模拟 2048 | 模拟 4096 | 模拟 8192 | 补充模拟 2176 | 2048→4096 降幅 | 4096→8192 降幅 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    has_postwarmup = False
    for row in summary["aggregates"]:
        if row["period"] != "after_first_16":
            continue
        has_postwarmup = True
        rates = [percentage(row["native_prestate_miss_rate"])] + [
            percentage(row[f"simulated_miss_rate_{capacity}"]) for capacity in base_table_capacities]
        deltas = [f"{row[f'simulated_miss_reduction_pp_{left}_to_{right}']:.2f} pp"
                  if row[f"simulated_miss_reduction_pp_{left}_to_{right}"] is not None else "—"
                  for left, right in zip(DEFAULT_CAPACITIES, DEFAULT_CAPACITIES[1:])]
        lines.append(f"| {row['workload']} | {row['steps']} | " + " | ".join(rates + deltas) + " |")
    if not has_postwarmup:
        lines += ["", "本次采集没有进入预热后的汇总区间。"]
    lines += [
        "", "## 各 rank 的工作负载容量曲线", "",
        "分别汇总各 rank 的预热后结果。rank 0 是主口径，rank 1 单独报告，"
        "两者不合并计算样本数或缺失率。",
        "",
        "| rank | 工作负载 | 步数 | 原生推导缺失率 | "
        + " | ".join(f"模拟 {capacity}" for capacity in curve_capacities)
        + " | 最小补充容量 ≤1 / 2 / 5 pp |",
        "|---:|---|---:|---:|" + "---:|" * len(curve_capacities) + "---|",
    ]
    for row in summary["rank_workload_after_first_16"]:
        curve = " | ".join(percentage(row[f"simulated_miss_rate_{capacity}"])
                           for capacity in curve_capacities)
        thresholds = " / ".join(str(value) if value is not None else "—"
                                for value in row["offline_hindsight_min_supplementary_capacity_by_pp"].values())
        lines.append(f"| {row['rank']} | {row['workload']} | {row['steps']} | "
                     f"{percentage(row['native_prestate_miss_rate'])} | {curve} | {thresholds} |")
    lines += [
        "", "## 逐请求缩容代价", "",
        "下表合并同一请求在 rank 0 的采样层，只计预热后。阈值按完整记录离线回放后判断，"
        "属于事后模拟结果，不能用作在线预测。补充容量阈值从已测试且低于 4096 的可行容量"
        "中取最小值，按同一记录与模拟 4096 比较。无满足项时留空。",
        "",
        "| 请求 | 工作负载 | 计入步数 | 原生推导缺失率 | 模拟缩容代价 4096→2048 | 模拟扩容收益 4096→8192 | 缩容代价 ≤1 / 2 / 5 pp | 最小补充容量 ≤1 / 2 / 5 pp |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in summary["primary_request_after_first_16"]:
        flags = " / ".join("—" if value is None else "是" if value else "否"
                           for value in row["offline_hindsight_shrink_cost_within_pp"].values())
        thresholds = " / ".join(str(value) if value is not None else "—"
                                for value in row["offline_hindsight_min_supplementary_capacity_by_pp"].values())
        lines.append(
            f"| {row['request']} | {row['workload']} | {row.get('steps', 0)} | "
            f"{percentage(row['native_prestate_miss_rate'])} | "
            f"{points(row['simulated_shrink_cost_pp_4096_to_2048'])} | "
            f"{points(row['simulated_expansion_benefit_pp_4096_to_8192'])} | {flags} | {thresholds} |"
        )
    lines += [
        "", "## 逐请求和逐层的窗口波动", "",
        "只统计 rank 0 预热后的完整 16 step 窗口。",
        "",
        "| 请求 | 层 | 完整窗口数 | 原生推导缺失率最小值 | 最大值 | 极差 | 模拟缩容代价最高值 | 对应窗口起始 step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["primary_request_layer_complete_16_step_bursts"]:
        lines.append(
            f"| {row['request']} | {row['layer']} | {row['complete_windows']} | "
            f"{percentage(row['native_prestate_miss_rate_min'])} | "
            f"{percentage(row['native_prestate_miss_rate_max'])} | "
            f"{points(row['native_prestate_miss_rate_range_pp'])} | "
            f"{points(row['simulated_shrink_cost_max_pp'])} | "
            f"{row['simulated_shrink_cost_max_window_start_step'] if row['simulated_shrink_cost_max_window_start_step'] is not None else '—'} |"
        )
    for row in summary["primary_phase_spread"]:
        if row["workload"] == "all" and row["metric"] == "native_prestate_miss_rate":
            lines += ["", f"预热后 {row['complete_windows']} 个完整 16 step 窗口中，"
                      f"原生推导缺失率的最小值、中位数、最大值分别为 "
                      f"{percentage(row['min'])}、{percentage(row['median'])}、{percentage(row['max'])}。"]
    lines += ["", "采样 rank 按请求、层和 step 对齐。对照 rank 仅用于核验或查看独立窗口，"
              "总计和阶段离散度只使用 rank 0。16 和 32 step 窗口见 CSV，"
              "完整区间、预热区间及预热后完整窗口的最小值、中位数和最大值见 JSON。"]
    for row in summary["capture"]["rank_selection_comparison"]:
        lines += ["", f"rank {row['rank']} 对齐 {row['matched_steps']} 步，"
                  f"其中 {row['selection_mismatch_steps']} 步选集不同。"
                  f"仅 rank 0 和仅对照 rank 的步数分别为 {row['primary_only_steps']} 和 "
                  f"{row['other_only_steps']}。"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="private runner directory")
    parser.add_argument("output", type=Path, help="new directory for sanitized reports")
    parser.add_argument("--supplementary-capacities", type=int, nargs="+", default=(),
                        help="additional simulated token capacities, alongside default 2176")
    args = parser.parse_args(argv)
    if any(capacity <= 0 for capacity in args.supplementary_capacities):
        parser.error("supplementary capacities must be positive integers")
    run, output = args.run.resolve(), args.output.resolve()
    if output.is_relative_to(run) or run.is_relative_to(output):
        parser.error("report directory must be separate from the raw run")
    summary = summarize(run, args.supplementary_capacities)
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
    with (output / "windows.csv").open("x", encoding="utf-8", newline="") as file:
        fields = list(summary["windows"][0]) if summary["windows"] else [
            "rank", "request", "workload", "layer", "window_steps", "window_start_step",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["windows"])
    (output / "report.md").write_text(report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
