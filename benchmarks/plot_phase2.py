"""Plot measured Phase 2 results as publication PNG and editable SVG."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from summarize import WORKLOADS
from summarize_capacity import ARMS, summarize as summarize_capacity


MODELS = {"qwen06": "Qwen3-0.6B-Base", "qwen4": "Qwen3-4B"}
LABELS = {"baseline": "Baseline", "v4": "V4", "v5": "V5",
          "v5-keep8": "V5 (keep 8)", "baseline-tuned": "Prior-tuned baseline"}
COLORS = {"baseline": "#596579", "v4": "#29828a", "v5": "#2455a4",
          "v5-keep8": "#ae601f", "baseline-tuned": "#ae601f"}
WORKLOAD_LABELS = [name.replace("_", "\n").title() for name in WORKLOADS]
GIB = 2**30


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig, output, name, caption):
    fig.supxlabel(caption, fontsize=9)
    for suffix in ("png", "svg"):
        destination = output / f"{name}.{suffix}"
        fig.savefig(destination, dpi=300)
        if suffix == "svg":
            destination.write_text("\n".join(line.rstrip() for line in
                                   destination.read_text(encoding="utf-8").splitlines())
                                   + "\n", encoding="utf-8")
    plt.close(fig)


def bars(ax, series, modes):
    width = 0.72 / len(modes)
    for j, mode in enumerate(modes):
        values, errors = series[mode]
        offsets = [i + (j - (len(modes) - 1) / 2) * width for i in range(len(values))]
        ax.bar(offsets, values, width, yerr=errors, capsize=3 if errors else 0,
               color=COLORS[mode], label=LABELS[mode], zorder=3)
    ax.grid(axis="y", color="#ddd", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def rates(rows, denominator):
    values = [row["output_tokens_per_s_median"] / scale for row, scale in zip(rows, denominator)]
    lows = [row["output_tokens_per_s_min"] / scale for row, scale in zip(rows, denominator)]
    highs = [row["output_tokens_per_s_max"] / scale for row, scale in zip(rows, denominator)]
    return values, [[v - l for v, l in zip(values, lows)], [h - v for v, h in zip(values, highs)]]


def exact(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), sharey=True, layout="constrained")
    for ax, (model, label) in zip(axes, MODELS.items()):
        denominator = [data[model, "graph", "baseline", work]["output_tokens_per_s_median"]
                       for work in WORKLOADS]
        modes = ("v4", "v5")
        bars(ax, {mode: rates([data[model, "graph", mode, work] for work in WORKLOADS], denominator)
                  for mode in modes}, modes)
        ax.axhline(1, color=COLORS["baseline"], linestyle="--", linewidth=1)
        ax.set_xticks(range(len(WORKLOADS)), WORKLOAD_LABELS)
        ax.set_title(label)
    axes[0].set_ylabel("Throughput / Graph baseline median")
    axes[1].set_ylim(0, max(1.16, axes[1].get_ylim()[1] + 0.08))
    axes[1].legend(loc="upper right", ncol=2, frameon=False)
    save(fig, output, "phase2_exact_throughput",
         "Bars show medians and whiskers show min–max over 3 repeats per workload.")


def padding(data, output):
    modes = ("baseline", "v5", "v5-keep8")
    if not all((model, "graph-pad", mode, work) in data
               for model in MODELS for mode in modes for work in WORKLOADS):
        return
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey="row",
                             layout="constrained")
    measures = (("ttft_s_median", 1000, "Median TTFT (ms)"),
                ("nvml_idle_median_bytes", 1 / GIB, "Idle GPU memory (GiB)"))
    for j, (model, label) in enumerate(MODELS.items()):
        axes[0, j].set_title(label)
        for i, (field, scale, ylabel) in enumerate(measures):
            ax = axes[i, j]
            bars(ax, {mode: ([data[model, "graph-pad", mode, work][field] * scale
                             for work in WORKLOADS], None) for mode in modes}, modes)
            axes[i, 0].set_ylabel(ylabel)
        axes[1, j].set_xticks(range(len(WORKLOADS)), WORKLOAD_LABELS)
    axes[0, 1].legend(frameon=False)
    save(fig, output, "phase2_padding_retention",
         "Padded decode Graph. Idle memory is the median NVML device usage after each workload.")


def capacity(directory, output):
    paths = {arm: directory / "capacity" / arm / "results.json" for arm in ARMS}
    if not all(path.is_file() for path in paths.values()):
        return
    fig, (cdf, mem, throughput) = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    summaries = {}
    for arm, path in paths.items():
        phases = read_json(path)["phases"]
        samples = sorted(req["ttft_s"] for phase in phases for req in phase["requests"])
        if len(samples) != 120:
            raise ValueError(f"{path}: expected 120 measured requests")
        cdf.step([samples[0], *samples], [i / len(samples) for i in range(len(samples) + 1)],
                 where="post", color=COLORS[arm], label=LABELS[arm])
        summaries[arm], _ = summarize_capacity(phases)
    cdf.set(xlabel="TTFT (s)", ylabel="Empirical cumulative fraction", ylim=(0, 1.02))
    cdf.grid(color="#ddd")
    cdf.legend(frameon=False, fontsize=8)

    fields = ("active_peak_device_used_bytes", "post_idle_device_used_bytes")
    bars(mem, {arm: ([row[field]["median"] / GIB for field in fields], None)
               for arm, row in summaries.items()}, ARMS)
    mem.set_xticks([0, 1], ["Active peak", "Post-burst idle"])
    mem.set_ylabel("GPU memory (GiB)")
    series = {}
    for arm, summary in summaries.items():
        rate = summary["output_tokens_per_s"]
        mid = rate["median"]
        series[arm] = [mid], [[mid - rate["min"]], [rate["max"] - mid]]
    bars(throughput, series, ARMS)
    throughput.set_xticks([0], ["Three measured bursts"])
    throughput.set_ylabel("Output tokens/s")
    save(fig, output, "phase2_capacity",
         "Controlled bursts: 120 measured requests per arm, excluding the first burst. "
         "Throughput bars show medians and whiskers span three bursts.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--results-dir", type=Path, default=root / "results" / "phase2")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output-dir", type=Path, default=root / "docs" / "figures")
    args = parser.parse_args()
    summary = args.summary or args.results_dir / "summary" / "summary.json"
    if args.summary is None and not summary.is_file():
        summary = args.results_dir / "summary-exact" / "summary.json"
    data = {(row["model"], row["execution"], row["mode"], row["workload"]): row
            for row in read_json(summary)["workloads"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exact(data, args.output_dir)
    padding(data, args.output_dir)
    capacity(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
