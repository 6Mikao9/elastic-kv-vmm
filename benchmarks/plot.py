"""Plot device memory and baseline-normalized throughput from summary.json."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save(fig, directory, stem):
    for extension in ("png", "svg"):
        fig.savefig(directory / f"{stem}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_memory(models, workloads, modes, rows, directory):
    fig, axes = plt.subplots(len(models), len(workloads), squeeze=False,
                             figsize=(3.2 * len(workloads), 3 * len(models)), sharey=True,
                             layout="constrained")
    for model, panels in zip(models, axes):
        model_modes = modes[model]
        for workload, axis in zip(workloads, panels):
            for offset, field, label, color in (
                (-0.19, "nvml_idle_median_bytes", "Idle median", "#31718e"),
                (0.19, "nvml_peak_bytes", "Sampled active peak", "#c06e2c"),
            ):
                axis.bar([i + offset for i in range(len(model_modes))],
                         [rows[model, mode, workload][field] / 2**30 for mode in model_modes],
                         width=0.36, label=label, color=color)
            axis.set_title(f"{model}\n{workload.replace('_', ' ')}")
            axis.set_xticks(range(len(model_modes)), model_modes)
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
        panels[0].set_ylabel("NVML device memory (GiB)")
    axes[0, 0].legend(fontsize=8)
    save(fig, directory, "device-memory")


def plot_throughput(models, workloads, modes, rows, directory):
    fig, axes = plt.subplots(len(models), len(workloads), squeeze=False,
                             figsize=(3.2 * len(workloads), 3 * len(models)), sharey=True,
                             layout="constrained")
    for model, panels in zip(models, axes):
        model_modes = modes[model]
        for workload, axis in zip(workloads, panels):
            reference = rows[model, "baseline", workload]["output_tokens_per_s_median"]
            selected = [rows[model, mode, workload] for mode in model_modes]
            centers = [row["output_tokens_per_s_median"] / reference for row in selected]
            lower = [(row["output_tokens_per_s_median"] - row["output_tokens_per_s_min"])
                     / reference for row in selected]
            upper = [(row["output_tokens_per_s_max"] - row["output_tokens_per_s_median"])
                     / reference for row in selected]
            axis.bar(model_modes, centers, color="#31718e", yerr=[lower, upper], capsize=3)
            axis.axhline(1, color="#333333", linewidth=1, linestyle="--")
            axis.set_title(f"{model}\n{workload.replace('_', ' ')}")
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
        panels[0].set_ylabel("Output throughput / baseline median")
    fig.supxlabel("Bars show medians. Whiskers show repeat minima and maxima.")
    save(fig, directory, "normalized-throughput")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if not summary["correctness_passed"]:
        parser.error("The summary contains failed correctness checks")
    records = summary["workloads"]
    rows = {(r["model"], r["mode"], r["workload"]): r for r in records}
    models = list(dict.fromkeys(r["model"] for r in records))
    modes = {model: sorted({r["mode"] for r in records if r["model"] == model})
             for model in models}
    workloads = list(dict.fromkeys(r["workload"] for r in records))
    if (not models or len(rows) != len(records)
            or any("baseline" not in modes[model] for model in models)
            or len(rows) != sum(len(modes[model]) * len(workloads) for model in models)):
        parser.error("Expected one record per workload for each measured model and mode")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_memory(models, workloads, modes, rows, args.output_dir)
    plot_throughput(models, workloads, modes, rows, args.output_dir)


if __name__ == "__main__":
    main()
