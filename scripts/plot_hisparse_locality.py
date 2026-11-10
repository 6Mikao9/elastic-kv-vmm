"""Plot sanitized locality evidence; simulation and native-derived metrics stay separate."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    colors = {"R01": "#3676b7", "R03": "#38a18b", "R04": "#b88b22",
              "R05": "#d15b50", "R06": "#925aaa"}
    rows = [r for r in data["primary_request_after_first_16"] if r.get("selected", 0)]
    capacities = [2048, 2560, 3072, 3584, 4096, 6144, 8192]
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    for row in rows:
        ax.plot(capacities, [100 * row[f"simulated_miss_rate_{c}"] for c in capacities],
                marker="o", ms=4, color=colors[row["request"]],
                label=f"{row['request']} ({row['workload']}, {row['input_tokens']:,} input)")
    ax.axvline(4096, ls="--", lw=1, color="#888888")
    ax.set(xlabel="Simulated resident capacity (tokens)", ylabel="Simulated miss rate (%)",
           title="Same real selection trace, different simulated capacities\nRank 0; sampled layers 5/15/25; after first 16 decode steps",
           xticks=capacities, ylim=(0, 25))
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc="upper right")
    for suffix in ("png", "svg"):
        fig.savefig(args.output / f"capacity-replay.{suffix}", dpi=160)
    plt.close(fig)

    windows = data.get("plot_windows", data.get("windows", []))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, sharey=True, layout="constrained")
    for ax, request in zip(axes, ("R05", "R06")):
        for layer, style in zip((5, 15, 25), ("-", "--", ":")):
            selected = sorted((w for w in windows if w["rank"] == 0
                               and w["request"] == request and w["layer"] == layer
                               and w["window_steps"] == 16 and w["steps"] == 16
                               and w["window_start_step"] >= 16),
                              key=lambda w: w["window_start_step"])
            ax.plot([w["window_start_step"] for w in selected],
                    [100*w["native_prestate_miss_rate"] for w in selected],
                    ls=style, lw=1.8, label=f"Layer {layer}")
        ax.set(title=f"{request}: long document", ylabel="Derived miss rate (%)", ylim=(0, 25))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Decode step at start of complete 16-step window (zero-based)")
    fig.suptitle("Actual native 4096 hot set: within-request variation\nRank 0; misses inferred from pre-swap residency, not transfer counters", fontsize=12)
    for suffix in ("png", "svg"):
        fig.savefig(args.output / f"native-temporal-windows.{suffix}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
