"""Plot matched HiSparse repeats with an isolated Matplotlib installation.

Install with python -m pip install --target cache/plot-deps matplotlib.
The first input is the pairing reference. Plot only one sampling mode.
"""

import argparse
import json
from pathlib import Path
from statistics import median
import sys

from summarize_hisparse import load, require


ROOT = Path(__file__).resolve().parents[1]


def raw_phases(directory):
    raw = json.loads((directory / "performance.json").read_text(encoding="utf-8"))
    return {(phase["workload"], phase["repeat"]): phase for phase in raw["phases"]}


def audit_pair(reference, candidate):
    require(reference.keys() == candidate.keys(), "Matched phase sets differ")
    fields = ("request_id", "prompt_sha256", "input_tokens", "requested_output_tokens")
    for key, base in reference.items():
        other = candidate[key]
        require(base["concurrency"] == other["concurrency"],
                f"Offered concurrency differs for {key}")
        prompts = lambda phase: sorted(tuple(r[field] for field in fields) for r in phase["requests"])
        require(prompts(base) == prompts(other), f"Measured prompts or shapes differ for {key}")


def plot(directory_args, output):
    cases = [load(path) for path in directory_args]
    reference = cases[0]
    raw = raw_phases(directory_args[0])
    for path, case in zip(directory_args[1:], cases[1:]):
        require(all(case[key] == reference[key] for key in ("sampling", "tp_size", "warmup_policy")),
                "Sampling, TP degree, or warmup policy differs")
        audit_pair(raw, raw_phases(path))

    sys.path.insert(0, str(ROOT / "cache" / "plot-deps"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    workloads = list(reference["workloads"])
    labels = [case["arm"] for case in cases]
    palette = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, len(cases), figsize=(3.25 * len(cases), 4.4),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    for index, (ax, case) in enumerate(zip(axes[0], cases)):
        for i, name in enumerate(workloads):
            row = case["workloads"][name]["throughput_output_tokens_per_s"]
            repeats = row["by_repeat"]
            jitter = [(j - (len(repeats) - 1) / 2) * 0.045 for j in range(len(repeats))]
            ax.scatter(repeats, [i + offset for offset in jitter], color=palette(index), s=22)
            ax.scatter(row["median"], i, marker="|", color="#161616", s=170)
        ax.set_title(case["arm"])
        ax.grid(axis="x", alpha=0.25)
        ax.set_xlabel("Output tokens/s")
        ax.set_yticks(range(len(workloads)), [name.replace("_", " ") for name in workloads])
    fig.suptitle(f"{reference['sampling'].capitalize()} throughput per measured repeat")
    fig.savefig(output / "hisparse-throughput.pdf")
    plt.close(fig)

    if len(cases) > 1:
        fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
        for index, case in enumerate(cases[1:], start=1):
            offset = (index - len(cases) / 2) * 0.16
            for i, name in enumerate(workloads):
                base = reference["workloads"][name]["throughput_output_tokens_per_s"]["by_repeat"]
                values = case["workloads"][name]["throughput_output_tokens_per_s"]["by_repeat"]
                changes = [100 * (new / old - 1) for new, old in zip(values, base, strict=True)]
                ax.scatter(changes, [i + offset] * len(changes), s=21, color=palette(index),
                           label=case["arm"] if i == 0 else None)
                ax.scatter(median(changes), i + offset, marker="|", color="#161616", s=150)
        ax.axvline(0, color="#555555", linewidth=0.9)
        ax.grid(axis="x", alpha=0.25)
        ax.set_yticks(range(len(workloads)), [name.replace("_", " ") for name in workloads])
        ax.set_xlabel(f"Paired throughput change vs {reference['arm']} (%)")
        ax.legend(loc="best", frameon=False)
        fig.savefig(output / "hisparse-paired-throughput.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    for index, case in enumerate(cases):
        peaks = [case["workloads"][name]["nvml_peak_synchronized_sum_bytes"] / (1 << 30)
                 for name in workloads]
        ax.plot(range(len(workloads)), peaks, marker="o", color=palette(index), label=case["arm"])
    ax.set_xticks(range(len(workloads)), [name.replace("_", " ") for name in workloads],
                  rotation=15)
    ax.set_ylabel("Peak synchronized TP-group NVML (GiB)")
    ax.legend(loc="best", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output / "hisparse-nvml.pdf")
    plt.close(fig)

    backing = []
    for path, label in zip(directory_args, labels):
        files = sorted((path / "vmm_stats").glob("tp-*.json"))
        if files:
            rows = [json.loads(file.read_text(encoding="utf-8")) for file in files]
            require({row["tp_rank"] for row in rows} == set(range(reference["tp_size"])),
                    f"Incomplete backing rank stats for {label}")
            backing.append((label, rows))
    if backing:
        fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
        for i, (_, rows) in enumerate(backing):
            for offset, key, color in ((-0.11, "backed_bytes", "tab:blue"),
                                       (0.11, "peak_backed_bytes", "tab:orange")):
                vals = [row[key] / (1 << 20) for row in rows]
                ax.scatter([i + offset] * len(vals), vals, color=color,
                           label=key.replace("_", " ") if not i else None)
        ax.set_xticks(range(len(backing)), [label for label, _ in backing], rotation=15)
        ax.set_ylabel("Sparse main KV backing per rank (MiB)")
        ax.legend(loc="best", frameon=False)
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(output / "hisparse-vmm-backing.pdf")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        plot(args.directory, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Plot rejected: {exc}\n")


if __name__ == "__main__":
    main()
