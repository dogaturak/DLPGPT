"""
aggregate_results.py  --  merge all per-N_PAIRS results into combined plots.

Reproduces:
  • Figure 1: horizontal bar chart  (accuracy by method × swarm size)
  • Figure 2: token consumption + accuracy table

Usage:
    python experiments/aggregate_results.py
    python experiments/aggregate_results.py --results_dir results  --out_dir results/figures
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.config import EXPERIMENT_CONFIGS

METHODS = [
    ("IO (single agent)",  "io_baseline"),
    ("Fully Connected",    "fully_connected"),
    ("Random Connected",   "random_connected"),
    ("REINFORCE",          "reinforce"),
    ("EA",                 "ea"),
]


def load_results(results_dir: str) -> dict:
    """Load one JSON per n_pairs; return {n_pairs: data}."""
    data = {}
    for n in EXPERIMENT_CONFIGS:
        path = os.path.join(results_dir, f"n_pairs_{n}", "mmlu_results.json")
        if not os.path.exists(path):
            print(f"  [warn] missing: {path}")
            continue
        with open(path) as f:
            data[n] = json.load(f)
    return data


def print_table(data: dict) -> None:
    """Print Figure-2-style ASCII table to stdout."""
    col_w = 22
    header = f"{'Swarm':<14}" + "".join(f"{'  '+ml:<{col_w}}" for ml, _ in METHODS)
    print("\n" + "=" * len(header))
    print("ACCURACY TABLE")
    print(header)
    print("-" * len(header))
    for n, d in sorted(data.items()):
        label = f"{n}T{n}A"
        row = f"{label:<14}"
        for _, key in METHODS:
            acc = d.get(key, {}).get("accuracy", float("nan"))
            row += f"{'  '+f'{acc:.3f}':<{col_w}}"
        print(row)
    print("=" * len(header))

    print("\nTOKEN CONSUMPTION TABLE")
    print(header)
    print("-" * len(header))
    for n, d in sorted(data.items()):
        label = f"{n}T{n}A"
        row = f"{label:<14}"
        for _, key in METHODS:
            tok = d.get(key, {}).get("total_tokens", 0)
            row += f"{'  '+f'{tok:,}':<{col_w}}"
        print(row)
    print("=" * len(header) + "\n")


def plot_figure1(data: dict, out_dir: str) -> None:
    """Figure 1: horizontal grouped bar chart — accuracy by method & swarm size."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available, skipping Figure 1")
        return

    n_list = sorted(data.keys())
    method_labels = [ml for ml, _ in METHODS]
    method_keys   = [mk for _, mk in METHODS]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452", "#8172B2"]

    # one group of bars per swarm size, one bar per method
    x = np.arange(len(n_list))
    bar_width = 0.15
    offsets = np.linspace(-(len(METHODS)-1)/2, (len(METHODS)-1)/2, len(METHODS)) * bar_width

    fig, ax = plt.subplots(figsize=(13, 5))
    for j, (ml, mk, color) in enumerate(zip(method_labels, method_keys, colors)):
        accs = [data[n].get(mk, {}).get("accuracy", 0) for n in n_list]
        bars = ax.bar(x + offsets[j], accs, bar_width, label=ml, color=color, alpha=0.85)
        for bar, a in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{a:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}T{n}A" for n in n_list], fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel("Swarm Configuration", fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.title("MMLU Accuracy by Method and Swarm Size", fontsize=14, pad=12)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "figure1_accuracy_by_swarm.png")
    plt.savefig(out, dpi=150)
    print(f"Figure 1 saved -> {out}")
    plt.close()


def plot_figure2(data: dict, out_dir: str) -> None:
    """Figure 2: token consumption × accuracy, one subplot per swarm size."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available, skipping Figure 2")
        return

    n_list = sorted(data.keys())
    colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452", "#8172B2"]

    fig, axes = plt.subplots(1, len(n_list), figsize=(5 * len(n_list), 5), sharey=False)
    if len(n_list) == 1:
        axes = [axes]

    for ax, n in zip(axes, n_list):
        d = data[n]
        names  = [ml for ml, _ in METHODS]
        tokens = [d.get(mk, {}).get("total_tokens", 0) for _, mk in METHODS]
        accs   = [d.get(mk, {}).get("accuracy", 0)     for _, mk in METHODS]

        x = np.arange(len(names))
        bars = ax.bar(x, tokens, color=colors, alpha=0.75, zorder=2)
        ax.set_yscale("log")
        ax.set_ylabel("Total Tokens (log)" if n == n_list[0] else "", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=1)
        ax.set_title(f"{n}T{n}A", fontsize=13, fontweight="bold")

        for bar, tok in zip(bars, tokens):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
                    f"{tok:,}", ha="center", va="bottom", fontsize=7, rotation=90)

        ax2 = ax.twinx()
        ax2.plot(x, accs, "o-", color="#2d2d2d", linewidth=2, markersize=7, zorder=3)
        for xi, a in zip(x, accs):
            ax2.annotate(f"{a:.2f}", (xi, a), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8, fontweight="bold")
        ax2.set_ylim(-0.05, 1.15)
        ax2.set_ylabel("Accuracy" if n == n_list[-1] else "", fontsize=10)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))

    plt.suptitle("Token Consumption vs Accuracy by Swarm Size", fontsize=14, y=1.02)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "figure2_tokens_vs_accuracy.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure 2 saved -> {out}")
    plt.close()


def plot_ea_convergence(data: dict, out_dir: str) -> None:
    """Bonus: EA fitness curve per swarm size."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n_list = sorted(data.keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    for n in n_list:
        gen_bests = data[n].get("ea", {}).get("gen_bests", [])
        if gen_bests:
            ax.plot(range(1, len(gen_bests)+1), gen_bests,
                    marker="o", markersize=4, label=f"{n}T{n}A")

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Fitness (batch accuracy)", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    plt.title("EA Fitness Convergence per Swarm Size", fontsize=13)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "ea_convergence.png")
    plt.savefig(out, dpi=150)
    print(f"EA convergence saved -> {out}")
    plt.close()


def plot_reinforce_convergence(data: dict, out_dir: str) -> None:
    """Figure 3: REINFORCE rolling-avg reward per swarm size, all on one chart."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    n_list = sorted(data.keys())
    fig, ax = plt.subplots(figsize=(9, 5))

    for n in n_list:
        rewards = data[n].get("reinforce", {}).get("train_rewards", [])
        if not rewards:
            continue
        window = max(5, len(rewards) // 5)
        rm = np.convolve(rewards, np.ones(window) / window, mode="valid")
        x_steps = range(window, len(rewards) + 1)
        styles = {1: "-", 3: "--", 5: "-.", 7: ":"}
        ax.plot(x_steps, rm, marker="o", markersize=4, label=f"{n}T{n}A",
                alpha=0.9, linestyle=styles.get(n, "-"), linewidth=2)

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Rolling Avg Reward (batch accuracy)", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    plt.title("REINFORCE Reward Convergence per Swarm Size", fontsize=13)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "figure3_reinforce_convergence.png")
    plt.savefig(out, dpi=150)
    print(f"Figure 3 saved -> {out}")
    plt.close()


def plot_edge_probabilities(data: dict, out_dir: str) -> None:
    """Figure 4: REINFORCE learned edge probabilities per swarm size."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    n_list = sorted(data.keys())
    fig, axes = plt.subplots(1, len(n_list), figsize=(4 * len(n_list), 4), squeeze=False)

    for ax, n in zip(axes[0], n_list):
        probs = data[n].get("reinforce", {}).get("final_edge_probs", [])
        if not probs:
            ax.set_title(f"{n}T{n}A — no data")
            continue
        x = np.arange(len(probs))
        colors = ["#C44E52" if p >= 0.5 else "#4C72B0" for p in probs]
        ax.bar(x, probs, color=colors, alpha=0.85)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="threshold=0.5")
        for xi, p in enumerate(probs):
            ax.text(xi, p + 0.03, f"{p:.2f}", ha="center", va="bottom", fontsize=7)
        active = sum(p >= 0.5 for p in probs)
        ax.set_xlabel("Edge index")
        ax.set_ylabel("Probability")
        ax.set_title(f"{n}T{n}A  ({active}/{len(probs)} active)")
        ax.set_ylim(0, 1.2)
        ax.set_xticks(x)
        ax.legend(fontsize=8)

    plt.suptitle("REINFORCE Learned Edge Probabilities\n(red = active ≥ 0.5, blue = pruned)", fontsize=13)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "figure4_edge_probabilities.png")
    plt.savefig(out, dpi=150)
    print(f"Figure 4 saved -> {out}")
    plt.close()


def plot_token_cost_vs_npairs(data: dict, out_dir: str) -> None:
    """Figure 5: total token cost per method across swarm sizes (n_pairs)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    n_list = sorted(data.keys())
    colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452", "#8172B2"]
    markers = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for (ml, mk), color, marker in zip(METHODS, colors, markers):
        tokens = [data[n].get(mk, {}).get("total_tokens", 0) for n in n_list]
        ax.plot(n_list, tokens, marker=marker, color=color, linewidth=2,
                markersize=7, label=ml)
        for n, t in zip(n_list, tokens):
            ax.annotate(f"{t:,}", (n, t), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color=color)

    ax.set_xticks(n_list)
    ax.set_xticklabels([f"{n}T{n}A" for n in n_list], fontsize=11)
    ax.set_yscale("log")
    ax.set_ylabel("Total Tokens (log scale)", fontsize=12)
    ax.set_xlabel("Swarm Configuration", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    plt.title("Token Cost Scaling per Method across Swarm Sizes", fontsize=13)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "figure5_token_cost_vs_npairs.png")
    plt.savefig(out, dpi=150)
    print(f"Figure 5 saved -> {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=os.path.join(ROOT, "results"))
    parser.add_argument("--out_dir",     default=os.path.join(ROOT, "results", "figures"))
    args = parser.parse_args()

    data = load_results(args.results_dir)
    if not data:
        print("No result files found. Run run_experiment.py first.")
        return

    print_table(data)
    plot_figure1(data, args.out_dir)
    plot_figure2(data, args.out_dir)
    plot_ea_convergence(data, args.out_dir)
    plot_reinforce_convergence(data, args.out_dir)
    plot_edge_probabilities(data, args.out_dir)
    plot_token_cost_vs_npairs(data, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
