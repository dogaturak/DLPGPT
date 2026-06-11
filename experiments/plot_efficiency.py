"""Plot accuracy-per-token efficiency across methods and swarm sizes."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results_server"
OUT_DIR = RESULTS_DIR / "new_figures"

METHODS = [
    ("IO (single agent)", "io_baseline"),
    ("Fully Connected",   "fully_connected"),
    ("Random Connected",  "random_connected"),
    ("REINFORCE",         "reinforce"),
    ("EA",                "ea"),
]
COLORS  = ["#4C72B0", "#55A868", "#C44E52", "#DD8452", "#8172B2"]
MARKERS = ["o", "s", "^", "D", "v"]
N_LIST  = [1, 3, 5, 7]


def load_data():
    data = {}
    for n in N_LIST:
        p = RESULTS_DIR / f"n_pairs_{n}" / "mmlu_results.json"
        if p.exists():
            data[n] = json.loads(p.read_text(encoding="utf-8"))
    return data


def main():
    data = load_data()
    if not data:
        print("No result files found.")
        return

    # ── Figure 1: accuracy / 1000 tokens per method across n_pairs ────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    for (label, key), color, marker in zip(METHODS, COLORS, MARKERS):
        efficiencies = []
        for n in N_LIST:
            d = data[n].get(key, {})
            acc = d.get("accuracy", 0)
            tok = d.get("total_tokens", 1)
            efficiencies.append(acc / (tok / 1000))  # accuracy per 1k tokens
        ax.plot(N_LIST, efficiencies, marker=marker, color=color,
                linewidth=2, markersize=7, label=label)
        for n, e in zip(N_LIST, efficiencies):
            ax.annotate(f"{e:.4f}", (n, e), textcoords="offset points",
                        xytext=(5, 4), fontsize=7, color=color)

    ax.set_xticks(N_LIST)
    ax.set_xticklabels([f"{n}T{n}A" for n in N_LIST], fontsize=11)
    ax.set_ylabel("Accuracy per 1k Tokens", fontsize=12)
    ax.set_xlabel("Swarm Configuration", fontsize=12)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    plt.title("Token Efficiency  (Accuracy / 1k Tokens)", fontsize=13)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out1 = OUT_DIR / "figure6_efficiency_vs_npairs.png"
    plt.savefig(out1, dpi=150)
    print(f"Saved -> {out1}")
    plt.close()

    # ── Figure 2: average efficiency bar chart across all n_pairs ─────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    avg_eff = []
    for label, key in METHODS:
        vals = []
        for n in N_LIST:
            d = data[n].get(key, {})
            acc = d.get("accuracy", 0)
            tok = d.get("total_tokens", 1)
            vals.append(acc / (tok / 1000))
        avg_eff.append(np.mean(vals))

    labels = [m[0] for m in METHODS]
    bars = ax.bar(labels, avg_eff, color=COLORS, alpha=0.85)
    for bar, v in zip(bars, avg_eff):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                f"{v:.5f}", ha="center", va="bottom", fontsize=9)

    ax.set_yscale("log")
    ax.set_ylabel("Avg Accuracy per 1k Tokens (log)", fontsize=12)
    ax.set_xlabel("Method", fontsize=12)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.title("Average Token Efficiency across All Swarm Sizes", fontsize=13)
    plt.tight_layout()
    out2 = OUT_DIR / "figure7_avg_efficiency.png"
    plt.savefig(out2, dpi=150)
    print(f"Saved -> {out2}")
    plt.close()


if __name__ == "__main__":
    main()
