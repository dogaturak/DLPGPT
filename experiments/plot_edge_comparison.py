"""Side-by-side REINFORCE edge probs vs EA best topology, colored by agent type."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results_server"
OUT_DIR = RESULTS_DIR / "new_figures"
N_LIST = [1, 3, 5, 7]


def main():
    data = {n: json.loads((RESULTS_DIR / f"n_pairs_{n}" / "mmlu_results.json").read_text(encoding="utf-8"))
            for n in N_LIST}

    fig, axes = plt.subplots(len(N_LIST), 2, figsize=(12, 3.5 * len(N_LIST)))

    for row, n in enumerate(N_LIST):
        d = data[n]
        probs = d["reinforce"]["final_edge_probs"]
        topo  = d["ea"]["best_topology"]

        truthful_idx    = list(range(n))
        adversarial_idx = list(range(n, 2 * n))
        x = np.arange(2 * n)

        colors = ["#4C72B0" if i < n else "#C44E52" for i in range(2 * n)]

        # ── Left: REINFORCE ────────────────────────────────────────────────────
        ax_rl = axes[row, 0]
        ax_rl.bar(x, probs, color=colors, alpha=0.85)
        ax_rl.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
        for xi, p in enumerate(probs):
            ax_rl.text(xi, p + 0.03, f"{p:.2f}", ha="center", fontsize=7)
        ax_rl.set_ylim(0, 1.2)
        ax_rl.set_xticks(x)
        ax_rl.set_xticklabels(
            [f"T{i}" for i in range(n)] + [f"A{i}" for i in range(n)],
            fontsize=8
        )
        ax_rl.set_ylabel("Edge Probability", fontsize=10)
        ax_rl.set_title(f"{n}T{n}A — REINFORCE", fontsize=11, fontweight="bold")
        ax_rl.grid(axis="y", linestyle="--", alpha=0.3)
        active_rl = sum(p >= 0.5 for p in probs)
        ax_rl.text(0.98, 0.95, f"Active: {active_rl}/{2*n}",
                   transform=ax_rl.transAxes, ha="right", va="top", fontsize=9,
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        # ── Right: EA ──────────────────────────────────────────────────────────
        ax_ea = axes[row, 1]
        ax_ea.bar(x, topo, color=colors, alpha=0.85)
        ax_ea.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)
        for xi, t in enumerate(topo):
            label = "ON" if t == 1 else "OFF"
            ax_ea.text(xi, t + 0.03, label, ha="center", fontsize=7,
                       fontweight="bold")
        ax_ea.set_ylim(0, 1.4)
        ax_ea.set_xticks(x)
        ax_ea.set_xticklabels(
            [f"T{i}" for i in range(n)] + [f"A{i}" for i in range(n)],
            fontsize=8
        )
        ax_ea.set_ylabel("Edge Active (0/1)", fontsize=10)
        ax_ea.set_title(f"{n}T{n}A — EA", fontsize=11, fontweight="bold")
        ax_ea.grid(axis="y", linestyle="--", alpha=0.3)
        active_ea = sum(topo)
        ax_ea.text(0.98, 0.95, f"Active: {active_ea}/{2*n}",
                   transform=ax_ea.transAxes, ha="right", va="top", fontsize=9,
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", alpha=0.85, label="Truthful agent"),
        Patch(facecolor="#C44E52", alpha=0.85, label="Adversarial agent"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2,
               fontsize=11, bbox_to_anchor=(0.5, 1.01))

    plt.suptitle("Learned Edge Selection: REINFORCE vs EA\n(threshold=0.5 dashed)",
                 fontsize=13, y=1.04)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "figure11_rl_vs_ea_edges.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
