"""Plot REINFORCE edge probabilities split by agent type (truthful vs adversarial)."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results_server"
OUT_DIR = RESULTS_DIR / "new_figures"
N_LIST = [1, 3, 5, 7]


def load_data():
    data = {}
    for n in N_LIST:
        p = RESULTS_DIR / f"n_pairs_{n}" / "mmlu_results.json"
        if p.exists():
            data[n] = json.loads(p.read_text(encoding="utf-8"))
    return data


def main():
    data = load_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Figure A: avg truthful vs adversarial edge prob per n_pairs ───────────
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(N_LIST))
    width = 0.35
    avg_truth = []
    avg_adv = []

    for n in N_LIST:
        probs = data[n]["reinforce"]["final_edge_probs"]
        avg_truth.append(np.mean(probs[:n]))
        avg_adv.append(np.mean(probs[n:]))

    bars_t = ax.bar(x - width/2, avg_truth, width, label="Truthful agents",
                    color="#4C72B0", alpha=0.85)
    bars_a = ax.bar(x + width/2, avg_adv, width, label="Adversarial agents",
                    color="#C44E52", alpha=0.85)

    for bar in bars_t:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)
    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5,
               label="Threshold (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}T{n}A" for n in N_LIST], fontsize=11)
    ax.set_ylabel("Avg Edge Probability", fontsize=12)
    ax.set_xlabel("Swarm Configuration", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.title("REINFORCE: Avg Edge Probability by Agent Type", fontsize=13)
    plt.tight_layout()
    out = OUT_DIR / "figure8_avg_prob_truthful_vs_adversarial.png"
    plt.savefig(out, dpi=150)
    print(f"Saved -> {out}")
    plt.close()

    # ── Figure B: individual edge probs per n_pairs, colored by type ──────────
    fig, axes = plt.subplots(1, len(N_LIST), figsize=(4.5 * len(N_LIST), 5), squeeze=False)

    for ax, n in zip(axes[0], N_LIST):
        probs = data[n]["reinforce"]["final_edge_probs"]
        truthful_probs = probs[:n]
        adversarial_probs = probs[n:]

        # jitter x positions slightly for readability
        t_x = np.arange(len(truthful_probs))
        a_x = np.arange(len(adversarial_probs))

        ax.scatter(t_x, truthful_probs, color="#4C72B0", s=100, zorder=3,
                   label="Truthful", marker="o")
        ax.scatter(a_x, adversarial_probs, color="#C44E52", s=100, zorder=3,
                   label="Adversarial", marker="^")

        for xi, p in enumerate(truthful_probs):
            ax.plot([xi, xi], [0, p], color="#4C72B0", alpha=0.3, linewidth=1.5)
        for xi, p in enumerate(adversarial_probs):
            ax.plot([xi, xi], [0, p], color="#C44E52", alpha=0.3, linewidth=1.5)

        ax.axhline(0.5, color="black", linestyle="--", linewidth=1,
                   alpha=0.6, label="Threshold")
        ax.axhspan(0.5, 1.0, alpha=0.04, color="blue")
        ax.axhspan(0.0, 0.5, alpha=0.04, color="red")

        ax.set_ylim(-0.05, 1.1)
        ax.set_xlabel("Agent index", fontsize=11)
        ax.set_ylabel("Edge Probability" if n == N_LIST[0] else "", fontsize=11)
        ax.set_title(f"{n}T{n}A", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # annotate averages
        ax.axhline(np.mean(truthful_probs), color="#4C72B0", linestyle=":",
                   linewidth=1.5, alpha=0.8)
        ax.axhline(np.mean(adversarial_probs), color="#C44E52", linestyle=":",
                   linewidth=1.5, alpha=0.8)

    plt.suptitle(
        "REINFORCE Edge Probabilities: Truthful vs Adversarial Agents\n"
        "(dotted lines = group averages, shaded = active/pruned zones)",
        fontsize=13
    )
    plt.tight_layout()
    out = OUT_DIR / "figure9_individual_edges_truthful_vs_adversarial.png"
    plt.savefig(out, dpi=150)
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
