import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Data ───────────────────────────────────────────────────────────────────────
with open("results/mmlu_results.json") as f:
    data = json.load(f)

rewards = data["reinforce"]["train_rewards"]
steps = list(range(1, len(rewards) + 1))

# Edge probabilities per step — parsed from training output
edge_probs = [
    [0.50, 0.50, 0.50],
    [0.50, 0.50, 0.50],
    [0.50, 0.50, 0.50],
    [0.50, 0.50, 0.50],
    [0.49, 0.51, 0.49],
    [0.50, 0.52, 0.48],
    [0.50, 0.52, 0.48],
    [0.50, 0.52, 0.47],
    [0.50, 0.52, 0.48],
    [0.50, 0.52, 0.48],
    [0.50, 0.52, 0.48],
    [0.50, 0.52, 0.48],
    [0.50, 0.52, 0.48],
    [0.50, 0.52, 0.48],
    [0.50, 0.52, 0.48],
    [0.50, 0.52, 0.48],
    [0.50, 0.52, 0.47],
    [0.50, 0.52, 0.47],
    [0.50, 0.52, 0.47],
    [0.50, 0.52, 0.47],
    [0.50, 0.52, 0.47],
    [0.50, 0.52, 0.47],
    [0.50, 0.52, 0.47],
    [0.50, 0.52, 0.47],
    [0.50, 0.52, 0.47],
    [0.51, 0.53, 0.46],
    [0.51, 0.53, 0.47],
    [0.51, 0.53, 0.47],
    [0.51, 0.53, 0.47],
    [0.52, 0.54, 0.46],
]
edge_probs = np.array(edge_probs)
edge_labels = ["AnalyticalAgent", "ExpertAgent", "EliminationAgent"]
edge_colors = ["#2196F3", "#4CAF50", "#F44336"]

# Rolling mean (window=5)
window = 5
rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
rolling_steps = steps[window - 1:]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
fig.suptitle("REINFORCE Edge Optimization on MMLU\n(llama3.2 3B, 30 train steps)", fontsize=13)

# ── Panel 1: Reward curve ──────────────────────────────────────────────────────
ax1.bar(steps, rewards, color="#BBDEFB", edgecolor="none", width=0.7, label="Step reward", zorder=2)
ax1.plot(rolling_steps, rolling, color="#1565C0", linewidth=2, label=f"Rolling mean (w={window})", zorder=3)
ax1.axhline(np.mean(rewards), color="#1565C0", linestyle="--", linewidth=1, alpha=0.5,
            label=f"Mean reward ({np.mean(rewards):.2f})")
ax1.set_ylabel("Reward (0/1)", fontsize=11)
ax1.set_ylim(-0.05, 1.2)
ax1.set_yticks([0, 0.5, 1.0])
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(axis="y", alpha=0.3)
ax1.set_title("Training Reward per Step", fontsize=11)

# ── Panel 2: Edge probabilities ────────────────────────────────────────────────
for i, (label, color) in enumerate(zip(edge_labels, edge_colors)):
    ax2.plot(steps, edge_probs[:, i], color=color, linewidth=2, label=label, marker="o",
             markersize=3)
ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="p=0.5 (chance)")
ax2.fill_between(steps, 0.5, 1.0, alpha=0.04, color="green")
ax2.fill_between(steps, 0.0, 0.5, alpha=0.04, color="red")
ax2.set_xlabel("Training Step", fontsize=11)
ax2.set_ylabel("Edge Probability  σ(w)", fontsize=11)
ax2.set_ylim(0.42, 0.60)
ax2.legend(fontsize=9, loc="center right")
ax2.grid(axis="y", alpha=0.3)
ax2.set_title("Edge Inclusion Probability Over Training", fontsize=11)

# Annotate final state
final_probs = data["reinforce"]["final_edge_probs"]
for i, (p, color) in enumerate(zip(final_probs, edge_colors)):
    active = p > 0.5
    marker = "✓" if active else "✗"
    ax2.annotate(f"{marker} p={p:.2f}", xy=(30, edge_probs[-1, i]),
                 xytext=(30.3, edge_probs[-1, i]), fontsize=8, color=color, va="center")

plt.tight_layout()
out = "results/reward_and_edges.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
plt.show()
