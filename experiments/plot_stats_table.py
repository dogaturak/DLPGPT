"""Create experiment stats table figure (nodes, edges, optimization time)."""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results_server"
OUT_DIR = RESULTS_DIR / "new_figures"
N_LIST = [1, 3, 5, 7]


def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    rows = []
    for n in N_LIST:
        d = json.load(open(RESULTS_DIR / f"n_pairs_{n}" / "mmlu_results.json"))
        nodes = n * 2
        edges = len(d["reinforce"]["final_edge_probs"])
        rl_time = d["reinforce"]["train_time_s"]
        ea_time = d["ea"]["train_time_s"]
        rl_acc  = d["reinforce"]["accuracy"]
        ea_acc  = d["ea"]["accuracy"]
        rows.append([
            f"{n}T{n}A",
            str(nodes),
            str(edges),
            fmt_time(rl_time),
            fmt_time(ea_time),
            f"{rl_acc:.2f}",
            f"{ea_acc:.2f}",
        ])

    col_labels = [
        "Config",
        "#Nodes",
        "#Potential\nEdges",
        "REINFORCE\nOpt. Time",
        "EA\nOpt. Time",
        "REINFORCE\nAccuracy",
        "EA\nAccuracy",
    ]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2d2d2d")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row colors
    for i in range(1, len(rows) + 1):
        color = "#f0f4f8" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(color)

    plt.title("Adversarial Swarm Experiment Statistics", fontsize=13,
              fontweight="bold", pad=20)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "figure10_stats_table.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
