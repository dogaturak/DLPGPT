import asyncio
import json
import os
import random
import time
from typing import Callable, List

from dataset.mmlu import load_mmlu, make_questions
from graph import CombineAnswerNode, Edge, Graph, LLMNode
from swarm.swarm import Swarm

# ── Config ─────────────────────────────────────────────────────────────────────
N_TOTAL   = 60   # total MMLU questions to load
N_TRAIN   = 30   # questions used for REINFORCE optimization
N_TEST    = 3 # questions used for final evaluation
RL_STEPS  = 30   # REINFORCE training steps
LR        = 0.1  # learning rate (Adam)
# ──────────────────────────────────────────────────────────────────────────────


def build_swarm_graph() -> Graph:
    direct_answer = LLMNode(
        "You are a knowledgeable expert in question answering. I will ask you a question. I will also give you 4 answers enumerated as A, B, C and D. Only one answer out of the offered 4 is correct. You must choose the correct answer to the question. Your response must be one of the 4 letters: A, B, C or D, corresponding to the correct answer. Only one letter (A, B, C or D) is allowed in your answer.",
        operation_description="direct_answer",
    )
    adversarial = LLMNode(
        "Answer a lie to the following question:",
        operation_description="adversarial",
    )
    aggregator = CombineAnswerNode(
        system_prompt=(
            "Two agents have answered a multiple-choice question, each giving a single letter (A, B, C, or D). "
            "Choose the letter that appears most often. If tied, use your best judgement. "
            "Output only a single letter: A, B, C, or D."
        ),
        operation_description="Aggregator",
    )
    graph = Graph(output_node=aggregator)
    for agent in (direct_answer, adversarial):
        graph.add_edge(Edge(agent, aggregator))
    return graph


async def evaluate(
    graph: Graph,
    questions: List[str],
    score_fns: List[Callable],
    label: str = "",
) -> float:
    correct = 0.0
    for i, (q, sfn) in enumerate(zip(questions, score_fns)):
        result = await graph.execute(q, sample=False)
        correct += sfn(result)
        if (i + 1) % 10 == 0 or (i + 1) == len(questions):
            print(f"  [{label}] {i + 1}/{len(questions)}  acc={correct / (i + 1):.3f}")
    return correct / len(questions)


async def main():
    print("Loading MMLU...")
    items = load_mmlu(subject="all", split="test", n=N_TOTAL)
    pairs = make_questions(items)
    train_pairs = pairs[:N_TRAIN]
    test_pairs  = pairs[N_TRAIN:N_TRAIN + N_TEST]
    train_q = [p[0] for p in train_pairs]
    train_s = [p[1] for p in train_pairs]
    test_q  = [p[0] for p in test_pairs]
    test_s  = [p[1] for p in test_pairs]
    print(f"Train: {len(train_q)}  Test: {len(test_q)}\n")

    results = {}

    # ── Baseline 1: Single IO agent ────────────────────────────────────────────
    print("=== [1/4] Baseline: Single IO Agent ===")
    io_node = LLMNode(
        "You are a knowledgeable expert in question answering. I will ask you a question. I will also give you 4 answers enumerated as A, B, C and D. Only one answer out of the offered 4 is correct. You must choose the correct answer to the question. Your response must be one of the 4 letters: A, B, C or D, corresponding to the correct answer. Only one letter (A, B, C or D) is allowed in your answer.",
        operation_description="IOAgent",
    )
    io_graph = Graph(output_node=io_node)
    t0 = time.time()
    acc = await evaluate(io_graph, test_q, test_s, label="IO")
    results["io_baseline"] = {"accuracy": acc, "time_s": round(time.time() - t0, 1)}
    print(f"IO accuracy: {acc:.3f}\n")

    # ── Baseline 2: Fully connected swarm (no optimization) ────────────────────
    print("=== [2/4] Baseline: Fully Connected Swarm ===")
    graph_fc = build_swarm_graph()
    for e in graph_fc.edges:
        e.active = True
    t0 = time.time()
    acc = await evaluate(graph_fc, test_q, test_s, label="FC")
    results["fully_connected"] = {"accuracy": acc, "time_s": round(time.time() - t0, 1)}
    print(f"Fully connected accuracy: {acc:.3f}\n")

    # ── Baseline 3: Randomly connected swarm ──────────────────────────────────
    print("=== [3/4] Baseline: Randomly Connected Swarm ===")
    graph_rand = build_swarm_graph()
    correct = 0.0
    t0 = time.time()
    for i, (q, sfn) in enumerate(zip(test_q, test_s)):
        for e in graph_rand.edges:
            e.active = random.random() < 0.5
        result = await graph_rand.execute(q, sample=False)
        correct += sfn(result)
        if (i + 1) % 10 == 0 or (i + 1) == len(test_q):
            print(f"  [Rand] {i + 1}/{len(test_q)}  acc={correct / (i + 1):.3f}")
    rand_acc = correct / len(test_q)
    results["random_connected"] = {"accuracy": rand_acc, "time_s": round(time.time() - t0, 1)}
    print(f"Random connected accuracy: {rand_acc:.3f}\n")


    # ── REINFORCE optimization ─────────────────────────────────────────────────
    print("=== [4/4] REINFORCE Optimization ===")
    graph_rl = build_swarm_graph()
    for e in graph_rl.edges:
        e.active = True
    swarm = Swarm(graph_rl, lr=LR, baseline_decay=0.9)
    t0 = time.time()
    rl_rewards = await swarm.optimize(
        list(zip(train_q, train_s)),
        n_iterations=RL_STEPS,
        verbose=True,
    )
    train_time = round(time.time() - t0, 1)

    print("\nEdge weights after training:")
    for e in graph_rl.edges:
        e.active = e.probability > 0.5
        print(f"  {repr(e)}  w={e.weight:.3f}  p={e.probability:.3f}  active={e.active}")

    acc = await evaluate(graph_rl, test_q, test_s, label="RL")
    results["reinforce"] = {
        "accuracy": acc,
        "train_rewards": [round(r, 3) for r in rl_rewards],
        "train_time_s": train_time,
        "final_edge_weights": [round(e.weight, 4) for e in graph_rl.edges],
        "final_edge_probs":   [round(e.probability, 4) for e in graph_rl.edges],
    }
    print(f"REINFORCE accuracy: {acc:.3f}\n")

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out_path = "results/mmlu_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}\n")

    print("=" * 42)
    print(f"{'Method':<22} {'Accuracy':>8}  {'Time':>8}")
    print("-" * 42)
    rows = [
        ("IO (single agent)",  "io_baseline"),
        ("Fully Connected",    "fully_connected"),
        ("Random Connected",   "random_connected"),
        ("REINFORCE",          "reinforce"),
    ]
    for label, key in rows:
        r = results[key]
        t = r.get("train_time_s", r.get("time_s", "-"))
        print(f"{label:<22} {r['accuracy']:>8.3f}  {str(t) + 's':>8}")
    print("=" * 42)


if __name__ == "__main__":
    asyncio.run(main())
