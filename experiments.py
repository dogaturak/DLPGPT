import asyncio
import json
import os
import random
import time
from typing import Callable, List

from dataset.mmlu import load_mmlu, make_questions
from graph import CombineAnswerNode, Edge, Graph, LLMNode
from swarm.swarm import Swarm
from swarm.ea_swarm import EASwarm

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL = "llama3.1"
N_TOTAL = 60
N_TRAIN = 30
N_TEST = 3
RL_STEPS = 30
LR = 0.1
EA_GENS = 10
EA_POP = 8
# ──────────────────────────────────────────────────────────────────────────────


def build_swarm_graph() -> Graph:
    direct_answer = LLMNode(
        "You are a knowledgeable expert in question answering. I will ask you a question. "
        "I will also give you 4 answers enumerated as A, B, C and D. Only one answer out of "
        "the offered 4 is correct. You must choose the correct answer to the question. Your "
        "response must be one of the 4 letters: A, B, C or D, corresponding to the correct "
        "answer. Only one letter (A, B, C or D) is allowed in your answer.",
        model=MODEL,
        operation_description="direct_answer",
    )
    adversarial = LLMNode(
        "Answer a lie to the following question:",
        model=MODEL,
        operation_description="adversarial",
    )
    aggregator = CombineAnswerNode(
        system_prompt=(
            "Two agents have answered a multiple-choice question, each giving a single letter "
            "(A, B, C, or D). Choose the letter that appears most often. If tied, use your best "
            "judgement. Output only a single letter: A, B, C, or D."
        ),
        model=MODEL,
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
            print(
                f"  [{label}] {i + 1}/{len(questions)}  acc={correct / (i + 1):.3f}")
    return correct / len(questions)


async def main():
    print("Loading MMLU...")
    items = load_mmlu(subject="all", split="test", n=N_TOTAL)
    pairs = make_questions(items)
    train_pairs = pairs[:N_TRAIN]
    test_pairs = pairs[N_TRAIN:N_TRAIN + N_TEST]
    train_q = [p[0] for p in train_pairs]
    train_s = [p[1] for p in train_pairs]
    test_q = [p[0] for p in test_pairs]
    test_s = [p[1] for p in test_pairs]
    print(f"Train: {len(train_q)}  Test: {len(test_q)}\n")

    results = {}

    # ── Baseline 1: Single IO agent ────────────────────────────────────────────
    print("=== [1/5] Baseline: Single IO Agent ===")
    io_node = LLMNode(
        "You are a knowledgeable expert in question answering. I will ask you a question. "
        "I will also give you 4 answers enumerated as A, B, C and D. Only one answer out of "
        "the offered 4 is correct. You must choose the correct answer to the question. Your "
        "response must be one of the 4 letters: A, B, C or D, corresponding to the correct "
        "answer. Only one letter (A, B, C or D) is allowed in your answer.",
        model=MODEL,
        operation_description="IOAgent",
    )
    io_graph = Graph(output_node=io_node)
    t0 = time.time()
    acc = await evaluate(io_graph, test_q, test_s, label="IO")
    results["io_baseline"] = {"accuracy": acc,
                              "time_s": round(time.time() - t0, 1)}
    print(f"IO accuracy: {acc:.3f}\n")

    # ── Baseline 2: Fully connected swarm ─────────────────────────────────────
    print("=== [2/5] Baseline: Fully Connected Swarm ===")
    graph_fc = build_swarm_graph()
    for e in graph_fc.edges:
        e.active = True
    t0 = time.time()
    acc = await evaluate(graph_fc, test_q, test_s, label="FC")
    results["fully_connected"] = {
        "accuracy": acc, "time_s": round(time.time() - t0, 1)}
    print(f"Fully connected accuracy: {acc:.3f}\n")

    # ── Baseline 3: Randomly connected swarm ──────────────────────────────────
    print("=== [3/5] Baseline: Randomly Connected Swarm ===")
    graph_rand = build_swarm_graph()
    correct = 0.0
    t0 = time.time()
    for i, (q, sfn) in enumerate(zip(test_q, test_s)):
        for e in graph_rand.edges:
            e.active = random.random() < 0.5
        result = await graph_rand.execute(q, sample=False)
        correct += sfn(result)
        if (i + 1) % 10 == 0 or (i + 1) == len(test_q):
            print(
                f"  [Rand] {i + 1}/{len(test_q)}  acc={correct / (i + 1):.3f}")
    rand_acc = correct / len(test_q)
    results["random_connected"] = {
        "accuracy": rand_acc, "time_s": round(time.time() - t0, 1)}
    print(f"Random connected accuracy: {rand_acc:.3f}\n")

    # ── REINFORCE optimization ─────────────────────────────────────────────────
    print("=== [4/5] REINFORCE Optimization ===")
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
    train_time_rl = round(time.time() - t0, 1)

    print("\nEdge weights after REINFORCE:")
    for e in graph_rl.edges:
        e.active = e.probability > 0.5
        print(
            f"  {repr(e)}  w={e.weight:.3f}  p={e.probability:.3f}  active={e.active}")

    acc = await evaluate(graph_rl, test_q, test_s, label="RL")
    results["reinforce"] = {
        "accuracy": acc,
        "train_rewards": [round(r, 3) for r in rl_rewards],
        "train_time_s": train_time_rl,
        "final_edge_weights": [round(e.weight, 4) for e in graph_rl.edges],
        "final_edge_probs":   [round(e.probability, 4) for e in graph_rl.edges],
    }
    print(f"REINFORCE accuracy: {acc:.3f}\n")

    # ── EA optimization ────────────────────────────────────────────────────────
    print("=== [5/5] EA Optimization ===")
    graph_ea = build_swarm_graph()
    ea = EASwarm(graph_ea, pop_size=EA_POP, mutation_rate=0.2, tournament_k=2)
    t0 = time.time()
    best_individual, gen_bests = await ea.optimize(
        questions=train_q,
        score_fns=train_s,
        n_generations=EA_GENS,
        batch_size=5,
        verbose=True,
    )
    train_time_ea = round(time.time() - t0, 1)

    ea._apply(best_individual)
    print(f"\nBest topology: {best_individual}")
    for e in graph_ea.edges:
        print(f"  {repr(e)}  active={e.active}")

    acc = await evaluate(graph_ea, test_q, test_s, label="EA")
    results["ea"] = {
        "accuracy": acc,
        "best_topology": best_individual,
        "gen_bests": [round(f, 3) for f in gen_bests],
        "train_time_s": train_time_ea,
    }
    print(f"EA accuracy: {acc:.3f}\n")

    # ── Save & print summary ───────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out_path = "results/mmlu_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}\n")

    print("=" * 50)
    print(f"{'Method':<26} {'Accuracy':>8}  {'Time':>8}")
    print("-" * 50)
    rows = [
        ("IO (single agent)",   "io_baseline"),
        ("Fully Connected",     "fully_connected"),
        ("Random Connected",    "random_connected"),
        ("REINFORCE",           "reinforce"),
        ("EA",                  "ea"),
    ]
    for label, key in rows:
        r = results[key]
        t = r.get("train_time_s", r.get("time_s", "-"))
        print(f"{label:<26} {r['accuracy']:>8.3f}  {str(t) + 's':>8}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
