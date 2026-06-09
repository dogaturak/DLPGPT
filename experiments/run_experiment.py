"""
run_experiment.py  --  run all 5 conditions for a single N_PAIRS value.

Usage (local or inside a SLURM job):
    python experiments/run_experiment.py --n_pairs 3

Outputs:
    results/n_pairs_{N}/mmlu_results.json
    results/n_pairs_{N}/accuracy_vs_tokens.png
"""

import argparse
import asyncio
import json
import os
import random
import sys
import time
from typing import Callable, List

# ── make sure project root is on sys.path regardless of cwd ───────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset.mmlu import load_mmlu, make_questions
from graph import CombineAnswerNode, Edge, Graph, LLMNode
from graph.token_tracker import tracker
from swarm.swarm import Swarm
from swarm.ea_swarm import EASwarm
from experiments.config import (
    MODEL, N_TOTAL, N_TRAIN, N_TEST, SEED,
    RL_STEPS, LR, BASELINE_DECAY,
    EA_GENS, EA_POP, EA_MUT_RATE, EA_TOURNAMENT_K, EA_BATCH_SIZE,
)


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_swarm_graph(n_pairs: int) -> Graph:
    agents = []
    for i in range(n_pairs):
        agents.append(LLMNode(
            system_prompt=(
                "You are a knowledgeable expert in question answering. "
                "I will ask you a question. I will also give you 4 answers "
                "enumerated as A, B, C and D. Only one answer out of the "
                "offered 4 is correct. You must choose the correct answer to "
                "the question. Your response must be one of the 4 letters: "
                "A, B, C or D, corresponding to the correct answer. Only one "
                "letter (A, B, C or D) is allowed in your answer."
            ),
            model=MODEL,
            operation_description=f"Truthful_{i}",
        ))
    for i in range(n_pairs):
        agents.append(LLMNode(
            system_prompt="Answer a lie to the following question:",
            model=MODEL,
            operation_description=f"Adversarial_{i}",
        ))

    total_agents = n_pairs * 2
    aggregator = CombineAnswerNode(
        system_prompt=(
            f"{total_agents} agents have answered a multiple-choice question, "
            "each giving a single letter (A, B, C, or D). Choose the letter "
            "that appears most often. If tied, use your best judgement. "
            "Output only a single letter: A, B, C, or D."
        ),
        model=MODEL,
        operation_description="Aggregator",
    )

    graph = Graph(output_node=aggregator)
    for agent in agents:
        graph.add_edge(Edge(agent, aggregator))
    return graph


# ── Evaluation helper ──────────────────────────────────────────────────────────

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
            print(f"  [{label}] {i+1}/{len(questions)}  acc={correct/(i+1):.3f}",
                  flush=True)
    return correct / len(questions)


# ── Main experiment ────────────────────────────────────────────────────────────

async def run(n_pairs: int, out_dir: str) -> dict:
    random.seed(SEED)

    label = f"{n_pairs}T{n_pairs}A"
    print(f"\n{'='*70}")
    print(f"  GPTSwarm experiment  —  {label}  ({n_pairs} truthful + {n_pairs} adversarial agents)")
    print(f"{'='*70}\n", flush=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading MMLU...", flush=True)
    items = load_mmlu(subject="all", split="test", n=N_TOTAL)
    pairs = make_questions(items)
    train_pairs = pairs[:N_TRAIN]
    test_pairs  = pairs[N_TRAIN:N_TRAIN + N_TEST]
    train_q = [p[0] for p in train_pairs]
    train_s = [p[1] for p in train_pairs]
    test_q  = [p[0] for p in test_pairs]
    test_s  = [p[1] for p in test_pairs]
    print(f"Train: {len(train_q)}  Test: {len(test_q)}\n", flush=True)

    results = {"n_pairs": n_pairs, "label": label}

    # ── [1/5] Single IO baseline ───────────────────────────────────────────────
    print(f"=== [1/5] Baseline: Single IO Agent ({label}) ===", flush=True)
    tracker.reset("io_baseline")
    io_node = LLMNode(
        system_prompt=(
            "You are a knowledgeable expert in question answering. "
            "I will ask you a question. I will also give you 4 answers "
            "enumerated as A, B, C and D. Only one answer out of the "
            "offered 4 is correct. You must choose the correct answer to "
            "the question. Your response must be one of the 4 letters: "
            "A, B, C or D, corresponding to the correct answer. Only one "
            "letter (A, B, C or D) is allowed in your answer."
        ),
        model=MODEL,
        operation_description="IOAgent",
    )
    io_graph = Graph(output_node=io_node)
    t0 = time.time()
    acc = await evaluate(io_graph, test_q, test_s, label="IO")
    tok = tracker.snapshot("io_baseline")
    results["io_baseline"] = {
        "accuracy": acc, "time_s": round(time.time() - t0, 1), **tok
    }
    print(f"IO accuracy: {acc:.3f}  tokens: {tok['total_tokens']}\n", flush=True)

    # ── [2/5] Fully connected ──────────────────────────────────────────────────
    print(f"=== [2/5] Baseline: Fully Connected Swarm ({label}) ===", flush=True)
    graph_fc = build_swarm_graph(n_pairs)
    for e in graph_fc.edges:
        e.active = True
    t0 = time.time()
    acc = await evaluate(graph_fc, test_q, test_s, label="FC")
    tok = tracker.snapshot("fully_connected")
    results["fully_connected"] = {
        "accuracy": acc, "time_s": round(time.time() - t0, 1), **tok
    }
    print(f"Fully connected accuracy: {acc:.3f}  tokens: {tok['total_tokens']}\n",
          flush=True)

    # ── [3/5] Randomly connected ───────────────────────────────────────────────
    print(f"=== [3/5] Baseline: Randomly Connected Swarm ({label}) ===", flush=True)
    graph_rand = build_swarm_graph(n_pairs)
    correct = 0.0
    t0 = time.time()
    for i, (q, sfn) in enumerate(zip(test_q, test_s)):
        for e in graph_rand.edges:
            e.active = random.random() < 0.5
        result = await graph_rand.execute(q, sample=False)
        correct += sfn(result)
        if (i + 1) % 10 == 0 or (i + 1) == len(test_q):
            print(f"  [Rand] {i+1}/{len(test_q)}  acc={correct/(i+1):.3f}",
                  flush=True)
    rand_acc = correct / len(test_q)
    tok = tracker.snapshot("random_connected")
    results["random_connected"] = {
        "accuracy": rand_acc, "time_s": round(time.time() - t0, 1), **tok
    }
    print(f"Random connected accuracy: {rand_acc:.3f}  tokens: {tok['total_tokens']}\n",
          flush=True)

    # ── [4/5] REINFORCE ────────────────────────────────────────────────────────
    print(f"=== [4/5] REINFORCE Optimization ({label}) ===", flush=True)
    graph_rl = build_swarm_graph(n_pairs)
    for e in graph_rl.edges:
        e.active = True
    swarm = Swarm(graph_rl, lr=LR, baseline_decay=BASELINE_DECAY)
    t0 = time.time()
    rl_rewards = await swarm.optimize(
        list(zip(train_q, train_s)),
        n_iterations=RL_STEPS,
        verbose=True,
    )
    train_time_rl = round(time.time() - t0, 1)

    print("\nEdge probabilities after REINFORCE:", flush=True)
    for e in graph_rl.edges:
        e.active = e.probability > 0.5
        print(f"  {repr(e)}  w={e.weight:.3f}  p={e.probability:.3f}  active={e.active}",
              flush=True)

    acc = await evaluate(graph_rl, test_q, test_s, label="RL")
    tok = tracker.snapshot("reinforce")
    results["reinforce"] = {
        "accuracy": acc,
        "train_rewards": [round(r, 3) for r in rl_rewards],
        "train_time_s": train_time_rl,
        "final_edge_weights": [round(e.weight, 4) for e in graph_rl.edges],
        "final_edge_probs":   [round(e.probability, 4) for e in graph_rl.edges],
        **tok,
    }
    print(f"REINFORCE accuracy: {acc:.3f}  tokens: {tok['total_tokens']}\n",
          flush=True)

    # ── [5/5] EA ───────────────────────────────────────────────────────────────
    print(f"=== [5/5] EA Optimization ({label}) ===", flush=True)
    graph_ea = build_swarm_graph(n_pairs)
    ea = EASwarm(
        graph_ea,
        pop_size=EA_POP,
        mutation_rate=EA_MUT_RATE,
        tournament_k=EA_TOURNAMENT_K,
    )
    t0 = time.time()
    best_individual, gen_bests = await ea.optimize(
        questions=train_q,
        score_fns=train_s,
        n_generations=EA_GENS,
        batch_size=EA_BATCH_SIZE,
        verbose=True,
    )
    train_time_ea = round(time.time() - t0, 1)

    ea._apply(best_individual)
    print(f"\nBest topology: {best_individual}", flush=True)
    for e in graph_ea.edges:
        print(f"  {repr(e)}  active={e.active}", flush=True)

    acc = await evaluate(graph_ea, test_q, test_s, label="EA")
    tok = tracker.snapshot("ea")
    results["ea"] = {
        "accuracy": acc,
        "best_topology": best_individual,
        "gen_bests": [round(f, 3) for f in gen_bests],
        "train_time_s": train_time_ea,
        **tok,
    }
    print(f"EA accuracy: {acc:.3f}  tokens: {tok['total_tokens']}\n", flush=True)

    # ── Save results ───────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mmlu_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}", flush=True)

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY  —  {label}")
    print(f"{'Method':<26} {'Accuracy':>8}  {'Total Tokens':>13}  {'Time':>8}")
    print("-" * 70)
    for method_label, key in [
        ("IO (single agent)",  "io_baseline"),
        ("Fully Connected",    "fully_connected"),
        ("Random Connected",   "random_connected"),
        ("REINFORCE",          "reinforce"),
        ("EA",                 "ea"),
    ]:
        r = results[key]
        t = r.get("train_time_s", r.get("time_s", "-"))
        print(f"{method_label:<26} {r['accuracy']:>8.3f}  {r['total_tokens']:>13,}  {str(t)+'s':>8}")
    print("=" * 70, flush=True)

    return results


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run GPTSwarm experiment for one N_PAIRS value")
    parser.add_argument("--n_pairs", type=int, required=True,
                        help="Number of truthful/adversarial agent pairs (1, 3, 5, or 7)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: results/n_pairs_<N>)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(ROOT, "results", f"n_pairs_{args.n_pairs}")
    asyncio.run(run(args.n_pairs, out_dir))


if __name__ == "__main__":
    main()
