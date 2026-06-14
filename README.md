# GPTSwarm: Replacing REINFORCE with Evolutionary Algorithms

**Deep Learning Practical — University of Groningen, June 2026**  
Baran Akkanat · Doga Turak · Madalina Gavat

Reproduction of [GPTSwarm (Zhuge et al., ICML 2024)](https://arxiv.org/abs/2402.16823) with a replacement of the REINFORCE edge optimizer by an Evolutionary Algorithm (EA). Evaluated on the adversarial MMLU benchmark using Llama 3.1:8b locally via Ollama and on the Habrok GPU cluster.

---

## Project Structure

```
DLPGPT/
├── graph/                  # Core graph execution layer
│   ├── graph.py            # Graph, topological sort, active-node pruning
│   ├── node.py             # Base node class
│   ├── llm_node.py         # LLM query node (Ollama backend)
│   ├── edge.py             # Edge with learned probability / binary state
│   ├── operations.py       # IO, Adversarial, MajorityVote node types
│   └── token_tracker.py    # Token usage tracking
├── swarm/
│   ├── swarm.py            # REINFORCE-based edge optimizer
│   └── ea_swarm.py         # EA-based edge optimizer (our contribution)
├── dataset/
│   └── mmlu.py             # MMLU loader (HuggingFace datasets)
├── experiments/
│   ├── config.py           # Central hyperparameter config
│   ├── run_experiment.py   # Main experiment runner (all 5 conditions)
│   ├── run_rl_only.py      # REINFORCE-only runner
│   ├── aggregate_results.py# Merge per-config JSON results
│   ├── job_1T1A.sh         # SLURM job scripts
│   ├── job_3T3A.sh
│   ├── job_5T5A.sh
│   ├── job_7T7A.sh
│   └── submit_all.sh       # Submit all SLURM jobs at once
├── results_server/         # Final experiment outputs
│   ├── new_figures/        # All paper figures
│   └── n_pairs_*/          # Per-config JSON results
├── requirements.txt
└── README.md
```

---

## Setup

### Local (macOS / Linux)

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd DLPGPT

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install and start Ollama, then pull the model
# https://ollama.com
ollama pull llama3.1:8b
ollama serve   # run in a separate terminal
```

### Habrok (SLURM)

```bash
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load ollama/0.6.0-GCCcore-13.3.0-CUDA-12.6.0

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running Experiments

### Single configuration (local)

```bash
# Runs all 5 conditions: IO, FC, Random, REINFORCE, EA
python experiments/run_experiment.py --n_pairs 3   # 3T3A
python experiments/run_experiment.py --n_pairs 1   # 1T1A
python experiments/run_experiment.py --n_pairs 5   # 5T5A
python experiments/run_experiment.py --n_pairs 7   # 7T7A
```

Output is saved to `results/n_pairs_{N}/mmlu_results.json`.

### All configurations on Habrok (SLURM)

```bash
cd experiments
bash submit_all.sh
```

This submits 4 jobs (`job_1T1A.sh` through `job_7T7A.sh`), each requesting 1 GPU, 16GB RAM, 4 CPUs on the `gpulong` partition with a 3-day time limit.

### Aggregate results across configs

```bash
python experiments/aggregate_results.py
```

---

## Hyperparameters

All hyperparameters are defined in `experiments/config.py`. Key values used in the paper experiments:

| Parameter | Value |
|---|---|
| Model | `llama3.1:8b` |
| N_TRAIN | 100 |
| N_TEST | 100 |
| Seed | 42 |
| REINFORCE steps | 200 |
| REINFORCE batch size | 4 |
| Learning rate | 0.1 |
| Baseline decay | 0.9 |
| EA generations | 15 |
| EA population size | 10 |
| EA mutation rate | 0.2 |
| EA tournament k | 2 |
| EA batch size | 20 |

---

## EA Design

The EA (`swarm/ea_swarm.py`) operates directly on binary adjacency matrices:

- **Genome**: binary vector over all potential edges (1 = active, 0 = pruned)
- **Fitness**: MMLU batch accuracy over `EA_BATCH_SIZE` questions
- **Selection**: tournament selection (k=2) + elitism (best individual always survives)
- **Crossover**: single-point crossover on edge vector
- **Mutation**: bit-flip with probability `EA_MUT_RATE` per edge

---

## Key Implementation Fixes

Three bugs/issues were identified and fixed during reproduction:

1. **Inactive agent execution**: `topological_sort()` was executing pruned agents, increasing token cost by ~40%. Fixed with backward reachability from the output node over active edges only.
2. **Silent total failure**: Ollama server saturation caused all agents to return empty outputs silently. Fixed with explicit failure detection that raises `RuntimeError` when all agents fail.
3. **Token tracking**: Bare `except Exception: pass` suppressed all token tracking errors. Replaced with tiered exception handling.

---

## Results Summary (3T3A, N_TEST=100)

| Method | Accuracy | Tokens |
|---|---|---|
| IO (single agent) | 0.49 | 21,531 |
| Fully Connected | 0.49 | 145,553 |
| Random Connected | 0.30 | 270,184 |
| REINFORCE | 0.50 | 460,291 |
| EA | 0.51 | 4,596,972 |

EA achieves comparable accuracy to REINFORCE at ~10× the token cost at 3T3A, and ~10× at 7T7A.

---

## Citation

```bibtex
@inproceedings{zhuge2024gptswarm,
  title     = {GPTSwarm: Language Agents as Optimizable Graphs},
  author    = {Zhuge, Mingchen and Wang, Wenyi and Kirsch, Louis and
               Faccio, Francesco and Khizbullin, Dmitrii and Schmidhuber, J{\"u}rgen},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year      = {2024},
  series    = {PMLR 235}
}
```
