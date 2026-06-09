"""
Central configuration for all GPTSwarm experiments.
All experiment scripts import from here.
Change values here to affect all runs simultaneously.
"""

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL = "llama3.1:8b"

# ── Dataset ────────────────────────────────────────────────────────────────────
N_TOTAL   = 220    # total MMLU samples to load
N_TRAIN   = 100    # training split (used for optimisation)
N_TEST    = 100    # test split (used for final evaluation)
SEED      = 42

# ── REINFORCE hyperparameters ──────────────────────────────────────────────────
RL_STEPS       = 50
LR             = 0.1
BASELINE_DECAY = 0.9

# ── EA hyperparameters ─────────────────────────────────────────────────────────
EA_GENS        = 15
EA_POP         = 10
EA_MUT_RATE    = 0.2
EA_TOURNAMENT_K = 2
EA_BATCH_SIZE  = 5

# ── Swarm configurations to run ────────────────────────────────────────────────
# Each entry produces one row group in the results tables (Figures 1-3 of report)
# N_PAIRS=1 → 1T1A, N_PAIRS=3 → 3T3A, etc.
EXPERIMENT_CONFIGS = [1, 3, 5, 7]
