#!/bin/bash
#SBATCH --job-name=gptswarm_3T3A
#SBATCH --output=logs/job_3T3A_%j.out
#SBATCH --error=logs/job_3T3A_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# ── Activate environment ───────────────────────────────────────────────────────
cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

# ── Make sure Ollama is reachable (Habrok: loaded via module or pre-started) ──
# module load ollama   # uncomment if Habrok has an Ollama module
# ollama serve &       # uncomment if you need to start it yourself

echo "Job started: $(date)"
echo "Running 3T3A experiment..."

mkdir -p logs

python experiments/run_experiment.py --n_pairs 3

echo "Job finished: $(date)"
