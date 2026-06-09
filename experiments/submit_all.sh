#!/bin/bash
# submit_all.sh — submit all 4 GPTSwarm experiments to Habrok SLURM
# Usage (from project root): bash experiments/submit_all.sh

set -e
mkdir -p logs

echo "Submitting GPTSwarm experiments..."

JID1=$(sbatch --parsable experiments/job_1T1A.sh)
echo "  Submitted 1T1A  → job $JID1"

JID3=$(sbatch --parsable experiments/job_3T3A.sh)
echo "  Submitted 3T3A  → job $JID3"

JID5=$(sbatch --parsable experiments/job_5T5A.sh)
echo "  Submitted 5T5A  → job $JID5"

JID7=$(sbatch --parsable experiments/job_7T7A.sh)
echo "  Submitted 7T7A  → job $JID7"

# Aggregate plots run only after all 4 complete
AGG_JID=$(sbatch --parsable \
  --job-name=gptswarm_aggregate \
  --output=logs/aggregate_%j.out \
  --error=logs/aggregate_%j.err \
  --time=00:10:00 \
  --partition=regular \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=4G \
  --dependency=afterok:${JID1}:${JID3}:${JID5}:${JID7} \
  --wrap="source .venv/bin/activate && python experiments/aggregate_results.py")

echo "  Submitted aggregate → job $AGG_JID (runs after all 4 complete)"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Logs in:       logs/"
echo "Results in:    results/n_pairs_*/"
