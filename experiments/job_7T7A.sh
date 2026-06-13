#!/bin/bash
#SBATCH --job-name=gptswarm_7T7A
#SBATCH --output=logs/job_7T7A_%j.out
#SBATCH --error=logs/job_7T7A_%j.err
#SBATCH --time=1-12:00:00
#SBATCH --partition=gpulong
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

cd $SLURM_SUBMIT_DIR

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load ollama/0.6.0-GCCcore-13.3.0-CUDA-12.6.0

source .venv/bin/activate

ollama serve &
OLLAMA_PID=$!
sleep 15

echo "Job started: $(date)"
echo "Running 7T7A experiment..."

mkdir -p logs

PYTHONUNBUFFERED=1 python -u experiments/run_experiment.py --n_pairs 7

kill $OLLAMA_PID
echo "Job finished: $(date)"
