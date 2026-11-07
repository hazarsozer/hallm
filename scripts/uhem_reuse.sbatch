#!/bin/bash
# Data-reuse runs (spec 2026-11-07): one seed per 4×A100 node, one per-GPU queue each.
#
#   cd ~/hallm/repo && mkdir -p runs/reuse-logs && sbatch scripts/uhem_reuse.sbatch 1337
#
# (Slurm opens the -o log before the script runs, so the directory must exist at submission.)
# Resubmitting the same seed resumes every unfinished run from its resume.pt. Results land in
# results/runs/<run-id>.json and runs/ladder/<run-id>/ (manifest, metrics, checkpoints).
#SBATCH -A htdm01
#SBATCH -p a100x4q
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --gres=gpu:4
#SBATCH -t 06:00:00
#SBATCH -J reuse
#SBATCH -o runs/reuse-logs/slurm-%j.out

SEED=${1:?usage: sbatch scripts/uhem_reuse.sbatch <seed>}
hostname
nvidia-smi -L
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g ~/hallm/venv/bin/python -u scripts/run_queue.py \
    --queue "configs/runs/queue-reuse-s$SEED-g$g.txt" --data data --results-dir results/runs \
    > "runs/reuse-logs/s$SEED-g$g.log" 2>&1 &
done
wait
grep -h "session complete\|Error\|nan" runs/reuse-logs/s$SEED-g*.log
