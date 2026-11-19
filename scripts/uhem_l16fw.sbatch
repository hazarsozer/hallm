#!/bin/bash
# L16 FineWeb-Edu budget grid (spec 2026-11-19, issue #20): one seed per 4×A100 node, one per-GPU queue each.
#
#   cd ~/hallm/repo && mkdir -p runs/l16fw-logs && sbatch scripts/uhem_l16fw.sbatch 1337
#
# (Slurm opens the -o log before the script runs, so the directory must exist at submission.)
# Needs data/fineweb with the sha256s of issue #20 §4. Resubmitting the same seed resumes every
# unfinished run from its resume.pt. Results land in results/runs/<run-id>.json and
# runs/ladder/<run-id>/ (manifest, metrics, checkpoints).
#SBATCH -A htdm01
#SBATCH -p a100x4q
#SBATCH -N 1
#SBATCH -n 64
#SBATCH --gres=gpu:4
#SBATCH -t 12:00:00
#SBATCH --exclude=a105
#SBATCH -J l16fw
#SBATCH -o runs/l16fw-logs/slurm-%j.out

SEED=${1:?usage: sbatch scripts/uhem_l16fw.sbatch <seed>}
hostname
nvidia-smi -L
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g ~/hallm/venv/bin/python -u scripts/run_queue.py \
    --queue "configs/runs/queue-l16fw-s$SEED-g$g.txt" --data data/fineweb --results-dir results/runs \
    > "runs/l16fw-logs/s$SEED-g$g.log" 2>&1 &
done
wait
grep -h "session complete\|Error\|nan" runs/l16fw-logs/s$SEED-g*.log
