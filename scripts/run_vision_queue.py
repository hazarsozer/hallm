"""Drain a vision queue file (spec 2026-09-16).

Usage:
  uv run python scripts/run_vision_queue.py --queue configs/runs/queue-vision.txt \
      --data data/in100 --results-dir results/runs
"""

from __future__ import annotations

import argparse

import torch

from hallm.vision_runqueue import drain_vision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--data", default="data/in100")
    ap.add_argument("--results-dir", default="results/runs")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--stop-step", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    drain_vision(args.queue, args.data, args.results_dir, device, args.max_runs, args.stop_step)


if __name__ == "__main__":
    main()
