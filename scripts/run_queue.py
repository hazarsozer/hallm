"""CLI for the campaign run queue (spec 06 §8.3). GPU, manual — never launched by automation.

Typical remote session (from the Mac, ssh'd into the Linux box, ideally under systemd-inhibit):

    uv run python scripts/run_queue.py --queue configs/ladder/queue.txt \
        --data data/ --results results/ladder.jsonl

Interrupt any time (Ctrl-C, reboot, Windows switch): the next invocation resumes mid-run from
runs/ladder/<name>/resume.pt. Use --max-runs 1 for a single-run session, --stop-step N to bound it.
"""

from __future__ import annotations

import argparse

import torch

from hallm.runqueue import drain


def main() -> None:
    ap = argparse.ArgumentParser(description="hallm campaign run queue")
    ap.add_argument("--queue", default="configs/ladder/queue.txt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results/ladder.jsonl")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--max-runs", type=int, default=None, help="finish at most N runs this session")
    ap.add_argument("--stop-step", type=int, default=None, help="pause the current run after step N")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = drain(args.queue, args.data, args.results, device, max_runs=args.max_runs, stop_step=args.stop_step)
    print(f"\nsession complete: {len(rows)} run(s) finished → {args.results}")


if __name__ == "__main__":
    main()
