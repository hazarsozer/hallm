"""CLI for the campaign run queue (spec 06 §8.3). GPU, manual — never launched by automation.

Typical remote session (from the Mac, ssh'd into the Linux box, ideally under systemd-inhibit):

    uv run python scripts/run_queue.py --queue configs/ladder/queue.txt \
        --data data/ --results results/ladder.jsonl

Interrupt any time (Ctrl-C, reboot, Windows switch): the next invocation resumes mid-run from
runs/ladder/<name>/resume.pt. Use --max-runs 1 for a single-run session, --stop-step N to bound it.
--stop-step is an ABSOLUTE step index (same semantics as train()'s `stop_step`), not a step count
relative to the current session — it bounds the ONE run active when the session ends; queue entries
behind it are left untouched for the next invocation.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from hallm.runqueue import drain


def main() -> None:
    ap = argparse.ArgumentParser(description="hallm campaign run queue")
    ap.add_argument("--queue", default="configs/runs/queue.txt")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results-dir", default="results/runs",
                    help="directory of per-run result JSONs (one file per run, never a shared ledger)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--max-runs", type=int, default=None, help="finish at most N runs this session")
    ap.add_argument("--stop-step", type=int, default=None,
                     help="pause the in-progress run at this ABSOLUTE step index (not a step count); "
                          "the session ends there without starting the next queue entry")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("SHAREDLM_NO_FLASH_SDP"):
        # workaround for wedged flash-attention backward kernels (2026-08-18 hangs);
        # math/mem-efficient SDP computes the identical attention, just slower
        torch.backends.cuda.enable_flash_sdp(False)
        print("flash SDP disabled (SHAREDLM_NO_FLASH_SDP)")
    failures: list[str] = []
    rows = drain(args.queue, args.data, args.results_dir, device, max_runs=args.max_runs,
                 stop_step=args.stop_step, failures=failures)
    print(f"\nsession complete: {len(rows)} run(s) finished, {len(failures)} failed "
          f"→ {args.results_dir}/<run-id>.json")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
