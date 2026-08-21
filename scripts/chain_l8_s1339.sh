#!/usr/bin/env bash
# Second-stage supervisor: take over the GPU when P1 finishes and run the L8 seed-1339 pair.
#
# These two runs were assigned to Alper (issue #1) but his L8 attempt failed on his side, so they
# come back in-house. They are not optional extras: L8-A0-s1339 is the BASELINE that makes the
# seed-1339 P1 ablations computable as a tax, and the pair also closes the L8 rung at three seeds.
#
# Same relaunch-is-safe property as the first supervisor: run_queue infers state from the
# filesystem, so finished runs skip and interrupted ones resume.
set -uo pipefail

REPO="$HOME/Dev/hallm"
LOG="$REPO/chain.log"
QUEUE="configs/runs/queue-l8-s1339.txt"
MAX_STRIKES=3
UV="$HOME/.local/bin/uv"

cd "$REPO" || exit 1
say() { echo "[$(date -Is)] [l8-s1339] $*" >>"$LOG"; }

completed() {
  local n=0
  while read -r cfg; do
    [ -z "$cfg" ] && continue
    local name; name="$(basename "$cfg" .yaml)"
    [ -f "runs/ladder/$name/$name.pt" ] && n=$((n+1))
  done < "$QUEUE"
  echo "$n"
}

say "armed; waiting for the P1 supervisor to finish before touching the GPU"
while pgrep -f "chain_next.sh" >/dev/null 2>&1; do sleep 60; done
say "P1 supervisor exited"

# Belt and braces: never run two trainers at once.
while pgrep -f "scripts/run_queue.py" >/dev/null 2>&1; do sleep 60; done

TOTAL="$(grep -cve '^[[:space:]]*$' "$QUEUE")"
strikes=0
say "supervision begins: $(completed)/$TOTAL complete"

while :; do
  before="$(completed)"
  if [ "$before" -ge "$TOTAL" ]; then
    say "COMPLETE: $before/$TOTAL runs finished"
    break
  fi
  say "launching run_queue (progress $before/$TOTAL, strike $strikes/$MAX_STRIKES)"
  "$UV" run python scripts/run_queue.py \
      --queue "$QUEUE" --data data/ --results-dir results/runs >>queue.log 2>&1
  rc=$?
  after="$(completed)"
  say "run_queue exited rc=$rc; progress $before -> $after"

  "$UV" run python scripts/build_reports.py --results results >>"$LOG" 2>&1 \
    && say "reports rebuilt" || say "WARN: build_reports failed"

  if [ "$after" -le "$before" ]; then
    strikes=$((strikes+1))
    if [ "$strikes" -ge "$MAX_STRIKES" ]; then
      say "ABORT: $MAX_STRIKES passes with no progress — stopping. GPU left idle deliberately."
      exit 1
    fi
    say "no progress; backing off 120s"
    sleep 120
  else
    strikes=0
  fi
done
say "done"
