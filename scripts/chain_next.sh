#!/usr/bin/env bash
# Overnight supervisor: finish the ladder, deploy P0 code, then drive P1 to completion.
#
# Why a supervisor and not a single launch: run_queue exits on an unhandled crash (CUDA fault,
# OOM, driver hiccup), and an unattended GPU that stops at 01:00 loses the whole night. Relaunching
# is always safe because run_queue infers state from the filesystem — finished runs are skipped,
# interrupted ones resume from resume.pt — so a relaunch is idempotent by construction.
set -uo pipefail

REPO="$HOME/Dev/hallm"
STAGE="$HOME/hallm-staging"
LOG="$REPO/chain.log"
WAIT_FOR="L16-A2-s1339"
QUEUE="configs/runs/queue-p1.txt"
MAX_STRIKES=3          # consecutive passes with zero progress before giving up
UV="$HOME/.local/bin/uv"

cd "$REPO" || exit 1
say() { echo "[$(date -Is)] $*" >>"$LOG"; }

completed() {  # count queue entries whose final checkpoint exists
  local n=0
  while read -r cfg; do
    [ -z "$cfg" ] && continue
    local name; name="$(basename "$cfg" .yaml)"
    [ -f "runs/ladder/$name/$name.pt" ] && n=$((n+1))
  done < "$QUEUE"
  echo "$n"
}

total_entries() { grep -cve '^[[:space:]]*$' "$QUEUE"; }

# ---- 1. wait for the in-flight ladder run -------------------------------------------------
say "supervisor armed; waiting for run_queue to exit (expecting $WAIT_FOR)"
while pgrep -f "scripts/run_queue.py" >/dev/null 2>&1; do sleep 60; done
say "run_queue exited"

if ! grep -q "\"run\": \"$WAIT_FOR\"" results/ladder.jsonl 2>/dev/null \
   && [ ! -f "results/runs/$WAIT_FOR.json" ]; then
  say "ABORT: $WAIT_FOR result absent — the run did not finish cleanly. P1 NOT launched."
  exit 1
fi
say "$WAIT_FOR result present — ladder complete"

# ---- 2. deploy P0 instrumentation --------------------------------------------------------
for d in src scripts configs; do
  [ -d "$STAGE/$d" ] || { say "ABORT: staging dir $STAGE/$d missing"; exit 1; }
done
for d in src scripts configs; do
  cp -r "$STAGE/$d/." "$REPO/$d/" || { say "ABORT: deploy of $d failed"; exit 1; }
done
find "$REPO/src" "$REPO/scripts" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
HALLM_GIT_COMMIT="$(cat "$STAGE/COMMIT" 2>/dev/null || echo unknown)"
export HALLM_GIT_COMMIT
say "P0 code deployed at commit $HALLM_GIT_COMMIT"

# ---- 3. fold the legacy ledger into per-run result files ---------------------------------
"$UV" run python scripts/migrate_results.py --results results >>"$LOG" 2>&1 \
  && say "legacy results migrated to results/runs/" \
  || say "WARN: migrate_results failed (non-fatal; P1 writes per-run files directly)"

# ---- 4. supervise P1 to completion -------------------------------------------------------
TOTAL="$(total_entries)"
strikes=0
say "P1 supervision begins: $(completed)/$TOTAL complete"

while :; do
  before="$(completed)"
  if [ "$before" -ge "$TOTAL" ]; then
    say "P1 COMPLETE: $before/$TOTAL runs finished"
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
      say "ABORT: $MAX_STRIKES consecutive passes with no progress — stopping to avoid a crash loop."
      say "ABORT: inspect queue.log; GPU left idle deliberately."
      exit 1
    fi
    say "no progress; backing off 120s then retrying"
    sleep 120
  else
    strikes=0
  fi
done

say "supervisor done"
