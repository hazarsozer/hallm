# Artifact layout design — GitHub repo + HuggingFace store

**Date:** 2026-08-19 · **Status:** approved (Hazar) · **executed 2026-08-21**
**Amended:** 2026-08-21 — results are one file per run, not a single ledger (see Amendment A).
**Scope:** directory/naming schema for `hazarsozer/hallm` (GitHub) and
`hallm-thesis/hallm-wikitext103` (HuggingFace), plus the one-pass migration plan.

> The HuggingFace repo moved from the personal namespace `hazarsozer/` into the
> `hallm-thesis` organization after this spec was written. Paths below use the org.

---

## Amendment A — per-run result files supersede the single ledger (2026-08-21)

The GitHub layout below originally specified `results/runs.jsonl`: one append-only ledger,
one row per run. **Rejected by Hazar during execution and replaced with one JSON file per
run** under `results/runs/<run-id>.json`.

The reasoning, recorded because the ledger form is the more obvious design and will look
tempting again:

1. **It cannot be re-derived.** A ledger is simultaneously the record and the view. Per-run
   files make the record primitive and every table a *derived* artifact that is always safe
   to delete and rebuild (`scripts/build_reports.py`).
2. **A single bad write corrupts everything after it.** One interleaved or truncated append
   damages the rows that follow. Per-run writes are atomic (tmp + `os.replace`) and isolated —
   a failure can only ever damage its own run.
3. **Re-running a run is ambiguous.** Appending duplicates the row; rewriting in place means
   parsing and rewriting the whole file. Writing `<run-id>.json` is idempotent by
   construction: a run owns exactly one file and touching it cannot disturb another.
4. **Concurrent writers.** Two machines finishing runs near-simultaneously conflict on one
   file and never on separate ones.

Cost accepted: many small files instead of one, and any consolidated view must be generated.
Both are cheap; the ledger's failure modes were not.

**Consequence for collaborators:** the deliverable named in issue #1
(`results/ladder-alper.jsonl`) is superseded. `scripts/migrate_results.py --ledger <file>`
converts any legacy ledger into per-run files.

## Problem

Three naming eras accumulated side by side:

- **HF:** flat `A0/…A3/`, `A0-deep/`, `A2-iso/` (each `model.pt` + config;
  Experiments 1–2), `runs/ladder/<name>/<name>.pt` (Experiment 3), `data/*.bin`
  (WikiText token streams). Inconsistent checkpoint filenames, no manifest linkage.
- **GitHub:** `configs/` mixes flat `arm*_*.yaml`, `iso/`, `iso2/`, and `ladder/`;
  `results/` mixes per-experiment prose+JSON pairs with the append-only
  `ladder.jsonl` + `manifests/`.

Decision (Hazar, 2026-08-19): **one uniform run-addressed scheme**, old eras
normalized into it — not experiment-numbered namespaces.

## Run ID grammar

Canonical run ID: **`L<depth>-A<arm>-s<seed>`** (d=512, ctx=512, WikiText-103
implied for the current campaign). The ID is a *key*, not a spec — the run's
`manifest.json` is the authority on full configuration.

Legacy runs map into the grammar with zero collisions (they are exactly the
unqueued seed-1337 slot-ins of the ladder):

| legacy | run ID |
|---|---|
| `A0` | `L8-A0-s1337` |
| `A1` | `L8-A1-s1337` |
| `A2` | `L8-A2-s1337` |
| `A3` | `L8-A3-s1337` |
| `A0-deep` | `L16-A0-s1337` |
| `A2-iso` | `L16-A2-s1337` |

### Extension rule (future datasets / training variants)

When a run varies a dimension the current grammar leaves implied, the ID grows an
explicit segment rather than overloading existing names — e.g. width
`L8d768-A2-s1337`, a different dataset `L8-A2-s1337-owt` (suffix per dataset
tag), or a budget variant `L8-A2-s1337-2x`. Rules:

1. Absence of a segment always means the campaign default (d=512, ctx=512,
   WikiText-103, standard budget).
2. New segments are appended, never inserted before arm/seed.
3. The manifest always carries the full config; tooling must key on the ID
   string, never parse dimensions out of it.
4. A new dataset also gets its own `data/<tag>/` tree on HF (see below).

## HuggingFace layout (`hallm-thesis/hallm-wikitext103`, private)

```
data/{train,val,test}.bin           # unchanged (issue #1 in flight)
checkpoints/<run-id>/model.pt       # uniform filename; the dir is the key
checkpoints/<run-id>/config.yaml
checkpoints/<run-id>/manifest.json  # legacy runs: minimal backfill, "retroactive": true
README.md                           # gains the layout + naming table
```

- All six legacy dirs and all `runs/ladder/*` checkpoints fold into
  `checkpoints/<run-id>/`.
- Future datasets: `data/<tag>/…` subtrees (current bins may move to
  `data/wikitext103/` in that same future pass — not now).
- **Deferred:** splitting `data/` into a separate HF dataset repo — revisit only
  after issue #1 closes and its URLs become irrelevant.

## GitHub layout (`hazarsozer/hallm`)

```
configs/runs/<run-id>.yaml     # ladder + ablation configs; arm1/arm3 renamed to
                               # L8-A1-s1337 / L8-A3-s1337
configs/runs/queue*.txt        # one queue per cohort the runner drains
configs/smoke.yaml
results/runs/<run-id>.json     # SOURCE OF TRUTH — one file per run, atomic and
                               # idempotent (Amendment A). Experiment 1–2 rows
                               # backfilled from the legacy JSONs.
results/manifests/<run-id>.json  # frozen provenance, written once at launch
results/reports/*.md           # GENERATED from results/runs/ — disposable, never
                               # hand-edited; existing prose comparisons moved here
COLLABORATOR.md                # tracked executor rules (CLAUDE.md is gitignored
                               # and personal to each machine)
```

- `arm0_none.yaml`, `arm2_halvit.yaml`, `iso/`, `iso2/` retire — their ladder
  twins (`L8-A0-s1337`, `L8-A2-s1337`, `L16-*-s1337`) already exist; history
  stays in git.
- `runs/` remains gitignored runtime scratch; the box's `runs/ladder/` tree is
  not restructured.
- Code/doc updates in the same commit: `run_queue.py` default paths (`--results`
  became `--results-dir`), README, RESULTS.md path references, eval-driver
  checkpoint discovery.
- **`.gitignore` must anchor `runs/` as `/runs/`.** An unanchored pattern matches
  *any* directory named `runs` at any depth, silently excluding both
  `results/runs/` and `configs/runs/` from version control. This actually
  happened during execution and cost a round trip with a collaborator whose
  fresh clone failed 3 tests against configs that were never committed.

## Migration plan (as planned 2026-08-19)

1. **HF — safe immediately.** The box queue never touches HF (uploads are
   manual) and Alper has not accepted the collaborator invite, so nothing is
   mid-download. Copy to new paths via `HfApi`, verify old-vs-new file hashes,
   then delete old paths. Backfill legacy manifests. Update HF README.
2. **Issue #1 — same sitting as (1).** Edit the issue body to the new
   `checkpoints/` paths (data URLs unchanged). Safe precisely because his work
   has not started.
3. **GitHub — after the queue drains** (7 runs remaining as of 2026-08-19
   night). The live queue reads `configs/ladder/queue.txt` on the box;
   renaming configs and rsyncing mid-queue would break it. One commit, then
   rsync to the box in a quiet window.
4. Delete the wiki pin `pin-2026-08-19-repo-layout.md` once (1)–(3) are done.

## Execution record (2026-08-21)

Both stores migrated. Where reality diverged from the plan above, recorded so the
plan's assumptions are not mistaken for what happened:

- **GitHub was migrated mid-queue, not after it drained** (contradicting step 3).
  It was safe because the box runs from an rsynced staging directory, not a git
  checkout, and the deploy step was chained to fire *after* the in-flight run
  finished. Path changes therefore never reached a running job.
- **Step 2 was overtaken.** Alper had already accepted the invite and been made an
  org admin, and had started work. Instead of editing issue #1 in place, issue #2
  was raised describing the migration and what to do about work already in flight.
- **HF migration used server-side `CommitOperationCopy`**, not download/re-upload:
  the Experiment 1–2 checkpoints existed *only* on the Hub, with no local copy
  anywhere. Copy and delete were separate commits so no file was ever absent, and
  deletion was gated on the destination's LFS sha256 matching the source. All 15
  source paths verified; none failed.
- **The repo was public.** The org transfer had silently dropped the private flag —
  an unauthenticated request served the full file listing including `data/train.bin`.
  Flipped to private during this pass; verified externally by a 401.
- **A read-only token blocked the first attempt.** The box's token predated the
  transfer into the org, so it could read the (public) repo but not write. This was
  the single root cause of three separate symptoms: the collaborator's failing HF
  push, the failed visibility flip, and checkpoints never syncing.

## Verification (performed)

- **HF:** 68 files, 21 runs, every run holding exactly `model.pt` + `config.yaml` +
  `manifest.json`; top level is only `checkpoints/`, `data/`, `README.md`,
  `.gitattributes`. Post-upload check confirmed no previously present path was lost.
  Legacy runs carry `"retroactive": true` manifests with environment fields recorded
  as `unknown` rather than guessed.
- **GitHub:** `uv run pytest` green **on a fresh clone**, not merely in a working
  tree — the distinction mattered: an unanchored `.gitignore` pattern had excluded
  every config from version control while the local tree still passed. Grep shows no
  stale `configs/ladder` references outside history pages.
