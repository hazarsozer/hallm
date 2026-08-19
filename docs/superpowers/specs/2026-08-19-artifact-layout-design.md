# Artifact layout design — GitHub repo + HuggingFace store

**Date:** 2026-08-19 · **Status:** approved (Hazar), execution pending
**Scope:** directory/naming schema for `hazarsozer/hallm` (GitHub) and
`hazarsozer/hallm-wikitext103` (HuggingFace), plus the one-pass migration plan.

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

## HuggingFace layout (`hazarsozer/hallm-wikitext103`, private)

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
configs/runs/<run-id>.yaml   # the 18 ladder configs, moved; arm1/arm3 renamed
                             # L8-A1-s1337 / L8-A3-s1337
configs/runs/queue.txt
configs/smoke.yaml
results/runs.jsonl           # single ledger, one row per run (ladder.jsonl renamed;
                             # Experiment 1–2 rows backfilled from existing JSONs)
results/manifests/<run-id>.json
results/reports/<experiment>.md   # existing prose comparisons, moved untouched
```

- `arm0_none.yaml`, `arm2_halvit.yaml`, `iso/`, `iso2/` retire — their ladder
  twins (`L8-A0-s1337`, `L8-A2-s1337`, `L16-*-s1337`) already exist; history
  stays in git.
- `runs/` remains gitignored runtime scratch; the box's `runs/ladder/` tree is
  not restructured.
- Code/doc updates in the same commit: `run_queue.py` default paths, README,
  RESULTS.md path references, eval-driver checkpoint discovery.

## Migration plan (one pass per store, ordered by in-flight constraints)

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

## Verification

- HF: post-migration file listing matches the schema exactly; hash comparison
  old-vs-new before any delete; a fresh `checkpoints/<id>/model.pt` download
  loads and evals to the recorded PPL for one spot-check run.
- GitHub: `uv run pytest` green after the rename commit; `run_queue.py` dry-run
  resolves the new queue path; grep shows no stale `configs/ladder` or
  `runs/ladder` references outside `wiki/` history pages.
