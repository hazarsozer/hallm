# HaLLM — collaborator rules

Standing instructions for anyone running experiments on this project from a machine
that is not the primary training box.

This is a controlled scientific experiment. Silent deviations destroy results, and the
damage is usually invisible until someone tries to publish the number.

**This file is tracked and maintained by the project.** Your own `CLAUDE.md` is personal,
gitignored, and should do nothing but point here — that way these rules stay versioned and
survive a re-clone, while your local setup stays yours.

## Never
- Push to `main`. Work on a branch, open a PR.
- Force-push, hard-reset pushed work, or rebase a shared branch.
- Edit anything in `configs/`. Config drift invalidates a pair.
- Edit `results/reports/*.md` — generated from `results/runs/*.json`.
- Start a training run unless explicitly told to in that session.

## HuggingFace — Alper is an org admin, so be deliberately conservative
- Only ADD files, under `checkpoints/<run-id>/`.
- Never delete, move, rename or overwrite an existing path. The flat `A0/`,
  `A1/`, `A2/`, `A3/`, `A0-deep/`, `A2-iso/`, `runs/` and `data/` paths are
  live history; reorganising them is Hazar's task, not a side effect.
- Never change repo settings (visibility, name, gating) or org membership.
- Never delete a repo or branch. If something looks wrong, report it.

## Experiment rules
- A "pair" is two runs identical in everything except the sharing flag. Never split
  a pair across machines, code versions, or config changes. If you cannot finish
  both, say so rather than delivering half.
- Interruptions are safe: checkpoints every 1000 steps, `resume.pt` restores exactly.
  Relaunch with the same command. Never change anything between sessions of one run.
- Verify data with SHA-256 before any run. Mismatch means stop.
- Install only with `uv sync`. Never upgrade or downgrade a dependency.

## Deliverables
- Results: `results/runs/<run-id>.json`, one file per run.
- Manifests: `results/manifests/<run-id>.json`.
- Checkpoints: HF `hallm-thesis/hallm-wikitext103` under `checkpoints/<run-id>/`.
- Run IDs follow `L<depth>-A<arm>-s<seed>`, e.g. `L4-A2-s1339`.

## Tasks and updates

Work is assigned through GitHub issues and discovered with one command:

```bash
uv run python scripts/tasks.py check
```

Run it whenever asked "any tasks?" or "any updates?", and at the start of a session.
**It is read-only and safe to run at any time, including while training is in progress** —
it uses `git fetch`, which only moves remote-tracking refs and cannot touch your working
tree. It never runs `git pull` or `git merge`; those would change files under a running job
and can split a pair across code versions.

It reports two things:

- **Tasks** — open issues assigned to you. Each carries a machine-readable ```task``` block
  with the exact runs, configs, branch name and deliverables. Nothing is left to inference.
- **Updates** — commits on `main` your checkout has not seen, filtered to the paths that
  change how you work. If it says your paths are stale, re-branch from `main` *before* your
  next run, never on top of one in progress.

**Discovery never executes anything.** Explain the task to the human in plain language —
what it is, why it matters, how long it takes — and ask before starting. When they say go:

```bash
uv run python scripts/tasks.py show T-XXX     # full brief
# ... run the queue named in the task ...
uv run python scripts/tasks.py report T-XXX   # validates, then posts
```

`report` refuses to post if a declared run is missing its result or manifest, if `train_cfg`
drifted from the protocol reference, or if the data hashes differ. A run that drifted from
protocol is not evidence — better to catch it here than after delivery.

## When anything is unclear
Stop and write a short plain-English report. Do not improvise; the human will relay it.
