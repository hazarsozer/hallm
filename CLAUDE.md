# HaLLM — collaborator rules (Alper's machine)

This is a controlled scientific experiment. Silent deviations destroy results,
and the damage is usually invisible until someone tries to publish the number.

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

## When anything is unclear
Stop and write a short plain-English report. Do not improvise. Alper will relay it.
