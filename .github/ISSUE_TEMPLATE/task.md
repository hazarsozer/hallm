---
name: Collaborator task
about: Assign work to a collaborator, discoverable by their agent
title: "T-XXX — short description"
labels: task
assignees: ''
---

<!-- Prose for the human. Explain WHY this matters, in plain language, without
     assuming they follow the mechanism argument. Attribute anything that came
     from outside this repo (WhatsApp, a call) so their agent can tell a
     reported fact from our inference. -->

## What this is

## Why it matters

<!-- Machine-readable block. `scripts/tasks.py` parses this; keep the fence tag `task`.
     Every field an agent would otherwise have to guess belongs here. -->

```task
task_id: T-XXX
type: run                      # run | action | announcement
assignee: alpericon
why: one sentence, plain language
what: one sentence, concrete
runs:
  - id: L4-A2ffn-s1337
    config: configs/runs/L4-A2ffn-s1337.yaml
    est_hours: 4
baseline_for_tax: L4-A0-s1337  # what these numbers are compared against
protocol_reference: L4-A0-s1338  # manifest to check for config drift before reporting
branch: alper/T-XXX-slug
deliverables:
  - results/runs/<run-id>.json
  - results/manifests/<run-id>.json
  - hf:checkpoints/<run-id>/
depends_on: []
```

## Guardrails

Standard collaborator rules apply — see `CLAUDE.md` in the repo root.
Discovery is read-only: `uv run python scripts/tasks.py check` is safe to run at any time,
including mid-training. Nothing starts without an explicit go-ahead.
