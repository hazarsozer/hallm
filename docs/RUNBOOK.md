# Runbook

How to set up a machine, run experiments, and keep the record straight. The rules behind these steps
are in [`COLLABORATOR.md`](../COLLABORATOR.md). They exist because silent deviations destroy results.

## 1. Setup

```bash
git clone git@github.com:hazarsozer/hallm.git && cd hallm
uv sync                                      # never pip; never up/downgrade a dependency
uv run pytest -q                             # CPU-only, no data needed
uv sync --group data --group analysis        # extra groups for data prep, ImageNet, symmetry
```

Python and torch versions are pinned by `uv.lock`. On Windows, run the same commands in PowerShell.

## 2. Data

Verify every corpus by SHA-256 against its `SOURCE.json` (or the hashes in
`scripts/hf_migrate_legacy.py` for WikiText) before any run. A mismatch means stop.

| corpus | get it | size |
|---|---|---|
| WikiText-103 bins | HF `hallm-thesis/hallm-wikitext103`, path `data/` (`hf download hallm-thesis/hallm-wikitext103 --include 'data/*' --local-dir .`), or rebuild with `scripts/run_real_training.py prepare` | ~0.25 GB |
| FineWeb-Edu bins | `uv run --group data python scripts/prepare_fineweb.py` (reads `sample/10BT` shards 000–003, holds out 013) | ~5.2 GB |
| ImageNet-1k memmaps | `uv run --group data --group analysis python scripts/prepare_imagenet1k.py --out data/in1k` | 65 GB, ~6 GB download |
| ImageNet-100 memmaps | `uv run --group data --group analysis python scripts/prepare_imagenet100.py` | 6.6 GB |
| LAMBADA, BLiMP | one-time fetch; see the `scripts/capability_eval.py` docstring | small |

**HF downloads stalling in `SYN-SENT`:** some networks blackhole IPv6 to the HF CDN. The ImageNet
prepare scripts take `--ipv4`. For other scripts, force IPv4 by patching `socket.getaddrinfo` (see
`_force_ipv4` in `scripts/prepare_imagenet1k.py`).

## 3. Running experiments

Configs are generated, never hand-edited (`scripts/gen_ladder_configs.py` and friends). A queue file
lists configs in order, and the runner drains it.

```bash
# language models
uv run python scripts/run_queue.py --queue configs/runs/<queue>.txt \
    --data <data | data/fineweb> --results-dir results/runs
# vision
uv run python scripts/run_vision_queue.py --queue configs/runs/<queue>.txt \
    --data data/in1k --results-dir results/runs
```

- **Interrupting is safe.** Checkpoints every 1000 steps; the next invocation resumes from
  `runs/<kind>/<run-id>/resume.pt` with optimizer state, data order and RNG restored. Relaunch the
  same command, and change nothing between sessions of one run.
- Bound a session with `--max-runs N` or `--stop-step N`.
- **Keep the machine awake.** A suspend killed the first-ever run at step 20.4k. On Linux, wrap the
  command in `systemd-inhibit --what=sleep:idle --who=hallm --why=q sh -c '…'` (it is polkit-denied
  over non-interactive SSH). On Windows, set the power plan to never sleep while plugged in.
- **Pairs never split** across machines, code versions or config changes. A pair is two runs
  identical except for the sharing flag. If you can't finish both on one setup, don't start.
- **8 GB GPUs:** the L8 pair may OOM on a 3070 Ti. Any workaround (e.g. smaller micro-batches with
  more `grad_accum` at the same tokens per step) must be identical for both arms of a pair, and it
  changes the config, so it gets new configs and a note in the manifest. Never patch a run mid-way.
- **Cloud:** use `resume.pt`. Colab sessions time out; an hourly rented GPU is better for multi-day
  runs. Copy `results/` and `runs/<run-id>/model.pt` off before the instance dies.

Run ID grammar: `L<depth>-A<arm>-s<seed>[-suffix]` for LMs, `V<depth>-A<arm>-s<seed>[-suffix]` for
ViTs. Arms: `A0` none · `A1` ALBERT · `A2` W+Wᵀ · `A3` both · `A2ffn` · `A2attn` · `A1u<k>` looped (k
blocks cycled) · `A1u<k>t|n|a` transposed loop. Suffixes: `-fw` FineWeb-Edu, `-in1k` ImageNet-1k,
`-p5k` probe, `-lr2x`, `-d720`, etc. Suffixed runs stay out of the main ladder tables.

## 4. After runs land

```bash
uv run python scripts/build_reports.py           # LM tables → results/reports/*.md
uv run python scripts/build_vision_report.py     # vision table → results/reports/vision.md
uv run python scripts/capability_eval.py --checkpoints 'runs/ladder/*/*.pt' \
    --lambada data/lambada_test.jsonl --blimp data/blimp --data data/ --out results/
uv run python scripts/eval_split.py …            # held-out test PPL for finished checkpoints
uv run --group analysis python scripts/ffn_symmetry.py --help   # FFN rotation share
```

- `results/runs/<run-id>.json` is the source of truth; commit it with its manifest
  (`results/manifests/<run-id>.json`). Reports are generated: rebuild, never hand-edit.
- **Appending analysis rows:** check the row count of `results/analysis/*.json` before and after.
  A past session silently truncated 20 rows.
- Write an outcome doc in `docs/analysis/` for any study with pre-registered predictions, and score
  every prediction, including the ones that fail.

## 5. Checkpoints (HF)

```bash
uv run python scripts/hf_sync.py …       # add-only upload of runs/<run-id>/model.pt + manifest
uv run python scripts/hf_fetch.py …      # download for probes/evals elsewhere
```

HF rules: only **add** under `checkpoints/<run-id>/`. Never delete, move, rename or overwrite, and
never change repo settings. Uploads need a token with **write scope on the `hallm-thesis` org**; a
token that predates the org transfer can read but not write. The repo is private and must stay so.

## 6. Pre-registration workflow (how every study so far ran)

1. **Spec** in `docs/superpowers/specs/<date>-<topic>-design.md`: question, arms, decision rules,
   numbered predictions, and what each outcome would mean. Written *before* any run.
2. **Plan** in `docs/superpowers/plans/`: tasks, tests first.
3. **Run**, then **outcome** in `docs/analysis/`: predictions scored, what it licenses and what it
   doesn't.
4. **Decision** entry in `docs/DECISIONS.md` if the question, method or a claim changed.
5. If a later result breaks an earlier claim, **mark the old one withdrawn where it is stated**, and
   don't delete it.

## 7. Known traps

- Manifests record `deterministic: true` as a request. Flash Attention's backward is
  non-deterministic, so runs are not bit-reproducible, and the `determinism` block records what
  happened.
- Four checkpoints have no weights (`L7-A0-s1337`, `L8-A0-s1339`, `L8-A2-s1339`,
  `L8-A2attn-s1339`). Probes skip them.
- Every perplexity labelled "test" before 2026-09-14 was validation.
- Tokens per step for the LM ladder is **12,288** (batch 12 × accum 2 × ctx 512), not 24,576.
  One spec got this wrong (DECISIONS 2026-09-27).
- Don't run other GPU or CPU-heavy work alongside a run: a concurrent eval slowed a 1.7 h run to 2.6 h.
