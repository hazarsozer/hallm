# HaLLM — Does W+Wᵀ Weight Sharing Generalize to Language Models?

ITU graduation thesis (Alper Düzgün, Hazar Utku Sözer; advisor: Prof. B. U. Töreyin).
Extends [HaLViT](https://github.com/sp4cing-itu/halvit) (CVPR 2024 W) — "half of the
weights are enough" via W/Wᵀ reuse — from Vision Transformers to autoregressive LMs,
and compares/composes it with ALBERT-style cross-layer sharing under a matched budget.

Companion code: [alpericon/wplusw-lm](https://github.com/alpericon/wplusw-lm) (independent implementation).

## Results (WikiText-103, GPT-2 BPE, d=512 L=8 ctx=512, 614M tokens/arm, seed 1337)

| arm | sharing | non-emb params | test PPL | Δ vs A0 |
|-----|---------|---------------|----------|---------|
| A0 | none | 25.2M | **26.06** | — |
| A1 | ALBERT cross-layer | 3.1M (−87.5%) | 35.63 | +36.7% |
| A2 | HaLViT W+Wᵀ | 12.6M (−50%) | **29.68** | **+13.9%** |
| A3 | both | 1.6M (−93.7%) | 43.30 | +66.2% |

Iso-parameter probe (same 25.2M stored params, 2× depth via sharing):

| model | depth | test PPL | Δ vs A0 |
|-------|-------|----------|---------|
| A2-iso (L=16, W+Wᵀ) | 16 | 27.01 | +3.6% |
| A0-deep (L=16, unshared, 50.3M) | 16 | **23.98** | **−8.0%** |

**Headline finding:** the W+Wᵀ "sharing tax" is a stable ~13–14% PPL whether measured at
matched shape or iso-compute — a predictable price for halving stored weights — but at
iso-storage the unshared model still wins: sharing is best-in-class graceful degradation,
not free capacity.

## Term-2: the sharing tax vs scale (Experiment 3, in progress)

Depth-scaled ladder at d=512 testing whether the tax *shrinks* as models grow
(hypothesis H-S; spec: `wiki/roadmap/06-scaling-campaign.md`). Single-seed so far,
seeds 1338/1339 in flight (distributed across two machines — see issue #1):

| rung | non-emb (unshared) | unshared PPL | shared PPL | tax |
|------|--------------------|-------------|------------|-----|
| L4 | 12.6M | 29.14 | 33.50 | **+14.9%** |
| L8 | 25.2M | 26.06 | 29.68 | +13.9% |
| L16 | 50.3M | 23.98 | 27.01 | +12.7% |

Monotone in the H-S-predicted direction; early seed spread is small (L4-A0:
29.14 vs 29.07 across seeds 1337/1338). Raw rows: one file per run under `results/runs/`; generated comparison tables in
`results/reports/` (rebuild with `uv run python scripts/build_reports.py`).

See [RESULTS.md](RESULTS.md) for the full story, caveats, and follow-ups.
Trained checkpoints: [hazarsozer/hallm-wikitext103](https://huggingface.co/hazarsozer/hallm-wikitext103) (private).

## Quick start

```bash
uv sync
uv run pytest                       # 50 tests: param-count proofs + smoke + pipeline + campaign infra
# real training (GPU):
uv run python scripts/run_real_training.py prepare --train wiki.train.raw --val wiki.valid.raw --out data/
uv run python scripts/run_real_training.py run --data data/ --configs configs --out runs/
```

## Term-2 campaign workflow (spec: wiki/roadmap/06-scaling-campaign.md)

One-time: `uv run python scripts/gen_ladder_configs.py` (configs + queue already committed).

Each GPU session (on the Linux box; auto-suspend must be off — note `systemd-inhibit`
is polkit-denied over non-interactive ssh, so don't wrap the command in it remotely):

    uv run python scripts/run_queue.py --queue configs/runs/queue.txt \
        --data data/ --results-dir results/runs

Interrupt freely — the next invocation resumes from `runs/ladder/<name>/resume.pt`.
Bound a session with `--max-runs 1` or `--stop-step N`.

Capability evals (inference-only, any machine; data fetch documented in the script docstring):

    uv run python scripts/capability_eval.py --checkpoints 'runs/*.pt' 'runs/ladder/*/*.pt' \
        --lambada data/lambada_test.jsonl --blimp data/blimp --data data/ --out results/

## Layout

`src/hallm/` harness · `tests/` proofs · `configs/` arm + iso configs · `results/` run outputs ·
`wiki/` curated research base (analyses, roadmap, cited sources) · `raw/papers/` cited PDFs
