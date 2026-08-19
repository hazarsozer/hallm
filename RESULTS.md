# Results — W+Wᵀ Weight Sharing in Language Models

All runs: WikiText-103 (GPT-2 BPE, 119.2M train tokens), matched budget of 50,000 steps ×
12,288 tokens/step ≈ 614M tokens per arm, identical hyperparameters and seed (1337),
bf16, single RTX 4070 Super. Only the sharing flags (and, for the iso probes, depth) differ.

## Experiment 1 — Four-arm comparison (shape s30: d=512, L=8)

| arm | sharing | non-emb params | test PPL | Δ vs A0 | <2% viable |
|-----|---------|---------------|----------|---------|------------|
| A0 | none | 25.17M | **26.06** | — | — |
| A1 | ALBERT cross-layer | 3.15M (−87.5%) | 35.63 | +36.7% | ✗ |
| A2 | HaLViT W+Wᵀ | 12.59M (−50.0%) | 29.68 | +13.9% | ✗ |
| A3 | both | 1.57M (−93.7%) | 43.30 | +66.2% | ✗ |

- Forward GFLOPs are identical across arms (56.4): sharing compresses **storage, not compute**.
- **W+Wᵀ dominates cross-layer sharing per parameter saved** (~0.28% PPL per % params removed
  vs ~0.42% for ALBERT).
- Composition (A3) is roughly additive in log-PPL — no catastrophic interaction between the axes.
- No arm meets the <2% degradation envelope at this budget.

## Experiment 2 — Iso-parameter / iso-compute 2×2 (does sharing buy free depth?)

A2-iso doubles depth with shared weights so its *stored* parameter count matches A0's;
A0-deep is the unshared control at the same depth (its "virtual" size).

| model | depth | stored non-emb | GFLOPs | test PPL | Δ vs A0 |
|-------|-------|---------------|--------|----------|---------|
| A0 (unshared) | 8 | 25.17M | 56.4 | 26.06 | — |
| A2-iso (W+Wᵀ) | 16 | 25.18M | 86.5 | 27.01 | +3.6% |
| A0-deep (unshared) | 16 | 50.35M | 86.5 | **23.98** | **−8.0%** |

- Depth genuinely pays at this budget (−8.0% PPL for the unshared 50M model).
- **The sharing tax is stable at ~13–14%** whether measured at matched shape
  (29.68 vs 26.06, +13.9%) or at iso-compute (27.01 vs 23.98, +12.7%).
- At **iso-storage**, the shallow unshared model still wins (26.06 < 27.01): W+Wᵀ-bought
  depth does not beat spending the same memory on unshared width at this scale.
- Notable: A2-iso reached a *lower final train loss* than A0 (3.29 vs 3.81) while testing
  worse — the shared-deep model fits better but generalizes worse (all runs use dropout 0.0).

## Experiment 3 — Scaling ladder (Term 2, IN PROGRESS)

Campaign spec: `wiki/roadmap/06-scaling-campaign.md`. Depth-scaled ladder at d=512 testing
**H-S: the sharing tax shrinks with scale.** New runs use the identical protocol (seeds 1338/1339
added); rows for L8/L16 seed 1337 are Experiments 1–2 restated in ladder form. Raw rows:
`results/ladder.jsonl`; frozen per-run manifests: `results/manifests/`.

| rung | seed | unshared PPL | shared (W+Wᵀ) PPL | tax |
|------|------|-------------|-------------------|-----|
| L4 (12.59M non-emb) | 1337 | 29.14 | 33.50 | **+14.9%** |
| L4 | 1338 | 29.07 | *(paused at ~4k steps)* | — |
| L8 (25.17M) | 1337 | 26.06 | 29.68 | +13.9% |
| L16 (50.35M) | 1337 | 23.98 | 27.01 | +12.7% |

- **Trend so far (single-seed): 14.9% → 13.9% → 12.7%, monotone in the H-S-predicted direction.**
  The new L4 point extends the line down-scale, where the hypothesis predicts the largest tax.
- Early seed-noise signal: L4-A0 across seeds 1337/1338 differs by only 0.26% PPL — if that
  spread holds, the rung-to-rung tax differences (~1 point each) are well outside noise.
- Status 2026-08-19 evening: 3 of 14 queued runs complete; queue paused (resumable mid-run);
  remaining: L4-A2-s1338 (from ~4k), L4 pair s1339, L8/L16 pairs seeds 1338/1339.

## Conclusions

1. HaLViT's W+Wᵀ mechanism **transfers to autoregressive LMs** as the best-in-class sharing
   scheme tested: a predictable ~14% PPL cost for −50% non-embedding storage.
2. It is a **graceful-degradation tool, not a superior parameter allocation**: the Pareto
   frontier at this scale is set by unshared models.
3. Cross-layer (ALBERT) sharing is substantially more damaging per parameter saved, and
   combining both axes inherits both costs additively.

## Caveats & follow-ups (Term 2)

Single seed; single scale (25–50M); 614M-token budget (shared arms are relatively
over-trained per stored parameter). Planned: multi-seed error bars; longer budgets;
dropout > 0 for the iso probe (the train/test gap suggests regularization may narrow it);
partial-depth sharing (official HaLViT code shares only later stages); cross-validation
against the independent implementation in alpericon/wplusw-lm.

Full experimental narrative and literature context: `wiki/analyses/four-arm-results-2026-08.md`.
