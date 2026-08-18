---
query: "First full four-arm matched-budget run: does W+Wᵀ sharing generalize to LMs?"
date: 2026-08-18
sources_consulted: ["wiki/analyses/halvit-vs-albert-cross-layer-sharing", "wiki/sources/halvit-official-code", "wiki/analyses/wplusw-lm-review-2026-08", "runs/results.json (this repo)"]
---

# Four-arm results — first full run (2026-08-18, RTX 4070 Super)

Setup: `sharedlm` harness, shape s30 (d=512, L=8, H=8, ctx=512, GPT-2 BPE vocab 50257), WikiText-103 (119.2M train tokens), matched budget: 50k steps × 12,288 tokens/step ≈ 614M tokens per arm, identical seed/hparams (lr 6e-4 cosine, wd 0.1, bf16). ~2.5h/arm, ~8 GiB peak. Checkpoints + `results.json` + `comparison.md` in `runs/`.

## Results

| arm | sharing | non-emb params | test PPL | Δ% vs A0 | <2% viable |
|-----|---------|---------------|----------|----------|------------|
| A0 | none | 25.17M | 26.06 | — | — |
| A1 | ALBERT cross-layer | 3.15M (−87.5%) | 35.63 | +36.7% | ✗ |
| A2 | HaLViT W+Wᵀ | 12.59M (−50.0%) | 29.68 | +13.9% | ✗ |
| A3 | both | 1.57M (−93.7%) | 43.30 | +66.2% | ✗ |

GFLOPs identical across arms (56.4/fwd) — sharing compresses storage, not compute, as designed.

## Findings

1. **W+Wᵀ generalizes to LMs in the ordering sense, not the free-lunch sense.** HaLViT intra-layer sharing costs +13.9% PPL for −50% non-emb params — much cheaper per parameter saved than ALBERT's +36.7% for −87.5%. Normalizing: A2 loses ~0.28% PPL per % params removed vs A1's ~0.42%.
2. **No arm meets the <2% envelope (C2/NFR-2) at this budget.** The vision-domain result (≈1pt top-1 loss) does not transfer at this scale/budget — the honest headline. Caveats before generalizing: single seed, single scale, 614M-token budget (≈5 tokens/param for A0 but ≈390/param for A3 — shared arms are relatively over-trained, baseline under-trained; Chinchilla-style budget sensitivity is a live confound).
3. **Composition is roughly additive in log-PPL** (A3 ≈ A1+A2 degradations), consistent with the two axes being orthogonal ([[analyses/halvit-vs-albert-cross-layer-sharing]]); no catastrophic interaction, echoing the official code's composed sharing ([[sources/halvit-official-code]]).
4. **Sanity vs literature**: ordering (none < HaLViT < ALBERT < both) matches expectations from [[sources/1909.11942]] (cross-layer degradation grows with depth budget) and HaLViT's own extreme-cross-layer ablation (67.6% top-1 collapse).

## Iso-parameter probe (2026-08-18 afternoon, same day)

Follow-up run `A2-iso`: HaLViT arm at **d=512, L=16** (shape `s30x2`) — stores 25.18M non-emb params, matching A0's 25.17M to within 0.03% (extra per-layer LayerNorms only). Same matched budget. Question: at equal *memory footprint*, does W+Wᵀ-bought depth beat unshared width?

| model | non-emb params | fwd GFLOPs | test PPL | Δ vs A0 |
|-------|---------------|-----------|----------|---------|
| A0 (L=8, unshared) | 25.17M | 56.4 | 26.06 | — |
| A2-iso (L=16, W+Wᵀ) | 25.18M | 86.5 | 27.01 | **+3.6%** |
| A0-deep (L=16, unshared) | 50.35M | 86.5 | **23.98** | **−8.0%** |

**A0-deep control (completed 2026-08-18 evening):** depth genuinely pays at this budget — the unshared 50.3M L=16 model improves PPL by 8.0%. So at iso-*compute* (L=16), W+Wᵀ sharing costs +12.7% (27.01 vs 23.98), closely matching the +13.9% it costs at L=8 — the sharing tax is ~13–14% regardless of arrangement, and at iso-*storage* the shallow unshared model still wins (26.06 vs 27.01). Completed 2×2 (params × sharing): the Pareto frontier at this scale is set by unshared models; W+Wᵀ's value is halving storage at a consistent, predictable ~14% PPL cost — not free depth.

**Finding: iso-param HaLViT does NOT beat the baseline** — +3.6% PPL despite ~1.5× the compute and a *lower final train loss* (3.29 vs ~3.81). The shared-deep model fits the training data better but generalizes slightly worse — the sharing constraint costs expressivity that depth-reuse does not fully buy back at this scale/budget. Honest headline: W+Wᵀ sharing is a graceful *degradation* mechanism (best-in-class per param removed, see table above), not a superior parameter allocation, on WikiText-103 at 25M/614M tokens. Caveats: single seed; the train/test gap suggests regularization (dropout>0, more data) could flip the sign — worth one Term-2 run.

## Term-2 follow-ups suggested by the data

- Longer-budget run (does the A2 gap shrink with more tokens? PPL-vs-steps curves are in each ckpt's history)
- Multi-seed error bars before quoting deltas in the report
- Partial-depth sharing arm (official-code precedent) — may land inside 2% envelope
- Cross-check against Alper's independent run ([[analyses/wplusw-lm-review-2026-08]]) once his artifacts arrive (init confound must be fixed first)

## Incidents (methods note)

First A0 attempt lost at step 20.4k to a system suspend (CUDA context death); full run restarted under `systemd-inhibit`. Vocab-size bug (32000 vs GPT-2's 50257) caught before launch and fixed in `SHAPES` — would have crashed on first high-id token.

## Related
[[analyses/halvit-vs-albert-cross-layer-sharing]], [[analyses/wplusw-lm-review-2026-08]], [[sources/halvit]], [[sources/halvit-official-code]], [[entities/halvit-model]], [[entities/albert-model]], [[concepts/weight-sharing]]
