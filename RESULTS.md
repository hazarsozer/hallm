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
- **Correction (2026-08-20).** An earlier version of this section read "A2-iso reached a lower
  final train loss than A0 (3.29 vs 3.81) while testing worse — fits better, generalizes worse."
  That comparison is **depth-confounded**: A2-iso is L=16 and A0 is L=8, so the lower train loss is
  a depth effect, not a sharing effect. The honest comparison is A2-iso vs **A0-deep** (both L=16),
  and it cannot be recovered — per-run loss curves were never saved. Training now writes a
  `metrics.jsonl` per run, so the train/test gap is answerable from the next cohort onward.
  No claim about sharing and generalization is supported by the current data.

## Experiment 3 — Scaling ladder (Term 2)

Campaign spec: `docs/superpowers/specs/2026-08-20-research-program-design.md`. Depth-scaled
ladder at d=512 testing **H-S: the sharing tax shrinks with scale.** Identical protocol
throughout; L8/L16 seed 1337 are Experiments 1–2 restated in ladder form. Per-run data:
`results/runs/`; frozen manifests: `results/manifests/`; generated tables: `results/reports/`.

| rung | seed | unshared PPL | shared (W+Wᵀ) PPL | tax |
|------|------|-------------|-------------------|-----|
| L4 (12.59M non-emb) | 1337 | 29.1429 | 33.4981 | +14.94% |
| L4 | 1338 | 29.0664 | 33.5997 | +15.60% |
| L8 (25.17M) | 1337 | 26.0610 | 29.6770 | +13.88% |
| L8 | 1338 | 25.9303 | 29.8213 | +15.01% |
| L16 (50.35M) | 1337 | 23.9769 | 27.0107 | +12.65% |
| L16 | 1338 | 23.9450 | 27.1578 | +13.42% |
| L16 | 1339 | 23.9421 | 26.8802 | +12.27% |

Per-rung means: L4 **15.27% ± 0.33**, L8 **14.44% ± 0.57**, L16 **12.78% ± 0.34** (SE over seeds).

**Decision rule (pre-registered).** Regress tax on log₂(non-embedding params); H-S is supported iff
the slope's 95% CI excludes zero on the negative side. Over 7 pairs: **slope −1.27 pp per doubling,
95% CI [−1.96, −0.57] → SUPPORTED**.

This replaces an earlier rule requiring adjacent rungs' min/max seed ranges not to overlap. That
rule was withdrawn because a min/max range can only *widen* with more seeds, so it became harder to
satisfy as evidence accumulated — it read "inconclusive" on a 0.06 pp overlap between L4 and L8
while the means were cleanly monotone.

**Honest extrapolation.** The fitted decay implies ~19B non-embedding parameters for a <2% tax.
Extrapolating that far from three rungs is not sound, and the gap is large enough that no plausible
functional form rescues it: the ladder characterises a decay *rate*, it does not lead to the
viability gate.

**Known confound.** The ladder scales depth at fixed width, so the aspect ratio d/L walks from 128
(L4) to 32 (L16) — progressively further from a typical design point. Separating "tax at scale"
from "tax at unusual aspect ratio" needs a width axis, which is planned, not done.

## Experiment 4 — Mechanism decomposition (which sublayer costs?)

W+Wᵀ is independently togglable for attention and FFN. `wiki/roadmap/00-master.md` §7 designated
this ablation as the instrument that converts a negative full-A2 result into a boundary finding;
it is now run at L8, seeds 1337 and 1338.

| arm | sharing | stored non-emb | mean tax | seed spread | cost per % saved |
|-----|---------|---------------|----------|-------------|------------------|
| A2attn | attention only | −16.7% | **+4.13%** | 0.53 | **0.247** |
| A2ffn | FFN only | −33.3% | +9.03% | 0.14 | 0.271 |
| A2 | both | −50.0% | +14.44% | 1.13 | 0.289 |

- **H-M1 (attention costs more than FFN by >2 pp): NOT SUPPORTED.** Attention costs 4.90 pp *less*,
  with the same sign in both seeds. `wiki/roadmap/01-mechanism.md` predicted the opposite: the FFN
  path had the rigorous column-space argument, the causal-attention path was flagged as fragile
  (risk R1). Measurement inverts it, and the inversion survives per-parameter normalisation.
- **H-M2 (taxes are additive): SUPPORTED.** Residuals +1.04 and +1.51 pp, inside the ±2 pp band.
- **Recommended configuration (pre-registered criterion — minimise cost per % storage saved, subject
  to absolute tax < 8%): A2attn.** It wins on both the ratio and the ceiling.

**What this suggests.** The three cost-per-%-saved figures rise monotonically (0.247 → 0.271 →
0.289) and the additivity residual is positive in both seeds. Rather than a strong path and a weak
path, the tax looks roughly **proportional to capacity removed** — ~0.25–0.29% PPL per 1% of
non-embedding storage — with a mild penalty for removing more. That rate reproduces every arm
measured, and it explains the viability gate directly: at 0.247, a <2% tax buys only ~8% storage
reduction.

**Caveat.** Two seeds, one rung. Whether the inversion is constant or scale-dependent is untested;
the L4 replication is the obvious next step.

## Memory accounting

Sharing compresses weights only — the KV cache is `2·d·L` per token regardless.

| | weights | KV @ ctx512×1 | KV @ ctx2048×8 | KV @ ctx8192×8 |
|---|---|---|---|---|
| A0 L8 | 50.3 MB | 8.4 MB | 268 MB | 1074 MB |
| A2-iso L16 | 50.3 MB | 16.8 MB | 537 MB | 2148 MB |

The iso-storage swap in Experiment 2 therefore holds weight memory flat and **doubles** the cache:
net **+8 MB worse at ctx512×1, +1074 MB worse at ctx8192×8**. And the ceiling on the whole idea is
low — non-embedding weights are 45.7% of inference memory at ctx512×1, 13.6% at ctx2048×8 and 4.3%
at ctx8192×8, so a −50% scheme can save at most 22.8% / 6.8% / **2.1%** respectively.

The defensible claim is about **stored, downloaded and loaded weight size** at small batch and short
context — not serving memory in general.

## Conclusions

1. **W+Wᵀ transfers to autoregressive LMs**, and its cost is predictable rather than
   catastrophic: ~14% PPL for −50% non-embedding storage, decaying with scale at
   −1.27 pp per doubling (95% CI [−1.96, −0.57]).
2. **The cost behaves like a price on capacity, not a property of a sublayer.** Attention-only,
   FFN-only and combined sharing all sit at 0.25–0.29% PPL per 1% of storage removed, rising
   mildly as more is removed. This replaces the strong-FFN / weak-attention framing the project
   started from, which measurement inverted.
3. **Attention-only sharing is the recommended configuration** on the pre-registered criterion:
   −16.7% storage for +4.13% PPL, the best ratio and the only arm under the 8% ceiling. It is also
   the closest anything has come to the <2% viability gate — while still missing it.
4. **No arm meets the <2% gate, and the reason is now quantitative rather than empirical:** at
   0.247% PPL per % saved, a 2% budget buys ~8% storage reduction. The gate and the mechanism are
   incompatible at these scales.
5. **It remains graceful degradation, not a superior parameter allocation.** At iso-storage the
   unshared model still wins on perplexity, and once the KV cache is counted it wins on memory too.
6. Cross-layer (ALBERT) sharing is substantially more damaging per parameter saved (0.420), and
   composing both axes inherits both costs additively.

## Caveats & follow-ups

**Scope.** 12–100M non-embedding parameters, one corpus, one budget, dropout 0.0. Two to three
seeds per rung. A1 and A3 remain single-seed. The mechanism decomposition is two seeds at one rung.

**Known confounds, stated rather than buried.**
- The ladder scales depth at fixed width, so aspect ratio drifts with the independent variable.
- Neither arm applies GPT-2's `1/√(2L)` residual-projection scaling. It is symmetric within a pair,
  but the ladder's independent variable *is* depth, so H-S is entangled with a depth-dependent init
  choice. Notably A2 **structurally cannot** take independently-scaled residual init: the output
  projection *is* `W_qᵀ`, so scaling the residual write path also scales `Q`.
- The 614M-token budget over-trains shared arms relative to stored parameters.
- Runs are not bit-reproducible: Flash Attention's backward is non-deterministic.

**Planned.** L4 replication of the mechanism decomposition; a width axis to break the aspect-ratio
confound; token budget 2×; dropout > 0; `sharing_warmup_steps` > 0 (implemented, never exercised);
an alternative transpose pairing; capability metrics beyond perplexity (LAMBADA, BLiMP, per-position
and per-frequency loss decomposition); a second corpus; cross-validation against the independent
implementation in alpericon/wplusw-lm.

**Out of scope, stated explicitly.** Conversational or reasoning-level evaluation is unreachable
here — a 1B-parameter pair is ~108 GPU-days on this hardware, WikiText-103 is 119M tokens against
the 20B such a model wants, and conversational ability is a post-training artifact this project
does not attempt. Benchmarks of that kind would read at chance for every arm and would be a null
produced by the floor, not by the mechanism.

Full experimental narrative and literature context: `wiki/analyses/four-arm-results-2026-08.md`.
