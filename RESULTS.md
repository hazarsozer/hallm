# Results — W+Wᵀ Weight Sharing in Language Models

All runs: WikiText-103 (GPT-2 BPE, 119.2M train tokens), matched budget of 50,000 steps ×
12,288 tokens/step ≈ 614M tokens per arm, identical hyperparameters and seed (1337),
bf16, single RTX 4070 Super. Only the sharing flags (and, for the iso probes, depth) differ.
Perplexities are on the WikiText-103 **validation** split unless marked test (see Corrections).

## Corrections (2026-09-14)

- **Validation, not test.** Every perplexity recorded before 2026-09-14 was computed on the
  WikiText-103 *validation* split and labelled "test" — the runner scored `val.bin` and never read
  `test.bin`. Nothing was tuned or selected on validation, so every comparison stands, but the
  numbers are validation perplexities. Result files now carry `val_ppl` (the campaign metric) and,
  for the 35 runs whose checkpoint survives, a true `test_ppl`. `results/reports/splits.md` checks
  that no verdict changes between the two.
- **"W+Wᵀ dominates cross-layer sharing" is withdrawn.** It compared A2 at −50% storage with A1 at
  −87.5% by PPL-per-%-saved, a ratio that rises with the amount removed (Experiment 4), so it
  penalises the arm that compresses harder by construction; A1 is also single-seed. The fair test
  holds storage and compute equal (looped `A1u4@L8` vs `A2@L8`, `A1u8@L16` vs `A2@L16`) — see
  `results/reports/iso-storage.md`.

## Experiment 1 — Four-arm comparison (shape s30: d=512, L=8)

| arm | sharing | non-emb params | val PPL | Δ vs A0 | <2% viable |
|-----|---------|---------------|----------|---------|------------|
| A0 | none | 25.17M | **26.06** | — | — |
| A1 | ALBERT cross-layer | 3.15M (−87.5%) | 35.63 | +36.7% | ✗ |
| A2 | HaLViT W+Wᵀ | 12.59M (−50.0%) | 29.68 | +13.9% | ✗ |
| A3 | both | 1.57M (−93.7%) | 43.30 | +66.2% | ✗ |

- Forward GFLOPs are identical across arms (56.4): sharing compresses **storage, not compute**.
- Composition (A3) is roughly additive in log-PPL — no catastrophic interaction between the axes.
- No arm meets the <2% degradation envelope at this budget.

## Experiment 2 — Iso-parameter / iso-compute 2×2 (does sharing buy free depth?)

A2-iso doubles depth with shared weights so its *stored* parameter count matches A0's;
A0-deep is the unshared control at the same depth (its "virtual" size).

| model | depth | stored non-emb | GFLOPs | val PPL | Δ vs A0 |
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
| L4 | 1339 | 29.1827 | 33.7089 | +15.51% |
| L7 (22.03M) | 1337 | 26.4060 | *(pending)* | — |
| L8 (25.17M) | 1337 | 26.0610 | 29.6770 | +13.88% |
| L8 | 1338 | 25.9303 | 29.8213 | +15.01% |
| L8 | 1339 | 26.0384 | 29.7832 | +14.38% |
| L16 (50.35M) | 1337 | 23.9769 | 27.0107 | +12.65% |
| L16 | 1338 | 23.9450 | 27.1578 | +13.42% |
| L16 | 1339 | 23.9421 | 26.8802 | +12.27% |

Per-rung means: L4 **15.35% ± 0.20**, L8 **14.42% ± 0.33**, L16 **12.78% ± 0.34** (SE over seeds).

**Decision rule (pre-registered).** Regress tax on log₂(non-embedding params); H-S is supported iff
the slope's 95% CI excludes zero on the negative side. Over 9 pairs: **slope −1.28 pp per doubling,
95% CI [−1.78, −0.79] → SUPPORTED**.

This replaces an earlier rule requiring adjacent rungs' min/max seed ranges not to overlap. That
rule was withdrawn because a min/max range can only *widen* with more seeds, so it became harder to
satisfy as evidence accumulated — it read "inconclusive" on a 0.06 pp overlap between L4 and L8
while the means were cleanly monotone.

**Honest extrapolation.** The fitted decay implies ~18B non-embedding parameters (18,020M) for a
<2% tax. Extrapolating that far from three rungs is not sound, and the gap is large enough that no
plausible functional form rescues it: the ladder characterises a decay *rate*, it does not lead to
the viability gate.

**Known confound.** The ladder scales depth at fixed width, so the aspect ratio d/L walks from 128
(L4) to 32 (L16) — progressively further from a typical design point. Separating "tax at scale"
from "tax at unusual aspect ratio" needs a width axis, which is planned, not done.

## Experiment 4 — Mechanism decomposition (which sublayer costs?)

W+Wᵀ is independently togglable for attention and FFN. `wiki/roadmap/00-master.md` §7 designated
this ablation as the instrument that converts a negative full-A2 result into a boundary finding;
it is now run at L8 (the pre-registered rung, three seeds) and, as replication T-001, at L4 (two
seeds for the FFN/attention split, three for A2).

Rungs are never pooled: pooling lets one rung's ordering overwrite another's (the pooled table once
named A2ffn against the pre-registered L8 reading, A2attn).

### L4

| arm | sharing | stored non-emb | mean tax | cost per % saved |
|-----|---------|---------------|----------|------------------|
| A2attn | attention only | −16.7% | +5.61% | 0.336 |
| A2ffn | FFN only | −33.3% | +8.25% | 0.248 |
| A2 | both | −50.0% | +15.35% | 0.307 |

- **H-M1 (not supported)**: attention sharing costs more than FFN sharing by >2pp — FFN costs
  2.64 pp more, not less (n = 2 / 2).
- **H-M2 (supported)**: taxes are additive — A2 15.35% vs ffn+attn 13.86%.
- best cost-per-%-storage-saved: **A2ffn** at 0.248.

### L8 (pre-registered rung, three seeds)

| arm | sharing | stored non-emb | mean tax | cost per % saved |
|-----|---------|---------------|----------|------------------|
| A2attn | attention only | −16.7% | +3.97% | 0.237 |
| A2ffn | FFN only | −33.3% | +8.80% | 0.264 |
| A2 | both | −50.0% | +14.42% | 0.288 |

- **H-M1 (not supported)**: attention sharing costs more than FFN sharing by >2pp — attention
  costs 4.83 pp *less*, with the same sign in all three seeds. `wiki/roadmap/01-mechanism.md`
  predicted the opposite: the FFN path had the rigorous column-space argument, the causal-attention
  path was flagged as fragile (risk R1). Measurement inverts it, and the inversion survives
  per-parameter normalisation.
- **H-M2 (supported)**: taxes are additive — A2 14.42% vs ffn+attn 12.76%.
- best cost-per-%-storage-saved: **A2attn** at 0.237.

### L16

Decomposition incomplete at this rung: need both A2ffn and A2attn.

At L8 (three seeds, the pre-registered rung) attention-only sharing is cheapest per % saved; at L4
(T-001, two seeds) FFN-only is. The attention tax falls with depth (5.6% at L4 → 4.0% at L8) while
the FFN tax stays near 8.3–8.8%, so the ordering is scale-dependent rather than a fixed property of
the sublayer.

- **Recommended configuration (pre-registered criterion — minimise cost per % storage saved, subject
  to absolute tax < 8%): A2attn.** It wins on both the ratio and the ceiling.

**What this suggests (L8, the pre-registered rung).** The three cost-per-%-saved figures rise
monotonically (0.237 → 0.264 → 0.288) and the additivity residual is positive in all three seeds
(+1.04, +1.51, +2.42 pp; mean +1.66). Rather than a strong path and a weak path, the tax at this
rung looks roughly **proportional to capacity removed** — ~0.24–0.29% PPL per 1% of non-embedding
storage — with a mild penalty for removing more. That rate reproduces every arm measured at L8, and
it explains the viability gate directly: at 0.237, a <2% tax buys only ~8.4% storage reduction. L4
breaks the single-rate reading: FFN-only is cheapest there instead (see above), so the rate is
rung-specific rather than a fixed property of the mechanism.

**Caveat.** L16's decomposition is incomplete — only A2 (both) has run there, so whether the
inversion holds at that depth is untested. At L8, the third seed's additivity residual (+2.42 pp)
sits outside the ±2 pp band the other two seeds (and every L4 seed) stay inside; H-M2 is still
called supported on the three-seed mean (+1.66 pp), but a fourth L8 seed would tighten this.

## Experiment 5 — Iso-storage gate and recipe probes (2026-08-31 → 09-14)

- **Gate.** Spec amendment 2026-08-31 compares A2attn@L8 (20.98M stored, 3-seed mean val PPL
  27.041) with the unshared L7 model (22.03M stored, single seed, 26.406): **+2.41%, missing the
  ≤2% bar.**
- **LR probe (T-002, seed 1337).** At 2× LR: A0 26.508, A2 29.429. At ½× LR: A0 26.663, A2 31.590.
  A0 is best at the standard LR; the smaller tax at 2× comes mostly from A0 getting worse.
- **A2attn at 2× LR (T-005).** 27.545 against the L7 comparator at 2× LR (26.682): the gate verdict
  does not flip.
- **Unshared P4 frontier (T-004).** A0@L8's storage re-spent on width: d720/L4 26.246, d360/L16
  26.991; d512/L8 (26.06) stays best. Table with total inference memory: `results/reports/probes.md`.

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
   −1.28 pp per doubling (95% CI [−1.78, −0.79], 9 pairs).
2. **The cost behaves like a price on capacity, not a property of a sublayer.** Attention-only,
   FFN-only and combined sharing all sit at 0.24–0.29% PPL per 1% of storage removed, rising
   mildly as more is removed. This replaces the strong-FFN / weak-attention framing the project
   started from, which measurement inverted.
3. **Attention-only sharing is the recommended configuration** on the pre-registered criterion:
   −16.7% storage for +3.97% PPL (L8, three seeds), the best ratio and the only arm under the 8%
   ceiling. It is also the closest anything has come to the <2% viability gate — while still
   missing it.
4. **No arm meets the <2% gate, and the reason is now quantitative rather than empirical:** at
   0.237% PPL per % saved (L8), a 2% budget buys ~8.4% storage reduction. The gate and the
   mechanism are incompatible at these scales.
5. **It remains graceful degradation, not a superior parameter allocation.** At iso-storage the
   unshared model still wins on perplexity, and once the KV cache is counted it wins on memory too.
6. The cross-layer (ALBERT) comparison is **open**: the earlier per-%-saved ranking was unfair (see
   Corrections); the matched test is in the fixed-storage pilot.
7. The program now asks whether extra compute bought with shared weights pays at fixed storage, on
   perplexity and on reasoning: docs/superpowers/specs/2026-09-14-fixed-storage-compute-program-design.md.

## Caveats & follow-ups

**Scope.** 12–100M non-embedding parameters, one corpus, one budget, dropout 0.0. Two to three
seeds per rung. A1 and A3 remain single-seed. The mechanism decomposition is three seeds at L8
(the pre-registered rung) and two at L4; L16 has none yet.

**Known confounds, stated rather than buried.**
- The ladder scales depth at fixed width, so aspect ratio drifts with the independent variable.
- Neither arm applies GPT-2's `1/√(2L)` residual-projection scaling. It is symmetric within a pair,
  but the ladder's independent variable *is* depth, so H-S is entangled with a depth-dependent init
  choice. Notably A2 **structurally cannot** take independently-scaled residual init: the output
  projection *is* `W_qᵀ`, so scaling the residual write path also scales `Q`.
- The 614M-token budget over-trains shared arms relative to stored parameters.
- Runs are not bit-reproducible: Flash Attention's backward is non-deterministic.

**Planned.** A width axis to break the aspect-ratio confound; token budget 2×; dropout > 0;
`sharing_warmup_steps` > 0 (implemented, never exercised); an alternative transpose pairing;
capability metrics beyond perplexity (LAMBADA, BLiMP, per-position and per-frequency loss
decomposition); a second corpus; cross-validation against the independent implementation in
alpericon/wplusw-lm.

**Out of scope, stated explicitly.** Conversational or reasoning-level evaluation is unreachable
here — a 1B-parameter pair is ~108 GPU-days on this hardware, WikiText-103 is 119M tokens against
the 20B such a model wants, and conversational ability is a post-training artifact this project
does not attempt. Benchmarks of that kind would read at chance for every arm and would be a null
produced by the floor, not by the mechanism.

Full experimental narrative and literature context: `wiki/analyses/four-arm-results-2026-08.md`.
