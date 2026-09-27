# HaLLM — project brief

The research equivalent of a PRD: what the project asks, why, what counts as an answer, what is out
of scope, and when it is due. For where things stand today, read [`HANDOVER.md`](../HANDOVER.md). For
the numbers, read [`RESULTS.md`](../RESULTS.md).

**Project:** Advanced Model Compression Techniques for Resource-Constrained AI Architectures
(ITU AI & Data Engineering graduation project, YZV 4901E/4902E)
**Team:** Alper Düzgün (project lead from 2026-09-27), Hazar Utku Sözer
**Advisor:** Prof. Dr. Behçet Uğur Töreyin. He asked that Ahmet Nuri Yılmaz (who assembled the
HaLViT code) be kept in the loop, and that every mail CC the whole team.

---

## 1. Background

[HaLViT](https://github.com/sp4cing-itu/halvit) (CVPR 2024 Workshops) halves a Vision Transformer's
weights by storing one matrix `W` per pair and using `Wᵀ` for the second transformation ("W+Wᵀ"
sharing). Its argument: after a nonlinearity, `F(Wx)` leaves the column space of `W`, so `Wᵀ·F(Wx)`
is a genuinely new transformation, not a linear reuse.

The thesis asked whether that carries over to **autoregressive language models**, and how it compares
with ALBERT-style **cross-layer** sharing (the same block reused at several depths). The advisor
locked this direction on 2026-06-12 (Option 2, research only, small scale, no deployment work).

## 2. The question

The question has moved twice. Each move is recorded in [`docs/DECISIONS.md`](DECISIONS.md).

1. **Aug 2026 — the tax.** How much perplexity does W+Wᵀ cost, where does the cost come from, and
   does it shrink with scale? Answered: ~14% perplexity for −50% non-embedding storage at L8,
   falling −1.28 pp per doubling of parameters (H-S supported), roughly proportional to capacity
   removed. No arm meets the charter's ≤2% bar.
2. **Since 2026-09-14 — fixed storage (current).**

> **At a fixed number of stored weights, does spending extra compute through sharing beat the
> unshared model — on perplexity and on reasoning — and which kind of sharing spends it better:
> W+Wᵀ (within a layer) or looping (across layers)?**

The ≤2% bar stays as a reported secondary verdict. Spec:
[`2026-09-14-fixed-storage-compute-program-design.md`](superpowers/specs/2026-09-14-fixed-storage-compute-program-design.md).

3. **Side question, answered 2026-09-18 — is language the problem?** If W+Wᵀ works in HaLViT and not
   here, is the domain the reason? Our own arms were trained as ViTs under our recipe. At matched data
   reuse **vision orders the schemes exactly as language does**, and W+Wᵀ loses in both. HaLViT's gain
   is attributable to its recipe, scale or augmentation rather than to the domain.

## 3. Hypotheses and decision rules (pre-registered)

| id | claim | rule | status |
|---|---|---|---|
| H-S | the sharing tax shrinks with scale | slope of tax on log₂(params), 95% CI below 0 | **supported** (−1.28 pp/doubling, 9 pairs) |
| H-M1/2 | attention sharing costs more than FFN; taxes add | per-rung means, 3 seeds at L8 | H-M1 **not supported** (inverted); H-M2 supported |
| H-C | a shared arm beats the unshared model at the same storage | lower mean **and** sign holds in every paired seed | **supported for looping** at L8 and L16; not for W+Wᵀ |
| H-L | which kind spends compute better (matched storage + compute) | sign in all 3 seeds and \|mean\| > 2 SE | **looping beats W+Wᵀ** at L8 and L16 |
| H-R | the same, on reasoning metrics | H-C/H-L rules on probe or Track 2 accuracy | Track 1: no W+Wᵀ signal; looping wins at L16. **Track 2: harness and validity check done (PR #13), grid not run** |
| H-T | a transposed loop beats the plain loop | spec 2026-09-15 §4 | **failed** (all three variants lose) |

Rules are fixed before the runs they judge. When a rule turned out to be badly designed, it was
replaced *in writing, with the reason*, never quietly (see the Experiment 3 note in RESULTS.md).

## 4. Scope

**In:** decoder-only LMs trained from scratch at matched budgets; WikiText-103 for the pilot,
FineWeb-Edu for the 124M rung; the controlled ViT comparison; perplexity, capability probes (LAMBADA,
BLiMP, per-position and per-frequency loss, copy gain), synthetic reasoning (Track 2), and at 124M
HellaSwag, ARC-Easy and PIQA with 95% CIs.

**Out:** products, UI, deployment and serving; quantization, pruning and distillation experiments
(future work only); conversational or post-trained evaluation (it would read at chance at this scale
and prove nothing).

## 5. What counts as success

- A complete, controlled answer to §2 at two scales (~25M pilot and 124M), with every claim scoped
  to what was measured and every withdrawn claim marked as withdrawn.
- **A clean negative is a valid result.** Most of the headline results so far are negatives
  (W+Wᵀ loses at fixed storage; the transposed loop loses; vision does not rescue it). The thesis
  stands on them as long as the controls are sound.
- **Novelty:** looping is known (Universal Transformers, ALBERT, Saunshi et al. 2025, Bae et al.
  2025), so it cannot be the headline contribution. The contribution has to come from the W/Wᵀ side:
  the controlled measurement of what W+Wᵀ costs, why, and that the domain does not explain HaLViT's
  gain.
- Course bar: the advisor must approve before the final exam and presentation; faculty templates
  are mandatory; code and data are submitted electronically, not pasted into the report.

## 6. Timeline

| when | what |
|---|---|
| 2026-10-05 | **Gate** (planned; see HANDOVER D1). Picks which variant of each kind runs at 124M (spec §6). T-006's build + sizing pilot is done (PR #13, awaiting merge). |
| 2026-10-06 → 11-30 | Phase 2: 124M grid on FineWeb-Edu (~300 GPU-h), full Track 2 grid. 350M only if 124M shows a signal. |
| 2026-11-30 | Experiments freeze. |
| December | Final report in the faculty template; full draft to Töreyin by ~2026-12-20. |
| mid-January | Final report and presentation (working assumption; confirm on Ninova when Design II deliverables are posted). |

## 7. Risks

- **Compute.** Hazar's 4070 box, which ran everything to date, is no longer the project's training
  machine. Alper's laptop (8 GB 3070 Ti) covers Track 2 and small runs. Phase 2 needs
  ~300 GPU-h on a rented GPU or a university cluster (SP4CING/UHEM access was to be requested from
  Töreyin). Seed 1337 first for all three arms keeps a partial grid a complete comparison.
- **Track 2 does not reproduce Saunshi et al.** Handled by its validity check: fix the pipeline
  before reading any W+Wᵀ result. Costs calendar, not conclusions.
- **The advisor has not been updated since 2026-08-14** and has not approved the reframe. 124M spend
  waits for his OK.
- **Scope of claims.** Two to three seeds per rung, one or two corpora, 12–100M parameters. Several
  results are one seed. Every write-up must say so.
