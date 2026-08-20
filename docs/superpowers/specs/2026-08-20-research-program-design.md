# Research Program Design — W+Wᵀ Weight Sharing in Causal LMs (Term 2 → January)

> **Status: approved 2026-08-20 (Hazar + Claude design session).** Supersedes the *organisation* of
> `wiki/roadmap/06-scaling-campaign.md`; its protocol constants (§3) and infrastructure (§8) carry
> forward verbatim. Experiments 1–3 are inputs, not superseded.
>
> **What changed and why.** 06 was organised by *hardware cost* (guaranteed / near-free / gated).
> That tells you what is affordable but never what is being asked, which is why it read as thin on
> "what and why". This document is organised by **question**. Every phase declares: the claim, the
> mechanism-level reason to believe it, the exact runs, a **pre-registered** decision rule, the cost,
> and what a negative result buys.

---

## 1. The corrected premise

The project's founding belief has been "same compute, half the memory, therefore a win." That is
true of the **weight file** and substantially false of **serving memory**, because sharing does not
touch the KV cache (`2·d·L` per token, independent of what is shared).

Measured, bf16, d=512:

| config | weights | KV @ ctx512×1 | KV @ ctx2048×8 | KV @ ctx8192×8 |
|---|---|---|---|---|
| A0 L8 | 50.3 MB | 8.4 MB | 268 MB | 1074 MB |
| A2 L8 | 25.2 MB | 8.4 MB | 268 MB | 1074 MB |
| A0 L16 | 100.7 MB | 16.8 MB | 537 MB | 2148 MB |
| A2-iso L16 | 50.3 MB | 16.8 MB | 537 MB | 2148 MB |

Two consequences that must appear in the thesis:

1. **Experiment 2's negative is stronger than reported.** The iso-storage swap A0@L8 → A2-iso@L16
   holds weight memory flat *and doubles the KV cache*: net **+8 MB worse at ctx512×1, +1074 MB
   worse at ctx8192×8**. Re-investing saved weights into depth pays the saving back, with interest,
   in cache. Perplexity was not the only thing that lost.
2. **There is a ceiling on the whole idea.** Non-embedding weights are 45.7% of inference memory at
   ctx512×1, 13.6% at ctx2048×8, 4.3% at ctx8192×8 — so the *maximum possible* saving from a −50%
   scheme is 22.8% / 6.8% / **2.1%** respectively.

**Scope the claim accordingly:** W+Wᵀ addresses *stored, downloaded and loaded weight size* at small
batch and short context. It is not a general memory-reduction technique. Stating this pre-emptively,
with these numbers, is strictly better than having it raised in a defence.

## 2. Standing methodology

Carried verbatim from 06 §3: WikiText-103, GPT-2 BPE (V=50257), 614M-token budget
(50k × 12,288), AdamW lr 6e-4 cosine → 6e-5, warmup 200, wd 0.1, grad clip 1.0, bf16, dropout 0.0,
batch 12 × accum 2. Pair = two runs identical but for the declared variable. Run-ID grammar
`L<depth>-A<arm>-s<seed>`; arm tags in run IDs carry no hyphen (`A2ffn`, `A2attn` → `ARMS["A2-ffn"]`,
`ARMS["A2-attn"]`).

Three additions:

- **Pre-registration.** Every phase's decision rule is fixed in this document *before* its runs
  execute. A conclusion selected after seeing results is not a finding. This is not new discipline —
  `00-master.md` §7 already states its ablation conditional in advance; this generalises it.
- **The recommendation criterion.** The `<2%` viability gate is an absolute bar that no arm has
  approached and FFN-only probably will not either. The recommendation is therefore made on
  **PPL cost per % storage saved** — where A2 = 0.28 and ALBERT = 0.42 are the standing baselines —
  subject to an absolute-cost ceiling declared per phase. The `<2%` gate is retained and reported as
  a separate, honest verdict.
- **Paired statistics.** Both arms of a pair share a seed, so the tax is a *paired* measurement and
  the relevant noise is pair-to-pair variance of the tax. Report mean ± SE across seeds. The 06 §4
  "adjacent min/max ranges must not overlap" rule is **withdrawn**: a min/max range can only widen
  with n, so that rule gets harder to satisfy as evidence accumulates. (It is already violated by a
  0.06 pp sliver at L4 vs L8, on means that are cleanly monotone.)

---

## P0 — Are the artifacts trustworthy?  *(no GPU; blocks everything)*

Not a research question, but every later phase's evidence passes through it.

| # | Repair | Why it blocks |
|---|---|---|
| 1 | Per-run metrics jsonl: step, train loss, val loss, lr, tokens, wall time, peak VRAM | Checkpoints store only `model/model_cfg/train_cfg`. No train-vs-test-gap analysis is possible at *any* rung today. |
| 2 | Record `git_commit` in the manifest | Currently `"unknown"` everywhere, because `~/Dev/hallm` on the box is not a git checkout. 06 §8.2 defines a valid pair as manifests differing only in declared variables. |
| 3 | Measure memory, do not infer it: peak VRAM, weight bytes, KV bytes at reference (ctx,batch) | §1 is the thesis's central axis and is currently only ever computed from parameter counts. |
| 4 | Final train **and** val loss into `ladder.jsonl` | Needed for the generalisation-gap story throughout. |
| 5 | Resolve the determinism claim | `deterministic: true` is recorded in every manifest, but Flash Attention's backward is non-deterministic (warned at runtime). Either enable deterministic SDP or stop asserting it. |
| 6 | Fix the confounded claim in `RESULTS.md` | "A2-iso fits better but generalises worse (3.29 vs 3.81)" compares **L16 against L8** — a depth effect, not a sharing effect. It cannot be recomputed: the curves were never saved. |
| 7 | Decide the residual init | Neither arm applies GPT-2's `1/√(2L)` residual scaling (`gpt.py` docstring). Symmetric within a pair, but the ladder's independent variable **is depth**, so H-S is entangled with a depth-dependent init pathology at 200 warmup steps. |

**Finding to record from item 7, independent of any number:** A2 *structurally cannot* take
independently-scaled residual init. The output projection **is** `W_qᵀ`, so scaling the residual
write path down by `1/√(2L)` also scales `Q`. The shared architecture is incompatible with the
standard depth fix. This is a mechanism-level cost of W+Wᵀ that owes nothing to perplexity.

**Exit criterion:** smoke suite green; one existing config re-run reproduces its `ladder.jsonl` row
within stated tolerance.

---

## P1 — *Where* does the tax come from?  *(~8 h GPU)*

The cheapest and most decisive experiment available, already implemented, never run.

**Why.** `01-mechanism.md` gives asymmetric priors. The FFN path has a genuine nonlinearity between
`W` and `Wᵀ` (`y = Wᵀ GELU(W x)`), so the column-space argument transfers directly — the *strong*
path. The attention path has `K = W_kv x` and `V = W_kvᵀ x`, **both linear in the same x**, with no
nonlinearity between them; only the causal softmax mixes, and causal masking *reduces* that mixing —
the *weak* path, logged as risk R1.

A second, sharper reason: in the QK/OV circuit decomposition, unshared attention generates the two
circuits from four independent matrices, so *what to attend to* is chosen independently of *what to
copy*. Under sharing both circuits are generated by `W_q` and `W_kv` alone, so they are **coupled**.
Induction — attend by match, copy the *next* token — needs exactly that independence.

**Claims (pre-registered).**
- **H-M1:** `tax(attn-only) > tax(ffn-only)`.
- **H-M2:** the taxes are approximately additive: `tax(A2) ≈ tax(ffn-only) + tax(attn-only)`.

**Instrument.** `A2-ffn` and `A2-attn` at L8, seeds 1337 and 1338 → **4 runs × 2.03 h ≈ 8.1 h**.
Storage saved: FFN-only −33.3%, attn-only −16.7%, both −50.0% (per layer: attn 4d², FFN 8d²).

**Decision rule.**
- H-M1 supported iff `tax(attn-only) − tax(ffn-only) > 2 pp` with consistent sign across both seeds.
- H-M2 supported iff `|tax(A2) − (tax_ffn + tax_attn)| < 2 pp`.
- The **recommended configuration** is the arm minimising cost-per-%-storage-saved, subject to
  absolute tax < 8% (chosen as roughly half A2's measured 13.9%, and fixed here in advance of the
  runs). If none qualifies, the recommendation is "no configuration recommended" and the
  phase's output is the boundary finding instead.

**What a negative buys.** If FFN-only is also expensive, the mechanism is broadly incompatible with
causal LMs — R1 confirmed, and a clean publishable boundary rather than a null. `00-master.md` §7
already commits to this reading.

**Optional (+2 h, 1 pair):** alternative transpose pairing. HaLViT pairs Q↔Out and K↔V; pairing
Q↔K and V↔Out is a different constraint at identical storage. Distinguishes "sharing is costly" from
"*this pairing* is costly."

**Depends on:** P0.

---

## P2 — *What* does the constraint forbid?  *(~2 h GPU)*

The rigorous form of "perplexity isn't the whole story." Averaged PPL can hide a lopsided deficit.

**Why.** The mechanism predicts *where* to look. FFN sharing forces neuron *i* to read and write
along the same direction `row_i(W)` — in the key-value-memory view of FFNs, **key = value**. The
layer can amplify or suppress a direction it detects; it cannot map A to a different B. Attention
sharing couples QK and OV as above. Together they predict damage concentrated on
associative/induction-like computation and relative sparing of local syntax.

**Claims (pre-registered).**
- **H-C1:** the A2−A0 loss gap **grows with token position** (long-range use degrades faster).
- **H-C2:** BLiMP (syntax) degrades proportionally **less** than LAMBADA (long-range) under sharing.
- **H-C3:** the gap is larger on **rare** tokens than common ones.

**Instrument, cheapest first.**
- **P2a — free, zero new data:** per-position loss curves and loss-by-frequency-decile over all
  existing checkpoints. Minutes of GPU.
- **P2b:** LAMBADA + BLiMP via `scripts/capability_eval.py` (already written); one-time data fetch
  (one curl, one shallow clone).
- **P2c:** synthetic in-context probes — induction and associative recall — evaluated in-context on
  existing checkpoints first; dedicated tiny models only if the in-context signal is null.

**Decision rule.** Each metric reported separately (`05-eval-protocol.md` no-composite rule). H-C1
supported iff the gap is monotone across position buckets in ≥3 of 4 rungs.

**What a negative buys.** "The tax is uniform" is itself a clean result: it says W+Wᵀ degrades the
model *evenly* rather than removing a specific capability — graceful degradation in the strong sense.

**Depends on:** P0. Independent of P1; run concurrently.

---

## P3 — How does the tax move with scale?  *(~20 h GPU)*

The existing ladder, re-scoped from headline to supporting characterisation.

**Status.** L4 and L8 complete at 2 seeds; L16 completes when `L16-A2-s1339` lands (running
2026-08-20). Current tax: L4 15.27, L8 14.45, L16 13.03 (means) — monotone, as H-S predicts.

**The honest framing, stated up front.** Fitting `tax% ≈ 17.61 − 1.12·log₂(L)` gives ≈ **−1.1 pp per
depth doubling**, hence L≈111 for <10%, L≈2,451 for <5%, **L≈15,689 for the <2% gate**. Extrapolating
ten doublings from three points across two is not sound, but the gap is large enough that no
plausible functional form rescues it. **H-S, even if it holds perfectly, does not lead to viability.**
Its value is a measured decay rate. The campaign says so rather than implying otherwise.

**The aspect-ratio confound.** Holding d=512 and scaling depth walks d/L from 128 (L4) → 64 (L8) →
32 (L16) → 16 (L32). L32 at d=512 is pathologically deep-narrow, so an L16→L32 bend may be about
aspect ratio rather than sharing. **Fix: add a width axis.**

**Instrument.**
- Finish `L16-A2-s1339` (2.85 h) — *in flight*.
- **Width axis:** L8 pairs at d=384 and d=640, seed 1337 → 4 runs ≈ 8 h. Separates "size" from
  "depth."
- **L32 rung:** VRAM pre-flight first, and **measured, not assumed** — `L16-A2` was observed at
  7,512 MiB of 12,282 during the 2026-08-20 session, so headroom at L32 (roughly double the
  activations and weights, and A0 stores twice what A2 does) is thin but not obviously negative.
  Pre-flight decides. If it does not fit, the fallback is `grad_checkpoint: true`, which is
  **numerically identical** (recompute, not a changed summation order) — a cleaner deviation than the
  micro-batch÷2 sanctioned for Alper, and it must be set on *both* arms and recorded in the manifest.
  ~9 h/pair with checkpointing, ~7 h without.

**Decision rule (replaces 06 §4).** Regress tax on `log₂(non-embedding params)` across all rungs and
seeds. H-S **supported** iff the slope is negative with a 95% CI excluding zero; **refuted** iff the
CI excludes zero on the positive side; **inconclusive** otherwise. Report the slope with CI as the
deliverable — a decay *rate*, not a verdict.

**Depends on:** P0; L32 sequenced after P1 so the ladder can be extended with the recommended
mechanism rather than only A2.

---

## P4 — Can the freed memory be re-invested to win?  *(~25 h GPU)*

The only phase that directly tests the project's founding belief, and the only place a win can appear.

**Why it is currently unanswered.** Every arm in 06 holds compute fixed and lets storage drop, so the
freed memory is a free variable that nothing consumes — such a design can only measure cost.
Experiment 2 is the sole re-investment attempt (savings → depth) and it lost. But it tried *one* axis
at *one* budget, and scored on weights alone (see §1).

**Claim (pre-registered).** **H-R:** at a fixed **total inference memory** budget, some shared
configuration achieves lower test PPL than the best unshared configuration.

**Instrument.** Fix S = 25.2M non-embedding params (A0@L8's storage). Populate the frontier:
A0(d512,L8) reference · A2(d512,L16) [measured, 27.01] · A2(d724,L8) · A0(d362,L16) · A0(d724,L4) ·
plus the P1-recommended mechanism at its own iso-S shapes. ~8–12 runs.

**Decision rule.** H-R supported iff a shared configuration wins at **both** ctx512×1 and ctx2048×8
reference points, scored on *total* inference memory (weights + embeddings + KV), not weights alone.
Winning at ctx512×1 only is reported as a **short-context-only** result and is not a general win.

**What a negative buys.** A quantitative Pareto frontier showing unshared models dominate at iso-
memory — the strongest possible form of the project's central negative, and far more defensible than
the single 2×2 point that carries it today.

**Depends on: P1.** This ordering matters more than any other in this document. If FFN-only is the
good mechanism, the frontier must be searched with FFN-only; searching it with A2 first would spend
~25 GPU-hours mapping a frontier for a mechanism about to be abandoned.

---

## P5 — Is the tax an artifact of the recipe?  *(~22 h GPU; attaches, does not stand alone)*

A recipe probe is only worth running against a conclusion it could overturn, so each is bound to one.

| probe | threatens | runs | cost |
|---|---|---|---|
| Token budget 2× (L8 pair, resume 50k→100k from existing `resume.pt`) | P1/P3 — shared arms are over-trained per stored param | 2 | 4.1 h |
| Dropout 0.1 (L16 pair) | P2/P4 — the generalisation-gap story | 2 | 5.7 h |
| `sharing_warmup_steps` > 0 (L8 pair) | P1 — implemented, documented as R3 mitigation, never once non-zero; an untested lever that could *reduce* the tax | 2 | 4.1 h |
| LR ±2× (L8) | everything — is the tax an artifact of an LR tuned on the baseline? | 4 | 8.1 h |

**Decision rule.** A probe overturns its target conclusion only if it moves the tax by **>2 pp with
consistent sign across both arms**. Anything smaller is reported as a robustness confirmation.

---

## P6 — Does it generalise beyond WikiText-103?  *(~10 h GPU)*

- **Dataset:** one L8 pair on a second corpus (OpenWebText or FineWeb-Edu slice), same token budget.
  Also relieves a standing problem: the 614M-token budget is already **5.15 epochs** of WikiText-103.
- **Vocabulary:** one L8 pair at larger BPE vocab — shifts the embedding/non-embedding split and so
  changes what a "−50% non-embedding" claim is worth.
- **Independent implementation:** cross-check against `alpericon/wplusw-lm`. Its init confound
  (`analyses/wplusw-lm-review-2026-08.md`) must be reconciled first.

---

## P7 — Capability scale: what is and is not reachable  *(scoping, not a commitment)*

Recorded because the question will be asked, and the answer should be a measurement rather than a
shrug. **Conversational or reasoning-level evaluation is not reachable from this hardware and corpus.**
Three independent walls, each sufficient alone:

| target | params | Chinchilla tokens | measured throughput | GPU time / run | WT103 epochs |
|---|---|---|---|---|---|
| current | 51M | 1.0B | 84,298 tok/s | 0.1 d | 9× |
| GPT-2 small | 124M | 2.5B | 34,671 tok/s | 0.8 d | 21× |
| GPT-2 medium | 355M | 7.1B | 12,110 tok/s | 6.8 d | 60× |
| small LLM | 1B | 20.0B | 4,299 tok/s | **53.8 d** | 168× |

1. **Compute.** A 1B pair is ~108 GPU-days. AdamW fp32 moments alone need 12.0 GB on a 12.3 GB card —
   it does not fit before activations.
2. **Data.** WikiText-103 is 119.2M tokens. Capability scale needs 20–170 epochs of it; a different
   corpus is mandatory before scale is even meaningful (→ P6).
3. **Objective.** Conversational ability is a *post-training* artifact — instruction tuning and
   preference tuning on dialogue data, none of which exists here. GPT-2 small cannot hold a
   conversation either, and it is 2.4× this size trained on 20× the data.

Running MMLU-style benchmarks on a 50M WikiText model yields chance ± noise for **every arm** — a
null produced by the floor, not by the mechanism. That is worse than no result, because it looks like
evidence.

**What legitimately substitutes.** Two things, both already in this program:
- **P2c** measures in-context-learning primitives (induction, associative recall) — the computational
  substrate that reasoning is built from, and which *does* emerge at this scale. Given the coupled-
  QK/OV prediction, this is the sharpest available test of the mechanism, not a consolation prize.
- **P3** is the bridge to scale. A measured decay rate with a CI is the only honest way to say
  anything about 1B-scale behaviour from a 4070 — which is precisely why architectural comparisons
  are conventionally run small.

**If a genuine capability-level claim is wanted**, exactly one route fits the budget: apply W+Wᵀ to an
*already-pretrained* open model and recover with finetuning, then run real benchmarks. Flagged, not
planned — it is a different question (post-hoc compression of a trained model, adjacent to the
dropped Option 1) and needs advisor sign-off before any GPU time.

---

## Schedule

2026-08-20 → January ≈ 20 weeks. At 2 overnight sessions/week × ~8 h ≈ **320 GPU-hours available**
against a program needing **~87 h** — roughly 3.7× headroom, which is what makes "run all of them"
the right call rather than an aspiration.

| phase | GPU | order |
|---|---|---|
| P0 | 0 h | immediate, blocks all |
| P1 | 8 h | first GPU work after P0 |
| P2 | 2 h | concurrent with P1 |
| P3 | 20 h | L16 closing now; width + L32 after P1 |
| P4 | 25 h | **after P1** |
| P5 | 22 h | attached to P1/P3/P4 as each concludes |
| P6 | 10 h | after P1 |

Slack absorbs re-runs from P0 item 7 (residual init) and any P5 probe that overturns a conclusion.

## Open risks

- **R1 (carried):** W+Wᵀ may not transfer under causal attention. P1 localises it; unchanged.
- **R6 (new):** P0 item 7 may force re-anchoring one or more ladder rungs. Budgeted in slack.
- **R7 (new):** the L32 rung may not fit even with gradient checkpointing. Then P3's fourth point is
  reached by width instead, and the aspect-ratio confound is reported rather than resolved.
- **R8 (new):** Alper's seed-1339 L4/L8 pairs are external (issue #1, no reply since 2026-08-20
  09:52 UTC). P3's decision rule uses a regression over available runs and does **not** require them;
  they tighten the CI rather than gate the conclusion.
