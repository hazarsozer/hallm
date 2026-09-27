# Decision log

Every decision that changed what the project asks, how it measures, or what it claims, newest first.
Each entry: what was decided, why, and where the detail lives. Add an entry when you decide something
a later reader would otherwise have to reverse-engineer from the diffs.

Withdrawn claims are listed too. Withdrawing a claim is a decision, and anything that cited it has
to follow.

---

### 2026-09-27 — Alper leads the project from here
Hazar moves his focus to another project. Alper owns `main`, the run plan, the open decisions below
and the advisor thread (CC Hazar). Compute moves off Hazar's 4070 box to Alper's laptop and rented
GPUs. Detail: [`HANDOVER.md`](../HANDOVER.md).

### 2026-09-27 — CORRECTION: the ViT re-run is not pass-matched; it is ~2× the LMs' reuse
The 2026-09-18 spec computed the LM ladder's tokens per step as 12 × 2 × **1024** = 24,576. The ladder
context is 512 (`SHAPES`, the run configs and the frozen manifests agree), so it is 12,288 tokens/step,
614.4M tokens, **5.15 passes** over WikiText-103, as README and RESULTS.md always said. Consequences:
the ImageNet-1k arms (9.99 passes) see their data **~1.94×** as often as the LMs; the first ImageNet-100
pass (101.03) was ~19.6× the LMs, not 10×; the 5,000-step probe (10.1) was also ~2×. The diagnosis
survives in direction (the depth effect flips sign between 101 and 10 passes, and nothing overfits at
10), but "matched data reuse" does not. A truly matched run would be ~25,800 steps on ImageNet-1k.
Re-running or stating the 2× gap as a caveat is an open decision (HANDOVER.md).

### 2026-09-18 — The ViT recipe is fixed by matching data reuse, not by adding augmentation
The first controlled-ViT pass failed its depth gate. The cause was the corpus, not the augmentation:
the ViT arms saw ImageNet-100 ~101 times while the LMs see WikiText-103 far fewer times. Fixed by
swapping to ImageNet-1k at 128px (9.99 passes at the same 50k steps), changing nothing else, and
testing the diagnosis first with a 10-minute probe on data already on disk (+1.38 points for depth).
Rejected: adding augmentation (reintroduces the confound the study exists to remove), a shorter
budget (post-hoc), a smaller model (breaks the scale match). Pre-committed: a second gate failure
would stop the study. It passed.
Detail: `docs/superpowers/specs/2026-09-18-vit-pass-matched-design.md`,
`docs/analysis/2026-09-18-vit-pass-matched-outcome.md`.
> ⚠ Corrected 2026-09-27 (entry above): the LMs see ~5.15 passes, not 10.31, so this run is ~2× the
> LMs' reuse, not matched.

### 2026-09-18 — WITHDRAWN: "vision FFNs are more symmetric than language FFNs"
Published by the 2026-09-17 outcome as the first pass's one clean controlled result. With the recipe
fixed, our ViTs measure 0.4437–0.4516 against our LMs' 0.4476–0.4774. No gap. Anything that cites the
2026-09-17 claim (including any advisor briefing) must carry the retraction.

### 2026-09-17 — The controlled ViT study gets its recipe fixed and re-run
The first pass failed its pre-registered depth gate (`V8-A0` 46.96% < `V4-A0` 48.14%), so by the
spec's own rule no sharing claim could be read from it. Chosen: fix the recipe and re-run rather than
report the study as inconclusive. How to fix it was left to a fresh session (the entry above).

### 2026-09-16 — Answer the vision question with our own ViTs, not an off-the-shelf DeiT
Comparing our LMs with a published DeiT confounds domain with recipe, data, scale and task. Train our
own sharing arms as ViTs under our recipe so only the domain differs.
Detail: `docs/superpowers/specs/2026-09-16-controlled-vit-design.md`.

### 2026-09-16 — The transposed loop lost; the symmetry argument is dropped as an explanation
All three variants (`t`, `n`, `a`) lose to the plain loop at matched storage and compute (+3.2% to
+4.5%), so none earned follow-up seeds. Predictions 2 and 3 failed, so by the spec's rule the
first-order symmetric/antisymmetric argument does not explain the ordering. No claim of the form
"W+Wᵀ fails in LMs *because* LM FFNs need rotation" is supported.
Detail: `docs/analysis/2026-09-16-transposed-loop-outcome.md`.

### 2026-09-15 — Test W/Wᵀ across depth (the transposed loop), not looping as the headline
Looping wins at fixed storage, but looping is prior work (Universal Transformers, ALBERT, Saunshi et
al. 2025, Bae et al. 2025), so the contribution has to come from W/Wᵀ. The transposed loop reuses
each block with Wᵀ on the second pass. Literature check: novel as specified (~70% confidence). Gate
amendment: a transposed winner would have taken the W+Wᵀ slot at 124M. There was none, so **that slot
is open** (see HANDOVER.md).
Detail: `docs/superpowers/specs/2026-09-15-transposed-loop-design.md`,
`wiki/analyses/transposed-loop-literature-2026-09.md`.

### 2026-09-14 — Reframe: does shared-weight compute pay at fixed storage?
The ≤2% gate asks how much a shared model loses. The question a user faces is: at a given storage
budget, is a shared model running more compute on the same bytes better than the unshared model of
that size? Two tracks (LM perplexity and probes; synthetic reasoning), a pilot at ~25M, a gate on
2026-10-05, and a 124M rung on FineWeb-Edu. The ≤2% bar stays as a secondary verdict.
Detail: `docs/superpowers/specs/2026-09-14-fixed-storage-compute-program-design.md`.

### 2026-09-14 — Probe amendment: copy gain replaces random-token induction and recall
Both sat at the floor on four checkpoints (recall 2–8%), so they could not separate arms. Replaced,
before any pilot probe result, by copy gain (loss drop on the second copy of a repeated passage).
Associative lookup moved to Track 2.

### 2026-09-14 — WITHDRAWN: "W+Wᵀ dominates cross-layer sharing"
It compared A2 at −50% storage with A1 at −87.5% by PPL per % saved, a ratio that rises with the
amount removed, so it penalised the arm that compresses harder; A1 was also one seed. Replaced by the
matched test (looped `A1u<k>` vs `A2` at equal storage and compute), which looping wins.

### 2026-09-14 — CORRECTION: every earlier "test PPL" was validation PPL
The runner scored `val.bin` and never read `test.bin`. Nothing was tuned on validation, so every
comparison stands. Result files now carry `val_ppl`, plus a real `test_ppl` where the checkpoint
survives. `results/reports/splits.md` confirms no verdict changes.

### 2026-09-14 — Four missing checkpoints written off
`L7-A0-s1337`, `L8-A0-s1339`, `L8-A2-s1339`, `L8-A2attn-s1339` have no weights anywhere. Their result
JSONs are intact, so every number stands; only checkpoint-based probes skip them.

### 2026-08-31 — The gate compares against an unshared model of equal *stored* size
Iso-storage gate: A2attn@L8 (20.98M stored) vs unshared L7 (22.03M). Result: +2.41%, missing ≤2%.

### 2026-08-20 — Scaling program: depth ladder, three seeds per rung, pre-registered rules
H-S (tax shrinks with scale) tested on L4/L8/L16 at d=512. The original adjacent-rung min/max rule
was later replaced by a regression-slope rule, because a min/max range can only widen as seeds are
added. Detail: `docs/superpowers/specs/2026-08-20-research-program-design.md`, RESULTS.md Exp. 3.

### 2026-06-12 — Advisor locks Option 2: W+Wᵀ → language models, research only
Prof. Töreyin chose Option 2 over HaLViT-Edge and steered it away from deployment. Model scale may
go below GPT-2, which moved the core comparison onto a single RTX 4070. (Relayed by phone; wording
approximate.) Detail: `wiki/analyses/thesis-options-2026-05.md`.
