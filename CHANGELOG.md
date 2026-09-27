# Changelog

What happened, newest first, one entry per study or milestone. Results live in RESULTS.md, reasons in
docs/DECISIONS.md. This file is the timeline that connects them.

## 2026-09-27 — Handover to Alper; documentation pass
- Alper takes over the project (HANDOVER.md).
- New docs: HANDOVER.md, CHANGELOG.md, docs/PROJECT.md (the brief), docs/DECISIONS.md,
  docs/RUNBOOK.md. README and RESULTS.md brought up to date with the pilot, the transposed loop, the
  FineWeb-Edu bridge and the ViT study. COLLABORATOR.md reflects the new roles.
- **Correction:** the ViT re-run is ~1.94× the LMs' data reuse, not matched. The LM ladder is 5.15
  passes, not 10.31; one spec used context 1024 instead of 512. Banners on the affected spec,
  plan and outcome.
- **Withdrawal banner** added to the 2026-09-17 outcome doc, whose symmetry claim was withdrawn on
  2026-09-18 without the doc itself saying so.
- `tests/test_prepare_imagenet1k.py` skips cleanly without pillow instead of failing collection.
- Branches `controlled-vit` and `vit-pass-matched` merged into `main`.

## 2026-09-18 — Track 2 built and calibrated (Alper, T-006, PR #13)
- Three synthetic tasks (p-hop induction, addition, variable-binding chains), each checked against an
  independent reference solver (two generator bugs found and fixed), and a masked-loss harness
  separate from the LM training loop.
- p-hop sizing pilot: p=4, d=128 (shallow 54.8%, deep 87.7%). Validity check passes: looped `A1u1@L4`
  85.0% vs unshared deep 87.7% (gap 2.7 pp ≤ 5). At p=3 it fails (gap 25.6 pp).
- PR open, not yet reviewed or merged.

## 2026-09-18 — ViT re-run on ImageNet-1k: depth gate passes, LM negatives generalize
- Diagnosis of the failed first pass: data reuse (101 passes over ImageNet-100), not augmentation.
  Tested first with a 5,000-step probe on ImageNet-100 (+1.38 points for depth).
- ImageNet-1k at 128px, 50k steps, 9.99 passes: depth gate **passes** (+1.23 points). No arm overfits.
- Arms (1 seed): `V4-A0` 39.27 · `V8-A0` 40.50 · `V8-A2` **31.33** · `V8-A1u4` **41.38** · `V8-A1u4t` 37.87.
  Same ordering as language; the one difference is that looping beats the unshared ceiling (+0.87).
- Predictions 2, 3, 4 fail → by the spec's rule, the LM negatives generalize to vision.
- **Withdrawn:** the 2026-09-17 symmetry claim (no vision-vs-language gap once the recipe is fixed).
- New code: 1000-class shapes, stratified eval subsample, parallel ImageNet-1k prepare, per-corpus
  vision report with pass counts and an overfitting flag.

## 2026-09-17 — Controlled ViT study, first pass: depth gate fails
- Five arms as ViTs on ImageNet-100 @112px, our LM recipe, minimal augmentation.
- `V8-A0` 46.96% < `V4-A0` 48.14%: the pre-registered depth gate **fails**, so no sharing claim.
  All four 8-layer arms within 0.4 points of each other.
- Reported the FFN symmetry gap as a controlled result (withdrawn the next day).
- `main` pushed through `cf37216`: 45 atomic commits. 15 new checkpoints synced to HF (50 runs total).

## 2026-09-16 — Transposed loop answered: it lost
- `L8-A1u4t/n/a` (second pass reuses each block with Wᵀ: added, subtracted, learned α):
  +3.39%, +4.50%, +3.22% vs the plain loop. None beats it; all beat plain W+Wᵀ.
- Predictions 2 and 3 fail → the symmetric/antisymmetric argument does not explain the ordering.
- FFN symmetry re-run over loop checkpoints: sharing does not move the rotation share.
- **FineWeb-Edu bridge pair** (`L8-A0-fw` val 45.56 / test 49.55, `L16-A2-fw` 46.45 / 50.56): the
  W+Wᵀ penalty halves on FineWeb-Edu (+1.94% vs +3.64% on WikiText). 8.8% val/test gap, unexplained.
- Controlled ViT study designed.

## 2026-09-15 — Pilot drained; transposed loop designed and built
- Pilot (8 runs + seeds, 0 failed): looping satisfies H-C at both sizes (`L8-A1u4` −3.21% vs `L4-A0`,
  `L16-A1u8` −2.47% vs `L8-A0`, 3/3 seeds) and beats W+Wᵀ at matched storage and compute (−5.25%,
  −6.10%). `L9-A2attn` +1.81%, `L10-A2attn` +0.05% then +0.71% at seed 1338: A2attn does not beat
  `L8-A0` at iso-storage.
- Track 1 probes over 21 checkpoints: no W+Wᵀ reasoning signal; looping wins at L16 on LAMBADA,
  BLiMP, late and rare loss.
- FFN rotation share measured: LMs 0.45–0.48, DeiT-small 0.40.
- Looping ruled out as the headline (prior work). Transposed loop specified, literature-checked
  (novel as specified, ~70%), built and merged.

## 2026-09-14 — Program reframed around fixed storage
- New question: does compute bought through sharing pay at fixed stored weights? Pilot → 10-05
  gate → 124M on FineWeb-Edu. Track 2 (synthetic reasoning) assigned to Alper (T-006, #12).
- `A1u<k>` looped arms (`n_unique_blocks`), A2attn L9/L10 shapes.
- Corrections: every earlier "test PPL" was validation (`val_ppl` + real `test_ppl`); "W+Wᵀ beats
  ALBERT" withdrawn. Copy-gain probe replaced floor-level induction/recall probes.
- FineWeb-Edu pipeline (2.6B train tokens, held-out shard 013).
- Alper's T-004 (#11) and T-005 (#10) merged.
- Reports per rung (no pooling); `probes.md` for suffixed run IDs.
- Training moved from the PopOS box to a new Omarchy desktop (same 4070 SUPER).

## 2026-09-01 → 09-07 — T-001, T-002, T-004, T-005 (Alper)
- T-001: L4 decomposition. The attention tax falls with depth (5.6% → 4.0%), FFN stays ~8.3–8.8%;
  at L4, A2ffn is cheapest per % saved.
- T-002: LR ±2× at L8. A0 is best at the standard LR; the smaller tax at 2× is mostly A0 getting worse.
- T-004 (unshared P4 frontier) and T-005 (A2attn at 2× LR) run 09-06/07, merged 09-14.

## 2026-08-31 — Iso-storage gate
- A2attn@L8 (3-seed 27.041) vs unshared L7 (26.406): +2.41%, misses ≤2%.

## 2026-08-19 → 08-21 — Scaling ladder and mechanism decomposition
- Depth ladder L4/L8/L16 at d=512, 3 seeds each: tax 15.35% / 14.42% / 12.78%. **H-S supported**
  (−1.28 pp per doubling, CI [−1.78, −0.79]).
- L8 decomposition (3 seeds): A2attn +3.97%, A2ffn +8.80%, A2 +14.42%. The attention path is the
  *cheapest*, inverting the roadmap's prediction; taxes are roughly additive.
- Infrastructure: exact resume, frozen manifests, a run queue, capability evals, per-run result
  files, generated reports, `tasks.py` for collaborator tasks, HF org `hallm-thesis`.
- Alper runs the seed-1339 ladder pairs on his 3070 Ti (#1).

## 2026-08-18 — First controlled results
- Four arms at d=512/L8, WikiText-103, 614M tokens: A0 26.06 · A1 35.63 · A2 29.68 · A3 43.30.
- Iso-parameter 2×2: A2-iso (L16, W+Wᵀ) 27.01 vs A0 26.06 vs A0-deep 23.98. Sharing doesn't buy free
  capacity at iso-storage.
- Repo assembled and pushed.

## 2026-06-12 — Thesis direction locked
- Advisor picks Option 2 (W+Wᵀ → language models), research only.

## 2026-04 → 06 — Design I
- Literature wiki (56 sources), the two thesis options, the interim report (`docs/interim-report.pdf`).
