# Pass-matched ViT re-run — outcome

Spec: `docs/superpowers/specs/2026-09-18-vit-pass-matched-design.md` (predictions 1–6 recorded before
any run of this pass, including the §3 probe). Plan:
`docs/superpowers/plans/2026-09-18-vit-pass-matched.md`. Runs: probe 2026-09-17 late, arms
2026-09-18. Artifacts: `results/reports/vision.md`, `results/runs/V*-in1k.json`,
`results/analysis/ffn-symmetry.json`.

> **Status: IN PROGRESS.** The gate pair has landed and the depth gate passes. The three sharing arms
> are running (ETA ~14:40). Sections 4 onward are written as their results land; nothing below is
> filled in from expectation.

## One-line summary

**The first pass's depth-gate failure was caused by the corpus being 10× too small for the step
budget, not by the augmentation floor. At matched data reuse the gate passes, and the sharing question
is askable for the first time.**

## 1. What the first pass got wrong

`docs/analysis/2026-09-17-controlled-vit-outcome.md` closed with the augmentation floor named as the
cause of the failed depth gate. That was wrong, and the error was visible in the two recipes all along:

| | per step | × 50,000 steps | corpus | **passes** |
|---|---|---|---|---|
| LM ladder | 24,576 tokens | 1.229B tokens | WikiText-103, 119.2M tokens | **10.31** |
| ViT arms, first pass | 256 images | 12.8M samples | ImageNet-100, 126,689 images | **101.03** |
| ViT arms, this pass | 256 images | 12.8M samples | ImageNet-1k, 1,281,167 images | **9.99** |

"Our recipe" was matched on optimizer, schedule, warmup, batch semantics, step count and model shape,
but never on how many times a model sees its corpus. A 10× difference in data reuse sat between the
language arms and the vision arms of a study whose entire purpose was to hold everything but the
domain constant.

## 2. The probe — the diagnosis tested before it was trusted

§3 of the spec required a ten-minute test on data already on disk before any download was spent:
re-run `V4-A0` and `V8-A0` on **ImageNet-100** with `max_steps: 5000` and the cosine resized to it —
10.1 passes, the same reuse factor, the same corpus as the failed run.

| arm | 5,000 steps (10.1 passes) |
|---|---|
| `V4-A0-s1337-p5k` | 41.62% |
| `V8-A0-s1337-p5k` | **43.00%** |

**+1.38 points for depth**, on the corpus where the 50,000-step run had it losing by 1.18. Same data,
same code, same seed; only the budget differs. That ruled out the augmentation floor as the cause
before a single byte of the new corpus was downloaded.

## 3. The depth gate, on ImageNet-1k

| arm | stored non-emb | layers | passes | top-1 | best | best step | val loss |
|---|---|---|---|---|---|---|---|
| `V4-A0-s1337-in1k` | 12.59M | 4 | 9.991 | 39.27% | 39.61% | 49,000 | 2.9872 |
| `V8-A0-s1337-in1k` | 25.17M | 8 | 9.991 | **40.50%** | 40.44% | 49,000 | 2.9765 |

**Spec §6 sanity gate 1 is met: +1.23 points.** Set against the first pass:

| corpus | passes | V4-A0 | V8-A0 | depth buys |
|---|---|---|---|---|
| ImageNet-100 | 101.03 | 48.14% | 46.96% | **−1.18 points** |
| ImageNet-1k | 9.99 | 39.27% | 40.50% | **+1.23 points** |

Absolute accuracies are not comparable across rows (100-way against 1000-way); the sign of the depth
effect is, and it flips on the pass count alone.

### 3.1 Overfitting is gone, not merely smaller

| arm | final | best | gap | flag (threshold 1.0 pt) |
|---|---|---|---|---|
| `V4-A0-in1k` | 39.27% | 39.61% @ 49,000 | +0.34 pt | clear |
| `V8-A0-in1k` | 40.50% | 40.44% @ 49,000 | −0.06 pt | clear |

Both arms peak at step 49,000 — the last eval before the end — exactly as every LM ladder run does
(`L8-A1u4`, `L16-A1u8`, `L9-A2attn` all best at 49,000, final within 0.01–0.59%). The first pass's
arms peaked at 37–40k and decayed from there. This is a second, independent signature of the same
cause, and it is why §7's overfitting flag fires on no arm here.

### 3.2 The crossover, at matched steps

The trajectory inverts relative to the first pass. There, `V8-A0` led through step 20k and crossed
under by 30k. Here the deeper model starts slower and crosses over upward:

| step | `V4-A0-in1k` | `V8-A0-in1k` | Δ |
|---|---|---|---|
| 8,000 | 13.67% | 11.99% | −1.68 |
| 14,000 | 21.15% | 21.17% | +0.02 |
| 17,000 | 24.60% | 24.03% | −0.57 |
| 31,000 | 32.93% | 34.65% | +1.72 |
| 36,000 | 35.07% | 37.39% | +2.32 |
| final | 39.27% | 40.50% | +1.23 |

With 10× the data and 1000 classes the 8-layer model takes longer to get going, then separates. The
first pass never reached this regime because it had already begun memorising by step 30k.

## 4. The sharing arms

*Pending — `V8-A2`, `V8-A1u4`, `V8-A1u4t` are running. This section is written when they land, with
the §2 pairs, the §6 follow-up-seed trigger, and no claim beyond one seed.*

## 5. Symmetry, re-measured

*Pending — requires all five checkpoints.*

## 6. Predictions, scored

*Predictions 1–4 carry over from `2026-09-16-controlled-vit-design.md` §6; 5 and 6 are new to this
pass. Scored when the arms land. Recorded now, before those results exist:*

- **1 — `V8-A0-in1k` beats `V4-A0-in1k` (the depth gate): HOLDS.** 40.50% vs 39.27%, +1.23 points.
- **5 — the pass count, not the augmentation floor, is what blocked depth: HOLDS IN BOTH HALVES.**
  The probe passes at 5,000 steps on ImageNet-100 (+1.38), and the gate passes on ImageNet-1k (+1.23).
  The spec wrote prediction 5 so that a probe-pass / gate-fail combination would be recorded as a
  partial falsification; that combination did not occur.
- 2, 3, 4, 6 — pending the sharing arms and the symmetry re-measurement.

## 7. Corrections to the spec, from execution

Two claims in `2026-09-18-vit-pass-matched-design.md` were wrong and are corrected here rather than
quietly left standing:

1. **§4's hardware claim.** The spec asserted the memmap reads would be "comfortably inside a Samsung
   990 PRO's range". `/home` is not on the 990 PRO — it is `/dev/mapper/root` on **sda, a BIWIN M100**;
   both NVMe drives are unmounted NTFS Windows volumes. Random reads over the 62.97 GB corpus measured
   **2.9 batches/s** against the >13.0 the loop needs, failing the plan's own Task 6 gate. Cause was
   latency, not bandwidth: fancy-indexing a memmap issues serialised page faults. Fixed by
   `madvise(MADV_WILLNEED)` over the whole batch before reading it (commit `c93b807`), measured through
   the real loader at **22.6 batches/s**. The ImageNet-100 seed-1337 canary test proves no returned
   data changed.
2. **§4's timing.** The spec's ~1.07 h per V8 run does not transfer: the measured rate here is
   ~7.2 steps/s for V8 and ~10.9 for V4, so the five arms cost ~11 h on this machine rather than the
   estimated 4.8 h. The corpus does not fit page cache and every batch pays real disk latency.

A third item in the execution ledger — a claim that manifest hashing cost ~15 min per arm — was itself
wrong and was withdrawn on measurement; it is ~2 min per arm.

## 8. Provenance of the corpus

`benjamin-paine/imagenet-1k-128x128`, revision `dd3adfd6ecfb3ef3ae26205733bf5d9d5410643b`, 1,281,167
train / 50,000 validation. `resized_at_prepare` is **0 for both splits**: every image in the mirror is
already exactly 128×128, so no resampling occurred anywhere in preparation, and the open question the
design left about the mirror's layout is answered.

Verified beyond the unit tests: 76 train indices and 30 validation indices were read back from the
memmaps and compared **byte-exact** against images decoded straight from the source parquet, touching
all 13 train shards and every shard boundary, with labels confirmed against the parquet's own label
column. Labels span 0–999 with all 1000 classes present in both splits.
