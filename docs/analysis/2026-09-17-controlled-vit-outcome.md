# Controlled ViT study — outcome

Spec: `docs/superpowers/specs/2026-09-16-controlled-vit-design.md` (predictions recorded before any
vision run). Runs: five arms, seed 1337, ImageNet-100 @112px, drained 2026-09-16/17, 0 failed.
Artifacts: `results/reports/vision.md`, `results/reports/ffn-symmetry.md`,
`results/analysis/ffn-symmetry.json`.

## One-line summary

**The symmetry half of the domain story is now a controlled result and it survives. The accuracy half
is unanswered, because the setup could not make depth pay and the pre-registered depth gate failed.**

## 1. The depth gate failed, so no sharing claim is made

| arm | stored non-emb | layers | top-1 | val loss |
|---|---|---|---|---|
| `V4-A0` | 12.59M | 4 | **48.14%** | 2.5436 |
| `V8-A0` | 25.17M | 8 | 46.96% | 2.7713 |
| `V8-A2` | 12.59M | 8 | 47.32% | 2.6926 |
| `V8-A1u4` | 12.59M | 8 | 47.36% | 2.6955 |
| `V8-A1u4t` | 12.59M | 8 | 47.28% | 2.6737 |

Spec §3 gate 1 requires `V8-A0` to beat `V4-A0`. It does not: 46.96% against 48.14%. By the spec's own
rule the study is **inconclusive on sharing**, and nothing below is offered as a sharing result.

All four 8-layer arms sit inside 0.4 points of each other while the 4-layer model beats every one of
them. The spread across sharing schemes is smaller than the penalty for adding depth.

## 2. Why depth failed: overfitting, not under-training

This is the opposite of the risk §9 R1 anticipated.

| step | V4 train | V8 train | V4 top-1 | V8 top-1 | V4 val loss | V8 val loss |
|---|---|---|---|---|---|---|
| 10,000 | 1.660 | **1.405** | 43.54% | 44.60% | 2.392 | 2.500 |
| 20,000 | 1.040 | 0.995 | 45.88% | 46.12% | 2.476 | 2.557 |
| 30,000 | 0.870 | 0.834 | 47.42% | 46.74% | 2.434 | 2.656 |
| 49,000 | 0.789 | 0.779 | 48.16% | 47.38% | 2.528 | **2.763** |

`V8-A0` reaches lower training loss earlier and carries higher validation loss throughout. It leads
through step 20k, then crosses below. Both arms peak and decay (`V4-A0` best 49.42% @37k, `V8-A0` best
48.06% @40k), so the gate fails at best checkpoints too, not only at the final step.

The cause is the augmentation floor. §4 deliberately restricted augmentation to a random crop and flip
precisely so HaLViT's recipe could not enter as a confound. That choice is what an 8-layer ViT cannot
carry at 126,689 images — §9 R3's stated trade-off, biting.

**The contrast with the language ladder is the useful diagnostic:**

| | depth, L8 vs L4 unshared |
|---|---|
| language | **−10.58% perplexity** |
| vision, this recipe | **−1.18 top-1 points** (a loss) |

In language, depth bought a large gain, which is what made fixed-storage comparisons meaningful —
sharing had something to trade against. Here there is no depth headroom at all, so no sharing scheme
can distinguish itself. **The re-plan's target is the recipe, not the sharing schemes.**

## 3. The symmetry result, which does survive

FFN Jacobian rotation share, `u` basis, one measurement applied unchanged to both domains:

| model | rotation share | layer 1 |
|---|---|---|
| DeiT-small (uncontrolled reference) | 0.395 | 0.281 |
| **our ViTs** (`V4-A0`, `V8-A0`, `V8-A1u4`, `V8-A1u4t`) | **0.416 – 0.422** | 0.359 – 0.395 |
| our LMs, unshared (`L4/L8/L16-A0`, 3 seeds) | 0.449 – 0.477 | 0.488 – 0.492 |
| random | 0.496 | 0.496 |

This is the point of the whole study. Before it, the LM-vs-DeiT gap was real but **uncontrolled** —
different recipe, data, scale and task — and so could carry no causal claim. Our ViTs share the
recipe, the optimizer, the schedule, the scale, the sharing code and the parameter count with our LMs;
only the domain differs. The gap survives that control: vision FFNs are measurably more symmetric than
language FFNs, sitting strictly between DeiT and our own LMs. The early-layer effect reproduces
directionally too (our ViTs 0.359–0.395 at layer 1 against our LMs' 0.488–0.492), though less
extremely than DeiT's 0.281.

**Sanity gate 3 passes exactly:** `V8-A2`'s rotation share is 0.0000 in the `u` basis, as `L8-A2`'s is
— a W+Wᵀ FFN cannot produce an antisymmetric update, and the measurement confirms it in both domains.

## 4. Scoring the §6 predictions

| # | prediction | outcome |
|---|---|---|
| 1 | `V8-A0` beats `V4-A0` on top-1 | **FALSE.** 46.96% vs 48.14%. This is the gate. |
| 2 | `V8-A2` vs `V4-A0` does not reproduce the LM's iso-storage loss | **NOT ASSESSABLE.** Its precondition (a setup where depth pays) failed. Descriptively: −0.82 points, below the pre-registered 1.0-point threshold, so reported as no difference detected at one seed. The direction matches the LM's (+1.83% ppl, also a loss) but the magnitude is not resolvable here. |
| 3 | our ViT's rotation share below our LMs', near DeiT's | **HOLDS.** 0.416–0.422 vs 0.449–0.477, DeiT 0.395. Independent of the gate, since it measures trained weights rather than accuracy. |
| 4 | `V8-A1u4t` vs `V8-A1u4` closer to level than the LM's +3.39% | **HOLDS VACUOUSLY.** −0.08 points (4 images in 5000) against the LM's +3.39%. But every arm is level here, so this says more about the setup's resolution than about the transposed loop. |

The spec's §6 decision rule anticipated "2 and 3 both hold" (domain story supported), "2 and 3 both
fail" (LM negatives generalize) and a split. What actually happened is a fourth case it did not
enumerate: **3 holds and 2 is unanswerable**, because the gate that licenses any accuracy claim failed.

## 5. What this does and does not license

**Does:** the LM-vs-vision FFN symmetry gap is now a controlled observation rather than an
uncontrolled one. That stands on its own and is reportable.

**Does not:** any claim that sharing behaves differently in vision, in either direction. The setup
could not make depth pay, so it cannot attribute an accuracy difference to sharing. It also does not
disprove HaLViT — §9 R3 stands: a null here shows the gain does not survive a common minimal recipe,
not that it fails on HaLViT's own terms.

## 6. Open, for the re-plan

1. **Make depth pay before re-running the arms.** The candidates are more augmentation (which
   reintroduces the confound the study exists to avoid, so it would need to be applied identically and
   declared), fewer steps with early stopping at the observed ~37–40k peak, a smaller model, or more
   data. This is the decision that gates any repeat.
2. **Both arms peaked around step 37–40k and decayed.** The final-step checkpoint is not the best
   model for any arm. Whether the protocol should report best-checkpoint rather than final is a
   methodology question that also affects the LM ladder.
3. **The symmetry result may deserve its own write-up** independent of the accuracy study, since it is
   the part that came out clean and controlled.
