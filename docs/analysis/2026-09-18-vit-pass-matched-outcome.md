# Pass-matched ViT re-run — outcome

> **CORRECTION 2026-09-27 — this run is ~2× the LMs' data reuse, not matched.** The LM figure used
> throughout (24,576 tokens/step, 10.31 passes) took the context as 1024; the ladder runs at 512, so
> the LMs see **5.15 passes**. Read "matched data reuse" below as "~1.94× the LMs' reuse", the first
> pass as ~19.6× (not 10×), and the §2 probe (10.1 passes) as ~2× as well. The direction of the finding
> survives: the depth effect flips sign between 101 and ~10 passes, and no arm overfits at ~10. Every
> claim that rests on the reuse being *equal* is weakened to "close, within 2×". Whether to re-run at
> ~25,800 steps is open. See `docs/DECISIONS.md` (2026-09-27).

Spec: `docs/superpowers/specs/2026-09-18-vit-pass-matched-design.md` (predictions 1–6 recorded before
any run of this pass, including the §3 probe). Plan:
`docs/superpowers/plans/2026-09-18-vit-pass-matched.md`. Runs: probe 2026-09-17 late, arms
2026-09-18. Artifacts: `results/reports/vision.md`, `results/runs/V*-in1k.json`,
`results/analysis/ffn-symmetry.json`.

> **Status: complete.** All five arms landed 2026-09-18, 0 failed, one seed. Symmetry re-measured.
> Predictions 1–6 scored in §6. The §6 follow-up-seed trigger fired and its six runs were not launched
> (§4.3) — that is Hazar's call.

## One-line summary

**The first pass's depth-gate failure was caused by the corpus being 10× too small for the step budget,
not by the augmentation floor. With that fixed the gate passes — and the answer to the study's actual
question is that the language negatives generalize: at matched data reuse, vision orders the sharing
schemes exactly as language does, W+Wᵀ loses badly in both, and the symmetry gap that was the first
pass's one surviving result does not reproduce.**

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

All five arms, one seed (1337), 9.991 passes each, 1000-way, final-step top-1 on the full
50,000-image validation split.

| arm | stored non-emb | layers | sharing | top-1 | best | @step | val loss |
|---|---|---|---|---|---|---|---|
| `V4-A0-in1k` | 12.59M | 4 | none | 39.27% | 39.61% | 49,000 | 2.9872 |
| `V8-A0-in1k` | 25.17M | 8 | none | 40.50% | 40.44% | 49,000 | 2.9765 |
| `V8-A2-in1k` | 12.59M | 8 | intra-layer W+Wᵀ | **31.33%** | 31.11% | 49,999 | 3.4358 |
| `V8-A1u4-in1k` | 12.59M | 8 | 4 blocks cycled ×2 | **41.38%** | 41.60% | 49,999 | 2.8929 |
| `V8-A1u4t-in1k` | 12.59M | 8 | 4 blocks, pass 2 transposed | 37.87% | 37.61% | 49,000 | 3.0865 |

No arm trips the §7 overfitting flag: the largest best/final gap is 0.34 points against a 1.0-point
threshold, and every arm's best sits at 49,000 or 49,999. This is the LM ladder's shape and the
opposite of the first pass, where all five arms peaked at 37–40k and decayed.

### 4.1 The pairs

| pair | kind | Δ | LM counterpart |
|---|---|---|---|
| `V8-A1u4` vs `V4-A0` | iso-storage | **+2.11 pts** | `L8-A1u4` 28.1615 vs `L4-A0` 29.1429 ppl (−3.37%, also a win) |
| `V8-A2` vs `V4-A0` | iso-storage | **−7.94 pts** | `L8-A2` 29.677 vs `L4-A0` 29.1429 ppl (+1.83%, also a loss) |
| `V8-A1u4` vs `V8-A2` | matched storage **and** compute | **+10.05 pts** | `L8-A1u4` vs `L8-A2` (−5.11%, also a win) |
| `V8-A1u4t` vs `V8-A1u4` | matched storage and compute | **−3.51 pts** | `L8-A1u4t` 29.1158 vs `L8-A1u4` 28.1615 ppl (+3.39%, also a loss) |
| `V8-A1u4` vs `V8-A0` | vs ceiling, 2× storage | **+0.87 pts** | `L8-A1u4` 28.1615 vs `L8-A0` 26.061 ppl (+8.06%, a **loss**) |

**Every sign in this table matches the LM ladder except the last one.** At fixed storage, looping pays
and W+Wᵀ loses, in both domains; the transposed loop loses to the plain loop, in both domains. The
ordering language produced reproduces in vision under a common recipe.

### 4.2 The one thing that does not reproduce

`V8-A1u4-in1k` at 12.59M stored weights **beats `V8-A0-in1k` at 25.17M** by 0.87 points — it is the
best arm in the study. In language the same comparison goes the other way and not narrowly: `L8-A1u4`
is 8.06% worse in perplexity than `L8-A0`. Looping buying more than the storage it saves is a vision
result that has no language counterpart here.

One seed. 0.87 points is small in absolute terms, and the §6 rule does not license a verdict from it.
It is reported as the single largest candidate domain difference the study found, and as the obvious
target for the follow-up seeds.

### 4.3 The follow-up trigger fired

§6 pre-registers seeds 1338 and 1339 for `V8-A2`, `V4-A0` and `V8-A1u4` if the `V8-A2` vs `V4-A0` gap
reaches 1.0 point. It is 7.94 points, so the trigger fires. Those six runs are ~13 h of GPU on this
machine and were **not** launched: plan Task 8 step 2 requires reporting the trigger rather than
spending unapproved GPU time unattended.

Worth noting when deciding: the trigger was written to discriminate a marginal effect. A 7.94-point
gap is not marginal, so the seeds would confirm an unambiguous result rather than resolve a doubtful
one. The comparison that would actually benefit from more seeds is §4.2's +0.87.

## 5. Symmetry, re-measured — the first pass's surviving result does not survive

FFN Jacobian rotation share `‖½(J − Jᵀ)‖²_F / ‖J‖²_F`, `u` basis, means over layers. W+Wᵀ arms
excluded from the ranges (they are structurally 0).

| model set | rotation share |
|---|---|
| DeiT-small (uncontrolled reference) | 0.3948 |
| **our ViTs, ImageNet-100** (first pass, 101 passes, 100-way) | **0.4161 – 0.4222** |
| **our ViTs, ImageNet-1k** (this pass, 9.99 passes, 1000-way) | **0.4437 – 0.4516** |
| our LMs, 18 runs | 0.4476 – 0.4774 |
| random | 0.4960 |

Per arm, against the LM floor of 0.4476 (`L16-A1u8-s1337`):

| arm | share | position |
|---|---|---|
| `V8-A1u4-in1k` | 0.4516 | **inside the LM range** |
| `V8-A0-in1k` | 0.4485 | **inside the LM range** |
| `V4-A0-in1k` | 0.4467 | below the floor by 0.0009 |
| `V8-A1u4t-in1k` | 0.4437 | below the floor by 0.0039 |

The first pass concluded that "vision FFNs are measurably more symmetric than language FFNs" and
offered this as its one clean, controlled result. **It does not reproduce.** With the recipe fixed —
the depth gate passing, no overfitting, 10× the data, 1000 classes — the separation from the language
band collapses to nothing: two arms sit inside it and the other two are within 0.004 of its floor.

**Sanity gate 3 passes exactly**: `V8-A2-in1k` measures 7.72e-33 in the `u` basis, zero to float
precision, as `L8-A2` does. The measurement is behaving.

**Caveat, stated rather than buried:** the in100 rows were measured on ImageNet-100 validation images
and the in1k rows on ImageNet-1k validation images, because `--vit-data` takes one path per
invocation. So training corpus and measurement inputs changed together, and this comparison cannot
separate them. What it does establish is that the first pass's numbers were not robust to fixing the
recipe, which is enough to withdraw the claim built on them.

## 6. Predictions, scored

| # | prediction | outcome |
|---|---|---|
| 1 | `V8-A0-in1k` beats `V4-A0-in1k` (depth gate) | **HOLDS.** 40.50% vs 39.27%, +1.23 points. |
| 2 | `V8-A2` vs `V4-A0` does **not** reproduce the LM's iso-storage loss | **FAILS.** It reproduces and amplifies it: −7.94 points against the LM's +1.83% perplexity. Vision is markedly *less* friendly to symmetric maps, not more. |
| 3 | our ViT's rotation share below our LMs', near DeiT's 0.395 | **FAILS.** 0.4437–0.4516 against the LM band 0.4476–0.4774 and DeiT's 0.3948 — inside or adjacent to the language band, nowhere near DeiT. |
| 4 | `V8-A1u4t` vs `V8-A1u4` closer to level than the LM's +3.39% | **FAILS.** −3.51 points, about −8.5% relative, larger than the LM's +3.39%, not closer to level. |
| 5 | the pass count, not the augmentation floor, is what blocked depth | **HOLDS IN BOTH HALVES.** Probe +1.38 at 5,000 steps on ImageNet-100; gate +1.23 on ImageNet-1k. The probe-pass / gate-fail combination the spec wrote as a partial falsification did not occur. |
| 6 | the in100 symmetry band reproduces within ±0.03 | **HOLDS LITERALLY, FAILS IN SUBSTANCE.** All four arms fall inside the numeric tolerance (0.386–0.452), but the clause that gave the tolerance meaning — "still strictly between DeiT-small and our LMs" — is false. Recorded as a failure; the tolerance was too loose to carry the claim it was attached to. |

**Applying the spec's own §6 decision rule:** predictions 2 and 3 both fail, and the rule says that
means *the LM negatives generalize* — sharing does not behave differently in vision under a common
recipe, and HaLViT's reported gain is attributable to its recipe, scale or augmentation rather than to
the domain. That is this study's answer.

With one qualification the rule did not anticipate: §4.2. The arm ordering generalizes, but looping's
*margin* does not — in vision it beats a model storing twice as much, and in language it does not.
The generalization is about which schemes win, not about how much they win by.

## 7. What this does and does not license

**Does:**
- The first pass's depth-gate failure is explained, and the explanation was tested before it was
  trusted (§2) rather than inferred from a curve.
- At matched data reuse, the vision arms order exactly as the language arms do: looping > unshared
  shallow > transposed loop > W+Wᵀ.
- HaLViT-style intra-layer W+Wᵀ does not pay in vision under our recipe. This is now an interpretable
  negative, unlike the first pass's, because the depth gate passed and sharing had headroom to trade
  against.
- The symmetry claim from 2026-09-17 is withdrawn.

**Does not:**
- Any verdict at one seed. The §6 trigger fired and its seeds were not run.
- Any claim that HaLViT is wrong on its own terms. §9 R3 of the first spec stands: a null under a
  common minimal recipe shows the gain does not survive that recipe, not that it fails under theirs.
- Any causal account of §4.2. That looping beats the unshared ceiling in vision and not in language is
  an observation at one seed, not a mechanism.

## 8. Open

1. **The follow-up seeds (§4.3)** — Hazar's call, ~13 h of GPU.
2. **§4.2 deserves its own seeds** more than the trigger's pair does. If looping really buys more than
   its storage in vision and not in language, that is the most interesting thing here.
3. **The 124M W+Wᵀ slot.** This pass adds evidence: W+Wᵀ loses badly in vision too, so the slot has
   even less reason to hold plain W+Wᵀ than it did after the transposed-loop negative.
4. **The withdrawn symmetry claim** needs to reach anything that cited it, including the briefing
   planned for Töreyin.
