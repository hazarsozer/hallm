# Data reuse — outcome

Spec (pre-registration): `docs/superpowers/specs/2026-11-07-data-reuse-design.md`, merged as PR #17
before any run. Runs: UHeM `a100x4q` (A100-SXM4-80GB), jobs 556620 (seed 1337), 556643 (1338), 556644
(1339), all 2026-11-07. Artifacts: `results/runs/L4-*-s133?-{uhem,tok2x,tok4x}.json`, manifests under
`results/manifests/`, checkpoints on HF under `checkpoints/<run-id>/`.

> **Status: complete.** 18 of 18 runs finished, 0 failed, no NaN. Hardware check passes. Reading
> rule applied as written: **supported**.

## One-line summary

**Seen 20 times instead of 5, WikiText-103 costs W+Wᵀ 12.4% in test perplexity instead of 15.4%. The
cost falls at every step in every seed (Δ = −3.09 points, 6.3 SE). It falls because A2 improves
faster than A0, not because A0 stops improving: A0 is still gaining at 20 passes.**

## 1. Results

Final test perplexity (`test.bin`), three seeds (1337, 1338, 1339), mean in the last column.

| budget | passes | A0 | A2 | mean A0 | mean A2 |
|---|---|---|---|---|---|
| 1× (`-uhem`, 50k steps) | 5.15 | 29.46 · 29.45 · 29.34 | 33.91 · 33.98 · 33.99 | 29.42 | 33.96 |
| 2× (`-tok2x`, 100k) | 10.3 | 28.00 · 28.08 · 28.08 | 31.76 · 32.01 · 31.66 | 28.05 | 31.81 |
| 4× (`-tok4x`, 200k) | 20.6 | 27.25 · 27.28 · 27.53 | 30.67 · 30.75 · 30.77 | 27.35 | 30.73 |

Cost of W+Wᵀ, g(b, s) = 100 · (A2 − A0) / A0, in percent:

| seed | g(1) | g(2) | g(4) | Δ = g(4) − g(1) |
|---|---|---|---|---|
| 1337 | 15.11 | 13.44 | 12.57 | −2.54 |
| 1338 | 15.38 | 14.00 | 12.71 | −2.67 |
| 1339 | 15.85 | 12.74 | 11.79 | −4.06 |
| **mean ± SE** | **15.44 ± 0.22** | **13.39 ± 0.37** | **12.35 ± 0.29** | **−3.09 ± 0.49** |

## 2. Reading rule (spec §3)

- g(1) > g(2) > g(4) in every seed: **yes**, 3 of 3.
- Mean Δ < 0 with |mean Δ| > 2 SE: **yes**, −3.09 against 2 SE = 0.97.

→ **Supported.**

Most of the drop comes from the first doubling: 1× → 2× moves g by −2.05 points on average, 2× → 4×
by −1.04. In every seed both steps are negative.

## 3. Descriptive (not part of the rule)

- **A0 does not stop improving.** It gains 1.37 ppl from 1× to 2× and 0.70 from 2× to 4×. So the
  hypothesis's "the unshared model can't use its extra capacity" side is not reached at 20 passes.
  The cost shrinks because A2 gains more at each step (2.15, then 1.08 ppl).
- **No overfitting.** In every run the lowest validation loss is in the last 6% of training (step
  188k–200k at 4×). The final evaluation is at most 0.05 nats above that minimum. Source: `metrics.jsonl` per run (on HF with the checkpoint).
- **Hardware check: passes.** Each `-uhem` run vs its local twin `L4-<arm>-s<seed>` (RTX 4070 SUPER,
  same config except `out_dir`):

  | seed | A0 | A2 |
  |---|---|---|
  | 1337 | +0.25% | +0.14% |
  | 1338 | +0.03% | +0.39% |
  | 1339 | −0.91% | −0.42% |

  All are under the 1% flag; the largest, 1339-A0, is close to it. The 1× gap on UHeM (15.44%)
  matches the local one (15.15%) within seed noise. Correction to the spec: the local 1339 twins
  ran on an RTX 3070 Ti Laptop GPU (their manifests; CHANGELOG 2026-08-19), not the 4070 SUPER.

## 4. What this licenses, and what it does not

- **Licensed:** at L4 on WikiText-103, W+Wᵀ's perplexity cost falls as the step budget, and with it
  the number of passes, grows from 5 to 20. The fall is consistent across seeds and well outside
  seed noise.
- **Not separated (spec §4):** on a fixed corpus, more passes means more tokens and more compute.
  This result is equally a statement about longer training. The follow-up that separates them, the
  pair at 2× on FineWeb-Edu with fresh tokens, has not been run.
- **Not shown:** that this explains HaLViT's small cost. One rung, one corpus, one domain. Going by
  the slope here, the gap at ~300 passes would have to keep falling at a similar rate, and that
  extrapolation is not tested.
- **Not shown:** the mechanism. A0 is not saturating, so the data say "A2 catches up with more
  training", not "A0 runs out of data to use its capacity on".

## 5. Cost

Three node runs of 3 h 46 min, 3 h 46 min and 3 h 43 min on `a100x4q`: about **2,880 CPU-h** at
256 CPU-h per node-hour, against the estimate of ~2,900.

## Acknowledgement

Computing resources used in this work were provided by the National Center for High Performance
Computing of Turkey (UHeM) under grant number 4029252026.
