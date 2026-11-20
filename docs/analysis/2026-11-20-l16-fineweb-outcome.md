# L16 FineWeb-Edu budget grid — outcome

Spec (pre-registration): `docs/superpowers/specs/2026-11-19-l16-fineweb-budget-design.md`, merged as
PR #21 before any run; task issue #20. Configs, test and sbatch: PR #22. Runs: UHeM `a100x4q`
(A100-SXM4-80GB), jobs 566271 (seed 1337, a148, 2026-11-19), 566503 (1338, a144) and 566504 (1339,
a148), both 2026-11-20, all at commit bac186b. Artifacts: `results/runs/L16-*-s133?-fw?x.json`,
manifests under `results/manifests/`, checkpoints and `metrics.jsonl` on HF under `checkpoints/<run-id>/`.

> **Status: complete.** 18 of 18 runs finished, 0 failed, no NaN. Reading rule applied as written:
> **shrinks**. Hardware note: within 1%.

## One-line summary

**Training L16 for 4× longer on FineWeb-Edu, without repeating data (≤ 1 pass), brings W+Wᵀ's cost down
from 15.1% to 12.2% in test perplexity. It falls at every step in every seed (Δ = −2.97 points,
8.5 SE). Both arms are still improving at 4×. The cost shrinks because A2 gains more, not because A0
stops gaining.**

## 1. Results

Final test perplexity (`data/fineweb/test.bin`), three seeds (1337 · 1338 · 1339), means in the last
two columns.

| budget | steps | passes | A0 | A2 | mean A0 | mean A2 |
|---|---|---|---|---|---|---|
| 1× (`-fw1x`) | 50k | 0.24 | 43.88 · 43.91 · 43.99 | 50.49 · 50.41 · 50.83 | 43.93 | 50.57 |
| 2× (`-fw2x`) | 100k | 0.47 | 40.67 · 40.51 · 40.63 | 45.74 · 45.77 · 46.22 | 40.61 | 45.91 |
| 4× (`-fw4x`) | 200k | 0.95 | 38.84 · 38.64 · 38.78 | 43.39 · 43.48 · 43.53 | 38.75 | 43.47 |

"Passes" is tokens drawn ÷ 2.6B. The loader samples random windows with replacement, so 4× is about
one epoch of random windows (spec §2).

Validation perplexity (`val.bin`), same order:

| budget | A0 | A2 |
|---|---|---|
| 1× | 40.49 · 40.55 · 40.57 | 46.47 · 46.44 · 46.75 |
| 2× | 37.60 · 37.44 · 37.57 | 42.12 · 42.20 · 42.60 |
| 4× | 35.95 · 35.75 · 35.87 | 40.07 · 40.14 · 40.17 |

Cost of W+Wᵀ, g(b, s) = 100 · (A2 − A0) / A0 on test perplexity, in percent:

| seed | g(1) | g(2) | g(4) | Δ = g(4) − g(1) |
|---|---|---|---|---|
| 1337 | 15.04 | 12.47 | 11.71 | −3.34 |
| 1338 | 14.80 | 12.98 | 12.53 | −2.27 |
| 1339 | 15.54 | 13.75 | 12.25 | −3.29 |
| **mean ± SE** | **15.13 ± 0.22** | **13.06 ± 0.37** | **12.16 ± 0.24** | **−2.97 ± 0.35** |

## 2. Reading rule (spec §7)

- g(1) > g(2) > g(4) in every seed: **yes**, 3 of 3.
- Mean Δ < 0 with |mean Δ| > 2 SE: **yes**, −2.97 against 2 SE = 0.70.

→ **Shrinks.**

As at L4, most of the drop comes from the first doubling. 1× → 2× moves g by −2.06 points on average
(per seed −2.58, −1.82, −1.79), and 2× → 4× by −0.90 (−0.76, −0.45, −1.50). Both steps are negative in
every seed.

## 3. Descriptive (not part of the rule)

- **A0 still improves from 2× to 4×.** It gains 1.83, 1.87 and 1.85 ppl per seed. A2 gains 2.35, 2.29
  and 2.69 over the same step, and from 1× to 2× it gains 4.74, 4.63, 4.61 against A0's 3.21, 3.39, 3.36.
  The cost shrinks because A2 gains more at each step.
- **Lowest validation loss is at the end.** In every run it falls in the last 8% of training: steps
  49k–50k at 1×, 97k–100k at 2×, and 185k–200k at 4×. The final evaluation is at most 0.035 nats above
  that minimum. Source: `metrics.jsonl` per run, on HF with the checkpoint.
- **Val/test gap.** Test perplexity is 8.1–8.7% above validation in every run (mean 8.35%). The gap
  shrinks slightly with budget (A0 8.37 → 8.09%, A2 8.64 → 8.33%, seed means). It matches the 8.8% of
  Experiment 8, so it belongs to the FineWeb-Edu val/test split, not to one run. WikiText shows ~1%.
  Why the two splits differ is still open. It affects both arms alike and does not touch the rule.
- **Comparison with the 11-07 L4 WikiText-103 test.** That test gave 15.44 / 13.39 / 12.35%, and this
  grid gives 15.13 / 13.06 / 12.16%. The two are close at every budget, and the Δ values are similar
  (−3.09 there, −2.97 here). Different corpus, different depth (4 vs 16 layers, both d=512), and here the
  data is not repeated, so this is a comparison, not a replication.
- **Hardware note: within 1%.** `L16-A2-s1337-fw1x` has test_ppl 50.4858 and the local
  `L16-A2-s1337-fw` (RTX 4070 SUPER, code 1231ae3) has 50.5587, a gap of −0.14%.

## 4. What this licenses, and what it does not

- **Licensed:** at L16 on FineWeb-Edu, W+Wᵀ's perplexity cost falls as the step budget grows from
  0.24 to 0.95 passes. The fall is consistent across seeds and well outside seed noise. Together with
  Experiment 11, data repetition is not needed for the cost to shrink: longer training alone does it,
  at two sizes and on two corpora.
- **Not shown (spec §7):** one size, one seed set, ≤ 1 pass, and a cosine schedule per budget. Each
  budget is its own run with its own schedule, not one run checkpointed along the way.
- **Not shown:** anything about 124M. That is Hazar's run, and this grid only supplies the seed-noise
  estimate (SE ≈ 0.2–0.4 points on g) it can be read against.
- **Not shown:** where the cost bottoms out. It still falls at 4× but by less than half as much as at
  2×. Where it stops is not measured.

## 5. Cost

Three node runs of 7 h 08 min, 7 h 20 min and 7 h 16 min on `a100x4q`: 21.74 node-hours, about
**5,570 CPU-h** at 256 CPU-h per node-hour, against the estimate of ~6,100. The 10-minute benchmark
before the grid (job 566232) added about 53 CPU-h. The rest of the allocation is untouched and kept as a
reserve, in case Hazar's 124M run (on his own allocation) needs continuing.

## Acknowledgement

Computing resources used in this work were provided by the National Center for High Performance
Computing of Turkey (UHeM) under grant number 4029252026.
