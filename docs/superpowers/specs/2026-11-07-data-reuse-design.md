# Data reuse — does W+Wᵀ's cost shrink as the model sees its data more often?

> **Status: pre-registration, approved by Alper 2026-11-07.** Written and pushed before any run
> below. Tests HANDOVER.md D5. Nothing here changes an earlier pre-registered question, rule or result.

---

## 1. Why

Our language models pay a clear price for W+Wᵀ: at the standard budget L4-A2's test perplexity is
15.0–15.3% worse than L4-A0's in each of three seeds (local runs, `results/runs/L4-*.json`).
HaLViT pays little: −2.6 points top-1, −1.1 at 600 epochs. Our own runs line up by how many times
a model sees its training data (HANDOVER D5 table): ~5 passes for our LMs (large cost), ~10 for our
ViTs on ImageNet-1k (large cost), ~101 for the failed ImageNet-100 pass (no measurable cost), ~300
for HaLViT (small cost).

**Hypothesis (D5).** Halving the weights costs little when the unshared model cannot use its extra
capacity (many passes over limited data), and a lot when capacity is the bottleneck (few passes,
still underfit). If it holds, it reconciles HaLViT's result with ours, and it is a finding specific
to W+Wᵀ, which the thesis needs (PROJECT.md §5).

D5's cheapest test is a ViT pair on ImageNet-1k, but ImageNet does not fit in the UHeM project's
25 GB quota. This is the language-model version D5 names: the same pair at 2× and 4× tokens.

## 2. Runs

The L4 pair (`s30h`, d=512, ctx 512, WikiText-103), unshared `A0` vs W+Wᵀ `A2`, at three step
budgets, three seeds (1337, 1338, 1339): 18 runs.

| run-ID segment | max_steps | passes over WikiText-103 |
|---|---|---|
| `-uhem` | 50,000 (the standard budget) | 5.15 |
| `-tok2x` | 100,000 | 10.3 |
| `-tok4x` | 200,000 | 20.6 |

- **Recipe:** the `TRAIN` block of `scripts/gen_ladder_configs.py` verbatim. `max_steps` is the
  only protocol change. The cosine schedule stretches with it, and warmup stays at 200 steps.
  `tests/test_ladder_configs.py` checks that each config equals `configs/runs/L4-<arm>-s<seed>.yaml`
  except for `max_steps` and `out_dir`.
- **All 18 runs train on UHeM** (A100-SXM4-80GB, `a100x4q`). The 1× point is re-run there as
  `-uhem` instead of pairing cluster runs with the existing local `L4-*` runs (RTX 4070 SUPER), so
  every comparison below is within one kind of hardware. Its config is byte-identical to the local
  run's except `out_dir`.
- **Layout:** one 4-GPU node per seed, each GPU draining one queue (`queue-reuse-s<seed>-g<gpu>.txt`):
  g0 `A0-tok4x`, g1 `A2-tok4x`, g2 `A0-uhem` then `A0-tok2x`, g3 `A2-uhem` then `A2-tok2x`.
  Every finished node is one complete seed. Measured throughput is 14.7–15.1 steps/s per GPU with
  four runs per node (benchmark job 556613), so a node takes ~3.8 h.
- **Order:** seed 1337 first, then 1338 and 1339 in parallel. Seeds 1338 and 1339 run whatever
  seed 1337 shows. The only reason to stop is a technical failure (crash, NaN, a run not
  finishing); a stop is reported, not quietly fixed.

## 3. Reading rule

Everything is read on **final test perplexity** (`test_ppl`, `test.bin`), the project's standing
metric. For budget b ∈ {1, 2, 4} and seed s, the cost of W+Wᵀ is

    g(b, s) = 100 · (test_ppl[A2, b, s] − test_ppl[A0, b, s]) / test_ppl[A0, b, s]   (percent)

and the primary contrast is Δ(s) = g(4, s) − g(1, s).

- **Supported:** g(1, s) > g(2, s) > g(4, s) in every seed, **and** mean Δ < 0 with |mean Δ| > 2 SE
  (SE of the three paired Δ).
- **Refuted:** mean Δ ≥ 0, i.e. the cost at 4× is not smaller than at 1×.
- **Inconclusive:** anything else, e.g. the cost shrinks on average but not in every seed, or not by
  more than 2 SE.

Reported alongside, descriptive only (not part of the rule):
- test_ppl per run and the mean and per-seed g(b, s) at each budget;
- whether A0 itself stops improving or gets worse from 2× to 4×, the "can't use its capacity" side
  of the hypothesis;
- the step of lowest validation loss per run, from `metrics.jsonl`;
- **hardware check:** each `-uhem` run vs its local twin `L4-<arm>-s<seed>`. A difference over 1%
  in test_ppl is flagged and explained before the result is written up.

## 4. What this cannot show

- On a fixed corpus, more passes means more tokens and more compute. This design cannot separate
  "seeing the same data again" from "training longer". A follow-up that does: the pair at 2× on
  FineWeb-Edu with fresh tokens (not run here).
- One rung (L4), one corpus, one domain. A positive result says the cost depends on reuse at this
  scale on WikiText. It does not show that this explains HaLViT.

## 5. Cost

Three node runs of ~3.8 h on `a100x4q` at 256 CPU-h per node-hour: **~2,900 CPU-h**, about 12% of
the 25,000 CPU-h allocation. Seed 1337 alone is ~970 CPU-h. Outputs acknowledge UHeM per the
allocation terms.
