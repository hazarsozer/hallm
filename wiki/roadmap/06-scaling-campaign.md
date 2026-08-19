# 06 — Term-2 Scaling Campaign: the sharing tax as a function of scale

> **Status: approved 2026-08-19 (Hazar + Claude design session).** Extends `00-master.md` with the
> post-first-results experimental program. The first four-arm run and the iso 2×2
> (`analyses/four-arm-results-2026-08.md`) are **inputs** to this campaign, not superseded by it.

## 1. Motivation and primary claim

The completed runs contain an unplanned observation: the W+Wᵀ sharing tax appears to **shrink as
scale grows**, at matched compute:

| pair (same shape, same compute, half stored params) | unshared PPL | shared PPL | tax |
|---|---|---|---|
| d=512, L=8 (~25M non-emb unshared) — A0 vs A2 | 26.06 | 29.68 | **+13.9%** |
| d=512, L=16 (~50M non-emb unshared) — A0-deep vs A2-iso | 23.98 | 27.01 | **+12.7%** |

Both measurements are single-seed, so the 1.2-point drop may be noise. The campaign's primary claim
to test:

> **H-S (scaling):** at matched compute budget, the relative PPL cost of W+Wᵀ sharing decreases
> monotonically with model size.

If H-S holds, the thesis headline upgrades from "sharing costs ~14%" to "the cost of halving
storage decays with scale" — a much stronger result for the same mechanism. If it fails (flat or
noisy trend), the stable-~14%-tax finding stands as before; either outcome is reportable.

Perplexity is not assumed to be the whole story: Tier 1.5 adds capability metrics, and Tier 3 tests
dataset/vocabulary dependence. Ranking follows the compute reality below.

## 2. Compute reality (constraints the plan is shaped by)

- **Guaranteed:** the RTX 4070 Super — but *intermittently*: the desktop dual-boots to Windows and
  is used interactively. Training happens when Hazar is remote and the box is idle. **No run may
  assume an unbroken multi-day window.**
- **Possible:** university GPUs or Colab Pro+ — not promised. Everything gated on them is Tier 3.
- Consequence: runs are ordered by evidence-per-GPU-hour, every completed run permanently extends
  the ladder, and the campaign is useful at any stopping point.

## 3. Fixed protocol (what never varies)

All Tier-1 runs inherit the Experiment-1 recipe verbatim (see `04-training-protocol.md` and
`configs/arm0_none.yaml`):

- WikiText-103, GPT-2 BPE (V=50257), 614M-token budget (50k steps × 12,288 tokens/step).
- AdamW, lr 6.0e-4 cosine → 6.0e-5, warmup 200, weight decay 0.1, grad clip 1.0, bf16,
  dropout 0.0, batch 12 × accum 2, deterministic.
- **Pair definition:** two runs identical in everything — shape, seed, all hyperparameters —
  except the A2 sharing flag. Same forward FLOPs, half stored non-embedding params.
  `tax = (PPL_shared − PPL_unshared) / PPL_unshared`.
- **Ladder rule:** width fixed at d=512; scale by depth (the rule the existing runs already
  follow). Rungs: **L=4 (~12.6M non-emb), L=8 (~25.2M), L=16 (~50.3M), L=32 (~100.7M, Tier 3)**.
- **Seeds:** 1337 (existing), 1338, 1339. Seed controls init + data order.
- **Control discipline:** between paired runs, nothing varies except {sharing flag}; across the
  campaign, nothing varies except {sharing flag, seed, depth} — every other change is a Tier-2
  probe with its own pair.
- **LR policy (stated for the thesis):** one LR for both arms of a pair, inherited from the
  baseline tuning. Whether the tax is LR-sensitive is itself a Tier-2 probe, not a per-arm re-tune.

## 4. Tier 1 — the core ladder (guaranteed hardware)

Existing runs slot in as seed 1337 (vocab-fix era, protocol-identical): L=8 pair = A0/A2,
L=16 pair = A0-deep/A2-iso. **14 new runs**, in execution order:

| order | runs | what it buys | est. GPU sessions |
|---|---|---|---|
| 1 | L=4 pair × 3 seeds (6 runs) | third rung, where H-S predicts the *largest* tax; cheapest runs (~½ the L=8 wall-time) | ~3 overnight |
| 2 | L=8 pair × seeds 1338, 1339 (4 runs) | error bars on the +13.9% point | ~4 overnight |
| 3 | L=16 pair × seeds 1338, 1339 (4 runs) | error bars on the +12.7% point; slowest, resume across sessions | ~6 sessions |

**Deliverable:** the headline figure — tax vs non-embedding size, 3 points, mean ± min/max over
3 seeds. Decision rule: H-S is *supported* if tax(L4) > tax(L8) > tax(L16) and the seed ranges at
adjacent rungs don't overlap; *inconclusive* if ranges overlap (report as such); *refuted* if the
ordering inverts.

## 5. Tier 1.5 — capability metrics beyond perplexity (near-free, runs early)

Inference-only over every checkpoint, past and future — no GPU lockdown, hours not days. At
12–100M scale most standard benchmarks read chance-level, so the set is chosen to discriminate at
tiny scale:

- **LAMBADA** (final-word accuracy): long-range context use.
- **BLiMP** (minimal-pair grammatical acceptability): syntactic ability, works even for tiny LMs.
- **Per-domain PPL slices** of the WikiText test set.

Question answered: is the ~13% tax *uniform*, or does averaging hide a lopsided deficit (e.g. the
shared model losing disproportionately on long-range prediction)? Extends `05-eval-protocol.md`;
each metric reported separately per the no-composite rule.

## 6. Tier 2 — sensitivity probes (is the tax an artifact of the recipe?)

Rule: **one attribute varied at a time; both arms rerun as a pair; seed fixed 1337.** The measurand
is always "how does the *tax* respond," never "how does one model respond." Ranked:

1. **Token budget 2×** — resume the L=8 pair from the step-50k checkpoints to 100k steps (needs
   optimizer-state resume, §8). Tests the standing caveat that shared arms are relatively
   over-trained per stored parameter. Cheapest probe per bit of information.
2. **Dropout 0.1 on the L=16 pair** — the A2-iso train/test gap (train loss 3.29 vs A0's 3.81,
   yet worse test PPL) says regularization is the probe most likely to move a conclusion.
   (Carried from the RESULTS.md follow-up list.)
3. **LR ±2× on the L=8 pair** (4 runs) — is the tax an artifact of an LR tuned for the baseline?
   Run only after 1–2; explicitly skippable under time pressure.

## 7. Tier 3 — stretch (gated on external compute)

In priority order; nothing in Tiers 1–2 depends on this tier.

- **(a) L=32 rung** (~100M non-emb), 1–3 seeds — the fourth trend point. Multi-day even on good
  hardware; on the 4070S only via many resumed sessions.
- **(b) Dataset generalization:** one L=8 pair on a second corpus (OpenWebText/FineWeb-Edu slice,
  same token budget) — is the tax a WikiText artifact?
- **(c) Vocabulary generalization:** one L=8 pair with a larger BPE vocab — tests the
  bigger-vocabulary hypothesis (embedding/head share of params shifts the non-emb picture).

Also carried: cross-validation against `alpericon/wplusw-lm` (independent-implementation check;
see `analyses/wplusw-lm-review-2026-08.md` for its init confound, which must be reconciled first).

## 8. Infrastructure prerequisites (before any new run)

Built and validated on `configs/smoke.yaml` + the pytest suite before touching real runs:

1. **Checkpoint–resume with optimizer state** (model + AdamW moments + scheduler + data-loader
   position + RNG state), checkpoint every ~1k steps. Serves the intermittent-GPU reality *and*
   makes the token-budget probe (§6, item 1) nearly free. Resumed-vs-unbroken equivalence verified on smoke.
2. **Run manifest**, frozen at launch next to the checkpoints: resolved config, git commit,
   data/tokenizer hash, torch version, GPU, seed. Two runs form a valid pair iff their manifests
   diff only in the declared variables.
3. **Run queue:** a pending-configs list the training script drains, so one ssh command starts
   "whatever is next" and interruption is always safe. Extends `run_real_training`; the
   GPU-only / never-launched-by-automation rule from `00-master.md` §5 stands.

## 9. Ledger

Live status belongs in `RESULTS.md` / `results/`; this section only fixes the campaign's run
naming: `<rung>-<arm>-s<seed>` (e.g. `L4-A2-s1338`), shapes named in `model/config.py` as the
existing s30 family extended down (L=4) and up (L=32).
