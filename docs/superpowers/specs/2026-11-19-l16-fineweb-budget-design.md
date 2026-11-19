# L16 data budget on FineWeb-Edu — does W+Wᵀ's cost shrink without repeated data?

> **Status: pre-registration, approved by Alper 2026-11-19.** Written and merged before any run below.
> Agreed by Hazar and Alper on 2026-11-19 after the Töreyin meeting (#19); task issue #20. Sections 1–3
> and 7 are copied from #20 unchanged, so the numbering follows the issue. #20 also fixes the data
> (§4, sha256 of every file checked before any run), the UHeM layout (§5) and the process (§6).
> Follows the 2026-11-07 data-reuse test (`2026-11-07-data-reuse-design.md`, RESULTS.md Experiment 11).
> Nothing here changes an earlier pre-registered question, rule or result.

---

## 1. What and why

The 11-07 test (L4, WikiText-103) showed W+Wᵀ's cost falling from 15.4% to 12.4% as the budget went
1× → 4×, but on a fixed small corpus that means 5 → 20 passes over the same text. This grid repeats the
design at **L16** on **FineWeb-Edu**, where 4× is about one pass over the training file. It asks:
does the cost still shrink when training gets longer **without** repeating data?

Tokens per parameter (A0 total params 76.34M / non-embedding 50.35M):
1× = 8 / 12, 2× = 16 / 24, 4× = 32 / 49.

## 2. Runs (18)

Arms: **A0** (unshared) and **A2** (W+Wᵀ in FFN **and** attention: `share_intra_ffn=True`,
`share_intra_attn=True`). Not `A2-attn`, not `A2-ffn`.
Shape: **`s30x2`** (vocab 50257, block 512, n_embd 512, n_layer 16, n_head 8, ffn_mult 4, dropout 0,
bias False, tied embeddings). Seeds: **1337, 1338, 1339**.

| run-ID suffix | max_steps | tokens | passes over train.bin (2.6B) |
|---|---|---|---|
| `-fw1x` | 50,000 | 614.4M | 0.24 |
| `-fw2x` | 100,000 | 1.229B | 0.47 |
| `-fw4x` | 200,000 | 2.458B | 0.95 |

"Passes" is tokens drawn ÷ 2.6B. The loader samples random 512-token windows with replacement
(`torch.randint`), so at 4× about 60% of train.bin is seen at least once and some windows twice. Do not
change the sampler; report it as "about one epoch, random windows".

Run IDs: `L16-<arm>-s<seed>-fw<b>x`, e.g. `L16-A0-s1337-fw4x`. Do **not** reuse `L16-A2-s1337-fw`
(the 2026-09-16 local run; it stays as is).

## 3. Exact config (one file per run, `configs/runs/<run-id>.yaml`)

```yaml
shape: s30x2
arm: A0            # or A2
train:
  lr: 0.0006
  min_lr: 6.0e-05
  warmup_steps: 200
  max_steps: 50000   # 100000 for fw2x, 200000 for fw4x — the ONLY value that changes with budget
  weight_decay: 0.1
  grad_clip: 1.0
  batch_size: 12
  grad_accum: 2
  dtype: bfloat16
  deterministic: true
  eval_interval: 1000
  checkpoint_interval: 1000
  dataset: fineweb-edu
  seed: 1337         # 1338 / 1339
  out_dir: runs/ladder/<run-id>
```

- Generate them by extending `scripts/gen_ladder_configs.py` (the `TRAIN` block verbatim, as the FineWeb
  bridge configs at line ~144 do). Do not hand-edit values.
- Add a test in `tests/test_ladder_configs.py`: each new config equals `configs/runs/L16-<arm>-s<seed>.yaml`
  except `max_steps`, `out_dir` and `dataset: fineweb-edu`; arm ∈ {A0, A2}; shape `s30x2`.
- Implicit defaults that must stay default (do not set them): `eval_iters` 100, `log_interval` 50,
  `grad_checkpoint` false, `block_size` 512. Cosine LR from 6e-4 to 6e-5 over `max_steps`, 200-step warmup
  (`cosine_lr` in `src/hallm/train.py`); the schedule stretches with `max_steps`, warmup does not.
- **Forbidden changes:** model code, sharing code, optimizer, LR schedule (no WSD here), batch size,
  grad accumulation, precision, `torch.compile`, attention kernel, dropout, init, tokenizer, data
  sampling, eval code, seeds. No new features in the training path for this task.

## 7. Reading rule (pre-registered; copy into the spec unchanged)

Metric: final **test perplexity** on `data/fineweb/test.bin` (`test_ppl`). For budget b ∈ {1, 2, 4},
seed s: g(b, s) = 100 · (test_ppl[A2,b,s] − test_ppl[A0,b,s]) / test_ppl[A0,b,s]. Δ(s) = g(4,s) − g(1,s).

- **Shrinks:** g(1,s) > g(2,s) > g(4,s) in every seed, **and** mean Δ < 0 with |mean Δ| > 2 SE.
- **Does not shrink:** mean Δ ≥ 0.
- **Inconclusive:** anything else.

Reported alongside, descriptive only: test_ppl and val_ppl per run; g per seed and mean ± SE per budget;
whether A0 still improves from 2× to 4×; step of lowest val loss per run; the val/test perplexity gap per
run (open question: 8.8% in Experiment 8 vs ~1% on WikiText); comparison with the 11-07 L4 WikiText
costs (15.4 / 13.4 / 12.4%), labelled as a different corpus and size; hardware note: `L16-A2-s1337-fw1x`
vs the local `L16-A2-s1337-fw` (test_ppl 50.56, RTX 4070 SUPER, older code 1231ae3), flagged if > 1% apart,
explained, not a gate.

What it cannot show: one size, one seed set, ≤ 1 pass, cosine schedule per budget. It says nothing about
124M; that is Hazar's run.
