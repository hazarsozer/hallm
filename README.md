# HaLLM — Does W+Wᵀ Weight Sharing Generalize to Language Models?

ITU graduation thesis (Alper Düzgün, Hazar Utku Sözer; advisor: Prof. B. U. Töreyin).
Extends [HaLViT](https://github.com/sp4cing-itu/halvit) (CVPR 2024 W) — "half of the
weights are enough" via W/Wᵀ reuse — from Vision Transformers to autoregressive LMs,
and compares/composes it with ALBERT-style cross-layer sharing under a matched budget.

Companion code: [alpericon/wplusw-lm](https://github.com/alpericon/wplusw-lm) (independent implementation).
Trained checkpoints: [hallm-thesis/hallm-wikitext103](https://huggingface.co/hallm-thesis/hallm-wikitext103) (private).

All runs share one protocol: WikiText-103, GPT-2 BPE (V=50257), d=512, ctx=512, 614M tokens
(50k steps × 12,288), AdamW lr 6e-4 cosine → 6e-5, bf16, dropout 0.0. Only the declared
variable differs within a comparison. **Forward GFLOPs are identical across arms** — sharing
compresses storage, never compute.

---

## Results

### The sharing tax and where it comes from (L8, seeds 1337–1338)

| arm | sharing | stored non-emb | mean tax | cost per % storage saved |
|-----|---------|---------------|----------|--------------------------|
| A0 | none | 25.2M | — | — |
| **A2attn** | **W+Wᵀ attention only** | −16.7% | **+4.13%** | **0.247** |
| A2ffn | W+Wᵀ FFN only | −33.3% | +9.03% | 0.271 |
| A2 | W+Wᵀ both | −50.0% | +14.44% | 0.289 |
| A1 | ALBERT cross-layer | −87.5% | +36.7%\* | 0.420 |
| A3 | both axes | −93.7% | +66.2%\* | 0.706 |

\* A1 and A3 are **single-seed (1337)**; every other row is a 2-seed mean. They have not been
replicated and their taxes carry no error estimate.

The decomposition **inverted the project's mechanistic prediction**. `wiki/roadmap/01-mechanism.md`
argued the FFN path was strong (a genuine nonlinearity sits between `W` and `Wᵀ`) and the causal
attention path fragile (K and V both linear in the same `x`, risk R1). Measured directly, attention
sharing is the *cheapest* mechanism — absolutely and per parameter saved. Taxes are additive
(residuals +1.04 and +1.51 pp, inside the ±2 pp pre-registered band).

What replaces the strong/weak story: **the cost is roughly proportional to capacity removed**,
~0.25–0.29% PPL per 1% of non-embedding storage, with a mild penalty for removing more. That single
rate reproduces every arm above, and it explains why nothing reaches the <2% viability gate —
at 0.247, a 2% tax buys only ~8% storage reduction.

### The tax shrinks with scale (H-S — supported)

| rung | non-emb (unshared) | seeds | mean tax | SE |
|------|--------------------|-------|----------|-----|
| L4 | 12.6M | 2 | 15.27% | 0.33 |
| L8 | 25.2M | 2 | 14.44% | 0.57 |
| L16 | 50.3M | 3 | 12.78% | 0.34 |

Pre-registered rule: regress tax on log₂(non-embedding params); supported iff the slope's 95% CI
excludes zero on the negative side. Result over 7 pairs: **−1.27 pp per doubling, CI [−1.96, −0.57]
→ SUPPORTED**.

Stated honestly: extrapolating that decay to a <2% tax implies ~19B non-embedding parameters. The
ladder characterises a *rate*; it does not lead to the viability gate.

### Iso-storage: sharing does not buy free capacity

| model | depth | stored non-emb | test PPL | Δ vs A0 |
|-------|-------|---------------|----------|---------|
| A0 | 8 | 25.17M | 26.06 | — |
| A2-iso | 16 | 25.18M | 27.01 | +3.6% |
| A0-deep | 16 | 50.35M | **23.98** | **−8.0%** |

At matched *stored* parameters the shallow unshared model still wins. And the comparison is worse
than perplexity alone suggests: sharing does not touch the KV cache, so re-investing saved weights
into depth **doubles** cache — net memory is +8 MB worse at ctx512×1 and +1074 MB worse at
ctx8192×8. The weight-only saving also has a ceiling: non-embedding weights are 45.7% of inference
memory at ctx512×1 but only 4.3% at ctx8192×8.

**Scope the memory claim accordingly:** W+Wᵀ reduces *stored/downloaded weight size* at small batch
and short context. It is not a general memory-reduction technique.

Full narrative, caveats and follow-ups: **[RESULTS.md](RESULTS.md)**.
Generated tables: `results/reports/` (rebuild with `uv run python scripts/build_reports.py`).

---

## Repository structure

```
src/hallm/
  model/config.py      ModelConfig · the six arms (ARMS) · named shapes (SHAPES)
  model/sharing.py     THE CRUX — W+Wᵀ intra-layer + ALBERT cross-layer mechanisms
  model/gpt.py         decoder-only GPT assembled from sharing-aware sublayers
  data/wikitext.py     GPT-2 BPE tokenization, .bin token streams, batch sampling
  train.py             matched-budget loop · exact resume · checkpoints · val probe
  eval.py              perplexity + the arm comparison table
  metrics.py           params · size · analytic FLOPs · KV-cache and memory accounting
  capeval.py           capability evals: LAMBADA, BLiMP, per-slice perplexity
  experiment.py        YAML → (ModelConfig, TrainConfig)
  manifest.py          frozen run manifests + the pair-validity diff
  results.py           per-run result files — the source of truth
  reports.py           derived statistics: tax, paired regression, H-S verdict
  runqueue.py          drain-the-queue runner (resumable, interrupt-safe)

configs/
  runs/<run-id>.yaml   one config per run, canonical and never hand-edited
  runs/queue*.txt      pending-run lists the runner drains in order
  smoke.yaml           micro shape used by the test suite

results/
  runs/<run-id>.json   SOURCE OF TRUTH — one file per run, atomic, idempotent
  manifests/<run-id>.json  frozen provenance, written once at launch
  reports/*.md         GENERATED and disposable — never hand-edit

scripts/
  run_queue.py         GPU session entry point; drains a queue, resumes mid-run
  gen_ladder_configs.py  generate ladder + ablation configs and their queues
  build_reports.py     rebuild every comparison table from results/runs/
  migrate_results.py   fold a legacy ledger into per-run result files
  hf_sync.py           add-only checkpoint upload to the HF store
  hf_migrate_legacy.py  hash-verified migration of legacy HF paths
  capability_eval.py   inference-only capability evals over checkpoints
  run_real_training.py  data prep + single-run training
  chain_*.sh           unattended GPU supervisors (relaunch on crash)

tests/                 12 files, 78 tests — param-count proofs, smoke, resume,
                       queue semantics, manifests, reports, metrics
docs/superpowers/      specs/ (approved designs) and plans/ (implementation)
wiki/                  curated research base: roadmap, analyses, concepts, sources
raw/papers/            cited PDFs
runs/                  gitignored runtime scratch (checkpoints, resume state)
```

### Run ID grammar

`L<depth>-A<arm>-s<seed>` — e.g. `L16-A2-s1339`. Everything unstated is the campaign default
(d=512, ctx=512, WikiText-103, standard budget). A run varying one of those gains an explicit
segment (`L8d768-A2-s1337`, `L8-A2-s1337-owt`) rather than overloading an existing one. The ID is
a key, not a spec — the manifest is the authority on configuration.

Arms: `A0` none · `A1` ALBERT cross-layer · `A2` W+Wᵀ both · `A3` both axes ·
`A2ffn` W+Wᵀ FFN only · `A2attn` W+Wᵀ attention only.

### Where each artifact lives

| artifact | location |
|---|---|
| code, configs, results, reports | this repo |
| model weights (`model.pt`) | HF `checkpoints/<run-id>/` |
| tokenized corpus (`*.bin`) | HF `data/` |
| per-run provenance | both — `results/manifests/` here, `checkpoints/<run-id>/manifest.json` on HF |
| training scratch, `resume.pt` | GPU box only, gitignored |

---

## Quick start

```bash
uv sync
uv run pytest                # 78 tests
```

Real training is GPU-only and never launched by an automated loop:

```bash
uv run python scripts/run_real_training.py prepare --train wiki.train.raw --val wiki.valid.raw --out data/
uv run python scripts/run_queue.py --queue configs/runs/queue.txt --data data/ --results-dir results/runs
```

Interrupt freely — the next invocation resumes from `runs/ladder/<run-id>/resume.pt` with optimizer
state, data order and RNG restored. Bound a session with `--max-runs N` or `--stop-step N`.

Rebuild the derived tables after new results land:

```bash
uv run python scripts/build_reports.py
```

Capability evals (inference-only, any machine; one-time data fetch in the script docstring):

```bash
uv run python scripts/capability_eval.py --checkpoints 'runs/ladder/*/*.pt' \
    --lambada data/lambada_test.jsonl --blimp data/blimp --data data/ --out results/
```

### Operational notes

- `systemd-inhibit` is polkit-denied over non-interactive ssh — don't wrap remote launches in it.
- Manifests record `deterministic: true` as a *request*; Flash Attention's backward is
  non-deterministic, so runs are not bit-reproducible. The `determinism` block records observed state.
- Uploading to the HF store needs a token with **write scope on the `hallm-thesis` org** — a token
  predating the org transfer can read but not write.

## Program and specs

The research program — what is being asked at each phase, the pre-registered decision rules, and
what a negative result buys — is in
[`docs/superpowers/specs/2026-08-20-research-program-design.md`](docs/superpowers/specs/2026-08-20-research-program-design.md).
Artifact naming and store layout:
[`docs/superpowers/specs/2026-08-19-artifact-layout-design.md`](docs/superpowers/specs/2026-08-19-artifact-layout-design.md).
