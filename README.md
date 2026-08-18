# HaLLM — Does W+Wᵀ Weight Sharing Generalize to Language Models?

ITU graduation thesis (Alper Düzgün, Hazar Utku Sözer; advisor: Prof. B. U. Töreyin).
Extends [HaLViT](https://github.com/sp4cing-itu/halvit) (CVPR 2024 W) — "half of the
weights are enough" via W/Wᵀ reuse — from Vision Transformers to autoregressive LMs,
and compares/composes it with ALBERT-style cross-layer sharing under a matched budget.

Companion code: [alpericon/wplusw-lm](https://github.com/alpericon/wplusw-lm) (independent implementation).

## Results (WikiText-103, GPT-2 BPE, d=512 L=8 ctx=512, 614M tokens/arm, seed 1337)

| arm | sharing | non-emb params | test PPL | Δ vs A0 |
|-----|---------|---------------|----------|---------|
| A0 | none | 25.2M | **26.06** | — |
| A1 | ALBERT cross-layer | 3.1M (−87.5%) | 35.63 | +36.7% |
| A2 | HaLViT W+Wᵀ | 12.6M (−50%) | **29.68** | **+13.9%** |
| A3 | both | 1.6M (−93.7%) | 43.30 | +66.2% |

Iso-parameter probe (same 25.2M stored params, 2× depth via sharing):

| model | depth | test PPL | Δ vs A0 |
|-------|-------|----------|---------|
| A2-iso (L=16, W+Wᵀ) | 16 | 27.01 | +3.6% |
| A0-deep (L=16, unshared, 50.3M) | 16 | **23.98** | **−8.0%** |

**Headline finding:** the W+Wᵀ "sharing tax" is a stable ~13–14% PPL whether measured at
matched shape or iso-compute — a predictable price for halving stored weights — but at
iso-storage the unshared model still wins: sharing is best-in-class graceful degradation,
not free capacity.

See [RESULTS.md](RESULTS.md) for the full story, caveats, and follow-ups.
Trained checkpoints: [hazarsozer/hallm-wikitext103](https://huggingface.co/hazarsozer/hallm-wikitext103) (private).

## Quick start

```bash
uv sync
uv run pytest                       # 26 tests: param-count proofs + smoke + pipeline
# real training (GPU):
uv run python scripts/run_real_training.py prepare --train wiki.train.raw --val wiki.valid.raw --out data/
uv run python scripts/run_real_training.py run --data data/ --configs configs --out runs/
```

## Layout

`src/hallm/` harness · `tests/` proofs · `configs/` arm + iso configs · `results/` run outputs ·
`wiki/` curated research base (analyses, roadmap, cited sources) · `raw/papers/` cited PDFs
