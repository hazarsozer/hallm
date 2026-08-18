---
query: "Review of Alper's wplusw-lm implementation + in-scope improvement suggestions"
date: 2026-08-17
sources_consulted: ["wiki/sources/halvit-official-code", "wiki/sources/halvit", "wiki/sources/1909.11942", "wiki/analyses/halvit-vs-albert-cross-layer-sharing", "src/sharedlm (this repo)"]
---

# Review: alpericon/wplusw-lm (Alper's four-arm implementation)

Inspected 2026-08-17, single commit `b8179e5` "Initial scaffold". Independent re-implementation of the proposal's experiment: GPT (L=8, H=8, d=512, ctx=512) from scratch on WikiText-103, arms `baseline` / `albert_cross` / `halvit_intra` / `combined` + FR-8 ablations (`halvit_ffn_only`, `halvit_attn_only`), matched budget (identical seed/hparams, 20k steps), `compare.py` builds the FR-6 table with the <2% envelope flag (C2/NFR-2).

## Mechanism (matches our `sharedlm` design)
- FFN: `FFN(x) = Wᵀ·GELU(W·x+b₁)+b₂`, one stored matrix (FR-1).
- Attention: `Wkv` yields K = x·Wkvᵀ and V = x·Wkv; `Wq` yields Q and the output projection — 4 matrices → 2. (Reconstruction from the paper; no reference ViT code exists — see [[sources/halvit-official-code]].)
- Cross-layer: one shared `Sublayers` module reused across all blocks; LayerNorms per-layer (SPIN, [[entities/spin-method]]).

## Defects found
1. **Repo does not run as-is**: `train.py`/`eval.py` import `src.data.dataset` and README invokes `src.data.prepare_wikitext103`, but **`src/data/` is absent from the repo** (forgotten add or gitignored). ModuleNotFoundError on first run.
2. **No results committed**: the 2-day run's `out/` dirs, `train_log.json`, `eval_result.json`, `comparison_table.json` exist only on Alper's machine.
3. **Init confound (methodological)**: shared arms use raw `nn.Parameter` + Kaiming-uniform, while baseline `nn.Linear` weights are re-initialized to N(0, 0.02) by `_init_weights`, which skips bare Parameters. Arms therefore start from differently-scaled weights — undermines the "only the sharing flag differs" claim (C3). The official code ([[sources/halvit-official-code]]) also mixes init schemes across files, so this deserves an explicit controlled choice, not silence.

## In-scope improvement suggestions (all within FR-1…FR-9 / C2/C3 of the proposal)
1. **Unify initialization** (fixes defect 3): initialize `W`, `Wkv`, `Wq` with the same N(0, 0.02) scheme as every other projection (or extend `_init_weights` to cover bare Parameters). Cheap, restores the causal control.
2. **Commit `src/data/` + run artifacts**: reproducibility is an NFR; logs and eval JSONs are tiny — push them.
3. **Deterministic full test-set perplexity**: `eval.py` samples random windows with a seeded generator. Replace with sequential non-overlapping (or strided) windows over the full WikiText-103 test set — the standard convention, removes sampling noise from the headline PPL, and makes the <2% gate meaningful (2% can be smaller than random-window variance at eval_iters=200).
4. **Separate eval RNG from training RNG**: `estimate_loss` consumes the same `torch.Generator` used for training batches, entangling eval frequency with the training data sequence. Harmless while all arms eval identically, but fragile — one dedicated generator for eval.
5. **AdamW decay hygiene**: weight decay currently applies to LayerNorms, biases, and embeddings. Standard practice (and our `sharedlm/train.py`) splits decay/no-decay groups; matters more here because sharing changes what fraction of parameters are LN/bias.
6. **Safe checkpoint load**: `torch.load` without `weights_only=True` in `eval.py`.
7. *(Optional, still FR-8-adjacent)* **Partial-depth sharing ablation**: the official code shares only later stages — a "share upper half of layers only" arm mirrors that precedent. Flag to advisor before adding; it grows the run matrix.

## Verdict
Design is sound and convergent with our tested `sharedlm` harness (26 green tests). With fixes 1–4 his runs become citable; having **two independent codebases produce matching perplexities** is a strong robustness argument for the thesis. Action: request `src/data/` + `out/` artifacts from Alper; apply fix 1 before trusting any cross-codebase comparison.

## Related
[[sources/halvit-official-code]], [[sources/halsp]], [[analyses/halvit-vs-albert-cross-layer-sharing]], [[entities/halvit-model]], [[entities/albert-model]], [[concepts/weight-sharing]]
