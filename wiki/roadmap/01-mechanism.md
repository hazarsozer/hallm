# 01 — The W+Wᵀ Mechanism: Formal Per-Component Analysis

> Companion to `ROADMAP.md` §2. This is the Term-2 "mathematical-validity analysis" (weeks 1–2,
> FR-7, G-pre) written up early. It states *where* the column-space argument holds, where it is only
> heuristic, and exactly what the numerical correctness check must verify. Claims are hedged per
> `codex-report.md` (use "generically/can", never "always/does").

## 1. The column-space argument

Let `W ∈ ℝ^{m×n}` be the single stored matrix and `F` a pointwise nonlinearity. A standard two-matrix
sublayer computes `y = W₂ · F(W₁ x)`. The HaLViT move sets `W₂ = W₁ᵀ`, giving `y = Wᵀ F(W x)`.

- If `F` were the identity, `Wᵀ F(Wx) = WᵀW x` — a single rank-≤`n` linear map; the second matrix
  adds no expressive direction it could not already reach. Tying would be a real capacity loss.
- With a genuine nonlinearity, `F(Wx)` **generically leaves the column space of `W`**: the activated
  vector has components outside `range(W)`, so `Wᵀ` acts on directions `W` alone never produced.
  `Wᵀ` is then a *distinct* transformation despite reusing the stored entries.

"Generically" is load-bearing: there exist `(W, x)` for which `F(Wx) ∈ range(W)` (measure-zero / special
activations), so the argument is **typical-case, not universal**. This is why the thesis is empirical.

## 2. FFN path — strong argument (core experiment)

```
standard:  FFN(x) = W₂ · GELU(W₁ x)        W₁ ∈ ℝ^{h×d}, W₂ ∈ ℝ^{d×h},  h = 4d
HaLViT:    FFN(x) = W₁ᵀ · GELU(W₁ x)        store W₁ only; 8d² → 4d²  (−50% FFN)
```

GELU is smooth, non-polynomial, and non-saturating on one side; `GELU(W₁x)` reliably exits
`range(W₁)`. The ViT argument transfers **directly**. This is the robust hypothesis the project leads
with. (Shapes: `W₁ᵀ ∈ ℝ^{d×h}` matches `W₂` exactly, so the substitution is drop-in.)

## 3. Attention path — weak argument (the open risk, R1)

```
K = W_kv · x      V = W_kvᵀ · x          (store W_kv, derive V-proj as transpose)
Q = W_q  · x      Out = W_qᵀ · x̂          (store W_q,  derive out-proj as transpose)
```

Keys and Values are **both linear** in the same `x` — there is **no nonlinearity between them**. The
only nonlinear mixing is the downstream softmax over `QKᵀ/√d`. Under **causal masking** the softmax is
taken over a strictly lower-triangular window, which *reduces* the effective mixing that would justify
treating `W_kvᵀ` as independent. So:
- the argument here is **heuristic**, relying on softmax+depth to supply effective nonlinearity;
- HaLViT shows it works empirically in (bidirectional) ViT, but **causal** LMs are untested — exactly
  the question. Expect the attention arm to be the fragile one; the per-sublayer ablation (§7 of the
  master roadmap, FR-8) isolates it.

## 4. Modern-LLM interactions (stretch only — FR-9, not in the core)

- **RoPE** is applied to `Q` and `K` *after* their projections; it does not touch `V`. It therefore
  composes with W+Wᵀ sharing without conflict (sharing acts on the projection weights, RoPE on the
  projected vectors).
- **SwiGLU** `FFN(x) = W_down · (SiLU(W_gate x) ⊙ (W_up x))`: `W_up ∈ ℝ^{h×d}`, `W_down ∈ ℝ^{d×h}` are
  transpose-shaped. The SiLU-gated elementwise product exits `range(W_up)`, so `W_down = W_upᵀ` is the
  HaLViT move — **3 matrices → 2** (−33% FFN, not −50%). `W_gate` stays free.
- **GQA** (multiple Q heads share one K/V pair) means a shared `W_kv` serves even more roles at once —
  the most aggressive compression point and the most empirically uncertain. Stretch finding only.

## 5. Numerical correctness check (FR-7 / NFR-5 — runs before any training)

The check that `model/sharing.py` must pass (encoded in `tests/test_sharing.py`):
1. **Single stored tensor:** the transpose path must reference the *same* `nn.Parameter` (id-equal),
   not a `.clone()`/`.detach()` copy. Verified by parameter identity + `param_count(unique)`.
2. **Gradient flows through both roles:** after a backward pass, the stored `W` has a non-None grad
   that is the *sum* of the contributions from its forward use and its transpose use — no stop-grad,
   no detached branch.
3. **Param count matches theory:** measured unique-parameter count equals the §02 formula for each
   arm (deviation > 1% must be explained — G2).
4. **Disabled == standard:** with sharing flags off, the module is bit-for-bit a standard FFN/attention
   (same param count, same forward given same init).

A clean pass here is the precondition for trusting any perplexity number.
