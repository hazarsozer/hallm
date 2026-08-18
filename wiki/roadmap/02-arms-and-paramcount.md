# 02 — Four Arms & Parameter-Count Derivation

> Companion to `ROADMAP.md` §3. Full symbolic + numeric param accounting and the exact invariants
> `tests/test_sharing.py` asserts. This is the spec the param-count test verifies against (FR-7, G2).

## 1. Symbolic accounting (per standard block, biases/LN omitted)

Dims: vocab `V`, context `T`, model dim `d`, layers `L`, FFN hidden `h = 4d`, MHA (no GQA).

| Component | Matrices | Params |
|-----------|----------|--------|
| Attention | `Q, K, V, O` each `d×d` | `4d²` |
| FFN | `W₁ (h×d)`, `W₂ (d×h)` | `2·h·d = 8d²` |
| **Block total** | | **`12d²`** |
| Token embedding | `V×d` | `V·d` |
| Position embedding (learned) | `T×d` | `T·d` |
| LM head | tied to token emb | `0` |

`P_embed = (V + T)·d`.  `P_block = 12d²`.  **Baseline `P(A0) = P_embed + L·P_block`.**

## 2. Per-arm formulas

| Arm | Shared | Block params | Total params | Non-emb ratio |
|-----|--------|--------------|--------------|---------------|
| **A0** baseline | — | `12d²` (×L) | `P_embed + 12L d²` | `1` |
| **A1** ALBERT (depth) | one block reused ×L | `12d²` (×1) | `P_embed + 12 d²` | `1/L` |
| **A2** HaLViT (width) | `W₂=W₁ᵀ` (FFN 8d²→4d²); `V=K`,`O=Q` transposes (attn 4d²→2d²) | `6d²` (×L) | `P_embed + 6L d²` | `1/2` |
| **A3** combined | one block, itself halved | `6d²` (×1) | `P_embed + 6 d²` | `1/2L` |

**Idealized affected-subblock reduction for A3 = `2L` (= 24× at L=12)** — this is the honest headline,
*before* embeddings/norms/biases (codex: NOT the discredited 52× whole-model figure).

## 3. Numeric tables

### (a) GPT-2-124M shape: `V=50257, T=1024, d=768, L=12`  → `P_embed ≈ 39.4M`, `P_block ≈ 7.08M`

| Arm | Block params | Whole-model | Whole-model ratio | Non-emb ratio |
|-----|--------------|-------------|-------------------|---------------|
| A0 | 84.9M | **124.3M** | 1.00 | 1.00 |
| A1 | 7.08M | **46.5M** | 0.37 | 0.083 |
| A2 | 42.5M | **81.9M** | 0.66 | 0.50 |
| A3 | 3.54M | **42.9M** | 0.35 | 0.042 |

### (b) Small "go-smaller" shape: `V=32000, T=512, d=512, L=8`  → `P_embed ≈ 16.7M`, `P_block ≈ 3.15M`

| Arm | Block params (×L) | Whole-model | Whole-model ratio | Non-emb ratio |
|-----|-------------------|-------------|-------------------|---------------|
| A0 | 25.2M | **41.9M** | 1.00 | 1.00 |
| A1 | 3.15M | **19.8M** | 0.47 | 0.125 (1/8) |
| A2 | 12.6M | **29.3M** | 0.70 | 0.50 |
| A3 | 1.57M | **18.3M** | 0.44 | 0.0625 (1/16) |

## 4. The embedding-floor problem (key methodological point)

At small scale the **unshared** token-embedding table dominates `P`, so whole-model ratios (0.35–0.70)
are far milder than the block-only ratios (0.04–0.50). Consequences, baked into the plan:

1. **Headline metric = non-embedding parameter count** (the axis the mechanism acts on). Whole-model is
   a secondary, honestly-labeled column. Never a single dramatic "Nx".
2. To make sharing visible whole-model, a **secondary deep-narrow config** (larger `L`, smaller `V` via
   16k–32k BPE) lets blocks dominate. Optional, logged as a config variant.
3. A1 replicates **cross-layer sharing only**, *not* ALBERT's embedding factorization, to isolate the
   depth-sharing variable. (Embedding factorization is a separate lever — flagged for the advisor.)

## 5. Test invariants (`tests/test_sharing.py` — measured == formula, tol 1%)

For a fixed tiny `(V,T,d,L,h=4d)`, assert unique-parameter counts:
- `params(A0) == P_embed + 12·L·d²`
- `params(A1) == P_embed + 12·d²`            ← independent of L (vary L: 2,4,8 → same block contribution)
- `params(A2) == P_embed + 6·L·d²`            ← FFN-only: `P_embed + (4+4)L d²`; attn-only: `P_embed + (2+8)L d²`
- `params(A3) == P_embed + 6·d²`
- Identity check: A2's stored `W₁` is the *same* `nn.Parameter` used in the transpose path (id-equal).
- Ratio checks: `nonemb(A1)/nonemb(A0) == 1/L`, `nonemb(A2)/nonemb(A0) == 1/2`, `nonemb(A3)/nonemb(A0) == 1/(2L)`.

These five families of assertions are the proof that the sharing is implemented correctly rather than
plausibly. They run in `< 5 min` on CPU (use-case scenario 2 / FR-7).
