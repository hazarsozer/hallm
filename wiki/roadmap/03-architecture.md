# 03 — Architecture & Config Schema

> Companion to `ROADMAP.md` §4. Concrete candidate shapes and the `ModelConfig` schema that
> `model/config.py` implements (NFR-6). Size is a *range*; the exact point is locked in Term 2 after a
> tokens/sec pilot (C4, R2).

## 1. Candidate shapes (non-embedding params; pick in Term 2)

`P_block = 12d²`, non-emb `= L·12d²`. Heads chosen so `d/heads = 64`.

| Tag | d | L | heads | h=4d | non-emb params | whole-model @ V=32k,T=512 |
|-----|---|---|-------|------|----------------|---------------------------|
| `s10` | 384 | 6 | 6 | 1536 | ≈ 10.6M | ≈ 27M |
| `s30` | 512 | 8 | 8 | 2048 | ≈ 25.2M | ≈ 42M |
| `s60` | 640 | 10 | 10 | 2560 | ≈ 49.2M | ≈ 70M |
| `s124` | 768 | 12 | 12 | 3072 | ≈ 84.9M | ≈ 124M (GPT-2 @ V=50k) |

Recommendation: **start at `s30` (512/8)** — four from-scratch arms fit the 4070 comfortably and the
Term-2 timeline; scale up only if tokens/sec allows. `smoke` config is a micro shape (`d=64,L=2,h=256`).

## 2. `ModelConfig` schema (dataclass / pydantic)

```
ModelConfig:
  # shape
  vocab_size: int = 32000          # GPT-2 BPE → 50257 for s124
  block_size: int = 512            # context T
  n_embd:     int = 512            # d
  n_layer:    int = 8              # L
  n_head:     int = 8              # d/n_head == head_dim (=64)
  ffn_mult:   int = 4              # h = ffn_mult * d
  # regularization / init
  dropout:    float = 0.0
  bias:       bool  = False        # nanoGPT-style: no bias in Linears/LN
  tie_embeddings: bool = True      # LM head shares token embedding
  # --- sharing knobs (the four arms live here) ---
  share_cross_layer: bool = False  # A1/A3: reuse one block across all L layers (ALBERT, depth)
  share_intra_ffn:   bool = False  # A2/A3: FFN W₂ = W₁ᵀ (HaLViT, width)
  share_intra_attn:  bool = False  # A2/A3: V=Kᵀ-path, O=Qᵀ-path (HaLViT, width)
  sharing_warmup_steps: int = 0    # R3: enforce ties only after N steps (0 = from start)
```

### Arm presets (exactly four; `configs/armX_*.yaml` set these)
| Arm | share_cross_layer | share_intra_ffn | share_intra_attn |
|-----|:-:|:-:|:-:|
| A0 none | F | F | F |
| A1 ALBERT | **T** | F | F |
| A2 HaLViT | F | **T** | **T** |
| A3 combined | **T** | **T** | **T** |

Ablation configs (G4/FR-8): A2 with `share_intra_attn=F` (FFN-only) and `share_intra_ffn=F` (attn-only).

## 3. Module decisions (recap, with the concrete picks)
- Decoder-only causal Transformer, **Pre-LN**, LayerNorm (no bias), **GELU** FFN.
- **Learned absolute** positions (core); RoPE deferred to the TinyLlama stretch.
- **Plain MHA**, `head_dim = 64`, no GQA in the core.
- **Weight-tied LM head** — keeps the embedding floor identical across all arms (clean comparison).
- LayerNorm params **never shared** (SPIN precaution, codex §Medium).
- `flash`/SDPA attention via `torch.nn.functional.scaled_dot_product_attention` where available.
