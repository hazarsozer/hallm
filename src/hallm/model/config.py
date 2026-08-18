"""Model configuration and the four experimental arms.

A single `ModelConfig` carries the architecture shape plus three orthogonal sharing flags. The four
thesis arms (and the two ablation variants) are just specific flag combinations — see `ARMS`. This
keeps the sharing mechanism fully decoupled from the architecture (NFR-6) and lets one training
harness instantiate every arm from one config (FR-3).

Mapping (ROADMAP.md §3, roadmap/03-architecture.md):
    flag                | axis                         | arms
    share_cross_layer   | depth  (ALBERT, reuse block) | A1, A3
    share_intra_ffn     | width  (HaLViT, W2 = W1ᵀ)     | A2, A3
    share_intra_attn    | width  (HaLViT, V=Kᵀ, O=Qᵀ)   | A2, A3
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelConfig:
    """Architecture + sharing configuration for one model arm.

    All four arms share an identical shape; only the three ``share_*`` flags differ, so any
    perplexity gap is attributable to the sharing scheme (matched-budget control).
    """

    # --- shape ---
    vocab_size: int = 32000
    block_size: int = 512          # context length T
    n_embd: int = 512              # model dim d
    n_layer: int = 8               # number of transformer blocks L
    n_head: int = 8                # n_embd must be divisible by n_head (head_dim = n_embd // n_head)
    ffn_mult: int = 4              # FFN hidden h = ffn_mult * n_embd

    # --- regularization / init ---
    dropout: float = 0.0
    bias: bool = False             # nanoGPT-style: no bias in Linear/LayerNorm
    tie_embeddings: bool = True    # LM head weight-tied to token embedding (keeps embed floor equal)

    # --- sharing knobs (the four arms live here) ---
    share_cross_layer: bool = False   # A1/A3 — reuse one block across all L layers (ALBERT, depth)
    share_intra_ffn: bool = False     # A2/A3 — FFN W2 = W1ᵀ (HaLViT, width)
    share_intra_attn: bool = False    # A2/A3 — V = K-path transpose, O = Q-path transpose (HaLViT)
    sharing_warmup_steps: int = 0     # R3 — enforce intra-layer ties only after N steps (0 = always)

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.ffn_mult < 1:
            raise ValueError(f"ffn_mult ({self.ffn_mult}) must be >= 1")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def ffn_hidden(self) -> int:
        return self.ffn_mult * self.n_embd

    @property
    def arm(self) -> str:
        """Canonical arm tag derived from the sharing flags (A0/A1/A2/A3 or an ablation label)."""
        for tag, flags in ARMS.items():
            if (
                flags["share_cross_layer"] == self.share_cross_layer
                and flags["share_intra_ffn"] == self.share_intra_ffn
                and flags["share_intra_attn"] == self.share_intra_attn
            ):
                return tag
        return "custom"


# --- the four arms (+ two ablation variants for G4/FR-8) ---
# Each maps a tag to the three sharing-flag values. A2 shares BOTH sublayers by default; the ablation
# variants isolate the FFN path (strong column-space argument) from the attention path (weak).
ARMS: dict[str, dict[str, bool]] = {
    "A0": dict(share_cross_layer=False, share_intra_ffn=False, share_intra_attn=False),  # baseline
    "A1": dict(share_cross_layer=True, share_intra_ffn=False, share_intra_attn=False),   # ALBERT
    "A2": dict(share_cross_layer=False, share_intra_ffn=True, share_intra_attn=True),    # HaLViT
    "A3": dict(share_cross_layer=True, share_intra_ffn=True, share_intra_attn=True),     # combined
    "A2-ffn": dict(share_cross_layer=False, share_intra_ffn=True, share_intra_attn=False),   # ablation
    "A2-attn": dict(share_cross_layer=False, share_intra_ffn=False, share_intra_attn=True),  # ablation
}


def arm_config(base: ModelConfig, arm: str) -> ModelConfig:
    """Return a copy of ``base`` with the sharing flags set for the named ``arm``.

    The shape (vocab/block/embd/layer/head/ffn) is preserved exactly so all arms are matched.
    """
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    return replace(base, **ARMS[arm])


# --- candidate shapes (roadmap/03-architecture.md §1); pick the size in Term 2 after a pilot ---
# head_dim is held at 64 (n_head = n_embd // 64). `smoke` is the micro shape used by the test suite.
SHAPES: dict[str, ModelConfig] = {
    "smoke": ModelConfig(vocab_size=256, block_size=64, n_embd=64, n_layer=2, n_head=2, ffn_mult=4),
    "s10": ModelConfig(vocab_size=50257, block_size=512, n_embd=384, n_layer=6, n_head=6),
    "s30": ModelConfig(vocab_size=50257, block_size=512, n_embd=512, n_layer=8, n_head=8),
    "s60": ModelConfig(vocab_size=50257, block_size=512, n_embd=640, n_layer=10, n_head=10),
    # iso-param probe: A2 at L=16 stores 6·d²·16 = 25.17M non-emb — exactly A0@s30's count (2× FLOPs)
    "s30x2": ModelConfig(vocab_size=50257, block_size=512, n_embd=512, n_layer=16, n_head=8),
    "s124": ModelConfig(vocab_size=50257, block_size=1024, n_embd=768, n_layer=12, n_head=12),
}
