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

import re
from dataclasses import dataclass, replace


# Transposed loop (spec 2026-09-15): how a looped arm's odd passes add their transposed update.
# Value → run-ID suffix: A1u<k>t / A1u<k>n / A1u<k>a.
LOOP_PASS2 = {"transpose": "t", "negate": "n", "scaled": "a"}


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
    causal: bool = True            # LM masks future positions; a ViT attends both ways

    # --- regularization / init ---
    dropout: float = 0.0
    bias: bool = False             # nanoGPT-style: no bias in Linear/LayerNorm
    tie_embeddings: bool = True    # LM head weight-tied to token embedding (keeps embed floor equal)

    # --- sharing knobs (the four arms live here) ---
    share_cross_layer: bool = False   # A1/A3 — reuse blocks across layers (ALBERT, depth)
    n_unique_blocks: int | None = None  # looped A1u<k>: k blocks cycled to depth L (None ⇒ 1 or L)
    share_intra_ffn: bool = False     # A2/A3 — FFN W2 = W1ᵀ (HaLViT, width)
    share_intra_attn: bool = False    # A2/A3 — V = K-path transpose, O = Q-path transpose (HaLViT)
    sharing_warmup_steps: int = 0     # R3 — enforce intra-layer ties only after N steps (0 = always)
    loop_pass2: str | None = None     # transposed loop: odd passes use Wᵀ ("transpose"|"negate"|"scaled")

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.ffn_mult < 1:
            raise ValueError(f"ffn_mult ({self.ffn_mult}) must be >= 1")
        if self.n_unique_blocks is not None:
            if not self.share_cross_layer:
                raise ValueError("n_unique_blocks requires share_cross_layer=True")
            k = self.n_unique_blocks
            if not 1 <= k <= self.n_layer or self.n_layer % k:
                raise ValueError(f"n_unique_blocks ({k}) must divide n_layer ({self.n_layer})")
        if self.loop_pass2 is not None:
            if self.loop_pass2 not in LOOP_PASS2:
                raise ValueError(f"loop_pass2 must be one of {sorted(LOOP_PASS2)}, got {self.loop_pass2!r}")
            if self.n_unique_blocks is None:
                raise ValueError("loop_pass2 requires a looped arm (n_unique_blocks set)")
            if self.share_intra_ffn or self.share_intra_attn:
                raise ValueError("loop_pass2 cannot be combined with intra-layer (W+Wᵀ) sharing")
            if self.bias:
                raise ValueError("loop_pass2 assumes bias=False (the transposed pass has no bias)")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def ffn_hidden(self) -> int:
        return self.ffn_mult * self.n_embd

    @property
    def arm(self) -> str:
        """Canonical arm tag derived from the sharing flags (A0/A1/A2/A3 or an ablation label)."""
        if self.loop_pass2 is not None:
            return f"A1u{self.n_unique_blocks}{LOOP_PASS2[self.loop_pass2]}"
        if self.n_unique_blocks not in (None, 1):
            intra = self.share_intra_ffn or self.share_intra_attn
            return "custom" if intra else f"A1u{self.n_unique_blocks}"
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


_LOOPED = re.compile(r"A1u(\d+)([tna]?)")
_PASS2_FOR_SUFFIX = {suffix: mode for mode, suffix in LOOP_PASS2.items()}


def arm_config(base: ModelConfig, arm: str) -> ModelConfig:
    """Return a copy of ``base`` with the sharing flags set for the named ``arm``.

    ``A1u<k>`` is the looped arm (spec 2026-09-14 §2): A1's cross-layer flag with k distinct blocks
    cycled to depth L. ``A1u<k>t|n|a`` is the transposed loop (spec 2026-09-15 §2): odd passes use the
    blocks' transposed weights, added, subtracted or scaled by a learned α. The shape
    (vocab/block/embd/layer/head/ffn) is preserved exactly.
    """
    m = _LOOPED.fullmatch(arm)
    if m:
        return replace(base, **ARMS["A1"], n_unique_blocks=int(m.group(1)),
                       loop_pass2=_PASS2_FOR_SUFFIX.get(m.group(2)))
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; choose from {sorted(ARMS)} or A1u<k>[t|n|a]")
    return replace(base, **ARMS[arm], n_unique_blocks=None, loop_pass2=None)


# --- candidate shapes (roadmap/03-architecture.md §1); pick the size in Term 2 after a pilot ---
# head_dim is held at 64 (n_head = n_embd // 64). `smoke` is the micro shape used by the test suite.
SHAPES: dict[str, ModelConfig] = {
    "smoke": ModelConfig(vocab_size=256, block_size=64, n_embd=64, n_layer=2, n_head=2, ffn_mult=4),
    "s10": ModelConfig(vocab_size=50257, block_size=512, n_embd=384, n_layer=6, n_head=6),
    "s30": ModelConfig(vocab_size=50257, block_size=512, n_embd=512, n_layer=8, n_head=8),
    "s60": ModelConfig(vocab_size=50257, block_size=512, n_embd=640, n_layer=10, n_head=10),
    # iso-param probe: A2 at L=16 stores 6·d²·16 = 25.17M non-emb — exactly A0@s30's count (2× FLOPs)
    "s30x2": ModelConfig(vocab_size=50257, block_size=512, n_embd=512, n_layer=16, n_head=8),
    # scaling-campaign ladder (wiki/roadmap/06-scaling-campaign.md §3): width fixed at d=512,
    # depth-scaled. s30h = L4 rung (~12.6M non-emb unshared); s30x4 = L32 stretch rung (~100.7M).
    "s30h": ModelConfig(vocab_size=50257, block_size=512, n_embd=512, n_layer=4, n_head=8),
    "s30x4": ModelConfig(vocab_size=50257, block_size=512, n_embd=512, n_layer=32, n_head=8),
    # iso-storage partner for A2attn@s30 (stores 20.98M non-emb): nearest unshared shape at the
    # campaign width d=512 is L=7 → 22.03M (+5% over target, i.e. a slightly stronger baseline).
    # Width kept at 512 so embeddings are identical and no aspect-ratio confound enters.
    "s30l7": ModelConfig(vocab_size=50257, block_size=512, n_embd=512, n_layer=7, n_head=8),
    # Fixed-storage pilot (spec 2026-09-14 §4.1): A2attn stores 10d² per layer, so at A0@L8's 96d²
    # it can buy ~20% extra depth. L9 (90d², −6%) and L10 (100d², +4%) bracket the target.
    "s30l9": ModelConfig(vocab_size=50257, block_size=512, n_embd=512, n_layer=9, n_head=8),
    "s30l10": ModelConfig(vocab_size=50257, block_size=512, n_embd=512, n_layer=10, n_head=8),
    # P4 unshared frontier points (spec amendment 2026-08-31 §A3): A0@L8's 25.17M non-emb storage
    # re-invested along width. Spec names d724/L4 and d362/L16; rounded to head-divisible widths
    # with head dim near the campaign's 64. Both ~24.9M non-emb (−1%), i.e. slightly weaker than
    # the reference — a loss against them is not explained by extra capacity.
    "p4w720l4": ModelConfig(vocab_size=50257, block_size=512, n_embd=720, n_layer=4, n_head=10),
    "p4w360l16": ModelConfig(vocab_size=50257, block_size=512, n_embd=360, n_layer=16, n_head=6),
    "s124": ModelConfig(vocab_size=50257, block_size=1024, n_embd=768, n_layer=12, n_head=12),
}


@dataclass(frozen=True)
class VisionConfig(ModelConfig):
    """A ViT's shape. Inherits every sharing flag, so `arm_config` and `build_blocks` work unchanged.

    `block_size` is the token count (patches + class token) and `vocab_size`/`tie_embeddings` are
    unused by `ViT` — they stay on the base config so manifests keep one schema.
    """

    image_size: int = 112
    patch_size: int = 16
    n_classes: int = 100
    causal: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.image_size % self.patch_size:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"
            )
        n_tokens = (self.image_size // self.patch_size) ** 2 + 1
        if self.block_size != n_tokens:
            raise ValueError(
                f"block_size ({self.block_size}) must equal patches + class token ({n_tokens})"
            )


# Controlled ViT study (spec 2026-09-16 §2): one geometry, two depths. 49 patches + class token.
VSHAPES: dict[str, VisionConfig] = {
    "v4": VisionConfig(vocab_size=1, block_size=50, n_embd=512, n_layer=4, n_head=8,
                       tie_embeddings=False),
    "v8": VisionConfig(vocab_size=1, block_size=50, n_embd=512, n_layer=8, n_head=8,
                       tie_embeddings=False),
}
