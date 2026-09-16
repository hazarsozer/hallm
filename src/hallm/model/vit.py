"""Vision transformer over the sharing-aware sublayers (spec 2026-09-16).

Identical to `gpt.py` in everything that matters for the study: the same `Block`, the same
`build_blocks` cross-layer helper and the same transposed-loop pass schedule. Only the ends differ —
a patch embedding and a class token instead of token embeddings, and a classifier head instead of a
weight-tied LM head — so an arm gap here is comparable with the same arm gap in language.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hallm.model.config import VisionConfig
from hallm.model.gpt import Block
from hallm.model.sharing import build_blocks


class ViT(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch = nn.Conv2d(3, cfg.n_embd, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.n_embd))
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = build_blocks(cfg, Block)
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.n_classes, bias=True)
        self.label_smoothing = 0.0   # set by the trainer; kept here so forward stays self-contained

        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, pixels: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B = pixels.shape[0]
        x = self.patch(pixels).flatten(2).transpose(1, 2)          # (B, n_patches, C)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        pos = torch.arange(x.shape[1], device=pixels.device)
        x = self.drop(x + self.pos_emb(pos))

        n_unique = len(self.blocks)
        for i in range(self.cfg.n_layer):
            block = self.blocks[i % n_unique]
            if self.cfg.loop_pass2 is not None and (i // n_unique) % 2 == 1:
                x = block(x, transposed=True)
            else:
                x = block(x)
        x = self.ln_f(x)
        logits = self.head(x[:, 0])                                 # class token
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        return logits, loss

    def loop_scales(self) -> dict[str, float]:
        if self.cfg.loop_pass2 != "scaled":
            return {}
        out: dict[str, float] = {}
        for j, block in enumerate(self.blocks):
            out[f"alpha_attn_{j}"] = round(block.alpha_attn.item(), 6)
            out[f"alpha_mlp_{j}"] = round(block.alpha_mlp.item(), 6)
        return out

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Unique parameters; `non_embedding` counts transformer blocks + final LN only, the vision
        analogue of the LM's non-embedding count (spec 2026-09-16 §2)."""
        seen: set[int] = set()
        modules = [self.blocks, self.ln_f] if non_embedding else [self]
        total = 0
        for m in modules:
            for p in m.parameters():
                if id(p) in seen:
                    continue
                seen.add(id(p))
                total += p.numel()
        return total
