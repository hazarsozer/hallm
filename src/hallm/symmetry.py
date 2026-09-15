"""FFN Jacobian symmetry (spec 2026-09-15 §5): how much of an FFN's input Jacobian is antisymmetric.

A W+Wᵀ FFN y = Wᵀ·GELU(W·u) has J = Wᵀ·D·W, exactly symmetric for any diagonal D, so it cannot
produce the antisymmetric ("rotation-like") part of an update. `rotation_share` measures how much of
that part a trained FFN actually uses: ‖½(J − Jᵀ)‖²_F / ‖J‖²_F at real inputs u, where
J(u) = W_down · diag(GELU′(W_up·u + b_up)) · W_up. Reference points: 0 for W+Wᵀ, ~0.5 random, 1 pure
rotation. Computed in float64: for a symmetric J the numerator is a difference of near-equal terms.
"""

from __future__ import annotations

import math

import torch

from hallm.model.sharing import MLP


def gelu_grad(z: torch.Tensor) -> torch.Tensor:
    """Exact GELU derivative Φ(z) + z·φ(z) (matches F.gelu with approximate='none')."""
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0))) + z * torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def antisymmetric_share(J: torch.Tensor) -> torch.Tensor:
    """(n, d, d) → (n,): ‖½(J − Jᵀ)‖²_F / ‖J‖²_F per matrix."""
    A = 0.5 * (J - J.transpose(1, 2))
    return A.pow(2).sum((1, 2)) / J.pow(2).sum((1, 2))


@torch.no_grad()
def rotation_share(w_up: torch.Tensor, w_down: torch.Tensor, u: torch.Tensor,
                   b_up: torch.Tensor | None = None, chunk: int = 32) -> float:
    """Mean rotation share over the N inputs. w_up (h, d) maps d→h; w_down (d, h) maps h→d; u (N, d)."""
    w_up, w_down, u = w_up.double(), w_down.double(), u.double()
    pre = u @ w_up.T
    if b_up is not None:
        pre = pre + b_up.double()
    g = gelu_grad(pre)                                                   # (N, h)
    shares = [antisymmetric_share(torch.einsum("ih,nh,hj->nij", w_down, g[i:i + chunk], w_up))
              for i in range(0, u.shape[0], chunk)]
    return float(torch.cat(shares).mean())


def effective_ffn_weights(mlp: MLP, transposed: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """(w_up (h, d), w_down (d, h), b_up) as the FFN actually applies them on this call. b_up is the
    up-projection bias — `mlp.up_bias` (shared) or `mlp.fc.bias` (unshared), both None under the
    default bias=False — and always None on a transposed pass (config forbids bias with loop_pass2)."""
    if mlp.shared:                 # HaLViT: up = W, down = Wᵀ
        return mlp.w, mlp.w.t(), mlp.up_bias
    if transposed:                 # transposed loop pass 2: up = W_downᵀ, down = W_upᵀ, no bias
        return mlp.proj.weight.t(), mlp.fc.weight.t(), None
    return mlp.fc.weight, mlp.proj.weight, mlp.fc.bias


@torch.no_grad()
def hallm_rotation_shares(model, idx: torch.Tensor, n_positions: int = 256, seed: int = 0) -> list[float]:
    """One rotation share per layer (FFN call, depth order), each over the SAME n_positions of the
    FFN's real inputs (the LN2 outputs), a single subset drawn once (fixed seed) from all B·T
    positions of `idx` — so shares across layers form a paired depth profile. Runs the forward pass
    in eval mode (dropout etc. off), restoring the model's original training mode afterward."""
    calls: list[tuple[MLP, torch.Tensor, bool]] = []
    hooks = [block.mlp.register_forward_pre_hook(
                 lambda m, args, kwargs: calls.append((m, args[0].detach(), kwargs.get("transposed", False))),
                 with_kwargs=True)
             for block in model.blocks]
    was_training = model.training
    model.eval()
    try:
        model(idx)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    n_total = calls[0][1].reshape(-1, calls[0][1].shape[-1]).shape[0]
    gen = torch.Generator().manual_seed(seed)
    pick = torch.randperm(n_total, generator=gen)[:n_positions]
    out = []
    for mlp, x, transposed in calls:
        u = x.reshape(-1, x.shape[-1])
        w_up, w_down, b_up = effective_ffn_weights(mlp, transposed)
        out.append(rotation_share(w_up, w_down, u[pick.to(u.device)], b_up=b_up))
    return out
