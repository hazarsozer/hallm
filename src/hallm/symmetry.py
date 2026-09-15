"""FFN Jacobian symmetry (spec 2026-09-15 §5): how much of an FFN's input Jacobian is antisymmetric.

A W+Wᵀ FFN y = Wᵀ·GELU(W·u) has J = Wᵀ·D·W, exactly symmetric for any diagonal D, so it cannot
produce the antisymmetric ("rotation-like") part of an update. `rotation_share` measures how much of
that part a trained FFN actually uses: ‖½(J − Jᵀ)‖²_F / ‖J‖²_F at real inputs u, where
J(u) = W_down · diag(GELU′(W_up·u + b_up)) · W_up. Reference points: 0 for W+Wᵀ, ~0.5 random, 1 pure
rotation. Computed in float64: for a symmetric J the numerator is a difference of near-equal terms.

The share is basis-dependent (spec 2026-09-15 §5, added after the final code review): it is measured
w.r.t. u = LN(x), which includes the LayerNorm gain γ. W.r.t. the pre-gain normalized input x̂ (u =
γ·x̂), the Jacobian is J·diag(γ) — not symmetric for a W+Wᵀ FFN unless γ is uniform. `rotation_share`'s
optional `in_scale` reports the share in that rescaled basis; `hallm_rotation_shares` and
`vit_rotation_shares` expose it as `basis="xhat"`.
"""

from __future__ import annotations

import math

import torch

from hallm.model.gpt import GPT
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
                   b_up: torch.Tensor | None = None, chunk: int = 32,
                   in_scale: torch.Tensor | None = None) -> float:
    """Mean rotation share over the N inputs. w_up (h, d) maps d→h; w_down (d, h) maps h→d; u (N, d).

    `in_scale` (d,), if given, right-multiplies the Jacobian by diag(in_scale) — i.e. reports the
    share w.r.t. a rescaled input basis (e.g. u = γ·x̂, so in_scale=γ measures w.r.t. x̂) while the
    gate pre-activation u @ w_up.T + b_up is left untouched: it is the FFN's real input, not the
    rescaled one. Implemented as `w_up * in_scale[None, :]` inside the einsum only."""
    w_up, w_down, u = w_up.double(), w_down.double(), u.double()
    pre = u @ w_up.T
    if b_up is not None:
        pre = pre + b_up.double()
    g = gelu_grad(pre)                                                   # (N, h)
    w_up_j = w_up if in_scale is None else w_up * in_scale.double()[None, :]
    shares = [antisymmetric_share(torch.einsum("ih,nh,hj->nij", w_down, g[i:i + chunk], w_up_j))
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
def hallm_rotation_shares(model: GPT, idx: torch.Tensor, n_positions: int = 256, seed: int = 0,
                          basis: str = "u") -> list[float]:
    """One rotation share per layer (FFN call, depth order), each over the SAME n_positions of the
    FFN's real inputs (the LN2 outputs), a single subset drawn once (fixed seed) from all B·T
    positions of `idx` — so shares across layers form a paired depth profile. Runs the forward pass
    in eval mode (dropout etc. off), restoring the model's original training mode afterward.

    `basis`: "u" (default) measures w.r.t. u = LN2(x), the real FFN input. "xhat" measures w.r.t.
    the pre-gain normalized input x̂ (u = γ·x̂), passing `in_scale=block.ln2.weight` — the share is
    basis-dependent (spec 2026-09-15 §5): a W+Wᵀ FFN is exactly symmetric only in the u basis."""
    if basis not in ("u", "xhat"):
        raise ValueError(f"basis must be 'u' or 'xhat', got {basis!r}")

    def _read_transposed(args: tuple, kwargs: dict) -> bool:
        return kwargs.get("transposed", args[1] if len(args) > 1 else False)

    calls: list[tuple[MLP, torch.Tensor, bool, torch.Tensor | None]] = []
    hooks = []
    for block in model.blocks:
        gain = block.ln2.weight if basis == "xhat" else None

        def _hook(m, args, kwargs, gain=gain):
            calls.append((m, args[0].detach(), _read_transposed(args, kwargs), gain))

        hooks.append(block.mlp.register_forward_pre_hook(_hook, with_kwargs=True))
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
    for mlp, x, transposed, gain in calls:
        u = x.reshape(-1, x.shape[-1])
        w_up, w_down, b_up = effective_ffn_weights(mlp, transposed)
        out.append(rotation_share(w_up, w_down, u[pick.to(u.device)], b_up=b_up, in_scale=gain))
    return out


@torch.no_grad()
def vit_rotation_shares(vit_model, pixel_values: torch.Tensor, n_positions: int = 256,
                        seed: int = 0, basis: str = "u") -> list[float]:
    """Same measure for a Hugging Face ViTModel (DeiT-small loads as one): the FFN input is the
    `layernorm_after` output, i.e. the input of the up projection (W_up, biased); the down projection
    is the second MLP linear. Requires exact GELU so gelu_grad is the right derivative.

    NOTE: as of the installed `transformers` (5.17.0), `ViTModel` has no `.encoder.layer` /
    `.intermediate.dense` / `.output.dense` — the encoder stack is `vit_model.layers` directly and
    each `ViTLayer`'s FFN is `layer.mlp` with `fc1` (up, d→h) / `fc2` (down, h→d); `fc1`'s input is
    still exactly the `layernorm_after` output (confirmed from `ViTLayer.forward` source), so the
    measured quantity is unchanged from the plan's description — only the attribute path differs.

    Draws ONE position subset shared by every layer (so all ViT layers see the same number of token
    positions, mirroring `hallm_rotation_shares`), and runs the model in eval mode, restoring its
    original training mode afterward.

    `basis`: "u" (default) measures w.r.t. u = layernorm_after(x), the real FFN input. "xhat"
    measures w.r.t. the pre-gain normalized input (in_scale=layer.layernorm_after.weight) — see
    `hallm_rotation_shares` and spec 2026-09-15 §5."""
    if vit_model.config.hidden_act != "gelu":
        raise ValueError(f"expected hidden_act='gelu', got {vit_model.config.hidden_act!r}")
    if basis not in ("u", "xhat"):
        raise ValueError(f"basis must be 'u' or 'xhat', got {basis!r}")
    layers = vit_model.layers
    inputs: dict[int, list[torch.Tensor]] = {i: [] for i in range(len(layers))}
    hooks = [layer.mlp.fc1.register_forward_pre_hook(
                 lambda m, args, i=i: inputs[i].append(args[0].detach().reshape(-1, args[0].shape[-1])))
             for i, layer in enumerate(layers)]
    was_training = vit_model.training
    vit_model.eval()
    try:
        vit_model(pixel_values=pixel_values)
    finally:
        for h in hooks:
            h.remove()
        vit_model.train(was_training)
    n_total = torch.cat(inputs[0]).shape[0]
    gen = torch.Generator().manual_seed(seed)
    pick = torch.randperm(n_total, generator=gen)[:n_positions]
    out = []
    for i, layer in enumerate(layers):
        u = torch.cat(inputs[i])
        up, down = layer.mlp.fc1, layer.mlp.fc2
        gain = layer.layernorm_after.weight if basis == "xhat" else None
        out.append(rotation_share(up.weight, down.weight, u[pick.to(u.device)], b_up=up.bias, in_scale=gain))
    return out
