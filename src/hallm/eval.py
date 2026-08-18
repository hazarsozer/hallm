"""Evaluation: perplexity + the four-way comparison table (FR-5/6, G1–G3).

Metrics are reported SEPARATELY (perplexity · params · size · FLOPs); the table also carries Δ% vs the
A0 baseline and the <2% viability gate (roadmap/05-eval-protocol.md). No opaque composite scalar.
"""

from __future__ import annotations

import math

import torch

from hallm.data import iter_eval_batches
from hallm.metrics import metrics_row
from hallm.model.config import ModelConfig
from hallm.model.gpt import GPT

VIABILITY_GATE = 0.02  # ≤2% perplexity degradation vs baseline (C2 / NFR-2 / G3)


@torch.no_grad()
def evaluate_perplexity(
    model: GPT, data, block_size: int, batch_size: int = 8, device=None
) -> float:
    """Token-level perplexity = exp(mean NLL) over non-overlapping windows of the test stream."""
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    total_nll, total_tok = 0.0, 0
    for x, y in iter_eval_batches(data, block_size, batch_size, device):
        _, loss = model(x, y)  # mean cross-entropy over tokens in the batch
        ntok = y.numel()
        total_nll += loss.item() * ntok
        total_tok += ntok
    if was_training:
        model.train()
    if total_tok == 0:
        return float("nan")
    return math.exp(total_nll / total_tok)


def evaluate_arm(
    model: GPT, cfg: ModelConfig, data, block_size: int | None = None, batch_size: int = 8, device=None
) -> dict:
    """One arm → a full results row (perplexity + params/size/FLOPs)."""
    ppl = evaluate_perplexity(model, data, block_size or cfg.block_size, batch_size, device)
    row = metrics_row(model, cfg)
    row["test_ppl"] = round(ppl, 4)
    return row


def comparison_table(rows: list[dict], baseline_arm: str = "A0") -> str:
    """Render the four-way comparison as a markdown table with Δ% vs baseline and the viability gate."""
    base = next((r for r in rows if r["arm"] == baseline_arm), None)
    base_ppl = base["test_ppl"] if base else None
    header = (
        "| arm | nonemb_M | total_M | size_fp32_MB | gflops | test_ppl | Δ% vs A0 | viable |\n"
        "|-----|----------|---------|--------------|--------|----------|----------|--------|"
    )
    lines = [header]
    for r in rows:
        if base_ppl:
            d = (r["test_ppl"] - base_ppl) / base_ppl * 100.0
            delta = f"{d:+.2f}%"
            viable = "—" if r["arm"] == baseline_arm else ("✓" if d <= VIABILITY_GATE * 100 else "✗")
        else:
            delta, viable = "n/a", "n/a"
        lines.append(
            f"| {r['arm']} | {r['non_embedding_params_M']} | {r['total_params_M']} | "
            f"{r['size_fp32_MB']} | {r['fwd_gflops']} | {r['test_ppl']} | {delta} | {viable} |"
        )
    return "\n".join(lines)
