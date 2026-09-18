"""Batch assembly for Track 2 synthetic tasks: fresh problems generated on the fly (not sampled
from a fixed corpus), with the loss masked to the answer positions only.

Follows the same (x, y) shift-by-one convention as `hallm.data.wikitext.get_batch` (x = tokens
up to the last, y = tokens shifted by one) so `GPT.forward`'s existing `ignore_index=-1` masking
is the only thing a synth training loop needs on top of the ordinary causal-LM loss.
"""

from __future__ import annotations

import random

import torch

from hallm.synth.tasks import AdditionTask, BindingChainTask, PHopInductionTask, SynthProblem, SynthTask

TASKS: dict[str, SynthTask] = {
    "p_hop_induction": PHopInductionTask(),
    "addition": AdditionTask(),
    "binding_chain": BindingChainTask(),
}


def make_problems(task: SynthTask, difficulty: int, n: int, rng: random.Random) -> list[SynthProblem]:
    return [task.generate(rng, difficulty) for _ in range(n)]


def make_batch(
    problems: list[SynthProblem], device: str | torch.device = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack problems (all must share the same total length and answer_start -- true within one
    task+difficulty, since every generator returns a fixed length per difficulty) into a masked
    (x, y) pair: `y` is -1 everywhere except the positions that predict an answer token."""
    lengths = {len(p.tokens) for p in problems}
    starts = {p.answer_start for p in problems}
    if len(lengths) != 1 or len(starts) != 1:
        raise ValueError("all problems in a batch must share the same length and answer_start")
    answer_start = starts.pop()

    tokens = torch.tensor([p.tokens for p in problems], dtype=torch.long)
    x = tokens[:, :-1]
    y = tokens[:, 1:].clone()
    cutoff = answer_start - 1  # first index in y that predicts an answer token
    y[:, :cutoff] = -1
    return x.to(device), y.to(device)
