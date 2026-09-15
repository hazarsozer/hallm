"""Golden logits of the model code as it was BEFORE the transposed-loop amendment (spec 2026-09-15
§7 invariant: with loop_pass2=None every existing arm runs exactly the same operations as before).

Written ONCE by `uv run python tests/prepatch_fixture.py` before the model is touched. Never
regenerate it afterwards, or test_default_path_is_bit_identical loses its meaning."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from hallm.model import GPT, SHAPES, arm_config

SMOKE = SHAPES["smoke"]
CASES = {"A0": SMOKE, "A2": SMOKE, "A2-attn": SMOKE, "A1u2": dataclasses.replace(SMOKE, n_layer=4)}
FIXTURE = Path(__file__).parent / "fixtures" / "prepatch_logits.pt"


def make() -> dict[str, torch.Tensor]:
    out = {}
    for arm, base in CASES.items():
        torch.manual_seed(0)
        model = GPT(arm_config(base, arm)).eval()
        x = torch.randint(0, base.vocab_size, (2, 16), generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            logits, _ = model(x, x)   # with targets → logits for every position
        out[arm] = logits
    return out


if __name__ == "__main__":
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(make(), FIXTURE)
    print(f"wrote {FIXTURE}")
