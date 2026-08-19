"""Ladder shapes hit the exact non-embedding counts from spec 06 §3, and the generator emits
protocol-identical configs that differ only in {shape, arm, seed, out_dir}."""
from __future__ import annotations

from hallm.experiment import load_experiment
from hallm.model import GPT, SHAPES, arm_config

from scripts.gen_ladder_configs import EXISTING, generate


def test_ladder_shape_param_counts():
    # non-embedding params follow 12·d²·L (core sublayers) + d(2L+1) LayerNorm params (roadmap §3,§4)
    # Shapes: s30h=L4, s30=L8, s30x2=L16, s30x4=L32, all at d=512
    def expected_params(L: int, d: int = 512) -> int:
        return 12 * d * d * L + d * (2 * L + 1)

    expected = {
        "s30h": expected_params(4),
        "s30": expected_params(8),
        "s30x2": expected_params(16),
        "s30x4": expected_params(32),
    }
    for shape, count in expected.items():
        model = GPT(arm_config(SHAPES[shape], "A0"))
        assert model.num_parameters(non_embedding=True) == count, shape


def test_generate_ladder(tmp_path):
    queue = generate(tmp_path)
    yamls = sorted(p.name for p in tmp_path.glob("*.yaml"))
    assert len(yamls) == 18  # 3 rungs × 2 arms × 3 seeds
    assert len(queue) == 18 - len(EXISTING) == 14
    assert all("s1337" not in q or "L4" in q for q in queue)  # only L4 keeps seed 1337 (others exist)
    assert [q for q in queue if "L4" in q] == queue[:6]  # L4 runs drain first (spec 06 §4)

    # every generated config loads, and a pair differs ONLY in sharing flags
    a0, _ = load_experiment(tmp_path / "L4-A0-s1338.yaml")
    a2, tc = load_experiment(tmp_path / "L4-A2-s1338.yaml")
    assert (a0.n_layer, a0.arm, a2.arm) == (4, "A0", "A2")
    assert tc.seed == 1338 and tc.max_steps == 50_000 and tc.batch_size == 12 and tc.grad_accum == 2

    qfile = (tmp_path / "queue.txt").read_text().splitlines()
    assert qfile == queue
