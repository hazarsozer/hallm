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


# --- P1: mechanism decomposition. Which sublayer's sharing causes the tax? ------------------

def test_generate_p1_creates_four_configs_with_correct_flags(tmp_path):
    from hallm.experiment import load_experiment
    from scripts.gen_ladder_configs import generate_p1

    queue = generate_p1(tmp_path)
    assert len(queue) == 6

    names = {p.split("/")[-1] for p in queue}
    assert names == {f"L8-{t}-s{s}.yaml" for t in ("A2ffn", "A2attn")
                     for s in (1337, 1338, 1339)}

    # seed-major ordering: both arms of seed 1337 must complete before 1338 starts, so a
    # self-contained decomposition exists as early as possible.
    order = [p.split("/")[-1] for p in queue]
    assert order[:2] == ["L8-A2ffn-s1337.yaml", "L8-A2attn-s1337.yaml"]
    assert order[-2:] == ["L8-A2ffn-s1339.yaml", "L8-A2attn-s1339.yaml"]

    mc, tc = load_experiment(tmp_path / "L8-A2ffn-s1337.yaml")
    assert mc.share_intra_ffn is True and mc.share_intra_attn is False
    assert mc.share_cross_layer is False
    assert mc.n_layer == 8 and tc.seed == 1337 and tc.max_steps == 50_000

    mc, _ = load_experiment(tmp_path / "L8-A2attn-s1338.yaml")
    assert mc.share_intra_ffn is False and mc.share_intra_attn is True


def test_p1_protocol_matches_the_completed_l8_runs(tmp_path):
    """P1 is compared against completed L8 A0/A2 runs, so the recipe must be byte-identical."""
    from hallm.experiment import load_experiment
    from scripts.gen_ladder_configs import generate_p1

    generate_p1(tmp_path)
    _, p1 = load_experiment(tmp_path / "L8-A2ffn-s1337.yaml")
    _, baseline = load_experiment("configs/runs/L8-A0-s1338.yaml")
    for field in ("lr", "min_lr", "warmup_steps", "max_steps", "weight_decay", "grad_clip",
                  "batch_size", "grad_accum", "dtype", "block_size"):
        assert getattr(p1, field) == getattr(baseline, field), f"{field} drifted from the protocol"


def test_p1_storage_savings_are_the_expected_thirds(tmp_path):
    """FFN-only saves 33.3% of per-layer non-embedding storage, attn-only 16.7%, both 50%."""
    from hallm.experiment import load_experiment
    from hallm.metrics import count_parameters
    from hallm.model.gpt import GPT
    from scripts.gen_ladder_configs import generate_p1

    generate_p1(tmp_path)
    base, _ = load_experiment("configs/runs/L8-A0-s1338.yaml")
    ffn, _ = load_experiment(tmp_path / "L8-A2ffn-s1337.yaml")
    attn, _ = load_experiment(tmp_path / "L8-A2attn-s1337.yaml")

    n = {k: count_parameters(GPT(c))["non_embedding"] for k, c in
         (("base", base), ("ffn", ffn), ("attn", attn))}
    assert abs((1 - n["ffn"] / n["base"]) - 1 / 3) < 0.01
    assert abs((1 - n["attn"] / n["base"]) - 1 / 6) < 0.01


def test_p4_unshared_frontier_shapes_land_within_2pct_of_a0_l8_storage():
    """P4 frontier (spec amendment A3): unshared points at ~25.2M non-emb, re-invested along width."""
    ref = GPT(arm_config(SHAPES["s30"], "A0")).num_parameters(non_embedding=True)
    for shape in ("p4w720l4", "p4w360l16"):
        n = GPT(arm_config(SHAPES[shape], "A0")).num_parameters(non_embedding=True)
        assert abs(n - ref) / ref < 0.02, (shape, n, ref)


# --- Fixed-storage pilot (spec 2026-09-14 §4.1) ---------------------------------------------

def test_generate_pilot_order_and_flags(tmp_path):
    from scripts.gen_ladder_configs import generate_pilot

    queue = generate_pilot(tmp_path)
    order = [p.split("/")[-1] for p in queue]
    assert order == [
        "L8-A1u4-s1337.yaml", "L16-A1u8-s1337.yaml", "L9-A2attn-s1337.yaml", "L10-A2attn-s1337.yaml",
        "L8-A1u4-s1338.yaml", "L16-A1u8-s1338.yaml", "L10-A2attn-s1338.yaml",
        "L8-A1u4-s1339.yaml", "L16-A1u8-s1339.yaml", "L10-A2attn-s1339.yaml",
    ]
    assert (tmp_path / "queue-pilot.txt").read_text().splitlines() == queue
    mc, tc = load_experiment(tmp_path / "L16-A1u8-s1338.yaml")
    assert (mc.n_layer, mc.n_unique_blocks, mc.arm, tc.seed) == (16, 8, "A1u8", 1338)
    mc, _ = load_experiment(tmp_path / "L9-A2attn-s1337.yaml")
    assert mc.n_layer == 9 and mc.share_intra_attn and not mc.share_intra_ffn


def test_pilot_protocol_matches_the_completed_runs(tmp_path):
    from scripts.gen_ladder_configs import generate_pilot

    generate_pilot(tmp_path)
    _, baseline = load_experiment("configs/runs/L8-A0-s1338.yaml")
    for name in ("L8-A1u4-s1337", "L16-A1u8-s1339", "L10-A2attn-s1337"):
        _, tc = load_experiment(tmp_path / f"{name}.yaml")
        for field in ("lr", "min_lr", "warmup_steps", "max_steps", "weight_decay", "grad_clip",
                      "batch_size", "grad_accum", "dtype", "block_size"):
            assert getattr(tc, field) == getattr(baseline, field), (name, field)


def _block_matrix(cfg) -> int:
    return sum(p.numel() for p in GPT(cfg).blocks.parameters() if p.ndim == 2)


def test_looped_and_wwt_pairs_store_identical_block_weights():
    """Matched pairs (spec §2): same storage AND same unrolled depth, hence same compute."""
    assert _block_matrix(arm_config(SHAPES["s30"], "A1u4")) == _block_matrix(arm_config(SHAPES["s30"], "A2"))
    assert _block_matrix(arm_config(SHAPES["s30x2"], "A1u8")) == _block_matrix(arm_config(SHAPES["s30x2"], "A2"))


def test_a2attn_depth_points_bracket_a0_l8_storage():
    ref = GPT(arm_config(SHAPES["s30"], "A0")).num_parameters(non_embedding=True)
    n9 = GPT(arm_config(SHAPES["s30l9"], "A2-attn")).num_parameters(non_embedding=True)
    n10 = GPT(arm_config(SHAPES["s30l10"], "A2-attn")).num_parameters(non_embedding=True)
    assert n9 < ref < n10


# --- FineWeb-Edu bridge (spec 2026-09-14 §4.3) -----------------------------------------------

def test_generate_bridge_is_the_iso_storage_pair_on_fineweb(tmp_path):
    from scripts.gen_ladder_configs import generate_bridge

    queue = generate_bridge(tmp_path)
    assert [p.split("/")[-1] for p in queue] == ["L8-A0-s1337-fw.yaml", "L16-A2-s1337-fw.yaml"]
    _, baseline = load_experiment("configs/runs/L8-A0-s1338.yaml")
    for name, (depth, arm) in (("L8-A0-s1337-fw", (8, "A0")), ("L16-A2-s1337-fw", (16, "A2"))):
        mc, tc = load_experiment(tmp_path / f"{name}.yaml")
        assert (mc.n_layer, mc.arm, tc.dataset, tc.seed) == (depth, arm, "fineweb-edu", 1337)
        for field in ("lr", "min_lr", "warmup_steps", "max_steps", "batch_size", "grad_accum", "dtype"):
            assert getattr(tc, field) == getattr(baseline, field), (name, field)


# --- Transposed loop (spec 2026-09-15 §2) ----------------------------------------------------

def test_generate_transposed_order_and_flags(tmp_path):
    from scripts.gen_ladder_configs import generate_transposed

    queue = generate_transposed(tmp_path)
    assert [p.split("/")[-1] for p in queue] == [
        "L8-A1u4t-s1337.yaml", "L8-A1u4n-s1337.yaml", "L8-A1u4a-s1337.yaml"]
    assert (tmp_path / "queue-transposed.txt").read_text().splitlines() == queue
    for suffix, mode in (("t", "transpose"), ("n", "negate"), ("a", "scaled")):
        mc, tc = load_experiment(tmp_path / f"L8-A1u4{suffix}-s1337.yaml")
        assert (mc.n_layer, mc.n_unique_blocks, mc.loop_pass2, mc.arm, tc.seed) == \
            (8, 4, mode, f"A1u4{suffix}", 1337)


def test_transposed_recipe_matches_the_plain_loop_run(tmp_path):
    """Everything but the run directory equals the completed L8-A1u4-s1337 run's config."""
    import dataclasses as dc

    from scripts.gen_ladder_configs import generate_transposed

    generate_transposed(tmp_path)
    _, loop = load_experiment("configs/runs/L8-A1u4-s1337.yaml")
    for suffix in "tna":
        _, tc = load_experiment(tmp_path / f"L8-A1u4{suffix}-s1337.yaml")
        assert dc.replace(tc, out_dir=loop.out_dir) == loop, suffix
