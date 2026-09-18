# T-006 sizing-pilot calibration notes

Search over `(d, p)` for p-hop induction, looking for a difficulty level where the unshared deep
reference (`A0 @ 4k`) reaches ≥90% accuracy at a level where the shallow reference (`A0 @ k`) is
≤50% (spec 2026-09-14 §5's "depth matters" precondition), *and* the looped model (`A1u1 @ 4k`)
comes within 5pp of the deep reference (the validity check against Saunshi et al.'s published
result). All runs: seed 0, 80,000 steps, batch 64, vocab 256, `n_head=4`.

| p | d | shallow A0@L1 | deep A0@L4 | looped A1u1@L4 | deep−looped gap |
|---|---|---|---|---|---|
| 3 | 160 | **39.9%** ✓ | **94.4%** ✓ | 68.8% | **25.6pp — fails** |
| 4 | 128 | 54.8% | 87.7% | 85.0% | **2.7pp — passes** |
| 5 | 160 | 69.5% | **95.6%** ✓ | 92.7% | **2.9pp — passes** |

## What this shows

No single `(d, p)` cleanly clears every literal threshold at once — the sizing precondition and
the validity check pull toward different difficulty levels. More importantly, **p=3 is a genuine
finding, not calibration noise**: the sizing thresholds are hit exactly there, but the looped
model badly fails to match the unshared deep reference (25.6pp, five times the tolerance) — the
opposite of what happens at p=4 and p=5, where the looped model tracks the deep reference closely.

This suggests the looped-vs-unshared match quality is itself difficulty-dependent in this small
regime, at least for p-hop induction specifically. Whether that's a real, interesting property of
looped models on compositional reasoning tasks, or an artifact of this tiny scale (d=128-160,
1-4 layers) not yet reproducing Saunshi et al.'s regime cleanly, isn't something this pilot can
resolve — worth a second look before Phase 2's full grid locks in a difficulty range.

## Locked-in config

**p=4, d=128** was chosen: the validity check passes comfortably (2.7pp), and while the sizing
thresholds aren't hit at the letter (54.8% vs ≤50%, 87.7% vs ≥90%), the substantive gap (~33pp)
is unambiguous evidence that depth matters on this task. See `configs/synth/pilot-p4-*.yaml` and
the three `synth-p_hop_induction-L*-*-s0.json` result files.

Also worth noting for whoever runs the p=5/p=3 configs later: `AdditionTask` and
`BindingChainTask` haven't been calibrated at all yet — only p-hop induction. The full grid across
all three tasks is Phase 2 scope, not part of this build+pilot deliverable.
