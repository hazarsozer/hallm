# T-006 sizing-pilot calibration notes

> **2026-10-12 — do not cite the numbers below as p-hop results. The pilot task has a position
> shortcut, found before the grid was run.**
>
> `PHopInductionTask` plants its hop chain by drawing each landing position uniformly from the room
> left in a short sequence (20 tokens at p=4). The chain collapses toward the start, so the answer
> sits at **position 1** far more often than anywhere else, and a strategy that follows no hop at
> all — "copy the token at position 1" — scores well above chance (1/256 = 0.4%):
>
> | p | answer is the token at position 1 | shallow `A0@L1` in the pilot |
> |---|---|---|
> | 3 | 46.9% | 39.9% |
> | 4 | 61.0% | 54.8% |
> | 5 | 72.8% | 69.5% |
>
> (`scripts/synth_shortcut_check.py`, 20,000 problems per row.) A diagnostic run confirms the
> shallow model learns exactly this: `A0@L1` at p=4 (seed 0, 20k steps) outputs the position-1
> token on 82.0% of 4,000 held-out problems, and is right on 84.4% of the problems whose answer is
> at position 1 and on 1.7–3.9% of the rest (overall 53.0%).
>
> What this means for everything below:
> - The shallow model's ~50% "ceiling" is the shortcut, not a capacity limit. Accuracy rose with p
>   because the shortcut gets stronger with p, not because the task got easier to reason about.
> - The deep and looped models' accuracies (85–95%) are a mix of the shortcut and whatever they do
>   beyond it. How much of that is hop-following was not measured.
> - **The validity check is not established.** "Looped within 5 pp of unshared deep" was read on
>   this task, so it does not show that the pipeline reproduces Saunshi et al.
> - The setup is also far from theirs: they use an alphabet of 4, sequences of 256 and p = 16–32
>   (chance 25%); the pilot used 256 symbols, 16–24 tokens and p = 3–5.
>
> The grid was not run. The generator needs replacing and the pilot redoing before any Track 2
> comparison is read. The original notes are kept below as the record of what was run.

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

**How the choice was made (read this before citing p=4).** The order of events was p=4 → p=5 →
p=3. p=4 ran first and passed the validity check with both sizing thresholds a few points off. p=5
and p=3 were then tried to hit the thresholds exactly: p=5 overshot the shallow ceiling, and p=3
hit both thresholds but failed the validity check. p=4 was locked *after* seeing all three, so the
difficulty was not chosen blind, and a validity check that passes at the chosen level and fails at
a neighbouring one is weaker evidence than one that passes everywhere. The p=3 and p=5 runs are
kept as result files (`synth-p_hop_induction-L*-*-s0-p3d160.json`, `…-p5d160.json`) and belong next
to p=4 in any write-up. Two more limits: p=3/p=5 ran at d=160 and p=4 at d=128, so the levels
differ in width as well as hops; and every number above is one seed (seed 0).

Also worth noting for whoever runs the p=5/p=3 configs later: `AdditionTask` and
`BindingChainTask` haven't been calibrated at all yet — only p-hop induction. The full grid across
all three tasks is Phase 2 scope, not part of this build+pilot deliverable.
