# Transposed loop: outcome and the §6 predictions scored

Scores the predictions recorded in `docs/superpowers/specs/2026-09-15-transposed-loop-design.md` §6
against the runs that landed 2026-09-16. Nothing here is generated; the numbers come from
`results/runs/*.json` and `results/analysis/ffn-symmetry.json`.

## The predictions

With M the change one pass makes, M = S + A (symmetric + antisymmetric), the first-order argument
says plain loop ≈ I + 2M, transposed ≈ I + 2S, negated ≈ I + 2A.

| # | Prediction | Outcome | Evidence |
|---|---|---|---|
| 1 | `t` worse than the plain loop on val_ppl | **held** | t +3.39% |
| 2 | `n` better than `t` | **failed** | n +4.50%, worse than t |
| 3 | FFN α in `a` drifts below 1, possibly negative | **failed** | mlp α 0.42, 1.72, 1.83, 2.16 |
| 4 | LM FFN rotation share well above 0 and above DeiT-small | **held** | LM 0.449–0.477, DeiT 0.395 |

Two of the three run-level predictions failed. The spec's own rule for that case: *"If 1–3 fail, the
first-order argument is wrong for trained LMs, which is itself worth reporting."*

## What survives

**The descriptive gap is real (prediction 4).** Trained LM FFNs are more rotation-like than
DeiT-small's, and the depth profile differs in kind, not only in level:

| model | mean | min layer | max layer | shape |
|---|---|---|---|---|
| L4-A0-s1337 | 0.477 | 0.466 | 0.488 | flat |
| L8-A0-s1337 | 0.461 | 0.436 | 0.489 | flat |
| L16-A0-s1337 | 0.449 | 0.394 | 0.492 | flat, slight dip mid-stack |
| deit-small-patch16-224 | 0.395 | 0.281 (layer 1) | 0.475 | rises with depth |
| random reference | 0.496 | — | — | — |

DeiT's early blocks sit far below anything the LMs reach; the LMs never go below 0.394 and mostly sit
at 0.44–0.49, i.e. near the random reference. So a W+Wᵀ transfer discards less in early vision blocks
than anywhere in an LM, which is consistent with HaLViT working in ViTs and W+Wᵀ failing here.

**Three caveats, all load-bearing:**

1. The comparison is uncontrolled — different recipe, scale, data and task. Spec §5 says so, and §9
   (R3) notes the controlled version (a small ViT trained with our recipe) was left out of scope.
2. DeiT's 0.395 is itself far from 0. Even in vision, FFN Jacobians are not close to symmetric, so
   "vision FFNs are nearly symmetric, language ones are not" is **not** what was measured. The
   honest claim is a gap in degree, concentrated in early layers.
3. Sharing does not move the share: looped LMs sit where unshared ones do (0.456–0.459 vs
   0.461–0.463), and the transposed variants come out slightly *more* asymmetric (n 0.477).

## What does not survive

The mechanism story cannot be used to explain the ordering of the variants. If the antisymmetric
part A were what the transposed loop throws away, then `n` (which keeps ≈ I + 2A) should have beaten
`t` (≈ I + 2S). It did not — `n` is the worst of the three, and the learned α values in `a` grow
with depth (up to 2.16) instead of shrinking toward 0 or going negative as prediction 3 expected.

So: **the symmetry gap explains nothing about why the transposed variants lose.** It stands only as
a descriptive difference between LM and ViT FFNs. Whatever makes `t`/`n`/`a` lose to the plain loop
is not captured by the first-order S/A decomposition — the likely suspects are the ones §6 flagged
as ignored (the GELU gates differ between passes, attention does not follow the argument, and the
LayerNorm gain is outside the decomposition).

A claim of the form "W+Wᵀ fails in LMs because LM FFNs need the rotation part" is **not** supported
by these runs. What is supported: W+Wᵀ and every transposed variant lose to plain looping at matched
storage and compute, and LM FFNs are measurably more rotation-like than DeiT-small's.
