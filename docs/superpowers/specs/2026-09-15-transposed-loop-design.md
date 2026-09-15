# Transposed Loop — W/Wᵀ Across Depth (exploratory amendment)

> **Status: approved in design 2026-09-15 (Hazar + Claude brainstorm); spec awaiting Hazar's review.**
> Amends `2026-09-14-fixed-storage-compute-program-design.md` (§4 pilot, §6 gate). Everything here
> is **exploratory**: conceived on 2026-09-15 *after* the pilot's perplexity and probe results were
> known, and written down *before* any run of the arms below. The 2026-09-14 pre-registered
> questions, rules and results are unchanged.
>
> **Decided with Hazar:** all three variants at seed 1337 (§2) · the symmetry measurement on LM
> checkpoints plus DeiT-small (§5) · the gate rule with plain W+Wᵀ kept at one 124M seed (§4).
>
> **Decided by Claude, open to override:** whole block transposed, not FFN-only (§2) · one α per
> sublayer (§2) · pass order (§2) · run-ID tags (§2) · implementation shape (§7) · worktree
> workflow (§8).

---

## 1. Why

The pilot (3 seeds, two sizes) answered the 2026-09-14 question for perplexity: at matched
storage and compute, looping beats W+Wᵀ (−5.25% at L8, −6.10% at L16), and W+Wᵀ shows no signal
against the same-storage unshared model on perplexity or on any Track 1 probe. Looping itself is
well studied (Universal Transformers, ALBERT, Saunshi et al. 2025, Bae et al. 2025), so it cannot be
the thesis's contribution.

The thesis asks whether HaLViT's W/Wᵀ idea carries over to language models. This amendment tests
it in a second form: **across depth instead of within a layer**. A looped model's second pass
reuses each stored block with its matrices transposed, so the two passes compute different
functions at zero extra storage and identical FLOPs. Plain looping's known weakness is that its
passes are identical; Relaxed Recursive Transformers untie them with LoRA, which costs parameters.
Transposition does it for free, and it is the lab's own mechanism.

Literature check (`wiki/analyses/transposed-loop-literature-2026-09.md`): novel as specified,
~70% confidence; closest work Haber & Ruthotto 2017, Chang et al. 2018, Bae et al. 2025.

## 2. Arms

All at L8 (d=512, ctx 512, WikiText-103): 4 stored blocks, each applied twice. Same stored weights
and same analytic FLOPs as the plain loop `L8-A1u4`.

| run | pass 1 (layers 0–3) | pass 2 (layers 4–7) | extra params |
|---|---|---|---|
| `L8-A1u4-s1337` (exists) | x + f_W(x) | x + f_W(x) | — |
| `L8-A1u4t-s1337` | x + f_W(x) | x + f_Wᵀ(x) | 0 |
| `L8-A1u4n-s1337` | x + f_W(x) | x − f_Wᵀ(x) | 0 |
| `L8-A1u4a-s1337` | x + f_W(x) | x + α·f_Wᵀ(x) | 8 |

- **f_Wᵀ, the transposed block.** Pre-LN as usual, with the block's own LayerNorm gains shared
  across passes. FFN: up-projection = W_downᵀ (d→4d), down-projection = W_upᵀ (4d→d). Attention:
  Q, K, V and O each use their own matrix transposed (all d×d). The transposed pass references the
  same stored tensors, so gradients accumulate from both roles.
- **α (`a` variant):** one scalar per sublayer per block (attention and FFN; 4 blocks → 8 scalars),
  initialised to 1, so the variant starts identical to `t`. α multiplies only pass-2 updates.
  Scalars are already exempt from weight decay (`configure_optimizer`: `ndim < 2`), so nothing
  pulls α toward 0.
- **Pass order.** Layer i uses block i mod k in pass ⌊i/k⌋, the same cycling as `L8-A1u4`. Odd
  passes are transposed; with more than two passes, W and Wᵀ alternate.
- **Whole block, not FFN-only.** The idea as specified and literature-checked. FFN-only is a
  follow-up if the results are ambiguous.
- **Recipe:** the pilot recipe verbatim (the `TRAIN` block of `gen_ladder_configs.py`: 50k steps,
  lr 6e-4 cosine, batch 12 × accum 2, bf16), seed 1337, identical data order. No biases anywhere
  and a uniform N(0, 0.02) init, so no bias or init-scale decisions arise.
- **Cost:** ~2 GPU-h per run at L8, ~6 GPU-h for the three.

## 3. Rules (written before any run)

- **H-T (transposition helps):** variant X beats `L8-A1u4` at matched storage and compute: the
  paired Δ% `(PPL_X − PPL_loop) / PPL_loop` is negative in all 3 seeds and |mean| > 2 SE. Otherwise
  "no difference detected" (or "loop better" if the reverse holds). The H-L rule of 2026-09-14 §4.4,
  applied to a new pair. Metric `val_ppl`; `test_ppl` is reported alongside.
- **Also scored:** each variant against `L4-A0` (H-C, same storage) and `L8-A2` (H-L, plain W+Wᵀ at
  matched storage and compute), and on every Track 1 probe (H-R).
- **Follow-up seeds:** a variant gets seeds 1338/1339 iff its seed-1337 `val_ppl` ≤ **28.4431**
  (the plain loop's 28.1615 + 1%). Others stay at one seed and are reported as such.
- **α is descriptive:** the final α values and their trajectories are reported, with no verdict
  rule attached.

## 4. Gate (amends 2026-09-14 §6)

- If a transposed variant passes H-T on 3 seeds, it takes the **W+Wᵀ slot** at 124M. The 124M grid
  becomes: unshared `L12-A0`, the transposed variant `L24-A1u12<t|n|a>`, and the plain loop
  `L24-A1u12` (which it has to beat at scale too). If more than one passes, the one with the most
  negative mean Δ vs the plain loop goes.
- In that case plain W+Wᵀ (the gate's W+Wᵀ pick under the 2026-09-14 rule) runs at 124M with
  **one seed, queued last**. The pilot's seed spread is under 0.1 PPL (~0.4%) against a 5–6% gap,
  so one seed shows whether the gap closes at scale. It is an observation, not a verdict, and the
  first thing cut if time or budget runs short.
- **If time and compute allow** after the main grid, plain W+Wᵀ can still get its full 3 seeds at
  124M, or go to 350M (Hazar, 2026-09-15).
- If no variant passes H-T, the 2026-09-14 gate stands unchanged.
- Dropping plain W+Wᵀ from the full 124M grid changes the thesis's original question, so it needs
  Töreyin's agreement at the term-start briefing.

## 5. Symmetry measurement (why W+Wᵀ fails in LMs)

**Claim tested.** A W+Wᵀ FFN, y = Wᵀ·GELU(W·u), has the input Jacobian J = Wᵀ·D·W with D
diagonal, which is **exactly symmetric** for any D. So a W+Wᵀ FFN cannot produce the
antisymmetric (rotation-like) part of an update. If trained LM FFNs rely on that part more than
vision FFNs do, the direct transfer throws away more in language than in vision, which would
explain why HaLViT works in ViTs but not here.

**Measure.** For an FFN with up-projection W_up (4d×d) and down-projection W_down (d×4d), at
FFN input u (the LN2 output):

    J(u) = W_down · diag(GELU′(W_up·u)) · W_up            (d × d)
    rotation share(u) = ‖½(J − Jᵀ)‖²_F / ‖J‖²_F

Averaged over 256 real input positions per layer, per layer and per model. Reference points: 0
for a W+Wᵀ layer (exact), ~0.5 for random matrices, 1 for a purely antisymmetric J. Biases shift
the pre-activation only, so DeiT's biased FFN is handled by the same formula.

> **Added 2026-09-15, after the final code review, before any measurement was run.** The share is
> basis-dependent: it is measured w.r.t. u = LN(x), which includes the LayerNorm gain γ. W.r.t. the
> pre-gain normalized input x̂ (u = γ·x̂), the Jacobian is J·diag(γ) — for a W+Wᵀ FFN this is not
> symmetric unless γ is uniform. So a W+Wᵀ layer gives exactly 0 only in the u basis; both the u and
> the x̂ basis (γ folded in) are reported.

**Models.**
- LM: unshared `L4-A0`, `L8-A0` and `L16-A0` at every seed with a checkpoint; inputs are WikiText-103
  validation positions.
- Vision: DeiT-small (`facebook/deit-small-patch16-224`, GELU, pre-LN, same FFN form), inputs from
  ~256 images of a small public image set (Imagenette validation).
- Sanity: an `L8-A2` checkpoint must give 0 (to float precision).

**Hypothesis (descriptive, no verdict rule):** LM FFNs have a clearly higher rotation share than
DeiT-small's. The cross-model comparison is uncontrolled (different recipe, scale and data), so it
is reported as evidence for a mechanism, not a test. Also recorded: per-layer depth profiles, and
the transposed-loop FFNs after training (§6 predicts the rotation part matters).

**Output:** `results/analysis/ffn-symmetry.json` and a table in `results/reports/`.

## 6. Predictions (from the first-order argument, recorded before any run)

With M the change one pass makes, M = S + A (symmetric + antisymmetric):
plain loop ≈ I + 2M, transposed ≈ I + 2S (A cancels), negated ≈ I + 2A (S cancels). The argument
is first order and ignores that GELU gates differ between passes; attention does not follow it
cleanly. It also ignores the LayerNorm gain (§5): S and A are the symmetric/antisymmetric parts of
J in the u basis, not of the gain-scaled J·diag(γ). Predictions:

1. `t` is worse than the plain loop on `val_ppl`.
2. `n` is better than `t`.
3. The FFN α values in `a` drift below 1, possibly negative. Attention α values need not follow.
4. The LM FFN rotation share is well above 0 (and above DeiT-small's, §5).

If 1–3 fail, the first-order argument is wrong for trained LMs, which is itself worth reporting.

## 7. Implementation

**Invariant:** with the new field at its default, every existing arm runs exactly the same
operations as today. Old checkpoints and resume files still load (new config keys only have
defaults; `run_one`'s resume check iterates the checkpoint's own keys).

- **`ModelConfig`**: `loop_pass2: str | None = None`, one of `"transpose"`, `"negate"`, `"scaled"`.
  Valid only with `n_unique_blocks` set and neither intra-layer flag on. `arm` returns
  `A1u<k>t|n|a`, and `arm_config` parses the same tags.
- **`sharing.py`**: `MLP.forward(x, transposed=False)` and `CausalSelfAttention.forward(x,
  transposed=False)`. The transposed path uses `F.linear(·, W.t())` on the existing
  `nn.Linear` weights (no copies, no new parameters).
- **`gpt.py`**: `Block.forward(x, transposed=False)`, with the pass-2 residual multiplier set by
  `loop_pass2` (+1, −1 or the block's `alpha_attn` / `alpha_mlp`, which exist only for
  `"scaled"`). `GPT.forward` passes `transposed = loop_pass2 is not None and (i // k) % 2 == 1`,
  and `GPT.loop_scales()` returns the α values (empty for every other arm).
- **`train.py`**: when `loop_scales()` is non-empty, its values are added to each logged metrics
  record. **`runqueue.py`**: the final α values go into the run's result row.
- **`scripts/gen_ladder_configs.py --transposed`**: writes the three seed-1337 configs and
  `configs/runs/queue-transposed.txt`. Follow-up seeds are added by a later commit if §3's rule fires.
- **`scripts/build_reports.py`**: three comparisons per variant: vs `L8-A1u4` (matched, H-T), vs
  `L4-A0` (iso-storage), vs `L8-A2` (matched).
- **Probes**: after training, `capability_eval.py` over the new checkpoints (same flags as the
  2026-09-15 pilot pass).
- **`src/hallm/symmetry.py`**: pure `rotation_share(w_up, w_down, u)` (exact GELU derivative computed inside) plus helpers that
  collect FFN inputs with forward hooks. **`scripts/ffn_symmetry.py`**: runs it over hallm
  checkpoints and DeiT-small. `transformers` and the image fetch go in a separate uv dependency
  group (`analysis`), so the training environment is unchanged.

## 8. Tests (written first) and workflow

- Config validation and tag round-trip (`A1u4t` ↔ flags).
- `t` and `n` have exactly `L8-A1u4`'s parameter count; `a` has 8 more.
- Gradients reach every stored matrix from both passes.
- At init, `a` gives the same output as `t`, and `n`'s pass-2 updates are `t`'s negated.
- Regression: `loop_pass2=None` gives bit-identical logits to the current code for A0, A2, A2attn
  and A1u4 on the `smoke` shape.
- Checkpoint round-trip and loading of pre-amendment checkpoints.
- `rotation_share`: 0 for a W+Wᵀ layer, ≈0.5 for random matrices, 1 for an antisymmetric J; the
  hook-based collector matches a direct computation on the `smoke` shape.

**Workflow.** Implementation happens on a git worktree branch, because every run's manifest
records `main`'s commit at launch and `L10-A2attn-s1339` has not started yet. Merge after the
A2attn seeds finish (~2026-09-15 19:50), then queue `queue-transposed.txt` (~6 GPU-h, overnight).
The symmetry measurement needs no training and runs whenever the branch is merged.

## 9. Risks

- **R1: the transposed pass destabilises training** (e.g. loss spikes at pass 2). The run is
  still reported. If all three variants diverge, the likely cause is scale: record it and consider
  an FFN-only transposition before concluding anything.
- **R2: novelty.** The literature check was ~70% confident (API coverage was limited). Before
  writing, repeat the search on Semantic Scholar/OpenReview and check the two unread items
  (§"Not checked" of the literature note).
- **R3: the symmetry comparison is uncontrolled across models.** Report it as mechanism evidence
  only. A controlled version would train a small ViT with our recipe, which is out of scope unless
  time allows.
- **R4: GPU contention before the gate.** Worst case is ~6 h (seed 1337) + ~12 h (follow-up seeds),
  plus the FineWeb bridge (~5 h). All of it fits before 2026-10-05 on the 4070.
