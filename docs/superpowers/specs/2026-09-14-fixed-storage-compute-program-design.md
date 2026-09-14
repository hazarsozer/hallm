# Research Program Design — Buying Compute With Shared Weights (Sep 2026 → January)

> **Status: approved in brainstorm 2026-09-14 (Hazar + Claude); spec under Hazar's review.**
> Supersedes the *question* of `2026-08-20-research-program-design.md` (and its 2026-08-31
> amendment). That document's standing methodology (§2) carries forward unchanged, and every result
> it produced (P1, P3, T-001–T-005) stands as input.
>
> **Decided with Hazar:** the reframed question (§1) · two measurement tracks (§4, §5) · top rung
> ~124M on FineWeb-Edu with 350M as a conditional stretch (§7) · pilot first, then scale (§3) ·
> repeating the grid on a second corpus is "if time" (§9) · cloud GPU allowed as overflow.
>
> **Decided by Claude, flagged for review:** the looped-arm definition and run-ID tags (§2) · the
> pilot run list (§4.1) · the Track 2 task set (§5) · the gate rule (§6) · the 124M arm shapes and
> recipe freeze (§7) · Track 2 assigned to Alper (§5).

---

## 1. The question

> **At a fixed number of stored weights, does spending extra compute through sharing beat the
> unshared model — on perplexity and on reasoning — and which kind of sharing spends it better:
> W+Wᵀ (within a layer) or looping (across layers)?**

**Why the reframe.**

- **The ≤2% gate asks the wrong thing on its own.** It asks how much a shared model loses. The
  question a user of shared weights faces is different: given a storage budget, is a shared model
  that runs *more compute* on the same stored bytes better than the unshared model that fits the
  same bytes? (Hazar, 2026-09-14.) The 2026-08-31 iso-storage gate was a first step toward this; this
  program makes it the organising axis. The ≤2% gate stays as a reported secondary verdict.
- **Perplexity may be the wrong instrument for extra compute.** Saunshi et al. (ICLR 2025,
  arXiv 2502.17416) find that looped transformers — k layers reused m times, i.e. ALBERT-style
  cross-layer sharing — have *worse* perplexity than an unshared model of the same effective depth
  but nearly match it on reasoning tasks (addition, p-hop induction, synthetic math), and at equal
  perplexity do better downstream. Their result concerns the depth axis; whether W+Wᵀ-bought depth
  behaves the same is untested.
- **The ALBERT comparison so far was unfair.** Experiment 1's "W+Wᵀ dominates cross-layer sharing
  (0.28 vs 0.42 % PPL per % saved)" compares A2 at −50% storage with A1 at −87.5%. P1 showed the
  cost per % saved *rises* with the amount removed (0.247 → 0.271 → 0.289), so that ratio
  penalises the arm that compresses harder by construction; A1 is also single-seed. The claim is
  withdrawn until the matched test in §4.1 runs.

**What we already know on this axis (both measured, both lost):**

| stored non-emb | shared arm, more compute | unshared arm | result |
|---|---|---|---|
| 25.2M | A2 @ L16, 86.5 GFLOPs: 27.02 (3 seeds) | A0 @ L8, 56.4 GFLOPs: 26.01 (3 seeds) | shared +3.9% worse |
| 21.0M vs 22.0M | A2attn @ L8, 56.4 GFLOPs: 27.04 (3 seeds) | A0 @ L7, 52.7 GFLOPs: 26.41 (1 seed) | shared +2.4% worse |

**Scope reminder (from the 2026-08-20 spec §1):** extra depth grows the KV cache, so every
iso-storage comparison also reports weights + KV memory at ctx512×1 and ctx2048×8. The claim is about
stored weight size; serving memory is reported, not optimised.

## 2. Arms

Storage is counted in non-embedding parameters, in units of d² (per layer: attention 4d², FFN 8d²).
Embeddings are identical within a rung.

| kind | arm tag | what is stored | per-layer / total storage |
|---|---|---|---|
| unshared | `A0` | L distinct blocks | 12d² × L |
| W+Wᵀ, full | `A2` | L blocks, each FFN and attention halved | 6d² × L |
| W+Wᵀ, attention only | `A2attn` | L blocks, attention halved | 10d² × L |
| looped | `A1u<k>` | **k distinct blocks, cycled** to depth L (layer i uses block i mod k) | 12d² × k |

- `A1u<k>` generalises A1 (which is `A1u1`). The cycle order (0,1,…,k−1, 0,1,…) matches Saunshi et
  al.'s "k-layer block looped L/k times" and the existing forward (`gpt.py:74`). LayerNorms live in
  the block and loop with it, as in A1.
- Code change: a `n_unique_blocks` field on `ModelConfig` (`None` = today's behaviour), used by
  `build_blocks` (`sharing.py:131`); `parse_run_id` maps `A1u8` → A1 flags + `n_unique_blocks=8`.
  Tests: storage invariant `12d²·k` independent of L; `A1u<L>` ≡ `A0`.
- Matched pairs by construction (same storage **and** same compute):
  - `A1u4 @ L8` vs `A2 @ L8`: both 48d² = 12.58M, 56.4 GFLOPs.
  - `A1u8 @ L16` vs `A2 @ L16`: both 96d² = 25.17M, 86.5 GFLOPs.

## 3. Shape of the program

```
Phase 0  now → Sep 20     housekeeping, code, email, issues          (no GPU)
Phase 1  Sep 15 → Oct 5   PILOT at ~25M on WikiText-103              (~25 GPU-h, 4070)
                          Track 1 perplexity grid + probes · Track 2 build + pilot (Alper)
                          FineWeb-Edu pipeline + bridge run
         Oct 5            GATE: which variant of each kind goes to 124M
Phase 2  Oct 6 → Nov 30   SCALE: 124M grid on FineWeb-Edu            (~300 GPU-h, 4070 + cloud)
                          full Track 2 grid · 350M only if 124M shows a signal
Phase 3  Dec → mid-Jan    WRITE (final report, advisor review, presentation)
```

Working assumption: final report due mid-January; experiments freeze 2026-11-30. Revisit when
Design II deliverables appear on Ninova.

## 4. Track 1 — language models

### 4.1 Pilot perplexity grid (WikiText-103, standing methodology)

| run | stored | GFLOPs | seeds | cost | answers |
|---|---|---|---|---|---|
| `L8-A1u4` | 48d² | 56.4 | 1337–1339 | ~6.1 h | looping vs W+Wᵀ, matched (vs `L8-A2`); vs unshared at same storage (`L4-A0`) |
| `L16-A1u8` | 96d² | 86.5 | 1337–1339 | ~8.6 h | looping vs W+Wᵀ, matched (vs `L16-A2`); vs unshared (`L8-A0`) |
| `L9-A2attn` | 90d² (−6%) | ~60.2 | 1337 | ~2.1 h | A2attn buys ~20% extra depth at iso-storage: does it beat `L8-A0`? |
| `L10-A2attn` | 100d² (+4%) | ~63.9 | 1337 | ~2.2 h | bracket from above (T-003, narrowed to the depth axis) |

Every comparison row already exists on the other side (`L4-A0`, `L8-A0`, `L8-A2`, `L16-A2`, `L16-A0`
all at 3 seeds). Extra A2attn seeds only if a seed-1337 point lands within 1% of `L8-A0`.

Structural note to record: A2attn saves only 16.7% per layer, so at iso-storage it can buy at most
~20% extra depth; full A2 buys 2×; looping buys any multiple. The kinds differ in *how much* compute
they can buy, not only in how well they spend it.

### 4.2 Probes (inference-only, existing and new checkpoints)

Per-position loss, loss by token-frequency decile, LAMBADA, BLiMP (all in `capeval.py` /
`scripts/capability_eval.py`), plus two new in-context probes: induction (repeated random-token
sequences, accuracy on the second occurrence) and associative recall (key–value pairs in context,
query a key). Run on every pilot arm and its comparators. The 4 checkpoints without weights
(`L7-A0-s1337`, `L8-A0-s1339`, `L8-A2-s1339`, `L8-A2attn-s1339`) are skipped; those comparisons run
at the seeds that exist.

### 4.3 Bridge run (FineWeb-Edu)

`L8-A0` and `L16-A2` (the Experiment 2 iso-storage pair), seed 1337, trained on FineWeb-Edu at the
WikiText token budget (614M). Connects the WikiText numbers to the 124M rung's corpus. ~5 GPU-h.
Needs the FineWeb-Edu pipeline: GPT-2 BPE, `uint16` token files like `data/*.bin`, a held-out
split, ≥2.5B training tokens tokenised once (~5 GB).

### 4.4 Pre-registered claims and decision rules (pilot)

- **H-C (compute pays at fixed storage):** for a shared arm X, `PPL(X) < PPL(unshared at the same
  storage)`. Supported for X iff the mean is lower **and** the sign holds in every paired seed.
- **H-L (which kind spends compute better):** at matched storage and compute (§2 pairs), report the
  paired PPL difference, mean ± SE over seeds. "X beats Y" iff the sign holds in all 3 seeds and
  |mean| > 2 SE; otherwise "no difference detected".
- **H-R (reasoning, Track 1 half):** for each probe, the same two rules applied to the probe metric
  (accuracy or loss) instead of PPL.

## 5. Track 2 — synthetic reasoning (Alper, 3070 Ti)

Small models trained from scratch on generated tasks, following Saunshi et al.'s setup.

- **Tasks (3), each with a difficulty knob:** p-hop induction (knob: hops) · multi-digit addition
  (knob: digits) · variable-binding chains, an i-GSM-lite (`a=3; b=a+2; c=b·2; c=?`, knob: chain
  length).
- **Arms at equal stored weights:** unshared k layers · looped k blocks × m (m = 2, 4) · W+Wᵀ at 2k
  layers · unshared k·m layers as the compute reference. Exact d, k and training steps are set in
  the Track 2 plan after a sizing pilot.
- **Metric:** exact-match accuracy on held-out problems, per difficulty level, 3 seeds.
- **Validity check before any comparison is read:** reproduce Saunshi et al.'s qualitative result
  (looped k×m ≈ unshared k·m on p-hop induction). If it does not reproduce, the pipeline is fixed
  first; W+Wᵀ results are not interpreted until it does.
- **H-R (Track 2 half):** the H-L / H-C rules of §4.4 applied to accuracy at each difficulty level.

Filed as a GitHub task (T-006, via `scripts/tasks.py` conventions): build + sizing pilot by Oct 5,
full grid in Phase 2.

## 6. The gate (Oct 5) — pre-registered

The gate decides **which variant of each kind** runs at 124M, not whether 124M runs. The 124M grid
is always three arms: unshared, the best W+Wᵀ variant, the best looped variant.

- Within each kind, a variant **shows a signal** if it satisfies H-C or H-R against the unshared
  model at the same storage (any Track 1 metric or any Track 2 task/difficulty).
- Pick the variant with a signal; if several, the one with the most measures showing a signal; if
  none, the one with the lowest iso-storage PPL gap.
- Default if the pilot is silent: full A2 at 2× depth (the thesis's own mechanism, and the only W+Wᵀ
  variant that buys 2× compute) and looped ×2.
- Hazar sets the cloud budget cap at the gate, after Töreyin answers on cluster access.

## 7. Phase 2 — the 124M rung (FineWeb-Edu)

- **Shapes:** `s124` (d=768, L=12, ctx 1024, V=50257; 144d² = 84.9M non-emb). Unshared `L12-A0`;
  W+Wᵀ `L24-A2` (144d²) or the gate's A2attn variant; looped `L24-A1u12` (144d²). Shared arms cost
  ~2× the unshared arm's compute.
- **Tokens:** 2.5B per arm (≈20 per parameter), identical data order across arms.
- **Recipe:** the nanoGPT GPT-2-small recipe, fixed in the Phase 2 plan before the first run and
  identical across arms. Not tuned per arm.
- **Seeds:** 1337–1339. Seed 1337 first for all three arms, so a partial grid is still a complete
  comparison.
- **Cost:** ~20 h (A0) + ~40 h + ~40 h per seed on the 4070 ≈ 100 h; 3 seeds ≈ 300 h (~12.5 days
  continuous). Cloud runs whole seeds in parallel if the calendar needs it. Colab sessions time out,
  so any cloud run uses `resume.pt`; an hourly rented GPU is preferred for multi-day runs.
- **Evals:** held-out FineWeb-Edu PPL, WikiText-103 test PPL (cross-corpus), the §4.2 probes, and
  standard benchmarks HellaSwag, ARC-Easy, PIQA. At 124M these sit a few points above chance, so
  every benchmark reports a 95% CI and the §4.4 rules; a null here is read as "below this scale's
  resolution", not as evidence.
- **350M stretch:** only if some shared arm shows a signal at 124M **and** the budget or cluster
  covers it: ~34 4070-days per seed (A0 ~7 d, each 2×-compute shared arm ~14 d). Otherwise the thesis reports 124M as the top rung.

## 8. Phase 3 — writing

Experiments freeze 2026-11-30. December: final report in the faculty template, figures from
`results/reports/`. Töreyin's approval is required before the final exam and presentation
(principles doc), so a full draft goes to him by ~Dec 20.

## 9. If time (not planned)

Grid on a second corpus (full P6) · P3 width axis · P5 leftovers (2× tokens, dropout 0.1,
`sharing_warmup_steps` > 0) · alternative transpose pairing (Q↔K, V↔Out) · A2attn width
re-investment (d576/L8 from T-003) · looping × W+Wᵀ combined at iso-storage.

## 10. Phase 0 — housekeeping (no GPU)

1. **RESULTS.md / README:** add T-001, T-002, the third L8 seed, the A2attn gate verdict (+2.41%),
   T-004/T-005; withdraw the "W+Wᵀ dominates cross-layer" claim (§1); state the reframed question.
2. **Reports:** a report for 4-part run IDs (probes, P4 frontier); `mechanism.md` reported per rung
   instead of pooled (the pooled table names A2ffn, contradicting the pre-registered L8 reading).
3. **Code:** `n_unique_blocks` + `A1u<k>` run IDs + tests (§2); `SHAPES` entries for A2attn at L9/L10.
4. **Advisor:** progress email to Töreyin, CC Alper and Ahmet Nuri Yılmaz — results so far, the
   reframed question, the pilot, a request for SP4CING/UHEM access, and his OK before 124M/cloud
   spend. Claude drafts; Hazar sends.
5. **Alper:** this spec pushed to `main` (shows up in `tasks.py check` as an update); T-006 filed.
6. **Old spec:** status line on `2026-08-20-research-program-design.md` pointing here.

## 11. Risks

- **R9 — Track 2 does not reproduce the published looped result.** Handled by §5's validity check;
  costs calendar, not conclusions.
- **R10 — 124M is slower than estimated** (the 34.7k tok/s figure is from P7's measurement). Seed
  1337 first keeps a complete comparison; cloud absorbs overflow.
- **R11 — Töreyin disagrees with the reframe.** The pilot is useful under either framing (it also
  fixes the ALBERT comparison); 124M spend waits for his reply.
- **R12 — Hazar's term load** (registration, Turkcell part-time). Runs are queued and low-touch;
  Phase 2 is mostly waiting.
- **R1 (carried):** unchanged; the P1 decomposition already localised it.
