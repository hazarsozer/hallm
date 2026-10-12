# Controlled ViT Study — Is the Sharing Result Domain-Specific?

> **Status: executed 2026-09-16/17; recipe superseded by `2026-09-18-vit-pass-matched-design.md`.**
> The first pass failed its depth gate because of data reuse, not the augmentation floor, and its
> symmetry result was withdrawn. Outcome: `docs/analysis/2026-09-17-controlled-vit-outcome.md`.
>
> **Originally: approved in design 2026-09-16 (Hazar + Claude brainstorm).**
> Follows `2026-09-15-transposed-loop-design.md` §5 and `docs/analysis/2026-09-16-transposed-loop-outcome.md`,
> which closed the LM side: W+Wᵀ and all three transposed variants lose to plain looping, and the
> LM-vs-DeiT symmetry gap is real but **uncontrolled** (different recipe, scale, data and task), so it
> cannot carry a causal claim. This spec removes that confound by training the same arms, with our
> recipe, on images. Written before any vision run.
>
> **Decided with Hazar:** full arms in vision rather than a symmetry-only measurement (§2) ·
> ImageNet-100 at 112px (§4) · five arms at one seed (§2, §3).
>
> **Decided by Claude, open to override:** the reuse boundary and one-flag change to the LM model
> code (§6) · the uint8 memmap data format (§4) · the augmentation floor (§4) · run-ID tags (§2) ·
> artifact shapes (§7).

---

## 1. Why

Two negative results stand: at fixed storage, W+Wᵀ loses to looping in language (−5.25% at L8,
−6.10% at L16, 3 seeds), and every transposed-loop variant loses to the plain loop (+3.22% to
+4.50%, 1 seed each). HaLViT (CVPR 2024 W) reports that the same family of intra-layer transpose
sharing *works* in vision transformers. Either vision is genuinely different, or HaLViT's gain comes
from its recipe, scale or augmentation and our negative result is general. Today we cannot tell
these apart, because the only vision number we have is an off-the-shelf DeiT-small measured for FFN
Jacobian symmetry — a model trained by other people, on other data, at other scale, with heavy
augmentation.

The study answers one question: **under our recipe, on our arms, does sharing behave differently in
vision than in language?** Both answers are publishable, and the design must be equally happy with
either (§8).

## 2. Arms

Shape fixed across every arm (matched-budget control, as on the LM side): `n_embd` 512, `n_head` 8,
`ffn_mult` 4, patch 16 at 112×112 → 49 patch tokens + 1 class token = 50 positions.

| Run ID | Layers | Sharing | Role |
|---|---|---|---|
| `V4-A0-s1337` | 4 | none | storage floor: the unshared model storing what the looped arm stores |
| `V8-A0-s1337` | 8 | none | ceiling: what the storage buys when nothing is shared |
| `V8-A2-s1337` | 8 | intra-layer W+Wᵀ (FFN + attention) | the HaLViT-style arm |
| `V8-A1u4-s1337` | 8 | 4 unique blocks, cycled ×2 | the looping arm |
| `V8-A1u4t-s1337` | 8 | 4 unique blocks, pass 2 transposed | the transposed arm |

`V` distinguishes vision runs from the `L` ladder everywhere — run IDs, configs, result files and
report tables. Arm tags (`A0`, `A2`, `A1u4`, `A1u4t`) keep their LM meaning because they are the
same `ModelConfig` flags over the same `sharing.py` code (§6).

**Comparisons, mirroring the LM fixed-storage pairs:**

| Pair | Kind | LM counterpart |
|---|---|---|
| `V8-A1u4` vs `V4-A0` | iso-storage (same stored weights, more compute) | `L8-A1u4` vs `L4-A0`, −3.21% |
| `V8-A2` vs `V4-A0` | iso-storage | `L8-A2` vs `L4-A0` |
| `V8-A1u4` vs `V8-A2` | matched storage **and** compute | `L8-A1u4` vs `L8-A2`, −5.25% |
| `V8-A1u4t` vs `V8-A1u4` | matched storage and compute | `L8-A1u4t` vs `L8-A1u4`, +3.39% |
| any vs `V8-A0` | ceiling reference, not a claim | — |

Storage is counted over transformer blocks only. Patch embedding, positional embedding, class token
and classifier head are the vision analogue of the LM's embedding floor: identical across arms at a
given depth, and excluded from the iso-storage accounting. The prepare step and the model must
report both numbers so the pairing is checkable rather than asserted.

## 3. Rules (written before any run)

**One seed (1337) for all five arms. No verdict is claimed from this pass.** The LM rules require
the sign to hold in every paired seed, and with one seed that rule cannot be applied. This pass
produces point differences, reported as such, in a table that says "1 seed, descriptive" in its
header.

**Follow-up trigger, pre-registered:** the decisive comparison is `V8-A2` vs `V4-A0` (does the
HaLViT-style arm pay at fixed storage in vision?). Seeds 1338 and 1339 are run for `V8-A2` and
`V4-A0` iff their top-1 gap at seed 1337 is **≥ 1.0 point in either direction**. A gap smaller than
that is reported as "no difference detected at one seed" and gets no further GPU time. If the
trigger fires, `V8-A1u4` also gets the two seeds, so the three-way comparison stays paired.

**Primary metric:** top-1 accuracy on the held-out validation split. **Secondary:** validation
cross-entropy. Both are reported for every arm; the primary decides the trigger above.

**Sanity gates, checked before any comparison is believed:**
- `V8-A0` beats `V4-A0` on top-1. If depth does not pay at all, the setup is too weak to say
  anything about sharing, and the study stops and is reported as inconclusive.
- Every arm reaches a top-1 clearly above the 1% chance floor.
- `V8-A2`'s FFN Jacobian rotation share measures 0 to float precision (the §5 sanity check that
  already holds for `L8-A2`).

## 4. Data and recipe

**Corpus.** `clane9/imagenet-100` from the Hugging Face Hub: 100 ImageNet-1k classes, shorter side
already 160px, ungated parquet, the standard ImageNet-100 split from the CMC paper
(arXiv:1906.05849). ~130k train, ~5k validation images.

**Prepare once** (`scripts/prepare_imagenet100.py`): resize shorter side to 128, center crop to
128×128, write `data/in100/train.bin` and `val.bin` as uint8 memmaps of shape `[N, 3, 128, 128]`,
plus `labels.npy` per split and a `SOURCE.json` recording the dataset revision SHA, the class list,
counts and the preprocessing parameters. Train is 6.4 GB, validation 245 MB. This mirrors the
existing `data/*.bin` token files: a batch is a memmap slice, so there is no DataLoader jitter and a
run is reproducible from the recorded SHA.

**Augmentation floor, identical across arms:** random 112×112 crop from the stored 128×128 plus
horizontal flip, both on GPU, and label smoothing 0.1. **No mixup, cutmix, RandAugment, erasing or
EMA.** This is deliberate: heavy augmentation is standard for ViTs trained from scratch and is part
of HaLViT's recipe, so importing it would reintroduce the confound this study exists to remove.
The cost is that absolute accuracies land below published ImageNet-100 numbers. Only the arm gaps
are claimed, and §9 records this as the study's main limitation.

**Optimization, identical across arms:** AdamW, cosine schedule with 200-step warmup, bf16
autocast, batch 256, 50,000 steps (~98 epochs). Same optimizer, schedule shape and step budget as
the LM ladder, which is what "our recipe" means here.

**Timing probe first.** Before the queue is written, one arm runs 200 steps to measure real
steps/min. Sequences are 50 positions against the LM's 1024, so the runs should be much cheaper,
but the queue is sized from the measured number, not from this estimate.

**Measured 2026-09-16 (V8-A0-s1337, RTX 4070 SUPER, bf16, batch 256).** The probe was run in three
legs so that startup could be cancelled out rather than estimated: 200 steps in 22 s including the
one-time 6.2 GB manifest hash, then 1000 steps in 78 s, then 2000 steps in 155 s. The last two legs
have identical startup shape (the manifest already exists), so their difference is the true marginal
cost: **77 s per 1000 steps including the eval that falls in that window — 13.0 steps/s, 779
steps/min.** That gives **~64 min (1.07 h) per 50,000-step V8 run**, ~32 min for the 4-block V4-A0,
and **~4.8 h for the whole five-arm queue**. Well under the §9 R4 re-plan threshold of 8 h for a
single run, so the queue was launched rather than re-planned. For the record, the a-priori estimate
from parameter counts (6·N per token, the LM's 50k-vocab head being about half its FLOPs, and near
identical tokens per step) predicted ~1.0 h per V8 run — 7% low.

## 5. Symmetry measurement (the controlled comparison)

After training, `scripts/ffn_symmetry.py` runs over the five ViT checkpoints with a collector that
feeds them validation images, reporting the same rotation share
`‖½(J − Jᵀ)‖²_F / ‖J‖²_F` in both the `u` and the pre-gain `x̂` basis, per layer and per model
(the basis caveat from 2026-09-15 §5 applies unchanged).

This is the controlled version of the DeiT comparison. Our ViT and our LM now share recipe,
optimizer, schedule, scale and parameter count; only the domain differs. The existing rows
(`L*-A0`, DeiT-small, `L8-A2`, random) stay in the same table, with DeiT-small explicitly labelled
as the uncontrolled reference it is.

## 6. Predictions (recorded before any run)

1. `V8-A0` beats `V4-A0` on top-1 (the depth sanity gate, §3).
2. `V8-A2` vs `V4-A0` does **not** reproduce the LM's clear iso-storage loss. If vision is friendlier
   to symmetric maps, W+Wᵀ should come at least level here.
3. Our ViT's FFN rotation share is below our LMs' (0.449–0.477) and near DeiT-small's 0.395,
   including the low early layers (DeiT's layer 1 is 0.281).
4. `V8-A1u4t` vs `V8-A1u4` is closer to level than the LM's +3.39%.

If 2 and 3 both hold, the domain story is supported and the symmetry gap survives as a controlled
result. If 2 and 3 both fail, the LM negatives generalize: sharing does not pay in either domain
under a common recipe, and HaLViT's gain is attributable to its recipe, scale or augmentation
rather than to vision. **Both outcomes are reportable and the study is designed to be indifferent
between them.** The awkward case is a split (2 holds, 3 fails, or the reverse): that means accuracy
and symmetry disagree, the first-order symmetry argument is wrong for vision too, and the write-up
says so rather than picking the half that flatters the thesis.

## 7. Implementation

**Reuse boundary** — the arms must be literally the same code, or the comparison is not controlled.

| Status | Component |
|---|---|
| Reused unchanged | `model/sharing.py` (MLP, attention, W+Wᵀ ties, transposed loop), `build_blocks`, the optimizer and cosine schedule, `manifest.py`, `results.py`, `symmetry.py` |
| Changed (one flag) | `ModelConfig.causal: bool = True`, threaded to the hardcoded `is_causal=True` at `model/sharing.py:128` |
| New | `model/vit.py`, `data/imagenet.py`, `vision_train.py`, `scripts/prepare_imagenet100.py`, vision configs and queue |

`causal` defaults to `True`, so every existing LM run stays bit-identical; a test asserts that an
unshared LM built from a saved config is unchanged by the flag's introduction.

**`model/vit.py`:** patch embedding (conv 16×16, stride 16), learned positional embedding, a class
token, blocks from `build_blocks` with `causal=False`, final LayerNorm, linear head to 100 classes.
Mean pooling is not used; the class token keeps the block stack identical to the LM's.

**`vision_train.py`:** its own loop, because the LM trainer is token-batch specific and is mid-campaign.
It imports the shared optimizer and schedule helpers, writes the same `metrics.jsonl`, `manifest.json`
and `resume.pt` files, and supports the same mid-run resume, so an interrupted vision run recovers
exactly as an LM run does.

**`data/imagenet.py`:** memmap open, batch sampling, GPU-side crop and flip, and the label array.

## 8. Artifacts

- `results/runs/V*.json` — the existing schema plus `top1` and `n_classes`.
- `results/manifests/V*.json` — copied on landing, as on the LM side.
- `results/reports/vision.md` — the five arms and the §2 pairs, header stating "1 seed, descriptive".
- `results/analysis/ffn-symmetry.json` — extended with the ViT rows; DeiT labelled uncontrolled.
- One commit per landed run, with the result, manifest and rebuilt reports together.

## 9. Risks

- **R1 — under-trained ViTs.** ViTs from scratch on 130k images with a minimal augmentation floor may
  land at low absolute accuracy. Mitigated by the §3 depth gate: if `V8-A0` does not beat `V4-A0`,
  the study is inconclusive and says so instead of comparing noise.
- **R2 — one seed.** Every number in this pass is a point estimate; the pre-registered trigger in §3
  is the only path to a verdict, and the report header must carry the caveat.
- **R3 — the augmentation choice.** Our floor is not HaLViT's recipe, so a null vision result does
  not disprove HaLViT on its own terms; it shows the gain does not survive a common recipe. Stated
  as a limitation, not hidden.
- **R4 — GPU contention before the 2026-10-05 gate.** Five runs at an unmeasured cost. The §4 timing
  probe runs first, and the queue is sized from it; the gate work has priority if they collide.
- **R5 — new trainer, new bugs.** A second training loop can silently differ from the LM one (loss
  reduction, schedule off-by-one, eval in train mode). Tests written first, plus the §3 sanity gates,
  plus a deterministic 50-step overfit-a-single-batch check.

## 10. Out of scope

No pretrained weights, no distillation tokens, no ImageNet-1k, no SOTA chasing, no vision probes
beyond top-1 and validation loss, no new sharing mechanisms. Five runs, one seed, one question.
