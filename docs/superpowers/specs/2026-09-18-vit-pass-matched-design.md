# Controlled ViT Study — Pass-Matched Re-Run

> **CORRECTION 2026-09-27 — the LM pass count below is wrong by 2×.** §1 computes the LM ladder's
> tokens per step as 12 × 2 × **1024**; the ladder context is **512** (`SHAPES`, the run configs and the
> frozen manifests agree). The LM ladder sees 12,288 tokens/step, 614.4M tokens, **5.15 passes**. So the
> first ViT pass (101.03) was ~19.6× the LMs, not 10×, and the ImageNet-1k design (9.99) is ~1.94× the
> LMs, not matched. True matching is ~25,800 steps on ImageNet-1k. The text is left as written, as the
> record of what was designed; see `docs/DECISIONS.md` (2026-09-27).

> **Status: approved in design 2026-09-18 (Hazar + Claude brainstorm); execution handed to Claude the
> same evening.** Supersedes the recipe of `2026-09-16-controlled-vit-design.md`; its arms, sharing
> code, measurement and predictions carry over unchanged. Written before any run of this pass,
> including the §3 probe.
>
> **Decided with Hazar:** more data rather than more augmentation, a shorter budget or a smaller model
> (§2) · if the depth gate fails a second time, the study stops and reports depth-doesn't-pay rather
> than escalating to augmentation (§6).
>
> **Decided by Claude, open to override:** the ImageNet-1k mirror and stored format (§4) · the §3
> probe as a precondition on the corpus swap · staged execution (§5) · final-step primary with a
> best-checkpoint overfitting flag (§7) · the 10,000-image periodic eval subsample (§7) · the `-in1k`
> run-ID suffix (§5).

---

## 1. Why

`docs/analysis/2026-09-17-controlled-vit-outcome.md` closed the first pass: the pre-registered depth
gate failed (`V8-A0` 46.96% against `V4-A0` 48.14%), so by spec §3 the study is inconclusive on
sharing. The diagnosis was overfitting rather than under-training, and the outcome doc named the
augmentation floor as the cause.

Re-reading the two recipes side by side says otherwise. "Our recipe" was matched between the LM ladder
and the ViT arms on optimizer, schedule shape, warmup, batch semantics, step count and model shape —
but never on how many times a model sees its corpus:

| | per step | × 50,000 steps | corpus | **passes** |
|---|---|---|---|---|
| LM ladder | 24,576 tokens (12 × 2 × 1024) | 1.229B tokens | WikiText-103, 119.2M tokens | **10.31** |
| ViT arms, first pass | 256 images | 12.8M samples | ImageNet-100, 126,689 images | **101.03** |

A 10× asymmetry, on the axis that decides whether extra capacity generalizes. The first pass's own
trajectory shows the consequence:

| step | ~passes | `V4-A0` top-1 | `V8-A0` top-1 |
|---|---|---|---|
| 10,000 | 20 | 43.54% | **44.60%** |
| 20,000 | 40 | 45.88% | **46.12%** |
| 30,000 | 61 | **47.42%** | 46.74% |
| 49,000 | 99 | **48.16%** | 47.38% |

**Depth pays in vision for roughly the first 40 passes and stops paying after.** The gate did not fail
because an 8-layer ViT cannot be trained under a minimal augmentation floor; it failed because the
budget ran to 101 passes over a corpus small enough that the extra capacity could only memorize. The
variable to fix is the one that was silently unmatched.

This matters beyond the gate. The study exists to hold everything constant but the domain. A 10×
difference in data reuse between the language arms and the vision arms is exactly the kind of
uncontrolled difference the study was built to remove, and it was present in the first pass.

## 2. The change — exactly one variable

| | first pass | this pass |
|---|---|---|
| corpus | ImageNet-100, 126,689 train | **ImageNet-1k, 1,281,167 train** |
| passes at 50,000 steps × 256 | 101.03 | **9.99** (LM ladder: 10.31) |
| classes | 100 | 1000 |
| steps, batch, LR, min_lr, warmup, weight decay, grad clip | | unchanged |
| augmentation floor: random 112 crop from 128, hflip, label smoothing 0.1 | | unchanged |
| `n_embd` 512, `n_head` 8, `ffn_mult` 4, patch 16, 50 positions | | unchanged |
| sharing code, arm definitions, `causal=False` flag | | unchanged |
| sequence length, and therefore GPU cost per run | | unchanged |

**No augmentation is added.** The confound the first spec's §4 was written to exclude stays excluded;
this pass fixes a mismatch rather than importing a recipe.

The classifier head grows from 512×100 to 512×1000 (51k → 0.51M parameters). It sits in the excluded
bucket under the first spec's §2 — the vision analogue of the LM's embedding floor, identical across
arms at a given depth — so the iso-storage accounting between arms is untouched. The chance floor
moves from 1% to 0.1%, and the §6 sanity gate is restated against the new floor.

Absolute top-1 is not comparable to the first pass (1000-way against 100-way). Only within-pass arm
gaps are claimed, as before.

## 3. Diagnostic probe — a precondition, not a formality

§1's account is inferred from a mid-training crossover, not tested. It is testable in ten minutes on
data already on disk, and until it passes, no download or prepare pass is spent.

**Probe:** re-run `V4-A0` and `V8-A0` on the **existing** ImageNet-100 memmaps with `max_steps: 5000`
and the cosine schedule resized to 5,000 steps — 10.1 passes, the same data-reuse factor this pass
targets, on the same corpus as the failed run. Everything else is the first pass's config. Warmup
stays at 200 absolute steps, as in the ladder, which makes it 4% of the schedule instead of 0.4%;
recorded here rather than silently adjusted. `eval_interval` drops to 500 for resolution.

At the measured 779 steps/min this is ~6.4 min for `V8-A0` and ~3.2 min for `V4-A0`.

**Probe gate, pre-registered:**
- **Pass:** `V8-A0` top-1 > `V4-A0` top-1 at step 5,000. The §1 account is confirmed and §4 proceeds.
- **Fail:** the account in §1 is wrong. Stop, report it, and re-plan before spending the corpus swap.
- **Either way:** both arms must land clearly above the 1% ImageNet-100 chance floor. If they land near
  chance, 10 passes is too few to train these models at all, and the design needs a pass count between
  10 and 101 rather than a corpus swap. This is R1 of §12 and the probe is its cheapest test.

The probe's runs are diagnostics, not arms. They are tagged `-p5k` and are excluded from every
comparison table; their purpose is to decide whether §4 happens.

## 4. Data

**Source.** `benjamin-paine/imagenet-1k-128x128` — ILSVRC-2012, ungated, already stored at 128×128,
1,281,167 train / 50,000 validation / 100,000 test, 6.14 GB of parquet. Standard class ordering
(label 0 = tench), so the class index is the usual ImageNet one. The test split has no public labels
and is not used.

The mirror's stored size matches the format `scripts/prepare_imagenet100.py` already targets, so the
prepare step is a decode-and-pack rather than a resize. `scripts/prepare_imagenet1k.py` nevertheless
handles both cases — if a decoded image is not exactly 128×128 it resizes the shorter side to 128 and
center-crops, the same operation the ImageNet-100 prepare applied — and records which path was taken,
with counts, in `SOURCE.json`. Whichever it is, it is written down rather than assumed.

**Output.** `data/in1k/train.bin` and `val.bin`, uint8 memmaps of shape `[N, 3, 128, 128]`, plus
`train_labels.npy` / `val_labels.npy` and `SOURCE.json` recording the revision SHA, split counts, the
class list, the preprocessing path taken and the resulting file hashes. Train is **62.97 GB**,
validation **2.46 GB**; 860 GB free at design time.

**Prepare cost is unmeasured.** The script runs first on a 10,000-image slice to measure decode
throughput and extrapolate, the same discipline the first pass applied to step timing, and the full
run only starts once that number exists. `--ipv4` carries over from `prepare_imagenet100.py`: this
host's IPv6 route to the HF CDN blackholes.

**Read throughput.** A 62.97 GB memmap under random access exceeds the 23 GB of available page cache,
so batches will come off NVMe rather than RAM. At 13.0 steps/s × 256 images × 49,152 B this is
~164 MB/s of 49 KB random reads — comfortably inside a Samsung 990 PRO's range, but measured with a
short read-only benchmark before the queue is launched, not assumed.

## 5. Arms and staged execution

The five arms are unchanged from the first spec's §2. Run IDs take an `-in1k` suffix, mirroring the
LM side's `-fw` for FineWeb-Edu: `V4-A0-s1337-in1k`, `V8-A0-s1337-in1k`, `V8-A2-s1337-in1k`,
`V8-A1u4-s1337-in1k`, `V8-A1u4t-s1337-in1k`. The first pass's 100-class results stay committed under
their existing IDs and are reported as the 101-pass contrast (§7).

| Stage | Work | Cost | Precondition |
|---|---|---|---|
| 0 | §3 probe on existing data | ~10 min GPU | — |
| 1 | prepare ImageNet-1k | 6.14 GB download, 62.97 GB disk, time measured in-stage | probe passed |
| 2 | `V4-A0-in1k`, `V8-A0-in1k` | ~1.6 h GPU | corpus prepared and spot-checked |
| 3 | `V8-A2-in1k`, `V8-A1u4-in1k`, `V8-A1u4t-in1k` | ~3.2 h GPU | **depth gate passed** |
| 4 | symmetry re-measurement over the five new checkpoints | minutes | stage 3 landed |

Stage 3 is gated on stage 2 rather than queued behind it. This is the first spec's §6 sanity gate
enforced by scheduling: if depth does not pay, the three sharing arms are never run, because nothing
they produce could be interpreted.

Each landed run is committed with its result, manifest and rebuilt report, as on the LM side.

## 6. Rules (written before any run)

**One seed (1337) for all five arms.** No verdict is claimed from this pass. Point differences are
reported as such, under a header stating "1 seed, descriptive", exactly as the first pass did.

**Follow-up trigger, carried over unchanged:** seeds 1338 and 1339 are run for `V8-A2` and `V4-A0` iff
their top-1 gap at seed 1337 is **≥ 1.0 point in either direction**; if it fires, `V8-A1u4` gets the
same two seeds so the three-way comparison stays paired. A smaller gap is reported as "no difference
detected at one seed" and gets no further GPU time.

**Sanity gates, checked before any comparison is believed:**
1. **Depth gate.** `V8-A0-in1k` beats `V4-A0-in1k` on final-step top-1.
2. Every arm reaches a top-1 clearly above the 0.1% chance floor.
3. `V8-A2-in1k`'s FFN Jacobian rotation share measures 0 to float precision, as `L8-A2` and the first
   pass's `V8-A2` both do.

**If the depth gate fails a second time — decided with Hazar, 2026-09-18 — the study stops.** It does
not escalate to augmentation, a third pass count, or a smaller model. The reported finding becomes:
under a minimal common recipe, at this scale, depth does not pay for ViTs, so the sharing question in
vision cannot be asked without importing HaLViT's own recipe — which is itself an answer about where
HaLViT's gain comes from. The symmetry result (§8) stands regardless, since it measures trained
weights rather than accuracy.

## 7. Metrics and reporting

**Primary:** final-step top-1 on the held-out validation split. **Secondary:** validation
cross-entropy. The primary decides the §6 trigger.

Final-step rather than best-checkpoint, because the LM ladder shows the two coincide there — across
`L8-A1u4`, `L16-A1u8`, `L9-A2attn` and `L8-A0-fw`, the best validation loss lands at step 49,000 and
the final step is +0.01% to +0.59% from it. Reporting best-checkpoint in vision while the ladder
reports final would introduce a second unmatched axis, which is the mistake this pass exists to fix.

**Overfitting flag, pre-registered.** Best-checkpoint top-1 is reported alongside final-step for every
arm. Any arm whose best/final gap exceeds **1.0 top-1 point** is labelled *still overfitting* in the
report. This closes open item 2 of the first pass's outcome without changing the LM protocol: the peak
is disclosed, never substituted for the reported number.

**Eval splits.** Periodic evaluation every 1,000 steps runs on a fixed, class-stratified 10,000-image
subsample of the 50,000-image validation split, drawn once with a recorded seed and stored in
`SOURCE.json`; the final reported number for every arm runs on all 50,000. Periodic eval on the full
split would cost ~10× the first pass's for no added resolution on the training curve.

**The 101-pass contrast.** `results/reports/vision.md` carries both passes: the ImageNet-100 arms at
101 passes and the ImageNet-1k arms at 10, each labelled with its corpus, class count and pass count.
Absolute accuracies across the two blocks are not comparable and the report says so. The pairing —
depth costing 1.18 points at 101 passes and its behaviour at 10 — is reported as a result about data
reuse, which is the one thing the failed first pass bought.

## 8. Symmetry re-measurement

`scripts/ffn_symmetry.py` runs unchanged over the five new checkpoints, feeding them ImageNet-1k
validation images, reporting the rotation share `‖½(J − Jᵀ)‖²_F / ‖J‖²_F` in both the `u` and the
pre-gain `x̂` basis, per layer and per model. Existing rows (`L*-A0`, `L8-A2`, DeiT-small labelled
uncontrolled, random) stay in the table; the first pass's ViT rows stay too, labelled with their
corpus.

This is a genuine replication rather than a restatement: if the 0.416–0.422 band reproduces at 10×
the data, 1000 classes and a different pass count, the domain gap is robust to the corpus as well as
controlled for recipe, scale and sharing code.

## 9. Predictions (recorded before any run of this pass, including the probe)

Predictions 1–4 carry over verbatim from `2026-09-16-controlled-vit-design.md` §6 and are restated
against the new corpus:

1. `V8-A0-in1k` beats `V4-A0-in1k` on top-1 (the depth gate, §6).
2. `V8-A2-in1k` vs `V4-A0-in1k` does **not** reproduce the LM's clear iso-storage loss.
3. Our ViT's FFN rotation share is below our LMs' (0.449–0.477) and near DeiT-small's 0.395,
   including the low early layers.
4. `V8-A1u4t-in1k` vs `V8-A1u4-in1k` is closer to level than the LM's +3.39%.

New to this pass:

5. **The pass count, not the augmentation floor, is what blocked depth.** `V8-A0` beats `V4-A0` in the
   §3 probe at 5,000 steps on ImageNet-100, and again on ImageNet-1k at 50,000 steps. If the probe
   passes and the ImageNet-1k gate then fails, prediction 5 is half-false and the write-up says the
   corpus swap introduced something the probe did not capture.
6. **The symmetry band reproduces.** The `-in1k` ViTs' rotation share lands within ±0.03 of the first
   pass's 0.416–0.422, i.e. still strictly between DeiT-small and our LMs.

The §6 decision rule of the first spec applies unchanged: if 2 and 3 both hold the domain story is
supported; if both fail the LM negatives generalize; a split means accuracy and symmetry disagree and
the write-up says so rather than picking the flattering half. Both outcomes remain reportable.

## 10. Implementation

**Reuse boundary — unchanged.** `model/vit.py`, `model/sharing.py`, `build_blocks`, `vision_train.py`,
the optimizer and cosine helpers, `manifest.py`, `results.py`, `symmetry.py` are all reused as they
stand. This pass adds no model code and changes no sharing code.

| Status | Component |
|---|---|
| Reused unchanged | everything in the first pass's §7 reuse table, plus `vision_train.py` and `scripts/run_vision_queue.py` |
| New | `scripts/prepare_imagenet1k.py`, `data/in1k/*`, five `V*-s1337-in1k.yaml` configs, two `-p5k` probe configs, `configs/runs/queue-vision-in1k.txt` |
| Changed | `data/imagenet.py` gains a class-count and eval-subsample argument; `scripts/build_vision_report.py` gains the corpus/pass-count columns and the overfitting flag |

`data/imagenet.py`'s changes must leave the ImageNet-100 path bit-identical: a test loads the existing
`data/in100` memmaps through the modified loader and asserts the same batches for a fixed seed.

**Tests first**, per the first pass's R5 and the repo's practice: the prepare script's dual-path
resize logic, the stratified subsample's determinism and class balance, the report's new columns and
the overfitting flag's threshold behaviour, and the in100 regression above.

## 11. Artifacts

- `results/runs/V*-in1k.json` — existing schema plus `top1`, `top1_best`, `best_step`, `n_classes`,
  `corpus`, `passes`.
- `results/manifests/V*-in1k.json` — copied on landing.
- `results/reports/vision.md` — both passes, labelled, with the "1 seed, descriptive" header.
- `results/analysis/ffn-symmetry.json` — extended with the `-in1k` rows.
- `docs/analysis/2026-09-18-vit-pass-matched-outcome.md` — written when the pass closes, scoring
  predictions 1–6 against what happened, including any that fail.
- One commit per landed run and per stage gate.

## 12. Risks

- **R1 — under-training replaces overfitting.** At 10 passes the arms may not converge, which is the
  first spec's R1 returning by the other door. The §3 probe is its cheapest test and its chance-floor
  check is the tripwire. The LM ladder converges at 10.31 passes, which is the reason to expect this
  is survivable.
- **R2 — the corpus swap changes more than the pass count.** 1000-way classification is a harder task
  than 100-way, not merely a bigger one. Mitigated by claiming only within-pass arm gaps, and by
  prediction 5, which is written so that a probe-pass / gate-fail combination is recorded as a partial
  falsification rather than explained away.
- **R3 — prepare cost.** 1.28M images decoded and packed is unmeasured. Measured on a 10,000-image
  slice before committing to the full run (§4). If extrapolation puts it beyond ~3 h, report the number
  and re-plan rather than starting it.
- **R4 — memmap read throughput.** §4 measures it before the queue rather than discovering it at
  step 400.
- **R5 — one seed.** Unchanged from the first pass: every number is a point estimate and the §6
  trigger is the only path to a verdict.
- **R6 — GPU contention before the 2026-10-05 gate (T-006, issue #12).** ~5 h of GPU across stages 0–3.
  The gate work has priority if they collide.

## 13. Out of scope

No augmentation changes, no pretrained weights, no distillation, no SOTA chasing, no new sharing
mechanisms, no change to the LM ladder's protocol, no re-run of the ImageNet-100 arms beyond the §3
probe. Five arms, one seed, one corpus swap, one question.
