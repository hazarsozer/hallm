# HaLLM — handover to Alper (2026-09-27)

From 2026-09-27 **Alper leads HaLLM**: he owns `main`, the run plan, the open decisions and the
advisor thread. Hazar has moved his focus to another project; he stays on CC and is reachable for
questions, but nothing waits on him any more. This file is the entry point. It says where things
stand, what is open, what to do first, and where everything lives.

## Reading order

1. **This file.**
2. [`docs/PROJECT.md`](docs/PROJECT.md): the question, hypotheses, scope, timeline, risks (the PRD).
3. [`RESULTS.md`](RESULTS.md): every result with its caveats. Generated tables in `results/reports/`.
4. [`docs/DECISIONS.md`](docs/DECISIONS.md): why the project looks the way it does, including every
   withdrawn claim.
5. [`CHANGELOG.md`](CHANGELOG.md): what happened when, study by study.
6. [`docs/RUNBOOK.md`](docs/RUNBOOK.md): how to set up, run, resume, report and sync.
7. [`COLLABORATOR.md`](COLLABORATOR.md): the experiment-hygiene rules, which still apply to everyone.

Specs (`docs/superpowers/specs/`) and outcome analyses (`docs/analysis/`) are the detailed record.
Each outcome doc scores its own pre-registered predictions.

---

## 1. Where things stand, in six lines

1. **W+Wᵀ costs ~14% perplexity for −50% storage at ~25M**, falling −1.28 pp per doubling. The cost
   is roughly proportional to capacity removed, whichever sublayer is shared. It never meets ≤2%.
2. **At fixed storage, looping beats the unshared model and W+Wᵀ does not** (L8 and L16, 3 seeds,
   every paired seed). At matched storage *and* compute, looping beats W+Wᵀ by 5–6%.
3. **The transposed loop (our W/Wᵀ-across-depth idea) lost** to the plain loop in all three variants.
4. **Vision does not rescue W+Wᵀ.** Our own arms trained as ViTs reproduce language's direction on
   every pair the study pre-registered: looping wins at fixed storage, W+Wᵀ loses (by 7.94 points),
   the transposed loop loses to the plain loop. Two differences: looping beats even the 2×-storage
   unshared model in vision, and the transposed loop loses to the shallow unshared model in vision
   while it roughly ties it in language. One seed, and ~2× the LMs' data reuse, not matched
   (corrected 2026-09-27).
5. **Withdrawn:** "W+Wᵀ beats ALBERT" (unfair ratio) and "vision FFNs are more symmetric" (did not
   reproduce). Neither may appear in the thesis except as a retraction.
6. **Track 2 (synthetic reasoning) is built, not yet run.** Alper's PR #13 (T-006): three tasks, a
   harness, and a p-hop sizing pilot at p=4, d=128. At p=4 the validity check passes on one seed
   (looped 85.0% vs unshared deep 87.7%, within the PR's own 5 pp tolerance); at p=3 it fails by
   25.6 pp. Unreviewed and unmerged since 2026-09-18.
   **Not started:** the Track 2 grid, the 2026-10-05 gate, the 124M rung.

## 2. What changes with the handover

| | before | from now |
|---|---|---|
| `main` | Hazar merged; Alper sent PRs | **Alper merges and pushes.** Branches and PRs are still good practice for anything you want reviewed. |
| decisions | Hazar | **Alper.** Big direction changes still go to Töreyin. |
| advisor | Hazar | **Alper**, CC Hazar and Ahmet Nuri Yılmaz on everything |
| compute | Hazar's RTX 4070 SUPER box | **Alper's laptop (8 GB 3070 Ti) + rented cloud GPU** (or UHEM/SP4CING if Töreyin grants it) |
| tasks | GitHub issues via `scripts/tasks.py` | Optional now. Keep using them if they help you or a future helper. |

COLLABORATOR.md's experiment rules (pairs never split across machines or code versions, configs never
hand-edited, reports generated, HF add-only) exist to protect results, not to limit a guest. They still
apply.

## 3. Open decisions (yours now)

Ordered by urgency. Each has the context and the previous read, which you are free to overrule.

### D1. The 2026-10-05 gate — hold it, or move it
The gate picks which W+Wᵀ variant and which looped variant run at 124M (spec §6). A variant shows
a signal if it satisfies H-C or H-R on any Track 1 metric **or any Track 2 task**, so the gate needs
at least the p-hop arms of the Track 2 grid (the pilot in PR #13 ran only the calibration arms). It
also needs Töreyin's OK on the reframe and the 124M spend (spec §11 R11). **Previous read:** hold the
gate until the Track 2 p-hop arms are in and the advisor has agreed, rather than decide 124M on
Track 1 alone.

### D2. Which W+Wᵀ variant takes the 124M slot
No transposed-loop variant passed H-T, so by spec 2026-09-15 §4 the 2026-09-14 gate rule (§6) stands
unchanged: pick the W+Wᵀ variant that shows a signal (H-C or H-R vs the unshared model at the same
storage); if none, the lowest iso-storage PPL gap; default full A2 at 2× depth. What the rule sees today:
- **`L10-A2attn` shows the only W+Wᵀ-family signal**: BLiMP at iso-storage vs `L8-A0`, SUPPORTED
  (`results/reports/capability.md`, n=2, Δ −0.66). Caveats: 2 seeds, and it stores 4% *more* than
  `L8-A0`. On perplexity it ties (`iso-storage.md`: +0.16% mean over 3 seeds, signs mixed). It is also
  the lowest iso-storage PPL gap of any W+Wᵀ variant. So as things stand the rule picks **A2attn**
  (spec §7 names "the gate's A2attn variant" as the alternative to `L24-A2`).
- Full `A2` loses at iso-storage everywhere measured (+2.2% at L8, +3.9% at L16; −7.94 points in
  vision). The 2026-09-18 outcome (§8.3) concluded the vision result gives the slot **even less reason
  to hold plain W+Wᵀ**.
- Track 2 can still change this: a W+Wᵀ arm with a reasoning signal there would count.
Whatever the rule picks, the choice and its evidence go to Töreyin with the compute ask. Dropping the
slot instead would save 40% per seed (20 + 40 + 40 h → 20 + 40 h), but leaves the thesis's own mechanism
out of its largest rung.

### D3. The ViT follow-up (seeds, and the 2× reuse correction)
Two things, best decided together:
- The spec's follow-up trigger fired (`V8-A2` vs `V4-A0` at 7.94 points ≥ 1.0), pre-registering seeds
  1338/1339 for `V8-A2`, `V4-A0`, `V8-A1u4`: 6 runs, ~13 h on a 4070-class GPU.
- The re-run turned out to be ~1.94× the LMs' data reuse, not matched (DECISIONS 2026-09-27). A truly
  matched run is ~25,800 steps on ImageNet-1k, about half the cost per run.
**Previous read (made before the 2× error was found):** the trigger was written for a marginal effect,
and 7.94 is not marginal. The result that deserves seeds is the outcome doc's §4.2: looping beats the unshared *ceiling*
at half the storage (+0.87), which never happens in language. **Suggested now:** if any vision GPU is
spent, spend it on the pass-matched arms at ~25.8k steps (5 arms × 1 seed, ~half the original cost),
then seeds on `V8-A1u4` vs `V8-A0` if that +0.87 survives. Or state the 2× gap as a caveat and spend
nothing. Downside of re-running: it costs cloud GPU time that Phase 2 also needs.

### D4. Compute and budget for Phase 2
~300 GPU-h on a 4070-class card for 3 seeds × 3 arms at 124M (spec §7): ~100 h for one seed. Seed
1337 first for all arms, so a partial grid is still complete. Ask Töreyin about UHEM/SP4CING access
first; then set a cloud budget cap. An hourly rented GPU beats Colab for multi-day runs (resume works,
but Colab sessions time out).

### D5. Parked questions (no deadline)
- FineWeb-Edu's val/test gap is 8.8%, against 0.5–2.4% (mean 1.2%) on WikiText. Unexplained. Check before the 124M
  rung reports test PPL on FineWeb-Edu.
- The transposed-loop rows in the reports read "pending (1/3 seeds)", although by the seed rule they
  are final at one seed. It's a report-generator wording fix.
- Four arXiv PDFs over 5 MB in `raw/papers/` (67 MB) are on the public repo; arXiv's default licence
  may not allow redistribution. Consider removing them from the tree (history keeps them anyway) and
  linking the abstracts instead.

## 4. First week: suggested order

1. **Pull `main`** (this file arrives with it) and read this file and PROJECT.md. `uv sync && uv run pytest`.
2. **Update Töreyin** (in person at term start if possible, CC Hazar and Ahmet Nuri Yılmaz). Cover:
   - the results so far (§1 above, and RESULTS.md's conclusions)
   - the 2026-09-14 reframe (fixed storage), and his OK on it
   - the negatives: the transposed loop, and W+Wᵀ in vision
   - **the retraction**: the 2026-09-17 symmetry claim is withdrawn, and the ViT re-run is ~2× the LMs'
     reuse, not matched. Tell him before anything cites either.
   - the 124M W+Wᵀ variant (D2) and the compute ask: UHEM/SP4CING access, or approval for cloud spend (D4)
   - the timeline (PROJECT.md §6) and the Design II deliverables once Ninova lists them
3. **Merge PR #13 (T-006)** after a read-through, then run the Track 2 p-hop grid at p=4 (the W+Wᵀ
   and looped arms at equal storage, 3 seeds). Two things to settle while reading it:
   - **p=3 fails the validity check badly** (looped 68.8% vs deep 94.4%) while p=4 and p=5 pass. p=4
     was chosen after seeing this. State that in the write-up, and consider reporting p=3 and p=5
     alongside p=4 rather than p=4 alone, so the choice of difficulty can't look selected.
   - The pilot's literal thresholds (shallow ≤50%, deep ≥90%) are not met at p=4 (54.8%, 87.7%), and
     the 5 pp validity tolerance was set in the PR, not the spec. Record both in the spec before the
     grid runs, not after. The validity check is one seed (seed 0).
   - Addition and binding chains are built but not calibrated.
4. **Set the gate date** (D1) once 2 and 3 are in.

## 5. Where everything lives

| artifact | where | notes |
|---|---|---|
| code, configs, results, reports, specs | GitHub `hazarsozer/hallm` (public) | `results/runs/*.json` is the source of truth; `results/reports/` is generated |
| LM checkpoints (`model.pt`) | HF `hallm-thesis/hallm-wikitext103` (private), `checkpoints/<run-id>/` | 4 have no weights anywhere (DECISIONS 2026-09-14); their numbers stand |
| WikiText-103 token bins | HF, same repo, `data/` | verify the SHA-256 before any run |
| FineWeb-Edu token bins (2.6B train tokens) | HF, same repo, `data/fineweb/` | provenance and hashes: `data/fineweb/SOURCE.json`; regenerable with `scripts/prepare_fineweb.py` |
| ImageNet-1k / -100 memmaps (65 GB / 6.5 GB) | not uploaded (licence) | regenerate with `scripts/prepare_imagenet1k.py` / `prepare_imagenet100.py`; provenance in `data/in1k/SOURCE.json` |
| ViT checkpoints | HF, same repo, `checkpoints/V*` | first pass, probes and ImageNet-1k runs |
| research wiki (literature, roadmap, analyses) | `wiki/` in this repo (Obsidian) | the roadmap predates the 2026-09-14 reframe; PROJECT.md is current |
| independent implementation | GitHub `alpericon/wplusw-lm` | cross-validation is still a planned item |
| course rules | ITU senior design principles doc (R1, 2023-10-01) | advisor approval before the final exam; faculty templates mandatory |

## 6. Hazar's last steps (done 2026-09-27)

- [x] Merged `controlled-vit` + `vit-pass-matched` into `main` and pushed, with the corrections above.
- [x] ViT checkpoints (12 runs) added to HF under `checkpoints/<run-id>/`.
- [x] FineWeb-Edu bins added to HF `data/fineweb/`, hashes matching `SOURCE.json`.
- [x] Alper has write access to the GitHub repo.
- [x] Every LM and ViT run on the 4070 box is on HF (`hf_sync.py --dry-run`: nothing to upload).
