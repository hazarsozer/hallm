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
4. **Vision does not rescue W+Wᵀ.** Our own arms trained as ViTs order exactly as in language, and
   W+Wᵀ loses by 7.94 points. One seed. The run is ~2× the LMs' data reuse, not matched (corrected
   2026-09-27).
5. **Withdrawn:** "W+Wᵀ beats ALBERT" (unfair ratio) and "vision FFNs are more symmetric" (did not
   reproduce). Neither may appear in the thesis except as a retraction.
6. **Track 2 (synthetic reasoning) is built and calibrated, not yet run.** Alper's PR #13 (T-006):
   three tasks, a harness, a p-hop sizing pilot (p=4, d=128), and a passing validity check
   (looped 85.0% vs unshared deep 87.7%). It has been unreviewed and unmerged since 2026-09-18.
   **Not started:** the Track 2 grid, the 2026-10-05 gate, the 124M rung.

## 2. What changes with the handover

| | before | from now |
|---|---|---|
| `main` | Hazar merged; Alper sent PRs | **Alper merges and pushes.** Branches and PRs are still good practice for anything you want reviewed. |
| decisions | Hazar | **Alper.** Big direction changes still go to Töreyin. |
| advisor | Hazar (no update sent since 2026-08-14) | **Alper**, CC Hazar and Ahmet Nuri Yılmaz on everything |
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
also needs Töreyin's OK on the reframe and the spend, and he has not been briefed. **Previous read:** move the gate to the week after the briefing rather than decide
124M without him.

### D2. What takes the W+Wᵀ slot at 124M
The 2026-09-15 amendment assumed a transposed-loop winner would fill it. There is none. Options:
- **plain `L24-A2`**: the thesis's own mechanism, and the spec's default. It loses in every setting
  measured so far, so 124M would confirm a negative at a larger scale.
- **a second looped point** (e.g. a different k): more likely to show a signal, but looping is prior
  work, so it adds little to the contribution.
- **drop the slot**: two arms, cheaper by ~⅓.
**Previous read:** plain A2 is still the honest choice. The thesis is about W+Wᵀ, and "the negative
holds at 124M" is a result. The vision study gives the slot even less reason to hold anything *else*.
Needs Töreyin's OK either way.

### D3. The ViT follow-up (seeds, and the 2× reuse correction)
Two things, best decided together:
- The spec's follow-up trigger fired (`V8-A2` vs `V4-A0` at 7.94 points ≥ 1.0), pre-registering seeds
  1338/1339 for `V8-A2`, `V4-A0`, `V8-A1u4`: 6 runs, ~13 h on a 4070-class GPU.
- The re-run turned out to be ~1.94× the LMs' data reuse, not matched (DECISIONS 2026-09-27). A truly
  matched run is ~25,800 steps on ImageNet-1k, about half the cost per run.
**Previous read (made before the 2× error was found):** the trigger was written for a marginal effect,
and 7.94 is not marginal. The result that deserves seeds is §4.2: looping beats the unshared *ceiling*
at half the storage (+0.87), which never happens in language. **Suggested now:** if any vision GPU is
spent, spend it on the pass-matched arms at ~25.8k steps (5 arms × 1 seed, ~half the original cost),
then seeds on `V8-A1u4` vs `V8-A0` if §4.2 survives. Or state the 2× gap as a caveat and spend
nothing. Downside of re-running: it costs cloud GPU time that Phase 2 also needs.

### D4. Compute and budget for Phase 2
~300 GPU-h on a 4070-class card for 3 seeds × 3 arms at 124M (spec §7): ~100 h for one seed. Seed
1337 first for all arms, so a partial grid is still complete. Ask Töreyin about UHEM/SP4CING access
first; then set a cloud budget cap. An hourly rented GPU beats Colab for multi-day runs (resume works,
but Colab sessions time out).

### D5. Parked questions (no deadline)
- FineWeb-Edu's val/test gap is 8.8%, against ~2% on WikiText. Unexplained. Check before the 124M
  rung reports test PPL on FineWeb-Edu.
- The transposed-loop rows in the reports read "pending (1/3 seeds)", although by the seed rule they
  are final at one seed. It's a report-generator wording fix.
- Four arXiv PDFs over 5 MB in `raw/papers/` (67 MB) are on the public repo; arXiv's default licence
  may not allow redistribution. Consider removing them from the tree (history keeps them anyway) and
  linking the abstracts instead.

## 4. First week: suggested order

1. **Pull `main`** and read this file and PROJECT.md. `uv sync && uv run pytest`.
2. **Brief Töreyin** (in person at term start if possible, CC Hazar and Ahmet Nuri Yılmaz). He has
   heard nothing since 2026-08-14. Cover:
   - the results so far (§1 above, and RESULTS.md's conclusions)
   - the 2026-09-14 reframe (fixed storage), which he has not approved yet
   - the negatives: the transposed loop, and W+Wᵀ in vision
   - **the retraction**: the 2026-09-17 symmetry claim is withdrawn, and the ViT re-run is ~2× the LMs'
     reuse, not matched. Tell him before anything cites either.
   - the open 124M slot (D2) and the compute ask: UHEM/SP4CING access, or approval for cloud spend (D4)
   - the timeline (PROJECT.md §6) and the Design II deliverables once Ninova lists them
3. **Merge PR #13 (T-006)** after a read-through, then run the Track 2 p-hop grid at p=4 (the W+Wᵀ
   and looped arms at equal storage, 3 seeds). Two things to settle while reading it:
   - **p=3 fails the validity check badly** (looped 68.8% vs deep 94.4%) while p=4 and p=5 pass. p=4
     was chosen after seeing this. State that in the write-up, and consider reporting p=3 and p=5
     alongside p=4 rather than p=4 alone, so the choice of difficulty can't look selected.
   - The pilot's literal thresholds (shallow ≤50%, deep ≥90%) are not met at p=4 (54.8%, 87.7%).
     Record the amended rule in the spec before the grid runs, not after.
   - Addition and binding chains are built but not calibrated.
4. **Set the gate date** (D1) once 2 and 3 are in.

## 5. Where everything lives

| artifact | where | notes |
|---|---|---|
| code, configs, results, reports, specs | GitHub `hazarsozer/hallm` (public) | `results/runs/*.json` is the source of truth; `results/reports/` is generated |
| LM checkpoints (`model.pt`) | HF `hallm-thesis/hallm-wikitext103` (private), `checkpoints/<run-id>/` | 4 have no weights anywhere (DECISIONS 2026-09-14); their numbers stand |
| WikiText-103 token bins | HF, same repo, `data/` | verify the SHA-256 before any run |
| FineWeb-Edu token bins (2.6B train tokens) | see §6 | provenance and hashes: `data/fineweb/SOURCE.json`; regenerable with `scripts/prepare_fineweb.py` |
| ImageNet-1k / -100 memmaps (65 GB / 6.6 GB) | not uploaded (licence) | regenerate with `scripts/prepare_imagenet1k.py` / `prepare_imagenet100.py`; provenance in `data/in1k/SOURCE.json` |
| ViT checkpoints | see §6 | |
| research wiki (literature, roadmap, analyses) | `wiki/` in this repo (Obsidian) | the roadmap predates the 2026-09-14 reframe; PROJECT.md is current |
| independent implementation | GitHub `alpericon/wplusw-lm` | cross-validation is still a planned item |
| course rules | ITU senior design principles doc (R1, 2023-10-01) | advisor approval before the final exam; faculty templates mandatory |

## 6. Hazar's last steps (to be ticked before this handover is complete)

- [ ] Merge `controlled-vit` + `vit-pass-matched` into `main` and push (with the corrections above).
- [ ] Upload the ViT checkpoints to HF under `checkpoints/<run-id>/` (add-only).
- [ ] Upload the FineWeb-Edu bins to HF `data/fineweb/`, or confirm regeneration is the plan.
- [ ] Confirm Alper has write access to the GitHub repo.
- [ ] Hand over anything left in `runs/` on the 4070 box that is not on HF.
