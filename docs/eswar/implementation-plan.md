# IL-7Rα Minibinder Generation — Implementation Plan (v2, corrected after code check)
### Role: Candidate Generation & Design Approaches (Eswar)

**Target:** IL-7Rα (CD127) · P16871-1 · residues 21–239 · ref 3DI3 chain B · minibinder 40–100 aa · ≤20 candidates.
**Important facts found by checking the code (these changed the plan):**
- Hotspot numbers are **UniProt numbers** (`audit.json`: `label_seq_id_equals_uniprot_position: true`). Moving to 21–239 does **not** renumber them.
- 3DI3 only has a structure for residues 37–231. So **generation keeps the 37–231 structure**; the real change to 21–239 is in **evaluation** (full sequence + new MSA).
- `modal_app.py` can only run **RFD4 + Boltz-2 + ipSAE**. It has **no ESMFold2, no Mosaic, no USalign, no MMseqs2**.
- ESMFold2 is the **official screening model**, and it has never been qualified on Modal (`docs/VERIFICATION.md`).

---

## Phase 1 — Prepare the target  *(CPU only, no GPU, no cost)*

| Task | What to do | Done when |
|---|---|---|
| 1.1 | Run the **existing** command (no new script needed): `cd pipeline && python -m binder_campaign prepare-targets --config config/targets-2026-11-13.json --output data/current-targets` | `data/current-targets/il7ra/manifest.json` has start 21, end 239, 219 aa, hashed. ✅ **Tested 16 Nov** (219 aa, starts E, ends D); output saved in `docs/eswar/phase1-data/` |
| 1.2 | Run the **existing** MSA script (it already accepts any manifest): `python scripts/fetch-target-msa.py --manifest data/current-targets/il7ra/manifest.json --output data/current-targets/il7ra/msa`. Do **not** trim the old 1–236 MSA. | `target.a3m` + `manifest.json` (rows, sha256) saved. ✅ **Tested 16 Nov** (1022 rows, target-only); saved in `docs/eswar/phase1-data/` |
| 1.3 | Keep the frozen 37–231 structure for generation. Write a short note: residues 21–36 and 232–239 have no structure; 3DI3 position 66 differs from UniProt. No silent repair. | ✅ **Done 16 Nov** — see `docs/eswar/phase1-reference-note.md` (finding: the shipped reference is the same fold as 3DI3 chain B, 0.78 Å, but not identical: residue 66 and two extra residues) |
| 1.4 | Build the conditioning file with **binder length and hotspots as parameters** (new `binder_campaign/conditioning.py` + test). Hotspots come from Samyuktha; until then use the historical ten, labelled provisional. | valid `conditions.json` + `audit.json`. ✅ **Built and tested 16 Nov** (50/50 tests pass; rebuilds the historical file byte for byte). The variable-length contig is a syntax check only until a GPU run. Hotspot options A/B/C prepared for the epitope decision: see `docs/eswar/hotspot-decision.md`. |

**Phase output:** on-target evaluation inputs (21–239 sequence + new MSA) and a parametrized conditioning file.
**Needs internet:** rest.uniprot.org and api.colabfold.com (blocked in the cloud session — run on your laptop).

---

## Phase 2 — Build the generator  *(code = CPU; first GPU run is Task 2.4)*

| Task | What to do | Done when |
|---|---|---|
| 2.1 | New `scripts/rfd4-generate.py` from `rfd4-il7ra-seed.py`: remove the seed whitelist and the `reference_rehearsal_only` role guard, read binder length from the audit. Keep identity/chain/finite checks. | dry-run passes input checks. ✅ **Built and tested 16 Nov**: checks in `binder_campaign/generation_input.py`, settings guarded against drift from the qualified recipe, `--dry-run` works without a GPU (22 tests; 76 in the suite). **The GPU part is untested until Task 2.4.** Note: upstream contigs accept length ranges (e.g. `"40-100, /0, A37-231"`), so a length range is possible, but per the upstream spec the **model samples the length itself** with a range; use fixed numbers (50, 60, 70, 80) for a controlled length sweep. Relax the script's `binder_length` equality check for ranges. |
| 2.2 | New `binder_campaign/dedup.py`: drop a design too similar (sequence overlap / backbone TM) to one already kept. Add `tests/test_dedup.py`. | tests pass. ✅ **Built and tested 16 Nov** (22 tests; 98 in the suite): drops copies by aligned sequence identity, contiguous overlap and backbone TM; checked on the one real design (RFD4 vs Boltz pose TM 0.99; unrelated chains 0.17–0.28). Thresholds are **uncalibrated defaults**. Needs `tmtools==0.3.0` (`requirements-eswar.txt`). |
| 2.3 | New `import_current()` (tag `competition_campaign`, `submission_eligible=False`) + test. | tests pass. ✅ **Built and tested 16–17 Nov**: put in its own module `binder_campaign/current_candidates.py` so the team's `candidates.py` stays untouched (plan changed). 20 tests; 120 in the suite. Records start unreviewed and not eligible; dropped-by-dedup designs are blocked; `export_draft` still refuses them until the reviews are filled in. **Follow-ups done 17 Nov:** (1) the real `rfd4-generate.py` now runs end to end on a CPU with only the model call replaced (biotite 1.6.0, 4 tests; planted bugs were caught), so only the real RFD4/GPU call is still unproven; (2) `docs/eswar/candidates-reexport.patch` makes the function reachable as `candidates.import_current` without editing the team's file (applies cleanly; 118 tests pass with it applied); (3) `import_current(..., require_reviewed_hotspots=True)` blocks designs made from unreviewed hotspots (default off). |
| 2.4 | Update `modal_app.py`: `stage_root` must load the **current** manifest + MSA (today it copies the historical ones). Add a `--length` option. Run **one** small test generation. | one on-target design generated, cost logged ✅ **Built and tested on a CPU 17 Nov, NOT yet run on Modal.** New `modal_current.py` (the team's `modal_app.py` is left alone; same pins and cache Volume), `binder_campaign/modal_plan.py`, `scripts/plan-modal-run.py`. Found and fixed before any GPU spend: the historical RFD4 image has no Biopython, which `generation_input.py` imports. The task's finish line (one on-target design on a GPU, cost logged) needs you to run it: see `docs/eswar/RUN-GPU.md`. |
| 2.5 | New batch driver `scripts/run-il7ra-generation.py` (seeds × lengths → generate → dedup → record), budget-capped. | resumable batch, failures kept ✅ **Built and tested on a CPU 17 Nov, NOT yet run on Modal.** Engine `binder_campaign/batch.py` (budget caps required, resumable, failures kept, dedup replayed on resume), `scripts/run-il7ra-generation.py` (free `plan` and `report`), and the `--stage batch` entry in `modal_current.py`. The whole chain (entry point, real function body, real generation script, importer, engine) runs end to end on a CPU with only the model call replaced. See `docs/eswar/RUN-GPU.md`, Stage C. |

**Phase output:** a batch of on-target designs as candidate records.

---

## Phase 3 — Evaluate and filter

| Task | What to do | Done when |
|---|---|---|
| 3.1 | **Qualify ESMFold2 on Modal first** (new image: `esm==3.4.1.post1`, ~25 GB pinned checkpoint, `env/esmfold2-checkpoints.json`). This is real work, not glue. | one known sequence folds on Modal |
| 3.2 | Multi-candidate evaluation script. The existing `prepare-candidate-evaluation.py` accepts **exactly one** candidate and only `reference_rehearsal` ones — write a new batch version. Run ESMFold2 + Boltz-2 + ipSAE, keep raw scores. | scores attached to every candidate |
| 3.3 | QC checks (length 40–100, single chain, clash/contact/pose-shift) using `pose_review.py`. | `qc_reasons` filled |
| 3.4 | Diversity: `review-il7ra-diversity.py` is **not reusable as-is** (it needs exactly 16 records, Earth paths, a USalign binary, `3DI3.pdb`). Extract only the pairwise TM + site-Jaccard + grouping logic into a new script; add USalign to the image. | cluster report on any pool |

**Phase output:** scored, QC-passed, clustered pool. Keep a design only if **both** ESMFold2 and Boltz-2 agree it is good (they disagree even on the real IL-7/IL-7Rα pair).

---

## Phase 4 — Compare design methods  *(optional before 17 Nov; propose first)*

| Task | What to do | Done when |
|---|---|---|
| 4.1 | Adapt `mosaic-il7ra.py`: it is hardwired to the historical manifest, first 128 MSA rows, 60 aa. Add a JAX image with the 3 pinned Mosaic revisions (not in `modal_app.py`). | Mosaic runs on current target. **Blocked on one input:** the repo records only commit hashes. `mosaic` and the `boltz` fork were found under `escalante-bio`; the `joltz` source URL is unknown — ask the captain. |
| 4.2 | Raise the Modal timeout for Mosaic: one design took **~16 min** on L40S; the current limit is **15 min (900 s)**. | no timeout |
| 4.3 | Compare RFD4 vs Mosaic at **matched GPU-hours, not matched count**: RFD4 ≈ 1 min/design, Mosaic ≈ 16 min/design. Metric: diverse, viable candidates per GPU-hour. ESMFold2 is the independent check (Mosaic optimizes through Boltz). | comparison table, failures kept |

**Budget correction:** 10 Mosaic designs ≈ **2.7 GPU-hours**, not "under 1 hour" as I said before.

---

## Phase 5 — Select and finalize

| Task | What to do | Done when |
|---|---|---|
| 5.1 | Novelty pre-screen with MMseqs2 (coverage ≥0.8, identity ≤0.7). Needs MMseqs2 + a public database (neither is in the repo). Flag only; cannot certify. | flag per candidate |
| 5.2 | Sequence liabilities via `sequence_review.py`; optional SolubleMPNN repair (lead101 pattern). | liabilities recorded |
| 5.3 | Select ≤20 maximizing distinct fold/site clusters, no duplicates; export with `export_draft`. | export guards pass |
| 5.4 | Send finalists to Stefanos for AlphaFold cross-check. | AF result per finalist |

---

## What is realistic by 17 Nov (tomorrow)
The brief asks you to **explore, report, and *propose* one small experiment** — not to finish the pipeline. Realistic: Phase 1 (cheap), maybe one small RFD4 run (2.4), and the 5-part write-up with the Phase 4 comparison as your **proposed** experiment. Phases 3–5 are the build for after the 18 Nov team decision.

## Tests
New `test_conditioning.py`, `test_dedup.py`; extend `test_candidates.py`. All must pass with the existing 35: `(cd pipeline && python -m unittest discover -s tests)`.

---

## Verification status (16 Nov) — what is proven and what is not

| Item | Status | Evidence |
|---|---|---|
| Task 1.1 target prep (21–239) | ✅ **Tested** | ran `prepare-targets`: 219 aa, E…D, hashed |
| Task 1.2 target-only MSA | ✅ **Tested** | ran `fetch-target-msa.py`: ticket COMPLETE, 1022 rows |
| Numbering unchanged | ✅ **Tested** | current[37..231] == frozen reference; hotspot residues identical |
| RFD4 entry point + 14 Hydra keys + `thinking_high` | ✅ **Checked in pinned source** (`ddd934f`) | `generate(overrides=…)`, all keys, `configs/recycle_strategy/thinking_high.yaml` exist |
| ESMFold2 API used by the repo | ✅ **Checked in `esm==3.4.1.post1`** | all classes, `fold()` args, `from_a3m`, `set_kernel_backend` exist |
| Mosaic imports used by the repo | ✅ **Checked in pinned source** (`5937f37`) | all 7 names exist |
| `joltz` source (Mosaic dependency) | ❌ **Unresolved** | URL not in repo; ask captain |
| Any GPU run (RFD4, Boltz, ESMFold2, Mosaic) | ❌ **Not run** | no Modal access in the cloud session |
| Runtime behaviour of the checked APIs | ❌ **Not run** | "name exists" is not "works correctly" |

### GPU verification checklist — run on your laptop (your own Modal profile)
1. `modal billing summary` — note the balance first.
2. `modal run modal_app.py --stage prepare`
3. `modal run modal_app.py --stage generate --run-id eswar-seed43-001 --seed 43`
4. `modal run modal_app.py --stage evaluate --run-id eswar-seed43-001`
5. Compare with `examples/verified-modal-run/summary.json`: binder is 60 aa; ipTM about 0.945; ipSAE about 0.78; generation about 58 s; refold about 68 s on an A10G. Small differences are possible; large ones are a problem to report.
6. `modal billing summary` again — record the cost.
