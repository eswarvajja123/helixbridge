# IL-7Rα minibinder candidates — Eswar's work (Changjiang Cup 2026)

**Role:** candidate generation and design approaches. **Target:** IL-7Rα (CD127), UniProt P16871-1, residues 21–239. **Binder:** 40–100 aa minibinder, at most 20 candidates.
**Written:** 18–19 Nov 2026, from the real result files in `docs/eswar/results/`. Every number below can be traced to a file there.

> **Read this first.** This is a *proposal* for the team discussion, not a final submission. **Nothing here shows that any design binds IL-7Rα.**
> Every score is a computer model's opinion, and the team's own pilot found negative controls scoring as high as binders. What the work does show is that the
> pipeline runs end to end on real GPUs, that the designs are *consistent across three independent predictors*, and that shuffled sequences score clearly lower.
> The team's export guard is **closed on purpose**: it needs three human reviews per design, which nobody has done yet.

---

## 1. At a glance

| Question | Answer |
|---|---|
| Is the pipeline built and tested? | **Yes.** Phases 1–5, 511 automated tests (each module also checked by deliberately planting bugs). |
| Was it run on real GPUs? | **Yes**, on Modal (A100, L40S, A10G). RFD4, Boltz-2, ESMFold2, Mosaic, AlphaFold-Multimer, SolubleMPNN repair. |
| How many designs? | **72** in the final pool (RFD4), 8 more from Mosaic. **22** pass every check including AlphaFold. |
| What is the proposed shortlist? | **20 designs**, table in section 5. File: `docs/eswar/results/selection-final/selection.csv`. |
| Does anything prove binding? | **No.** See "What this does not show" below. |
| Can the shortlist be exported to the organizers? | **Not yet.** The export guard refuses until a person records the novelty, structure and screen reviews. |

### What the evidence does show
1. The designs fold into the intended complex in **three independent predictors** (ESMFold2, Boltz-2, AlphaFold-Multimer): for the final 20, ipTM is 0.82–0.97 in each.
2. **Shuffled copies of the same sequences do not.** Over 20 controls, 0 pass structure QC; AlphaFold ipTM is 0.28 median (max 0.61) against 0.91 for the designs.
3. The result does **not depend on one score**: designs were chosen for coverage of distinct folds and binding sites, not for the highest number.

### What this does not show
- That any design binds. A prediction of a complex is not an experiment.
- That the targeted epitope is the right one. All three hotspot sets are **unreviewed** (Samyuktha decides).
- Novelty against the organizers' private database. Only a **public** database was searched.
- That ESMFold2 as run here equals the organizers' screen (it is an approximation, same weights).

---

## 2. What was built (phases from the implementation plan)

| Phase | What | Status |
|---|---|---|
| 1. Prepare the target | 21–239 sequence, 1,022-row target-only MSA, conditioning with length and hotspots as parameters | Done |
| 2. Generator | RFD4-Proteina batch driver (seeds × lengths, budget caps, resumable, failures kept), near-duplicate gate | Done, run on GPU |
| 3. Evaluate and filter | ESMFold2 + Boltz-2 + ipSAE scoring, structure QC, diversity report (TM-score + binding-site overlap) | Done, run on GPU |
| 4. Compare methods | Mosaic (optimizes a sequence through Boltz-2) vs RFD4 at matched GPU-hours | Done, run on GPU |
| 5. Select and finalize | novelty pre-screen, liability review, selection, AlphaFold cross-check, optional repair | Done (see open items) |

Extras added for calibration: shuffled **negative controls**, **AlphaFold gate** in selection, **repair** of sequence liabilities.

**Design rules followed:** hotspot numbers are UniProt numbers; the structure available for generation covers residues 37–231 (reference 3DI3 chain B, 0.78 Å over 193 residues), the evaluation uses the full 21–239 sequence; nothing in code ever sets a human review field (`novelty_review`, `independent_prediction_status`, `screen_status`, `submission_eligible`).

---

## 3. Results

### 3.1 Generation and scoring
- **RFD4-Proteina** (pinned commit `ddd934f`, the team's `thinking_high` settings), lengths 50–80, three hotspot options:
  - **A** — provisional historical set (100–102, 158, 159, 161, 211–214).
  - **B** — exposed set on the natural IL-7 site (78, 79, 98, 100, 102, 103, 158, 159, 211, 212), proposed by Eswar.
  - **C** — off-site exploratory (170–179, 200).
- **Share of designs passing every QC check:** B 17 of 35 (49 %), A 10 of 36 (28 %), **C 0 of 20** — for option C both predictors put every binder about 55 Å from where it was designed, so neither independent model supports off-site binding.
- Most common failure: **Boltz-2 relocating the binder** (33 of 71 designs). One design got a high Boltz-2 ipSAE from a pose 19.6 Å away from the designed one, so a score alone would have hidden the problem — that is why pose checks are part of QC.
- **Repeatability:** ipTM changes between runs are about 0.002 (ESMFold2) and 0.003 (Boltz-2) across GPU types; the three ESMFold2 kernel backends agree (0.9492, 0.9492, 0.9493); QC outcomes were identical for 39 designs scored twice. One weak design swung by 0.29, so single numbers for low-confidence designs are unreliable.

### 3.2 Diversity, novelty, sequence liabilities
- **Diversity:** 15 fold groups among designs passing QC, but only **3 distinct binding sites** — a fixed hotspot set gives different folds on the same site. (TM-score validated against a real USalign build on 28 pairs.)
- **Novelty pre-screen** (MMseqs2, flag if query coverage ≥ 0.8 and identity > 0.7, public database of Swiss-Prot + PDB chains, 756,426 sequences): no design flagged. *This cannot certify novelty; it only gives an early warning.*
- **Sequence liabilities** (flags only, nothing removed): N-glycosylation sequon, odd cysteine count, Asn-Gly, Asp-Pro.

### 3.3 RFD4 vs Mosaic (Phase 4)
| | RFD4 | Mosaic |
|---|---|---|
| Designs | 72 | 8 |
| GPU seconds per design | 131 | 695 |
| Clean in structure QC | 27 of 71 | **0 of 8** |
| ESMFold2 ipTM (median) | 0.917 | **0.229** |
| Boltz-2 ipTM (median) | 0.889 | 0.712 |

Mosaic's designs look fine to Boltz-2 (it optimizes through Boltz-2) but not to ESMFold2. Small sample, untuned settings: this says "do not use Mosaic as is", not "Mosaic cannot work".

### 3.4 Negative controls (shuffled sequences)
20 shuffled copies of the shortlisted sequences, scored and QC'd exactly like the designs (`results/controls/`):
- **Structure QC:** 0 of 20 controls pass; 20 of 20 designs pass. *(Biased: the shortlist was already QC-filtered.)*
- **ESMFold2 ipTM:** controls median 0.18 (max 0.64). **Boltz-2 ipTM:** controls median 0.61 (**max 0.86**) — Boltz-2 alone is easily fooled; ESMFold2 separates better.
- A design scores above a control in 100 % of ESMFold2 and 99.5 % of Boltz-2 comparisons.

### 3.5 AlphaFold-Multimer cross-check (`results/alphafold/`, `alphafold-rest/`)
Same settings for everything: AlphaFold-Multimer v3 (ColabFold 1.6.3), our own 1,022-row target alignment plus the binder as a single sequence (no public MSA server, so no sequence leaves the container), 2 models, 3 recycles, best by ipTM.
- **Shortlisted designs of the first list (20):** ipTM median 0.91 (range 0.27–0.93); **16 agree** with the generator and both other predictors on the binder pose, **4 do not** (15–25 Å shifts; two have ipTM 0.27 and 0.53, inside the control range).
- **Shuffled controls:** ipTM median 0.28, max 0.61. A design beats a control in 96 % of pairs.
- The 4 unsupported designs and one more (`bc6f6d`, ipTM 0.44) are excluded from the final shortlist.
- This run is a **measurement**. It sets no review field. **Stefanos's own AlphaFold check is still needed.**

### 3.6 Repair of sequence liabilities (`results/repair/`)
Native SolubleMPNN (upstream `26ec57ac`, hash-checked weights `7af52d09…`) run in a Modal CPU container on the 8 shortlisted designs that had liabilities, per the team's lead101 protocol:
- **12 variants** produced (up to 2 per parent); all scored on ESMFold2 and Boltz-2; scores stay within 0.06 ipTM of the parent (one exception, below).
- Structure QC: **10 of 12 clean**, 2 flagged (`22e6bf-r1`, `d8ae7b-r1`). Novelty pre-screen: no hit for the 10 searched.
- **AlphaFold caught a failure the other two predictors missed:** both variants of `d8fd8a` (parent ipTM 0.87) fall to 0.31 and 0.36 with the binder 20–42 Å away, yet ESMFold2 and Boltz-2 scored them like the parent. Lesson: **rerun an independent predictor after any sequence change.**
- Only **3 variants** are clean on QC, novelty and AlphaFold together: `8e93b8-r1`, `0a592d-r1`, `0a592d-r2`. They are offered as replacements for their parents, **not** in the list.

---

## 4. How the final shortlist was chosen
1. **Eligible** = passes structure QC with no flags, public novelty pre-screen not flagged, scored by both predictors, liabilities reviewed, **and** AlphaFold gate: ipTM **above every shuffled control (> 0.61)** and pose within 2.5 Å of the generator and 3.0 Å of the ESMFold2 / Boltz-2 refolds. The rule was set before looking at the result.
2. **22 of 72** designs are eligible.
3. **Picked by coverage, not score**: round-robin over fold groups, then an unused binding site, then fewer liability flags, then the best agreement between the two predictors. `pick_order` is the order of picking, **not a ranking**.
4. Result: 20 designs, 13 of 13 fold groups and 2 of 2 binding sites among the eligible ones. (The first list covered 15 folds and 3 sites; the AlphaFold gate removed the only members of 2 folds and 1 site.)

## 5. The proposed shortlist (`selection-eswar-s4`)
Hotspot option A = provisional historical, B = proposed by Eswar (both **unreviewed**). F = fold group, S = binding-site group. Scores are ipTM. Order is the order of picking, not quality.

| # | Design (`il7ra-rfd4-…`) | Length | Option | Fold / Site | ESMFold2 | Boltz-2 | AlphaFold | Sequence flags |
|---|---|---|---|---|---|---|---|---|
| 1 | `08f9233a8a90` | 70 | B | F2 / S1 | 0.95 | 0.95 | 0.93 | none |
| 2 | `90b6d9399028` | 50 | A | F8 / S2 | 0.95 | 0.96 | 0.92 | none |
| 3 | `026e12ae07bc` | 60 | B | F1 / S1 | 0.95 | 0.95 | 0.92 | none |
| 4 | `67f05b49fd19` | 50 | A | F4 / S1 | 0.92 | 0.94 | 0.91 | none |
| 5 | `7fc8ab07c02f` | 50 | A | F7 / S2 | 0.91 | 0.92 | 0.88 | none |
| 6 | `13af2bd8fd8a` | 80 | B | F3 / S1 | 0.88 | 0.92 | 0.87 | N-glyc. sequon, Asn-Gly, Asp-Pro |
| 7 | `2e2ec88e93b8` | 70 | A | F5 / S2 | 0.92 | 0.92 | 0.91 | odd Cys |
| 8 | `8d368322e6bf` | 60 | A | F6 / S2 | 0.91 | 0.82 | 0.84 | odd Cys |
| 9 | `b25d020a592d` | 80 | B | F10 / S1 | 0.93 | 0.94 | 0.91 | N-glyc. sequon |
| 10 | `b71bcf79d966` | 70 | A | F11 / S1 | 0.92 | 0.94 | 0.92 | none |
| 11 | `c79f951a2a43` | 80 | B | F13 / S1 | 0.90 | 0.95 | 0.91 | none |
| 12 | `e45f4580e44e` | 60 | B | F14 / S1 | 0.97 | 0.96 | 0.93 | none |
| 13 | `fa91ad8e2470` | 60 | B | F15 / S1 | 0.88 | 0.94 | 0.89 | none |
| 14 | `798bb23118c0` | 60 | B | F2 / S1 | 0.93 | 0.93 | 0.91 | none |
| 15 | `f5d48afb1a53` | 50 | B | F8 / S1 | 0.93 | 0.94 | 0.91 | none |
| 16 | `154c11f68ccc` | 70 | B | F1 / S1 | 0.95 | 0.94 | 0.91 | N-glyc. sequon |
| 17 | `1ce95bd8ae7b` | 60 | A | F4 / S1 | 0.88 | 0.92 | 0.87 | Asp-Pro |
| 18 | `65355a2ab485` | 80 | B | F7 / S1 | 0.91 | 0.94 | 0.91 | N-glyc. sequon |
| 19 | `dbb68e85db49` | 60 | B | F2 / S1 | 0.92 | 0.92 | 0.91 | none |
| 20 | `a0f0f8c0cd1c` | 60 | A | F8 / S1 | 0.90 | 0.93 | 0.90 | none |

Sequences, ipSAE values and per-design details: `docs/eswar/results/selection-final/selection.csv` and `selection.json`.

---

## 6. What went wrong along the way (kept on record)
- **AlphaFold hangs (about 2.5 GPU hours wasted).** First attempt used the public ColabFold MSA server and waited 52 minutes per chunk (and **sent about 12 unpublished binder sequences to a third-party server** before I stopped it). Next two attempts hung on the first GPU transfer. A stack dump showed ColabFold's default CUDA unified-memory setting hangs inside Modal; `--disable-unified-memory` fixes it. Records: `results/alphafold-attempts/`.
- **Repair command bug:** my first repair run passed a `--device` option upstream does not have. Fixed; a test now checks every option against the pinned upstream `run.py` (the old test's fake program had invented the option).
- **Earlier bugs found only on real GPUs:** an unpicklable torch version string, blocked storage hosts, Windows line endings breaking checksum checks, a Modal proxy package missing. All fixed and tested.
- A committed `__pycache__` folder was removed from the repository.
- One unexplained item: seed 43 reproduces across two A100 sizes, differs by one residue on an A10, and does **not** reproduce the binder in the team's earlier verified run.

---

## 7. Open items — what still needs a person
| # | Item | Who |
|---|---|---|
| 1 | Three reviews per design (novelty, independent structure review, screen review). The export guard stays closed until they are recorded. | team |
| 2 | **Stefanos's own AlphaFold check** of the final 20 | Stefanos |
| 3 | **Epitope decision.** All three hotspot sets are unreviewed; the final 20 sit on only two sites. | Samyuktha |
| 4 | Confirm the `joltz` source (found at `github.com/nboyd/joltz`, pinned commit `ed0f042`) | captain |
| 5 | Decide about the repaired variants (3 clean ones offered as replacements) | team |
| 6 | **Replace and revoke the Modal tokens** that were pasted in chat (they are not in this repository) | Eswar |
| 7 | After the epitope decision: regenerate on the agreed hotspots and repeat the selection (the AlphaFold gate is one flag) | Eswar |

---

## 8. How to reproduce or continue
**Get the files on top of the team's clone** (full instructions and every stage in `docs/eswar/RUN-GPU.md`):
```bash
git clone https://github.com/RishyanthReddy/helixbridge helixbridge-team
cd helixbridge-team
git config core.autocrlf false          # required on Windows: inputs are checked against a SHA-256
git remote add eswar https://github.com/eswarvajja123/helixbridge
git fetch eswar main
git checkout eswar/main -- pipeline docs/eswar requirements-eswar.txt modal_current.py modal_esmfold2.py modal_evaluate.py modal_mosaic.py modal_repair.py modal_alphafold.py
```
**Do not commit or push from `helixbridge-team`.** Run the free CPU checks first (Section 2 of the run guide), then Modal login and balance (Section 3). Modal needs a workspace with credit; weights are cached in a Volume (`helixbridge-model-cache`) and download once per workspace.

| Stage | What | Cost |
|---|---|---|
| A–C | prove Modal + RFD4 + Boltz-2; one design; a small budget-capped batch | GPU |
| D–E | qualify ESMFold2; score many designs with ESMFold2 + Boltz-2 | GPU |
| F–G | structure QC; diversity report | free (CPU) |
| H | merge batches into one pool | free (CPU) |
| J–L | novelty pre-screen; liabilities; shortlist and export guard | free (CPU) |
| M–N | repair tool; hand-off for the AlphaFold cross-check | free / person |
| O | Mosaic and the RFD4 comparison | GPU |
| P | native SolubleMPNN repair (Modal CPU container) | CPU |
| Q | AlphaFold-Multimer on a GPU; shuffled controls | GPU |

Every GPU engine is **budget-capped and resumable**: failures are kept on record and a re-run continues where it stopped.

**Where things are**
| Path | What |
|---|---|
| `docs/eswar/eswars-notes.md` | the five-part report for the team |
| `docs/eswar/gpu-runs.md` | the log of every GPU run, with numbers |
| `docs/eswar/gpu-coverage.md` | what has and has not run on a real GPU |
| `docs/eswar/implementation-plan.md` | the phased plan with status per task |
| `docs/eswar/hotspot-decision.md`, `hotspot-options/` | the three hotspot sets |
| `docs/eswar/RUN-GPU.md` | step-by-step commands for every stage |
| `docs/eswar/results/` | all result files (scores, QC, diversity, novelty, controls, AlphaFold, repair, final selection) |
| `pipeline/binder_campaign/`, `pipeline/scripts/`, `pipeline/tests/` | code and tests |
| `modal_*.py` | Modal apps (generation, ESMFold2, evaluation, Mosaic, repair, AlphaFold) |

**Pinned versions:** RFD4-Proteina `ddd934f7e6b86f7582ce487d5b19f6c526c644b5`; ESMFold2 (`esm==3.4.1.post1`, revision `69869f73…`); Boltz-2 (`boltz==2.2.1`); AlphaFold-Multimer v3 via ColabFold 1.6.3; Mosaic runtime `5937f37`, joltz `ed0f042`, Boltz fork `7ea718a`; LigandMPNN `26ec57ac`.

**Cost:** about $19 of Modal credit on the first account (billed cost $0); the later AlphaFold, repair and control runs were on a second account and were not measured. Roughly $0.10 per design from generation to both predictors.

**Security:** no token or secret is stored in this repository. Never paste Modal tokens into chat, documents or logs. Revoke any token that was pasted in chat.
