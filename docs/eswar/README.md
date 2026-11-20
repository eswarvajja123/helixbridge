# IL-7Rα minibinder candidates — Eswar's work (Changjiang Cup 2026)

**Role:** candidate generation and design approaches. **Target:** IL-7Rα (CD127), UniProt P16871-1, residues 21–239. **Binder:** 40–100 aa minibinder, at most 20 candidates.
**Written:** 18–19 Nov 2026 (final update 19 Nov, after the harder controls and the larger pool), from the real result files in `docs/eswar/results/`. Every number below can be traced to a file there.

> **Read this first.** This is a *proposal* for the team discussion, not a final submission. **Nothing here shows that any design binds IL-7Rα.**
> Every score is a computer model's opinion, and the team's own pilot found negative controls scoring as high as binders. What the work does show is that the
> pipeline runs end to end on real GPUs, that the designs are *consistent across three independent predictors*, and that shuffled sequences score clearly lower.
> The team's export guard is **closed on purpose**: it needs three human reviews per design, which nobody has done yet.

---

## 1. At a glance

| Question | Answer |
|---|---|
| Is the pipeline built and tested? | **Yes.** Phases 1–5, 529 automated tests (each module also checked by deliberately planting bugs). |
| Was it run on real GPUs? | **Yes**, on Modal (A100, L40S, A10G). RFD4, Boltz-2, ESMFold2, Mosaic, AlphaFold-Multimer, SolubleMPNN repair. |
| How many designs? | **120** in the final pool (RFD4: 72 first, 48 added later), 8 more from Mosaic. **32** pass every check, including AlphaFold and the decoy check. |
| What is the proposed shortlist? | **20 designs** (`selection-eswar-s5`), table in section 5. File: `docs/eswar/results/selection-final/selection.csv`. |
| Does anything prove binding? | **No.** See "What this does not show" below. |
| Can the shortlist be exported to the organizers? | **Not yet.** The export guard refuses until a person records the novelty, structure and screen reviews. |

### What the evidence does show
1. The designs fold into the intended complex in **three independent predictors** (ESMFold2, Boltz-2, AlphaFold-Multimer): for the final 20, ipTM is 0.77–0.97 in each.
2. **Shuffled copies of the same sequences do not.** Over 20 controls, 0 pass structure QC; AlphaFold ipTM is 0.28 median (max 0.61) against 0.91 for the designs.
3. **A known designed binder behaves the same way.** The published IL-7Rα antagonist 7OPB gets AlphaFold ipTM 0.90 and is placed within 1.2–1.35 Å of its crystal pose; our designs touch the same site.
4. **They are mostly specific.** Against two unrelated proteins (GFP and lysozyme) the designs score low (median ipTM 0.17 and 0.20); the 4 designs that confidently docked to an unrelated protein (ipTM 0.68–0.90) were removed from the shortlist.
5. The result does **not depend on one score**: designs were chosen for coverage of distinct folds and binding sites, not for the highest number.

### What this does not show
- That any design binds. A prediction of a complex is not an experiment. AlphaFold gives the **natural IL-7 / IL-7Rα complex an ipTM of 0.79**, lower than the designs' 0.91, so a high ipTM is not a measure of how strongly something binds.
- That the targeted epitope is the right one. All three hotspot sets are **unreviewed** (Samyuktha decides). What is known: options A and B lie on the site where a published designed antagonist (PDB 7OPB) binds, and our 20 designs touch most of that site (section 3.8).
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
- **Share of designs passing every structure-QC check** (all batches, 120 designs): **B 32 of 60 (53 %)**, **A 15 of 60 (25 %)**; option C (earlier, 20 designs): **0 of 20** — both predictors put every option-C binder about 55 Å from where it was designed, so neither independent model supports off-site binding. The 48 designs added later behaved the same way (B 15 of 24, A 5 of 24).
- Most common failure: **Boltz-2 relocating the binder** (33 of 71 designs). One design got a high Boltz-2 ipSAE from a pose 19.6 Å away from the designed one, so a score alone would have hidden the problem — that is why pose checks are part of QC.
- **Repeatability:** ipTM changes between runs are about 0.002 (ESMFold2) and 0.003 (Boltz-2) across GPU types; the three ESMFold2 kernel backends agree (0.9492, 0.9492, 0.9493); QC outcomes were identical for 39 designs scored twice. One weak design swung by 0.29, so single numbers for low-confidence designs are unreliable. A second full scoring of 71 designs agrees: Boltz-2 ipTM median difference 0.0002 (max 0.03); ESMFold2 median 0.0007, 95th percentile 0.09, max 0.34 (a weak design).

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

**Retry with doubled optimizer steps** (`results/mosaic-retry/`; soft steps 75 → 150 and sharp steps 50 → 100, the only change; same seed and lengths as the first attempt, 4 designs for each hotspot option; the change is recorded in every design's audit): **0 of 8 clean in structure QC** again; ESMFold2 ipTM median 0.32 (range 0.05–0.80; first attempt 0.23), Boltz-2 median 0.60 (first attempt 0.71); 1,106 GPU seconds per design (first attempt 695). One design (length 80) reached ESMFold2 ipTM 0.80 but fails QC on a close atom pair. More steps did not fix the disagreement between Boltz-2 and ESMFold2.

**AlphaFold as the third opinion on all 16 Mosaic designs** (`results/mosaic-retry/alphafold/`): it sides with ESMFold2, not with Boltz-2. First attempt: AlphaFold ipTM 0.29–0.79 (median 0.61), ipSAE at most 0.60; only 2 of 8 reach ipTM 0.78, and their ipSAE (0.54–0.60) is at the level of the wrong-target binder LCB1 (0.52), far below our designs (at least 0.65). **Retry with doubled steps: AlphaFold ipTM 0.15–0.36 (median 0.27), ipSAE at most 0.08, all 8 unsupported** — so the extra steps made the independent agreement *worse* while Boltz-2 (the model inside Mosaic's loop) stayed high. This confirms the mechanism: Mosaic overfits the predictor it optimizes through. Two of three predictors reject these designs, so none was added to the shortlist.

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
- Run on all designs that passed the earlier checks (47 of the 120): **36 pass the AlphaFold gate** (ipTM above every control, pose within the team's limits); the 11 that do not include the 4 above.
- This run is a **measurement**. It sets no review field. **Stefanos's own AlphaFold check is still needed.**

### 3.6 Repair of sequence liabilities (`results/repair/`)
Native SolubleMPNN (upstream `26ec57ac`, hash-checked weights `7af52d09…`) run in a Modal CPU container on the 8 shortlisted designs that had liabilities, per the team's lead101 protocol:
- **12 variants** produced (up to 2 per parent); all scored on ESMFold2 and Boltz-2; scores stay within 0.06 ipTM of the parent (one exception, below).
- Structure QC: **10 of 12 clean**, 2 flagged (`22e6bf-r1`, `d8ae7b-r1`). Novelty pre-screen: no hit for the 10 searched.
- **AlphaFold caught a failure the other two predictors missed:** both variants of `d8fd8a` (parent ipTM 0.87) fall to 0.31 and 0.36 with the binder 20–42 Å away, yet ESMFold2 and Boltz-2 scored them like the parent. Lesson: **rerun an independent predictor after any sequence change.**
- Only **3 variants** are clean on QC, novelty and AlphaFold together: `8e93b8-r1`, `0a592d-r1`, `0a592d-r2`. They are offered as replacements for their parents, **not** in the list.
- **`d8fd8a` is repaired with a targeted option** (`results/repair-d8fd8a/`): repairing only its N-glycosylation sequon (`--only-types sequon`, temperature 0.3, seed 403) gives 6 variants; 4 are clean on QC and the novelty pre-screen (the other 2 have one close atom pair under Boltz-2) and **all 4 pass AlphaFold** (ipTM 0.83–0.91, binder within 1.9 Å of the other predictors' poses; ESMFold2 ipTM 0.88–0.89, Boltz-2 0.91–0.92). Its Asn-Gly and Asp-Pro motifs are left as they are. These variants are not in the shortlist.
- **A wider, warmer second screen worked much better** (`results/repair-wide/`). The team's protocol (temperature 0.1) keeps proposing the same few sequences, so I re-ran SolubleMPNN at **temperature 0.3, seed 402** (a deliberate, recorded deviation) and kept up to 6 variants per parent, for the 7 flagged designs of the previous shortlist: 24 variants, all scored on both predictors; **19 clean on QC and the novelty pre-screen; 15 pass the AlphaFold gate, from 4 of the 7 parents**. All variants of `d8fd8a` (three liabilities at once) still fail under AlphaFold. These variants are **not** in the shortlist: they are available as replacements for their parents, and each needs its own reviews.

---

### 3.7 Harder controls: a real binder and unrelated proteins (`results/controls-hard/`)
Shuffled sequences are an easy negative, so two harder tests were run with AlphaFold-Multimer (same settings; every protein got its own deep alignment):
- **Positive control, IL-7 + IL-7Rα** (the natural pair): ipTM **0.79**, ipSAE 0.76. AlphaFold does recognise a real complex, but scores it *lower* than our designs (median 0.91) — designs are idealized helical bundles that AlphaFold finds easy, so **a high ipTM does not mean strong binding**. (IL-7 with no alignment of its own gave only 0.24, and the real target with no alignment cannot be folded at all, pLDDT about 38: alignments matter for every protein in a comparison.)
- **Decoy targets, GFP and chicken lysozyme:** every design was predicted with each. Medians: GFP 0.17, lysozyme 0.20 — most designs do not dock to unrelated proteins. **4 of the 36 AlphaFold-supported designs do**: `1febf6` (GFP, ipTM 0.90), `d8ae7b` (lysozyme, 0.86), `4dea48` (lysozyme, 0.82), `07c02f` (GFP, 0.68). They are excluded by a decoy rule (ipTM above the shuffled-control maximum, 0.61). Two further designs sit at 0.48 and are kept.

---

### 3.8 A known designed IL-7Rα binder as a real positive control (`results/known-binder/`)
PDB **7OPB** is a published, experimentally confirmed de novo antagonist of IL-7Rα (55 aa, 2.14 Å crystal structure; Cao et al., Nature 605:551, 2022). It was run through the same AlphaFold setup as our designs and compared with its crystal structure:
- **AlphaFold ipTM 0.898**, pLDDT 89.9 — the same range as our designs (median 0.91).
- **Pose reproduced:** the predicted binder is within **1.2–1.35 Å** (C-alpha RMSD) of the crystal binder after superposing the target, and 13–14 of its 14–15 crystal contact residues are predicted contacts.
- **Same site as our hotspots:** the crystal binder touches 9 of the 10 residues of option A and 7 of the 10 of option B. **Our 20 designs touch 9–13 of the 14 residues it touches (17 of 20 touch at least 10).** So the unreviewed hotspot sets and the designs lie on the site where a published antagonist is known to bind; this supports the site choice but is **not** the epitope decision (Samyuktha).
- Not done for lack of a suitable setup: antibody Fabs (PDB 6P50, 6P67) need a three-chain input the tools do not have; TSLP (5J11) needs its receptor TSLPR.

### 3.9 A designed binder for another target, and the known binder against decoys (`results/other-target-control/`)
- **LCB1** (PDB 7JZL; a published de novo binder of the SARS-CoV-2 spike, 55 aa, same design style and length as 7OPB) with IL-7Rα: **AlphaFold ipTM 0.78, ipSAE 0.52, pLDDT 87**. AlphaFold puts it in the **same IL-7 groove** our designs use (17 contacts, 9 of the 10 residues of option A and 9 of the 10 of option B). So AlphaFold's score can be confidently wrong for a binder that was not designed for this target, and the 0.61 floor set by shuffled sequences is **too low as a null**.
- Margin: our 20 final designs have AlphaFold ipTM 0.86–0.94 and ipSAE 0.65–0.82 — all above LCB1 on both. The shortlist is **unchanged** if the ipTM floor is raised to LCB1's level (0.78): the same 20 designs are selected. The margin is real but modest, and ipSAE separates the designs from LCB1 (0.52) at least as well as ipTM.
- The known binder 7OPB against GFP and lysozyme: ipTM **0.26 and 0.26** — a real binder does not dock to unrelated proteins.
- Not done: designs made *for another target* by this pipeline (needs a new target prepared). LCB1 is a binder for another target, but it was designed elsewhere.

---

## 4. How the final shortlist was chosen
1. **Eligible** = passes structure QC with no flags, public novelty pre-screen not flagged, scored by both predictors, liabilities reviewed, **AlphaFold gate** (ipTM **above every shuffled control, > 0.61**, and pose within 2.5 Å of the generator and 3.0 Å of the ESMFold2 / Boltz-2 refolds) **and decoy gate** (ipTM with GFP and lysozyme not above 0.61). The rules were set before looking at the results.
2. **32 of 120** designs are eligible (47 pass the earlier checks; 36 of those pass AlphaFold; 4 more are removed by the decoy gate).
3. **Picked by coverage, not score**: round-robin over fold groups, then an unused binding site, then fewer liability flags, then the best agreement between the two predictors. `pick_order` is the order of picking, **not a ranking**.
4. Result: 20 designs covering **18 of 18 fold groups and 2 of 2 binding sites** among the eligible ones. (The first list covered 15 folds and 3 sites; the AlphaFold gate had cut it to 13 folds and 2 sites; the larger pool brought the fold count back up, but all designs still sit on two binding sites.)

## 5. The proposed shortlist (`selection-eswar-s5`)
Hotspot option A = provisional historical, B = proposed by Eswar (both **unreviewed**). F = fold group, S = binding-site group. Scores are ipTM; "Decoy" is the highest ipTM with GFP or lysozyme (lower is better). Order is the order of picking, not quality. 15 designs use option B and 5 use option A; lengths: 4 at 50, 6 at 60, 6 at 70, 4 at 80; 7 carry a sequence flag.

| # | Design (`il7ra-rfd4-…`) | Length | Option | Fold / Site | ESMFold2 | Boltz-2 | AlphaFold | Decoy | Sequence flags |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `08f9233a8a90` | 70 | B | F2 / S1 | 0.95 | 0.95 | 0.93 | 0.32 (lyz) | none |
| 2 | `7ec112a2ac7a` | 60 | B | F11 / S1 | 0.96 | 0.96 | 0.93 | 0.18 (lyz) | none |
| 3 | `026e12ae07bc` | 60 | B | F1 / S1 | 0.95 | 0.95 | 0.92 | 0.48 (gfp) | none |
| 4 | `90b6d9399028` | 50 | A | F13 / S2 | 0.96 | 0.96 | 0.92 | 0.19 (lyz) | none |
| 5 | `5094a0d5dad8` | 50 | B | F4 / S1 | 0.96 | 0.95 | 0.94 | 0.19 (lyz) | none |
| 6 | `ee56c58e81b5` | 50 | B | F10 / S1 | 0.94 | 0.94 | 0.91 | 0.26 (lyz) | none |
| 7 | `f5206fc1de1f` | 70 | A | F8 / S2 | 0.84 | 0.77 | 0.87 | 0.42 (gfp) | Asp-Pro |
| 8 | `13af2bd8fd8a` | 80 | B | F3 / S1 | 0.89 | 0.92 | 0.87 | 0.22 (lyz) | N-glyc. sequon, Asn-Gly, Asp-Pro |
| 9 | `246e59d61754` | 60 | A | F5 / S1 | 0.94 | 0.95 | 0.91 | 0.17 (lyz) | none |
| 10 | `2e2ec88e93b8` | 70 | A | F6 / S2 | 0.92 | 0.92 | 0.91 | 0.23 (lyz) | odd Cys |
| 11 | `32fb546d40cc` | 70 | B | F7 / S1 | 0.95 | 0.94 | 0.90 | 0.21 (lyz) | N-glyc. sequon |
| 12 | `3dbfb374de02` | 70 | B | F9 / S1 | 0.93 | 0.94 | 0.90 | 0.16 (lyz) | N-glyc. sequon, Asn-Gly |
| 13 | `b25d020a592d` | 80 | B | F17 / S1 | 0.93 | 0.93 | 0.91 | 0.23 (gfp) | N-glyc. sequon |
| 14 | `b6eedc7da36b` | 80 | B | F19 / S1 | 0.96 | 0.94 | 0.93 | 0.22 (gfp) | none |
| 15 | `b71bcf79d966` | 70 | A | F20 / S2 | 0.92 | 0.94 | 0.92 | 0.48 (gfp) | none |
| 16 | `c79f951a2a43` | 80 | B | F21 / S1 | 0.90 | 0.95 | 0.91 | 0.17 (lyz) | none |
| 17 | `e45f4580e44e` | 60 | B | F22 / S1 | 0.97 | 0.96 | 0.93 | 0.29 (gfp) | none |
| 18 | `f49e60a8d7a3` | 50 | B | F23 / S1 | 0.94 | 0.95 | 0.92 | 0.20 (lyz) | N-glyc. sequon |
| 19 | `1fea9fc40149` | 60 | B | F2 / S1 | 0.96 | 0.93 | 0.92 | 0.25 (gfp) | none |
| 20 | `dbb68e85db49` | 60 | B | F11 / S1 | 0.92 | 0.92 | 0.91 | 0.17 (lyz) | none |

Group numbers come from the diversity report of the whole 120-design pool; numbers of groups with no eligible design are absent. `selection-eswar-s4` (the previous list, before the larger pool and the decoy check) is kept in `results/selection-s4/`; 10 of its 20 designs are in this list.

Sequences, ipSAE values and per-design details: `docs/eswar/results/selection-final/selection.csv` and `selection.json`.

---

## 6. What went wrong along the way (kept on record)
- **AlphaFold hangs (about 2.5 GPU hours wasted).** First attempt used the public ColabFold MSA server and waited 52 minutes per chunk (and **sent about 12 unpublished binder sequences to a third-party server** before I stopped it). Next two attempts hung on the first GPU transfer. A stack dump showed ColabFold's default CUDA unified-memory setting hangs inside Modal; `--disable-unified-memory` fixes it. Records: `results/alphafold-attempts/`.
- **Repair command bug:** my first repair run passed a `--device` option upstream does not have. Fixed; a test now checks every option against the pinned upstream `run.py` (the old test's fake program had invented the option).
- **Earlier bugs found only on real GPUs:** an unpicklable torch version string, blocked storage hosts, Windows line endings breaking checksum checks, a Modal proxy package missing. All fixed and tested.
- A committed `__pycache__` folder was removed from the repository.
- **Disclosure:** the first AlphaFold attempt used ColabFold's default alignment setting, which sends each chain to a public server; I chose that default without weighing that the binder sequences are unpublished. At least 3, and up to 12, designs were sent (`results/alphafold-attempts/EXPOSURE.md`); it cannot be undone. All later runs send no binder sequence anywhere.
- **More slips in the last round (all fixed, none changed a result):** the local `run` mode of the repair script still passed the removed `--device` option (found by reading, now tested); a launch environment missing `tmtools` made 8 generation designs fail until I installed it and resumed; two run-ID clashes on the Modal volume; I stopped every running app once, which also stopped a generation batch (it was resumable and finished); a first decoy run used single-sequence alignments after the alignment stage failed, because my launch script did not stop on error (it cost seconds and was discarded; the scripts now stop at the first failure).
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
| 7 | After the epitope decision: regenerate on the agreed hotspots and repeat the selection (the AlphaFold gate and the decoy gate are one flag each) | Eswar |
| 8 | Mosaic was retried with **twice the optimizer steps** (see 3.3) and is still negative, now also by AlphaFold (retry: ipTM 0.15–0.36); a real fix needs an independent predictor inside the loop, which the Mosaic framework does not offer here | team decision |
| 9 | **Decide whether designs whose sequences may have reached the public ColabFold server count as disclosed** (`results/alphafold-attempts/EXPOSURE.md`: 12 designs, at least 3 certain; 6 of them are in `s5`). An alternative shortlist without all 12 is `results/selection-s7/` | team |

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

**Cost:** about $19 of Modal credit on the first account (billed cost $0); the later AlphaFold, repair, control and larger-pool runs were on a second account (about 6.9 reported GPU hours for the second scoring alone) and were not measured in dollars. Roughly $0.10 per design from generation to both predictors.

**Security:** no token or secret is stored in this repository. Never paste Modal tokens into chat, documents or logs. Revoke any token that was pasted in chat.
