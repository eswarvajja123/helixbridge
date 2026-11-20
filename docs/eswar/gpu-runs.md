# Real GPU runs

What has actually run on a GPU, with where each number came from. Nothing here is a binding result; every design is a rehearsal and not submission eligible.

## 17 Nov 2026 — team's historical pipeline (`modal_app.py`), workspace `eswarvajja10`, run on the laptop

| Step | Command | Outcome |
|---|---|---|
| Stage A prepare | `modal run modal_app.py --stage prepare` | RFD4 and Boltz-2 images built; public checkpoints cached on the volume. |
| Stage A generate | `modal run modal_app.py --stage generate --run-id eswar-seed43-01 --seed 43` | A100-SXM4-40GB, 57.6 s (65.8 s worker), peak 2,564 MiB. Binder (60 aa): `SIQEQLQTILGELHSEVTDSYTDEFKNLLEKALEENDLNVLLEMSNSTLRSKIKDVINSL` |
| Stage A evaluate | `modal run modal_app.py --stage evaluate --run-id eswar-seed43-01` | Boltz-2 + ipSAE: ipTM 0.7752, pTM 0.8467, complex pLDDT 0.8350, max ipSAE 0.3757 (49.8 s; 66.3 s worker). |

Provenance: these numbers were relayed from the laptop session's summary. The raw `generate.json` and `evaluate.json` have **not** been read here, so treat them as unverified until those files are read.

Comparison with the team's verified run (seed 43; ipTM 0.945, ipSAE 0.78, binder `SVQEKLQEIQASLHSKVAKENVDEFKSLLEKANEENDLSVLLQASNPELLSLIKEALNSL`): the binder differs, the scores are lower, and time and memory match. It is a different design from the same seed, not a reproduction. One sample cannot say whether this is ordinary variation between designs or hardware non-determinism.

Findings:
- Windows Git converted line endings to CRLF and broke the hash check on the reference input; fixed with `git config core.autocrlf false` and Unix line endings (now in RUN-GPU.md step 0).
- This was the team's historical target (1-236) and the team's `modal_app.py`, **not** the new `modal_current.py` (current target 21-239). Stages B to E have not run.

## 17 Nov 2026 — Task 2.4 with the new code (`modal_current.py`), run from the cloud session

Run id `eswar-cur-001`, seed 43, option A (hotspots 100-102, 158, 159, 161, 211-214; **unreviewed**, `provisional_historical`). Read from the real result file `generate-current.json`, not from a summary.

| Check | Result |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| Time | 72.6 s generation, 86.4 s worker; peak memory 2,564 MiB |
| Binder | 60 aa, `SIQEQLQTILGELHSEVTDSYTDEFKNLLEKALEENDLNVLLEMSNSTLRSKIKDVINSL` |
| Generator | RFD4-Proteina at the pinned commit ddd934f7, `recycle_strategy=thinking_high`, seed 43 |
| Record | candidate `il7ra-rfd4-9bf9771de68d`; `qc_reasons` empty; `submission_eligible` false; `screen_status` not_run; target 21-239 |
| Cost | about $0.05 of credit (credits line 2.01 to 2.06; billed cost $0.00) |

**The binder is identical to the one the laptop run produced with the team's `modal_app.py`** (A100-40GB, same seed). So seed 43 reproduced across two A100 sizes and across both code paths (the team's and mine). It is **not** bit-for-bit across GPU families: the Stage C batch's seed-43, 60-residue design ran on an A10 and differs from this binder at one of 60 positions (18: S instead of T). It also means the difference from the team's recorded verified design (`SVQEKLQE...`) comes from something else in how that run was made, not from GPU size; that is not yet explained.

Task 2.4's finish line (one on-target design on a GPU, cost logged) is reached. Nothing here says the design binds.

### How a cloud session reaches Modal (found 17 Nov)

The first attempts failed with "Could not connect to the Modal server". The cause was not a protocol limit, as an earlier environment note claimed: the Modal client needs `pip install 'modal[api-proxy-support]'` (it adds `python-socks`) to use the session's proxy, and the environment's network allowlist needs both `api.modal.com` and `input-plane.us-east.modal.com`. The second host is where function calls go; without it the image builds but the run is refused with `403 Forbidden`.

## 17 Nov 2026 — Task 2.5, Stage C: a budget-capped batch (run from the cloud session)

`modal run modal_current.py --stage batch --batch-id eswar-b1 --seeds 43,44,45 --lengths 50,60,70 --option A --max-designs 6 --max-gpu-seconds 900 --max-failures 2`. Read from the real `batch-report.json` and `candidates.json` (copies in `docs/eswar/results/`).

| Check | Result |
|---|---|
| Outcome | stopped by the `max_designs` cap: 6 of 9 planned designs done (seeds 43 and 44; seed 45 not started), 0 failures, 0 interrupted |
| GPU | seed 43 designs on an A10 (fallback; ~80-84 s each), seed 44 designs on an A100-80GB (~50-58 s each) |
| GPU seconds reported | 406.6 of the 900 cap |
| Near-duplicate gate | all six kept |
| Records | all six unreviewed, `qc_reasons` empty, `submission_eligible` false |
| Cost | about $0.23 of credit (credits line 2.09 to 2.32); billed cost $0.00 |

Task 2.5's finish line (a small batch with a budget cap and a report) is reached.

### Task 3.4 on these six real designs (generator poses; CPU, no GPU)

`diversity-report.py --records results/batch-eswar-b1/candidates.json --batch-dir results/batch-eswar-b1` (re-run later with all three sources: `docs/eswar/results/diversity-eswar-d3*`).
- **Fold:** the designs are largely different folds. Highest pairwise min-TM is 0.67 (one pair); the rest are 0.36-0.46. At TM 0.5 there are 5 groups (one pair joined); at 0.7 every design is its own group.
- **Site:** they all touch largely the same target region: site Jaccard 0.54-0.82 between most pairs, so at Jaccard 0.3 and 0.5 all six form one site group, and at 0.7 there are four groups.
- Reading: with one fixed hotspot set (option A), the generator gives different folds on one site. That is what fixed hotspots should do, but it also means this pool says nothing about other sites. Descriptive only; these are generator poses, not refolds, and nothing here is a binding result.
- The ESMFold2 and Boltz-2 sources were skipped on this first run (no evaluation folder yet); see the later section for all three sources.

Note: the structure paths stored inside `docs/eswar/results/*candidates.json` are paths in the cloud container that ran the batch; the structures themselves are not in this repo.

## 17 Nov 2026 — Task 3.1, Stage D: ESMFold2 on Modal (run from the cloud session)

`prepare`: the pinned checkpoint (biohub/ESMFold2 at revision 69869f73) downloaded to the volume and **every file's size and SHA-256 verified**: 11 files, 26.7 GB, 168 s. The ESMFold2 image built on Modal from the frozen 113-package list in 40 s, and its `Python.h` and C-compiler checks passed (so the missing-header failure the team hit on its cluster does not occur here).

`qualify` (`eswar-esm-003`; the verified 60-aa design against the current target 21-239 with its target-only MSA), read from the real result file:

| Check | Result |
|---|---|
| GPU / kernels | NVIDIA A100-SXM4-80GB, `fused` kernels worked |
| Time and memory | load 17.5 s, fold 27.1 s; peak 26,272 MiB reserved (25,419 MiB allocated) |
| Output | 219 + 60 = 279 residues, validated; ptm 0.885, ipTM 0.949, mean pLDDT 0.888 (raw model outputs) |
| Versions | torch 2.11.0+cu130, esm 3.4.1.post1, triton 3.6.0 |

This is a software check, not an accuracy result. Two runs failed first, and both exposed real bugs that the CPU stand-ins could not:
1. The run returned `torch.__version__`, a string subclass defined by torch; a laptop without torch cannot unpickle it, so a **successful fold looked like a failure** (`DeserializationError ... 'torch' module is not available`). Fixed: metrics are forced to plain Python types, and a test now checks this.
2. Any other failure inside the GPU function now comes back as readable text (type, message, traceback) instead of a torch exception object.

## 17 Nov 2026 — Task 3.2, Stage E: scoring with ESMFold2 and Boltz-2 (run from the cloud session)

`modal run modal_evaluate.py --stage evaluate --candidates results/batch-eswar-b1/candidates.json --eval-id eswar-e1 --predictors esmfold2,boltz2 --chunk-size 4 --max-calls 6 --max-gpu-seconds 1800`.

The first call ran on the GPU but its result could not be downloaded (the allowlist lacked Modal's storage hosts: `clobber.modal-storage.com`, `duot2hadvvkdh.cloudfront.net`). The engine recorded the four failures and stopped at its failure cap, as designed. After the hosts were allowed I resumed the **same** evaluation with `--retry-failed --max-failures 6`; the four failed attempts stay on record as history. Result (read from the real files): 5 calls, 12 of 12 pairs done, 0 failures remaining, 565 GPU seconds reported.

| design (seed, length) | ESMFold2 ipTM / ipSAE max | Boltz-2 ipTM / ipSAE max |
|---|---|---|
| ...53f79 (43, 50) | 0.942 / 0.850 | 0.931 / 0.832 |
| ...c93f6f (43, 60) | 0.934 / 0.810 | 0.754 / **0.328** |
| ...eaa213 (43, 70) | 0.891 / 0.728 | 0.875 / 0.731 |
| ...ba5da81f (44, 50) | 0.911 / 0.754 | 0.941 / 0.780 |
| ...875483 (44, 60) | 0.870 / 0.708 | 0.771 / **0.340** |
| ...3fbb19 (44, 70) | 0.887 / 0.732 | 0.911 / 0.802 |

- The two models disagree sharply on the two 60-residue designs (ipSAE about 0.8 and 0.7 from ESMFold2, about 0.33 from Boltz-2). Per the team's guidance that is a reason for a person to look, not a pass or fail.
- Every ESMFold2 ipTM is at least 0.87, so these numbers do not separate the designs. The team's pilot found negative controls scoring as high as binders, so none of this says a design binds. Nothing is ranked.

## 17 Nov 2026 — Task 3.3, Stage F: structure QC on the six real candidates (CPU)

`qc-candidates.py --evaluation results/eval-eswar-e1 --batch-dir results/batch-eswar-b1`: 3 flagged, 3 with no flags, 0 incomplete.

| design | status | what tripped |
|---|---|---|
| ...53f79, ...ba5da81f, ...3fbb19 | no_flags | |
| ...c93f6f | flagged | Boltz-2 pose is 15.8 Å from the generated pose (15.6 Å from ESMFold2's); one Boltz close pair below 2 Å |
| ...eaa213 | flagged | Boltz-2 pose is 19.6 Å from the generated pose (19.5 Å from ESMFold2's) |
| ...875483 | flagged | Boltz-2 pose 19.3 Å away; one close pair below 2 Å in the **generator's own** structure |

ESMFold2 kept the generated binder pose within 0.8-1.7 Å for all six; every flagged design is flagged because Boltz-2 puts the binder somewhere else. The flagged ...eaa213 is the notable case: Boltz-2 gives it a high ipSAE (0.73) from a pose 19.6 Å from where the generator put it, so the score alone would not have shown the problem. I recomputed three of the pose shifts with a separate from-scratch Kabsch fit; it agrees with the QC to 0.01 Å. The limits are the team's provisional ones, not validated thresholds.

## 17 Nov 2026 — Task 3.4, Stage G: diversity of the real designs (CPU)

`diversity-report.py` on the QC output, with all six designs (`docs/eswar/results/diversity-eswar-d3*`) and with only the three unflagged ones:
- **Fold:** designs are different folds in all three sources. Min-TM between designs is 0.35-0.68; at TM 0.5 the six designs form 5 groups (one pair joined), at 0.7 they are 6 singletons. The three unflagged designs are 3 singletons at every threshold.
- **Site:** the designs touch largely the same target region (site Jaccard 0.43-0.83): all six form one site group at Jaccard 0.3 and 0.5 (two groups for Boltz-2 at 0.5), four at 0.7. The unflagged three form one site group at 0.3 and 0.5.
- Reading: one fixed hotspot set gives different folds on one site. It says nothing about other sites, and the groups are descriptive only.

## Totals

Cost for everything above, from the billing summary's credits line: 2.01 to 3.28, about $1.27 of credit; billed cost stays $0.00. Includes the first (failed-to-download) scoring call, three ESMFold2 qualify attempts, the weight download and image builds.

## 17-18 Nov 2026 — the larger pool: Tasks 2.5, 3.2 to 3.4 and 5.1 to 5.3 on real candidates (run from the cloud session)

All numbers below are read from the real result files in `docs/eswar/results/`.

**Generation (Task 2.5, three batches of 20):** seeds 101-105 x lengths 50/60/70/80, one batch per hotspot option (A `provisional_historical`, B `proposed_by_eswar_unreviewed`, C `proposed_by_eswar_unreviewed_exploratory`). All three completed: 60 of 60 designs, 0 failures, 0 interrupted, 3,912 GPU seconds reported (A 1,792; B 1,135; C 985). GPUs used: A10G and A10 (batch A, mostly), A100-80GB and **L40S** (batch B), A100-80GB PCIe (batch C). Lengths 50 to 80 and options B and C ran on a GPU for the first time. A single extra design with a length range (`--length-min 50 --length-max 80`) also ran: the model chose 53 residues.

**Pool merge (new `pool.py`):** 60 designs, 59 kept. One design from batch B was dropped as a backbone duplicate of an earlier design (reason `backbone_tm`), the first time the near-duplicate gate dropped a real design.

**Scoring (Task 3.2):** ESMFold2 and Boltz-2 on the 59 kept designs: 118 of 118 predictions, 21 calls, 3,758 GPU seconds, 0 failures. The run was cut off after call 12 when the container paused overnight (state intact: all 59 ESMFold2 scores and 6 Boltz-2 scores saved, 6 marked started); resuming the same evaluation with `--retry-interrupted` finished it. That is the first real use of the interrupted-call path.

**Structure QC (Task 3.3):** 13 of 59 designs have no flags, 46 are flagged, 1 is incomplete (the merged duplicate was never scored).

| batch (hotspot option) | no flags | flagged | generated pose vs ESMFold2 / Boltz-2 (median) |
|---|---|---|---|
| A (provisional historical) | 6 of 20 | 14 | 1.3 Å / 3.4 Å |
| B (exposed natural site) | 7 of 19 | 12 | 0.9 Å / 1.5 Å |
| **C (off-site, exploratory)** | **0 of 20** | **20** | **54.9 Å / 59.3 Å** |

Option C is the clear result: both independent predictors place every one of those binders about 55 Å from where the generator put them, so the designed off-site binding is not supported by either model. That is a reason to drop option C, not a statement about the generator. Across all flagged designs the most common flags are a Boltz-2 pose far from the generated one (37 designs) and no contact with a design hotspot (21 for Boltz-2, 18 for ESMFold2). The limits are the team's provisional ones.

**Novelty pre-screen (Task 5.1):** public database = Swiss-Prot (576,073 entries) + unique PDB protein chains (180,353); 13 designs searched (the rest already carry QC reasons): 12 have no public hit at all, 1 has a weak hit, **none is flagged**. This says nothing about the organizers' private database.

**Liabilities (Task 5.2), all 60 designs:** 23 have an N-glycosylation sequon (N-X-S/T), 12 an Asp-Pro motif, 7 an odd number of cysteines, 1 an Asn-Gly motif; 23 have none of the four. Whether a sequon matters depends on how the binders will be expressed, which this project has not decided.

**Diversity (Task 3.4) on the 13 unflagged designs:** 7 fold groups at TM 0.5 (8 at 0.6, 10 at 0.7) and 2 site groups at Jaccard 0.5 (1 at 0.3, 7 at 0.7) in all three structure sources.

**Selection (Task 5.3):** the proposal holds **13** designs (all that pass every earlier check, so the limit of 20 is not reached): all 7 fold groups and both site groups are covered. `pick_order` is the order of picking, not a ranking. The team's export guard refuses the selection with "Required novelty/structure review is incomplete": for every design a person still has to record the novelty review, the independent (AlphaFold) structure review (Task 5.4) and the screen review. Nothing was set automatically.

Caveats: 12 of the 13 shortlisted designs come from options A and B, whose hotspot sets are still unreviewed (Samyuktha's decision); this is a small pool at one site; and none of this shows that any design binds.

## Cost

From the billing summary's credits line: 2.01 at the start of the first GPU run to 9.25 at the end of the pool run, about $7.2 of credit in total (the pool run alone, from 3.16, about $6.1, slightly under the $7 cap set for it); billed cost is still $0.00.

## 18 Nov 2026 — extending the pool to reach 20, and the final shortlist

Superseded by this section: the 13-design shortlist above (`selection-eswar-s2`) is kept as history. All numbers are from the real files in `docs/eswar/results/`.

**More designs:** 32 more on options A and B (seeds 106-109 x lengths 50-80, batches `eswar-qA` and `eswar-qB`): 32 of 32, 0 failures, 2,106 GPU seconds, about $1.8. Option C was dropped from the pool (all 20 of its designs failed QC; they stay in the earlier results as a negative result). Merged pool (`pool-eswar-p2`): 72 designs, 71 kept by the near-duplicate gate (the same one batch-B backbone duplicate).

**Scoring (`eswar-e3`, 71 designs x 2 predictors = 142 predictions):** 25 calls, 4,424 GPU seconds, 0 failures remaining. It was interrupted twice by the environment (a worker restart dropped the proxy during Boltz-2 calls; the engine stopped at its failure cap with every finished score saved) and resumed with `--retry-failed --retry-interrupted --max-failures 12`; GPUs: A100-80GB, A100-PCIe, L40S, A10. Cost of this round about $6 of credit (more than my $4 estimate, because of the interrupted calls and re-scoring the 39 older designs on purpose).

**Reproducibility, measured for the first time (39 designs scored in both `e2` and `e3`, same seed and settings, mostly different GPU types):**

| predictor | median change in ipTM | 90th percentile | largest |
|---|---|---|---|
| ESMFold2 | 0.0015 | 0.025 | 0.29 (one design; ipTM 0.445 -> 0.153, a weak design that Boltz-2 also scores low) |
| Boltz-2 | 0.0031 | 0.015 | 0.020 |

Typical scores repeat to the second decimal across GPU types, but a weak design can swing a lot between runs, so a single ESMFold2 number for a low-confidence design should not be trusted. **Structure-QC status was identical in both runs for all 39 designs** (26 flagged, 13 not).

**Structure QC (71 scored designs):** 27 no flags, 44 flagged, 1 incomplete. Pass rate by hotspot option: **B (exposed natural site) 17 of 35 (49%)**, **A (provisional historical) 10 of 36 (28%)**, C 0 of 20 (earlier run). Most common flags: Boltz-2 puts the binder far from the generated pose (33 designs) and ESMFold2 and Boltz-2 disagree on the pose (30).

**Novelty pre-screen (public Swiss-Prot + PDB):** 27 designs searched; 25 have no public hit, 2 a weak hit; none flagged.

**Liabilities (all 72):** 24 have an N-glycosylation sequon, 11 an Asp-Pro motif, 6 an odd cysteine count, 2 an Asn-Gly motif; 34 have none.

**Diversity (27 unflagged designs):** at TM 0.5 there are 15 fold groups in the generator poses (14 in each refold source) and 3 site groups at Jaccard 0.5 in all three sources.

**Shortlist (`selection-eswar-s3`): 20 designs, the target.** It covers all 15 fold groups and all 3 site groups; 11 come from option B and 9 from option A; lengths: 4 at 50, 7 at 60, 5 at 70, 4 at 80; 8 carry at least one liability flag (the repair plan, `repair-plan-eswar-r2.json`, lists exactly which positions would be redesigned; the real SolubleMPNN run is not done). `pick_order` is the order of picking, not a ranking. The export guard still refuses until a person records the three reviews for every design. Inputs for the AlphaFold cross-check are in `docs/eswar/results/alphafold-handoff/` (inputs and a blank results sheet only).

## Cost summary

Credits line from the first GPU run (2.01) to the end of this round (15.09): about $13.1 of credit in total across about 100 generated designs and 272 predictions (6 + 59 + 71 designs, two predictors each); billed cost is still $0.00. Roughly $0.10 per design from generation through both predictors.

## 18 Nov 2026 — Phase 4: Mosaic on the current target, and RFD4 vs Mosaic per GPU-hour

All numbers are from the real files in `docs/eswar/results/mosaic/` and `docs/eswar/results/`.

**Source of the missing `joltz`:** `github.com/nboyd/joltz` at commit `ed0f042` is exactly the commit the team pinned; with `mosaic` (escalante-bio, `5937f37`) and the Boltz fork (escalante-bio, `7ea718a`) all three pinned commits are fetchable and MIT-licensed. The captain should still confirm these are the intended sources.

**Task 4.1 — Mosaic on the current target (done, on a real GPU).** `modal_mosaic.py` clones the three sources at the pinned commits (refusing any other or modified revision), installs the team's scoped stack, and runs the team's optimizer settings unchanged on the current construct (UniProt 21-239, first 128 MSA rows, hotspot options as indices from the construct start). The image built and passed its Boltz-2 and ProteinMPNN imports on the first try (46 s); the resolved 124 packages were frozen (`docs/eswar/mosaic-image-constraints.txt`) and the frozen image rebuilt in 57 s. Smoke test (one loss and gradient evaluation on the real Boltz-2 weights): loss 15.05, gradient norm 5.64, finite, 144 s. First design (option A, 60 aa, seed 101): 555 s on an A100-80GB (60 s load, 434 s optimizing, 49 s predicting); the binder touches 27 target residues including 8 of the 10 hotspots with no close atom pairs, but its own Boltz-2 interface PAE averages 16 Å and the binder pLDDT 0.60. The structure writer's reported ipTM was exactly 0.0, which I treat as an unreliable field, not a measurement (the independent scores below are the measurement).

**Task 4.2 — timeout (done).** The function timeout is one hour. Eight designs took 493 to 835 s (mean 622 s) of worker time on A100-80GB (7) and L40S (1); the old 900 s limit would have been close to the slowest, and the team's 16 minutes on an L40S would have exceeded it.

**Task 4.3 — comparison at matched GPU time (done, with a small Mosaic sample).** Eight Mosaic designs (options A and B, lengths 50/60/70/80, seed 102): 8 of 8 produced, 0 failures; one attempt was cut off by a container restart and retried (9 attempts), so the batch cap had to be raised by one to finish the last design. Scored with ESMFold2 and Boltz-2 (16 of 16 predictions, 587 GPU s, 0 failures) and put through the same structure QC.

| | RFD4 (options A and B) | Mosaic (options A and B) |
|---|---|---|
| designs | 72 | 8 |
| GPU seconds per design (generation + scoring) | 131 | 695 |
| designs per GPU-hour | 27.4 | 5.2 |
| clean in structure QC | 27 of 71 (38%) | **0 of 8** |
| clean per GPU-hour | 10.3 | 0 |
| clean if every Boltz-2 flag is ignored | 60 of 71 (85%) | **0 of 8** |
| distinct fold groups among the clean designs | 15 | 0 |
| ESMFold2 ipTM (min / median / max) | 0.153 / 0.917 / 0.966 | 0.156 / 0.229 / 0.289 |
| Boltz-2 ipTM (min / median / max) | 0.254 / 0.889 / 0.968 | 0.517 / 0.712 / 0.865 |

At Mosaic's GPU budget (5,560 s) the measured RFD4 rate would give about 42 designs, 16 clean, and around 9 fold groups (a scaling, not a run).

**What the Mosaic result shows.** Mosaic optimizes THROUGH Boltz-2, and the designs look good to Boltz-2 (median ipTM 0.71) but not to ESMFold2, the independent model: ESMFold2 gives them ipTM 0.16 to 0.29 and ipSAE at most 0.017, and places the binder 8 to 31 Å from where Mosaic's own structure put it. All 8 are QC-flagged even when the Boltz-2 flags are ignored. That is the classic result of optimizing against one predictor: the sequences are specific to that model. The same ESMFold2 pipeline gives RFD4 designs a median ipTM of 0.92, so the independent check is not simply harsh. It does not mean Mosaic can never work here.

**Limits of this comparison.** Only 8 Mosaic designs (the tool prints counts with every figure); one seed per length; the optimizer settings are the team's, untuned (125 steps, 128-row MSA, MPNN-recovery weight 5), and the designs are low-complexity and rich in lysine and glutamate; two hotspot sets only. Seconds of the interrupted attempt are not in the reported GPU seconds (Modal billed it). Nothing here says any design binds.

**Cost:** Phase 4 (image builds, smoke, first design, 8 designs with the retry, scoring): credits line 15.00 to 21.05, about $6.1. Total across the whole project so far: about $19 of credit (2.01 to 21.05); billed cost $0.00.

## Shuffled negative controls and the AlphaFold cross-check (Task 5.4, Phase 3 calibration)

- **Controls (done, GPU).** 20 shuffled copies of the shortlisted sequences (`il7ra-shuf-*`) scored by ESMFold2 and Boltz-2 (about 1,332 GPU seconds reported, 20 of 20 done each, no failures) and QC'd the same way. 0 of 20 controls pass QC; 20 of 20 shortlisted designs do. A design scores above a control in 100 % of ESMFold2 ipTM/ipSAE comparisons and 99.5 % of Boltz-2 ones. ESMFold2 median ipTM: controls 0.18 (max 0.64). Boltz-2 median ipTM: controls 0.61 (max 0.86), so Boltz-2 alone is easily fooled. The shortlist was already QC-filtered, so the QC comparison is biased toward the designs. Files: `results/controls/`.
- **AlphaFold-Multimer (done, GPU).** Attempt 1 used the public ColabFold MSA server: two chunks each hit the 52-minute timeout waiting in its queue and the third was stopped by hand (it would also have sent unpublished binder sequences to a third party; about 12 designs' sequences were submitted before it was stopped). Attempts 2 to 4 used our own target alignment and hung on the first GPU transfer; a stack dump showed ColabFold's default CUDA unified memory is the cause inside Modal. About 2.5 GPU hours were spent on the failures. With `--disable-unified-memory` all 20 shortlisted designs ran in 5 chunks (about 190 to 250 s per chunk of 4, A100 and L40S) on the `rishyanthreddy101` account, and the 20 shuffled controls the same way. Designs: ipTM median 0.91 (0.27 to 0.93); 16 of 20 agree with the other predictors on the pose. Controls: ipTM median 0.28 (max 0.61). A design beats a control in 96 % of pairs; two designs sit inside the control range. Files: `results/alphafold/` and `results/alphafold-attempts/`.

## Native SolubleMPNN repair (Task 5.2, optional), 18 Nov

Run in a Modal CPU container on the `rishyanthreddy101` account after the owner's go-ahead: upstream LigandMPNN at `26ec57ac976ade5379920dbd43c7f97a91cf82de` with `solublempnn_v_48_020.pt` (sha256 `7af52d09...a6e31a`, verified). Attempt 1 failed in the container (an invalid `--device` option in our command, my bug, now fixed and covered by a test); attempt 2 ran for 8 designs at about 10 s each. 12 variants were produced from 8 parents, scored by ESMFold2 and Boltz-2 (12 of 12 done, about 707 reported GPU seconds; this needed the ESMFold2 and Boltz-2 weights to be downloaded once into the new account) and QC'd against their parents' backbones: 10 clean, 2 flagged. Variants keep their parents' scores to within 0.06 ipTM except `d8ae7b-r1` (Boltz-2 ipTM 0.44 vs 0.92 for the parent). Not yet done for the variants: novelty pre-screen and AlphaFold. Files: `results/repair/`.

## AlphaFold and novelty on the 12 repaired variants, and the AlphaFold gaps in the shortlist (18 Nov)

**Variants.** Novelty pre-screen (public database only): 10 of 12 no hit, 2 not searched because structure QC had flagged them. AlphaFold-Multimer (own-alignment MSA, same settings as the shortlist, 3 chunks, about 4 minutes each): the repair kept AlphaFold support for `8e93b8-r1`, `0a592d-r1` and `0a592d-r2` (ipTM 0.90 to 0.91, pose within 1.6 A of the generator and both refolds, no close pairs), and for `22e6bf-r1` (ipTM 0.88; but QC flagged its Boltz-2 pose 2.83 A from the generator) and `d8ae7b-r1` (ipTM 0.85; QC flagged, Boltz-2 places the binder 22 A away). **Repair broke both variants of `d8fd8a`**: its parent had AlphaFold ipTM 0.87 and a matching pose, but the variants have ipTM 0.31 and 0.36 and the binder 20 to 42 A away, even though ESMFold2 and Boltz-2 scored them like the parent. So ESMFold2 and Boltz-2 alone missed this. Variants of the three parents AlphaFold already did not support (`6e8955`, `64bcd4`, `21d726`) are still unsupported (ipTM 0.24 to 0.28 for two of them; `21d726` has the same 22 A pose shift). Only `8e93b8-r1`, `0a592d-r1` and `0a592d-r2` are clean on structure QC, novelty pre-screen and AlphaFold together. Files: `results/repair/alphafold/` and `results/repair/novelty/`.

**Shortlist gaps (selection `s3`, unchanged).** AlphaFold does not support 4 of the 20 shortlisted designs: `47761f` (binder 25 A from the other predictors' pose), `6e8955` (ipTM 0.27, 16 A), `64bcd4` (ipTM 0.53, 15 A) and `21d726` (22 A). The selection tool does not use AlphaFold, so they are still in `selection-eswar-s3`; replacing them is a decision for the team, not something I changed.

## Final shortlist `selection-eswar-s4` (AlphaFold gate), 18 Nov

The selection tool now has an optional AlphaFold gate (`select-finalists.py --alphafold-results ...`). Rule, set before looking at the result and justified by the shuffled controls run through the same AlphaFold settings: ipTM **above** every control (the controls' maximum was 0.61), binder pose within 2.5 A of the generator and within 3.0 A of the ESMFold2 and Boltz-2 refolds (the team's own QC limits), and AlphaFold must have been run on the design. AlphaFold was run on all 27 designs that passed every earlier check (the 20 of `s3` plus 7 more, `results/alphafold-rest/`).

**Result:** 22 of the 72 designs are eligible (27 passed the earlier checks; 5 fail the AlphaFold gate: `47761f`, `6e8955`, `64bcd4`, `21d726`, `bc6f6d`). The 20 selected are 16 from `s3` and 4 new (`f68ccc`, `2ab485`, `85db49`, `c0cd1c`). The 4 designs AlphaFold did not support are gone. Coverage is lower than `s3`: 13 of 13 fold groups and 2 of 2 site groups among the eligible designs, against 15 and 3 before, because the dropped designs were the only members of 2 fold groups and 1 site group. 12 designs come from the unreviewed option B hotspots and 8 from option A; lengths: 4 at 50, 8 at 60, 4 at 70, 4 at 80. 7 carry at least one liability flag. The 3 repaired variants that are clean on QC, novelty and AlphaFold (`8e93b8-r1`, `0a592d-r1`, `0a592d-r2`) are offered as replacements for their parents only; they are not in the list. The export guard still refuses the list (`Required novelty/structure review is incomplete`), because the three human reviews are not set: that is correct. Files: `results/selection-final/`, `results/alphafold-rest/`.

## Harder controls, wider repair, larger pool, final shortlist `s5` (19 Nov)

- **Positive control and decoys (AlphaFold-Multimer).** IL-7 (UniProt P13232, residues 26 to 177) with IL-7Ralpha: ipTM 0.79, ipSAE 0.76 when IL-7 has its own deep alignment (0.24 without). GFP (P42212) and chicken lysozyme (P00698, residues 19 to 147) as decoy targets, each with its own deep alignment built by the new `--stage msa` (only the decoy sequence is sent to the public server): the 36 AlphaFold-supported designs were tested with both. Median ipTM over the first 20 designs tested: GFP 0.17, lysozyme 0.20. Four designs dock confidently to a decoy and are excluded: `1febf6` (GFP 0.90), `d8ae7b` (lysozyme 0.86), `4dea48` (lysozyme 0.82), `07c02f` (GFP 0.68). A single-sequence version of this test was tried first and discarded: the real target cannot be folded without its alignment (pLDDT about 38), so it proved nothing. Files: `results/controls-hard/` (`decoys-1/` and `decoys-2/` are the two runs).
- **Wider, warmer repair (`results/repair-wide/`).** SolubleMPNN at temperature 0.3, seed 402 (a deviation from the team's 0.1 / 401, recorded in every variant), up to 6 variants per parent, for the 7 flagged designs of the previous shortlist: 24 variants; all scored on ESMFold2 and Boltz-2 (24 of 24; 6 Boltz-2 calls first failed on run-ID collisions and were retried); 19 clean on QC and the novelty pre-screen; 15 pass the AlphaFold gate, from 4 of 7 parents (all variants of `d8fd8a` fail).
- **Larger pool (`results/pool3/`).** 48 more RFD4 designs (options A and B, seeds 110 to 115, lengths 50 to 80; 8 of the designs first failed because the launch environment lacked `tmtools` and were resumed); merged with the earlier batches: 120 designs, 118 kept by the near-duplicate gate. All 118 scored again on ESMFold2 and Boltz-2 (118 of 118, about 1.9 reported GPU hours): for the 71 designs that were scored before, Boltz-2 ipTM differences are median 0.0002 (max 0.03) and ESMFold2 median 0.0007 (95th percentile 0.09, max 0.34). Structure QC: 47 clean, 71 flagged; of the 48 new designs, option B 15 of 24 clean, option A 5 of 24. Novelty pre-screen: no hit for the 47. AlphaFold on the 20 newly eligible designs (5 chunks): 36 of the 47 eligible designs pass the AlphaFold gate. Decoys for the 16 not yet tested (32 predictions).
- **Final shortlist `selection-eswar-s5`** (QC, novelty, AlphaFold gate, decoy gate): 32 eligible, 20 selected, 18 of 18 fold groups and 2 of 2 site groups among the eligible; 15 designs from option B and 5 from A; lengths 50 (4), 60 (6), 70 (6), 80 (4); 7 carry a sequence flag; AlphaFold ipTM median 0.92. 10 of the 20 designs were also in `s4` (kept in `results/selection-s4/`). The export guard still refuses until the three human reviews are recorded.
- **Not done: Mosaic retry.** Mosaic's optimizer settings are the team's and are fixed in several places; more steps through Boltz-2 would likely overfit Boltz-2. A real fix needs an independent predictor in the optimization loop, which the Mosaic framework does not offer here.

## Known designed binder, PDB 7OPB (20 Nov)

AlphaFold-Multimer (the same setup as the designs) on the published designed IL-7Ralpha antagonist 7OPB (55 aa) with our 219-residue construct: ipTM 0.898, pLDDT 89.9, ipSAE 0.748, 90 s on an A100. Against the crystal structure (three copies): binder C-alpha RMSD 1.2 to 1.35 A after superposing the target (target RMSD 0.56 to 0.62 A); 13 to 14 of the 14 to 15 crystal contacts are predicted contacts. The crystal binder touches 9 of 10 residues of hotspot option A and 7 of 10 of option B, and the 20 shortlisted designs touch 9 to 13 of its 14 contact residues. Files and the source citation: `results/known-binder/`. New tested tool: `pipeline/binder_campaign/pose_compare.py`.

## Controls with another target's binder, targeted repair, Mosaic retry, disclosure record (20 Nov)

- **LCB1 (PDB 7JZL chain E; designed spike binder) with IL-7Ralpha:** AlphaFold ipTM 0.779, ipSAE 0.517, pLDDT 87; contacts 17 IL-7Ralpha residues, including 9 of 10 of hotspot option A and 9 of 10 of option B. All 20 shortlisted designs are above it on both scores (ipTM >= 0.86, ipSAE >= 0.65); with the floor raised to 0.78 the selection is the same 20 designs (`selection-eswar-s6`, not saved separately). 7OPB with GFP and with lysozyme: ipTM 0.257 and 0.258. Files: `results/other-target-control/`.
- **`d8fd8a` sequon-only repair:** `modal_repair.py --only-types sequon --temperature 0.3 --seed 403`: 6 variants; 6 of 6 scored on ESMFold2 and Boltz-2 (about 380 reported GPU seconds); QC 4 clean; novelty pre-screen no hit; AlphaFold 4 of 4 pass (ipTM 0.83 to 0.91). Files: `results/repair-d8fd8a/`.
- **Mosaic retry:** `modal_mosaic.py --soft-steps 150 --sharp-steps 100` (the only change; recorded in each audit as `settings_deviations_from_team`), seed 102, lengths 50/60/70/80, options A and B: 8 of 8 designs made (8,846 reported GPU seconds, 1,106 per design); all scored by ESMFold2 and Boltz-2; structure QC 0 of 8 clean; ESMFold2 ipTM median 0.32 (0.05 to 0.80), Boltz-2 median 0.60 (0.21 to 0.85). Files: `results/mosaic-retry/`.
- **Disclosure record:** `results/alphafold-attempts/EXPOSURE.md`; alternative shortlist without the 12 possibly disclosed designs: `results/selection-s7/` (24 eligible, 20 selected, 14 also in `s5`, 14 of 14 fold groups, 2 of 2 sites; built with the new `select-finalists.py --exclude`).

## AlphaFold on all 16 Mosaic designs (20 Nov)

AlphaFold-Multimer (same setup as the designs) on the 8 Mosaic designs of the first attempt and the 8 of the retry with doubled steps (4 chunks, about 280 s each). First attempt: ipTM 0.29, 0.35, 0.37, 0.52, 0.69, 0.73, 0.78, 0.79 (median 0.61), ipSAE at most 0.60. Retry: ipTM 0.15 to 0.36 (median 0.27), ipSAE at most 0.08. The retry designs are rejected by AlphaFold as by ESMFold2; only Boltz-2, which Mosaic optimizes through, rates them. For comparison the shortlisted RFD4 designs have AlphaFold ipTM at least 0.86 and ipSAE at least 0.65. Files: `results/mosaic-retry/alphafold/`.
