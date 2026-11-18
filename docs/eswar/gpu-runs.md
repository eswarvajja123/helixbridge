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
