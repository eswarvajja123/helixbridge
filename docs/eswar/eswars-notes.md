# Eswar's notes: candidate generation and design approaches (IL-7Rα minibinders)

*Draft prepared with Claude from the real result files in this repo (`docs/eswar/gpu-runs.md`, `docs/eswar/results/`). Check the numbers and edit the wording before you send it. Written 18 Nov 2026 for the team discussion.*

## 1. What I explored

- **The target.** IL-7Rα (UniProt P16871-1), residues 21-239 (219 aa), with a 1,022-row target-only MSA. The shipped reference core (residues 37-231) has the same fold as 3DI3 chain B (CA RMSD 0.78 Å over 193 shared residues), so it is a sound conditioning input.
- **Three hotspot sets** (`docs/eswar/hotspot-decision.md`): A, the provisional historical set (100-102, 158, 159, 161, 211-214); B, an exposed set on the natural IL-7 site (78, 79, 98, 100, 102, 103, 158, 159, 211, 212); C, an off-site exploratory set (170-179, 200). All three are still **unreviewed**.
- **One generator, RFD4-Proteina** at the pinned commit, with the team's qualified settings, binder lengths 50-80, and a model-chosen length range as one extra test. Mosaic was not explored: its `joltz` source is unknown.
- **Two independent predictors** for checking designs, ESMFold2 (the competition's screening model, approximated locally) and Boltz-2, plus the team's ipSAE scorer.
- **What a design must survive before shortlisting:** structure QC (clash, contact, hotspot contact, pose shift, the team's own checks), a near-duplicate gate, diversity of fold and site, a public-database novelty pre-screen, and a sequence-liability review.

## 2. What I tried

- Built the whole pipeline as tested code (425 tests, each module also checked by deliberately planting bugs) and ran every stage on real Modal GPUs, not only on a CPU with stand-ins.
- **About 100 designs** generated (first design, 6, then 60 across options A, B, C, then 32 more on A and B, plus one length-range design), **272 predictions** scored (ESMFold2 and Boltz-2 on 6 + 59 + 71 designs), on A100-40GB, A100-80GB, A100-PCIe, L40S, A10G and A10 GPUs.
- Checked ESMFold2 on Modal with all three kernel backends (they agree: ipTM 0.9492, 0.9492, 0.9493), and scored 39 designs twice to measure repeatability.
- Selected a shortlist of 20 by coverage of distinct folds and sites, not by score, and wrote the inputs for the AlphaFold cross-check.
- Total cost: about $13 of Modal credit (billed cost $0), roughly $0.10 per design from generation to both predictors.

## 3. What I found

1. **The pipeline works end to end on real GPUs**, and the first real runs exposed bugs no CPU test could (an unpicklable torch version string, blocked storage hosts, Windows line endings breaking checksum checks). All are fixed and tested.
2. **Hotspot option B is the best-behaved; option C does not work.** Share of designs passing every QC check: B 17 of 35 (49%), A 10 of 36 (28%), C 0 of 20. For option C both predictors put every binder about 55 Å from where the generator placed it, so neither independent model supports the designed off-site binding.
3. **The most common failure is Boltz-2 relocating the binder** (33 of 71 designs), while ESMFold2 mostly reproduces the generated pose. One design (…eaa213 in the first pool) got a high Boltz-2 ipSAE from a pose 19.6 Å from the designed one, so a score alone would have hidden the problem.
4. **Scores repeat well but not always:** typical ipTM changes between runs are about 0.002 (ESMFold2) and 0.003 (Boltz-2) across GPU types, and structure-QC outcomes were identical for all 39 designs scored twice. One weak design swung by 0.29 in ESMFold2, so single numbers for low-confidence designs are unreliable.
5. **Diversity:** designs are different folds (15 groups among the 27 that pass QC) but only three distinct binding sites: a fixed hotspot set gives different folds on the same site.
6. **Shortlist of 20:** 11 from option B, 9 from option A; lengths 50 (4), 60 (7), 70 (5), 80 (4); all 15 fold groups and 3 site groups covered; 8 carry at least one sequence flag (N-glycosylation sequon, odd cysteine, Asp-Pro). No design was flagged by the public-database novelty pre-screen.
7. **Seed 43 reproduces across two A100 sizes** (same binder), differs by one residue on an A10, and does **not** reproduce the binder recorded in the team's earlier verified run. I could not explain that difference.

**What this does not show:** that any design binds. Every design scores high on ESMFold2 and Boltz-2, and the team's own pilot found negative controls scoring as high as binders, so the scores cannot separate good from bad designs. The novelty check used a public database, not the organizers' private one. No design has been reviewed.

## 4. What I would improve

- **Put controls through the same pipeline** (a known binder, the natural IL-7 pair, scrambled negatives). Without them the QC limits and scores cannot be calibrated; the limits used are the team's provisional ones.
- **Settle the epitope** (Samyuktha). Option B looks best, but its hotspots and A's are unreviewed, and the 20 designs all sit on three sites.
- **Run the sequence repair for real** (the tool is built, the plan lists the 8 shortlisted designs and the exact positions; the real SolubleMPNN run was not done, see `docs/eswar/gpu-coverage.md`).
- **Make the scoring more robust:** score weak or borderline designs more than once, and keep both models' scores visible instead of one number.
- Run the new tools on a laptop (Windows) once; they were tested on Linux.

## 5. Next experiment

1. **AlphaFold cross-check of the 20** (Stefanos): inputs and a blank results sheet are in `docs/eswar/results/alphafold-handoff/`. This is the independent check the plan calls for, and nothing in the pipeline sets its review field.
2. **Controls run:** the same stages on a few known binders and negatives, to learn what the scores and QC flags mean on this target.
3. **Repair run:** SolubleMPNN on the 8 flagged shortlisted designs (temperature 0.1, seed 401, top two by MPNN confidence), then re-score and re-check the variants.
4. **After the epitope decision:** regenerate on the agreed hotspots, with a larger pool aimed at the chosen site, and repeat the selection.
5. A person records the novelty, structure and screen reviews, and only then can the team's export guard pass.
