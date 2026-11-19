# Eswar's notes: candidate generation and design approaches (IL-7Rα minibinders)

*Draft prepared with Claude from the real result files in this repo (`docs/eswar/gpu-runs.md`, `docs/eswar/results/`). Check the numbers and edit the wording before you send it. Written 18 Nov 2026 for the team discussion; updated 19 Nov with the final results (controls, AlphaFold, repair, final shortlist).*

## 1. What I explored

- **The target.** IL-7Rα (UniProt P16871-1), residues 21-239 (219 aa), with a 1,022-row target-only MSA. The shipped reference core (residues 37-231) has the same fold as 3DI3 chain B (CA RMSD 0.78 Å over 193 shared residues), so it is a sound conditioning input.
- **Three hotspot sets** (`docs/eswar/hotspot-decision.md`): A, the provisional historical set (100-102, 158, 159, 161, 211-214); B, an exposed set on the natural IL-7 site (78, 79, 98, 100, 102, 103, 158, 159, 211, 212); C, an off-site exploratory set (170-179, 200). All three are still **unreviewed**.
- **One generator, RFD4-Proteina** at the pinned commit, with the team's qualified settings, binder lengths 50-80, and a model-chosen length range as one extra test. Mosaic, a second method that optimizes a sequence through Boltz-2, was also run (8 designs); its `joltz` source turned out to be `nboyd/joltz` at the team's pinned commit.
- **Two independent predictors** for checking designs, ESMFold2 (the competition's screening model, approximated locally) and Boltz-2, plus the team's ipSAE scorer.
- **What a design must survive before shortlisting:** structure QC (clash, contact, hotspot contact, pose shift, the team's own checks), a near-duplicate gate, diversity of fold and site, a public-database novelty pre-screen, and a sequence-liability review.

## 2. What I tried

- Built the whole pipeline as tested code (511 tests, each module also checked by deliberately planting bugs) and ran every stage on real Modal GPUs, not only on a CPU with stand-ins.
- **About 100 designs** generated (first design, 6, then 60 across options A, B, C, then 32 more on A and B, plus one length-range design), **272 predictions** scored (ESMFold2 and Boltz-2 on 6 + 59 + 71 designs), on A100-40GB, A100-80GB, A100-PCIe, L40S, A10G and A10 GPUs.
- Checked ESMFold2 on Modal with all three kernel backends (they agree: ipTM 0.9492, 0.9492, 0.9493), and scored 39 designs twice to measure repeatability.
- Selected a shortlist of 20 by coverage of distinct folds and sites, not by score. Then ran AlphaFold-Multimer on every eligible design and on 20 shuffled negative controls, ran the native SolubleMPNN repair (12 variants from 8 designs), and rebuilt the shortlist with an AlphaFold rule (final list `selection-eswar-s4`).
- Cost: about $19 of Modal credit on the first account (billed cost $0), roughly $0.10 per design from generation to both predictors; the later AlphaFold, repair and control runs were on a second account whose spend I did not measure. About 2.5 GPU hours were wasted on AlphaFold runs that hung (cause: ColabFold's default CUDA unified memory inside Modal; fixed).

## 3. What I found

1. **The pipeline works end to end on real GPUs**, and the first real runs exposed bugs no CPU test could (an unpicklable torch version string, blocked storage hosts, Windows line endings breaking checksum checks). All are fixed and tested.
2. **Hotspot option B is the best-behaved; option C does not work.** Share of designs passing every QC check: B 17 of 35 (49%), A 10 of 36 (28%), C 0 of 20. For option C both predictors put every binder about 55 Å from where the generator placed it, so neither independent model supports the designed off-site binding.
3. **The most common failure is Boltz-2 relocating the binder** (33 of 71 designs), while ESMFold2 mostly reproduces the generated pose. One design (…eaa213 in the first pool) got a high Boltz-2 ipSAE from a pose 19.6 Å from the designed one, so a score alone would have hidden the problem.
4. **Scores repeat well but not always:** typical ipTM changes between runs are about 0.002 (ESMFold2) and 0.003 (Boltz-2) across GPU types, and structure-QC outcomes were identical for all 39 designs scored twice. One weak design swung by 0.29 in ESMFold2, so single numbers for low-confidence designs are unreliable.
5. **Diversity:** designs are different folds (15 groups among the 27 that pass QC) but only three distinct binding sites: a fixed hotspot set gives different folds on the same site.
6. **First shortlist of 20 (`s3`):** 11 from option B, 9 from option A; all 15 fold groups and 3 site groups covered; 8 with a sequence flag. No design was flagged by the public-database novelty pre-screen. **This list was replaced (item 11).**
7. **Mosaic does not hold up against an independent predictor.** Eight Mosaic designs on the same target gave 0 clean designs in structure QC; ESMFold2 scored them ipTM 0.16 to 0.29 (median 0.23) while Boltz-2, which Mosaic optimizes through, scored them 0.52 to 0.87. RFD4 gave 10.3 clean designs per GPU-hour and 15 fold groups; Mosaic took about 5 times longer per design. Small sample and untuned settings, so this is evidence about this setup, not a verdict on the method.
8. **Seed 43 reproduces across two A100 sizes** (same binder), differs by one residue on an A10, and does **not** reproduce the binder recorded in the team's earlier verified run. I could not explain that difference.

9. **Negative controls (done).** 20 shuffled copies of the shortlisted sequences, scored and QC'd the same way: 0 of 20 pass QC (20 of 20 designs do); a design scores above a control in 100 % of ESMFold2 and 99.5 % of Boltz-2 comparisons. But Boltz-2 gave one shuffled sequence an ipTM of 0.86, so Boltz-2 alone is easily fooled; ESMFold2 separates better. The shortlist was already QC-filtered, so the QC comparison favours the designs.
10. **AlphaFold-Multimer (own target alignment, no server).** Shortlisted designs: ipTM median 0.91 (0.27 to 0.93); 16 of 20 agree with the generator and both other predictors on the binder pose, 4 do not (15 to 25 Å shifts; two with ipTM 0.27 and 0.53, inside the control range of 0.21 to 0.61). Shuffled controls: ipTM median 0.28, max 0.61; a design beats a control in 96 % of pairs. **Repair (12 variants from 8 designs):** variants keep their parent's ESMFold2/Boltz-2 scores to within 0.06 ipTM, but repair broke both variants of `d8fd8a` under AlphaFold (ipTM 0.87 → 0.31/0.36, binder 20 to 42 Å away) while ESMFold2 and Boltz-2 scored them like the parent. Only 3 variants (`8e93b8-r1`, `0a592d-r1`, `0a592d-r2`) are clean on QC, novelty and AlphaFold together.
11. **Final shortlist (`selection-eswar-s4`, AlphaFold gate).** Rule set before looking: ipTM above every control (> 0.61) and the pose within the team's QC limits. 22 of the 72 designs are eligible; 20 selected: 16 from `s3` plus 4 new (`f68ccc`, `2ab485`, `85db49`, `c0cd1c`); the 4 AlphaFold did not support are out. 12 from option B, 8 from option A; lengths 50 (4), 60 (8), 70 (4), 80 (4); 7 with a sequence flag. Coverage is lower than `s3`: 13 of 13 fold groups and 2 of 2 site groups among the eligible designs, against 15 and 3 before. The team's export guard still refuses the list until the three human reviews are recorded.

**What this does not show:** that any design binds. Every design scores high on ESMFold2 and Boltz-2, and the team's own pilot found negative controls scoring as high as binders, so the scores cannot separate good from bad designs (shuffled sequences did score lower here, which shows the scores respond to sequence, not that anything binds). The novelty check used a public database, not the organizers' private one. No design has been reviewed by a person, and the AlphaFold run is a measurement, not Stefanos's check.

## 4. What I would improve

- **Controls beyond shuffles.** Shuffled copies are an easy negative. A known binder, the natural IL-7 pair and designs for a different target would show what the scores mean; the QC limits are still the team's provisional ones.
- **Settle the epitope** (Samyuktha). Option B looks best, but its hotspots and A's are unreviewed, and the final 20 designs sit on only two sites.
- **Treat the repaired variants with care.** Only 3 of 12 are clean everywhere; repair can break a design that ESMFold2 and Boltz-2 still score well, so AlphaFold (or another independent model) must be rerun after any sequence change.
- **Retry Mosaic only with changes**, not the same settings: more optimizer steps or a different objective, and an independent predictor inside the loop; and score any new Mosaic design with ESMFold2 first.
- **Make the scoring more robust:** score weak or borderline designs more than once, and keep both models' scores visible instead of one number.
- Run the new tools on a laptop (Windows) once; they were tested on Linux. Rotate the Modal tokens that were pasted in chat (revoke the old ones).

## 5. Next experiment

1. **Stefanos's own AlphaFold check** of the final 20 (`docs/eswar/results/selection-final/`; inputs and a blank results sheet are in `docs/eswar/results/alphafold-handoff/` for the earlier list). The cloud AlphaFold run only recorded measurements and set no review field.
2. **A person records the novelty, structure and screen reviews** for each of the 20, and only then can the team's export guard pass.
3. **Decide about the repaired variants:** offer `8e93b8-r1`, `0a592d-r1`, `0a592d-r2` in place of their parents, or leave the list as it is.
4. **After the epitope decision:** regenerate on the agreed hotspots, with a larger pool aimed at the chosen site, and repeat the selection (the AlphaFold gate is one flag).
5. The captain confirms the `joltz` source (found at `nboyd/joltz`, pinned commit `ed0f042`).
