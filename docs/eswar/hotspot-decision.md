# Hotspot decision memo — IL-7Rα (for Samyuktha and the team)

16 Nov 2026. Prepared by Eswar's pipeline work. **This is evidence for a decision, not the decision.** Nothing here is reviewed by the target-biology owner, and no option has been run through a generator.

## The question
Which residues should the generator aim at ("hotspots")? Phase 1 currently uses the historical ten, marked `provisional_historical`.

## Three options, measured the same way
All numbers come from 3DI3 chain B alone (`pipeline/scripts/evaluate-hotspot-options.py`; full table in `hotspot-options/hotspot-options.json`). Exposure is solvent-accessible area of the isolated chain. Ready-to-use conditioning files are in `hotspot-options/A|B|C/`.

| | **A — current (provisional)** | **B — exposed natural site** | **C — off-site (exploratory)** |
|---|---|---|---|
| Residues | 100 101 102 158 159 161 211 212 213 214 | 78 79 98 100 102 103 158 159 211 212 | 170 172–179 200 |
| In the natural IL-7 interface | 8 of 10 | 10 of 10 | 0 of 10 |
| Mean exposure (Å²) | 92.8 | 100.1 | 86.6 |
| Partly buried (<30 Å²) | 101 | none | none |
| Closest observed glycan | 12.0 Å | 12.0 Å | 22.0 Å |
| Spread (largest CA–CA distance) | 18.4 Å | **28.1 Å** | 14.9 Å |
| Rule | the historical set, unchanged | natural-interface residues with exposure ≥ 30 Å² and ≥ 10 Å from glycans; top 10 by exposure | best exposed patch ≥ 15 Å from IL-7 and glycans, excluding residues ≥ 220 (next to the C-terminal tag / membrane) |

## What the earlier experiments say (read from the team's reports)
- **[read]** At the natural site (the control arm), one of two seeds gave a strong, pose-stable design (ESMFold2 ipTM 0.863, pose shift 0.6 Å). The other seed failed. Evidence is two designs.
- **[read]** Four designs at two different off-site patches did **not** keep their intended pose (0 of 4). Those patches were not these residues (their exact lists are not in the repo), so Option C is untested, but the prior is not encouraging.
- **[read]** The natural-site hotspots were contacted by 8–10 of 10 in the native generated structures, so the generator does follow its hotspot list.

## My reading (inference, not tested)
1. **Keep the natural site as the working default.** It is the only region where this pipeline has produced a pose-stable complex.
2. **A versus B is the real choice inside that site.** B is more exposed (no buried residue) but its residues are spread over 28 Å, which a 60-residue binder may struggle to cover; A is tighter but includes the partly buried residue 101. This is cheap to test: same everything, only the hotspot list differs.
3. **Treat C as exploratory.** Aim there only after A/B, and only if the team wants a second, different site for diversity.

## What only the target-biology owner can answer
- Does the competition scoring care whether the binder **blocks IL-7**? The natural site (A, B) would compete with IL-7. Affinity ranking alone does not require it.
- Where do the **common gamma chain and TSLPR** contact IL-7Rα? I did not examine other partners; Option C might sit on one of them.
- Is any option **inaccessible in the assay** (the receptor is a stalk-anchored membrane protein in vivo; the assay uses a purified His-tagged fragment)?
- Do the observed **glycans** (N49, N65, N151) represent the assay protein? Only the first two sugars of each are seen.

## Limits
- Isolated chain, crystal glycans, one structure, one set of cutoffs (all in `thresholds` inside the JSON). Changing a cutoff changes B and C.
- Exposure and distances do not predict binding.
- The reference used for generation differs slightly from 3DI3 (see `phase1-reference-note.md`); the hotspot numbers are UniProt numbers, so they carry over.

## How to adopt a choice
Pick a list (any list within 37–231), then:

```bash
python scripts/build-il7ra-conditions.py --reference-dir ../examples/il7ra-rehearsal \
  --output-dir data/conditioning/il7ra-final --hotspots <comma-separated> \
  --binder-length 60 --hotspot-source "reviewed_by_<name>_<date>"
```

Then recheck the list with `check-glycan-context.py` and `compare-reference-to-pdb.py --hotspots …`.
