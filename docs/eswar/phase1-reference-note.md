# Phase 1, Task 1.3 — Reference structure note (IL-7Rα, 21–239)

Written 16 Nov 2026. Each statement is tagged **[tested]** (computed from files here), **[read]** (taken from a team report, not rechecked) or **[inference]** (my reading, not confirmed).

## What the competition asks for
- **[read]** Official construct: UniProt P16871-1, residues 21–239 (Glu21–Asp239), reference structure 3DI3 chain B, C-terminal His-tag (`docs/official-targets-2026-11-13.json`).
- **[tested]** The current sequence from UniProt is 219 aa, starting E and ending D.

## What generation actually uses
- **[tested]** The frozen reference (`examples/il7ra-rehearsal/target.cif`) has coordinates for UniProt positions **37–231 (195 residues)**, one contiguous run. Its sequence matches UniProt at all 195 positions.
- **[tested]** **24 positions in the official 21–239 interval have no coordinates: 21–36 (16) and 232–239 (8).** Generation cannot see them; they only enter at evaluation, through the full sequence and MSA.
- **[tested]** Numbering is UniProt numbering, so hotspots do not change when the interval moves from 1–236 to 21–239.

## Open problem: the reference is probably not 3DI3 chain B
- **[tested]** The shipped reference has **I at position 66**, which matches UniProt.
- **[read]** The team's reports say 3DI3 has **T at position 66** and supplies **193** mapped residues (`outputs/pipeline-execution-status.md`). The shipped reference has 195.
- **[tested]** The shipped file's own provenance field says its experimental or predicted origin is **not established**.
- **[inference]** The BindCraft2 config (`pipeline/config/il7ra-focused.json`) names a "WT shipped reference" (`hIL7RA.pdb`). The shipped reference is probably that wild-type model, not 3DI3. This is not confirmed.

**Why it matters:** designs are generated against a structure that may differ from the competition's named reference. The difference at residue 66 is a single residue, but the origin question is open. Samyuktha's target-biology work should confirm which structure is right.

**Not done:** I could not download 3DI3 to compare directly, because this session's network blocks the RCSB site. Checking the PDB file itself needs that access, or a download on your laptop.

## Other limits (no silent repair)
- **[read]** No glycans, membrane, receptor partners or assay construct are modelled in the reference.
- Nothing was repaired or filled in. Missing residues stay missing.
