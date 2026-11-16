# Phase 1, Task 1.3 — Reference structure note (IL-7Rα, 21–239)

Updated 16 Nov 2026 (revision 2). Every statement is tagged **[tested]** (computed from files in this repo or from the downloaded 3DI3 file), **[read]** (taken from a team report, not rechecked) or **[inference]** (my reading, not confirmed).

**Correction to revision 1:** it said the shipped reference was "probably not 3DI3 chain B". That was too strong. After downloading 3DI3 and comparing directly, the two structures are the same fold. See below.

## What the competition asks for
- **[read]** UniProt P16871-1, residues 21–239 (Glu21–Asp239), reference structure 3DI3 chain B, C-terminal His-tag (`docs/official-targets-2026-11-13.json`).
- **[tested]** The current UniProt sequence is 219 aa, starting E and ending D (`phase1-data/il7ra-manifest.json`).

## What 3DI3 chain B contains
Compared with `pipeline/scripts/compare-reference-to-pdb.py`; full output in `phase1-data/reference-vs-3DI3.json`.
- **[tested]** The file's own DBREF maps PDB residues 1–219 to UniProt 21–239 (offset +20).
- **[tested]** Chain B has coordinates for **193 residues, UniProt 37–229, with no gaps inside that range**. Coordinates are missing for official positions 21–36 and 230–239 (26 positions).
- **[tested]** 3DI3 has **T at UniProt position 66**, where the current UniProt sequence has **I**. This is the only difference from current UniProt across the 193 residues. The file's SEQADV records list only expression tags and one engineered change at PDB 118 (UniProt 138); at 138 the file matches current UniProt.

## What generation actually uses
- **[tested]** The shipped reference (`examples/il7ra-rehearsal/target.cif`) has coordinates for UniProt **37–231 (195 residues)** and matches current UniProt at every position, including **I at 66**.
- **[tested]** Against 3DI3 chain B: 193 shared residues, **CA RMSD 0.78 Å** after superposition. The shipped reference has two extra residues (230, 231) that 3DI3 lacks.
- **[tested]** **24 positions of the official interval have no coordinates in the shipped reference: 21–36 and 232–239.** Generation cannot see them; they only enter at evaluation through the full sequence and MSA.
- **[tested]** Numbering is UniProt numbering, so hotspots do not change when the interval moves from 1–236 to 21–239.

## What the natural ligand covers (3DI3 chain A = IL-7)
- **[tested]** 19 residues of chain B have a heavy atom within 5 Å of IL-7: 51, 53, 77–79, 97–103, 122, 124, 158, 159, 211–213.
- **[tested]** **8 of the 10 provisional hotspots (100, 101, 102, 158, 159, 211, 212, 213) are in that interface.** The other two sit just outside it (161 at 6.1 Å, 214 at 9.5 Å). So the provisional epitope is essentially the **natural IL-7 binding site**.
- **[tested]** Residue 66 is **30.2 Å** from IL-7, so the I/T difference does not touch it.
- Limit: this is one definition (5 Å, heavy atoms, one crystal). Other partners (the common gamma chain, TSLP) were not examined.
- **Why it matters for Samyuktha's site choice:** a binder on this site would compete with IL-7. Whether the competition wants that, or an off-site epitope, is a biology decision.

## Glycans (3DI3 crystal; `phase1-data/glycan-context.json`)
- **[tested]** The target sequence has six N-X-S/T sequons: 49, 65, 151, 182, 232, 233. 3DI3 shows an **observed glycan at 49, 65 and 151** (two NAG sugars each). 182 is in the resolved region with **no** glycan seen. 232 and 233 lie outside the resolved region, so nothing is known there.
- **[tested]** **No provisional hotspot is within 8 Å of an observed glycan.** Closest: the N151 glycan to residues 158 and 161 (12 Å) and the N49 glycan to residues 158 and 159 (15 Å). The N65 glycan, which sits next to residue 66, is 31 Å from the hotspots.
- **Limits:** only the first two sugars are seen, so real glycans are larger and could reach further. Crystal glycosylation is not the occupancy in the assay protein. The 8 Å screen is my choice, not a published rule. **Any new hotspot set must be rechecked with `check-glycan-context.py`.**

## What this means
- The shipped reference is the same fold as the competition's named structure, within 0.78 Å, but it is **not the same coordinates**: residue 66 differs (I vs T) and it has two extra C-terminal residues.
- **[tested]** For the historical hotspots (100–102, 158–161, 211–214), the nearest CA-to-CA distance to residue 66 in the shipped reference is **28.2 Å**, so the residue-66 difference is unlikely to touch that epitope. **A new epitope choice must be rechecked against residue 66.**
- **[inference]** The BindCraft2 config calls it a "WT shipped reference" (`pipeline/config/il7ra-focused.json`), so it is probably a wild-type model derived from this domain. Its exact origin is still **not established** (its own provenance field says so).
- Samyuktha's target-biology work should decide whether designing against this reference, instead of 3DI3 itself, is acceptable.

## Other limits (no silent repair)
- **[read]** No glycans, membrane, receptor partners or assay construct are modelled.
- Nothing was repaired or filled in. Missing residues stay missing.
- 3DI3 is a complex with IL-7 (chain A); its interface is covered above. Other binding partners and glycan positions were not examined.
