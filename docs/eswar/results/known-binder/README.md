# Positive control: a known designed IL-7Rα binder (PDB 7OPB)

**Source.** PDB 7OPB, "IL7R in complex with an antagonist": a de novo designed 55-residue binder (chains D/E/F) bound to the IL-7Rα
ectodomain (chains A/B/C, residues 36 to 231), 2.14 Å X-ray structure. Cao L, Coventry B, Goreshnik I, ... Baker D. "Design of protein-binding
proteins from the target structure alone." Nature 605, 551-560 (2022), doi 10.1038/s41586-022-04654-9, PMID 35332283.
Binder sequence: `SVIEKLRKLEKQARKQGDEVLVMLARMVLEYLEKGWVSEEDADESADRIEEVLKK`.

**What was done.** The binder and our 219-residue IL-7Rα construct were predicted with AlphaFold-Multimer in exactly the way the designs were
(own deep target alignment, binder as a single sequence, 2 models, 3 recycles; `alphafold-input.csv`, `run.json`, `c1.tar.gz`), then the
prediction was compared with the crystal structure (`known-binder-comparison.json`): the targets were superposed on their 196 shared
C-alpha atoms and the binder's C-alpha RMSD was measured; contact residues use a 4.5 Å heavy-atom cutoff. The comparison was run as a
one-off script using the tested functions in `pipeline/binder_campaign/pose_compare.py`; the script itself is not part of the tested tools.

**Result.**
- AlphaFold ipTM 0.898, pLDDT 89.9, ipSAE 0.748: the same range as our designs (median ipTM 0.91).
- Predicted vs crystal binder pose (three copies in the crystal): C-alpha RMSD 1.2 to 1.35 Å after superposing the target (target RMSD 0.56 to 0.62 Å).
- Contacts: the crystal binder touches 14 to 15 IL-7Rα residues (100, 102, 103, 122, 124, 125, 158, 159, 161, 210 to 214, plus 78/79 in some copies);
  13 to 14 of them are also predicted contacts (Jaccard 0.72 to 0.82).
- **Same site as our hotspots:** the crystal contacts include 9 of the 10 residues of hotspot option A and 7 of the 10 of option B.
- **Same site as our designs** (`site-vs-7opb.json`): the 20 shortlisted designs touch 9 to 13 of the 14 residues the crystal binder touches;
  17 of 20 touch at least 10 (median 11.5). Our designs have larger interfaces (15 to 31 contact residues against 14), so the Jaccard overlap is lower (median 0.45).

**What it does and does not show.** It shows the AlphaFold setup used here recognises a real designed binder and places it where the crystal
shows it, and that our hotspot sets and designs lie on the same site as a published, experimentally confirmed antagonist. It does not show that
any of our designs binds, and it is not the epitope decision (Samyuktha's).
