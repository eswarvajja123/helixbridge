"""Compare a shipped reference structure with one chain of a PDB entry (CPU only).

Reads the PDB file's own DBREF record to map PDB numbers to UniProt numbers, then reports
coverage, gaps, identity differences against a UniProt sequence, and the CA RMSD over the
shared residues. Downloads nothing; pass a local PDB file.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from Bio.PDB import Superimposer
from Bio.PDB.Atom import Atom
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.SeqUtils import seq1

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pdb", type=Path, required=True)
parser.add_argument("--chain", required=True)
parser.add_argument("--reference-cif", type=Path, required=True, help="mmCIF whose label_seq_id is the UniProt position")
parser.add_argument("--manifest", type=Path, required=True, help="target manifest with start and sequence (UniProt numbering)")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--partner-chain", help="optional: chain of the natural binding partner (3DI3: A, IL-7)")
parser.add_argument("--hotspots", help="optional: comma-separated UniProt hotspots to compare with the partner interface")
parser.add_argument("--interface-cutoff", type=float, default=5.0, help="heavy-atom distance in angstrom")
args = parser.parse_args()
if args.output.exists():
    raise ValueError("Use a fresh output file")

manifest = json.loads(args.manifest.read_text())
uniprot = {manifest["start"] + i: aa for i, aa in enumerate(manifest["sequence"])}
lines = args.pdb.read_text().splitlines()
dbref = [l for l in lines if l.startswith("DBREF") and l[12] == args.chain and l[26:32].strip() == "UNP"]
if len(dbref) != 1:
    raise ValueError(f"Expected one UniProt DBREF for chain {args.chain}, found {len(dbref)}")
seq_begin, db_begin = int(dbref[0][14:18]), int(dbref[0][55:60])
offset = db_begin - seq_begin                      # UniProt position = PDB number + offset
pdb = {}
for l in lines:
    if l.startswith("ATOM") and l[21] == args.chain and l[12:16].strip() == "CA" and l[16] in " A":
        pdb[int(l[22:26]) + offset] = (seq1(l[17:20]), np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]))
positions = sorted(pdb)

cif = MMCIF2Dict(str(args.reference_cif))
reference = {}
for name, res, seq, x, y, z in zip(cif["_atom_site.label_atom_id"], cif["_atom_site.label_comp_id"],
                                   cif["_atom_site.label_seq_id"], cif["_atom_site.Cartn_x"],
                                   cif["_atom_site.Cartn_y"], cif["_atom_site.Cartn_z"]):
    if name == "CA":
        reference[int(seq)] = (seq1(res), np.array([float(x), float(y), float(z)]))

common = sorted(set(reference) & set(pdb))
if len(common) < 10:
    raise ValueError("Too few shared residues to superimpose")
make = lambda coords: [Atom("CA", c, 0, 1, " ", "CA", i, element="C") for i, c in enumerate(coords)]
fit = Superimposer()
fit.set_atoms(make([pdb[p][1] for p in common]), make([reference[p][1] for p in common]))
official = range(manifest["start"], manifest["end"] + 1)
report = {
    "pdb_file": args.pdb.name, "chain": args.chain, "dbref_offset_pdb_to_uniprot": offset,
    "pdb_residues": len(positions), "pdb_uniprot_range": [positions[0], positions[-1]],
    "pdb_gaps_inside_range": [p for p in range(positions[0], positions[-1] + 1) if p not in pdb],
    "pdb_vs_uniprot_identity_differences": [{"position": p, "pdb": pdb[p][0], "uniprot": uniprot[p]}
                                            for p in positions if p in uniprot and pdb[p][0] != uniprot[p]],
    "reference_residues": len(reference), "reference_range": [min(reference), max(reference)],
    "reference_vs_uniprot_identity_differences": [p for p in reference if p in uniprot and reference[p][0] != uniprot[p]],
    "shared_residues": len(common),
    "only_in_reference": sorted(set(reference) - set(pdb)), "only_in_pdb": sorted(set(pdb) - set(reference)),
    "ca_rmsd_angstrom_over_shared": round(float(fit.rms), 3),
    "official_positions_without_pdb_coordinates": [p for p in official if p not in pdb],
    "official_positions_without_reference_coordinates": [p for p in official if p not in reference],
}
if args.partner_chain:
    def heavy_atoms(chain):  # protein ATOM records only (no waters or glycans), first alternate location
        out = []
        for l in lines:
            if l.startswith("ATOM") and l[21] == chain and l[16] in " A" and l[76:78].strip() not in ("H", "D"):
                out.append((int(l[22:26]) + offset, np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
        return out
    own, partner = heavy_atoms(args.chain), heavy_atoms(args.partner_chain)
    if not own or not partner:
        raise ValueError("No atoms found for the chain or its partner")
    partner_xyz = np.array([c for _, c in partner])
    nearest = {}
    for position, xyz in own:
        nearest[position] = min(nearest.get(position, 1e9), float(np.linalg.norm(partner_xyz - xyz, axis=1).min()))
    interface = sorted(p for p, d in nearest.items() if d <= args.interface_cutoff)
    report["partner_chain"] = args.partner_chain
    report["interface_cutoff_angstrom"] = args.interface_cutoff
    report["interface_residues_uniprot"] = interface
    report["interface_count"] = len(interface)
    if 66 in nearest:
        report["residue_66_min_heavy_atom_distance_to_partner_angstrom"] = round(nearest[66], 2)
    if args.hotspots:
        hotspots = sorted(int(x) for x in args.hotspots.split(","))
        report["hotspots"] = hotspots
        report["hotspots_in_partner_interface"] = [h for h in hotspots if h in interface]
        report["hotspots_not_in_partner_interface"] = [h for h in hotspots if h not in interface]
        report["hotspot_min_distance_to_partner_angstrom"] = {str(h): round(nearest[h], 2) for h in hotspots if h in nearest}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: report[k] for k in ("pdb_residues", "pdb_uniprot_range", "pdb_vs_uniprot_identity_differences",
                                         "shared_residues", "only_in_reference", "ca_rmsd_angstrom_over_shared")}))
