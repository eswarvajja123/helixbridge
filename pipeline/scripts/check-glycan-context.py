"""Measure how close observed glycans are to candidate hotspots in a PDB entry (CPU only).

Reads the entry's LINK records for ASN-NAG bonds on one chain, maps PDB numbers to UniProt
numbers through the file's own DBREF, and reports the minimum heavy-atom distance from each
glycan to each hotspot residue. Also lists sequons in the target sequence and which of them
have an observed glycan. Observed in a crystal is not the same as occupied in the assay cell.
"""
import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pdb", type=Path, required=True)
parser.add_argument("--chain", required=True)
parser.add_argument("--partner-chain", help="optional natural partner chain, to report glycan distance to it")
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--hotspots", required=True, help="comma-separated UniProt positions")
parser.add_argument("--screen-cutoff", type=float, default=8.0, help="flag a hotspot closer than this to a glycan (angstrom)")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise ValueError("Use a fresh output file")

lines = args.pdb.read_text().splitlines()
dbref = [l for l in lines if l.startswith("DBREF") and l[12] == args.chain and l[26:32].strip() == "UNP"]
if len(dbref) != 1:
    raise ValueError("Expected one UniProt DBREF for the chain")
offset = int(dbref[0][55:60]) - int(dbref[0][14:18])
manifest = json.loads(args.manifest.read_text())
seq, start = manifest["sequence"], manifest["start"]
hotspots = sorted(int(x) for x in args.hotspots.split(","))


def xyz(l):
    return np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])


def heavy(l):
    return l[16] in " A" and l[76:78].strip() not in ("H", "D")


# ASN (on the chain) -> sugar chain, from LINK records
roots = {}
for l in lines:
    if l.startswith("LINK") and l[12:16].strip() == "ND2" and l[17:20] == "ASN" and l[21] == args.chain and l[47:50] == "NAG":
        roots[int(l[22:26]) + offset] = l[51]
sugar = {}
for l in lines:
    if l.startswith("HETATM") and l[17:20] == "NAG" and heavy(l):
        sugar.setdefault(l[21], []).append((l[22:26].strip(), xyz(l)))
protein = {}
for l in lines:
    if l.startswith("ATOM") and heavy(l):
        protein.setdefault(l[21], {}).setdefault(int(l[22:26]) + offset, []).append(xyz(l))
if args.chain not in protein:
    raise ValueError("Chain not found")

sequons = [start + i for i in range(len(seq) - 2) if seq[i] == "N" and seq[i + 1] != "P" and seq[i + 2] in "ST"]
glycans = []
for asn, sugar_chain in sorted(roots.items()):
    atoms = np.array([c for _, c in sugar[sugar_chain]])
    distances = {}
    for h in hotspots:
        if h in protein[args.chain]:
            distances[str(h)] = round(float(min(np.linalg.norm(atoms - p, axis=1).min() for p in protein[args.chain][h])), 2)
    entry = {"asn_uniprot": asn, "sugar_chain": sugar_chain, "sugar_residues": len({r for r, _ in sugar[sugar_chain]}),
             "in_sequon_list": asn in sequons, "min_distance_to_hotspots_angstrom": distances,
             "hotspots_within_screen_cutoff": [int(h) for h, d in distances.items() if d <= args.screen_cutoff]}
    if args.partner_chain:
        partner = np.array([p for coords in protein[args.partner_chain].values() for p in coords])
        entry["min_distance_to_partner_chain_angstrom"] = round(float(min(np.linalg.norm(partner - a, axis=1).min() for a in atoms)), 2)
    glycans.append(entry)

resolved = sorted(protein[args.chain])
report = {
    "pdb_file": args.pdb.name, "chain": args.chain, "partner_chain": args.partner_chain,
    "sequons_in_target_sequence": sequons,
    "sequons_with_observed_glycan": sorted(a for a in roots if a in sequons),
    "sequons_resolved_without_glycan": [s for s in sequons if resolved[0] <= s <= resolved[-1] and s not in roots],
    "sequons_outside_resolved_range": [s for s in sequons if not resolved[0] <= s <= resolved[-1]],
    "glycans": glycans, "hotspots": hotspots, "screen_cutoff_angstrom": args.screen_cutoff,
    "hotspots_near_any_glycan": sorted({h for g in glycans for h in g["hotspots_within_screen_cutoff"]}),
    "limits": "Crystal glycans only; occupancy in the assay protein is unknown. Distances are to the observed NAG residues, not full glycans.",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: report[k] for k in ("sequons_in_target_sequence", "sequons_with_observed_glycan",
                                         "sequons_resolved_without_glycan", "sequons_outside_resolved_range",
                                         "hotspots_near_any_glycan")}))
