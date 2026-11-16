"""Build and measure candidate hotspot sets for one receptor chain of a PDB entry (CPU only).

Everything is computed from the PDB file: isolated-chain solvent exposure, distance to the natural
partner chain, distance to observed glycans, and patch compactness. Three options are produced with
fixed, stated rules. This only organises evidence for a human epitope decision; it does not predict
binding, and a different PDB entry or cutoff changes the result.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, ShrakeRupley
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pdb", type=Path, required=True)
parser.add_argument("--chain", required=True)
parser.add_argument("--partner-chain", required=True)
parser.add_argument("--current-hotspots", required=True, help="comma-separated UniProt positions (Option A, unchanged)")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--size", type=int, default=10, help="hotspots per option")
parser.add_argument("--interface-cutoff", type=float, default=5.0)
parser.add_argument("--b-min-sasa", type=float, default=30.0)
parser.add_argument("--b-min-glycan", type=float, default=10.0)
parser.add_argument("--c-min-sasa", type=float, default=30.0)
parser.add_argument("--c-min-partner", type=float, default=15.0)
parser.add_argument("--c-min-glycan", type=float, default=15.0)
parser.add_argument("--patch-radius", type=float, default=10.0)
parser.add_argument("--c-exclude-from", type=int, default=220,
                    help="Option C skips residues from this UniProt position on (membrane-proximal / next to the C-terminal tag)")
args = parser.parse_args()
if args.output.exists():
    raise ValueError("Use a fresh output file")

lines = args.pdb.read_text().splitlines()
dbref = [l for l in lines if l.startswith("DBREF") and l[12] == args.chain and l[26:32].strip() == "UNP"]
if len(dbref) != 1:
    raise ValueError("Expected one UniProt DBREF for the chain")
offset = int(dbref[0][55:60]) - int(dbref[0][14:18])
model = PDBParser(QUIET=True).get_structure("entry", str(args.pdb))[0]


def heavy(residue):
    return [a.coord for a in residue if a.element not in ("H", "D")]


# isolated-chain exposure (protein residues only: no partner, waters or sugars)
own = copy.deepcopy(model[args.chain])
for residue in list(own):
    if residue.id[0] != " ":
        own.detach_child(residue.id)
holder = Structure("chain")
holder.add(Model(0))
holder[0].add(own)
ShrakeRupley(n_points=960).compute(holder, level="R")

partner = np.array([c for r in model[args.partner_chain] if r.id[0] == " " for c in heavy(r)])
sugars = np.array([c for chain in model for r in chain if r.get_resname() == "NAG" for c in heavy(r)])
if not len(sugars):
    raise ValueError("No observed glycans in the entry")
table = {}
for residue in own:
    xyz = np.array(heavy(residue))
    u = residue.id[1] + offset
    table[u] = {"aa": residue.get_resname(), "sasa": round(float(residue.sasa), 1),
                "partner_dist": round(float(np.linalg.norm(partner[None] - xyz[:, None], axis=2).min()), 2),
                "glycan_dist": round(float(np.linalg.norm(sugars[None] - xyz[:, None], axis=2).min()), 2),
                "ca": own[residue.id]["CA"].coord}
for u, row in table.items():
    row["in_interface"] = row["partner_dist"] <= args.interface_cutoff

# Option A: the current set, unchanged
current = sorted(int(x) for x in args.current_hotspots.split(","))
missing = [h for h in current if h not in table]
if missing:
    raise ValueError(f"Current hotspots without coordinates in this chain: {missing}")

# Option B: most exposed natural-site residues away from glycans
eligible_b = [u for u, r in table.items() if r["in_interface"] and r["sasa"] >= args.b_min_sasa and r["glycan_dist"] >= args.b_min_glycan]
option_b = sorted(sorted(eligible_b, key=lambda u: (-table[u]["sasa"], u))[:args.size])

# Option C: exposed patch away from the natural site and from glycans (exploratory)
ok_c = [u for u, r in table.items() if r["sasa"] >= args.c_min_sasa and r["partner_dist"] >= args.c_min_partner and r["glycan_dist"] >= args.c_min_glycan and u < args.c_exclude_from]
best = None
for center in ok_c:
    near = [u for u in ok_c if np.linalg.norm(table[u]["ca"] - table[center]["ca"]) <= args.patch_radius]
    key = (len(near), sum(table[u]["sasa"] for u in near), -center)
    if best is None or key > best[0]:
        best = (key, center, near)
_, center_c, near_c = best
option_c = sorted(sorted(near_c, key=lambda u: (float(np.linalg.norm(table[u]["ca"] - table[center_c]["ca"])), u))[:args.size])


def metrics(hotspots):
    cas = np.array([table[h]["ca"] for h in hotspots])
    spread = max(float(np.linalg.norm(a - b)) for a in cas for b in cas)
    return {"hotspots": hotspots, "count": len(hotspots),
            "in_natural_interface": [h for h in hotspots if table[h]["in_interface"]],
            "mean_sasa": round(float(np.mean([table[h]["sasa"] for h in hotspots])), 1),
            "buried_below_30": [h for h in hotspots if table[h]["sasa"] < 30],
            "min_glycan_distance": min(table[h]["glycan_dist"] for h in hotspots),
            "min_partner_distance": min(table[h]["partner_dist"] for h in hotspots),
            "max_ca_ca_spread": round(spread, 1)}


report = {
    "pdb_file": args.pdb.name, "chain": args.chain, "partner_chain": args.partner_chain,
    "thresholds": {k: getattr(args, k) for k in ("interface_cutoff", "b_min_sasa", "b_min_glycan", "c_min_sasa",
                                                 "c_min_partner", "c_min_glycan", "patch_radius", "c_exclude_from", "size")},
    "options": {
        "A_current_provisional": {"rule": "unchanged historical set", **metrics(current)},
        "B_exposed_natural_site": {"rule": "natural-interface residues with exposure and glycan-distance floors, top by exposure",
                                   **metrics(option_b)},
        "C_offsite_exploratory": {"rule": f"best exposed patch away from the natural site and glycans (center {center_c})",
                                  "center": center_c, **metrics(option_c)}},
    "residues": {str(u): {k: v for k, v in r.items() if k != "ca"} for u, r in sorted(table.items())},
    "limits": "Isolated chain only: no membrane, gamma-c, TSLP receptor or full glycans. Crystal glycans only. Not a binding prediction.",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
for name, o in report["options"].items():
    print(f"{name}: {o['hotspots']} | in_interface={len(o['in_natural_interface'])}/{o['count']} mean_sasa={o['mean_sasa']} "
          f"buried={o['buried_below_30']} glycan>={o['min_glycan_distance']} spread={o['max_ca_ca_spread']}")
