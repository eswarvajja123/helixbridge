"""Measure the AlphaFold-Multimer predictions of the shortlist (Task 5.4). CPU only.

Reads the chunk archives written by modal_alphafold.py and, for each shortlisted design, writes AlphaFold's own confidence (pTM, ipTM,
pLDDT, interface PAE), contacts, hotspot contacts, close pairs, and how far the binder sits from the generator, ESMFold2 and Boltz-2
poses (after aligning on the shared reference core). It records measurements only: no threshold, no ranking, and the review fields
(independent_prediction_status, the 'interface looks as designed' judgement) are left for a person.

Example:
  python pipeline/scripts/review-alphafold.py --records results/liabilities-x/liabilities-candidates.json --selection results/selection-x/selection.json \
      --evaluation results/eval-x --batch-dir results/pool-x --alphafold results/alphafold-1 --out results/alphafold-review-1
"""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.alphafold_review import read_archives, review_design
from binder_campaign.qc import structures_for
from binder_campaign.runner import write_json

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--records", type=Path, required=True)
parser.add_argument("--selection", type=Path, required=True)
parser.add_argument("--evaluation", type=Path, required=True)
parser.add_argument("--batch-dir", type=Path, default=None)
parser.add_argument("--alphafold", type=Path, required=True, help="the folder modal_alphafold.py wrote (c1.tar.gz, ...)")
parser.add_argument("--target-manifest", type=Path, default=ROOT / "docs/eswar/phase1-data/il7ra-manifest.json")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
try:
    if args.out.exists():
        raise ValueError("Use a fresh output directory")
    wanted = [s["candidate_id"] for s in json.loads(args.selection.read_text())["selected"]]
    by_id = {r["candidate_id"]: r for r in json.loads(args.records.read_text())["candidates"]}
    target = json.loads(args.target_manifest.read_text())
    predictions = read_archives(args.alphafold)
    results, missing = [], []
    for cid in wanted:
        if cid not in by_id:
            raise ValueError("Not in the records: " + cid)
        if cid not in predictions:
            missing.append(cid)
            continue
        found = structures_for(by_id[cid], args.evaluation, args.batch_dir)
        others = {name: info["path"] for name, info in found.items()}
        results.append(review_design(by_id[cid], predictions[cid][0], predictions[cid][1], target, others))
    args.out.mkdir(parents=True)
    write_json(args.out / "alphafold-results.json", {"results": results, "missing_predictions": missing, "designs_requested": len(wanted),
               "scope": "Measurements of AlphaFold-Multimer predictions only. No threshold or ranking; independent_prediction_status is not set."})
    columns = ["candidate_id", "binder_length", "ptm", "iptm", "plddt_binder_mean", "plddt_target_mean", "interface_pae_mean_A", "target_contact_residues",
               "hotspot_contacts", "heavy_atom_pairs_lt_2A", "min_heavy_distance_A", "pose_shift_generator_A", "pose_shift_esmfold2_A", "pose_shift_boltz2_A"]
    with (args.out / "alphafold-results.csv").open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(columns)
        for r in results:
            s = r.get("pose_shift_A") or {}
            w.writerow([r["candidate_id"], r["binder_length"], r["ptm"], r["iptm"], r["plddt_binder_mean"], r["plddt_target_mean"], r["interface_pae_mean_A"],
                        r.get("target_contact_residues", ""), " ".join(map(str, r.get("hotspot_contacts") or [])), r.get("heavy_atom_pairs_lt_2A", ""),
                        r.get("min_heavy_distance_A", ""), s.get("generator", ""), s.get("esmfold2", ""), s.get("boltz2", "")])
    print(json.dumps({"output_folder": str(args.out), "designs": len(results), "missing_predictions": missing}, indent=2))
except (ValueError, FileNotFoundError, KeyError) as error:
    sys.exit(f"REFUSED: {error}")
