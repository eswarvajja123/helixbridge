"""Propose a shortlist of at most 20 designs covering distinct folds and sites (Task 5.3). CPU only.

Input: one records file that has been through scoring, structure QC, the novelty pre-screen and the liability review (the output of
review-liabilities.py), and a diversity report. It picks by COVERAGE, not by score (see binder_campaign/selection.py), writes the
proposal, then runs the team's own export guard and reports whether it passes or exactly what a person still has to review.

The guard needs human decisions (novelty review, independent structure review, screen review). They are never set automatically: give
them in a reviews file (reviewer, date, per-design fields) and the guard is run again.

Example:
  python pipeline/scripts/select-finalists.py --records results/liabilities-eswar-l1/liabilities-candidates.json \
        --diversity results/diversity-eswar-d4/diversity.json --out results/selection-eswar-s1
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.selection import ALPHAFOLD_MIN_IPTM, MAX_FINALISTS, apply_reviews, export_check, review_gaps, select, write_outputs

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--records", type=Path, required=True)
parser.add_argument("--diversity", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--limit", type=int, default=MAX_FINALISTS)
parser.add_argument("--source", default="generator", help="structure source whose groups are used: generator, esmfold2 or boltz2")
parser.add_argument("--fold-threshold", type=float, default=0.5)
parser.add_argument("--site-threshold", type=float, default=0.5)
parser.add_argument("--alphafold-results", type=Path, nargs="+", default=None,
                    help="alphafold-results.json files from review-alphafold.py; turns on the AlphaFold gate (ipTM above every control, pose limits)")
parser.add_argument("--alphafold-min-iptm", type=float, default=ALPHAFOLD_MIN_IPTM)
parser.add_argument("--decoy-results", type=Path, default=None,
                    help="a decoy summary from review-decoys.py; designs that bind an unrelated protein confidently (ipTM above the controls) are excluded")
parser.add_argument("--reviews", type=Path, default=None, help="a person's review decisions (JSON); see binder_campaign/selection.py")
args = parser.parse_args()
try:
    if args.out.exists():
        raise ValueError("Use a fresh output directory")
    records = json.loads(args.records.read_text())["candidates"]
    diversity = json.loads(args.diversity.read_text())
    alphafold = None
    if args.alphafold_results:
        alphafold = {}
        for path in args.alphafold_results:
            for entry in json.loads(path.read_text())["results"]:
                if entry["candidate_id"] in alphafold:
                    raise ValueError("A design appears in two AlphaFold result files: " + entry["candidate_id"])
                alphafold[entry["candidate_id"]] = entry
    decoys, decoy_names = None, ()
    if args.decoy_results:
        doc = json.loads(args.decoy_results.read_text())
        decoys, decoy_names = doc["designs"], tuple(doc["decoys"])
    selected, report = select(records, diversity, args.limit, args.source, args.fold_threshold, args.site_threshold, alphafold, args.alphafold_min_iptm,
                              decoys, decoy_names)
    ids = [s["candidate_id"] for s in selected]
    reviewed = apply_reviews(records, ids, json.loads(args.reviews.read_text())) if args.reviews else [r for r in records if r["candidate_id"] in ids]
    args.out.mkdir(parents=True)
    check = export_check(reviewed, ids, args.out / "draft-submission.csv")
    write_outputs(args.out, selected, report, reviewed, {"check": check, "gaps": review_gaps(reviewed, ids)})
    print(json.dumps({"output_folder": str(args.out), "pool": report["pool"], "eligible": report["eligible"], "selected": report["selected"],
                      "fold_groups": f"{report['fold_groups_selected']} of {report['fold_groups_available']}",
                      "site_groups": f"{report['site_groups_selected']} of {report['site_groups_available']}",
                      "export_guard_passes": check["passes"], "export_guard_message": check["message"]}, indent=2))
except (ValueError, FileNotFoundError, KeyError) as error:
    sys.exit(f"REFUSED: {error}")
