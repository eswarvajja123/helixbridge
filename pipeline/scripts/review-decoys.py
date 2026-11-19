"""Summarize an AlphaFold run of designs against decoy targets (CPU only). See binder_campaign/decoys.py.

Example:
  python pipeline/scripts/review-decoys.py --alphafold results/alphafold-x --decoys gfp,lyz --out results/decoys-x.json
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.decoys import read_decoy_iptm, summarize, write_summary

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--alphafold", type=Path, nargs="+", required=True, help="run folders holding the chunk archives")
parser.add_argument("--decoys", required=True, help="comma-separated decoy names used as id prefixes, for example gfp,lyz")
parser.add_argument("--candidate-prefix", default="il7ra-rfd4-")
parser.add_argument("--out", type=Path, required=True, help="the summary JSON to write")
args = parser.parse_args()
try:
    if args.out.exists():
        raise ValueError("Use a fresh output file")
    names = [d for d in args.decoys.split(",") if d]
    found = read_decoy_iptm(args.alphafold, names, args.candidate_prefix)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_summary(args.out, found, names)
    top = summarize(found)
    print(f"{len(found)} designs; highest decoy ipTM: " + ", ".join(f"{c[-6:]} {v['max_decoy']} {v['max_iptm']}" for c, v in sorted(top.items(), key=lambda kv: -kv[1]['max_iptm'])[:5]))
except (ValueError, FileNotFoundError, KeyError) as error:
    sys.exit(f"REFUSED: {error}")
