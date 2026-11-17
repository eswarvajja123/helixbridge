"""Sequence liabilities of a candidate pool (Task 5.2). CPU only; needs no Modal login.

Records the team's descriptive sequence diagnostics per design and four categorical flags (odd cysteine count, N-glycosylation
sequon, Asn-Gly, Asp-Pro). No numeric thresholds, no repair, and nothing is removed: the flags inform selection.

Example:
  python pipeline/scripts/review-liabilities.py --records results/novelty-eswar-n1/novelty-candidates.json --out results/liabilities-eswar-l1
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.liabilities import run_review

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--records", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
try:
    report = run_review(args.records, args.out, log=lambda line: print(line, file=sys.stderr))
    print(json.dumps({"output_folder": str(args.out), **{k: report[k] for k in ("candidates", "flag_counts", "designs_with_no_flags")}}, indent=2))
except (ValueError, FileNotFoundError, KeyError) as error:
    sys.exit(f"REFUSED: {error}")
