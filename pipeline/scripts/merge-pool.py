"""Merge generation batches into one candidate pool (CPU only; needs no Modal login).

Each batch removes near-duplicates only inside itself. A pool built from several batches (for example one per hotspot option) needs
the gate run again across all of them before the designs are scored and compared. The first batch given wins ties.

Example:
  python pipeline/scripts/merge-pool.py --batch results/batch-eswar-pA --batch results/batch-eswar-pB --batch results/batch-eswar-pC \
        --out results/pool-eswar-p1

Writes candidates.json (same shape as a batch's, usable by evaluate-candidates.py), a designs/ folder holding every structure, and
pool-report.json. The input batch folders are never changed.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.pool import merge_batches

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--batch", type=Path, action="append", required=True, help="a batch folder; repeat for each, in priority order")
parser.add_argument("--out", type=Path, required=True, help="new pool folder")
args = parser.parse_args()
try:
    report = merge_batches(args.batch, args.out, log=lambda line: print(line, file=sys.stderr))
    print(json.dumps({"output_folder": str(args.out), **{k: report[k] for k in ("designs", "kept_by_gate", "dropped_by_gate",
                                                                               "dropped_only_across_batches", "batches")}}, indent=2))
except (ValueError, FileNotFoundError, KeyError) as error:
    sys.exit(f"REFUSED: {error}")
