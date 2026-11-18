"""Write shuffled-sequence negative controls for a shortlist (Phase 3 calibration). CPU only.

One control per shortlisted design: same length and composition, random order. Score them with evaluate-candidates / modal_evaluate.py
and check them with qc-candidates.py --sources esmfold2,boltz2, then compare with controls-report.py.

Example:
  python pipeline/scripts/make-controls.py --records results/liabilities-x/liabilities-candidates.json --selection results/selection-x/selection.json --out results/controls-1
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.controls import make_controls
from binder_campaign.runner import write_json

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--records", type=Path, required=True)
parser.add_argument("--selection", type=Path, required=True)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
try:
    if args.out.exists():
        raise ValueError("Use a fresh output directory")
    wanted = [s["candidate_id"] for s in json.loads(args.selection.read_text())["selected"]]
    by_id = {r["candidate_id"]: r for r in json.loads(args.records.read_text())["candidates"]}
    missing = [w for w in wanted if w not in by_id]
    if missing:
        raise ValueError("Not in the records: " + ", ".join(missing))
    controls = make_controls([by_id[w] for w in wanted], args.seed)
    args.out.mkdir(parents=True)
    write_json(args.out / "candidates.json", {"candidates": controls, "count": len(controls),
               "note": "Shuffled-sequence negative controls: same length and composition as their parents, random order. Not designs."})
    print(json.dumps({"output_folder": str(args.out), "controls": len(controls)}, indent=2))
except (ValueError, FileNotFoundError, KeyError) as error:
    sys.exit(f"REFUSED: {error}")
