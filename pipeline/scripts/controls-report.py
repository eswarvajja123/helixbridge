"""Compare shortlisted designs with their shuffled negative controls (Phase 3 calibration). CPU only.

Both inputs are qc-candidates.json files made by qc-candidates.py with --sources esmfold2,boltz2 (so the two groups are checked the same
way: the two predictors' poses and contacts, no generator structure). Writes controls-report.json.

Example:
  python pipeline/scripts/controls-report.py --designs results/qc-d/qc-candidates.json --selection results/selection-x/selection.json \
      --controls results/qc-c/qc-candidates.json --out results/controls-report-1
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.controls import report
from binder_campaign.runner import write_json

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--designs", type=Path, required=True)
parser.add_argument("--selection", type=Path, required=True, help="only these designs are compared")
parser.add_argument("--controls", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
try:
    if args.out.exists():
        raise ValueError("Use a fresh output directory")
    wanted = {s["candidate_id"] for s in json.loads(args.selection.read_text())["selected"]}
    designs = [r for r in json.loads(args.designs.read_text())["candidates"] if r["candidate_id"] in wanted]
    controls = json.loads(args.controls.read_text())["candidates"]
    if len(designs) != len(wanted):
        raise ValueError("Some shortlisted designs are missing from the designs file")
    result = report(designs, controls)
    args.out.mkdir(parents=True)
    write_json(args.out / "controls-report.json", result)
    print(json.dumps({"output_folder": str(args.out), "designs": result["designs"], "controls": result["controls"], "qc_clean": result["qc_clean"],
                      "p_design_above_control": {k: v["p_design_above_control"] for k, v in result["measures"].items()}}, indent=2))
except (ValueError, FileNotFoundError, KeyError) as error:
    sys.exit(f"REFUSED: {error}")
