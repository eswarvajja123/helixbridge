"""Write the hand-off package for the AlphaFold cross-check of the shortlist (Task 5.4): inputs and a blank results sheet only.

Example:
  python pipeline/scripts/export-alphafold-inputs.py --records results/liabilities-eswar-l2/liabilities-candidates.json \
        --selection results/selection-eswar-s2/selection.json --out results/alphafold-handoff-eswar-h1
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.handoff import export

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--records", type=Path, required=True)
parser.add_argument("--selection", type=Path, required=True)
parser.add_argument("--target-manifest", type=Path, default=ROOT / "docs/eswar/phase1-data/il7ra-manifest.json")
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
try:
    records = json.loads(args.records.read_text())["candidates"]
    ids = [s["candidate_id"] for s in json.loads(args.selection.read_text())["selected"]]
    manifest = export(records, ids, json.loads(args.target_manifest.read_text()), args.out)
    print(json.dumps({"output_folder": str(args.out), "designs": len(manifest["designs"]), "status": manifest["status"]}, indent=2))
except (ValueError, FileNotFoundError, KeyError) as error:
    sys.exit(f"REFUSED: {error}")
