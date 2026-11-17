"""Check one on-target generation locally, spending nothing (CPU only; no Modal login needed).

  python pipeline/scripts/plan-modal-run.py --option A --seed 7
  python pipeline/scripts/plan-modal-run.py --hotspots 100,101,102 --hotspot-source my_note --length-min 50 --length-max 80

Prints the exact plan that `modal run modal_current.py --stage generate ...` would send, after running the
generation script's real input checks. Run it from the project root (the folder that holds pipeline/ and examples/).
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.modal_plan import build_plan, local_dry_run

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2], help="project root")
parser.add_argument("--run-id", default="")
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--option", default="", help="hotspot option A, B or C from docs/eswar/hotspot-options (default A)")
parser.add_argument("--hotspots", default="")
parser.add_argument("--hotspot-source", default="")
parser.add_argument("--length", type=int, default=60)
parser.add_argument("--length-min", type=int, default=0)
parser.add_argument("--length-max", type=int, default=0)
args = parser.parse_args()
try:
    plan = build_plan(args.root, run_id=args.run_id, seed=args.seed, option=args.option, hotspots=args.hotspots,
                      hotspot_source=args.hotspot_source, length=args.length, length_min=args.length_min,
                      length_max=args.length_max)
    checked = local_dry_run(args.root, plan)
except ValueError as error:
    sys.exit(f"REFUSED: {error}")
print(json.dumps({"plan": plan, "input_checks": "passed", "generation_settings": checked["overrides"]}, indent=2))
