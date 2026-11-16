"""Build one validated RFD4 conditioning input set (CPU only; runs no model).

Example (provisional historical hotspots, fixed 60-residue binder):
  python scripts/build-il7ra-conditions.py --reference-dir ../examples/il7ra-rehearsal \
      --output-dir data/conditioning/il7ra-60 --hotspots 100,101,102,158,159,161,211,212,213,214 \
      --binder-length 60 --hotspot-source provisional_historical
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.conditioning import write_conditioning

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--reference-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--hotspots", required=True, help="comma-separated UniProt positions")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--binder-length", type=int)
group.add_argument("--binder-range", nargs=2, type=int, metavar=("MIN", "MAX"))
parser.add_argument("--hotspot-source", required=True,
                    help="where the hotspots came from, e.g. provisional_historical or a reviewed epitope note")
args = parser.parse_args()
try:
    hotspots = [int(x) for x in args.hotspots.split(",")]
except ValueError:
    parser.error("--hotspots must be comma-separated integers")
length = args.binder_length if args.binder_length is not None else tuple(args.binder_range)
audit = write_conditioning(args.reference_dir, args.output_dir, hotspots, length,
                           hotspot_source=args.hotspot_source)
print(f"Wrote conditioning to {args.output_dir}: binder_length={audit['binder_length']}, "
      f"hotspots={audit['hotspots_uniprot']}, source={audit['hotspot_source']}")
