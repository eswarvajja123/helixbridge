"""Structure QC of evaluated candidates (Phase 3, Task 3.3). CPU only; needs no Modal login and no GPU.

Reads an evaluation folder (the output of modal_evaluate.py) and the generation batch folder it came from, checks every candidate's
structures, and writes a QC copy of the candidate records with `qc_reasons` filled in:

  length 40-100, single binder chain, close heavy-atom pairs (clash), target contact, hotspot contact, and binder pose shift
  (generator vs each refold, and ESMFold2 vs Boltz-2, after aligning on the shared reference core 37-231).

The flags are geometric consistency checks with the team's provisional limits. They are not binding predictions, and a candidate
with no flags is not thereby a good design.

Example:
  python pipeline/scripts/qc-candidates.py --evaluation results/eval-eswar-e1 --batch-dir results/batch-eswar-b1 --out results/qc-eswar-q1

Writes qc-candidates.json, qc.csv and qc-report.json into the (new) --out folder; the evaluation folder is never changed.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.qc import SOURCES, Thresholds, run_qc

ROOT = Path(__file__).resolve().parents[2]
defaults = Thresholds()
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--evaluation", type=Path, required=True, help="evaluation folder holding evaluations.json and evals/")
parser.add_argument("--batch-dir", type=Path, default=None, help="generation batch folder (used if a recorded structure path no longer exists)")
parser.add_argument("--out", type=Path, required=True, help="new folder for the QC output")
parser.add_argument("--target-manifest", type=Path, default=ROOT / "docs/eswar/phase1-data/il7ra-manifest.json")
parser.add_argument("--sources", default=",".join(SOURCES), help="structures to check, comma separated (default: all)")
parser.add_argument("--max-close-pairs", type=int, default=defaults.max_close_pairs)
parser.add_argument("--min-contact-residues", type=int, default=defaults.min_target_contact_residues)
parser.add_argument("--no-hotspot-requirement", action="store_true", help="do not require contact with a design hotspot")
parser.add_argument("--max-shift-generator", type=float, default=defaults.max_pose_shift_generator_A)
parser.add_argument("--max-shift-between", type=float, default=defaults.max_pose_shift_between_predictors_A)
args = parser.parse_args()

try:
    expected = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    if not expected or set(expected) - set(SOURCES):
        raise ValueError(f"--sources must be a subset of {', '.join(SOURCES)}")
    thresholds = Thresholds(max_close_pairs=args.max_close_pairs, min_target_contact_residues=args.min_contact_residues,
                            require_hotspot_contact=not args.no_hotspot_requirement,
                            max_pose_shift_generator_A=args.max_shift_generator,
                            max_pose_shift_between_predictors_A=args.max_shift_between)
    report = run_qc(args.evaluation, args.target_manifest, args.out, thresholds, args.batch_dir, expected,
                    log=lambda line: print(line, file=sys.stderr))
    print(json.dumps({"output_folder": str(args.out), **{k: report[k] for k in ("candidates", "status_counts", "reason_counts",
                                                                              "thresholds")}}, indent=2))
except (ValueError, FileNotFoundError, KeyError) as error:
    sys.exit(f"REFUSED: {error}")
