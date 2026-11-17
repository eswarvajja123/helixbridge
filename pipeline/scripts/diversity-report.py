"""Diversity report for a pool of binder designs (Phase 3, Task 3.4). CPU only; needs no Modal login and no GPU.

For every pair of designs, within each structure source (the RFD4 generator pose, the ESMFold2 refold, the Boltz-2 refold):
fold similarity (TM-score of the binder backbones), site similarity (Jaccard overlap of the target residues each binder touches)
and sequence identity; then groups the pool at several thresholds, greedy complete-link and connected components.

Two ways to name the pool:
  --records FILE   a candidates.json, evaluations.json or qc-candidates.json (with --evaluation DIR for the refolds and
                   --batch-dir DIR for the generator structures). Records with QC flags are left out unless --include-flagged.
  --pool FILE      any pool: {"target_sequence": ..., "positions": [...], "members": [{"id", "binder_sequence", "structure"}]}

The groups are descriptive: not SCOP/CATH classes, evolutionary families or binding predictions.

Example:
  python pipeline/scripts/diversity-report.py --records results/qc-eswar-q1/qc-candidates.json --evaluation results/eval-eswar-e1 \
        --batch-dir results/batch-eswar-b1 --out results/diversity-eswar-d1
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.diversity import FOLD_THRESHOLDS, MAX_POOL, SITE_THRESHOLDS, run_diversity, run_diversity_pool
from binder_campaign.qc import SOURCES

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--records", type=Path)
parser.add_argument("--pool", type=Path)
parser.add_argument("--evaluation", type=Path, default=None, help="evaluation folder holding evals/ (for the refolded structures)")
parser.add_argument("--batch-dir", type=Path, default=None)
parser.add_argument("--target-manifest", type=Path, default=ROOT / "docs/eswar/phase1-data/il7ra-manifest.json")
parser.add_argument("--out", type=Path, required=True, help="new folder for the report")
parser.add_argument("--sources", default=",".join(SOURCES))
parser.add_argument("--include-flagged", action="store_true")
parser.add_argument("--fold-thresholds", default=",".join(str(t) for t in FOLD_THRESHOLDS))
parser.add_argument("--site-thresholds", default=",".join(str(t) for t in SITE_THRESHOLDS))
parser.add_argument("--max-pool", type=int, default=MAX_POOL)
args = parser.parse_args()

try:
    if bool(args.records) == bool(args.pool):
        raise ValueError("Give exactly one of --records and --pool")
    fold = [float(x) for x in args.fold_thresholds.split(",") if x.strip()]
    site = [float(x) for x in args.site_thresholds.split(",") if x.strip()]
    log = lambda line: print(line, file=sys.stderr)
    if args.pool:
        report = run_diversity_pool(json.loads(args.pool.read_text()), args.out, fold, site, args.max_pool, log)
    else:
        sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
        if not sources or set(sources) - set(SOURCES):
            raise ValueError(f"--sources must be a subset of {', '.join(SOURCES)}")
        records = json.loads(args.records.read_text())["candidates"]
        target = json.loads(args.target_manifest.read_text())
        report = run_diversity(records, target, args.out, args.evaluation, args.batch_dir, sources, args.include_flagged, fold, site,
                               args.max_pool, log)
    print(json.dumps({"output_folder": str(args.out),
                      "sources": {s: (a["summary"] if "summary" in a else a.get("skipped")) for s, a in report["sources"].items()},
                      "left_out_as_flagged": len(report["left_out_as_flagged"])}, indent=2))
except (ValueError, FileNotFoundError, KeyError) as error:
    sys.exit(f"REFUSED: {error}")
