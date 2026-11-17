"""Novelty pre-screen of a candidate pool against a PUBLIC sequence database (Task 5.1). CPU only. Flags only; cannot certify novelty.

A design is flagged when MMseqs2 finds a public hit with query coverage >= 0.80 and identity > 0.70 (the team's recorded reading of the
organizers' rule, applied to a public database). No flag does NOT mean novel: the organizers' database is private.

Example:
  python pipeline/scripts/novelty-prescreen.py --records results/pool-eswar-p1/candidates.json --database data/public-db/publicDB \
        --out results/novelty-eswar-n1
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.novelty import run_mmseqs, run_prescreen

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--records", type=Path, required=True)
parser.add_argument("--database", type=Path, required=True, help="MMseqs2 database prefix (e.g. data/public-db/publicDB)")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--mmseqs", default="mmseqs")
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--include-flagged", action="store_true", help="also search designs that already carry QC reasons")
args = parser.parse_args()
try:
    if not Path(str(args.database) + ".dbtype").is_file():
        raise ValueError(f"No MMseqs2 database at {args.database} (build one with build-public-db.py)")
    manifest = args.database.parent / "public-db-manifest.json"
    note = (json.loads(manifest.read_text()) if manifest.is_file() else {}).get("scope", str(args.database))
    note = f"{args.database} ({note})"
    search = lambda q, d, o: run_mmseqs(q, d, o, binary=args.mmseqs, threads=args.threads)
    report = run_prescreen(args.records, args.database, args.out, search, args.include_flagged, note,
                           log=lambda line: print(line, file=sys.stderr))
    print(json.dumps({"output_folder": str(args.out), **{k: report[k] for k in ("candidates", "status_counts", "rule")}}, indent=2))
except (ValueError, RuntimeError, FileNotFoundError, KeyError) as error:
    sys.exit(f"REFUSED: {error}")
