"""Build the public sequence database for the novelty pre-screen (Task 5.1). Needs internet and the `mmseqs` program.

Downloads UniProtKB/Swiss-Prot and the PDB's protein sequences (about 210 MB), builds an MMseqs2 database and writes a manifest with
hashes and counts. Takes a few minutes. Install MMseqs2 first (Linux: `apt install mmseqs2`; Windows/macOS: see mmseqs.com).

Example:
  python pipeline/scripts/build-public-db.py --out data/public-db
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.public_db import build

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=Path, required=True, help="new folder for the database")
parser.add_argument("--mmseqs", default="mmseqs", help="the mmseqs program")
args = parser.parse_args()
try:
    manifest = build(args.out, args.mmseqs, log=lambda line: print(line, file=sys.stderr))
    print(json.dumps({k: manifest[k] for k in ("swissprot", "pdb_protein_chains", "pdb_unique_sequences", "mmseqs_version", "database")}, indent=2))
except (ValueError, RuntimeError, OSError) as error:
    sys.exit(f"REFUSED: {error}")
