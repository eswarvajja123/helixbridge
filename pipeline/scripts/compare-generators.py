"""Compare design methods per GPU-hour (Task 4.3). CPU only.

One --method per design method: LABEL=BATCH_REPORTS|EVAL_REPORTS|QC_RECORDS|DIVERSITY, where the first two fields are comma-separated
lists of report files and the last two are single files (DIVERSITY may be empty). Example:
  python pipeline/scripts/compare-generators.py --out results/compare-eswar-c1 \
    --method "RFD4=batch-a/batch-report.json,batch-b/batch-report.json|eval-e3/evaluation-report.json|qc-q3/qc-candidates.json|diversity-d5/diversity.json" \
    --method "Mosaic=batch-m/batch-report.json|eval-m/evaluation-report.json|qc-m/qc-candidates.json|diversity-m/diversity.json"
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.comparison import compare, summarize

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--method", action="append", required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
try:
    rows = []
    for spec in args.method:
        label, _, rest = spec.partition("=")
        parts = rest.split("|")
        if not label or len(parts) != 4:
            raise ValueError("Each --method is LABEL=BATCH_REPORTS|EVAL_REPORTS|QC_RECORDS|DIVERSITY")
        load = lambda p: json.loads(Path(p).read_text())
        batches = [load(p) for p in parts[0].split(",") if p]
        evaluations = [load(p) for p in parts[1].split(",") if p]
        qc_records = load(parts[2])["candidates"]
        diversity = load(parts[3]) if parts[3] else None
        rows.append(summarize(label, batches, evaluations, qc_records, diversity))
    result = compare(rows, args.out)
    print(json.dumps({"output_folder": str(args.out), "methods": [{k: r[k] for k in ("label", "designs_generated", "gpu_seconds_total", "qc_clean",
                      "qc_flagged", "per_gpu_hour")} for r in result["methods"]]}, indent=2))
except (ValueError, FileNotFoundError, KeyError) as error:
    sys.exit(f"REFUSED: {error}")
