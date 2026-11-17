"""Sequence liabilities for a candidate pool (Phase 5, Task 5.2), built on the team's `sequence_review.sequence_diagnostic`.

Records, per design, the team's descriptive diagnostics (cysteines, N-glycosylation sequons, Asn-Gly and Asp-Pro motifs, D/E, K/R and
hydrophobic fractions, longest identical run, entropy) and derives a short list of CATEGORICAL flags from them. There are deliberately
no numeric pass/fail thresholds: none is supportable (the team noted a 55 % D/E design as a concern but set no limit), so the numbers
are recorded and a person decides. The flags are information for selection; they do NOT add qc_reasons and do not remove a design.

Flags: odd_cysteine_count (an unpaired Cys), n_glycosylation_sequon (N-X-S/T, X not P), asn_gly_motif (NG, deamidation-prone),
asp_pro_motif (DP, acid-labile bond). Optional sequence repair (the lead101 SolubleMPNN pattern) is NOT done here.
"""
import csv
import json
from pathlib import Path
import statistics

from .runner import write_json
from .sequence_review import sequence_diagnostic

SCHEMA_VERSION = 1
FLAGS = ("odd_cysteine_count", "n_glycosylation_sequon", "asn_gly_motif", "asp_pro_motif")
NUMBERS = ("DE_fraction", "KR_fraction", "hydrophobic_AVILMFWY_fraction", "longest_identical_run", "whole_sequence_shannon_entropy_bits")


def flags_for(diagnostic):
    out = []
    if diagnostic["cysteine_count_odd"]:
        out.append("odd_cysteine_count")
    if diagnostic["N_glycosylation_sequons"]:
        out.append("n_glycosylation_sequon")
    if diagnostic["Asn_Gly_motifs"]:
        out.append("asn_gly_motif")
    if diagnostic["Asp_Pro_motifs"]:
        out.append("asp_pro_motif")
    return out


def review(records):
    """{candidate_id: {"diagnostic": ..., "flags": [...]}}; a sequence the team's code cannot diagnose is reported, not skipped."""
    out = {}
    for record in records:
        try:
            diagnostic = sequence_diagnostic(record["sequence"])
        except ValueError as error:
            out[record["candidate_id"]] = {"diagnostic": None, "flags": ["not_diagnosable"], "error": str(error)}
            continue
        out[record["candidate_id"]] = {"diagnostic": diagnostic, "flags": flags_for(diagnostic)}
    return out


def run_review(records_path, out_dir, log=print):
    """Write liabilities-candidates.json (records plus a `liabilities` block), liabilities.csv and liabilities-report.json."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError("Use a fresh output directory")
    doc = json.loads(Path(records_path).read_text())
    records = doc["candidates"]
    results = review(records)
    out_dir.mkdir(parents=True)
    updated = [{**r, "liabilities": {**results[r["candidate_id"]], "note": "Descriptive motifs and counts only; no numeric thresholds, "
                                     "no expression, aggregation or binding prediction. Flags do not change qc_reasons."}} for r in records]
    write_json(out_dir / "liabilities-candidates.json", {**{k: v for k, v in doc.items() if k not in ("candidates", "count")},
                                                         "candidates": updated, "count": len(updated)})
    with (out_dir / "liabilities.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["candidate_id", "length", "flags", "cysteines", "n_sequons", "NG", "DP", *NUMBERS])
        for item in updated:
            d = item["liabilities"]["diagnostic"]
            if d is None:
                writer.writerow([item["candidate_id"], len(item["sequence"]), ";".join(item["liabilities"]["flags"])])
                continue
            writer.writerow([item["candidate_id"], d["length"], ";".join(item["liabilities"]["flags"]), len(d["cysteine_positions"]),
                             len(d["N_glycosylation_sequons"]), len(d["Asn_Gly_motifs"]), len(d["Asp_Pro_motifs"]),
                             *[round(d[k], 4) if isinstance(d[k], float) else d[k] for k in NUMBERS]])
    counts = {flag: sum(1 for r in results.values() if flag in r["flags"]) for flag in FLAGS + ("not_diagnosable",)}
    spread = {}
    for key in NUMBERS:
        values = [r["diagnostic"][key] for r in results.values() if r["diagnostic"]]
        spread[key] = ({"n": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)} if values else None)
    report = {"schema": SCHEMA_VERSION, "candidates": len(updated), "flag_counts": counts,
              "designs_with_no_flags": sum(1 for r in results.values() if not r["flags"]), "spread": spread,
              "scope": "Descriptive motifs and counts only. No numeric pass/fail thresholds, and no repair was attempted."}
    write_json(out_dir / "liabilities-report.json", report)
    for item in updated:
        log(f"{item['candidate_id']}: {','.join(item['liabilities']['flags']) or 'no flags'}")
    return report
