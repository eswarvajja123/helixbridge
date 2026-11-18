"""Compare design methods at matched GPU time (Phase 4, Task 4.3): diverse, viable designs per GPU-hour, failures kept.

The plan asks for a comparison per GPU-hour, not per design count, because the methods cost very different amounts (RFD4 about a minute
a design, Mosaic about ten to sixteen). For each method this adds up the GPU seconds actually reported (generation, from the batch
reports, and scoring, from the evaluation reports), counts what the structure QC calls clean, and counts the distinct fold and site
groups among the clean designs (from a diversity report), then divides by the GPU hours. It also scales a method's rate to the other
method's GPU budget, so the two can be read at the same spend.

What the numbers are and are not:
  * "viable" means: no structure-QC flags (the team's provisional limits). It is not "binds" and not a quality ranking;
  * Mosaic optimizes THROUGH Boltz-2, so its Boltz-2 based checks are not independent. ESMFold2-only agreement is reported separately,
    so the comparison can be read on the independent model alone;
  * small counts mean wide uncertainty; every figure is shown with its count, and failed attempts stay in the denominator of the GPU
    time (their seconds are in the reports) and in the failure counts;
  * a design counts as generated only if the batch finished it; failed and interrupted attempts are listed, never dropped. The GPU seconds
    come from the function's own count for FINISHED designs, so the seconds of a failed or interrupted attempt are not in them: the number
    of attempts that produced no design is reported so the reader can see how much time that may hide.
"""
import csv
import json
from pathlib import Path
import statistics

from .runner import write_json

SCHEMA_VERSION = 1
SOURCES = ("generator", "esmfold2", "boltz2")


def _seconds(reports, key):
    return sum(float(r[key]) for r in reports)


def summarize(label, batch_reports, evaluation_reports, qc_records, diversity=None, fold_threshold="0.5", site_threshold="0.5"):
    """One method's row. `qc_records` are the records written by qc-candidates.py (with a `qc` block)."""
    designs = sum(r["done"] for r in batch_reports)
    failed = sum(len(r["failed"]) for r in batch_reports)
    interrupted = sum(len(r["interrupted"]) for r in batch_reports)
    generation_s = _seconds(batch_reports, "gpu_seconds_reported")
    scored = sum(r["by_predictor"]["esmfold2"]["done"] for r in evaluation_reports)
    scoring_s = _seconds(evaluation_reports, "gpu_seconds_reported")
    total_s = generation_s + scoring_s
    status = {"no_flags": 0, "flagged": 0, "incomplete": 0}
    esm_only = 0
    for record in qc_records:
        qc = record["qc"]
        status[qc["status"]] += 1
        independent = [x for x in qc["reasons"] if ":boltz2" not in x and "_boltz2" not in x and "boltz2_vs" not in x
                       and not x.startswith("clash:generator")]
        if qc["status"] != "incomplete" and not independent:
            esm_only += 1                                   # clean when every Boltz-2-related flag is ignored
    attempts = sum(r.get("attempts", r["done"] + len(r["failed"]) + len(r["interrupted"])) for r in batch_reports)
    row = {"label": label, "designs_generated": designs, "attempts_total": attempts, "attempts_not_producing_a_design": attempts - designs,
           "attempts_failed": failed, "attempts_interrupted": interrupted,
           "designs_scored": scored, "gpu_seconds_generation": round(generation_s, 1), "gpu_seconds_scoring": round(scoring_s, 1),
           "gpu_seconds_total": round(total_s, 1), "gpu_seconds_per_design_generation": round(generation_s / designs, 1) if designs else None,
           "gpu_seconds_per_design_total": round(total_s / designs, 1) if designs else None,
           "qc_clean": status["no_flags"], "qc_flagged": status["flagged"], "qc_incomplete": status["incomplete"],
           "qc_clean_share": round(status["no_flags"] / max(1, status["no_flags"] + status["flagged"]), 3),
           "clean_if_boltz2_flags_are_ignored": esm_only}
    if diversity is not None and "generator" in diversity["sources"] and "summary" in diversity["sources"]["generator"]:
        s = diversity["sources"]["generator"]["summary"]
        row["distinct_fold_groups_among_clean"] = s["fold"][fold_threshold]["complete_link_greedy"]["groups"]
        row["distinct_site_groups_among_clean"] = s["site"][site_threshold]["complete_link_greedy"]["groups"]
        row["designs_in_diversity_report"] = len(diversity["sources"]["generator"]["members"])
    else:
        row["distinct_fold_groups_among_clean"] = row["distinct_site_groups_among_clean"] = row["designs_in_diversity_report"] = None
    if row["qc_clean"] == 0 and row["distinct_fold_groups_among_clean"] is None:
        row["distinct_fold_groups_among_clean"] = row["distinct_site_groups_among_clean"] = 0         # no clean design, so no groups
    for predictor in ("esmfold2", "boltz2"):                                                           # raw scores, described, never ranked
        values = [r["evaluations"][predictor]["score"]["iptm"] for r in qc_records
                  if (r.get("evaluations") or {}).get(predictor, {}).get("status") == "done"]
        row[f"{predictor}_iptm"] = ({"n": len(values), "min": round(min(values), 3), "median": round(statistics.median(values), 3),
                                     "max": round(max(values), 3)} if values else None)
    hours = total_s / 3600.0
    if hours > 0:
        row["per_gpu_hour"] = {"designs": round(designs / hours, 2), "qc_clean": round(status["no_flags"] / hours, 2),
                               "distinct_fold_groups_among_clean": (round(row["distinct_fold_groups_among_clean"] / hours, 2)
                                                                    if row["distinct_fold_groups_among_clean"] is not None else None)}
    else:
        row["per_gpu_hour"] = None
    return row


def at_budget(row, gpu_seconds):
    """What `row`'s method would be expected to give at `gpu_seconds`, assuming its measured rate holds (a scaling, not a run)."""
    if not row["per_gpu_hour"] or gpu_seconds <= 0:
        return None
    hours = gpu_seconds / 3600.0
    return {"gpu_seconds": round(gpu_seconds, 1), "designs": round(row["per_gpu_hour"]["designs"] * hours, 1),
            "qc_clean": round(row["per_gpu_hour"]["qc_clean"] * hours, 1),
            "distinct_fold_groups_among_clean": (round(row["per_gpu_hour"]["distinct_fold_groups_among_clean"] * hours, 1)
                                                 if row["per_gpu_hour"]["distinct_fold_groups_among_clean"] is not None else None),
            "note": "scaled from the measured rate; distinct-group counts do not scale linearly, so read this as an order of magnitude"}


def compare(rows, out_dir):
    """Write comparison.json, comparison.csv and comparison.md for two or more method rows, each scaled to the others' budgets."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError("Use a fresh output directory")
    if len(rows) < 2 or len({r["label"] for r in rows}) != len(rows):
        raise ValueError("Give two or more methods with distinct labels")
    out_dir.mkdir(parents=True)
    scaled = {r["label"]: {other["label"]: at_budget(r, other["gpu_seconds_total"]) for other in rows if other is not r} for r in rows}
    write_json(out_dir / "comparison.json", {"schema": SCHEMA_VERSION, "methods": rows, "at_the_other_methods_budget": scaled,
                                             "scope": "Counts of designs without structure-QC flags per GPU hour. Not binding, not a ranking of designs; "
                                                      "Boltz-2 based checks are not independent for a method that optimizes through Boltz-2."})
    columns = ["label", "designs_generated", "attempts_total", "attempts_not_producing_a_design", "attempts_failed", "attempts_interrupted", "designs_scored", "gpu_seconds_generation",
               "gpu_seconds_scoring", "gpu_seconds_total", "gpu_seconds_per_design_generation", "gpu_seconds_per_design_total", "qc_clean",
               "qc_flagged", "qc_incomplete", "qc_clean_share", "clean_if_boltz2_flags_are_ignored", "distinct_fold_groups_among_clean",
               "distinct_site_groups_among_clean"]
    extra = [("esmfold2_iptm_median", lambda r: (r["esmfold2_iptm"] or {}).get("median", "")),
             ("boltz2_iptm_median", lambda r: (r["boltz2_iptm"] or {}).get("median", ""))]
    with (out_dir / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns + [n for n, _ in extra] + ["designs_per_gpu_hour", "qc_clean_per_gpu_hour", "fold_groups_per_gpu_hour"])
        for r in rows:
            p = r["per_gpu_hour"] or {}
            writer.writerow([r[c] for c in columns] + [f(r) for _, f in extra]
                            + [p.get("designs", ""), p.get("qc_clean", ""), p.get("distinct_fold_groups_among_clean", "")])
    lines = ["| method | designs | attempts that made no design | GPU s per design (total) | clean by QC | clean per GPU-hour | fold groups among clean | ESMFold2 ipTM median | Boltz-2 ipTM median |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p = r["per_gpu_hour"] or {}
        lines.append(f"| {r['label']} | {r['designs_generated']} | {r['attempts_not_producing_a_design']} | {r['gpu_seconds_per_design_total']} | "
                     f"{r['qc_clean']} of {r['qc_clean'] + r['qc_flagged']} | {p.get('qc_clean', '')} | {r['distinct_fold_groups_among_clean']} | "
                     f"{(r['esmfold2_iptm'] or {}).get('median', '')} | {(r['boltz2_iptm'] or {}).get('median', '')} |")
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    return {"methods": rows, "at_the_other_methods_budget": scaled}
