"""Novelty pre-screen against a PUBLIC sequence database with MMseqs2 (Phase 5, Task 5.1). Flags only; it cannot certify novelty.

The competition rule (as recorded by the team) is: a design counts as too similar if MMseqs2 finds a hit with at least 80 %
coverage and more than 70 % sequence identity, in an organizer-PRIVATE database. This module runs the same kind of search against a
public database that is available here and flags designs that would trip the rule on it. It is a cheap early warning, nothing more:

  * a flag is real information (the design resembles a known public protein);
  * the ABSENCE of a flag proves nothing, because the organizers' database is private and the exact command, the coverage mode and
    the identity definition they use are not confirmed;
  * `novelty_review` stays `pending` and `submission_eligible` stays false whatever this finds.

Interpretation used here (conservative): a hit trips the flag when query coverage >= 0.80 and identity (fraction of identical
aligned positions) > 0.70. Target coverage is recorded too and `both_coverages_ok` says whether the stricter reading (both
coverages >= 0.80, the MMseqs2 default coverage mode) also holds. The best hit by bit score is always recorded, flagged or not.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from .evaluate import select_candidates
from .runner import write_json

SCHEMA_VERSION = 1
COVERAGE = 0.80
IDENTITY = 0.70
FIELDS = "query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits"
COLUMNS = FIELDS.split(",")


def run_mmseqs(queries_fasta, database, out_path, binary="mmseqs", threads=4, sensitivity=7.5, evalue=1e-3, max_seqs=1000):
    """Search with MMseqs2 (easy-search) and write a tab-separated table with FIELDS."""
    with tempfile.TemporaryDirectory() as tmp:
        command = [binary, "easy-search", str(queries_fasta), str(database), str(out_path), tmp, "--format-output", FIELDS,
                   "-s", str(sensitivity), "-e", str(evalue), "--max-seqs", str(max_seqs), "--threads", str(threads),
                   "-v", "1"]
        done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError("MMseqs2 failed: " + (done.stderr or done.stdout).strip()[-400:])
    return command


def parse_hits(path):
    hits = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != len(COLUMNS):
            raise ValueError(f"Unexpected MMseqs2 output on line {line_number}")
        row = dict(zip(COLUMNS, parts))
        try:
            for key in ("fident", "qcov", "tcov", "evalue", "bits"):
                row[key] = float(row[key])
            for key in ("alnlen", "qlen", "tlen"):
                row[key] = int(row[key])
        except ValueError as error:
            raise ValueError(f"Unreadable number on line {line_number}") from error
        if not (0 <= row["fident"] <= 1 and 0 <= row["qcov"] <= 1 and 0 <= row["tcov"] <= 1):
            raise ValueError(f"Identity or coverage outside 0..1 on line {line_number}")
        hits.append(row)
    return hits


def judge(hits, coverage=COVERAGE, identity=IDENTITY):
    """Per query: the best hit, and whether any hit trips the rule."""
    by_query = {}
    for hit in hits:
        by_query.setdefault(hit["query"], []).append(hit)
    out = {}
    for query, rows in by_query.items():
        best = max(rows, key=lambda r: (r["bits"], r["fident"]))
        tripping = [r for r in rows if r["qcov"] >= coverage and r["fident"] > identity]
        worst = max(tripping, key=lambda r: (r["fident"], r["bits"])) if tripping else None
        out[query] = {"hits": len(rows), "best_hit": _summary(best), "tripping_hits": len(tripping),
                      "strongest_tripping_hit": _summary(worst) if worst else None,
                      "both_coverages_ok": bool(worst and worst["qcov"] >= coverage and worst["tcov"] >= coverage)}
    return out


def _summary(row):
    return {"target": row["target"], "identity": round(row["fident"], 4), "query_coverage": round(row["qcov"], 4),
            "target_coverage": round(row["tcov"], 4), "aligned_length": row["alnlen"], "target_length": row["tlen"],
            "evalue": row["evalue"], "bits": row["bits"]}


def prescreen(records, database, search=run_mmseqs, include_flagged=False, coverage=COVERAGE, identity=IDENTITY, workdir=None):
    """Return {candidate_id: result}. Only candidates without QC reasons are searched unless include_flagged."""
    chosen, skipped = select_candidates(records, include_flagged)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(workdir or tmp)
        queries, hits_path = tmp / "queries.fasta", tmp / "hits.tsv"
        queries.write_text("".join(f">{r['candidate_id']}\n{r['sequence']}\n" for r in chosen))
        search(queries, database, hits_path)
        judged = judge(parse_hits(hits_path), coverage, identity)
    results = {}
    for record in chosen:
        j = judged.get(record["candidate_id"])
        if j is None:
            results[record["candidate_id"]] = {"status": "no_public_hit", "hits": 0, "best_hit": None, "tripping_hits": 0,
                                               "strongest_tripping_hit": None, "both_coverages_ok": False}
        else:
            results[record["candidate_id"]] = {"status": "flagged" if j["tripping_hits"] else "no_tripping_hit", **j}
    for item in skipped:
        results[item["candidate_id"]] = {"status": "not_searched", "reasons": item["reasons"]}
    return results


def run_prescreen(records_path, database, out_dir, search=run_mmseqs, include_flagged=False, database_note="", log=print):
    """Pre-screen a candidates file. Writes novelty-candidates.json, novelty.csv and novelty-report.json into a fresh folder."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError("Use a fresh output directory")
    doc = json.loads(Path(records_path).read_text())
    records = doc["candidates"]
    results = prescreen(records, database, search, include_flagged)
    out_dir.mkdir(parents=True)
    updated = []
    for record in records:
        result = results[record["candidate_id"]]
        reasons = list(record.get("qc_reasons") or [])
        if result["status"] == "flagged":
            top = result["strongest_tripping_hit"]
            reasons.append(f"public_similarity:{top['target']}:{top['identity']:.2f}id_{top['query_coverage']:.2f}cov")
        updated.append({**record, "qc_reasons": list(dict.fromkeys(reasons)), "novelty_prescreen": {
            **result, "rule": f"query coverage >= {COVERAGE} and identity > {IDENTITY} (assumed reading of the organizer rule)",
            "database": database_note or str(database),
            "note": "Public database only. Cannot certify novelty; novelty_review stays pending."}})
        log(f"{record['candidate_id']}: {result['status']}")
    write_json(out_dir / "novelty-candidates.json", {**{k: v for k, v in doc.items() if k not in ("candidates", "count")},
                                                     "candidates": updated, "count": len(updated)})
    with (out_dir / "novelty.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["candidate_id", "status", "hits", "best_target", "best_identity", "best_query_coverage", "best_target_coverage",
                         "tripping_hits", "both_coverages_ok"])
        for item in updated:
            r = item["novelty_prescreen"]
            b = r.get("best_hit") or {}
            writer.writerow([item["candidate_id"], r["status"], r.get("hits", ""), b.get("target", ""), b.get("identity", ""),
                             b.get("query_coverage", ""), b.get("target_coverage", ""), r.get("tripping_hits", ""),
                             r.get("both_coverages_ok", "")])
    counts = {}
    for item in updated:
        s = item["novelty_prescreen"]["status"]
        counts[s] = counts.get(s, 0) + 1
    report = {"schema": SCHEMA_VERSION, "candidates": len(updated), "status_counts": counts,
              "rule": f"flag when query coverage >= {COVERAGE} and identity > {IDENTITY}", "database": database_note or str(database),
              "scope": "Public-database pre-screen only. A flag is informative; no flag proves nothing, because the organizers' "
                       "database is private and their exact search settings are unconfirmed. novelty_review stays pending."}
    write_json(out_dir / "novelty-report.json", report)
    return report
