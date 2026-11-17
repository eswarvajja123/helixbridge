"""Merge several generation batches into one candidate pool (Phase 3 support for Phase 5).

Each batch (Task 2.5) removes near-duplicates only within itself. A pool made of several batches, for example one per hotspot
option, therefore needs the same gate run again across all of them, in a fixed order, before the designs are scored and compared.

What this does:
  * copies each design's folder (structure, audit, record) into one self-contained pool folder and re-checks its SHA-256;
  * refuses batches made for a different target, generator revision or dedup thresholds, and duplicate candidate ids;
  * replays the near-duplicate gate over every design (batches in the order given, designs in the order each batch kept them);
    an earlier design always wins, so the first batch given wins ties;
  * writes `candidates.json` in the same shape the batch writes, with the paths rewritten to the pool folder and any
    `near_duplicate:*` reason recomputed (other reasons are kept); the batch's own decision is kept under `dedup_in_batch`.

Nothing is ranked or promoted: screen_status, novelty_review and submission_eligible are copied unchanged.
"""
import json
from pathlib import Path
import shutil

from .dedup import DedupGate, Thresholds, read_chain_ca
from .runner import digest, write_json

SCHEMA_VERSION = 1
DESIGN_FILES = ("model.cif", "audit.json", "record.json")
SAME_ACROSS_BATCHES = ("target_manifest_sha256", "generator_revision", "target_id")


def _load_batch(path):
    path = Path(path)
    candidates = json.loads((path / "candidates.json").read_text())
    state = json.loads((path / "state.json").read_text())
    return candidates["candidates"], state["identity"], digest(path / "candidates.json")


def _run_folder(record):
    structure = Path(record["generator_structure"])
    return structure.parent.name


def merge_batches(batch_dirs, out_dir, log=print):
    """Merge batch folders into a fresh pool folder. Returns the report dict (also written as pool-report.json)."""
    batch_dirs = [Path(b) for b in batch_dirs]
    out_dir = Path(out_dir)
    if not batch_dirs:
        raise ValueError("Give at least one batch folder")
    if len({b.resolve() for b in batch_dirs}) != len(batch_dirs):
        raise ValueError("The same batch folder was given twice")
    if out_dir.exists():
        raise ValueError("Use a fresh output directory")
    loaded = [(b, *_load_batch(b)) for b in batch_dirs]
    thresholds = {json.dumps(identity["thresholds"], sort_keys=True) for _, _, identity, _ in loaded}
    if len(thresholds) != 1:
        raise ValueError("The batches used different near-duplicate thresholds")
    thresholds = loaded[0][2]["thresholds"]
    ids_seen, runs_seen = set(), set()
    first = next((r for _, records, _, _ in loaded for r in records), None)
    for batch, records, identity, _ in loaded:
        for record in records:
            for key in SAME_ACROSS_BATCHES:
                if record[key] != first[key]:
                    raise ValueError(f"{batch} was made for a different {key.replace('_', ' ')}")
            run = _run_folder(record)
            if record["candidate_id"] in ids_seen or run in runs_seen:
                raise ValueError("Duplicate candidate id or run id across batches: " + record["candidate_id"])
            ids_seen.add(record["candidate_id"])
            runs_seen.add(run)

    out_dir.mkdir(parents=True)
    (out_dir / "designs").mkdir()
    gate = DedupGate(Thresholds(**thresholds))
    merged, per_batch, cross = [], [], []
    for batch, records, identity, source_sha in loaded:
        for record in records:
            run = _run_folder(record)
            source = batch / "designs" / run
            target = out_dir / "designs" / run
            target.mkdir()
            for name in DESIGN_FILES:
                if (source / name).is_file():
                    shutil.copyfile(source / name, target / name)
            if digest(target / "model.cif") != record["generator_structure_sha256"]:
                raise ValueError("A design's structure does not match its recorded SHA-256: " + run)
            audit = json.loads((target / "audit.json").read_text())
            if digest(target / "model.cif") != audit["generated_sha256"]:
                raise ValueError("A design's structure does not match its audit: " + run)
            sequence, coords = read_chain_ca(target / "model.cif", record["generator_binder_chain"])
            if sequence != record["sequence"]:
                raise ValueError("A design's structure holds a different binder sequence than its record: " + run)
            decision = gate.offer(record["candidate_id"], sequence, coords)
            reasons = [r for r in record.get("qc_reasons", []) if not r.startswith("near_duplicate:")]
            if not decision.kept:
                reasons.append("near_duplicate:" + decision.reason)
            new = dict(record)
            new["qc_reasons"] = reasons
            new["dedup_in_batch"] = record.get("dedup")
            new["dedup"] = decision.as_dict()
            new["generator_structure"] = str(target / "model.cif")
            new["source_audit"] = str(target / "audit.json")
            new["generator_audit"] = {**record["generator_audit"], "generated_structure": str(target / "model.cif")}
            new["pool"] = {"batch_id": identity["batch_id"], "batch_candidates_sha256": source_sha, "run_id": run,
                           "hotspot_source": record.get("hotspot_source")}
            merged.append(new)
            was_dropped_in_batch = bool((record.get("dedup") or {}).get("kept") is False)
            if not decision.kept and not was_dropped_in_batch:
                cross.append({"candidate_id": record["candidate_id"], "batch": identity["batch_id"], "reason": decision.reason,
                              "nearest": decision.nearest_id})
        per_batch.append({"batch_id": identity["batch_id"], "candidates": len(records), "hotspots": identity["hotspots"],
                          "hotspot_source": identity["hotspot_source"]})
        log(f"{identity['batch_id']}: {len(records)} designs")
    kept = sum(1 for r in merged if not any(x.startswith("near_duplicate:") for x in r["qc_reasons"]))
    write_json(out_dir / "candidates.json", {
        "candidates": merged, "count": len(merged),
        "note": "Pool of designs from several batches. Designs are copied with their structures; near-duplicates are re-checked across "
                "all batches (earlier batch wins). Nothing is ranked or promoted; statuses are unchanged."})
    report = {"schema": SCHEMA_VERSION, "batches": per_batch, "designs": len(merged), "kept_by_gate": kept,
              "dropped_by_gate": len(merged) - kept, "dropped_only_across_batches": cross,
              "thresholds": thresholds, "scope": "Near-duplicate gate only; not a novelty check and not a quality ranking."}
    write_json(out_dir / "pool-report.json", report)
    return report
