"""Structure QC for candidates (Phase 3, Task 3.3): length, single chain, clash, contact and pose shift.

Reads the structures the pipeline already holds for each candidate:
  * `generator`  the RFD4 structure saved by the generation batch (target core + the designed binder);
  * `esmfold2`, `boltz2`  the refolded complexes saved by the evaluation (full target 21-239 + binder);
and applies the team's own geometric flags through its `pose_review.py` (contacts, close heavy-atom pairs, target-aligned binder
pose), adding the missing checks (length, chain count, hotspot contact) around them. Everything that trips is written as a short
reason string into a COPY of the candidate record's `qc_reasons`; the originals are never altered.

What these flags are, and are not:
  * they are geometric consistency checks, not calibrated filters and not binding predictions;
  * the numeric limits are the team's own provisional ones (outputs/lead101-repair-protocol: zero heavy-atom pairs < 2 A, binder
    pose within 2.5 A of the generated pose and within 3.0 A between the two refolds, after aligning on the shared reference core).
    The team recorded them as "a predeclared consistency screen, not a validated experimental threshold". They are parameters here,
    so a different rule is one argument away, and every raw measurement is stored next to the reasons;
  * the team's pilot found high-scoring non-binders with zero close pairs, so "no flags" is never evidence that a design is good.

A candidate with no flags but a missing structure is reported as `incomplete`, never as clean.
"""
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from .runner import digest, write_json
from .scoring import PREDICTORS

SCHEMA_VERSION = 1
SOURCES = ("generator",) + PREDICTORS
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Thresholds:
    max_close_pairs: int = 0                      # interchain heavy-atom pairs below 2 A (the 2 A cutoff is fixed in pose_review)
    min_target_contact_residues: int = 1          # target residues within 5 A of the binder
    require_hotspot_contact: bool = True          # the structure touches at least one hotspot the design was conditioned on
    max_pose_shift_generator_A: float = 2.5       # binder CA RMSD, refold vs generated pose, target core aligned
    max_pose_shift_between_predictors_A: float = 3.0   # same, ESMFold2 vs Boltz-2

    def __post_init__(self):
        for name in ("max_close_pairs", "min_target_contact_residues"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("max_pose_shift_generator_A", "max_pose_shift_between_predictors_A"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive number")
        if not isinstance(self.require_hotspot_contact, bool):
            raise ValueError("require_hotspot_contact must be true or false")

    def as_dict(self):
        return dict(self.__dict__)


def _short(error):
    text = " ".join(str(error).split())
    return (text[:100] + "...") if len(text) > 100 else text


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def chain_sequences(path):
    """{chain id: sequence} from the CA atoms of an mmCIF, in file order. Refuses files with several models."""
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from .structure import AA3
    data = MMCIF2Dict(str(path))
    models = set(data.get("_atom_site.pdbx_PDB_model_num", []))
    if len(models) > 1:
        raise ValueError("The structure has several models")
    ids, names, comps = data["_atom_site.label_asym_id"], data["_atom_site.label_atom_id"], data["_atom_site.label_comp_id"]
    out = {}
    for chain, name, comp in zip(ids, names, comps):
        if name == "CA":
            if comp not in AA3:
                raise ValueError("Nonstandard residue: " + comp)
            out[chain] = out.get(chain, "") + AA3[comp]
        else:
            out.setdefault(chain, "")
    return out


def inspect_structure(path, source, target_sequence, binder_sequence, positions, expected_sha256=None):
    """Read one complex with the team's `read_pair`. Returns (pose or None, [reasons])."""
    path = Path(path)
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        return None, [f"structure_hash_mismatch:{source}"]
    try:
        chains = chain_sequences(path)
    except (ValueError, KeyError, OSError) as error:
        return None, [f"structure_unreadable:{source}:{_short(error)}"]
    if len(chains) != 2:
        return None, [f"unexpected_chains:{source}:found_{len(chains)}_expected_2"]
    reasons = []
    if sum(1 for s in chains.values() if s == binder_sequence) != 1:
        reasons.append(f"binder_sequence_mismatch:{source}")
    if sum(1 for s in chains.values() if s == target_sequence) != 1:
        reasons.append(f"target_sequence_mismatch:{source}")
    if reasons:
        return None, reasons
    from .pose_review import read_pair
    try:
        return read_pair(path, target_sequence, binder_sequence, list(positions)), []
    except (ValueError, KeyError, IndexError) as error:
        return None, [f"structure_unreadable:{source}:{_short(error)}"]


def _measure(pose, hotspots):
    contacts = [int(p) for p in pose["target_contacts_lt_5A"]]
    out = {"status": "checked", "target_contact_residues": len(contacts), "target_contacts": contacts,
           "hotspot_contacts": sorted(set(contacts) & set(hotspots)) if hotspots is not None else None,
           "heavy_atom_pairs_lt_2A": int(pose["heavy_atom_pairs_lt_2A"]), "min_heavy_distance_A": round(float(pose["min_heavy_distance_A"]), 4)}
    return out


def _flags(source, measured, thresholds):
    reasons = []
    if measured["heavy_atom_pairs_lt_2A"] > thresholds.max_close_pairs:
        reasons.append(f"clash:{source}:{measured['heavy_atom_pairs_lt_2A']}_heavy_atom_pairs_lt_2A")
    if measured["target_contact_residues"] < thresholds.min_target_contact_residues:
        reasons.append(f"no_target_contact:{source}")
    elif (thresholds.require_hotspot_contact and measured["hotspot_contacts"] is not None
            and not measured["hotspot_contacts"]):
        reasons.append(f"no_hotspot_contact:{source}")
    return reasons


def _shift(fixed, moving, core):
    from .pose_review import align_pair
    result = align_pair(fixed, moving, core)
    return round(float(result["target_aligned_binder_ca_rmsd_A"]), 4), round(float(result["target_fit_rmsd_A"]), 4)


def check_candidate(record, structures, target, thresholds=None, expected=SOURCES):
    """QC one candidate.

    A record may carry `generator_structure_sequence`: for a sequence-repaired variant the generator structure still holds the parent's
    binder sequence, and that is what the generator source is checked against.
    `structures` maps a source name ('generator', 'esmfold2', 'boltz2') to {"path": ..., "sha256": ... or None}; a source that is
    absent counts as not checked. `target` is the target manifest (sequence, start, end, min_length, max_length).
    Returns the `qc` dict: measurements, reasons and a status of `flagged`, `no_flags` or `incomplete`.
    """
    thresholds = thresholds or Thresholds()
    sequence = record["sequence"]
    reasons = []
    if not target["min_length"] <= len(sequence) <= target["max_length"]:
        reasons.append(f"length_out_of_range:{len(sequence)}")
    hotspots = record.get("hotspots_uniprot")
    reference = (record.get("generator_audit") or {}).get("reference_audit") or {}
    refold_positions = list(range(target["start"], target["end"] + 1))
    poses, sources = {}, {}
    for source in SOURCES:
        if source not in expected:
            continue
        info = structures.get(source)
        if info is None:
            sources[source] = {"status": "not_checked"}
            continue
        if source == "generator":
            seq, positions = reference.get("reference_sequence"), reference.get("positions")
            if not seq or not positions:
                sources[source] = {"status": "unreadable"}
                reasons.append("structure_unreadable:generator:the record has no reference audit")
                continue
        else:
            seq, positions = target["sequence"], refold_positions
        structure_binder = (record.get("generator_structure_sequence") or sequence) if source == "generator" else sequence
        pose, problems = inspect_structure(info["path"], source, seq, structure_binder, positions, info.get("sha256"))
        if pose is None:
            sources[source] = {"status": "unreadable"}
            reasons += problems
            continue
        sources[source] = _measure(pose, hotspots)
        poses[source] = pose
        reasons += _flags(source, sources[source], thresholds)
    core = sorted(set(reference.get("positions") or []) & set(refold_positions))
    shifts = {}
    if poses and core:
        if len(core) < 30:
            reasons.append(f"alignment_core_too_small:{len(core)}")
        else:
            pairs = [("generator", p, thresholds.max_pose_shift_generator_A) for p in PREDICTORS if p in poses and "generator" in poses]
            pairs += [(a, b, thresholds.max_pose_shift_between_predictors_A) for i, a in enumerate(PREDICTORS)
                      for b in PREDICTORS[i + 1:] if a in poses and b in poses]
            for fixed, moving, limit in pairs:
                name = f"{fixed}_vs_{moving}"
                try:
                    rmsd, fit = _shift(poses[fixed], poses[moving], core)
                except (ValueError, KeyError, IndexError) as error:
                    shifts[name] = {"status": "failed"}
                    reasons.append(f"alignment_check_failed:{name}:{_short(error)}")
                    continue
                shifts[name] = {"binder_ca_rmsd_A": rmsd, "target_core_fit_rmsd_A": fit, "limit_A": limit}
                if rmsd > limit:
                    reasons.append(f"pose_shift:{name}:{rmsd:.2f}A>{limit}A")
    unchecked = [s for s in expected if sources.get(s, {"status": "not_checked"})["status"] == "not_checked"]
    reasons = list(dict.fromkeys(reasons))
    status = "flagged" if reasons else ("incomplete" if unchecked else "no_flags")
    return {"schema": SCHEMA_VERSION, "status": status, "reasons": reasons, "length": len(sequence), "sources": sources,
            "pose_shift": shifts, "alignment_core": [core[0], core[-1]] if core else None, "unchecked": unchecked,
            "note": "Geometric consistency flags with provisional limits; not a binding prediction. No flags is not evidence of a good design."}


def structures_for(item, evaluation_dir, batch_dir=None, expected=SOURCES):
    """Locate the structures of one evaluated record, with the hash each was stored under."""
    out = {}
    audit = item.get("generator_audit") or {}
    if "generator" in expected and item.get("generator_structure"):
        path = Path(item["generator_structure"])
        if not path.is_file() and batch_dir is not None:
            path = Path(batch_dir) / "designs" / path.parent.name / "model.cif"
        if path.is_file():
            out["generator"] = {"path": path, "sha256": audit.get("generated_sha256")}
    for predictor in PREDICTORS:
        entry = (item.get("evaluations") or {}).get(predictor) or {}
        if predictor not in expected or entry.get("status") != "done" or evaluation_dir is None:
            continue
        path = Path(evaluation_dir) / "evals" / item["candidate_id"] / predictor / "model.cif"
        if path.is_file():
            out[predictor] = {"path": path, "sha256": (entry.get("artifacts") or {}).get("model.cif")}
    return out


def run_qc(evaluation_dir, target_manifest, out_dir, thresholds=None, batch_dir=None, expected=SOURCES, log=print):
    """QC every record of an evaluation folder. Writes qc-candidates.json, qc.csv and qc-report.json into a fresh `out_dir`."""
    thresholds = thresholds or Thresholds()
    evaluation_dir, out_dir = Path(evaluation_dir), Path(out_dir)
    stored = json.loads((evaluation_dir / "evaluations.json").read_text())
    target = json.loads(Path(target_manifest).read_text())
    if out_dir.exists():
        raise ValueError("Use a fresh output directory")
    for item in stored["candidates"]:
        if "qc" in item:
            raise ValueError("A record already holds a QC result; run QC on the evaluation folder, not on a QC output")
    checked = []
    for item in stored["candidates"]:
        qc = check_candidate(item, structures_for(item, evaluation_dir, batch_dir, expected), target, thresholds, expected)
        merged = list(dict.fromkeys(list(item.get("qc_reasons") or []) + qc["reasons"]))
        checked.append({**item, "qc_reasons": merged, "qc": qc})
        log(f"{item['candidate_id']}: {qc['status']}" + (": " + "; ".join(qc["reasons"]) if qc["reasons"] else ""))
    out_dir.mkdir(parents=True)
    write_json(out_dir / "qc-candidates.json", {**{k: v for k, v in stored.items() if k not in ("candidates", "count")},
               "candidates": checked, "count": len(checked),
               "qc_note": "Copies of the evaluated records with qc_reasons extended and a qc block added. Originals are untouched; "
                          "screen_status, novelty_review and submission_eligible are unchanged."})
    _write_csv(out_dir / "qc.csv", checked, expected)
    counts, histogram = {"flagged": 0, "no_flags": 0, "incomplete": 0}, {}
    for item in checked:
        counts[item["qc"]["status"]] += 1
        for reason in item["qc"]["reasons"]:
            code = reason.split(":")
            key = code[0] if code[0] in ("length_out_of_range", "alignment_core_too_small") else ":".join(code[:2])
            histogram[key] = histogram.get(key, 0) + 1
    report = {"candidates": len(checked), "status_counts": counts, "reason_counts": dict(sorted(histogram.items())),
              "thresholds": thresholds.as_dict(), "target_length_range": [target["min_length"], target["max_length"]],
              "sources_expected": list(expected), "pose_review_sha256": digest(HERE / "pose_review.py"),
              "scope": "Geometric consistency flags only. Provisional limits from the team's own consistency screen, not validated "
                       "thresholds; no flags is not evidence that a design is good (the team's pilot had high-scoring non-binders with "
                       "zero close pairs). Nothing is ranked or promoted."}
    write_json(out_dir / "qc-report.json", report)
    return report


def _write_csv(path, checked, expected):
    columns = ["candidate_id", "qc_status", "length", "reasons"]
    for source in expected:
        columns += [f"{source}_{k}" for k in ("status", "target_contact_residues", "hotspot_contacts", "heavy_atom_pairs_lt_2A",
                                              "min_heavy_distance_A")]
    columns += ["generator_vs_esmfold2_A", "generator_vs_boltz2_A", "esmfold2_vs_boltz2_A"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in checked:
            qc = item["qc"]
            row = {"candidate_id": item["candidate_id"], "qc_status": qc["status"], "length": qc["length"], "reasons": ";".join(qc["reasons"])}
            for source in expected:
                s = qc["sources"].get(source, {"status": "not_checked"})
                row[f"{source}_status"] = s["status"]
                for k in ("target_contact_residues", "heavy_atom_pairs_lt_2A", "min_heavy_distance_A"):
                    row[f"{source}_{k}"] = s.get(k, "")
                hot = s.get("hotspot_contacts")
                row[f"{source}_hotspot_contacts"] = "" if hot is None else " ".join(map(str, hot))
            for name in ("generator_vs_esmfold2", "generator_vs_boltz2", "esmfold2_vs_boltz2"):
                row[name + "_A"] = qc["pose_shift"].get(name, {}).get("binder_ca_rmsd_A", "")
            writer.writerow(row)
