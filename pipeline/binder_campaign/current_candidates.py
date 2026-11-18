"""Import one conditioned RFD4 generation as a candidate record (Phase 2, Task 2.3).

The counterpart of candidates.import_rfd4 for the current competition target. It reads the audit written
by scripts/rfd4-generate.py, re-checks it against the target manifest and the saved structure, and writes a
record that candidates.export_draft will accept ONLY after the later reviews are filled in. Every record starts
as not eligible, unreviewed and not screened; importing never promotes a design to an entry.

Lives in its own module so the team's candidates.py stays untouched.
"""
import json
from pathlib import Path
import re

from .generation_input import ALLOWED_ROLES, check_generated_chains
from .runner import digest, write_json

SCHEMA_VERSION = 1


def _decision_dict(decision):
    if decision is None:
        return None
    return decision.as_dict() if hasattr(decision, "as_dict") else dict(decision)


def import_current(audit_path, target_manifest, output, dedup_decision=None, require_reviewed_hotspots=False):
    """Write one candidate record for a generation audit.

    `dedup_decision` is a dedup.Decision (or its dict). With `require_reviewed_hotspots=True`, a design made from
    unreviewed hotspots gets the QC reason `hotspots_not_reviewed`, which keeps it out of export_draft until the
    epitope has been reviewed. The default only records the status.
    """
    audit_path, target_manifest, output = Path(audit_path), Path(target_manifest), Path(output)
    if output.exists():
        raise ValueError("Use a fresh output file")
    audit = json.loads(audit_path.read_text())
    target = json.loads(target_manifest.read_text())
    if not str(audit.get("status", "")).startswith("conditioned_generation"):
        raise ValueError("Not a conditioned-generation audit (was it written by rfd4-generate.py?)")
    reference = audit["reference_audit"]
    if reference.get("role") not in ALLOWED_ROLES or reference["target_id"] != target["id"]:
        raise ValueError("The audited reference does not match the candidate target")
    if audit["target_manifest_sha256"] != digest(target_manifest):
        raise ValueError("The target manifest differs from the one used for generation")
    structure = Path(audit["generated_structure"])
    if digest(structure) != audit["generated_sha256"]:
        raise ValueError("Generated structure changed after its identity check")
    target_chain, binder_chain = check_generated_chains(audit["chains"], reference["reference_sequence"],
                                                        audit["binder_length_requested"])
    if ((target_chain["id"], binder_chain["id"]) != (audit["target_chain"], audit["binder_chain"])
            or binder_chain["sequence"] != audit["binder_sequence"]):
        raise ValueError("Audit chains disagree with its recorded target and binder")

    sequence = audit["binder_sequence"]
    reasons = []
    if not target["min_length"] <= len(sequence) <= target["max_length"]:
        reasons.append("wrong_length")
    if target["format"] != "minibinder":
        reasons.append("format_and_framework_review_pending")
    decision = _decision_dict(dedup_decision)
    if decision is not None and not decision["kept"]:
        reasons.append("near_duplicate:" + decision["reason"])
    source = reference.get("hotspot_source")
    hotspot_review = "reviewed" if str(source).startswith("reviewed_by_") else "unreviewed"
    if require_reviewed_hotspots and hotspot_review != "reviewed":
        reasons.append("hotspots_not_reviewed")
    identity = digest(audit_path)[:12]
    family = audit.get("generator_family", "rfd4")              # a Mosaic design says "mosaic"; the default keeps every RFD4 id unchanged
    if not re.fullmatch(r"[a-z0-9]{2,12}", str(family)):
        raise ValueError("generator_family must be 2 to 12 lowercase letters or digits")
    record = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": f"{target['id']}-{family}-{identity}",
        "target_id": target["id"], "sequence": sequence, "format": target["format"],
        "generator": audit.get("generator_name", "RFD4-Proteina"), "generator_revision": audit["code_revision"],
        "campaign_kind": "competition_campaign", "seed": audit["seed"],
        "binder_length_requested": audit["binder_length_requested"],
        "hotspots_uniprot": reference["hotspots_uniprot"], "hotspot_source": source,
        "hotspot_review": hotspot_review,
        "construct": {"uniprot": target["uniprot"], "start": target["start"], "end": target["end"],
                      "target_sequence_sha256": target.get("sequence_sha256"),
                      "construct_status": target.get("construct_status")},
        "target_manifest_sha256": audit["target_manifest_sha256"],
        "generator_audit": audit, "generator_structure": str(structure.resolve()),
        "generator_structure_sha256": digest(structure),
        "generator_target_chain": audit["target_chain"], "generator_binder_chain": audit["binder_chain"],
        "backbone_family_id": f"{target['id']}-{family}-seed{audit['seed']}-{identity}",
        "dedup": decision, "source_audit": str(audit_path.resolve()), "source_sha256": digest(audit_path),
        "qc_reasons": reasons, "screen_status": "not_run", "independent_prediction_status": "not_run",
        "novelty_review": "pending", "submission_eligible": False,
    }
    write_json(output, {"candidates": [record], "count": 1,
                        "note": "Conditioned-generation import only; independent prediction, novelty and screen reviews pending"})
    return [record]
