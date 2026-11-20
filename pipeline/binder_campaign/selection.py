"""Propose a shortlist of at most 20 designs that covers as many distinct folds and sites as possible (Phase 5, Task 5.3).

This PROPOSES; it never decides. It reads a pool that has been through scoring, structure QC, the novelty pre-screen and the liability
review (one records file holding all four), plus a diversity report, and picks designs by coverage, not by score:

  1. ELIGIBLE: no qc_reasons; structure QC status `no_flags` (not `incomplete`); novelty pre-screen not flagged; both predictors
     scored; liabilities reviewed. Everything else is listed with its reasons.
  2. GROUPS: the designs' fold groups and site groups are read from the diversity report (one structure source, one threshold each).
  3. PICK: round-robin over fold groups, one design per group per round, until the limit or nothing is left. Inside a group the next
     design is the one whose SITE group is not yet represented, then the one with fewest liability flags, then the one on which the
     two predictors agree best (smallest gap in ipTM and ipSAE: agreement between independent models, never raw strength), then id.

`pick_order` is the order of picking, NOT a quality ranking: nothing here says one selected design is better than another, and a
design not selected is not worse. The team's pilot found negative controls scoring as high as binders, so scores are not used to rank.

Export: the team's `candidates.export_draft` refuses a design unless a person has set novelty_review to `passed`,
independent_prediction_status to `reviewed` and screen_status to a reviewed value. Those are human decisions; this module never sets
them by itself. `apply_reviews` records them from a reviews file with the reviewer's name and date, and `export_check` then runs the
team's own guard and reports whether it passes or exactly why it refuses.
"""
import csv
import json
from pathlib import Path
import tempfile

from .candidates import export_draft
from .decoys import DECOY_MAX_IPTM, gate_reasons as decoy_reasons
from .runner import write_json

SCHEMA_VERSION = 1
PREDICTORS = ("esmfold2", "boltz2")
REVIEW_FIELDS = {"novelty_review": {"passed"}, "independent_prediction_status": {"reviewed"},
                 "screen_status": {"approximation_reviewed", "official_protocol_reviewed"}}
MAX_FINALISTS = 20
# AlphaFold gate (optional). The ipTM floor is set by the shuffled negative controls run through the same AlphaFold settings: a design must
# score ABOVE every control (the controls' maximum was 0.61). The pose limits are the team's own structure-QC limits.
ALPHAFOLD_MIN_IPTM = 0.61
ALPHAFOLD_MAX_SHIFT = {"generator": 2.5, "esmfold2": 3.0, "boltz2": 3.0}


def _score_values(record):
    out = {}
    for predictor in PREDICTORS:
        entry = (record.get("evaluations") or {}).get(predictor) or {}
        score = entry.get("score") if entry.get("status") == "done" else None
        out[predictor] = score and {"iptm": score["iptm"], "ipsae_max": score["ipsae"]["max"]}
    return out


def disagreement(record):
    """Largest gap between the two predictors, over ipTM and max ipSAE. None if either is missing."""
    values = _score_values(record)
    if not all(values.values()):
        return None
    a, b = values["esmfold2"], values["boltz2"]
    return round(max(abs(a["iptm"] - b["iptm"]), abs(a["ipsae_max"] - b["ipsae_max"])), 6)


def alphafold_reasons(result, min_iptm=ALPHAFOLD_MIN_IPTM, max_shift=ALPHAFOLD_MAX_SHIFT):
    """Why AlphaFold-Multimer does not support a design. `result` is one entry of alphafold-results.json, or None if it was not run."""
    if result is None:
        return ["alphafold_missing"]
    reasons = []
    iptm = result.get("iptm")
    if iptm is None or iptm <= min_iptm:
        reasons.append(f"alphafold_iptm:{iptm}<={min_iptm}")
    shifts = result.get("pose_shift_A") or {}
    for source, limit in max_shift.items():
        value = shifts.get(source)
        if value is None:
            reasons.append(f"alphafold_pose_unchecked:{source}")
        elif value > limit:
            reasons.append(f"alphafold_pose_shift:{source}:{value}A>{limit}A")
    return reasons


def eligibility(record, alphafold=None, min_iptm=ALPHAFOLD_MIN_IPTM, decoys=None, decoy_names=(), excluded=None):
    """The reasons a design cannot be selected; an empty list means eligible. With `alphafold` ({candidate_id: result}) its gate applies too."""
    reasons = [f"qc_reason:{r}" for r in record.get("qc_reasons") or []]
    qc = record.get("qc")
    if not qc:
        reasons.append("structure_qc_missing")
    elif qc["status"] != "no_flags":
        reasons.append(f"structure_qc_{qc['status']}")
    novelty = (record.get("novelty_prescreen") or {}).get("status")
    if novelty is None:
        reasons.append("novelty_prescreen_missing")
    elif novelty not in ("no_public_hit", "no_tripping_hit"):
        reasons.append(f"novelty_prescreen_{novelty}")
    if disagreement(record) is None:
        reasons.append("not_scored_by_both_predictors")
    if not record.get("liabilities") or record["liabilities"].get("diagnostic") is None:
        reasons.append("liabilities_not_reviewed")
    if excluded and record["candidate_id"] in excluded:           # a design the owner has asked to leave out, with the stated reason
        reasons.append("excluded_by_request:" + excluded[record["candidate_id"]])
    if alphafold is not None:
        reasons += alphafold_reasons(alphafold.get(record["candidate_id"]), min_iptm)
    if decoys is not None:                                   # specificity: confident binding of an unrelated protein excludes a design
        reasons += decoy_reasons(decoys.get(record["candidate_id"]), decoy_names)
    return list(dict.fromkeys(reasons))


def _labels(groups):
    """{member: 'F1'...} from a list of groups, numbered in the order the diversity report wrote them."""
    return {member: f"{number}" for number, group in enumerate(groups, 1) for member in group}


def read_groups(diversity, source, fold_threshold, site_threshold, method="complete_link_greedy"):
    if source not in diversity["sources"] or "groups" not in diversity["sources"][source]:
        raise ValueError(f"The diversity report has no groups for source '{source}'")
    groups = diversity["sources"][source]["groups"]
    for kind, threshold in (("fold", fold_threshold), ("site", site_threshold)):
        if str(threshold) not in groups[kind]:
            raise ValueError(f"The diversity report has no {kind} groups at threshold {threshold}")
    return (_labels(groups["fold"][str(fold_threshold)][method]), _labels(groups["site"][str(site_threshold)][method]))


def select(records, diversity, limit=MAX_FINALISTS, source="generator", fold_threshold=0.5, site_threshold=0.5, alphafold=None,
           min_iptm=ALPHAFOLD_MIN_IPTM, decoys=None, decoy_names=(), leave_out=None):
    """Return (selected, report). selected: [{candidate_id, pick_order, fold_group, site_group, ...}]."""
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_FINALISTS:
        raise ValueError(f"limit must be a whole number from 1 to {MAX_FINALISTS}")
    fold_of, site_of = read_groups(diversity, source, fold_threshold, site_threshold)
    by_id, excluded, candidates = {}, [], []
    for record in records:
        if record["candidate_id"] in by_id:
            raise ValueError("Duplicate candidate_id: " + record["candidate_id"])
        by_id[record["candidate_id"]] = record
        reasons = eligibility(record, alphafold, min_iptm, decoys, decoy_names, leave_out)
        if not reasons and record["candidate_id"] not in fold_of:
            reasons.append("not_in_the_diversity_report")
        if reasons:
            excluded.append({"candidate_id": record["candidate_id"], "reasons": reasons})
        else:
            candidates.append(record["candidate_id"])
    sequences = {}
    for cid in candidates:                                                   # an exact duplicate sequence can never be exported twice
        sequences.setdefault(by_id[cid]["sequence"], []).append(cid)
    dropped = {c for ids in sequences.values() for c in ids[1:]}
    for cid in sorted(dropped):
        excluded.append({"candidate_id": cid, "reasons": ["exact_duplicate_sequence_of_an_earlier_design"]})
    candidates = [c for c in candidates if c not in dropped]

    groups = {}
    for cid in sorted(candidates):
        groups.setdefault(fold_of[cid], []).append(cid)
    order = sorted(groups, key=lambda g: (-len(groups[g]), min(groups[g])))   # larger fold groups first; ties by id
    used_sites, selected = set(), []

    def key(cid):
        r = by_id[cid]
        return (site_of[cid] in used_sites, len(r["liabilities"]["flags"]), disagreement(r), cid)

    while len(selected) < limit and any(groups[g] for g in order):
        for g in order:
            if not groups[g] or len(selected) >= limit:
                continue
            pick = min(groups[g], key=key)
            groups[g].remove(pick)
            used_sites.add(site_of[pick])
            r = by_id[pick]
            selected.append({"pick_order": len(selected) + 1, "candidate_id": pick, "fold_group": fold_of[pick], "site_group": site_of[pick],
                             "liability_flags": r["liabilities"]["flags"], "predictor_disagreement": disagreement(r),
                             "hotspot_source": r.get("hotspot_source"), "seed": r.get("seed"), "length": len(r["sequence"]), "sequence": r["sequence"],
                             "scores": _score_values(r)})
    eligible_ids = [c for c in by_id if c not in {e["candidate_id"] for e in excluded}]
    report = {"schema": SCHEMA_VERSION, "pool": len(records), "eligible": len(eligible_ids), "excluded": excluded,
              "selected": len(selected), "limit": limit, "source": source, "fold_threshold": fold_threshold,
              "site_threshold": site_threshold,
              "excluded_by_request": sorted(leave_out) if leave_out else [],
              "decoy_gate": None if decoys is None else {"decoys": list(decoy_names), "max_iptm_allowed": DECOY_MAX_IPTM},
              "alphafold_gate": None if alphafold is None else {"min_iptm_exclusive": min_iptm, "max_pose_shift_A": ALPHAFOLD_MAX_SHIFT,
                                                                "basis": "ipTM above every shuffled control; the team's pose limits"},
              "fold_groups_available": len({fold_of[c] for c in eligible_ids}), "fold_groups_selected": len({s["fold_group"] for s in selected}),
              "site_groups_available": len({site_of[c] for c in eligible_ids}), "site_groups_selected": len({s["site_group"] for s in selected}),
              "scope": "A proposal by coverage, not a ranking: pick_order is the order of picking, no design is better than another, and "
                       "scores were not used to rank. Review fields are untouched, so the export guard still needs a person's reviews."}
    return selected, report


def apply_reviews(records, selected_ids, reviews):
    """Copies of the selected records with a person's review decisions recorded. Anything unlisted stays as it was."""
    if not isinstance(reviews, dict) or not reviews.get("reviewer") or not reviews.get("date") or not isinstance(reviews.get("candidates"), dict):
        raise ValueError("The reviews file needs 'reviewer', 'date' and 'candidates'")
    unknown = set(reviews["candidates"]) - set(selected_ids)
    if unknown:
        raise ValueError("Reviews for designs that are not in the selection: " + ", ".join(sorted(unknown)))
    out = []
    for record in records:
        if record["candidate_id"] not in selected_ids:
            continue
        changes = reviews["candidates"].get(record["candidate_id"], {})
        for field, value in changes.items():
            if field not in REVIEW_FIELDS or value not in REVIEW_FIELDS[field]:
                raise ValueError(f"Not an allowed review value: {field}={value!r}")
        new = {**record, **changes}
        if changes:
            new["review_provenance"] = {"reviewer": reviews["reviewer"], "date": reviews["date"], "fields": sorted(changes)}
        out.append(new)
    return out


def export_check(records, selected_ids, out_csv):
    """Run the team's own export guard on the selection. Returns {"passes": bool, "message": ...}; writes the draft only if it passes."""
    chosen = [r for r in records if r["candidate_id"] in selected_ids]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "records.json"
        path.write_text(json.dumps({"candidates": chosen}))
        try:
            export_draft(path, out_csv, list(selected_ids))
        except ValueError as error:
            return {"passes": False, "message": str(error)}
    return {"passes": True, "message": "The team's export guard accepted the selection (internal draft only; the official schema is unconfirmed)."}


def review_gaps(records, selected_ids):
    """What a person still has to do, per selected design, before the export guard can pass."""
    gaps = {}
    for r in records:
        if r["candidate_id"] not in selected_ids:
            continue
        need = []
        if r.get("novelty_review") != "passed":
            need.append("novelty_review (pre-screen is public-database only; needs the organizers' check or your own judgement)")
        if r.get("independent_prediction_status") != "reviewed":
            need.append("independent_prediction_status (the AlphaFold cross-check, Task 5.4)")
        if r.get("screen_status") not in ("approximation_reviewed", "official_protocol_reviewed"):
            need.append("screen_status (a person reviews the ESMFold2 screen)")
        gaps[r["candidate_id"]] = need
    return gaps


def write_outputs(out_dir, selected, report, records, exported):
    out_dir = Path(out_dir)
    write_json(out_dir / "selection.json", {"selected": selected, "report": report, "export_check": exported["check"],
                                            "review_gaps": exported["gaps"]})
    with (out_dir / "selection.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pick_order", "candidate_id", "sequence", "length", "seed", "hotspot_source", "fold_group", "site_group", "liability_flags",
                         "predictor_disagreement", "esmfold2_iptm", "esmfold2_ipsae_max", "boltz2_iptm", "boltz2_ipsae_max"])
        for s in selected:
            sc = s["scores"]
            writer.writerow([s["pick_order"], s["candidate_id"], s["sequence"], s["length"], s["seed"], s["hotspot_source"], s["fold_group"], s["site_group"],
                             ";".join(s["liability_flags"]), s["predictor_disagreement"], sc["esmfold2"]["iptm"], sc["esmfold2"]["ipsae_max"],
                             sc["boltz2"]["iptm"], sc["boltz2"]["ipsae_max"]])
