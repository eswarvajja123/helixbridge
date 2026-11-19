"""Specificity check against decoy targets (calibration for Phase 5). Pure Python.

A design is predicted together with each of a few unrelated proteins (for example GFP and lysozyme) in the same way as with the real target.
A design that AlphaFold-Multimer is just as confident about with an unrelated protein is not specific, so its confidence for the real
target says little. This reads such a run and gives each design its ipTM for every decoy; `gate_reasons` says which decoys it binds
confidently. The limit is the highest ipTM that shuffled-sequence controls reached (0.61 in the team's AlphaFold run), the same floor
the real-target gate uses. Nothing here sets a review field.
"""
import json
from pathlib import Path

from .alphafold_review import load_scores, read_archives

DECOY_MAX_IPTM = 0.61


def read_decoy_iptm(folders, decoys, candidate_prefix="il7ra-rfd4-"):
    """{candidate_id: {decoy: ipTM}} from AlphaFold run folders whose design ids are `<decoy>-<candidate id without the prefix>`."""
    if not decoys or len(set(decoys)) != len(decoys):
        raise ValueError("Give each decoy name once")
    found = {}
    for folder in ([folders] if isinstance(folders, (str, Path)) else folders):
        for design, (_, scores_text) in read_archives(folder).items():
            owner = [d for d in decoys if design.startswith(d + "-")]
            if len(owner) != 1:
                raise ValueError("A prediction does not belong to exactly one decoy: " + design)
            cid = candidate_prefix + design[len(owner[0]) + 1:]
            entry = found.setdefault(cid, {})
            if owner[0] in entry:
                raise ValueError(f"Predicted twice: {design}")
            entry[owner[0]] = round(load_scores(scores_text)["iptm"], 4)
    return found


def summarize(per_candidate):
    """Adds the highest decoy ipTM and which decoy it was."""
    out = {}
    for cid, values in sorted(per_candidate.items()):
        top = max(values, key=lambda d: (values[d], d))
        out[cid] = {"iptm": dict(sorted(values.items())), "max_iptm": values[top], "max_decoy": top}
    return out


def gate_reasons(entry, decoys, max_iptm=DECOY_MAX_IPTM):
    """Why a design is not specific. `entry` is one summarize() entry, or None if no decoy was run for it."""
    if entry is None:
        return ["decoy_missing"]
    reasons = [f"decoy_missing:{d}" for d in decoys if d not in entry["iptm"]]
    reasons += [f"decoy_binding:{d}:{v}>{max_iptm}" for d, v in sorted(entry["iptm"].items()) if v > max_iptm]
    return reasons


def write_summary(path, per_candidate, decoys, source_note=""):
    Path(path).write_text(json.dumps({"decoys": list(decoys), "max_iptm_allowed": DECOY_MAX_IPTM, "designs": summarize(per_candidate),
                                       "note": source_note or "ipTM with unrelated proteins; a high value means the design is not specific"}, indent=1) + "\n")
