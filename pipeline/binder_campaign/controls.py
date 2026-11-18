"""Shuffled-sequence negative controls, and how well the pipeline's numbers separate designs from them (Phase 3 calibration).

Every earlier result says "high scores do not show binding" because the team's own pilot found negative controls scoring as high as
binders. This tests that on OUR pipeline: each shortlisted design gets a control with the same length and the same amino-acid
composition but a random order, so it has the same charge and size and no designed fold. Controls run through exactly the same
scoring and structure QC. If the controls score and pass QC about as often as the designs, the numbers cannot tell designs from junk.

Reported honestly: for each measure, the distribution for designs and for controls, and the probability that a random design beats a
random control (0.5 = no separation, 1.0 = perfect). Shuffled sequences are an easy negative (they have no designed fold), so a design
beating them says little about binding: this is a floor, not a validation.
"""
import json
import random
import statistics

SCHEMA_VERSION = 1


def shuffle_sequence(sequence, seed):
    """Same length and composition, different order; never returns the original unless the sequence cannot be reordered."""
    letters = list(sequence)
    if len(set(letters)) == 1:
        raise ValueError("A sequence of one repeated residue cannot be shuffled")
    rng = random.Random(seed)
    for _ in range(100):
        rng.shuffle(letters)
        out = "".join(letters)
        if out != sequence:
            return out
    raise ValueError("Could not reorder the sequence")


def make_controls(records, seed=1):
    """One shuffled control per record. Controls carry the parent's reference audit (so QC can align on the core) but no structure."""
    out, seen = [], set()
    for i, record in enumerate(records):
        shuffled = shuffle_sequence(record["sequence"], seed * 100003 + i)
        cid = "il7ra-shuf-" + record["candidate_id"].rsplit("-", 1)[-1]
        if cid in seen:
            raise ValueError("Two parents would give the same control id: " + cid)
        seen.add(cid)
        out.append({"candidate_id": cid, "target_id": record.get("target_id", "il7ra"), "sequence": shuffled, "qc_reasons": [],
                    "seed": record.get("seed"), "hotspots_uniprot": record.get("hotspots_uniprot"), "hotspot_source": record.get("hotspot_source"),
                    "generator_audit": {"reference_audit": (record.get("generator_audit") or {}).get("reference_audit")},
                    "control": {"kind": "shuffled_negative", "parent_candidate_id": record["candidate_id"], "shuffle_seed": seed * 100003 + i,
                                "same_length_and_composition_as_parent": True},
                    "screen_status": "not_run", "independent_prediction_status": "not_run", "novelty_review": "pending",
                    "submission_eligible": False})
    return out


def separation(design_values, control_values):
    """P(a random design scores above a random control), ties counting half. 0.5 means no separation."""
    if not design_values or not control_values:
        return None
    wins = sum((d > c) + 0.5 * (d == c) for d in design_values for c in control_values)
    return round(wins / (len(design_values) * len(control_values)), 3)


def _spread(values):
    return {"n": len(values), "min": round(min(values), 3), "median": round(statistics.median(values), 3), "max": round(max(values), 3)} if values else None


MEASURES = (("esmfold2_iptm", lambda r: r["evaluations"]["esmfold2"]["score"]["iptm"]),
            ("esmfold2_ipsae_max", lambda r: r["evaluations"]["esmfold2"]["score"]["ipsae"]["max"]),
            ("boltz2_iptm", lambda r: r["evaluations"]["boltz2"]["score"]["iptm"]),
            ("boltz2_ipsae_max", lambda r: r["evaluations"]["boltz2"]["score"]["ipsae"]["max"]))


def _values(records, getter):
    out = []
    for r in records:
        try:
            if all((r.get("evaluations") or {}).get(p, {}).get("status") == "done" for p in ("esmfold2", "boltz2")):
                out.append(getter(r))
        except (KeyError, TypeError):
            pass
    return out


def report(designs, controls, thresholds=(0.8,)):
    """`designs` and `controls` are QC records (with `evaluations` and `qc`). Everything is descriptive."""
    result = {"schema": SCHEMA_VERSION, "designs": len(designs), "controls": len(controls), "measures": {}}
    for name, getter in MEASURES:
        d, c = _values(designs, getter), _values(controls, getter)
        entry = {"designs": _spread(d), "controls": _spread(c), "p_design_above_control": separation(d, c)}
        if "iptm" in name:
            entry["share_at_or_above"] = {str(t): {"designs": round(sum(v >= t for v in d) / len(d), 3) if d else None,
                                                  "controls": round(sum(v >= t for v in c) / len(c), 3) if c else None} for t in thresholds}
        result["measures"][name] = entry
    clean = lambda rs: sum(1 for r in rs if r["qc"]["status"] == "no_flags")
    result["qc_clean"] = {"designs": clean(designs), "controls": clean(controls),
                          "design_share": round(clean(designs) / len(designs), 3) if designs else None,
                          "control_share": round(clean(controls) / len(controls), 3) if controls else None}
    result["scope"] = ("Shuffled-sequence negatives (same length and composition). An easy negative: it shows what the numbers cannot do, not that a "
                       "design binds. Probability 0.5 means no separation; the QC compares the two predictors' poses and contacts only (no generator).")
    return result
