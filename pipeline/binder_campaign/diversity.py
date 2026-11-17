"""Diversity report for any pool of binder complexes (Phase 3, Task 3.4).

Extracted from the team's `scripts/review-il7ra-diversity.py`, which cannot be reused as it stands (it needs exactly sixteen
records, the team's cluster paths, a compiled USalign binary and the 3DI3 file). What is kept is the part that answers "how
alike are these designs": for every pair of designs, within one structure source,

  * fold similarity  TM-score of the two binder backbones (both length normalisations) and the aligned length;
  * site similarity  Jaccard overlap of the target residues each binder touches (heavy atoms within 5 A, via pose_review);
  * sequence identity of the two binders (a cheap extra column);

and two groupings of the pool at several thresholds: greedy complete-link (every member related to every other member of its
group) and connected components (related through any chain of members). The two can differ and both are reported, as the team did.

TM-scores come from `tmtools` (TM-align), already pinned for the near-duplicate gate. On 78 real backbone pairs it reproduced the
real USalign binary's TM-scores, aligned lengths and RMSDs (see tests/fixtures/usalign_reference.json), so no USalign build is
needed and nothing has to be added to a container image: the report is a CPU job that takes seconds.

The groups are descriptive. They are not SCOP/CATH classes, evolutionary families or binding predictions, and the complete-link
grouping depends on the order members are visited (sorted by id here, as in the team's script).
"""
import csv
import itertools
import json
import math
from pathlib import Path

from .dedup import sequence_identity
from .qc import SOURCES, inspect_structure, structures_for
from .runner import write_json

SCHEMA_VERSION = 1
FOLD_THRESHOLDS = (0.5, 0.6, 0.7)
SITE_THRESHOLDS = (0.3, 0.5, 0.7)
COVERAGE_FRACTION = 0.8          # the team's rule: aligned length >= 48 of a 60-residue binder
MAX_POOL = 300


def _check_thresholds(values, name):
    if not values or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 < v <= 1 for v in values):
        raise ValueError(f"{name} must be a non-empty list of numbers in (0, 1]")
    return tuple(float(v) for v in values)


def covered(aligned_length, length_a, length_b, fraction=COVERAGE_FRACTION):
    """True if the alignment spans at least `fraction` of the shorter binder (integer ceiling, so 60 residues -> 48)."""
    need = -(-int(round(fraction * 100)) * min(length_a, length_b) // 100)
    return aligned_length >= need


def tm_pair(coords_a, seq_a, coords_b, seq_b):
    from tmtools import tm_align
    result = tm_align(coords_a, coords_b, seq_a, seq_b)
    return {"tm_left": float(result.tm_norm_chain1), "tm_right": float(result.tm_norm_chain2),
            "aligned_length": sum(1 for c in result.seqM if c in ":."), "rmsd_A": float(result.rmsd)}


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0         # two empty sets count as no overlap, as in the team's script


def complete_link_groups(ids, related):
    """The team's greedy complete-link grouping, visiting ids in sorted order."""
    groups = []
    for identifier in sorted(ids):
        for group in groups:
            if all(related(identifier, other) for other in group):
                group.append(identifier)
                break
        else:
            groups.append([identifier])
    return groups


def connected_components(ids, related):
    """Groups of members linked through any chain of related pairs (the team's closure, with sorted members)."""
    remaining, out = set(ids), []
    while remaining:
        group = {min(remaining)}
        while True:
            expanded = group | {b for b in remaining if any(a != b and related(a, b) for a in group)}
            if expanded == group:
                break
            group = expanded
        out.append(sorted(group))
        remaining -= group
    return out


def group_pool(ids, pairs, field, threshold, lengths=None):
    lookup = {tuple(sorted((p["left"], p["right"]))): p for p in pairs}

    def related(a, b):
        p = lookup[tuple(sorted((a, b)))]
        if field == "tm":
            return (min(p["tm_left"], p["tm_right"]) >= threshold
                    and covered(p["aligned_length"], lengths[p["left"]], lengths[p["right"]]))
        return p["site_jaccard"] >= threshold
    return {"complete_link_greedy": complete_link_groups(ids, related), "connected_components": connected_components(ids, related)}


def analyse_source(members, target_sequence, positions, label, fold_thresholds=FOLD_THRESHOLDS, site_thresholds=SITE_THRESHOLDS,
                   max_pool=MAX_POOL, log=print):
    """Compare every pair of one source's structures. members: [{"id", "binder_sequence", "structure", "sha256"?}]."""
    fold_thresholds = _check_thresholds(fold_thresholds, "fold thresholds")
    site_thresholds = _check_thresholds(site_thresholds, "site thresholds")
    ids = [m["id"] for m in members]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate member ids in the pool")
    if len(members) > max_pool:
        raise ValueError(f"The pool has {len(members)} members; the limit is {max_pool} (the comparison is quadratic). Raise it on purpose.")
    poses, excluded = {}, []
    for member in sorted(members, key=lambda m: m["id"]):
        pose, problems = inspect_structure(member["structure"], label, target_sequence, member["binder_sequence"], positions,
                                           member.get("sha256"))
        if pose is None:
            excluded.append({"id": member["id"], "reasons": problems})
            log(f"[{label}] {member['id']} left out: {'; '.join(problems)}")
        else:
            poses[member["id"]] = (member["binder_sequence"], pose)
    ordered = sorted(poses)
    pairs = []
    for a, b in itertools.combinations(ordered, 2):
        (sa, pa), (sb, pb) = poses[a], poses[b]
        pair = {"left": a, "right": b, **tm_pair(pa["binder_ca"], sa, pb["binder_ca"], sb),
                "site_jaccard": jaccard(pa["target_contacts_lt_5A"], pb["target_contacts_lt_5A"]),
                "shared_contact_residues": len(set(pa["target_contacts_lt_5A"]) & set(pb["target_contacts_lt_5A"])),
                "sequence_identity": sequence_identity(sa, sb)}
        pairs.append(pair)
    lengths = {i: len(poses[i][0]) for i in ordered}
    groups = {"fold": {str(t): group_pool(ordered, pairs, "tm", t, lengths) for t in fold_thresholds},
              "site": {str(t): group_pool(ordered, pairs, "site", t) for t in site_thresholds}}
    summary = {kind: {t: {method: {"groups": len(g), "largest": max((len(x) for x in g), default=0),
                                   "singletons": sum(1 for x in g if len(x) == 1)}
                          for method, g in by.items()} for t, by in table.items()} for kind, table in groups.items()}
    contacts = {i: sorted(int(c) for c in poses[i][1]["target_contacts_lt_5A"]) for i in ordered}
    return {"source": label, "members": ordered, "excluded": excluded, "pairs": pairs, "groups": groups, "summary": summary,
            "target_contacts": contacts}


def pool_from_records(records, evaluation_dir, batch_dir, sources=SOURCES, include_flagged=False):
    """Split records into per-source member lists. Flagged records (non-empty qc_reasons) are left out unless asked for."""
    chosen = [r for r in records if include_flagged or not r.get("qc_reasons")]
    kept = {r["candidate_id"] for r in chosen}
    left_out = [{"id": r["candidate_id"], "reasons": list(r["qc_reasons"])} for r in records if r["candidate_id"] not in kept]
    by_source = {s: [] for s in sources}
    for record in chosen:
        found = structures_for(record, evaluation_dir, batch_dir, sources)
        for source in sources:
            if source in found:
                by_source[source].append({"id": record["candidate_id"], "binder_sequence": record["sequence"],
                                          "structure": found[source]["path"], "sha256": found[source]["sha256"],
                                          "record": record})
    return by_source, left_out


def source_frame(source, record, target):
    """Target sequence and numbering a source's structure uses: the generator holds the reference core, refolds the full target."""
    if source == "generator":
        reference = (record.get("generator_audit") or {}).get("reference_audit") or {}
        return reference.get("reference_sequence"), reference.get("positions")
    return target["sequence"], list(range(target["start"], target["end"] + 1))


def run_diversity(records, target, out_dir, evaluation_dir=None, batch_dir=None, sources=SOURCES, include_flagged=False,
                  fold_thresholds=FOLD_THRESHOLDS, site_thresholds=SITE_THRESHOLDS, max_pool=MAX_POOL, log=print):
    """Diversity report for the records of a pipeline run. Writes diversity.json, pairs.csv and groups.csv into a fresh folder."""
    out_dir = Path(out_dir)
    fold_thresholds = _check_thresholds(fold_thresholds, "fold thresholds")
    site_thresholds = _check_thresholds(site_thresholds, "site thresholds")
    if out_dir.exists():
        raise ValueError("Use a fresh output directory")
    by_source, left_out = pool_from_records(records, evaluation_dir, batch_dir, sources, include_flagged)
    analyses = {}
    for source in sources:
        members = by_source[source]
        if len(members) < 2:
            why = ("no --evaluation folder was given, so the refolded structures cannot be found"
                   if source != "generator" and evaluation_dir is None else f"{len(members)} structure(s) available; at least 2 are needed")
            analyses[source] = {"source": source, "skipped": why}
            log(f"[{source}] skipped: fewer than two structures")
            continue
        frames = {json.dumps(source_frame(source, m["record"], target)) for m in members}
        if len(frames) != 1:
            raise ValueError(f"The {source} structures of this pool do not share one target numbering")
        sequence, positions = source_frame(source, members[0]["record"], target)
        if not sequence or not positions:
            analyses[source] = {"source": source, "skipped": "the records carry no reference audit for this source"}
            continue
        analyses[source] = analyse_source([{k: v for k, v in m.items() if k != "record"} for m in members], sequence, positions,
                                          source, fold_thresholds, site_thresholds, max_pool, log)
    return _write(out_dir, analyses, left_out, fold_thresholds, site_thresholds, include_flagged)


def run_diversity_pool(pool, out_dir, fold_thresholds=FOLD_THRESHOLDS, site_thresholds=SITE_THRESHOLDS, max_pool=MAX_POOL, log=print):
    """Diversity report for any pool: {"target_sequence", "positions", "members": [{"id", "binder_sequence", "structure"}]}."""
    out_dir = Path(out_dir)
    fold_thresholds = _check_thresholds(fold_thresholds, "fold thresholds")
    site_thresholds = _check_thresholds(site_thresholds, "site thresholds")
    if out_dir.exists():
        raise ValueError("Use a fresh output directory")
    for key in ("target_sequence", "positions", "members"):
        if key not in pool:
            raise ValueError(f"The pool file lacks '{key}'")
    analysis = analyse_source(pool["members"], pool["target_sequence"], list(pool["positions"]), pool.get("label", "pool"),
                              fold_thresholds, site_thresholds, max_pool, log)
    return _write(out_dir, {analysis["source"]: analysis}, [], fold_thresholds, site_thresholds, True)


def _write(out_dir, analyses, left_out, fold_thresholds, site_thresholds, include_flagged):
    out_dir.mkdir(parents=True)
    report = {"schema": SCHEMA_VERSION, "sources": analyses, "left_out_as_flagged": left_out, "include_flagged": include_flagged,
              "fold_thresholds": list(fold_thresholds), "site_thresholds": list(site_thresholds),
              "coverage_rule": f"aligned length >= {int(COVERAGE_FRACTION * 100)}% of the shorter binder (48 of 60)",
              "scope": "Descriptive similarity groups. Not SCOP/CATH classes, evolutionary families or binding predictions; "
                       "the complete-link grouping depends on member order (sorted by id)."}
    write_json(out_dir / "diversity.json", report)
    with (out_dir / "pairs.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "left", "right", "tm_left", "tm_right", "tm_min", "aligned_length", "rmsd_A", "site_jaccard",
                         "shared_contact_residues", "sequence_identity"])
        for source, a in analyses.items():
            for p in a.get("pairs", []):
                writer.writerow([source, p["left"], p["right"], f"{p['tm_left']:.5f}", f"{p['tm_right']:.5f}",
                                 f"{min(p['tm_left'], p['tm_right']):.5f}", p["aligned_length"], f"{p['rmsd_A']:.3f}",
                                 f"{p['site_jaccard']:.4f}", p["shared_contact_residues"], f"{p['sequence_identity']:.4f}"])
    with (out_dir / "groups.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "candidate_id", "kind", "threshold", "method", "group"])
        for source, a in analyses.items():
            for kind, table in a.get("groups", {}).items():
                for threshold, methods in table.items():
                    for method, groups in methods.items():
                        for number, group in enumerate(groups, 1):
                            for member in group:
                                writer.writerow([source, member, kind, threshold, method, f"{kind[0].upper()}{number}"])
    return report
