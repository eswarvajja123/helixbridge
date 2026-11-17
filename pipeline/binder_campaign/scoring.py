"""Raw interface scores for one predicted complex (Phase 3, Task 3.2). No GPU, no model, no thresholds.

Both predictors (ESMFold2 and Boltz-2) are reduced to the same record: ipTM, pTM, and the ipSAE rows exactly as the
DunbrackLab scorer prints them (pinned commit 6174cf9), with the team's settings (PAE cutoff 10, distance cutoff 10).
Chain A is the target and chain B the binder, as in every request made here.

Nothing is combined, ranked or thresholded. These are raw model outputs: not affinities, not probabilities, and numbers from
different predictors are not on a common scale. The organizer's own aggregation of ipSAE is unconfirmed.
"""
import math
from pathlib import Path
import subprocess
import sys

PAE_CUTOFF = 10
DIST_CUTOFF = 10
IPSAE_COMMIT = "6174cf9e71cb1bd660cc805856a18c4871a6dec3"
PREDICTORS = ("esmfold2", "boltz2")


def parse_ipsae(text):
    """Rows of the scorer's output file: {'chain1','chain2','type' ('asym' or 'max'),'ipsae'} (the team's parsing rule)."""
    rows = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) > 5 and fields[4] in ("asym", "max"):
            value = float(fields[5])
            if not math.isfinite(value):
                raise ValueError("Nonfinite ipSAE")
            rows.append({"chain1": fields[0], "chain2": fields[1], "type": fields[4], "ipsae": value})
    return rows


def check_rows(rows, chain_ids=("A", "B")):
    """Exactly the two directional rows for these two chains, plus the symmetric maximum."""
    first, second = chain_ids
    asym = {(r["chain1"], r["chain2"]) for r in rows if r["type"] == "asym"}
    if asym != {(first, second), (second, first)} or sum(r["type"] == "asym" for r in rows) != 2:
        raise ValueError("Missing or unexpected directional ipSAE pairs")
    if sum(r["type"] == "max" for r in rows) != 1:
        raise ValueError("Expected exactly one symmetric-maximum ipSAE row")
    if not all(isinstance(r["ipsae"], float) and math.isfinite(r["ipsae"]) for r in rows):
        raise ValueError("Nonfinite ipSAE")
    return rows


def summarise_rows(rows, chain_ids=("A", "B")):
    """a_b / b_a are the two directional rows in the order of chain_ids; min_directional is the lower of the two."""
    check_rows(rows, chain_ids)
    first, second = chain_ids
    value = {(r["chain1"], r["chain2"]): r["ipsae"] for r in rows if r["type"] == "asym"}
    return {"a_b": value[(first, second)], "b_a": value[(second, first)],
            "max": next(r["ipsae"] for r in rows if r["type"] == "max"),
            "min_directional": min(value.values())}


def run_ipsae(ipsae_script, pae_json, cif, chain_ids=("A", "B"), *, timeout=120):
    """Run the scorer on one prediction and return its checked rows. It writes next to the structure file."""
    cif = Path(cif)
    log = cif.with_name(cif.stem + ".ipsae.log")
    with log.open("w") as stream:
        subprocess.run([sys.executable, str(ipsae_script), str(pae_json), str(cif), str(PAE_CUTOFF), str(DIST_CUTOFF)],
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    output = cif.with_name(f"{cif.stem}_{PAE_CUTOFF}_{DIST_CUTOFF}.txt")
    if not output.is_file():
        raise ValueError("The ipSAE scorer produced no score file")
    return check_rows(parse_ipsae(output.read_text()), chain_ids)


def build_score(predictor, *, iptm, ptm, rows, seconds, peak_reserved_mib, plddt_raw_mean=None, gpu=None, settings=None):
    """One predictor's raw scores for one candidate, in the shape the evaluation engine stores."""
    if predictor not in PREDICTORS:
        raise ValueError("Unknown predictor: " + repr(predictor))
    numbers = [iptm, ptm, seconds, peak_reserved_mib] + ([] if plddt_raw_mean is None else [plddt_raw_mean])
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in numbers):
        raise ValueError("Nonfinite or non-numeric confidence or resource values")
    return {"predictor": predictor, "iptm": float(iptm), "ptm": float(ptm),
            "plddt_raw_mean": None if plddt_raw_mean is None else float(plddt_raw_mean),
            "ipsae": {"pae_cutoff": PAE_CUTOFF, "distance_cutoff": DIST_CUTOFF, "pairwise": check_rows(list(rows)),
                      **summarise_rows(rows)},
            "seconds": float(seconds), "peak_reserved_mib": float(peak_reserved_mib), "gpu": gpu, "settings": settings,
            "note": "Raw model outputs. No threshold, ranking or combination applied; not affinity; "
                    "scales differ between predictors; organizer aggregation unconfirmed."}


def check_score(score, predictor):
    """Re-validate a score received from a container before it is stored."""
    if not isinstance(score, dict) or score.get("predictor") != predictor:
        raise ValueError("Score is missing or for another predictor")
    return build_score(predictor, iptm=score.get("iptm"), ptm=score.get("ptm"), rows=score["ipsae"]["pairwise"],
                       seconds=score.get("seconds"), peak_reserved_mib=score.get("peak_reserved_mib"),
                       plddt_raw_mean=score.get("plddt_raw_mean"), gpu=score.get("gpu"), settings=score.get("settings"))


def boltz_score(metrics, gpu=None):
    """Convert the metrics.json written by the team's scripts/boltz-predict.py into a score."""
    return build_score("boltz2", iptm=metrics["iptm"], ptm=metrics["ptm"], rows=metrics["ipsae"]["pairwise"],
                       seconds=metrics["prediction_seconds"], peak_reserved_mib=metrics["peak_reserved_mib"], gpu=gpu,
                       settings={"seed": metrics["seed"], "samples": metrics["samples"],
                                 "target_only_precomputed_msa": metrics["target_only_precomputed_msa"]})
