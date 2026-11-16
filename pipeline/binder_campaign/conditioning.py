"""Build RFD4 conditioning inputs (contig + hotspots) from a reviewed reference.

Hotspots use UniProt numbering (the reference's label_seq_id equals the UniProt
position) and must lie on residues the reference structure actually resolves,
not merely inside the official interval. Nothing here predicts or implies binding.
"""
from pathlib import Path
import json
import shutil

from .runner import digest, write_json

MIN_BINDER = 40
MAX_BINDER = 100


def _bounds(binder_length):
    """Validate a fixed length or a (min, max) range; return (low, high)."""
    if isinstance(binder_length, int) and not isinstance(binder_length, bool):
        low = high = binder_length
    else:
        try:
            low, high = binder_length
        except (TypeError, ValueError):
            raise ValueError("binder_length must be an int or a (min, max) pair")
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (low, high)):
            raise ValueError("binder_length bounds must be integers")
    if not MIN_BINDER <= low <= high <= MAX_BINDER:
        raise ValueError(f"Binder length must be within {MIN_BINDER}-{MAX_BINDER} with min <= max")
    return low, high


def _length_token(binder_length):
    low, high = _bounds(binder_length)
    return str(low) if low == high else f"{low}-{high}"


def _audit_length(binder_length):
    """Fixed length stays an int (existing scripts compare it); a range is [min, max]."""
    low, high = _bounds(binder_length)
    return low if low == high else [low, high]


def _check_hotspots(hotspots, resolved):
    hotspots = list(hotspots)
    if not hotspots:
        raise ValueError("At least one hotspot is required")
    if any(isinstance(h, bool) or not isinstance(h, int) for h in hotspots):
        raise ValueError("Hotspots must be integer UniProt positions")
    if len(set(hotspots)) != len(hotspots):
        raise ValueError("Duplicate hotspot positions")
    missing = sorted(set(hotspots) - set(resolved))
    if missing:
        raise ValueError("Hotspots not resolved in the reference structure: " + ", ".join(map(str, missing)))
    return sorted(hotspots)


def build_conditions(resolved_positions, hotspots, binder_length, *, name="il7ra",
                     input_name="target.cif", chain="A"):
    """Return the RFD4 inputs mapping for one target, validated against resolved positions."""
    resolved = sorted(resolved_positions)
    if not resolved or resolved != list(range(resolved[0], resolved[-1] + 1)):
        raise ValueError("Resolved reference positions must be one contiguous run")
    hotspots = _check_hotspots(hotspots, resolved)
    token = _length_token(binder_length)
    selection = ",".join(map(str, hotspots))
    return {name: {
        "input": input_name,
        "contig": f"{token}, /0, {chain}{resolved[0]}-{resolved[-1]}",
        "conditions": {
            "C_CRDGRP": {"1": [{"select": f"sel('{chain}/*/*/[N,CA,C,O]')"}]},
            "C_SEQ": {"true": [{"select": f"sel('{chain}')"}]},
            "C_HOT": {"true": [{"select": f"sel('{chain}/*/[{selection}]')"}]},
        },
    }}


def write_conditioning(reference_dir, output_dir, hotspots, binder_length, *, hotspot_source):
    """Write target.cif, conditions.json and audit.json for one hashed, checked input set.

    `hotspot_source` is recorded so provisional hotspots can never be mistaken for a
    reviewed epitope choice (for example "provisional_historical" until it is reviewed).
    """
    if not hotspot_source or not str(hotspot_source).strip():
        raise ValueError("State where the hotspots came from (hotspot_source)")
    reference_dir, output_dir = Path(reference_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use a fresh output directory")
    reference = json.loads((reference_dir / "audit.json").read_text())
    for filename, expected in reference["files"].items():
        if digest(reference_dir / filename) != expected:
            raise ValueError("Reference input changed after preparation: " + filename)
    old = json.loads((reference_dir / "conditions.json").read_text())
    name = next(iter(old))
    conditions = build_conditions(reference["positions"], hotspots, binder_length, name=name,
                                  input_name=old[name]["input"], chain=reference["reference_chain"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(reference_dir / "target.cif", output_dir / "target.cif")
    write_json(output_dir / "conditions.json", conditions)
    sorted_hotspots = _check_hotspots(hotspots, reference["positions"])
    audit = {**reference,
             "role": "conditioning_input; not a submission-eligible candidate or binding result",
             "binder_length": _audit_length(binder_length),
             "hotspots_uniprot": sorted_hotspots,
             "hotspot_source": str(hotspot_source),
             "files": {"target.cif": digest(output_dir / "target.cif"),
                       "conditions.json": digest(output_dir / "conditions.json")}}
    write_json(output_dir / "audit.json", audit)
    return audit
