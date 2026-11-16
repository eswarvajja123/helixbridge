"""Pure checks for one RFD4 generation: reference identity, seed and generated chains.

No GPU or model code is imported here, so every check can be tested on any machine.
The generation script calls these before and after the model runs.
"""
from pathlib import Path
import json

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from .conditioning import CONDITIONING_ROLE
from .runner import digest
from .structure import AA3

ALLOWED_ROLES = {"reference_rehearsal_only", CONDITIONING_ROLE}
ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


def check_seed(seed):
    """Any non-negative integer; the historical seed whitelist is intentionally gone."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a non-negative integer")
    return seed


def build_overrides(spec, out, seed, model_record):
    """The Hydra overrides for one single-design RFD4 run (same recipe as the qualified seed script)."""
    return [
        "inference_experiment=quickstart", f"inputs={spec}", f"out_dir={out}",
        f"hf_ckpt_path={model_record['path']}", f"hf_ckpt_revision={model_record['revision']}",
        "num_replicates=1", "batch_size=1", "num_workers=0", f"seed={check_seed(seed)}",
        "compile_lengths=null", "recycle_strategy=thinking_high", "compress_outputs=false",
        "skip_existing=false", "prevalidate_inputs=true",
    ]


def length_allowed(length, spec):
    """`spec` is the audit's binder_length: an int (exact) or a [min, max] pair (inclusive)."""
    if isinstance(spec, int) and not isinstance(spec, bool):
        return length == spec
    if isinstance(spec, (list, tuple)) and len(spec) == 2 and all(isinstance(v, int) for v in spec):
        return spec[0] <= length <= spec[1]
    raise ValueError("binder_length in the audit must be an int or a [min, max] pair")


def read_reference_ca(cif_path):
    """Return (positions, sequence) of model-1 CA atoms; label_seq_id is the UniProt position."""
    cif = MMCIF2Dict(str(cif_path))
    models = cif.get("_atom_site.pdbx_PDB_model_num")
    first = models[0] if models else None
    positions, letters = [], []
    for i, atom in enumerate(cif["_atom_site.label_atom_id"]):
        if atom != "CA" or (models and models[i] != first):
            continue
        residue = cif["_atom_site.label_comp_id"][i]
        if residue not in AA3:
            raise ValueError("Nonstandard residue in reference: " + residue)
        positions.append(int(cif["_atom_site.label_seq_id"][i]))
        letters.append(AA3[residue])
    return positions, "".join(letters)


def load_reference(reference_dir, manifest_path):
    """Validate a conditioning/reference directory against the target manifest; return its audit."""
    reference_dir = Path(reference_dir)
    audit = json.loads((reference_dir / "audit.json").read_text())
    manifest = json.loads(Path(manifest_path).read_text())
    if audit.get("role") not in ALLOWED_ROLES:
        raise ValueError("Unrecognized reference role: " + repr(audit.get("role")))
    if audit["target_id"] != manifest["id"]:
        raise ValueError("Reference target does not match the manifest")
    for name, expected in audit["files"].items():
        if digest(reference_dir / name) != expected:
            raise ValueError("Reference input changed after preparation: " + name)
    positions, sequence = read_reference_ca(reference_dir / "target.cif")
    if positions != audit["positions"] or sequence != audit["reference_sequence"]:
        raise ValueError("Reference coordinates differ from the audited positions or sequence")
    start, end = manifest["start"], manifest["end"]
    if not all(start <= p <= end for p in positions):
        raise ValueError("Reference positions fall outside the manifest interval")
    if sequence != "".join(manifest["sequence"][p - start] for p in positions):
        raise ValueError("Reference mapping/identity differs from the manifest sequence")
    if not set(audit["hotspots_uniprot"]) <= set(positions):
        raise ValueError("Audited hotspots are not resolved in the reference")
    length_allowed(0, audit["binder_length"])  # validates the spec's shape
    return {**audit, "target_sequence": sequence}


def check_generated_chains(chains, target_sequence, binder_length):
    """Exactly one target chain and one valid binder chain; returns (target_chain, binder_chain)."""
    if len(chains) != 2:
        raise ValueError(f"Expected two chains, found {len(chains)}")
    targets = [c for c in chains if c["sequence"] == target_sequence]
    binders = [c for c in chains if c["sequence"] != target_sequence]
    if len(targets) != 1 or len(binders) != 1:
        raise ValueError("Generated target identity or one-chain binder validation failed")
    sequence = binders[0]["sequence"]
    if not sequence or set(sequence) - ALPHABET:
        raise ValueError("Binder has an invalid or empty sequence")
    if not length_allowed(len(sequence), binder_length):
        raise ValueError(f"Binder length {len(sequence)} is outside the requested {binder_length}")
    return targets[0], binders[0]
