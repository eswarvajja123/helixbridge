"""ESMFold2 helpers that need no GPU, no torch and no model (Phase 3, Task 3.1).

Pure code used by modal_esmfold2.py: the checkpoint integrity check, the fold request, and the checks on a prediction.
They mirror what the team's own pipeline/scripts/esmfold2-predict.py does, so a result from Modal is judged the same way.

SCREEN_SETTINGS are the team's local APPROXIMATION of the organizer's screen (ESMFold2 + ipSAE, one seed, one sample, target
MSA only, no templates). The organizer's exact implementation and parameters are unconfirmed, so nothing here is an
official admission score.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np

REPO = "biohub/ESMFold2"
REVISION = "69869f737beffec5294845ede23db5fc0b4f509e"
# Copied from pipeline/scripts/prepare-screen-control.py (the settings the team ran on its cluster).
SCREEN_SETTINGS = {"num_loops": 20, "num_sampling_steps": 200, "num_diffusion_samples": 1, "seed": 0, "lm_dropout": 0.3,
                   "msa_max_depth": 1024, "msa_column_mask_rate": 0.1, "kernel_backend": "fused"}
KERNEL_BACKENDS = ("fused", "cuequivariance", None)          # what esm 3.4.1.post1 accepts in set_kernel_backend
PROTOCOL_STATUS = "local approximation of the organizer screen; exact implementation and parameters unconfirmed"


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("repo") != REPO or manifest.get("revision") != REVISION:
        raise ValueError("The checkpoint manifest is not for the pinned ESMFold2 revision")
    if not manifest.get("files") or manifest.get("esmc_backbone_bundled") is not True:
        raise ValueError("The manifest must list the files and a bundled ESMC backbone")
    for name, info in manifest["files"].items():
        if not (isinstance(info.get("bytes"), int) and len(str(info.get("sha256", ""))) == 64):
            raise ValueError(f"Bad manifest entry for {name}")
    return manifest


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(directory, manifest, *, hash_files=True):
    """Check a downloaded checkpoint folder against the manifest. Sizes always; SHA-256 too when hash_files is true.

    Extra files are ignored (the download may carry others); a missing, resized or altered listed file is an error.
    """
    directory = Path(directory)
    problems = []
    for name, expected in manifest["files"].items():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            problems.append(f"missing or not a regular file: {name}")
        elif path.stat().st_size != expected["bytes"]:
            problems.append(f"wrong size: {name} ({path.stat().st_size} instead of {expected['bytes']})")
        elif hash_files and _sha256(path) != expected["sha256"]:
            problems.append(f"hash mismatch: {name}")
    if problems:
        raise ValueError("Checkpoint does not match the manifest: " + "; ".join(problems))
    return {"files": len(manifest["files"]), "bytes": sum(f["bytes"] for f in manifest["files"].values()), "hashed": hash_files}


def checkpoint_record(directory, manifest):
    """The env/esmfold2-checkpoints.json record the team's scripts read (a list with one entry)."""
    return [{"repo": manifest["repo"], "revision": manifest["revision"], "path": str(directory),
             "files": {n: {"sha256": f["sha256"], "bytes": f["bytes"]} for n, f in manifest["files"].items()}}]


def check_settings(settings):
    settings = dict(settings)
    if settings.get("kernel_backend") not in KERNEL_BACKENDS:
        raise ValueError("kernel_backend must be one of " + ", ".join(map(repr, KERNEL_BACKENDS)))
    if settings.get("num_diffusion_samples") != 1:
        raise ValueError("The screen approximation uses exactly one diffusion sample")
    seed = settings.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return settings


def build_request(target_sequence, binder_sequence, msa_path, candidate_id, *, role, seed=0, kernel_backend="fused"):
    """The fold request: target chain A with the target-only MSA, binder chain B with no MSA, no templates."""
    settings = check_settings({**SCREEN_SETTINGS, "seed": seed, "kernel_backend": kernel_backend})
    for label, sequence in (("target", target_sequence), ("binder", binder_sequence)):
        if not isinstance(sequence, str) or not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError(f"The {label} sequence must use the 20 standard amino acids")
    return {"role": role, "candidate_id": candidate_id, "target": {"sequence": target_sequence, "msa": str(msa_path)},
            "partner": {"id": "B", "sequence": binder_sequence, "msa": None}, "settings": settings,
            "protocol_status": PROTOCOL_STATUS}


def validate_prediction(pae, cif_text, target_sequence, binder_sequence):
    """The team's checks on one prediction: PAE shape and finiteness, residue count, chain order, coordinates."""
    from biotite.structure.io.pdbx import CIFFile, get_structure
    from .structure import AA3
    pae = np.asarray(pae)
    expected = len(target_sequence) + len(binder_sequence)
    if pae.shape != (expected, expected) or not np.isfinite(pae).all():
        raise ValueError("PAE does not have one finite token per input residue")
    atoms = get_structure(CIFFile.read(io.StringIO(cif_text)), model=1, use_author_fields=False)
    ca = atoms[atoms.atom_name == "CA"]
    if len(ca) != expected:
        raise ValueError("CIF and PAE residue counts differ")
    chains = list(dict.fromkeys(ca.chain_id.tolist()))
    observed = ["".join(AA3[x] for x in ca[ca.chain_id == c].res_name) for c in chains]
    if observed != [target_sequence, binder_sequence]:
        raise ValueError("CIF chain order or sequence differs from the PAE input convention")
    if not np.isfinite(ca.coord).all():
        raise ValueError("Invalid coordinates")
    return {"chains": [{"id": str(c), "length": int(sum(ca.chain_id == c))} for c in chains], "residues": int(expected)}


def confidence_summary(plddt, ptm, iptm):
    """Finite confidence numbers; raw model outputs, not calibrated probabilities or affinities."""
    mean = float(np.asarray(plddt).mean())
    values = [mean, float(ptm), float(iptm)]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite confidence metrics")
    return {"plddt_raw_mean": values[0], "ptm": values[1], "iptm": values[2]}
