"""Inputs, settings and result packaging for Mosaic on the CURRENT target (Phase 4, Task 4.1). Pure Python: no JAX, no GPU.

Mosaic designs a binder by optimizing a sequence through a differentiable Boltz-2 (JAX), then folds the result once with Boltz-2. It is
a different design method from RFD4 (which builds a backbone first). This module holds everything around the optimizer that can be
checked without a GPU, taken from the team's `mosaic-il7ra.py` (historical target 1-236, 60 aa, fixed hotspots) and generalized:

  * the three pinned source revisions (mosaic, joltz, the Boltz fork) and the optimizer settings are the team's, unchanged;
  * the target is the current construct (UniProt 21-239) with the first 128 rows of its target-only MSA, and the hotspots are
    UniProt positions mapped to target indices relative to the construct start (the team's script assumed a start of 1);
  * a finished design is wrapped in the same audit shape the RFD4 generation uses, so the existing batch engine (budget caps, resume,
    failures kept, near-duplicate gate) and every later stage work on it without change.

Boltz-2 is inside Mosaic's own objective, so Boltz-2 agreement on a Mosaic design is NOT independent evidence; ESMFold2 is the check.
Nothing here marks a design as eligible or reviewed.
"""
import json
from pathlib import Path

from .conditioning import CONDITIONING_ROLE
from .generation_input import check_generated_chains
from .qc import chain_sequences
from .runner import digest

SCHEMA_VERSION = 1
REPOS = {
    "mosaic-runtime": ("https://github.com/escalante-bio/mosaic.git", "5937f37164bcb28bc6f1000f10134145438e5971"),
    "joltz": ("https://github.com/nboyd/joltz.git", "ed0f04257dac85bd4b7bf45521cc280f86d38ded"),
    "mosaic-boltz": ("https://github.com/escalante-bio/boltz.git", "7ea718a1440ecbe8e2418317192b501544147edd"),
}
REVISIONS = {name: rev for name, (_, rev) in REPOS.items()}
MSA_ROWS = 128
# The team's predeclared optimizer settings (mosaic-il7ra.py), unchanged. Only the length, seed and hotspots are parameters.
SETTINGS = {"soft_steps": 75, "soft_stepsize": 0.1, "sharp_steps": 50, "sharp_stepsize": 0.5, "sharp_scale": 1.5, "momentum": 0.0,
            "design_recycles": 1, "design_diffusion_steps": 25, "mpnn_samples": 4, "mpnn_temperature": 0.01,
            "loss_weights": {"binder_target_contact": 2, "within_binder_contact": 1, "inverse_folding_sequence_recovery": 5},
            "initial": "random simplex (gumbel 0.5), no known-positive input", "templates": False}
LENGTH_RANGE = (40, 100)
ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


def read_msa(path, rows=MSA_ROWS):
    """The first `rows` records of an A3M, in order, insertion notation preserved: [(header line, sequence line)]."""
    entries = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            entries.append([line, ""])
        elif entries:
            entries[-1][1] += line.strip()
    return [(h, s) for h, s in entries[:rows]]


def build_inputs(manifest, msa_path, hotspots, length, rows=MSA_ROWS):
    """Validate and assemble the optimizer's inputs for the current target. Raises ValueError with a plain reason."""
    start, end, target = manifest["start"], manifest["end"], manifest["sequence"]
    if len(target) != end - start + 1:
        raise ValueError("The target manifest's sequence does not match its interval")
    if not isinstance(length, int) or isinstance(length, bool) or not LENGTH_RANGE[0] <= length <= LENGTH_RANGE[1]:
        raise ValueError(f"Binder length must be a whole number from {LENGTH_RANGE[0]} to {LENGTH_RANGE[1]}")
    if not hotspots or len(set(hotspots)) != len(hotspots) or any(isinstance(h, bool) or not isinstance(h, int) for h in hotspots):
        raise ValueError("Give at least one hotspot, without repeats, as whole UniProt positions")
    if not all(start <= h <= end for h in hotspots):
        raise ValueError(f"Every hotspot must lie inside the construct {start}-{end}")
    entries = read_msa(msa_path, rows)
    if not entries:
        raise ValueError("The MSA is empty")
    query = "".join(c for c in entries[0][1] if not c.islower() and c != ".")
    if query != target:
        raise ValueError("The MSA's first row is not the current target sequence")
    text = "\n".join(line for entry in entries for line in entry) + "\n"
    epitope = [h - start for h in sorted(hotspots)]                 # index into the construct, 0 = its first residue
    return {"target_sequence": target, "construct": [start, end], "msa_a3m": text, "msa_rows": len(entries), "epitope_idx": epitope,
            "hotspots_uniprot": sorted(hotspots), "binder_length": length, "settings": SETTINGS}


def check_revisions(found):
    """`found` maps source name -> {"head": sha, "dirty": bool}. Raises unless every pinned source is exactly as pinned and unmodified."""
    for name, expected in REVISIONS.items():
        got = found.get(name)
        if got is None or got["head"] != expected or got["dirty"]:
            raise ValueError(f"Unreviewed source revision or modifications: {name}")


def build_audit(*, run_id, seed, hotspots, hotspot_source, binder_length, manifest_path, structure_path, extras):
    """The audit for one finished Mosaic design, in the shape the RFD4 generation writes (see import_current)."""
    manifest = json.loads(Path(manifest_path).read_text())
    chains = chain_sequences(structure_path)
    shaped = [{"id": cid, "sequence": seq} for cid, seq in chains.items()]
    target_chain, binder_chain = check_generated_chains(shaped, manifest["sequence"], binder_length)
    start, end = manifest["start"], manifest["end"]
    return {
        "status": "conditioned_generation; Mosaic optimization through Boltz-2; not a submission-eligible candidate or binding result",
        "generator_name": "Mosaic", "generator_family": "mosaic",
        "reference_audit": {"target_id": manifest["id"], "role": CONDITIONING_ROLE, "reference_sequence": manifest["sequence"],
                            "positions": list(range(start, end + 1)), "alignment_positions": list(range(37, 232)),
                            "hotspots_uniprot": sorted(hotspots), "hotspot_source": hotspot_source,
                            "note": "The Mosaic structure holds the whole construct; pose shifts are measured on the reference core 37-231."},
        "target_manifest_sha256": digest(manifest_path), "generated_structure": str(structure_path),
        "generated_sha256": digest(structure_path), "target_chain": target_chain["id"], "binder_chain": binder_chain["id"],
        "binder_sequence": binder_chain["sequence"], "binder_length_requested": binder_length, "chains": shaped,
        "code_revision": REVISIONS["mosaic-runtime"], "revisions": REVISIONS, "seed": seed, "run_id": run_id, "settings": SETTINGS,
        "evidence_note": "Boltz-2 is inside the design objective: its agreement is related evidence, not independent. ESMFold2 is the check.",
        **extras}
