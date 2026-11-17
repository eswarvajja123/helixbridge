"""Realistic generation results for the batch tests, built from the real verified design (no GPU, no Modal).

A result has exactly the shape modal_current.generate_current returns: an audit like rfd4-generate.py writes, and a
gzip archive whose members are <run_id>/generation/model.cif and <run_id>/generation/audit.json. The structure files
hold only CA atoms (that is all the dedup gate and importer read). The coordinates are real: the verified RFD4 binder
and stretches of its real target chain, so "same backbone" and "different backbone" are real, not invented.
"""
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile

from binder_campaign.conditioning import write_conditioning
from binder_campaign.dedup import read_chain_ca
from binder_campaign.generation_input import load_reference
from binder_campaign.structure import AA3

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/il7ra-rehearsal"
REAL = ROOT / "examples/verified-modal-run"
MANIFEST = ROOT / "docs/eswar/phase1-data/il7ra-manifest.json"
HOTSPOTS = [100, 101, 102, 158, 159, 161, 211, 212, 213, 214]
DESIGN = "SVQEKLQEIQASLHSKVAKENVDEFKSLLEKANEENDLSVLLQASNPELLSLIKEALNSL"
ONE_TO_THREE = {v: k for k, v in AA3.items()}
AVAILABLE = EXAMPLE.is_dir() and REAL.is_dir() and MANIFEST.is_file()


def random_sequence(seed, length=60):
    rng = random.Random(seed)
    return "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(length))


def mutate(sequence, positions):
    out = list(sequence)
    for i in positions:
        out[i] = "G" if out[i] != "G" else "A"
    return "".join(out)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def cif(chains):
    """A minimal CA-only mmCIF. chains: {chain_id: (sequence, coords)}"""
    rows = ["data_test", "loop_", "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol", "_atom_site.label_atom_id",
            "_atom_site.label_comp_id", "_atom_site.label_asym_id", "_atom_site.label_seq_id", "_atom_site.Cartn_x",
            "_atom_site.Cartn_y", "_atom_site.Cartn_z", "_atom_site.pdbx_PDB_model_num"]
    n = 0
    for chain, (sequence, coords) in chains.items():
        for i, (aa, (x, y, z)) in enumerate(zip(sequence, coords), 1):
            n += 1
            rows.append(f"ATOM {n} C CA {ONE_TO_THREE[aa]} {chain} {i} {x:.3f} {y:.3f} {z:.3f} 1")
    return "\n".join(rows) + "\n"


class Designs:
    """Builds results. Each `kind` is a design with a known relationship to the first one ('base')."""

    def __init__(self, workdir):
        workdir = Path(workdir)
        write_conditioning(EXAMPLE, workdir / "ref", HOTSPOTS, 60, hotspot_source="provisional_historical")
        reference = load_reference(workdir / "ref", MANIFEST)
        self.reference = {k: v for k, v in reference.items() if k != "target_sequence"}
        self.target_sequence = reference["target_sequence"]
        _, self.base_xyz = read_chain_ca(REAL / "generator.cif", "B")
        _, self.target_xyz = read_chain_ca(REAL / "generator.cif", "A")
        self.reference_sha = sha((workdir / "ref/target.cif").read_bytes())
        self.manifest_sha = sha(MANIFEST.read_bytes())
        self.kinds = {                                   # kind -> (sequence, CA coordinates)
            "base": (DESIGN, self.base_xyz),
            "exact": (DESIGN, self.target_xyz[0:60]),                                  # same sequence
            "twin": (random_sequence(11), self.base_xyz),                              # different sequence, same backbone
            "mutant": (mutate(DESIGN, [6, 18, 30, 42, 54]), self.target_xyz[30:90]),   # 92% identical sequence
            "fold1": (random_sequence(21), self.target_xyz[0:60]),                     # different sequence and backbone
            "fold2": (random_sequence(22), self.target_xyz[60:120]),
            "fold3": (random_sequence(23), self.target_xyz[120:180]),
        }

    def result(self, run_id, seed, kind, length=60, worker_seconds=60.0, break_hash=False, wrong_manifest=False):
        sequence, xyz = self.kinds[kind]
        sequence, xyz = sequence[:length], xyz[:length]
        model = cif({"B": (sequence, xyz), "A": (self.target_sequence, self.target_xyz)}).encode()
        audit = {
            "status": "conditioned_generation; not a submission-eligible candidate or binding result",
            "reference_audit": self.reference, "reference_audit_sha256": "0" * 64,
            "target_manifest_sha256": "f" * 64 if wrong_manifest else self.manifest_sha,
            "spec_sha256": "0" * 64, "reference_sha256": self.reference_sha, "target_chain": "A", "binder_chain": "B",
            "binder_sequence": sequence, "binder_length_requested": length if isinstance(length, int) else list(length),
            "model": {"path": "/ckpt", "revision": "r"}, "code_revision": "ddd934f7e6b86f7582ce487d5b19f6c526c644b5",
            "seed": seed, "replicates": 1, "batch_size": 1, "compilation": False, "external_metrics": False,
            "recycle_strategy": "thinking_high", "generated_structure": f"/cache/runs/{run_id}/generation/model.cif",
            "generated_sha256": "0" * 64 if break_hash else sha(model),
            "chains": [{"id": "B", "sequence": sequence}, {"id": "A", "sequence": self.target_sequence}],
            "raw_structure_hashes": {}, "generation_seconds": worker_seconds, "peak_reserved_mib": 1.0,
            "peak_allocated_mib": 1.0, "host_peak_rss_mib": 1.0}
        return {"run_id": run_id, "audit": audit, "worker_seconds": worker_seconds, "gpu": "Fake GPU", "record": {},
                "archive": archive({f"{run_id}/generation/model.cif": model,
                                    f"{run_id}/generation/audit.json": json.dumps(audit).encode()})}


def archive(members):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return stream.getvalue()
