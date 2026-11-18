"""Measure AlphaFold-Multimer (ColabFold) predictions of the shortlisted designs (Phase 5, Task 5.4).

Reads what `modal_alphafold.py` returns (one PDB and one scores file per design) and measures, with the same tested functions the structure
QC uses: AlphaFold's own confidence (pTM, ipTM, pLDDT, interface PAE), the contacts with the target and with the design hotspots, close
atom pairs, and how far the binder sits from where the generator, ESMFold2 and Boltz-2 put it (after aligning on the shared reference core).

This records what AlphaFold says. It does NOT decide anything: `independent_prediction_status` stays a person's, the 'interface looks as
designed' judgement is left blank, and no threshold is applied (the team's pilot found high-confidence scores on non-binders).
"""
import json
from pathlib import Path
import re
import tarfile

import numpy as np

from .qc import _measure, _shift, inspect_structure
from .structure import AA3

SCHEMA_VERSION = 1
CIF_COLUMNS = ["group_PDB", "id", "type_symbol", "label_atom_id", "label_comp_id", "label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y",
               "Cartn_z", "pdbx_PDB_model_num"]


def parse_pdb(text):
    """[(chain, residue number, residue name, atom name, element, (x, y, z))] for the ATOM records of the first model."""
    atoms = []
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        name, comp, chain = line[12:16].strip(), line[17:20].strip(), line[21]
        element = (line[76:78].strip() or re.sub(r"[^A-Za-z]", "", name)[:1]).upper()
        if comp not in AA3:
            raise ValueError("Nonstandard residue in the prediction: " + comp)
        atoms.append((chain, int(line[22:26]), comp, name, element, (float(line[30:38]), float(line[38:46]), float(line[46:54]))))
    if not atoms:
        raise ValueError("The prediction holds no atoms")
    return atoms


def pdb_to_cif(text):
    """A minimal mmCIF (the columns the pipeline reads) from a PDB. Hydrogens are dropped."""
    rows = ["data_alphafold", "loop_"] + [f"_atom_site.{c}" for c in CIF_COLUMNS]
    n = 0
    for chain, number, comp, name, element, (x, y, z) in parse_pdb(text):
        if element in ("H", "D"):
            continue
        n += 1
        rows.append(f"ATOM {n} {element} {name} {comp} {chain} {number} {x:.3f} {y:.3f} {z:.3f} 1")
    return "\n".join(rows) + "\n"


def load_scores(text):
    """pTM, ipTM, pLDDT and the PAE matrix from a ColabFold scores file; refuses anything non-finite or inconsistent."""
    d = json.loads(text)
    for key in ("ptm", "iptm", "plddt", "pae"):
        if key not in d:
            raise ValueError("The scores file lacks " + key)
    plddt, pae = np.asarray(d["plddt"], float), np.asarray(d["pae"], float)
    if not all(np.isfinite(v).all() for v in (plddt, pae)) or not np.isfinite([d["ptm"], d["iptm"]]).all():
        raise ValueError("Non-finite confidence values")
    if pae.shape != (len(plddt), len(plddt)):
        raise ValueError("The PAE matrix does not match the number of residues")
    return {"ptm": float(d["ptm"]), "iptm": float(d["iptm"]), "plddt": plddt, "pae": pae}


def confidence(scores, target_length, binder_length):
    """AlphaFold's own numbers for one design. Chain order in the input is TARGET then BINDER."""
    n = target_length + binder_length
    if len(scores["plddt"]) != n:
        raise ValueError("The prediction has a different number of residues than the design")
    pae, plddt = scores["pae"], scores["plddt"]
    inter = np.concatenate([pae[:target_length, target_length:].ravel(), pae[target_length:, :target_length].ravel()])
    return {"ptm": round(scores["ptm"], 4), "iptm": round(scores["iptm"], 4), "plddt_target_mean": round(float(plddt[:target_length].mean()), 2),
            "plddt_binder_mean": round(float(plddt[target_length:].mean()), 2), "interface_pae_mean_A": round(float(inter.mean()), 2),
            "interface_pae_min_A": round(float(inter.min()), 2)}


def review_design(record, af_pdb_text, af_scores_text, target, other_structures):
    """Everything measured for one design. `other_structures` maps a source name to the path of its structure (generator, esmfold2, boltz2)."""
    import tempfile
    seq = record["sequence"]
    scores = load_scores(af_scores_text)
    out = {"candidate_id": record["candidate_id"], "binder_length": len(seq), **confidence(scores, len(target["sequence"]), len(seq))}
    positions = list(range(target["start"], target["end"] + 1))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "alphafold.cif"
        path.write_text(pdb_to_cif(af_pdb_text))
        pose, problems = inspect_structure(path, "alphafold2", target["sequence"], seq, positions)
        if pose is None:
            return {**out, "structure": "unreadable", "problems": problems}
        out["structure"] = "checked"
        out.update({k: v for k, v in _measure(pose, record.get("hotspots_uniprot")).items() if k != "status"})
        reference = (record.get("generator_audit") or {}).get("reference_audit") or {}
        core = sorted(set(reference.get("alignment_positions") or reference.get("positions") or []) & set(positions))
        out["pose_shift_A"] = {}
        for name, source_path in other_structures.items():
            if source_path is None:
                continue
            if name == "generator":
                seq_ref, pos_ref = reference.get("reference_sequence"), reference.get("positions")
                other, bad = inspect_structure(source_path, name, seq_ref, record.get("generator_structure_sequence") or seq, pos_ref) if seq_ref else (None, ["no reference"])
            else:
                other, bad = inspect_structure(source_path, name, target["sequence"], seq, positions)
            if other is None or len(core) < 30:
                out["pose_shift_A"][name] = None
                continue
            try:
                out["pose_shift_A"][name] = _shift(other, pose, core)[0]
            except ValueError:
                out["pose_shift_A"][name] = None
    return out


def read_archives(folder):
    """{design id: (pdb text, scores text)} from the chunk archives modal_alphafold.py wrote."""
    found = {}
    for archive in sorted(Path(folder).glob("c*.tar.gz")):
        with tarfile.open(archive, "r:gz") as tar:
            members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
            for name, member in members.items():
                m = re.fullmatch(r"(.+)_unrelaxed_rank_001_.*\.pdb", name)
                if not m:
                    continue
                design = m.group(1)
                scores = [n for n in members if n.startswith(f"{design}_scores_rank_001_") and n.endswith(".json")]
                if len(scores) != 1:
                    raise ValueError("Expected exactly one best-model scores file for " + design)
                if design in found:
                    raise ValueError("Design predicted twice: " + design)
                found[design] = (tar.extractfile(member).read().decode(), tar.extractfile(members[scores[0]]).read().decode())
    return found
