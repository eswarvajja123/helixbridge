"""Full-atom test structures derived from the REAL verified RFD4 design (examples/verified-modal-run/generator.cif).

The real file holds the 195-residue reference core (UniProt 37-231) and the 60-residue designed binder, every heavy atom. From it
this module builds what the pipeline reads later:
  * the generator structure of a candidate (the real coordinates, with the binder relabelled to any sequence), and
  * a refolded complex in the shape the predictors write: the full 219-residue target (UniProt 21-239) plus the binder.
The geometry is real, so contacts and distances are real; changes (a rigid shift of the binder, an extra atom placed 1.2 A from a
target atom, a third chain) are applied on purpose so each test knows the right answer from the construction.
Relabelling a residue name keeps the original atoms, so these files are not chemically valid: they test code, not chemistry.
"""
from pathlib import Path

import numpy as np

from binder_campaign.structure import AA3

ROOT = Path(__file__).resolve().parents[3]
REAL = ROOT / "examples/verified-modal-run/generator.cif"
ONE_TO_THREE = {v: k for k, v in AA3.items()}
TARGET_START, TARGET_END = 21, 239
CORE_START, CORE_END = 37, 231


def real_atoms():
    """[(chain, seq_id, comp, atom name, element, (x, y, z))] from the real generator file."""
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    d = MMCIF2Dict(str(REAL))
    return [(d["_atom_site.label_asym_id"][i], int(d["_atom_site.label_seq_id"][i]), d["_atom_site.label_comp_id"][i],
             d["_atom_site.label_atom_id"][i], d["_atom_site.type_symbol"][i],
             tuple(float(d[k][i]) for k in ("_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z")))
            for i in range(len(d["_atom_site.id"]))]


def write_cif(atoms):
    """Slim mmCIF with the columns the pipeline reads. atoms: [(chain, seq_id, comp, name, element, xyz)]."""
    rows = ["data_test", "loop_", "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol", "_atom_site.label_atom_id",
            "_atom_site.label_comp_id", "_atom_site.label_asym_id", "_atom_site.label_seq_id", "_atom_site.Cartn_x",
            "_atom_site.Cartn_y", "_atom_site.Cartn_z", "_atom_site.pdbx_PDB_model_num"]
    for n, (chain, seq_id, comp, name, element, (x, y, z)) in enumerate(atoms, 1):
        rows.append(f"ATOM {n} {element} {name} {comp} {chain} {seq_id} {x:.3f} {y:.3f} {z:.3f} 1")
    return ("\n".join(rows) + "\n").encode()


def real_sequences():
    atoms = real_atoms()
    out = {}
    for chain, _, comp, name, *_ in atoms:
        if name == "CA":
            out[chain] = out.get(chain, "") + AA3[comp]
    return out["A"], out["B"]


def _binder(atoms, sequence, shift=(0.0, 0.0, 0.0)):
    shift = np.asarray(shift, float)
    out = []
    for chain, seq_id, comp, name, element, xyz in atoms:
        if chain == "B":
            out.append(("B", seq_id, ONE_TO_THREE[sequence[seq_id - 1]], name, element, tuple(np.asarray(xyz) + shift)))
    return out


def _target_core(atoms):
    return [a for a in atoms if a[0] == "A"]


def generator_structure(binder_sequence, shift=(0.0, 0.0, 0.0)):
    """The generator file for a design: real core + the real binder relabelled to `binder_sequence`."""
    atoms = real_atoms()
    return write_cif(_target_core(atoms) + _binder(atoms, binder_sequence, shift))


def refold_structure(target_sequence, binder_sequence, shift=(0.0, 0.0, 0.0), extra=(), third_chain=False, binder_override=None,
                     _binder_atoms=None):
    """A predictor's complex: full target 21-239 (real core coordinates for 37-231, single far-away CA atoms for the loose ends)
    plus the binder shifted rigidly by `shift`. `extra` are additional binder atoms [(seq_id, name, element, xyz)]."""
    atoms = real_atoms()
    out = []
    for position in range(TARGET_START, TARGET_END + 1):
        seq_id = position - TARGET_START + 1
        comp = ONE_TO_THREE[target_sequence[seq_id - 1]]
        if CORE_START <= position <= CORE_END:
            for chain, rid, rcomp, name, element, xyz in atoms:
                if chain == "A" and rid == position - CORE_START + 1:
                    assert rcomp == comp, "the real core and the target manifest disagree"
                    out.append(("A", seq_id, comp, name, element, xyz))
        else:
            out.append(("A", seq_id, comp, "CA", "C", (150.0 + 3.8 * seq_id, 150.0, 150.0)))
    binder = list(_binder_atoms) if _binder_atoms is not None else _binder(atoms, binder_override or binder_sequence, shift)
    for seq_id, name, element, xyz in extra:
        binder.append(("B", seq_id, ONE_TO_THREE[(binder_override or binder_sequence)[seq_id - 1]], name, element, tuple(xyz)))
    out += binder
    if third_chain:
        out += [("C", i + 1, a[2], a[3], a[4], tuple(np.asarray(a[5]) + 60.0)) for i, a in enumerate(b for b in binder if b[3] == "CA")]
    return write_cif(out)


def refold_custom(target_sequence, binder_atoms):
    """Like refold_structure, but with any binder atoms: [(chain 'B', seq_id, comp, name, element, xyz)]."""
    return refold_structure(target_sequence, "A" * 60, binder_override=None, _binder_atoms=binder_atoms)


def other_fold_binder(first=31, last=90, shift=(0.0, 0.0, 0.0)):
    """A different fold: real target residues first..last (ids in the 195-residue core) used as a 'binder' (chain B, ids from 1),
    moved by `shift`. Returns (sequence, atoms)."""
    shift = np.asarray(shift, float)
    atoms = [a for a in real_atoms() if a[0] == "A" and first <= a[1] <= last]
    sequence = "".join(AA3[a[2]] for a in atoms if a[3] == "CA")
    return sequence, [("B", a[1] - first + 1, a[2], a[3], a[4], tuple(np.asarray(a[5]) + shift)) for a in atoms]


def clash_atom(seed=0, distance=1.2):
    """A point `distance` from one real target atom and no closer than 2.2 A to any other target atom, away from the binder."""
    atoms = real_atoms()
    target = [a for a in atoms if a[0] == "A" and a[1] == 100 - CORE_START + 1 and a[3] == "CA"][0]
    center = np.asarray(target[5])
    others = np.array([a[5] for a in atoms if a[0] == "A" and a is not target])
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(3000):
        v = rng.normal(size=3)
        point = center + distance * v / np.linalg.norm(v)
        gap = np.linalg.norm(others - point, axis=1).min()
        if best is None or gap > best[0]:
            best = (gap, point)
    assert best[0] >= 1.9, "no clean place for the planted atom"
    return tuple(best[1])


def brute_force(cif_bytes, target_chain="A", binder_chain="B"):
    """Independent re-computation, straight from the file text: (close heavy-atom pairs < 2 A, min distance, contact residue ids)."""
    rows = [line.split() for line in cif_bytes.decode().splitlines() if line.startswith("ATOM ")]
    by = {}
    for r in rows:
        by.setdefault(r[5], []).append((int(r[6]), np.array([float(r[7]), float(r[8]), float(r[9])]), r[2]))
    t, b = by[target_chain], by[binder_chain]
    tx, bx = np.array([x for _, x, e in t if e != "H"]), np.array([x for _, x, e in b if e != "H"])
    tid = [i for i, _, e in t if e != "H"]
    dist = np.linalg.norm(tx[:, None] - bx[None], axis=-1)
    return int((dist < 2).sum()), float(dist.min()), sorted({tid[i] for i in np.where((dist < 5).any(axis=1))[0]})
