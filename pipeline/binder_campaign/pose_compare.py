"""Compare a predicted binder pose with a crystal structure of a known binder (positive control). Pure Python plus numpy.

The target in the crystal and in the prediction are superposed on their shared residues (C-alpha, Kabsch), and the RMSD of the binder's
C-alpha atoms is measured after that superposition: it says how far the predicted binder sits from where the crystal shows it. Contacts
(any heavy-atom pair within a cutoff) are compared as sets of target residues. Residue numbers are matched, not assumed: the target by a
numbering offset, the binder by position.
"""
import numpy as np


def kabsch(moving, fixed):
    """Rotation and translation taking `moving` onto `fixed` (both N x 3, N >= 3)."""
    moving, fixed = np.asarray(moving, float), np.asarray(fixed, float)
    if moving.shape != fixed.shape or moving.ndim != 2 or moving.shape[1] != 3 or len(moving) < 3:
        raise ValueError("Need two matching N x 3 coordinate sets with N of at least 3")
    mc, fc = moving.mean(axis=0), fixed.mean(axis=0)
    u, _, vt = np.linalg.svd((moving - mc).T @ (fixed - fc))
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    return rotation, fc - rotation @ mc


def rmsd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def binder_shift(ref_target, mob_target, ref_binder, mob_binder):
    """(target RMSD after superposition, binder C-alpha RMSD after superposing the targets)."""
    rotation, shift = kabsch(mob_target, ref_target)
    moved_target = np.asarray(mob_target, float) @ rotation.T + shift
    moved_binder = np.asarray(mob_binder, float) @ rotation.T + shift
    if len(ref_binder) != len(mob_binder) or len(ref_binder) == 0:
        raise ValueError("The binders must have the same number of residues")
    return rmsd(ref_target, moved_target), rmsd(ref_binder, moved_binder)


def contact_residues(target_atoms, binder_atoms, cutoff=4.5):
    """Target residue ids with any heavy atom within `cutoff` of any binder heavy atom. Atoms are (residue id, x, y, z)."""
    t = np.array([[x, y, z] for _, x, y, z in target_atoms], float)
    b = np.array([[x, y, z] for _, x, y, z in binder_atoms], float)
    ids = np.array([r for r, *_ in target_atoms])
    if not len(t) or not len(b):
        return set()
    d2 = ((t[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return {int(r) for r in ids[(d2 <= cutoff * cutoff).any(axis=1)]}


def overlap(a, b):
    """(shared, only in a, only in b, Jaccard) for two sets of residue ids."""
    a, b = set(a), set(b)
    union = a | b
    return sorted(a & b), sorted(a - b), sorted(b - a), (len(a & b) / len(union) if union else 1.0)
