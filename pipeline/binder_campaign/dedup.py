"""Near-duplicate gate for candidate binders, applied BEFORE spending evaluation compute (Phase 2, Task 2.2).

A design is dropped when it is too similar to one already kept, by any of:
  * aligned sequence identity (global alignment, relative to the shorter sequence), which catches
    copies with scattered point changes that a contiguous-match test misses;
  * a long identical contiguous stretch (the repository's existing >90% overlap diagnostic);
  * backbone similarity (TM-score of CA traces), when both designs supply coordinates.

The gate is GREEDY and ORDER-DEPENDENT: the first design offered is kept. Offer designs in generation
order. Dropped designs are never discarded silently: every offer returns a Decision, and `decisions`
keeps all of them with the metrics and the kept design they matched.

The default thresholds are starting values, NOT calibrated on a population of real sibling designs
(the repository holds one real design). They are chosen to be conservative; tune them on real pools.
Backbone similarity needs the optional `tmtools` package (TM-align), imported only when coordinates are used.
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

from Bio.Align import PairwiseAligner
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from .sequence_review import ALPHABET, longest_common_substring
from .structure import AA3

_ALIGNER = PairwiseAligner()
_ALIGNER.mode = "global"
_ALIGNER.match_score = 1
_ALIGNER.mismatch_score = 0
_ALIGNER.open_gap_score = -1
_ALIGNER.extend_gap_score = -0.5


@dataclass(frozen=True)
class Thresholds:
    identity: float = 0.80    # aligned identity over the shorter sequence
    contiguous: float = 0.90  # longest identical stretch over the shorter sequence
    tm: float = 0.90          # backbone TM-score (the lower of the two normalizations)

    def __post_init__(self):
        for name in ("identity", "contiguous", "tm"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value <= 1:
                raise ValueError(f"Threshold {name} must be in (0, 1]")


@dataclass(frozen=True)
class Decision:
    candidate_id: str
    kept: bool
    reason: str                 # "kept" or why it was dropped
    nearest_id: str | None      # the most similar design already kept (also recorded for kept designs)
    identity: float | None
    contiguous: float | None
    tm: float | None

    def as_dict(self):
        return asdict(self)


def check_sequence(sequence):
    if not isinstance(sequence, str) or not sequence or set(sequence) - ALPHABET:
        raise ValueError("Require a nonempty sequence of the 20 standard uppercase amino acids")
    return sequence


def sequence_identity(a, b):
    """Identical aligned positions in the best global alignment, divided by the shorter length."""
    check_sequence(a), check_sequence(b)
    identities = _ALIGNER.align(a, b)[0].counts().identities
    return identities / min(len(a), len(b))


def contiguous_fraction(a, b):
    """Longest identical contiguous stretch divided by the shorter length."""
    check_sequence(a), check_sequence(b)
    return longest_common_substring(a, b)[0] / min(len(a), len(b))


def backbone_tm(coords_a, seq_a, coords_b, seq_b):
    """TM-score of two CA traces; returns the lower of the two length normalizations."""
    try:
        from tmtools import tm_align
    except ImportError as error:
        raise ImportError("Backbone comparison needs the 'tmtools' package (pip install tmtools==0.3.0)") from error
    result = tm_align(coords_a, coords_b, seq_a, seq_b)
    return float(min(result.tm_norm_chain1, result.tm_norm_chain2))


def read_chain_ca(cif_path, chain):
    """(sequence, CA coordinates) of one chain of model 1 of an mmCIF, by label_asym_id."""
    import numpy as np
    cif = MMCIF2Dict(str(Path(cif_path)))
    models = cif.get("_atom_site.pdbx_PDB_model_num")
    first = models[0] if models else None
    letters, coords = [], []
    for i, atom in enumerate(cif["_atom_site.label_atom_id"]):
        if atom != "CA" or cif["_atom_site.label_asym_id"][i] != chain or (models and models[i] != first):
            continue
        residue = cif["_atom_site.label_comp_id"][i]
        if residue not in AA3:
            raise ValueError("Nonstandard residue in chain: " + residue)
        letters.append(AA3[residue])
        coords.append([float(cif[k][i]) for k in ("_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z")])
    if not letters:
        raise ValueError(f"No CA atoms found for chain {chain}")
    return "".join(letters), np.array(coords)


class DedupGate:
    def __init__(self, thresholds=None):
        self.thresholds = thresholds or Thresholds()
        self._kept = {}          # candidate_id -> (sequence, coords or None)
        self.decisions = []

    def offer(self, candidate_id, sequence, coords=None):
        """Decide whether one design is new enough to keep. Never raises for a duplicate; returns the Decision."""
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("A nonempty candidate id is required")
        if candidate_id in {d.candidate_id for d in self.decisions}:
            raise ValueError("Candidate id offered twice: " + candidate_id)
        check_sequence(sequence)
        if coords is not None:
            if len(coords) != len(sequence) or any(len(c) != 3 or not all(math.isfinite(v) for v in c) for c in coords):
                raise ValueError("Coordinates must be one finite (x, y, z) per residue")
        t = self.thresholds
        compared = []  # one row per kept design: (score, other_id, identity, contiguous, tm, reason or None)
        for other_id, (other_seq, other_coords) in self._kept.items():
            identity = sequence_identity(sequence, other_seq)
            contiguous = contiguous_fraction(sequence, other_seq)
            tm = backbone_tm(coords, sequence, other_coords, other_seq) if coords is not None and other_coords is not None else None
            if sequence == other_seq:
                reason = "exact_duplicate"
            elif identity >= t.identity:
                reason = "sequence_identity"
            elif contiguous > t.contiguous:
                reason = "contiguous_overlap"
            elif tm is not None and tm >= t.tm:
                reason = "backbone_tm"
            else:
                reason = None
            compared.append((max(identity, contiguous, tm or 0.0), other_id, identity, contiguous, tm, reason))
        triggered = [row for row in compared if row[5]]
        if triggered:  # report the most similar kept design among those that caused the drop
            _, other_id, identity, contiguous, tm, reason = max(triggered, key=lambda row: row[0])
            decision = Decision(candidate_id, False, reason, other_id, identity, contiguous, tm)
        else:
            nearest = max(compared, key=lambda row: row[0], default=None)  # recorded even though it was kept
            decision = Decision(candidate_id, True, "kept", *(nearest[1:5] if nearest else (None,) * 4))
            self._kept[candidate_id] = (sequence, coords)
        self.decisions.append(decision)
        return decision

    @property
    def kept_ids(self):
        return list(self._kept)

    @property
    def dropped(self):
        return [d for d in self.decisions if not d.kept]

    def summary(self):
        by_reason = {}
        for d in self.dropped:
            by_reason[d.reason] = by_reason.get(d.reason, 0) + 1
        return {"offered": len(self.decisions), "kept": len(self._kept), "dropped": len(self.dropped),
                "dropped_by_reason": by_reason, "thresholds": asdict(self.thresholds)}
