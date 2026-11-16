from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.dedup import (DedupGate, Thresholds, backbone_tm, check_sequence, contiguous_fraction,
                                   read_chain_ca, sequence_identity)

try:
    import tmtools  # noqa: F401
    HAVE_TM = True
except ImportError:
    HAVE_TM = False

REAL = Path(__file__).resolve().parents[2] / "examples/verified-modal-run"
DESIGN = "SVQEKLQEIQASLHSKVAKENVDEFKSLLEKANEENDLSVLLQASNPELLSLIKEALNSL"  # the real verified 60-aa RFD4 design


def random_sequence(seed, length=60):
    rng = random.Random(seed)
    return "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(length))


def mutate(sequence, positions):
    out = list(sequence)
    for i in positions:
        out[i] = "G" if out[i] != "G" else "A"
    return "".join(out)


class SequenceTests(unittest.TestCase):
    def test_identical_and_symmetric(self):
        other = random_sequence(1)
        self.assertEqual(sequence_identity(DESIGN, DESIGN), 1.0)
        for a, b in ((DESIGN, other), (DESIGN, DESIGN[:50]), (other, mutate(other, [3, 30]))):
            self.assertEqual(sequence_identity(a, b), sequence_identity(b, a))
            self.assertEqual(contiguous_fraction(a, b), contiguous_fraction(b, a))

    def test_scattered_near_copy_is_caught_by_identity_but_not_by_contiguity(self):
        copy = mutate(DESIGN, [6, 18, 30, 42, 54])
        self.assertAlmostEqual(sequence_identity(DESIGN, copy), 55 / 60)
        self.assertLess(contiguous_fraction(DESIGN, copy), 0.5)  # why a contiguous test alone is not enough

    def test_unrelated_designs_are_far_apart(self):
        for seed in range(2, 8):
            self.assertLess(sequence_identity(DESIGN, random_sequence(seed)), 0.5)

    def test_truncation_and_insertion_are_still_duplicates(self):
        self.assertEqual(sequence_identity(DESIGN, DESIGN[:50]), 1.0)  # relative to the shorter one
        inserted = DESIGN[:30] + "GGG" + DESIGN[30:]
        self.assertGreaterEqual(sequence_identity(DESIGN, inserted), 0.95)

    def test_bad_sequences_are_rejected(self):
        for bad in ("", "acdef", "ACDEX", "ACD EF", None, 5, ["A"]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                check_sequence(bad)


class ThresholdTests(unittest.TestCase):
    def test_valid_and_invalid(self):
        self.assertEqual(Thresholds(identity=1.0).identity, 1.0)
        for bad in (0, -0.1, 1.01, True, "0.8", None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                Thresholds(identity=bad)
            with self.assertRaises(ValueError, msg=repr(bad)):
                Thresholds(tm=bad)


class GateSequenceTests(unittest.TestCase):
    def test_first_design_is_kept_with_no_neighbour(self):
        decision = DedupGate().offer("d1", DESIGN)
        self.assertTrue(decision.kept)
        self.assertEqual((decision.reason, decision.nearest_id, decision.identity), ("kept", None, None))

    def test_exact_and_near_duplicates_are_dropped_with_reasons(self):
        gate = DedupGate()
        gate.offer("d1", DESIGN)
        exact = gate.offer("d2", DESIGN)
        near = gate.offer("d3", mutate(DESIGN, [6, 18, 30, 42, 54]))
        self.assertEqual((exact.kept, exact.reason, exact.nearest_id), (False, "exact_duplicate", "d1"))
        self.assertEqual((near.kept, near.reason, near.nearest_id), (False, "sequence_identity", "d1"))
        self.assertAlmostEqual(near.identity, 55 / 60)

    def test_unrelated_design_is_kept_and_nearest_neighbour_is_recorded(self):
        gate = DedupGate()
        gate.offer("d1", DESIGN)
        decision = gate.offer("d2", random_sequence(9))
        self.assertTrue(decision.kept)
        self.assertEqual(decision.nearest_id, "d1")
        self.assertLess(decision.identity, 0.5)

    def test_a_design_just_under_the_threshold_is_kept(self):
        gate = DedupGate(Thresholds(identity=0.80))
        gate.offer("d1", DESIGN)
        under = mutate(DESIGN, range(0, 60, 4))   # 15 changes of 60 -> 75% identical
        self.assertAlmostEqual(sequence_identity(DESIGN, under), 45 / 60)
        self.assertTrue(gate.offer("d2", under).kept)
        over = mutate(DESIGN, range(0, 48, 6))     # 8 changes of 60 -> about 87% identical
        self.assertFalse(gate.offer("d3", over).kept)

    def test_contiguous_overlap_reason_with_a_stricter_identity_threshold(self):
        gate = DedupGate(Thresholds(identity=1.0, contiguous=0.9))
        gate.offer("d1", DESIGN)
        decision = gate.offer("d2", mutate(DESIGN, [59]))   # one change at the end: 59 identical in a row
        self.assertEqual((decision.kept, decision.reason), (False, "contiguous_overlap"))

    def test_order_matters_and_nothing_is_lost(self):
        sibling = mutate(DESIGN, [6, 18, 30, 42, 54])
        forward, backward = DedupGate(), DedupGate()
        forward.offer("a", DESIGN), forward.offer("b", sibling)
        backward.offer("b", sibling), backward.offer("a", DESIGN)
        self.assertEqual((forward.kept_ids, backward.kept_ids), (["a"], ["b"]))
        self.assertEqual(len(forward.decisions), 2)      # the dropped design is still on record
        self.assertEqual([d.candidate_id for d in forward.dropped], ["b"])

    def test_most_similar_matching_design_is_reported(self):
        gate = DedupGate()
        gate.offer("far", mutate(DESIGN, range(0, 60, 10)))   # 6 changes: about 90% identical to DESIGN
        gate.offer("other", random_sequence(11))
        decision = gate.offer("probe", DESIGN)
        self.assertFalse(decision.kept)
        self.assertEqual(decision.nearest_id, "far")

    def test_ids_and_summary(self):
        gate = DedupGate()
        gate.offer("d1", DESIGN)
        gate.offer("d2", DESIGN)
        gate.offer("d3", random_sequence(5))
        with self.assertRaises(ValueError):
            gate.offer("d1", random_sequence(6))
        for bad in ("", None, 3):
            with self.assertRaises(ValueError):
                gate.offer(bad, random_sequence(7))
        summary = gate.summary()
        self.assertEqual((summary["offered"], summary["kept"], summary["dropped"]), (3, 2, 1))
        self.assertEqual(summary["dropped_by_reason"], {"exact_duplicate": 1})
        self.assertEqual(summary["thresholds"], {"identity": 0.8, "contiguous": 0.9, "tm": 0.9})

    def test_rejected_input_does_not_change_the_gate(self):
        gate = DedupGate()
        gate.offer("d1", DESIGN)
        with self.assertRaises(ValueError):
            gate.offer("d2", "NOT A SEQUENCE")
        self.assertEqual((gate.kept_ids, len(gate.decisions)), (["d1"], 1))

    def test_bad_coordinates_are_rejected(self):
        gate = DedupGate()
        good = [[float(i), 0.0, 0.0] for i in range(60)]
        for bad in (good[:59], [[0.0, 0.0]] * 60, [[float("nan"), 0.0, 0.0]] * 60):
            with self.assertRaises(ValueError):
                gate.offer("x", DESIGN, bad)
        self.assertTrue(gate.offer("x", DESIGN, good).kept)


@unittest.skipUnless(HAVE_TM and REAL.is_dir(), "needs tmtools and examples/verified-modal-run")
class BackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seq_gen, cls.xyz_gen = read_chain_ca(REAL / "generator.cif", "B")      # RFD4 pose of the real design
        cls.seq_boltz, cls.xyz_boltz = read_chain_ca(REAL / "boltz.cif", "B")      # Boltz-2 refold of the same design
        _, target = read_chain_ca(REAL / "generator.cif", "A")
        cls.seq_other = random_sequence(21)
        cls.xyz_unrelated = target[60:120]                                         # a different protein, 60 residues

    def test_chain_reader(self):
        self.assertEqual(self.seq_gen, DESIGN)
        self.assertEqual(self.xyz_gen.shape, (60, 3))
        with self.assertRaises(ValueError):
            read_chain_ca(REAL / "generator.cif", "Z")

    def test_same_design_from_two_models_is_the_same_backbone(self):
        self.assertGreater(backbone_tm(self.xyz_gen, self.seq_gen, self.xyz_boltz, self.seq_boltz), 0.95)

    def test_unrelated_chain_is_a_different_backbone(self):
        self.assertLess(backbone_tm(self.xyz_gen, self.seq_gen, self.xyz_unrelated, self.seq_other), 0.4)

    def test_same_backbone_with_a_different_sequence_is_dropped_by_backbone(self):
        gate = DedupGate()
        gate.offer("rfd4", DESIGN, self.xyz_gen)
        self.assertLess(sequence_identity(DESIGN, self.seq_other), 0.5)             # sequences alone would keep it
        decision = gate.offer("same-fold", self.seq_other, self.xyz_boltz)
        self.assertEqual((decision.kept, decision.reason, decision.nearest_id), (False, "backbone_tm", "rfd4"))
        self.assertGreater(decision.tm, 0.95)

    def test_different_backbone_and_sequence_is_kept(self):
        gate = DedupGate()
        gate.offer("rfd4", DESIGN, self.xyz_gen)
        decision = gate.offer("other", self.seq_other, self.xyz_unrelated)
        self.assertTrue(decision.kept)
        self.assertLess(decision.tm, 0.4)

    def test_without_coordinates_only_sequences_are_compared(self):
        gate = DedupGate()
        gate.offer("rfd4", DESIGN, self.xyz_gen)
        decision = gate.offer("no-coords", self.seq_other)   # same fold would be missed: documented limit
        self.assertTrue(decision.kept)
        self.assertIsNone(decision.tm)


if __name__ == "__main__":
    unittest.main()
