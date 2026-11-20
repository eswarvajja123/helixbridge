from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign import pose_compare as P  # noqa: E402


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class Kabsch(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.points = rng.normal(size=(30, 3)) * 10

    def test_recovers_a_rigid_motion(self):
        r = rotation_z(0.7)
        moved = self.points @ r.T + np.array([3.0, -2.0, 5.0])
        rotation, shift = P.kabsch(moved, self.points)
        self.assertLess(P.rmsd(moved @ rotation.T + shift, self.points), 1e-8)

    def test_a_mirror_image_is_not_accepted_as_a_rotation(self):
        mirrored = self.points * np.array([1, 1, -1])
        rotation, _ = P.kabsch(mirrored, self.points)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)                  # a proper rotation only
        self.assertGreater(P.rmsd(mirrored @ rotation.T, self.points), 1.0)

    def test_refusals(self):
        with self.assertRaises(ValueError):
            P.kabsch(self.points[:2], self.points[:2])
        with self.assertRaises(ValueError):
            P.kabsch(self.points, self.points[:5])


class BinderShift(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.target = rng.normal(size=(40, 3)) * 10
        self.binder = rng.normal(size=(15, 3)) * 3 + np.array([20.0, 0, 0])

    def test_the_same_complex_moved_rigidly_has_no_shift(self):
        r = rotation_z(1.1)
        t = self.target @ r.T + 7
        b = self.binder @ r.T + 7
        target_rmsd, binder_rmsd = P.binder_shift(self.target, t, self.binder, b)
        self.assertLess(max(target_rmsd, binder_rmsd), 1e-8)

    def test_a_displaced_binder_is_measured_after_the_targets_are_superposed(self):
        r = rotation_z(0.4)
        t = self.target @ r.T
        b = (self.binder + np.array([0, 5.0, 0])) @ r.T                              # the binder sits 5 A elsewhere
        _, binder_rmsd = P.binder_shift(self.target, t, self.binder, b)
        self.assertAlmostEqual(binder_rmsd, 5.0, places=6)

    def test_binders_of_different_length_are_refused(self):
        with self.assertRaises(ValueError):
            P.binder_shift(self.target, self.target, self.binder, self.binder[:5])


class Contacts(unittest.TestCase):
    def test_cutoff_is_inclusive_and_residues_are_reported_once(self):
        target = [(10, 0, 0, 0), (10, 1, 0, 0), (11, 10, 0, 0), (12, 0, 4.5, 0)]
        binder = [(1, 0, 0, 1.0)]
        self.assertEqual(P.contact_residues(target, binder, 4.5), {10})                 # residue 10 twice counts once; 11 is far; 12 is 4.61 A away
        self.assertEqual(P.contact_residues([(12, 0, 4.5, 0)], [(1, 0, 0, 0)], 4.5), {12})        # exactly at the cutoff counts
        self.assertEqual(P.contact_residues([(12, 0, 4.6, 0)], [(1, 0, 0, 0)], 4.5), set())
        self.assertEqual(P.contact_residues([], binder), set())

    def test_overlap(self):
        shared, only_a, only_b, jaccard = P.overlap({1, 2, 3}, {2, 3, 4, 5})
        self.assertEqual((shared, only_a, only_b), ([2, 3], [1], [4, 5]))
        self.assertAlmostEqual(jaccard, 2 / 5)
        self.assertEqual(P.overlap(set(), set())[3], 1.0)


if __name__ == "__main__":
    unittest.main()
