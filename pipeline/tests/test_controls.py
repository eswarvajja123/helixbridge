import collections
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign import controls as C

AA = "ACDEFGHIKLMNPQRSTVWY"


def seq(seed, n=60):
    rng = random.Random(seed)
    return "".join(rng.choice(AA) for _ in range(n))


def parent(i):
    return {"candidate_id": f"il7ra-rfd4-{i:012x}", "target_id": "il7ra", "sequence": seq(i), "seed": i, "hotspots_uniprot": [100, 101],
            "hotspot_source": "x", "generator_audit": {"reference_audit": {"reference_sequence": "ACD", "positions": [37, 38, 39]}}}


class Shuffle(unittest.TestCase):
    def test_same_length_and_composition_different_order(self):
        for i in range(200):
            s = seq(i, random.Random(i).randint(40, 100))
            t = C.shuffle_sequence(s, i)
            self.assertEqual((len(t), collections.Counter(t)), (len(s), collections.Counter(s)))
            self.assertNotEqual(t, s)

    def test_a_short_sequence_is_never_returned_unchanged(self):
        for seed in range(300):                                    # with two letters a plain shuffle returns the original about half the time
            self.assertNotEqual(C.shuffle_sequence("AB", seed), "AB")
            self.assertNotEqual(C.shuffle_sequence("AAAB", seed), "AAAB")

    def test_deterministic_and_seed_dependent(self):
        s = seq(1)
        self.assertEqual(C.shuffle_sequence(s, 5), C.shuffle_sequence(s, 5))
        self.assertNotEqual(C.shuffle_sequence(s, 5), C.shuffle_sequence(s, 6))

    def test_cannot_shuffle_a_single_repeated_residue(self):
        with self.assertRaises(ValueError):
            C.shuffle_sequence("A" * 50, 1)
        self.assertNotEqual(C.shuffle_sequence("AAAAAAAAAB", 1), "AAAAAAAAAB")


class Make(unittest.TestCase):
    def test_one_control_per_parent_with_provenance_and_no_promotion(self):
        parents = [parent(i) for i in range(1, 6)]
        out = C.make_controls(parents, seed=3)
        self.assertEqual(len(out), 5)
        for p, c in zip(parents, out):
            self.assertEqual(collections.Counter(c["sequence"]), collections.Counter(p["sequence"]))
            self.assertNotEqual(c["sequence"], p["sequence"])
            self.assertEqual(c["control"]["parent_candidate_id"], p["candidate_id"])
            self.assertEqual(c["generator_audit"]["reference_audit"]["positions"], [37, 38, 39])
            self.assertEqual((c["qc_reasons"], c["submission_eligible"], c["novelty_review"]), ([], False, "pending"))
            self.assertTrue(c["candidate_id"].startswith("il7ra-shuf-"))
        self.assertEqual(len({c["candidate_id"] for c in out}), 5)
        self.assertEqual([c["sequence"] for c in out], [c["sequence"] for c in C.make_controls(parents, seed=3)])

    def test_two_parents_with_the_same_id_tail_are_refused(self):
        a, b = parent(1), parent(2)
        b["candidate_id"] = "other-" + a["candidate_id"].rsplit("-", 1)[-1]
        with self.assertRaisesRegex(ValueError, "same control id"):
            C.make_controls([a, b])


class Separation(unittest.TestCase):
    def test_probabilities(self):
        self.assertEqual(C.separation([3, 4], [1, 2]), 1.0)
        self.assertEqual(C.separation([1, 2], [3, 4]), 0.0)
        self.assertEqual(C.separation([1, 2], [1, 2]), 0.5)
        self.assertEqual(C.separation([2, 2], [2, 2]), 0.5)
        self.assertEqual(C.separation([1, 3], [2]), 0.5)
        self.assertIsNone(C.separation([], [1]))

    def test_matches_a_brute_force_on_random_data(self):
        rng = random.Random(2)
        for _ in range(100):
            d = [rng.random() for _ in range(rng.randint(1, 12))]
            c = [rng.random() for _ in range(rng.randint(1, 12))]
            expect = sum(1 for x in d for y in c if x > y) / (len(d) * len(c))
            self.assertAlmostEqual(C.separation(d, c), expect, delta=0.0006)


def scored(esm_iptm, esm_ipsae, bz_iptm, bz_ipsae, status):
    mk = lambda i, s: {"status": "done", "score": {"iptm": i, "ipsae": {"max": s}}}
    return {"evaluations": {"esmfold2": mk(esm_iptm, esm_ipsae), "boltz2": mk(bz_iptm, bz_ipsae)}, "qc": {"status": status}}


class Report(unittest.TestCase):
    def test_numbers(self):
        designs = [scored(0.9, 0.8, 0.9, 0.8, "no_flags"), scored(0.95, 0.85, 0.85, 0.7, "no_flags"), scored(0.85, 0.7, 0.9, 0.8, "flagged")]
        controls = [scored(0.3, 0.0, 0.5, 0.1, "flagged"), scored(0.92, 0.5, 0.9, 0.6, "no_flags")]
        r = C.report(designs, controls)
        self.assertEqual((r["designs"], r["controls"]), (3, 2))
        self.assertEqual(r["measures"]["esmfold2_iptm"]["designs"], {"n": 3, "min": 0.85, "median": 0.9, "max": 0.95})
        self.assertEqual(r["measures"]["esmfold2_iptm"]["p_design_above_control"], round(4 / 6, 3))      # 0.9 and 0.85 beat 0.3 but not 0.92; 0.95 beats both
        self.assertEqual(r["measures"]["esmfold2_iptm"]["share_at_or_above"]["0.8"], {"designs": 1.0, "controls": 0.5})
        self.assertEqual(r["qc_clean"], {"designs": 2, "controls": 1, "design_share": round(2 / 3, 3), "control_share": 0.5})
        self.assertIn("easy negative", r["scope"])

    def test_an_incomplete_check_is_not_counted_as_clean(self):
        r = C.report([scored(0.9, 0.8, 0.9, 0.8, "incomplete"), scored(0.9, 0.8, 0.9, 0.8, "no_flags")], [scored(0.3, 0, 0.3, 0, "incomplete")])
        self.assertEqual(r["qc_clean"], {"designs": 1, "controls": 0, "design_share": 0.5, "control_share": 0.0})

    def test_a_record_missing_a_predictor_is_left_out_of_the_distribution(self):
        bad = {"evaluations": {"esmfold2": {"status": "failed"}, "boltz2": {"status": "done", "score": {"iptm": 0.1, "ipsae": {"max": 0}}}}, "qc": {"status": "flagged"}}
        r = C.report([scored(0.9, 0.8, 0.9, 0.8, "no_flags"), bad], [scored(0.3, 0, 0.3, 0, "flagged")])
        self.assertEqual(r["measures"]["boltz2_iptm"]["designs"]["n"], 1)


class Scripts(unittest.TestCase):
    def test_make_and_report(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            ps = [parent(i) for i in range(1, 4)]
            (t / "r.json").write_text(json.dumps({"candidates": ps}))
            (t / "s.json").write_text(json.dumps({"selected": [{"candidate_id": p["candidate_id"]} for p in ps]}))
            run = lambda name, *a: subprocess.run([sys.executable, "-I", str(Path(__file__).resolve().parents[1] / "scripts" / name), *a],
                                                  capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
            done = run("make-controls.py", "--records", str(t / "r.json"), "--selection", str(t / "s.json"), "--out", str(t / "c"))
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(json.loads((t / "c/candidates.json").read_text())["count"], 3)
            again = run("make-controls.py", "--records", str(t / "r.json"), "--selection", str(t / "s.json"), "--out", str(t / "c"))
            self.assertTrue(again.returncode != 0 and "REFUSED" in again.stderr and "Traceback" not in again.stderr)
            designs = [{**p, **scored(0.9, 0.8, 0.9, 0.8, "no_flags")} for p in ps]
            ctrl = [{"candidate_id": f"c{i}", **scored(0.3, 0.0, 0.4, 0.1, "flagged")} for i in range(3)]
            (t / "d.json").write_text(json.dumps({"candidates": designs}))
            (t / "k.json").write_text(json.dumps({"candidates": ctrl}))
            rep = run("controls-report.py", "--designs", str(t / "d.json"), "--selection", str(t / "s.json"), "--controls", str(t / "k.json"), "--out", str(t / "o"))
            self.assertEqual(rep.returncode, 0, rep.stderr)
            self.assertEqual(json.loads(rep.stdout)["p_design_above_control"]["esmfold2_iptm"], 1.0)
            miss = run("controls-report.py", "--designs", str(t / "d.json"), "--selection", str(t / "s.json"), "--controls", str(t / "nope.json"), "--out", str(t / "o2"))
            self.assertTrue(miss.returncode != 0 and "REFUSED" in miss.stderr and "Traceback" not in miss.stderr)


if __name__ == "__main__":
    unittest.main()
