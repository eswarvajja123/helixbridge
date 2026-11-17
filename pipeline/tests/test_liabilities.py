import csv
import io
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign import liabilities

AA = "ACDEFGHIKLMNPQRSTVWY"


def clean(seed, n=60):
    """A sequence with none of the four liabilities: no C, no N followed by non-P then S/T, no NG, no DP."""
    rng = random.Random(seed)
    letters = "ADEFGHIKLMQRSTVWY"
    out = "".join(rng.choice(letters) for _ in range(n))
    assert not (set(liabilities.flags_for(liabilities.sequence_diagnostic(out))))
    return out


def rec(cid, sequence, reasons=()):
    return {"candidate_id": cid, "sequence": sequence, "qc_reasons": list(reasons), "submission_eligible": False}


def independent_flags(seq):
    """The four flags recomputed from the sequence text, without the team's module."""
    out = []
    if seq.count("C") % 2 == 1:
        out.append("odd_cysteine_count")
    if any(seq[i] == "N" and seq[i + 1] != "P" and seq[i + 2] in "ST" for i in range(len(seq) - 2)):
        out.append("n_glycosylation_sequon")
    if "NG" in seq:
        out.append("asn_gly_motif")
    if "DP" in seq:
        out.append("asp_pro_motif")
    return out


class Flags(unittest.TestCase):
    def of(self, seq):
        return liabilities.review([rec("x", seq)])["x"]["flags"]

    def test_each_flag_alone(self):
        base = clean(1)
        self.assertEqual(self.of(base), [])
        self.assertEqual(self.of(base[:10] + "C" + base[11:]), ["odd_cysteine_count"])
        self.assertEqual(self.of(base[:10] + "CC" + base[12:]), [])                       # an even number is not flagged
        self.assertEqual(self.of(base[:10] + "NAS" + base[13:]), ["n_glycosylation_sequon"])
        self.assertEqual(self.of(base[:10] + "NPS" + base[13:]), [])                      # N-P-S is not a sequon
        self.assertEqual(self.of(base[:10] + "NAT" + base[13:]), ["n_glycosylation_sequon"])
        self.assertEqual(self.of(base[:10] + "NGA" + base[13:]), ["asn_gly_motif"])
        self.assertEqual(self.of(base[:10] + "DP" + base[12:]), ["asp_pro_motif"])

    def test_several_flags_together_and_order(self):
        base = clean(2)
        seq = base[:5] + "C" + base[6:15] + "NGA" + base[18:25] + "DP" + base[27:35] + "NAS" + base[38:]
        self.assertEqual(self.of(seq), ["odd_cysteine_count", "n_glycosylation_sequon", "asn_gly_motif", "asp_pro_motif"])

    def test_agrees_with_an_independent_recomputation_on_500_random_sequences(self):
        rng = random.Random(7)
        for _ in range(500):
            seq = "".join(rng.choice(AA) for _ in range(rng.randint(40, 100)))
            self.assertEqual(self.of(seq), independent_flags(seq), seq)

    def test_numbers_are_the_teams_not_ours(self):
        seq = "DEDEDEDEDE" + clean(3, 50)
        d = liabilities.review([rec("x", seq)])["x"]["diagnostic"]
        self.assertEqual(d["length"], 60)
        self.assertAlmostEqual(d["DE_fraction"], sum(c in "DE" for c in seq) / 60)

    def test_a_sequence_the_teams_code_cannot_read_is_reported_not_skipped(self):
        r = liabilities.review([rec("short", "ACDE"), rec("odd", "A" * 59 + "B")])
        self.assertEqual(r["short"]["flags"], ["not_diagnosable"])
        self.assertEqual(r["odd"]["flags"], ["not_diagnosable"])
        self.assertIsNone(r["short"]["diagnostic"])


class Run(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        base = clean(4)
        self.recs = [rec("a", clean(5)), rec("b", base[:10] + "C" + base[11:]), rec("c", base[:10] + "NGA" + base[13:], ["near_duplicate:x"]),
                     rec("d", "ACDE")]
        self.path = self.dir / "c.json"
        self.path.write_text(json.dumps({"candidates": self.recs, "count": 4, "note": "n"}))

    def test_outputs_and_that_nothing_is_removed_or_promoted(self):
        before = self.path.read_bytes()
        report = liabilities.run_review(self.path, self.dir / "o", log=lambda *_: None)
        self.assertEqual(self.path.read_bytes(), before)
        out = {r["candidate_id"]: r for r in json.loads((self.dir / "o" / "liabilities-candidates.json").read_text())["candidates"]}
        self.assertEqual(list(out), ["a", "b", "c", "d"])                                 # every design is still there
        for original in self.recs:
            self.assertEqual(out[original["candidate_id"]]["qc_reasons"], original["qc_reasons"])   # flags never add or remove reasons
            self.assertIs(out[original["candidate_id"]]["submission_eligible"], False)
        self.assertEqual(report["flag_counts"], {"odd_cysteine_count": 1, "n_glycosylation_sequon": 0, "asn_gly_motif": 1,
                                                 "asp_pro_motif": 0, "not_diagnosable": 1})
        self.assertEqual(report["designs_with_no_flags"], 1)
        self.assertIn("No numeric pass/fail thresholds", report["scope"])
        rows = list(csv.DictReader(io.StringIO((self.dir / "o" / "liabilities.csv").read_text())))
        self.assertEqual([r["flags"] for r in rows], ["", "odd_cysteine_count", "asn_gly_motif", "not_diagnosable"])
        self.assertEqual(rows[1]["cysteines"], "1")

    def test_fresh_folder_required(self):
        liabilities.run_review(self.path, self.dir / "o", log=lambda *_: None)
        with self.assertRaisesRegex(ValueError, "fresh output"):
            liabilities.run_review(self.path, self.dir / "o", log=lambda *_: None)

    def test_script(self):
        script = Path(__file__).resolve().parents[1] / "scripts/review-liabilities.py"
        run = lambda *a: subprocess.run([sys.executable, "-I", str(script), *a], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
        done = run("--records", str(self.path), "--out", str(self.dir / "s1"))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout)["designs_with_no_flags"], 1)
        again = run("--records", str(self.path), "--out", str(self.dir / "s1"))
        self.assertTrue(again.returncode != 0 and "REFUSED" in again.stderr and "Traceback" not in again.stderr)
        missing = run("--records", str(self.dir / "nope.json"), "--out", str(self.dir / "s2"))
        self.assertTrue(missing.returncode != 0 and "REFUSED" in missing.stderr and "Traceback" not in missing.stderr)


if __name__ == "__main__":
    unittest.main()
