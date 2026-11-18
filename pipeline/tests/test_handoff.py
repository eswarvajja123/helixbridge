import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign import handoff

TARGET = {"uniprot": "P16871-1", "start": 21, "end": 239, "sequence": "ESGYAQNGDLEDAELDDYSFSCYSQLEVNGSQHSLTCAFEDPDVNITNLEFEICGALVEV"}
RECS = [{"candidate_id": f"il7ra-rfd4-{c}", "sequence": s, "seed": i, "hotspot_source": "x", "novelty_review": "pending",
         "independent_prediction_status": "not_run"} for i, (c, s) in enumerate([("a", "ACDEFGHIKLMNPQRSTVWY" * 3), ("b", "MKLVINGKTL" * 6), ("c", "SSSTTT" * 10)])]


class Handoff(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)

    def test_inputs_match_the_records_and_the_target(self):
        ids = ["il7ra-rfd4-a", "il7ra-rfd4-c"]
        m = handoff.export(RECS, ids, TARGET, self.dir / "o")
        fasta = (self.dir / "o" / "alphafold-inputs.fasta").read_text().splitlines()
        self.assertEqual(fasta, [">il7ra-rfd4-a", TARGET["sequence"] + ":" + RECS[0]["sequence"], ">il7ra-rfd4-c", TARGET["sequence"] + ":" + RECS[2]["sequence"]])
        rows = list(csv.DictReader(io.StringIO((self.dir / "o" / "colabfold-batch.csv").read_text())))
        self.assertEqual([r["id"] for r in rows], ids)
        self.assertEqual(rows[1]["sequence"].split(":"), [TARGET["sequence"], RECS[2]["sequence"]])
        self.assertEqual((m["status"], m["designs"]), ("inputs_only_no_results", ids))

    def test_the_results_sheet_is_blank_and_no_review_is_set(self):
        handoff.export(RECS, ["il7ra-rfd4-b"], TARGET, self.dir / "o")
        rows = list(csv.DictReader(io.StringIO((self.dir / "o" / "alphafold-results-template.csv").read_text())))
        self.assertEqual(len(rows), 1)
        self.assertEqual({k: v for k, v in rows[0].items() if k != "candidate_id"}, {k: "" for k in handoff.SHEET_COLUMNS[1:]})
        self.assertEqual(RECS[1]["independent_prediction_status"], "not_run")
        self.assertIn("nothing here sets it", (self.dir / "o" / "README.md").read_text().lower().replace("nothing here sets it.", "nothing here sets it"))
        self.assertIn("residues 21-239", (self.dir / "o" / "README.md").read_text())

    def test_refusals(self):
        for ids in ([], ["il7ra-rfd4-a", "il7ra-rfd4-a"], ["nope"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                handoff.export(RECS, ids, TARGET, self.dir / f"x{len(ids)}{ids[:1]}")
        handoff.export(RECS, ["il7ra-rfd4-a"], TARGET, self.dir / "o")
        with self.assertRaisesRegex(ValueError, "fresh output"):
            handoff.export(RECS, ["il7ra-rfd4-a"], TARGET, self.dir / "o")
        with self.assertRaisesRegex(ValueError, "target sequence"):
            handoff.export(RECS, ["il7ra-rfd4-a"], {**TARGET, "sequence": "ACDXZ"}, self.dir / "o2")
        bad = [{**RECS[0], "sequence": "ACDE*"}]
        with self.assertRaisesRegex(ValueError, "non-canonical"):
            handoff.export(bad, ["il7ra-rfd4-a"], TARGET, self.dir / "o3")

    def test_script(self):
        script = Path(__file__).resolve().parents[1] / "scripts/export-alphafold-inputs.py"
        (self.dir / "r.json").write_text(json.dumps({"candidates": RECS}))
        (self.dir / "s.json").write_text(json.dumps({"selected": [{"candidate_id": "il7ra-rfd4-a"}, {"candidate_id": "il7ra-rfd4-b"}]}))
        (self.dir / "t.json").write_text(json.dumps(TARGET))
        run = lambda *a: subprocess.run([sys.executable, "-I", str(script), *a], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
        done = run("--records", str(self.dir / "r.json"), "--selection", str(self.dir / "s.json"), "--target-manifest", str(self.dir / "t.json"),
                   "--out", str(self.dir / "so"))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout)["designs"], 2)
        again = run("--records", str(self.dir / "r.json"), "--selection", str(self.dir / "s.json"), "--target-manifest", str(self.dir / "t.json"),
                    "--out", str(self.dir / "so"))
        self.assertTrue(again.returncode != 0 and "REFUSED" in again.stderr and "Traceback" not in again.stderr)


if __name__ == "__main__":
    unittest.main()
