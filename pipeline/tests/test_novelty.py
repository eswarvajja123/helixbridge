import csv
import gzip
import io
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign import novelty, public_db

MMSEQS = shutil.which("mmseqs")
AA = "ACDEFGHIKLMNPQRSTVWY"


def seq(seed, n=80):
    rng = random.Random(seed)
    return "".join(rng.choice(AA) for _ in range(n))


def mutate(s, fraction, seed=0):
    rng = random.Random(seed)
    out = list(s)
    for i in rng.sample(range(len(s)), round(len(s) * fraction)):
        out[i] = rng.choice([a for a in AA if a != out[i]])
    return "".join(out)


def hit(query, target, fident, qcov, tcov, bits=100.0, alnlen=60, qlen=60, tlen=60, evalue=1e-10):
    return "\t".join(map(str, (query, target, fident, alnlen, qlen, tlen, qcov, tcov, evalue, bits)))


def fake_search(lines):
    def run(queries, database, out_path):
        Path(out_path).write_text("\n".join(lines) + ("\n" if lines else ""))
    return run


def record(cid, sequence, reasons=()):
    return {"candidate_id": cid, "sequence": sequence, "qc_reasons": list(reasons), "novelty_review": "pending",
            "submission_eligible": False, "screen_status": "not_run", "independent_prediction_status": "not_run"}


class Parsing(unittest.TestCase):
    def table(self, lines):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "h.tsv"
            p.write_text("\n".join(lines) + "\n")
            return novelty.parse_hits(p)

    def test_reads_a_good_table(self):
        rows = self.table([hit("q1", "sp|P1|X", 0.91, 1.0, 0.5), "", hit("q2", "pdb|1abc_A", 0.4, 0.9, 0.9)])
        self.assertEqual([r["query"] for r in rows], ["q1", "q2"])
        self.assertEqual((rows[0]["fident"], rows[0]["alnlen"]), (0.91, 60))

    def test_bad_tables_are_refused(self):
        for bad in (["q1\tonly\tthree"], [hit("q", "t", "abc", 1, 1)], [hit("q", "t", 1.5, 1, 1)], [hit("q", "t", 0.5, 1.2, 1)],
                    [hit("q", "t", 0.5, 1, -0.1)]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.table(bad)


class Judging(unittest.TestCase):
    def judged(self, *lines):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "h.tsv"
            p.write_text("\n".join(lines) + "\n")
            return novelty.judge(novelty.parse_hits(p))

    def test_identity_must_be_strictly_above_70_and_coverage_at_least_80(self):
        cases = ((0.70, 0.80, False), (0.7001, 0.80, True), (0.71, 0.7999, False), (0.71, 0.80, True), (1.0, 1.0, True), (0.3, 1.0, False))
        for fident, qcov, expect in cases:
            with self.subTest(fident=fident, qcov=qcov):
                j = self.judged(hit("q", "t", fident, qcov, 0.5))["q"]
                self.assertEqual(j["tripping_hits"] > 0, expect)

    def test_both_coverages_is_the_stricter_reading(self):
        self.assertFalse(self.judged(hit("q", "t", 0.9, 1.0, 0.3))["q"]["both_coverages_ok"])
        self.assertTrue(self.judged(hit("q", "t", 0.9, 1.0, 0.8))["q"]["both_coverages_ok"])

    def test_best_hit_is_by_bit_score_and_the_strongest_tripping_hit_by_identity(self):
        j = self.judged(hit("q", "weak_but_long", 0.55, 1.0, 1.0, bits=300), hit("q", "close", 0.95, 0.9, 0.2, bits=120),
                        hit("q", "closer", 0.99, 0.85, 0.1, bits=110))["q"]
        self.assertEqual(j["best_hit"]["target"], "weak_but_long")
        self.assertEqual(j["strongest_tripping_hit"]["target"], "closer")
        self.assertEqual((j["hits"], j["tripping_hits"]), (3, 2))


class Prescreen(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.records = [record("a", seq(1)), record("b", seq(2)), record("c", seq(3), ["near_duplicate:backbone_tm"]),
                        record("d", seq(4))]
        self.path = self.dir / "candidates.json"
        self.path.write_text(json.dumps({"candidates": self.records, "count": 4, "note": "x"}))

    def go(self, lines, out="out", **kw):
        return novelty.run_prescreen(self.path, self.dir / "db", self.dir / out, fake_search(lines), log=lambda *_: None, **kw)

    def result(self, out="out"):
        return {r["candidate_id"]: r for r in json.loads((self.dir / out / "novelty-candidates.json").read_text())["candidates"]}

    def test_statuses(self):
        report = self.go([hit("a", "sp|P16871|IL7RA_HUMAN", 0.95, 1.0, 0.3), hit("b", "sp|Q1|OTHER", 0.5, 0.9, 0.9)])
        r = self.result()
        self.assertEqual([r[k]["novelty_prescreen"]["status"] for k in "abcd"], ["flagged", "no_tripping_hit", "not_searched", "no_public_hit"])
        self.assertEqual(report["status_counts"], {"flagged": 1, "no_tripping_hit": 1, "not_searched": 1, "no_public_hit": 1})

    def test_a_flag_adds_a_reason_that_blocks_export_but_never_certifies_anything(self):
        self.go([hit("a", "sp|P16871|IL7RA_HUMAN", 0.95, 1.0, 0.3)])
        r = self.result()
        self.assertEqual(r["a"]["qc_reasons"], ["public_similarity:sp|P16871|IL7RA_HUMAN:0.95id_1.00cov"])
        self.assertEqual(r["c"]["qc_reasons"], ["near_duplicate:backbone_tm"])                 # earlier reasons are kept, not searched
        for item in r.values():
            self.assertEqual((item["novelty_review"], item["submission_eligible"]), ("pending", False))
            self.assertIn("Cannot certify novelty", item["novelty_prescreen"].get("note", "Cannot certify novelty"))

    def test_designs_with_qc_reasons_are_not_searched_unless_asked(self):
        seen = {}
        def spy(queries, database, out):
            seen["ids"] = [l[1:] for l in Path(queries).read_text().splitlines() if l.startswith(">")]
            Path(out).write_text("")
        novelty.run_prescreen(self.path, self.dir / "db", self.dir / "o1", spy, log=lambda *_: None)
        self.assertEqual(seen["ids"], ["a", "b", "d"])
        novelty.run_prescreen(self.path, self.dir / "db", self.dir / "o2", spy, include_flagged=True, log=lambda *_: None)
        self.assertEqual(seen["ids"], ["a", "b", "c", "d"])

    def test_inputs_are_untouched_and_outputs_exist(self):
        before = self.path.read_bytes()
        self.go([hit("a", "t", 0.9, 1.0, 1.0)])
        self.assertEqual(self.path.read_bytes(), before)
        rows = list(csv.DictReader(io.StringIO((self.dir / "out" / "novelty.csv").read_text())))
        self.assertEqual([r["candidate_id"] for r in rows], list("abcd"))
        self.assertEqual(rows[0]["status"], "flagged")
        with self.assertRaisesRegex(ValueError, "fresh output"):
            self.go([])

    def test_a_failing_search_is_an_error_not_a_silent_pass(self):
        def broken(*a):
            raise RuntimeError("MMseqs2 failed: boom")
        with self.assertRaisesRegex(RuntimeError, "boom"):
            novelty.run_prescreen(self.path, self.dir / "db", self.dir / "o3", broken, log=lambda *_: None)
        self.assertFalse((self.dir / "o3").exists())


class PublicDbCombine(unittest.TestCase):
    def test_combine_keeps_swissprot_whole_and_pdb_proteins_once(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            with gzip.open(t / "sp.gz", "wt") as f:
                f.write(">sp|P1|A_HUMAN x\nMKTAYIAKQR\nQISFVKSHFS\n>sp|P2|B_HUMAN y\nMKLVINGKTL\n")
            long = seq(9, 30)
            with gzip.open(t / "pdb.gz", "wt") as f:
                f.write(f">1abc_A mol:protein length:30  X\n{long}\n>1abc_B mol:protein length:30  X\n{long}\n"
                        f">1abc_C mol:na length:12  DNA\nACGTACGTACGT\n>2xyz_A mol:protein length:5  SHORT\nMKLVI\n"
                        f">3pqr_A mol:protein length:30  Y\n{seq(10, 29)}J\n")
            counts = public_db.combine(t / "sp.gz", t / "pdb.gz", t / "out.fasta")
            self.assertEqual(counts, {"swissprot": 2, "pdb_protein_chains": 2, "pdb_unique_sequences": 1})
            text = (t / "out.fasta").read_text()
            self.assertIn(">sp|P1|A_HUMAN x\nMKTAYIAKQR\nQISFVKSHFS", text)
            self.assertIn(">pdb|1abc_A\n" + long, text)
            self.assertNotIn("1abc_B", text)          # identical sequence kept once
            self.assertNotIn("1abc_C", text)          # nucleic acid left out
            self.assertNotIn("2xyz_A", text)          # shorter than 20 residues

    def test_an_empty_download_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            with gzip.open(t / "sp.gz", "wt") as f:
                f.write("")
            with gzip.open(t / "pdb.gz", "wt") as f:
                f.write(">1abc_A mol:protein length:3  X\nMKL\n")
            with self.assertRaisesRegex(ValueError, "empty database"):
                public_db.combine(t / "sp.gz", t / "pdb.gz", t / "o.fasta")


@unittest.skipUnless(MMSEQS, "needs the mmseqs program")
class RealMmseqs(unittest.TestCase):
    """The real MMseqs2 on a tiny database that this test builds itself."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.temp.name)
        cls.target = {f"sp|T{i}|DB": seq(100 + i, 90) for i in range(30)}
        (cls.dir / "db.fasta").write_text("".join(f">{k}\n{v}\n" for k, v in cls.target.items()))
        done = subprocess.run([MMSEQS, "createdb", str(cls.dir / "db.fasta"), str(cls.dir / "db"), "-v", "1"], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def screen(self, recs):
        return novelty.prescreen(recs, self.dir / "db", novelty.run_mmseqs)

    def test_known_relationships(self):
        original = self.target["sp|T3|DB"][:70]
        res = self.screen([record("exact", original), record("m10", mutate(original, 0.10, 1)), record("m20", mutate(original, 0.20, 2)),
                           record("m45", mutate(original, 0.45, 3)), record("unrelated", seq(999, 70))])
        self.assertEqual(res["exact"]["status"], "flagged")
        self.assertAlmostEqual(res["exact"]["best_hit"]["identity"], 1.0, places=2)
        self.assertEqual(res["exact"]["best_hit"]["target"], "T3")                  # MMseqs2 reports the accession part of the header
        self.assertAlmostEqual(res["exact"]["best_hit"]["query_coverage"], 1.0, places=2)
        self.assertLess(res["exact"]["best_hit"]["target_coverage"], 0.8)             # a fragment of a longer protein
        self.assertFalse(res["exact"]["both_coverages_ok"])
        self.assertEqual(res["m10"]["status"], "flagged")
        self.assertAlmostEqual(res["m10"]["best_hit"]["identity"], 0.9, delta=0.03)
        self.assertEqual(res["m20"]["status"], "flagged")
        self.assertNotEqual(res["m45"]["status"], "flagged")                           # identity about 0.55: below the rule
        self.assertEqual(res["unrelated"]["status"], "no_public_hit")

    def test_a_missing_database_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "MMseqs2 failed"):
            novelty.prescreen([record("x", seq(5, 60))], self.dir / "nope", novelty.run_mmseqs)

    def test_the_database_builder_end_to_end_on_tiny_inputs(self):
        def fetch(url, path):
            with gzip.open(path, "wt") as f:
                if "uniprot" in url:
                    f.write("".join(f">sp|S{i}|X\n{seq(200 + i, 60)}\n" for i in range(5)))
                else:
                    f.write("".join(f">{i}abc_A mol:protein length:60  P\n{seq(300 + i, 60)}\n" for i in range(5)))
        out = self.dir / "built"
        manifest = public_db.build(out, MMSEQS, fetch=fetch, log=lambda *_: None)
        self.assertEqual((manifest["swissprot"], manifest["pdb_unique_sequences"]), (5, 5))
        self.assertEqual(len(manifest["sources"][0]["sha256"]), 64)
        self.assertTrue((out / "publicDB.dbtype").is_file() and (out / "public-db-manifest.json").is_file())
        res = novelty.prescreen([record("hit", seq(302, 60)), record("miss", seq(77, 60))], out / "publicDB", novelty.run_mmseqs)
        self.assertEqual((res["hit"]["status"], res["miss"]["status"]), ("flagged", "no_public_hit"))
        with self.assertRaisesRegex(ValueError, "fresh output"):
            public_db.build(out, MMSEQS, fetch=fetch, log=lambda *_: None)


class Script(unittest.TestCase):
    SCRIPT = Path(__file__).resolve().parents[1] / "scripts/novelty-prescreen.py"

    @unittest.skipUnless(MMSEQS, "needs the mmseqs program")
    def test_end_to_end_with_the_real_program(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            known = seq(42, 80)
            (t / "db.fasta").write_text(f">sp|KNOWN|X\n{known}\n")
            subprocess.run([MMSEQS, "createdb", str(t / "db.fasta"), str(t / "publicDB"), "-v", "1"], check=True, capture_output=True)
            (t / "c.json").write_text(json.dumps({"candidates": [record("same", known[:60]), record("other", seq(43, 60))]}))
            run = lambda *a: subprocess.run([sys.executable, "-I", str(self.SCRIPT), *a], capture_output=True, text=True,
                                            env={"PATH": "/usr/bin:/bin"})
            done = run("--records", str(t / "c.json"), "--database", str(t / "publicDB"), "--out", str(t / "o"))
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(json.loads(done.stdout)["status_counts"], {"flagged": 1, "no_public_hit": 1})
            again = run("--records", str(t / "c.json"), "--database", str(t / "publicDB"), "--out", str(t / "o"))
            self.assertTrue(again.returncode != 0 and "REFUSED" in again.stderr and "Traceback" not in again.stderr)
            nodb = run("--records", str(t / "c.json"), "--database", str(t / "missing"), "--out", str(t / "o2"))
            self.assertTrue(nodb.returncode != 0 and "No MMseqs2 database" in nodb.stderr and "Traceback" not in nodb.stderr)


if __name__ == "__main__":
    unittest.main()
