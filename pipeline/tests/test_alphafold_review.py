import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _designs as D

try:
    import numpy as np
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False

RUN = unittest.skipUnless(HAVE and D.AVAILABLE, "needs numpy, Biopython and the examples")
if HAVE and D.AVAILABLE:
    from support import synth_structures as S
    from binder_campaign import alphafold_review as AR, repair, qc

MANIFEST = json.loads(D.MANIFEST.read_text()) if D.AVAILABLE else None
HOTSPOTS = D.HOTSPOTS


def scores_text(n_target, n_binder, ptm=0.8, iptm=0.7, plddt=80.0, pae=5.0, inter=12.0):
    n = n_target + n_binder
    pae_m = np.full((n, n), pae)
    pae_m[:n_target, n_target:] = inter
    pae_m[n_target:, :n_target] = inter
    return json.dumps({"ptm": ptm, "iptm": iptm, "plddt": [plddt] * n, "pae": pae_m.tolist(), "max_pae": 31.75})


@RUN
class Parsing(unittest.TestCase):
    def setUp(self):
        self.core, self.binder = S.real_sequences()
        self.cif = S.refold_structure(MANIFEST["sequence"], self.binder)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cif_path = Path(self.tmp.name) / "m.cif"
        self.cif_path.write_bytes(self.cif)
        self.pdb, _ = repair.cif_to_pdb(self.cif_path)

    def test_pdb_to_cif_round_trips_through_the_pipelines_own_reader(self):
        back = Path(self.tmp.name) / "back.cif"
        back.write_text(AR.pdb_to_cif(self.pdb))
        a, pa = qc.inspect_structure(self.cif_path, "x", MANIFEST["sequence"], self.binder, list(range(21, 240)))
        b, pb = qc.inspect_structure(back, "x", MANIFEST["sequence"], self.binder, list(range(21, 240)))
        self.assertEqual((pa, pb), ([], []))
        self.assertEqual(a["target_contacts_lt_5A"], b["target_contacts_lt_5A"])
        self.assertEqual((a["heavy_atom_pairs_lt_2A"], round(a["min_heavy_distance_A"], 2)), (b["heavy_atom_pairs_lt_2A"], round(b["min_heavy_distance_A"], 2)))

    def test_bad_pdb_is_refused(self):
        with self.assertRaises(ValueError):
            AR.parse_pdb("REMARK nothing\n")
        with self.assertRaisesRegex(ValueError, "Nonstandard"):
            AR.parse_pdb("ATOM      1  CA  XXX A   1       0.000   0.000   0.000  1.00  0.00           C\n")

    def test_only_the_first_model_is_read(self):
        two = self.pdb.replace("END\n", "") + "ENDMDL\n" + self.pdb
        self.assertEqual(len(AR.parse_pdb(two)), len(AR.parse_pdb(self.pdb)))

    def test_scores(self):
        s = AR.load_scores(scores_text(219, 60))
        self.assertEqual((s["ptm"], s["iptm"], s["pae"].shape), (0.8, 0.7, (279, 279)))
        for bad in ('{"ptm": 1}', json.dumps({"ptm": 1, "iptm": float("nan"), "plddt": [1.0], "pae": [[1.0]]}),
                    json.dumps({"ptm": 1, "iptm": 1, "plddt": [1.0, 2.0], "pae": [[1.0]]})):
            with self.subTest(bad=bad[:40]), self.assertRaises(ValueError):
                AR.load_scores(bad)

    def test_confidence_numbers(self):
        c = AR.confidence(AR.load_scores(scores_text(219, 60, plddt=77.0, inter=12.0)), 219, 60)
        self.assertEqual((c["ptm"], c["iptm"], c["plddt_binder_mean"], c["interface_pae_mean_A"], c["interface_pae_min_A"]), (0.8, 0.7, 77.0, 12.0, 12.0))
        with self.assertRaisesRegex(ValueError, "different number"):
            AR.confidence(AR.load_scores(scores_text(219, 60)), 219, 61)


@RUN
class Review(unittest.TestCase):
    def setUp(self):
        self.core, self.binder = S.real_sequences()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.record = {"candidate_id": "il7ra-rfd4-x", "sequence": self.binder, "hotspots_uniprot": HOTSPOTS,
                       "generator_audit": {"reference_audit": {"reference_sequence": self.core, "positions": list(range(37, 232))}}}
        (self.dir / "gen.cif").write_bytes(S.generator_structure(self.binder))
        (self.dir / "esm.cif").write_bytes(S.refold_structure(MANIFEST["sequence"], self.binder, shift=(0, 0, 1.0)))

    def pdb_of(self, shift):
        path = self.dir / "af.cif"
        path.write_bytes(S.refold_structure(MANIFEST["sequence"], self.binder, shift=shift))
        return repair.cif_to_pdb(path)[0]

    def test_pose_shifts_are_measured_against_each_other_structure(self):
        got = AR.review_design(self.record, self.pdb_of((0, 0, 3.0)), scores_text(219, 60), MANIFEST,
                               {"generator": self.dir / "gen.cif", "esmfold2": self.dir / "esm.cif", "boltz2": None})
        self.assertEqual(got["structure"], "checked")
        self.assertAlmostEqual(got["pose_shift_A"]["generator"], 3.0, places=2)
        self.assertAlmostEqual(got["pose_shift_A"]["esmfold2"], 2.0, places=2)
        self.assertNotIn("boltz2", got["pose_shift_A"])
        self.assertEqual((got["ptm"], got["iptm"]), (0.8, 0.7))
        self.assertTrue(set(got["hotspot_contacts"]) <= set(HOTSPOTS))

    def test_a_prediction_with_the_wrong_binder_is_reported_not_trusted(self):
        other = self.binder[:5] + ("G" if self.binder[5] != "G" else "A") + self.binder[6:]
        path = self.dir / "wrong.cif"
        path.write_bytes(S.refold_structure(MANIFEST["sequence"], other))
        got = AR.review_design(self.record, repair.cif_to_pdb(path)[0], scores_text(219, 60), MANIFEST, {})
        self.assertEqual(got["structure"], "unreadable")
        self.assertIn("binder_sequence_mismatch:alphafold2", got["problems"])

    def test_no_review_field_is_ever_produced(self):
        got = AR.review_design(self.record, self.pdb_of((0, 0, 0)), scores_text(219, 60), MANIFEST, {})
        self.assertFalse({"independent_prediction_status", "novelty_review", "submission_eligible", "interface_looks_as_designed"} & set(got))


@RUN
class Archives(unittest.TestCase):
    def make(self, folder, name, designs):
        with tarfile.open(Path(folder) / name, "w:gz") as tar:
            for design, (pdb, sc) in designs.items():
                for fname, data in ((f"{design}_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb", pdb),
                                    (f"{design}_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json", sc)):
                    info = tarfile.TarInfo(f"run/{fname}")
                    info.size = len(data.encode())
                    tar.addfile(info, io.BytesIO(data.encode()))

    def test_reads_every_chunk_and_refuses_duplicates_and_missing_scores(self):
        with tempfile.TemporaryDirectory() as t:
            self.make(t, "c1.tar.gz", {"a": ("P1", "S1")})
            self.make(t, "c2.tar.gz", {"b": ("P2", "S2")})
            self.assertEqual(AR.read_archives(t), {"a": ("P1", "S1"), "b": ("P2", "S2")})
            self.make(t, "c3.tar.gz", {"a": ("P3", "S3")})
            with self.assertRaisesRegex(ValueError, "twice"):
                AR.read_archives(t)
        with tempfile.TemporaryDirectory() as t:
            with tarfile.open(Path(t) / "c1.tar.gz", "w:gz") as tar:
                info = tarfile.TarInfo("run/a_unrelaxed_rank_001_x.pdb")
                info.size = 2
                tar.addfile(info, io.BytesIO(b"P1"))
            with self.assertRaisesRegex(ValueError, "exactly one"):
                AR.read_archives(t)


@RUN
class Script(unittest.TestCase):
    SCRIPT = Path(__file__).resolve().parents[1] / "scripts/review-alphafold.py"

    def test_end_to_end_and_missing_predictions_are_listed(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            core, binder = S.real_sequences()
            gen = t / "batch/designs/r1/model.cif"
            gen.parent.mkdir(parents=True)
            gen.write_bytes(S.generator_structure(binder))
            rec = {"candidate_id": "il7ra-rfd4-a", "sequence": binder, "hotspots_uniprot": HOTSPOTS, "generator_structure": str(gen),
                   "generator_audit": {"generated_sha256": None, "reference_audit": {"reference_sequence": core, "positions": list(range(37, 232))}}}
            rec2 = {**rec, "candidate_id": "il7ra-rfd4-b"}
            (t / "r.json").write_text(json.dumps({"candidates": [rec, rec2]}))
            (t / "s.json").write_text(json.dumps({"selected": [{"candidate_id": "il7ra-rfd4-a"}, {"candidate_id": "il7ra-rfd4-b"}]}))
            (t / "eval").mkdir()
            af = t / "af"
            af.mkdir()
            cif = t / "x.cif"
            cif.write_bytes(S.refold_structure(MANIFEST["sequence"], binder))
            Archives().make(af, "c1.tar.gz", {"il7ra-rfd4-a": (repair.cif_to_pdb(cif)[0], scores_text(219, 60))})
            run = lambda *a: subprocess.run([sys.executable, "-I", str(self.SCRIPT), *a], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
            done = run("--records", str(t / "r.json"), "--selection", str(t / "s.json"), "--evaluation", str(t / "eval"), "--batch-dir", str(t / "batch"),
                       "--alphafold", str(af), "--target-manifest", str(D.MANIFEST), "--out", str(t / "o"))
            self.assertEqual(done.returncode, 0, done.stderr)
            summary = json.loads(done.stdout)
            self.assertEqual((summary["designs"], summary["missing_predictions"]), (1, ["il7ra-rfd4-b"]))
            result = json.loads((t / "o/alphafold-results.json").read_text())["results"][0]
            self.assertAlmostEqual(result["pose_shift_A"]["generator"], 0.0, places=2)
            again = run("--records", str(t / "r.json"), "--selection", str(t / "s.json"), "--evaluation", str(t / "eval"), "--alphafold", str(af), "--out", str(t / "o"))
            self.assertTrue(again.returncode != 0 and "REFUSED" in again.stderr and "Traceback" not in again.stderr)


if __name__ == "__main__":
    unittest.main()
