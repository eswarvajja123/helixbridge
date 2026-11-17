"""Run the REAL esm ESMFold2 code (tiny random-weight model, CPU) and check OUR code against its real output.

Skipped unless HELIXBRIDGE_ESM_PYTHON names a Python that has the real `esm==3.4.1.post1` and `torch==2.11.0` installed
(for example an environment made with `uv pip install torch==2.11.0 esm==3.4.1.post1`). Takes about two minutes. Set
HELIXBRIDGE_IPSAE as well to also run the real pinned ipSAE scorer on the real CIF. This does not test model quality:
the weights are random. It tests that our assumptions about the real library's inputs and outputs are true.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.esm_eval import confidence_summary, validate_prediction
from binder_campaign.scoring import build_score, run_ipsae

ROOT = Path(__file__).resolve().parents[2]
PYTHON = os.environ.get("HELIXBRIDGE_ESM_PYTHON", "")
IPSAE = os.environ.get("HELIXBRIDGE_IPSAE", "")
SCRIPT = Path(__file__).resolve().parent / "support/tiny_esmfold2_fold.py"
CONFIG = Path(__file__).resolve().parent / "fixtures/esmfold2_config.json"
PHASE1 = ROOT / "docs/eswar/phase1-data"
SUMMARY = ROOT / "examples/verified-modal-run/summary.json"
MANIFEST = ROOT / "docs/eswar/esmfold2-checkpoint-manifest.json"
READY = PYTHON and Path(PYTHON).is_file() and CONFIG.is_file() and PHASE1.is_dir() and SUMMARY.is_file()


@unittest.skipUnless(READY, "set HELIXBRIDGE_ESM_PYTHON to a Python with the real esm and torch (see the module docstring)")
class RealLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.temp.name)
        cls.target = json.loads((PHASE1 / "il7ra-manifest.json").read_text())["sequence"]
        cls.binder = json.loads(SUMMARY.read_text())["binder_sequence"]
        done = subprocess.run([PYTHON, str(SCRIPT), "--config", str(CONFIG), "--target-manifest", str(PHASE1 / "il7ra-manifest.json"),
                               "--msa", str(PHASE1 / "target.a3m"), "--binder", cls.binder, "--out", str(cls.out)],
                              capture_output=True, text=True, timeout=900)
        if done.returncode != 0:
            raise AssertionError("the real fold did not run:\n" + done.stderr[-1500:])
        cls.result = json.loads((cls.out / "result.json").read_text())
        cls.cif = (cls.out / "model.cif").read_text()
        cls.pae = json.loads((cls.out / "pae.json").read_text())["pae"]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_the_pinned_versions_are_the_ones_that_ran(self):
        v = self.result["versions"]
        self.assertEqual((v["torch"], v["esm"]), ("2.11.0", "3.4.1.post1"))

    def test_the_real_result_has_the_fields_and_shapes_our_code_reads(self):
        n = len(self.target) + len(self.binder)
        self.assertEqual((self.result["pae_shape"], self.result["plddt_shape"]), ([n, n], [n]))
        self.assertEqual(self.result["msa_depth"], 1022)                       # our real MSA, all rows, loaded by the real parser
        for key in ("ptm", "iptm"):
            self.assertTrue(0.0 <= self.result[key] <= 1.0)
        self.assertTrue(0.0 <= self.result["plddt_min"] <= self.result["plddt_max"] <= 1.0)   # pLDDT is on a 0-1 scale here

    def test_our_validator_accepts_the_real_exporters_structure(self):
        checks = validate_prediction(self.pae, self.cif, self.target, self.binder)
        self.assertEqual(checks["chains"], [{"id": "A", "length": len(self.target)}, {"id": "B", "length": len(self.binder)}])

    def test_the_teams_pose_review_and_our_qc_read_the_real_exporters_structure(self):
        """QC (Task 3.3) reads the real exporter's mmCIF with the team's pose_review: numbering, chains and atoms are as assumed."""
        from binder_campaign import qc
        path = self.out / "model.cif"
        positions = list(range(21, 240))
        pose, problems = qc.inspect_structure(path, "esmfold2", self.target, self.binder, positions)
        self.assertEqual(problems, [])
        self.assertEqual((len(pose["target_ca"]), len(pose["binder_ca"])), (len(self.target), len(self.binder)))
        self.assertTrue(all(isinstance(c, int) and 21 <= c <= 239 for c in pose["target_contacts_lt_5A"]))
        self.assertTrue(np.isfinite(pose["min_heavy_distance_A"]))

    def test_our_confidence_summary_accepts_the_real_values(self):
        summary = confidence_summary(np.full(10, self.result["plddt_mean"]), self.result["ptm"], self.result["iptm"])
        self.assertAlmostEqual(summary["iptm"], self.result["iptm"])

    def test_the_real_structure_is_full_atom_with_cb_which_ipsae_needs(self):
        from biotite.structure.io.pdbx import CIFFile, get_structure
        import io
        atoms = get_structure(CIFFile.read(io.StringIO(self.cif)), model=1, use_author_fields=False)
        self.assertIn("CB", set(atoms.atom_name.tolist()))
        self.assertGreater(len(atoms) / int((atoms.atom_name == "CA").sum()), 5)       # side chains are present

    @unittest.skipUnless(IPSAE and Path(IPSAE).is_file(), "set HELIXBRIDGE_IPSAE to also run the real scorer")
    def test_the_real_scorer_runs_on_the_real_structure_and_our_score_record_builds(self):
        work = self.out / "ipsae"
        work.mkdir()
        (work / "model.cif").write_text(self.cif)
        (work / "pae.json").write_text(json.dumps({"pae": self.pae}))
        rows = run_ipsae(IPSAE, work / "pae.json", work / "model.cif")
        score = build_score("esmfold2", iptm=self.result["iptm"], ptm=self.result["ptm"], rows=rows, seconds=1.0,
                            peak_reserved_mib=1.0, plddt_raw_mean=self.result["plddt_mean"])
        self.assertEqual(score["ipsae"]["pae_cutoff"], 10)


if __name__ == "__main__":
    unittest.main()
