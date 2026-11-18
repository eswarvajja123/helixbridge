import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import modal  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "modal_alphafold.py"


class Static(unittest.TestCase):
    def test_settings_and_pins(self):
        s = MODULE.read_text()
        for fragment in ("COLABFOLD = \"1.6.3\"", "alphafold2_multimer_v3", "\"num_models\": 2", "\"num_recycle\": 3", "\"templates\": False", "\"relax\": False",
                         "--num-relax", "download_alphafold_params", "a3mpath", "no MSA server",
                         "'jax[cuda12]>=0.6.2,<0.7'", "timeout=3300"):
            self.assertIn(fragment, s, fragment)

    def test_no_public_msa_server_is_requested(self):
        s = MODULE.read_text()
        self.assertNotIn("--msa-mode", s)                       # a custom a3m per design is given instead, so no server is asked
        self.assertIn("contacted the MSA server", s)            # and a run that did contact it is rejected

    def test_it_never_sets_a_review_field(self):
        s = MODULE.read_text()
        for field in ("independent_prediction_status", "novelty_review", "submission_eligible"):
            self.assertNotIn(field, s)


@unittest.skipUnless(HAVE and MODULE.is_file(), "needs modal")
class Entrypoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.saved = (list(sys.path), dict(os.environ), {k: v for k, v in sys.modules.items() if k == "binder_campaign" or k.startswith("binder_campaign.")})
        self.addCleanup(self.restore)
        spec = importlib.util.spec_from_file_location("modal_alphafold_under_test", MODULE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.csv = self.dir / "batch.csv"
        with self.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sequence"])
            for i in range(1, 8):
                w.writerow([f"d{i}", "ACDEFGHIKL:MNPQRSTVWY"])

    def restore(self):
        sys.path[:] = self.saved[0]
        os.environ.clear()
        os.environ.update(self.saved[1])
        for name in [n for n in sys.modules if n == "binder_campaign" or n.startswith("binder_campaign.")]:
            sys.modules.pop(name)
        sys.modules.update(self.saved[2])

    def fake(self, fail=()):
        calls = []
        def remote(run_id, rows, target_a3m):
            calls.append([r["id"] for r in rows])
            if any(r["id"] in fail for r in rows):
                raise RuntimeError("container died")
            return {"run_id": run_id, "ids": [r["id"] for r in rows], "settings": self.module.SETTINGS, "command": ["x"], "device": "Fake",
                    "seconds": 1.0, "archive": b"tar"}
        self.module.predict_chunk = types.SimpleNamespace(remote=remote)
        return calls

    def run_main(self, **kw):
        self.module.main.info.raw_f(stage="predict", batch=str(self.csv), out=str(self.dir / "out"), **kw)

    def test_chunks_of_four_and_a_rerun_skips_finished_chunks(self):
        calls = self.fake()
        self.run_main()
        self.assertEqual(calls, [["d1", "d2", "d3", "d4"], ["d5", "d6", "d7"]])
        state = json.loads((self.dir / "out/run.json").read_text())
        self.assertEqual({k: v["status"] for k, v in state["chunks"].items()}, {"c1": "done", "c2": "done"})
        self.assertEqual((self.dir / "out/c1.tar.gz").read_bytes(), b"tar")
        calls.clear()
        self.run_main()
        self.assertEqual(calls, [])

    def test_a_failed_chunk_is_kept_on_record_and_retried_on_rerun_while_others_still_run(self):
        calls = self.fake(fail={"d2"})
        self.run_main()
        state = json.loads((self.dir / "out/run.json").read_text())
        self.assertEqual({k: v["status"] for k, v in state["chunks"].items()}, {"c1": "failed", "c2": "done"})
        self.assertIn("container died", state["chunks"]["c1"]["error"])
        calls = self.fake()
        self.run_main()
        self.assertEqual(calls, [["d1", "d2", "d3", "d4"]])
        state = json.loads((self.dir / "out/run.json").read_text())
        self.assertEqual(state["chunks"]["c1"]["status"], "done")
        self.assertEqual(len(state["chunks"]["c1"]["attempts"]), 2)                # both attempts stay on record

    def test_bad_requests_are_refused(self):
        self.fake()
        for kw in ({"stage": "oops"},):
            with self.assertRaises(ValueError):
                self.module.main.info.raw_f(batch=str(self.csv), out=str(self.dir / "o1"), **kw)
        with self.assertRaises(ValueError):
            self.module.main.info.raw_f(stage="predict", batch=str(self.csv), out="")
        (self.dir / "empty.csv").write_text("id,sequence\n")
        with self.assertRaises(ValueError):
            self.module.main.info.raw_f(stage="predict", batch=str(self.dir / "empty.csv"), out=str(self.dir / "o2"))

    def test_changed_settings_are_refused_on_resume(self):
        self.fake()
        self.run_main()
        state = json.loads((self.dir / "out/run.json").read_text())
        state["settings"]["num_models"] = 5
        (self.dir / "out/run.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "settings changed"):
            self.run_main()


if __name__ == "__main__":
    unittest.main()
