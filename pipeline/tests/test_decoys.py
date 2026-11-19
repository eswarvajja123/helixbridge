import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign import decoys as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def scores_text(iptm):
    n = 4
    return json.dumps({"ptm": 0.5, "iptm": iptm, "plddt": [80.0] * n, "pae": [[1.0] * n for _ in range(n)]})


def write_archive(folder, name, ids_iptm):
    folder.mkdir(parents=True, exist_ok=True)
    with tarfile.open(folder / name, "w:gz") as tar:
        for design, iptm in ids_iptm.items():
            for fname, text in ((f"{design}_unrelaxed_rank_001_x.pdb", "ATOM\n"), (f"{design}_scores_rank_001_x.json", scores_text(iptm))):
                data = text.encode()
                info = tarfile.TarInfo(f"run/{fname}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))


class Decoys(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        write_archive(self.dir / "run", "c1.tar.gz", {"gfp-aaa": 0.2, "lyz-aaa": 0.7, "gfp-bbb": 0.1, "lyz-bbb": 0.3})

    def test_reads_each_design_for_each_decoy(self):
        found = D.read_decoy_iptm(self.dir / "run", ["gfp", "lyz"])
        self.assertEqual(found, {"il7ra-rfd4-aaa": {"gfp": 0.2, "lyz": 0.7}, "il7ra-rfd4-bbb": {"gfp": 0.1, "lyz": 0.3}})

    def test_summary_names_the_strongest_decoy(self):
        top = D.summarize(D.read_decoy_iptm(self.dir / "run", ["gfp", "lyz"]))
        self.assertEqual((top["il7ra-rfd4-aaa"]["max_iptm"], top["il7ra-rfd4-aaa"]["max_decoy"]), (0.7, "lyz"))

    def test_gate_reasons(self):
        top = D.summarize(D.read_decoy_iptm(self.dir / "run", ["gfp", "lyz"]))
        self.assertEqual(D.gate_reasons(top["il7ra-rfd4-aaa"], ["gfp", "lyz"]), ["decoy_binding:lyz:0.7>0.61"])
        self.assertEqual(D.gate_reasons(top["il7ra-rfd4-bbb"], ["gfp", "lyz"]), [])
        self.assertEqual(D.gate_reasons(None, ["gfp"]), ["decoy_missing"])
        self.assertEqual(D.gate_reasons(top["il7ra-rfd4-bbb"], ["gfp", "lyz", "other"]), ["decoy_missing:other"])
        self.assertEqual(D.gate_reasons({"iptm": {"gfp": 0.61}}, ["gfp"]), [])                         # the limit is inclusive: ipTM must exceed it

    def test_refusals(self):
        with self.assertRaises(ValueError):
            D.read_decoy_iptm(self.dir / "run", [])
        with self.assertRaises(ValueError):
            D.read_decoy_iptm(self.dir / "run", ["gfp", "gfp"])
        with self.assertRaisesRegex(ValueError, "exactly one decoy"):
            D.read_decoy_iptm(self.dir / "run", ["gfp"])                                              # lyz-* belongs to no named decoy
        write_archive(self.dir / "dup", "c1.tar.gz", {"gfp-aaa": 0.2})
        with self.assertRaisesRegex(ValueError, "twice"):
            D.read_decoy_iptm([self.dir / "run", self.dir / "dup"], ["gfp", "lyz"])

    def test_the_script_writes_a_summary_and_refuses_to_overwrite(self):
        out = self.dir / "o" / "decoys.json"
        done = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/review-decoys.py"), "--alphafold", str(self.dir / "run"), "--decoys", "gfp,lyz",
                               "--out", str(out)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        doc = json.loads(out.read_text())
        self.assertEqual(doc["decoys"], ["gfp", "lyz"])
        self.assertEqual(doc["designs"]["il7ra-rfd4-aaa"]["max_decoy"], "lyz")
        again = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/review-decoys.py"), "--alphafold", str(self.dir / "run"), "--decoys", "gfp,lyz",
                                "--out", str(out)], capture_output=True, text=True)
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("REFUSED", again.stderr)


if __name__ == "__main__":
    unittest.main()
