import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _designs as D

try:
    import modal  # noqa: F401
    import numpy as np  # noqa: F401
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "modal_repair.py"
RUN = unittest.skipUnless(HAVE and D.AVAILABLE and MODULE.is_file(), "needs modal, numpy, Biopython, the examples and modal_repair.py")
if HAVE and D.AVAILABLE:
    from support import synth_structures as S


class StaticChecks(unittest.TestCase):
    def test_pins_and_safety_steps(self):
        source = MODULE.read_text()
        for fragment in ("26ec57ac976ade5379920dbd43c7f97a91cf82de", "7af52d090172c230c7f0e9d21e02203f6b3a38b16db58d3c7a3960e0a9a6e31a",
                         "sha256sum -c -", "weights_only=True", "files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt", "'torch==2.2.1'",
                         "'numpy==1.23.5'", "'ProDy==2.4.1'", '"/opt/pipeline"'):
            self.assertIn(fragment, source, fragment)
        self.assertNotIn("subprocess.run([sys.executable, str(MPNN_CODE", source)          # the upstream runs only through the tested wrapper
        self.assertIn("repair.run_native(", source)

    def test_the_team_protocol_settings_are_the_wrappers_defaults(self):
        from binder_campaign import repair
        self.assertEqual((repair.TEMPERATURE, repair.SEED, repair.SAMPLES, repair.MAX_VARIANTS), (0.1, 401, 16, 2))
        lead = ROOT / "outputs/lead101-repair-protocol-2026-11-02.md"
        if lead.is_file():
            text = lead.read_text()
            for fragment in ("temperature0.1", "seed401", "16 samples"):
                self.assertIn(fragment, text)


@RUN
class BodyLocally(unittest.TestCase):
    STUB = (
        "import argparse, json, os, sys\n"
        "p = argparse.ArgumentParser()\n"
        "for k in ['model_type','checkpoint_soluble_mpnn','pdb_path','out_folder','redesigned_residues','omit_AA_per_residue','temperature','seed',"
        "'batch_size','number_of_batches','save_stats','pack_side_chains','device']:\n"
        "    p.add_argument('--' + k)\n"
        "a = p.parse_args()\n"
        "os.makedirs(a.out_folder + '/seqs', exist_ok=True)\n"
        "json.dump(vars(a), open(a.out_folder + '/args.json', 'w'))\n"
        "open(a.out_folder + '/seqs/design.fa', 'w').write('>design, T=0.1\\nAAA:CCC\\n>design, id=0, T=0.1, seed=401, overall_confidence=0.5\\nAAA:CLC\\n')\n")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        (self.dir / "code").mkdir()
        (self.dir / "code/run.py").write_text(self.STUB)
        (self.dir / "w.pt").write_text("w")
        opt = self.dir / "opt"
        shutil.copytree(ROOT / "pipeline", opt / "pipeline", ignore=shutil.ignore_patterns("tests", "__pycache__", "data", "*.pyc"))
        self.saved = (list(sys.path), dict(os.environ), {k: v for k, v in sys.modules.items() if k == "binder_campaign" or k.startswith("binder_campaign.")})
        os.environ.update(HELIXBRIDGE_OPT=str(opt), HELIXBRIDGE_MPNN_CODE=str(self.dir / "code"), HELIXBRIDGE_MPNN_WEIGHTS=str(self.dir / "w.pt"))
        self.addCleanup(self.restore)
        spec = importlib.util.spec_from_file_location("modal_repair_under_test", MODULE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def restore(self):
        sys.path[:] = self.saved[0]
        os.environ.clear()
        os.environ.update(self.saved[1])
        for name in [n for n in sys.modules if n == "binder_campaign" or n.startswith("binder_campaign.")]:
            sys.modules.pop(name)
        sys.modules.update(self.saved[2])

    def test_one_design(self):
        result = self.module.native_repair.local("il7ra-rfd4-x", "END\n", {"28": "C", "30": "N"}, "B")
        self.assertEqual(result["parent"], "il7ra-rfd4-x")
        self.assertIn("overall_confidence=0.5", result["fasta"])
        self.assertEqual(result["revision"], "26ec57ac976ade5379920dbd43c7f97a91cf82de")
        self.assertIn("--redesigned_residues", result["command"])
        self.assertEqual(result["command"][result["command"].index("--redesigned_residues") + 1], "B28 B30")
        self.assertEqual(result["command"][result["command"].index("--seed") + 1], "401")

    def test_temperature_and_seed_are_passed_to_the_upstream_command(self):
        result = self.module.native_repair.local("x", "END\n", {"28": "C"}, "B", 0.3, 402)
        cmd = result["command"]
        self.assertEqual((cmd[cmd.index("--temperature") + 1], cmd[cmd.index("--seed") + 1]), ("0.3", "402"))

    def test_the_entrypoint_sends_the_chosen_temperature_and_seed(self):
        core, binder = S.real_sequences()
        sequence = binder[:9] + "C" + binder[10:]
        gen = self.dir / "gen3.cif"
        gen.write_bytes(S.generator_structure(sequence))
        records = self.dir / "r3.json"
        records.write_text(json.dumps({"candidates": [{"candidate_id": "a", "sequence": sequence, "generator_structure": str(gen), "generator_binder_chain": "B"}]}))
        seen = []
        def fake_remote(parent_id, pdb_text, edits, binder_chain, temperature=0.1, seed=401):
            seen.append((temperature, seed))
            return {"parent": parent_id, "fasta_name": "design.fa", "fasta": ">x\nAAA:CCC\n", "command": ["run.py"], "seconds": 1.0}
        self.module.native_repair = types.SimpleNamespace(remote=fake_remote)
        self.module.main.info.raw_f(records=str(records), out=str(self.dir / "out3"), temperature=0.3, seed=402)
        self.assertEqual(seen, [(0.3, 402)])
        doc = json.loads((self.dir / "out3" / "native-run.json").read_text())
        self.assertEqual((doc["temperature"], doc["seed"]), (0.3, 402))

    def test_bad_requests_are_refused_before_anything_runs(self):
        for bad in ({"temperature": 0}, {"temperature": 1.5}, {"temperature": True}, {"seed": -1}, {"seed": 1.5}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.module.native_repair.local("x", "END\n", {"5": "C"}, "B", **bad)
        for edits in ({}, {"5": ""}, {"5": "Z"}, {"5": "c"}):
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                self.module.native_repair.local("x", "END\n", edits, "B")

    def test_the_entrypoint_writes_native_outputs_where_ingest_expects_them(self):
        core, binder = S.real_sequences()
        sequence = binder[:9] + "C" + binder[10:]
        gen = self.dir / "gen.cif"
        gen.write_bytes(S.generator_structure(sequence))
        records = self.dir / "r.json"
        records.write_text(json.dumps({"candidates": [
            {"candidate_id": "a", "sequence": sequence, "generator_structure": str(gen), "generator_binder_chain": "B"},
            {"candidate_id": "b", "sequence": binder[:9] + "L" + binder[10:], "generator_structure": str(gen), "generator_binder_chain": "B"}]}))
        calls = []
        def fake_remote(parent_id, pdb_text, edits, binder_chain, temperature=0.1, seed=401):
            calls.append((parent_id, edits, binder_chain, pdb_text.count("ATOM")))
            return {"parent": parent_id, "fasta_name": "design.fa", "fasta": ">x\nAAA:CCC\n", "command": ["run.py"], "seconds": 1.0}
        self.module.native_repair = types.SimpleNamespace(remote=fake_remote)
        out = self.dir / "out"
        self.module.main.info.raw_f(records=str(records), out=str(out))
        self.assertEqual([(c[0], c[1], c[2]) for c in calls], [("a", {"10": "C"}, "B")])          # only the design with a liability is sent
        self.assertGreater(calls[0][3], 100)                                                       # a real backbone was sent
        self.assertTrue((out / "native/a/seqs/design.fa").is_file())
        self.assertEqual(json.loads((out / "native-run.json").read_text())["designs"][0]["status"], "ran")
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.module.main.info.raw_f(records=str(records), out=str(out))

    def test_one_failed_design_does_not_hide_the_others(self):
        core, binder = S.real_sequences()
        gen = self.dir / "gen.cif"
        gen.write_bytes(S.generator_structure(binder[:9] + "C" + binder[10:]))
        gen2 = self.dir / "gen2.cif"
        gen2.write_bytes(S.generator_structure(binder[:20] + "C" + binder[21:]))
        records = self.dir / "r.json"
        records.write_text(json.dumps({"candidates": [
            {"candidate_id": "a", "sequence": binder[:9] + "C" + binder[10:], "generator_structure": str(gen), "generator_binder_chain": "B"},
            {"candidate_id": "b", "sequence": binder[:20] + "C" + binder[21:], "generator_structure": str(gen2), "generator_binder_chain": "B"}]}))
        def fake_remote(parent_id, pdb_text, edits, binder_chain, temperature=0.1, seed=401):
            if parent_id == "a":
                raise RuntimeError("container died")
            return {"parent": parent_id, "fasta_name": "design.fa", "fasta": ">x\nAAA:CCC\n", "command": ["run.py"], "seconds": 1.0}
        self.module.native_repair = types.SimpleNamespace(remote=fake_remote)
        out = self.dir / "out2"
        self.module.main.info.raw_f(records=str(records), out=str(out))
        status = {r["candidate_id"]: r["status"] for r in json.loads((out / "native-run.json").read_text())["designs"]}
        self.assertEqual(status, {"a": "failed", "b": "ran"})


if __name__ == "__main__":
    unittest.main()
