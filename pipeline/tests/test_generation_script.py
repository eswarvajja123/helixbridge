"""Run the REAL scripts/rfd4-generate.py end to end on a CPU, replacing only the model call.

Stand-ins are used for `torch` (a GPU that always reports available, with no-op memory counters) and for
`rfd4_proteina.generate` (which copies a real structure into the output folder instead of running RFD4).
Everything else is the real code: argument handling, input checks, the real biotite parsing of the structure,
the chain checks, the audit that is written, and the hand-off to import_current. This is NOT a GPU run and says
nothing about RFD4 itself; it checks the wrapper we wrote around it.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.conditioning import write_conditioning
from binder_campaign.current_candidates import import_current
from binder_campaign.generation_input import build_overrides
from binder_campaign.runner import digest

try:
    import biotite  # noqa: F401
    HAVE_BIOTITE = True
except ImportError:
    HAVE_BIOTITE = False

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/il7ra-rehearsal"
REAL = ROOT / "examples/verified-modal-run"
MANIFEST = ROOT / "docs/eswar/phase1-data/il7ra-manifest.json"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rfd4-generate.py"
HOTSPOTS = [100, 101, 102, 158, 159, 161, 211, 212, 213, 214]
DESIGN = "SVQEKLQEIQASLHSKVAKENVDEFKSLLEKANEENDLSVLLQASNPELLSLIKEALNSL"

TORCH_STUB = '''
class _Cuda:
    is_available = staticmethod(lambda: True)
    reset_peak_memory_stats = staticmethod(lambda: None)
    synchronize = staticmethod(lambda: None)
    max_memory_reserved = staticmethod(lambda: 2564.0 * 1024 ** 2)
    max_memory_allocated = staticmethod(lambda: 1000.0 * 1024 ** 2)
cuda = _Cuda()
'''
RFD4_STUB = '''
import json, os, shutil
from pathlib import Path

def generate(model=None, *, overrides=()):
    settings = dict(o.split("=", 1) for o in overrides)
    out = Path(settings["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(os.environ["STUB_STRUCTURE"], out / "design_generated.cif")
    Path(os.environ["STUB_CAPTURE"]).write_text(json.dumps(list(overrides)))
'''


@unittest.skipUnless(HAVE_BIOTITE and EXAMPLE.is_dir() and REAL.is_dir() and MANIFEST.is_file(),
                     "needs biotite, examples/ and docs/eswar/phase1-data")
class GenerationScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        for package, source in (("torch", TORCH_STUB), ("rfd4_proteina", RFD4_STUB)):
            (self.dir / "stubs" / package).mkdir(parents=True)
            (self.dir / "stubs" / package / "__init__.py").write_text(source)
        self.task_root = self.dir / "task"
        (self.task_root / "env").mkdir(parents=True)
        (self.task_root / "data/rfd4").mkdir(parents=True)
        self.record = {"path": "/ckpt", "revision": "rev-1"}
        (self.task_root / "env/rfd4-checkpoints.json").write_text(json.dumps(self.record))
        self.ref = self.dir / "cond"
        write_conditioning(EXAMPLE, self.ref, HOTSPOTS, 60, hotspot_source="provisional_historical")

    def run_script(self, structure, seed=7, out="out"):
        env = dict(os.environ, PYTHONPATH=str(self.dir / "stubs"), TASK_ROOT=str(self.task_root),
                   STUB_STRUCTURE=str(structure), STUB_CAPTURE=str(self.dir / "captured.json"))
        return subprocess.run([sys.executable, str(SCRIPT), "--reference-dir", str(self.ref), "--manifest", str(MANIFEST),
                               "--out", str(self.dir / out), "--seed", str(seed)], env=env, capture_output=True, text=True)

    def test_full_path_writes_a_valid_audit_and_the_importer_accepts_it(self):
        done = self.run_script(REAL / "generator.cif", seed=7)
        self.assertEqual(done.returncode, 0, done.stderr)
        out = self.dir / "out"
        audit = json.loads((out / "audit.json").read_text())
        self.assertTrue(audit["status"].startswith("conditioned_generation"))
        self.assertEqual((audit["seed"], audit["binder_sequence"], audit["binder_chain"], audit["target_chain"]),
                         (7, DESIGN, "B", "A"))
        self.assertEqual(audit["binder_length_requested"], 60)
        self.assertEqual(audit["generated_sha256"], digest(out / "model.cif"))
        self.assertEqual(audit["target_manifest_sha256"], digest(MANIFEST))
        self.assertEqual({c["id"]: len(c["sequence"]) for c in audit["chains"]}, {"B": 60, "A": 195})
        self.assertGreater(audit["peak_reserved_mib"], 0)
        # the settings handed to the model are exactly the tested recipe, for the seed that was requested
        sent = json.loads((self.dir / "captured.json").read_text())
        self.assertEqual(sent, build_overrides(self.ref / "conditions.json", out.resolve(), 7, self.record))
        # and the audit this real script wrote is accepted by the importer
        record = import_current(out / "audit.json", MANIFEST, self.dir / "records.json")[0]
        self.assertEqual((record["sequence"], record["seed"], record["qc_reasons"]), (DESIGN, 7, []))

    def test_a_structure_with_the_wrong_target_is_refused_and_no_audit_is_written(self):
        # the Boltz refold has a 236-residue target chain, which is not the audited 195-residue reference
        done = self.run_script(REAL / "boltz.cif")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("ValueError", done.stderr)
        self.assertFalse((self.dir / "out/audit.json").exists())

    def test_input_checks_run_before_the_model(self):
        (self.ref / "target.cif").write_bytes((self.ref / "target.cif").read_bytes() + b"\n")
        done = self.run_script(REAL / "generator.cif")
        self.assertNotEqual(done.returncode, 0)
        self.assertFalse((self.dir / "captured.json").exists())     # the model stand-in was never called
        self.assertFalse((self.dir / "out").exists())

    def test_missing_gpu_is_a_clear_error(self):
        (self.dir / "stubs/torch/__init__.py").write_text(TORCH_STUB.replace("lambda: True", "lambda: False"))
        done = self.run_script(REAL / "generator.cif")
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("CUDA GPU is required", done.stderr)


if __name__ == "__main__":
    unittest.main()
