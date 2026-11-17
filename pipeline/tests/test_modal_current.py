"""Checks for modal_current.py that need no Modal login, GPU or network.

`Function.local()` runs a Modal function's body in this process; the torch and RFD4 libraries are replaced by tiny
stand-ins that copy a real structure, exactly as in test_generation_script.py. This tests our orchestration (the
folders it builds, the commands it runs, the record and archive it returns). It is not a Modal or GPU run.
"""
import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

try:
    import modal  # noqa: F401
    HAVE_MODAL = True
except ImportError:
    HAVE_MODAL = False
try:
    import biotite  # noqa: F401
    HAVE_BIOTITE = True
except ImportError:
    HAVE_BIOTITE = False

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "pipeline"
MODULE = ROOT / "modal_current.py"
TEAM = ROOT / "modal_app.py"
REAL = ROOT / "examples/verified-modal-run"
HAVE_FILES = (MODULE.is_file() and (ROOT / "examples/il7ra-rehearsal").is_dir() and REAL.is_dir()
              and (ROOT / "docs/eswar/phase1-data/il7ra-manifest.json").is_file())

# what the generation container is allowed to import, and what supplies each package in the image
CONTAINER_FILES = ["scripts/rfd4-generate.py", "binder_campaign/generation_input.py", "binder_campaign/conditioning.py",
                   "binder_campaign/current_candidates.py", "binder_campaign/modal_plan.py", "binder_campaign/runner.py",
                   "binder_campaign/structure.py"]
IMAGE_PROVIDES = {"numpy": "torch", "torch": "torch==2.14.1", "biotite": "biotite==1.6.0", "rfd4_proteina": "RFD4-Proteina",
                  "Bio": "biopython==1.88"}

TORCH_STUB = '''
class _Cuda:
    is_available = staticmethod(lambda: True)
    reset_peak_memory_stats = staticmethod(lambda: None)
    synchronize = staticmethod(lambda: None)
    max_memory_reserved = staticmethod(lambda: 2564.0 * 1024 ** 2)
    max_memory_allocated = staticmethod(lambda: 1000.0 * 1024 ** 2)
    get_device_name = staticmethod(lambda: "Stub GPU")
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
'''


def third_party(path):
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return {m for m in found if m not in sys.stdlib_module_names and m not in ("binder_campaign", "modal")}


@unittest.skipUnless(HAVE_FILES, "needs modal_current.py, examples/ and docs/eswar")
class ImageContentTests(unittest.TestCase):
    def test_every_third_party_import_in_the_container_path_is_installed_by_the_image(self):
        source = MODULE.read_text()
        used = set()
        for name in CONTAINER_FILES:
            used |= third_party(PIPELINE / name)
        self.assertTrue(used, "found no imports to check")
        self.assertEqual(used - set(IMAGE_PROVIDES), set(), "an import the image does not provide")
        for package in used:
            self.assertIn(IMAGE_PROVIDES[package].split("==")[0], source, f"{package} is not installed by the image")
        self.assertIn("biopython==1.88", source)         # the package that was missing from the historical image

    def test_the_image_mounts_every_folder_the_function_reads(self):
        source = MODULE.read_text()
        body = source[source.index("def generate_current"):source.index("@app.local_entrypoint")]
        for read_from, mount in (("pipeline/scripts/rfd4-generate.py", '"/opt/pipeline"'), ("examples/il7ra-rehearsal", '"/opt/examples"'),
                                 ("docs/eswar/phase1-data/il7ra-manifest.json", '"/opt/docs/eswar"')):
            self.assertIn(read_from, body)
            self.assertIn(mount, source, f"{read_from} is read but {mount} is not mounted")

    def test_pins_match_the_team_module(self):
        if not TEAM.is_file():
            self.skipTest("modal_app.py not present")
        ours, theirs = MODULE.read_text(), TEAM.read_text()
        for pin in ("ddd934f7e6b86f7582ce487d5b19f6c526c644b5", "uv==0.12.9", "'torch==2.14.1'", "'atomworks[ml,openbabel]==3.0.4'",
                    "'biotite==1.6.0'", "--torch-backend cu132", "helixbridge-model-cache",
                    "from rfproteina.generate import run_generation"):
            self.assertIn(pin, ours, pin)
            self.assertIn(pin, theirs, pin)


@unittest.skipUnless(HAVE_MODAL and HAVE_BIOTITE and HAVE_FILES, "needs modal, biotite, modal_current.py and the examples")
class RunBodyLocallyTests(unittest.TestCase):
    @classmethod
    def load(cls, opt, cache):
        os.environ.update(HELIXBRIDGE_OPT=str(opt), HELIXBRIDGE_CACHE=str(cache), HELIXBRIDGE_LOCAL_TEST="1")
        spec = importlib.util.spec_from_file_location("modal_current_under_test", MODULE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.opt, self.cache = self.dir / "opt", self.dir / "cache"
        self.opt.mkdir()
        for name in ("pipeline", "examples", "docs"):
            (self.opt / name).symlink_to(ROOT / name)
        (self.cache / "env").mkdir(parents=True)
        (self.cache / "env/rfd4-checkpoints.json").write_text(json.dumps({"path": "/ckpt", "revision": "rev-1"}))
        stubs = self.dir / "stubs"
        for package, source in (("torch", TORCH_STUB), ("rfd4_proteina", RFD4_STUB)):
            (stubs / package).mkdir(parents=True)
            (stubs / package / "__init__.py").write_text(source)
        self.saved = (list(sys.path), dict(os.environ), {k: v for k, v in sys.modules.items() if k in ("torch", "rfd4_proteina")})
        sys.path.insert(0, str(stubs))
        os.environ["PYTHONPATH"] = str(stubs) + os.pathsep + os.environ.get("PYTHONPATH", "")
        os.environ["STUB_STRUCTURE"] = str(REAL / "generator.cif")
        for name in ("torch", "rfd4_proteina"):
            sys.modules.pop(name, None)
        self.addCleanup(self.restore)
        self.module = self.load(self.opt, self.cache)

    def restore(self):
        sys.path[:] = self.saved[0]
        os.environ.clear()
        os.environ.update(self.saved[1])
        for name in ("torch", "rfd4_proteina"):
            sys.modules.pop(name, None)
        sys.modules.update(self.saved[2])

    def call(self, **overrides):
        args = dict(run_id="run-one", seed=7, hotspots=[100, 101, 102], hotspot_source="provisional_historical", binder_length=[60, 60])
        args.update(overrides)
        return self.module.generate_current.local(**args)

    def test_app_and_function_are_defined_as_planned(self):
        function = self.module.generate_current
        self.assertEqual(self.module.app.name, "helixbridge-current")
        self.assertTrue(hasattr(function, "local") and hasattr(function, "remote"))
        self.assertEqual(self.module.GPU, ["A100-80GB", "L40S", "A10G"])

    def test_gpu_can_be_forced_from_the_environment(self):
        os.environ["HELIXBRIDGE_GPU"] = "A10G"
        self.assertEqual(self.load(self.opt, self.cache).GPU, ["A10G"])

    def test_full_body_builds_inputs_runs_the_script_and_records_the_design(self):
        result = self.call()
        self.assertEqual((result["gpu"], result["run_id"]), ("Stub GPU", "run-one"))
        self.assertEqual(result["audit"]["seed"], 7)
        self.assertEqual(result["audit"]["binder_sequence"], result["record"]["sequence"])
        self.assertEqual(len(result["record"]["sequence"]), 60)
        self.assertEqual((result["record"]["hotspots_uniprot"], result["record"]["hotspot_review"]),
                         ([100, 101, 102], "unreviewed"))
        self.assertEqual(result["record"]["campaign_kind"], "competition_campaign")
        self.assertIs(result["record"]["submission_eligible"], False)
        root = self.cache / "runs/run-one"
        for part in ("inputs/conditioning/conditions.json", "inputs/il7ra-manifest.json", "generation/audit.json",
                     "generation/model.cif", "generation/environment.txt", "candidates.json"):
            self.assertTrue((root / part).is_file(), part)

    def test_returned_archive_holds_the_run(self):
        names = tarfile.open(fileobj=io.BytesIO(self.call()["archive"]), mode="r:gz").getnames()
        for part in ("run-one/generation/audit.json", "run-one/candidates.json", "run-one/inputs/conditioning/audit.json"):
            self.assertIn(part, names)

    def test_a_length_range_is_passed_through_to_the_conditioning(self):
        # the stand-in always returns a 60-residue design, which fits 50-80
        self.call(run_id="run-range", binder_length=[50, 80])
        contig = json.loads((self.cache / "runs/run-range/inputs/conditioning/conditions.json").read_text())["il7ra"]["contig"]
        self.assertEqual(contig, "50-80, /0, A37-231")

    def test_bad_requests_are_refused_before_anything_runs(self):
        for kwargs in ({"seed": -1}, {"run_id": "bad id"}, {"binder_length": [30, 60]}, {"hotspots": [25, 100]}):
            with self.assertRaises(ValueError, msg=repr(kwargs)):
                self.call(run_id=kwargs.get("run_id", "fresh-run"), **{k: v for k, v in kwargs.items() if k != "run_id"})
            self.assertFalse((self.cache / "runs/fresh-run/generation").exists(), repr(kwargs))

    def test_a_used_run_id_is_refused(self):
        self.call()
        with self.assertRaises(ValueError):
            self.call()

    def test_a_design_that_does_not_fit_the_request_fails_loudly(self):
        # exact 80 requested, but the stand-in returns a 60-residue design: the script must refuse and nothing is recorded
        with self.assertRaises(Exception):
            self.call(run_id="run-bad", binder_length=[80, 80])
        self.assertFalse((self.cache / "runs/run-bad/candidates.json").exists())


if __name__ == "__main__":
    unittest.main()
