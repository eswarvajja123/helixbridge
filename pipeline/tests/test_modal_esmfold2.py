"""Checks for modal_esmfold2.py that need no Modal login, GPU, network or 25 GB of weights.

`Function.local()` runs a Modal function's body in this process. The libraries that need a GPU or the weights (torch, esm,
huggingface_hub) are replaced by small stand-ins that record how they were called and hand back a valid structure. So this
tests our orchestration: the integrity check before loading, the order and arguments of the calls, the validation of the
result, and what is written. It is not a Modal run, and it says nothing about ESMFold2 itself.
"""
import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import modal  # noqa: F401
    import biotite  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "modal_esmfold2.py"
PHASE1 = ROOT / "docs/eswar/phase1-data"
SUMMARY = ROOT / "examples/verified-modal-run/summary.json"
AVAILABLE = MODULE.is_file() and PHASE1.is_dir() and SUMMARY.is_file()
TEAM_SCREEN = ROOT / "pipeline/scripts/prepare-screen-control.py"
TEAM_INSTALL = ROOT / "pipeline/slurm/install-esmfold2.sbatch"

STUBS = {
    "torch/__init__.py": '''
__version__ = "2.11.0-stub"
class _Cuda:
    is_available = staticmethod(lambda: True)
    reset_peak_memory_stats = staticmethod(lambda: None)
    synchronize = staticmethod(lambda: None)
    max_memory_reserved = staticmethod(lambda: 26000.0 * 1024 ** 2)
    max_memory_allocated = staticmethod(lambda: 20000.0 * 1024 ** 2)
    get_device_name = staticmethod(lambda: "Stub GPU")
cuda = _Cuda()
''',
    "esm/__init__.py": "",
    "esm/models/__init__.py": "",
    "esm/utils/__init__.py": "",
    "esm/utils/msa/__init__.py": '''
CALLS = []
class MSA:
    @classmethod
    def from_a3m(cls, path, remove_insertions=False, max_sequences=None):
        CALLS.append((str(path), remove_insertions, max_sequences))
        return cls()
''',
    "esm/models/esmfold2/__init__.py": '''
import os
from pathlib import Path
import numpy as np
CALLS = []
class _Config:
    esmc_config = object()
class EsmFold2Model:
    @classmethod
    def from_pretrained(cls, path, load_esmc=False, device="cuda"):
        CALLS.append(("from_pretrained", path, load_esmc, device))
        model = cls()
        model.esmc = None if os.environ.get("STUB_NO_ESMC") else object()
        model.config = _Config()
        return model
    def eval(self):
        return self
    def set_kernel_backend(self, backend):
        CALLS.append(("set_kernel_backend", backend))
class ProteinInput:
    def __init__(self, id, sequence, msa=None):
        self.id, self.sequence, self.msa = id, sequence, msa
class StructurePredictionInput:
    def __init__(self, sequences):
        self.sequences = sequences
class _Complex:
    def to_mmcif(self):
        return Path(os.environ["STUB_CIF"]).read_text()
class _Result:
    pass
class ESMFold2InputBuilder:
    def __init__(self, ccd_cache):
        CALLS.append(("builder", str(ccd_cache), os.environ.get("ESMCFOLD_CCD_PATH")))
    def fold(self, model, spi, **settings):
        CALLS.append(("fold", [(p.id, p.sequence, p.msa is not None) for p in spi.sequences], settings))
        n = sum(len(p.sequence) for p in spi.sequences)
        result = _Result()
        result.pae = np.full((n, n), 5.0)
        result.plddt = np.full(n, 0.8)
        result.ptm, result.iptm, result.complex = 0.7, 0.6, _Complex()
        return result
''',
    "huggingface_hub/__init__.py": '''
import json, os
from pathlib import Path
CALLS = []
def snapshot_download(repo_id, revision, local_dir, allow_patterns, max_workers, token):
    CALLS.append({"repo_id": repo_id, "revision": revision, "allow_patterns": allow_patterns, "token": token})
    for name, text in json.loads(Path(os.environ["STUB_FILES"]).read_text()).items():
        path = Path(local_dir) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return local_dir
''',
}
PACKAGES = ("torch", "esm", "huggingface_hub", "binder_campaign")


def third_party(path):
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return {m for m in found if m not in sys.stdlib_module_names and m not in ("binder_campaign", "modal")}


@unittest.skipUnless(AVAILABLE, "needs modal_esmfold2.py, docs/eswar/phase1-data and examples/")
class ImageAndPinTests(unittest.TestCase):
    def test_pins_and_build_checks_are_in_the_image(self):
        source = MODULE.read_text()
        for needed in ("'torch==2.11.0'", "'esm==3.4.1.post1'", "'biopython==1.88'", "'huggingface-hub'", "uv==0.12.9", "Python.h",
                       "gcc -fsyntax-only", "import triton", "from esm.models.esmfold2 import", "build-essential"):
            self.assertIn(needed, source, needed)

    def test_every_third_party_import_is_provided_by_the_image(self):
        provides = {"numpy": "esm==3.4.1.post1", "biotite": "esm==3.4.1.post1", "torch": "torch==2.11.0",
                    "esm": "esm==3.4.1.post1", "huggingface_hub": "huggingface-hub"}
        used = third_party(MODULE) | third_party(ROOT / "pipeline/binder_campaign/esm_eval.py")
        self.assertTrue(used)
        self.assertEqual(used - set(provides), set(), "an import that nothing in the image provides")
        source = MODULE.read_text()
        for package in used:
            self.assertIn(provides[package], source, package)

    def test_the_smaller_gpu_is_left_out_and_the_choice_can_be_forced(self):
        source = MODULE.read_text()
        self.assertIn('["A100-80GB", "L40S"]', source)
        self.assertNotIn("A10G", source.split("GPU = ")[1].split("\n")[0])

    @unittest.skipUnless(TEAM_INSTALL.is_file(), "needs the team's install script")
    def test_the_esm_version_is_the_one_the_team_ran(self):
        self.assertIn("esm==3.4.1.post1", TEAM_INSTALL.read_text())

    @unittest.skipUnless(TEAM_SCREEN.is_file(), "needs the team's screen script")
    def test_the_screen_settings_match_the_teams_script_exactly(self):
        from binder_campaign.esm_eval import SCREEN_SETTINGS
        text = TEAM_SCREEN.read_text()
        start = text.index("'settings':{") + len("'settings':")
        depth, end = 0, start
        for i, ch in enumerate(text[start:], start):
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0:
                end = i + 1
                break
        theirs = ast.literal_eval(text[start:end])
        self.assertEqual(theirs, SCREEN_SETTINGS)


@unittest.skipUnless(HAVE and AVAILABLE, "needs modal, biotite, modal_esmfold2.py and the Phase 1 data")
class BodyTests(unittest.TestCase):
    def setUp(self):
        import test_esm_eval as T
        from binder_campaign import esm_eval as E
        self.E = E
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.target = json.loads((PHASE1 / "il7ra-manifest.json").read_text())["sequence"]
        self.binder = json.loads(SUMMARY.read_text())["binder_sequence"]
        # checkpoint: two tiny files that stand in for the 11 real ones, with a matching test manifest
        self.cache, self.opt = self.dir / "cache", self.dir / "opt"
        self.folder = self.cache / "esmfold2" / E.REVISION
        self.folder.mkdir(parents=True)
        self.files = {"config.json": '{"esmc_config": {}}', "weights.safetensors": "w" * 3000}
        self.manifest = T.make_files(self.folder, {k: v.encode() for k, v in self.files.items()})
        self.manifest["allow_patterns"] = ["*.json", "*.safetensors"]
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest))
        # project folders as the container sees them
        self.opt.mkdir()
        (self.opt / "pipeline").symlink_to(ROOT / "pipeline")
        (self.opt / "examples").symlink_to(ROOT / "examples")
        (self.opt / "docs").symlink_to(ROOT / "docs")
        rng = __import__("numpy").random.default_rng(1)
        coords = lambda n: rng.normal(size=(n, 3)) * 10
        self.good_cif = T.full_cif({"A": (self.target, coords(len(self.target))), "B": (self.binder, coords(len(self.binder)))})
        self.swapped_cif = T.full_cif({"B": (self.binder, coords(len(self.binder))), "A": (self.target, coords(len(self.target)))})
        (self.dir / "good.cif").write_text(self.good_cif)
        (self.dir / "swapped.cif").write_text(self.swapped_cif)
        stubs = self.dir / "stubs"
        for name, source in STUBS.items():
            path = stubs / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        self.saved = (list(sys.path), dict(os.environ),
                      {k: v for k, v in sys.modules.items() if k.split(".")[0] in PACKAGES})
        for name in [n for n in sys.modules if n.split(".")[0] in PACKAGES[:3]]:
            sys.modules.pop(name)
        sys.path.insert(0, str(stubs))
        os.environ.update(HELIXBRIDGE_OPT=str(self.opt), HELIXBRIDGE_CACHE=str(self.cache), HELIXBRIDGE_LOCAL_TEST="1",
                          HELIXBRIDGE_ESM_MANIFEST=str(self.dir / "manifest.json"), HELIXBRIDGE_RESULTS=str(self.dir / "results"),
                          STUB_CIF=str(self.dir / "good.cif"), STUB_FILES=json.dumps({}))
        (self.dir / "files.json").write_text(json.dumps(self.files))
        os.environ["STUB_FILES"] = str(self.dir / "files.json")
        os.environ.pop("STUB_NO_ESMC", None)
        self.addCleanup(self.restore)
        spec = importlib.util.spec_from_file_location("modal_esmfold2_under_test", MODULE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def restore(self):
        sys.path[:] = self.saved[0]
        os.environ.clear()
        os.environ.update(self.saved[1])
        for name in [n for n in sys.modules if n.split(".")[0] in PACKAGES]:
            sys.modules.pop(name)
        sys.modules.update(self.saved[2])

    def calls(self):
        module = sys.modules.get("esm.models.esmfold2")      # never imported means the model was never reached
        return module.CALLS if module else []

    def qualify(self, **kwargs):
        args = dict(run_id="q1", seed=0, kernel_backend="fused")
        args.update(kwargs)
        return self.module.qualify_esmfold2.local(**args)

    # ---- prepare
    def test_prepare_downloads_the_pinned_revision_verifies_it_and_writes_the_record(self):
        shutil.rmtree(self.folder)                                       # start from an empty volume
        result = self.module.prepare_esmfold2.local(True)
        self.assertEqual((result["status"], result["files"], result["hashed"]), ("checkpoint verified", 2, True))
        call = sys.modules["huggingface_hub"].CALLS[0]
        self.assertEqual((call["repo_id"], call["revision"], call["token"]), (self.E.REPO, self.E.REVISION, False))
        self.assertEqual(call["allow_patterns"], ["*.json", "*.safetensors"])
        record = json.loads((self.cache / "env/esmfold2-checkpoints.json").read_text())
        self.assertEqual((record[0]["repo"], record[0]["revision"], record[0]["path"]), (self.E.REPO, self.E.REVISION, str(self.folder)))

    def test_prepare_refuses_a_download_whose_content_differs(self):
        shutil.rmtree(self.folder)
        bad = dict(self.files, **{"weights.safetensors": "x" * 3000})    # right size, wrong content
        (self.dir / "files.json").write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.module.prepare_esmfold2.local(True)
        self.assertFalse((self.cache / "env/esmfold2-checkpoints.json").exists())     # nothing is recorded as verified

    def test_a_size_only_prepare_cannot_see_same_size_damage_which_is_why_hashing_is_the_default(self):
        shutil.rmtree(self.folder)
        (self.dir / "files.json").write_text(json.dumps(dict(self.files, **{"weights.safetensors": "x" * 3000})))
        self.assertFalse(self.module.prepare_esmfold2.local(False)["hashed"])

    def test_prepare_refuses_a_missing_file(self):
        shutil.rmtree(self.folder)
        (self.dir / "files.json").write_text(json.dumps({"config.json": self.files["config.json"]}))
        with self.assertRaisesRegex(ValueError, "weights.safetensors"):
            self.module.prepare_esmfold2.local(True)

    # ---- qualify
    def test_qualify_loads_folds_validates_and_reports(self):
        result = self.qualify(seed=3)
        m = result["metrics"]
        self.assertEqual((m["gpu"], m["ptm"], m["iptm"], m["peak_reserved_mib"]), ("Stub GPU", 0.7, 0.6, 26000.0))
        self.assertEqual((m["checks"]["residues"], m["target_residues"], m["binder_residues"]), (279, 219, 60))
        self.assertEqual(m["versions"]["torch"], "2.11.0-stub")
        self.assertEqual((m["kernel_backend"], m["settings"]["seed"]), ("fused", 3))
        self.assertIn("not accuracy", m["status"])
        names = tarfile.open(fileobj=io.BytesIO(result["archive"]), mode="r:gz").getnames()
        for part in ("q1/esmfold2/model.cif", "q1/esmfold2/pae.json", "q1/esmfold2/metrics.json", "q1/esmfold2/request.json"):
            self.assertIn(part, names)

    def test_the_model_is_called_exactly_as_the_teams_script_calls_it(self):
        self.qualify(seed=5)
        by_kind = {c[0]: c for c in self.calls()}
        self.assertEqual(by_kind["from_pretrained"][1:], (str(self.folder), False, "cuda"))
        self.assertEqual(by_kind["set_kernel_backend"][1], "fused")
        self.assertEqual(by_kind["builder"][1:], (str(self.folder), str(self.folder)))          # ccd path set before folding
        _, inputs, settings = by_kind["fold"]
        self.assertEqual(inputs, [("A", self.target, True), ("B", self.binder, False)])         # target with MSA, binder without
        self.assertEqual(settings, {"num_loops": 20, "num_sampling_steps": 200, "num_diffusion_samples": 1, "seed": 5,
                                    "lm_dropout": 0.3, "msa_max_depth": 1024, "msa_column_mask_rate": 0.1})
        msa = sys.modules["esm.utils.msa"].CALLS[0]
        self.assertEqual((Path(msa[0]).name, msa[1], msa[2]), ("target.a3m", True, 1024))

    def test_kernel_backend_choices(self):
        self.qualify(run_id="b1", kernel_backend="none")
        self.assertIn(("set_kernel_backend", None), self.calls())
        self.qualify(run_id="b2", kernel_backend="cuequivariance")
        self.assertIn(("set_kernel_backend", "cuequivariance"), self.calls())
        before = len(self.calls())
        with self.assertRaises(ValueError):
            self.qualify(run_id="b3", kernel_backend="turbo")
        self.assertEqual(len(self.calls()), before)                      # refused before anything was loaded

    def test_refusals_happen_before_the_model_is_loaded(self):
        (self.folder / "weights.safetensors").write_text("short")          # wrong size
        for label, kwargs in (("damaged checkpoint", {}), ("bad run id", {"run_id": "bad id"}), ("bad seed", {"seed": -1})):
            with self.assertRaises(ValueError, msg=label):
                self.qualify(**kwargs)
        self.assertFalse(any(c[0] == "from_pretrained" for c in self.calls()))

    def test_a_missing_checkpoint_is_refused(self):
        shutil.rmtree(self.folder)
        with self.assertRaisesRegex(ValueError, "missing or not a regular file"):
            self.qualify()

    def test_a_msa_for_another_target_is_refused(self):
        docs = self.dir / "docs2/eswar"
        shutil.copytree(PHASE1, docs / "phase1-data")
        a3m = docs / "phase1-data/target.a3m"
        lines = a3m.read_text().splitlines()
        lines[1] = "A" + lines[1][1:]
        a3m.write_text("\n".join(lines) + "\n")
        (self.opt / "docs").unlink()
        (self.opt / "docs").symlink_to(self.dir / "docs2")
        with self.assertRaisesRegex(ValueError, "query row"):
            self.qualify()
        self.assertFalse(any(c[0] == "from_pretrained" for c in self.calls()))

    def test_a_checkpoint_without_the_bundled_backbone_is_refused(self):
        os.environ["STUB_NO_ESMC"] = "1"
        with self.assertRaisesRegex(ValueError, "bundled ESMC"):
            self.qualify()
        self.assertFalse(any(c[0] == "fold" for c in self.calls()))

    def test_an_output_with_the_wrong_chain_order_is_refused_and_nothing_is_recorded(self):
        os.environ["STUB_CIF"] = str(self.dir / "swapped.cif")
        with self.assertRaisesRegex(ValueError, "chain order"):
            self.qualify()
        self.assertFalse((self.cache / "runs/q1").exists())

    def test_a_used_run_id_is_refused(self):
        self.qualify()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.qualify()

    # ---- entry point
    def test_entrypoint_prepare_and_qualify_stages(self):
        import types
        real_q, real_p = self.module.qualify_esmfold2, self.module.prepare_esmfold2
        self.module.qualify_esmfold2 = types.SimpleNamespace(remote=real_q.local)
        self.module.prepare_esmfold2 = types.SimpleNamespace(remote=real_p.local)
        raw = self.module.main.info.raw_f
        raw(stage="prepare")
        raw(stage="qualify", run_id="ep1", seed=2)
        saved = json.loads((self.dir / "results/ep1/esmfold2-qualify.json").read_text())
        self.assertEqual((saved["metrics"]["settings"]["seed"], saved["run_id"]), (2, "ep1"))
        self.assertTrue((self.dir / "results/ep1/esmfold2-qualify.tar.gz").is_file())
        with self.assertRaises(ValueError):                               # a local result is never overwritten
            raw(stage="qualify", run_id="ep1")

    def test_entrypoint_refuses_bad_requests_before_launching_anything(self):
        import types
        launched = []
        self.module.qualify_esmfold2 = types.SimpleNamespace(remote=lambda *a: launched.append(a))
        raw = self.module.main.info.raw_f
        for kwargs in ({"stage": "nonsense"}, {"stage": "qualify", "run_id": "bad id"},
                       {"stage": "qualify", "run_id": "x1", "kernel_backend": "turbo"}, {"stage": "qualify", "run_id": "x2", "seed": -1}):
            with self.assertRaises(ValueError, msg=repr(kwargs)):
                raw(**kwargs)
        self.assertEqual(launched, [])


if __name__ == "__main__":
    unittest.main()
