"""Checks for modal_evaluate.py that need no Modal login, GPU, network or model weights.

The function bodies run locally (`Function.local()`). torch, esm and Boltz are replaced by stand-ins that build a correct CA-only
structure for whatever sequence they are given, so every candidate really goes through our validation and scoring. This tests
our orchestration, not ESMFold2 or Boltz-2. It is not a Modal run.
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

import _designs as D

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "modal_evaluate.py"
PHASE1 = ROOT / "docs/eswar/phase1-data"
FIXTURE = Path(__file__).resolve().parent / "fixtures/ipsae_model_10_10.txt"
REAL_IPSAE = os.environ.get("HELIXBRIDGE_IPSAE", "")
TEAM_MODAL = ROOT / "modal_app.py"
TEAM_BOLTZ = ROOT / "pipeline/scripts/boltz-predict.py"
AVAILABLE = MODULE.is_file() and PHASE1.is_dir() and FIXTURE.is_file()
PACKAGES = ("torch", "esm", "huggingface_hub", "binder_campaign")

CIF_CODE = '''
import numpy as np
THREE = %r
COLS = ["group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id", "label_asym_id", "label_entity_id",
        "label_seq_id", "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy", "B_iso_or_equiv", "auth_seq_id",
        "auth_comp_id", "auth_asym_id", "auth_atom_id", "pdbx_PDB_model_num"]
def make_cif(chains):
    rows = ["data_t", "loop_"] + ["_atom_site." + c for c in COLS]
    rng, n = np.random.default_rng(1), 0
    for entity, (chain, seq) in enumerate(chains, 1):
        for i, (aa, (x, y, z)) in enumerate(zip(seq, np.cumsum(rng.normal(size=(len(seq), 3)), axis=0) * 3), 1):
            res = THREE[aa]
            atoms = [("CA", x, y, z)] + ([] if res == "GLY" else [("CB", x + 1.5, y + 0.5, z)])   # the real scorer tracks CB (CA for Gly)
            for name, ax, ay, az in atoms:
                n += 1
                rows.append(f"ATOM {n} C {name} . {res} {chain} {entity} {i} ? {ax:.3f} {ay:.3f} {az:.3f} 1.00 50.00 {i} {res} {chain} {name} 1")
    return "\\n".join(rows) + "\\n"
''' % ({v: k for k, v in __import__("binder_campaign.structure", fromlist=["AA3"]).AA3.items()},)

ESM_STUB = CIF_CODE + '''
import os
CALLS = []
class _Config:
    esmc_config = object()
class EsmFold2Model:
    @classmethod
    def from_pretrained(cls, path, load_esmc=False, device="cuda"):
        CALLS.append(("from_pretrained", path))
        model = cls()
        model.esmc, model.config = object(), _Config()
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
    def __init__(self, spi):
        self.spi = spi
    def to_mmcif(self):
        chains = [(p.id, p.sequence) for p in self.spi.sequences]
        if self.spi.sequences[1].sequence == os.environ.get("STUB_BAD_ORDER_SEQ"):
            chains.reverse()                                   # a prediction whose chain order differs from the request
        return make_cif(chains)
class _Result:
    pass
class ESMFold2InputBuilder:
    def __init__(self, ccd_cache):
        pass
    def fold(self, model, spi, **settings):
        binder = spi.sequences[1].sequence
        CALLS.append(("fold", binder, settings))
        if binder == os.environ.get("STUB_FAIL_SEQ"):
            raise RuntimeError("CUDA out of memory (stand-in)")
        n = sum(len(p.sequence) for p in spi.sequences)
        result = _Result()
        result.pae = np.full((n, n), 5.0)
        result.plddt = np.full(n, 0.8)
        result.ptm, result.iptm, result.complex = 0.7, 0.6, _Complex(spi)
        return result
'''
MSA_STUB = "CALLS = []\nclass MSA:\n    @classmethod\n    def from_a3m(cls, path, remove_insertions=False, max_sequences=None):\n        CALLS.append((str(path), remove_insertions, max_sequences))\n        return cls()\n"
TORCH_STUB = '''
class _Cuda:
    reset_peak_memory_stats = staticmethod(lambda: None)
    synchronize = staticmethod(lambda: None)
    max_memory_reserved = staticmethod(lambda: 26000.0 * 1024 ** 2)
    get_device_name = staticmethod(lambda: "Stub GPU")
cuda = _Cuda()
'''
FAKE_BOLTZ = CIF_CODE + '''
import json, os, sys
from pathlib import Path
request = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2]); out.mkdir(parents=True)
Path(os.environ["BOLTZ_CALLS"]).open("a").write(json.dumps({"request": request, "script_dir": str(Path(__file__).resolve().parent)}) + "\\n")
if request["partner"]["sequence"] == os.environ.get("STUB_FAIL_SEQ"):
    sys.stderr.write("RuntimeError: Boltz could not fold this (stand-in)\\n")
    raise SystemExit(3)
target, binder = request["target"]["sequence"], request["partner"]["sequence"]
n = len(target) + len(binder)
(out / "model.cif").write_text(make_cif([("A", target), ("B", binder)]))
(out / "pae.json").write_text(json.dumps({"pae": np.full((n, n), 4.0).tolist()}))
rows = [{"chain1": "A", "chain2": "B", "type": "asym", "ipsae": 0.4}, {"chain1": "B", "chain2": "A", "type": "asym", "ipsae": 0.5},
        {"chain1": "A", "chain2": "B", "type": "max", "ipsae": 0.5}]
(out / "metrics.json").write_text(json.dumps({"iptm": 0.8, "ptm": 0.9, "ipsae": {"pairwise": rows}, "prediction_seconds": 12.0,
    "peak_reserved_mib": 9000.0, "seed": request["settings"]["seed"], "samples": 1, "target_only_precomputed_msa": True}))
'''
FAKE_IPSAE = "import sys, pathlib\npae, cif, a, b = sys.argv[1:5]\nc = pathlib.Path(cif)\nc.with_name(c.stem + '_' + a + '_' + b + '.txt').write_text(open(%r).read())\n" % str(FIXTURE)


def reachable_third_party(entry_modules):
    """Third-party imports in the given binder_campaign modules AND every binder_campaign module they import."""
    package = ROOT / "pipeline/binder_campaign"
    seen, found, todo = set(), set(), list(entry_modules)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        for node in ast.walk(ast.parse((package / f"{name}.py").read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                todo.append(node.module.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.startswith("binder_campaign."):
                todo.append(node.module.split(".")[1])
            elif isinstance(node, ast.ImportFrom) and node.level == 1 and not node.module:
                todo.extend(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return {m for m in found if m not in sys.stdlib_module_names and m != "binder_campaign"}


@unittest.skipUnless(AVAILABLE, "needs modal_evaluate.py, docs/eswar/phase1-data and the fixture")
class ImageAndPinTests(unittest.TestCase):
    def test_every_import_the_esm_function_can_reach_is_provided_by_its_image(self):
        provided = {"numpy": "esm==3.4.1.post1", "biotite": "esm==3.4.1.post1", "Bio": "biopython==1.88", "torch": "torch==2.11.0",
                    "esm": "esm==3.4.1.post1", "huggingface_hub": "huggingface-hub"}
        source = MODULE.read_text()
        used = reachable_third_party(["modal_plan", "esm_eval", "scoring"])
        self.assertIn("Bio", used)                                    # the transitive import that a shallow scan would miss
        self.assertEqual(used - set(provided), set())
        for package in used:
            self.assertIn(provided[package], source, package)

    def test_every_import_the_boltz_function_can_reach_is_provided_by_its_image(self):
        used = reachable_third_party(["modal_plan", "scoring"])
        self.assertEqual(used - {"Bio"}, set())
        self.assertIn('"biopython"', MODULE.read_text())              # the team's boltz image line installs it

    def test_the_esm_image_recipe_matches_modal_esmfold2(self):
        ours, theirs = MODULE.read_text(), (ROOT / "modal_esmfold2.py").read_text()
        for part in ("'torch==2.11.0' 'esm==3.4.1.post1' 'biopython==1.88' 'huggingface-hub'", "assert os.path.exists(p), p",
                     "gcc -fsyntax-only", "from esm.models.esmfold2 import EsmFold2Model, ESMFold2InputBuilder, ProteinInput, "
                     "StructurePredictionInput; ", "import triton"):
            self.assertIn(part, ours, part)
            self.assertIn(part, theirs, part)

    @unittest.skipUnless(TEAM_MODAL.is_file(), "needs the team's modal_app.py")
    def test_the_boltz_image_and_scorer_commit_match_the_teams(self):
        theirs = TEAM_MODAL.read_text()
        self.assertIn('base.uv_pip_install("boltz==2.2.1", "huggingface-hub", "biopython")', theirs)
        self.assertIn('base.uv_pip_install("boltz==2.2.1", "huggingface-hub", "biopython")', MODULE.read_text())
        from binder_campaign.scoring import IPSAE_COMMIT
        self.assertIn(IPSAE_COMMIT, theirs)
        self.assertIn(f'IPSAE_COMMIT = "{IPSAE_COMMIT}"', MODULE.read_text())

    def test_gpu_choices_mounts_and_script_path(self):
        source = MODULE.read_text()
        self.assertIn('["A100-80GB", "L40S"]', source)
        self.assertNotIn("A10G", [l for l in source.splitlines() if l.startswith("ESM_GPU")][0])
        self.assertIn('"A10G"', [l for l in source.splitlines() if l.startswith("BOLTZ_GPU")][0])   # Boltz ran on an A10G
        for mount in ('"/opt/pipeline"', '"/opt/docs/eswar"'):
            self.assertIn(mount, source)
        self.assertIn('pipeline/scripts/boltz-predict.py', source)
        if TEAM_BOLTZ.is_file():
            self.assertTrue(TEAM_BOLTZ.is_file())


@unittest.skipUnless(HAVE and AVAILABLE, "needs modal, biotite, modal_evaluate.py and the Phase 1 data")
class BodyTests(unittest.TestCase):
    def setUp(self):
        import test_esm_eval as T
        from binder_campaign import esm_eval as E
        self.E = E
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.cache, self.opt = self.dir / "cache", self.dir / "opt"
        self.folder = self.cache / "esmfold2" / E.REVISION
        self.folder.mkdir(parents=True)
        manifest = T.make_files(self.folder, {"config.json": b'{"esmc_config": {}}', "weights.safetensors": b"w" * 2000})
        manifest["allow_patterns"] = ["*.json"]
        (self.dir / "manifest.json").write_text(json.dumps(manifest))
        (self.cache / "env").mkdir()
        (self.cache / "env/boltz-checkpoints.json").write_text(json.dumps({"cache": "/cache/data/weights/boltz2"}))
        self.opt.mkdir()
        (self.opt / "pipeline").symlink_to(ROOT / "pipeline")
        (self.opt / "docs").symlink_to(ROOT / "docs")
        (self.opt / "IPSAE").mkdir()
        if REAL_IPSAE and Path(REAL_IPSAE).is_file():
            shutil.copyfile(REAL_IPSAE, self.opt / "IPSAE/ipsae.py")        # the real pinned scorer, when it is available
        else:
            (self.opt / "IPSAE/ipsae.py").write_text(FAKE_IPSAE)
        (self.dir / "fake_boltz.py").write_text(FAKE_BOLTZ)
        stubs = self.dir / "stubs"
        for name, source in {"torch/__init__.py": TORCH_STUB, "esm/__init__.py": "", "esm/models/__init__.py": "",
                             "esm/utils/__init__.py": "", "esm/utils/msa/__init__.py": MSA_STUB,
                             "esm/models/esmfold2/__init__.py": ESM_STUB}.items():
            path = stubs / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        self.saved = (list(sys.path), dict(os.environ), {k: v for k, v in sys.modules.items() if k.split(".")[0] in PACKAGES})
        for name in [n for n in sys.modules if n.split(".")[0] in PACKAGES[:2]]:
            sys.modules.pop(name)
        sys.path.insert(0, str(stubs))
        os.environ.update(HELIXBRIDGE_OPT=str(self.opt), HELIXBRIDGE_CACHE=str(self.cache), HELIXBRIDGE_LOCAL_TEST="1",
                          HELIXBRIDGE_ESM_MANIFEST=str(self.dir / "manifest.json"), HELIXBRIDGE_BOLTZ_SCRIPT=str(self.dir / "fake_boltz.py"),
                          HELIXBRIDGE_RESULTS=str(self.dir / "results"), BOLTZ_CALLS=str(self.dir / "boltz-calls.jsonl"))
        os.environ.pop("STUB_FAIL_SEQ", None)
        os.environ.pop("STUB_BAD_ORDER_SEQ", None)
        self.addCleanup(self.restore)
        spec = importlib.util.spec_from_file_location("modal_evaluate_under_test", MODULE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.seqs = [D.random_sequence(300 + i) for i in range(1, 5)]
        self.ids = [f"il7ra-rfd4-{i:012x}" for i in range(1, 5)]

    def restore(self):
        sys.path[:] = self.saved[0]
        os.environ.clear()
        os.environ.update(self.saved[1])
        for name in [n for n in sys.modules if n.split(".")[0] in PACKAGES]:
            sys.modules.pop(name)
        sys.modules.update(self.saved[2])

    def esm_calls(self):
        module = sys.modules.get("esm.models.esmfold2")
        return module.CALLS if module else []

    def request(self, predictor="esmfold2", run_id="r1", n=3, **extra):
        return {"run_id": run_id, "predictor": predictor, "seed": 0, "kernel_backend": "fused",
                "candidates": [{"id": self.ids[i], "sequence": self.seqs[i]} for i in range(n)], **extra}

    def members(self, response):
        return sorted(tarfile.open(fileobj=io.BytesIO(response["archive"]), mode="r:gz").getnames())

    # ---- ESMFold2
    def test_esm_loads_the_model_once_folds_every_candidate_and_scores_it(self):
        response = self.module.esmfold2_evaluate.local(self.request())
        self.assertEqual([r["status"] for r in response["results"]], ["ok"] * 3)
        calls = self.esm_calls()
        self.assertEqual(sum(c[0] == "from_pretrained" for c in calls), 1)                       # the 25 GB load happens once
        self.assertEqual([c[1] for c in calls if c[0] == "fold"], self.seqs[:3])
        score = response["results"][0]["score"]
        self.assertEqual((score["predictor"], score["gpu"], score["settings"]["revision"]), ("esmfold2", "Stub GPU", self.E.REVISION))
        self.assertEqual(score["ipsae"]["pae_cutoff"], 10)
        if REAL_IPSAE and Path(REAL_IPSAE).is_file():          # the real scorer ran: the numbers must be its own, not constants
            self.assertTrue(any(r["ipsae"] != 0.0 for r in score["ipsae"]["pairwise"]))
        else:                                                   # the stand-in scorer replays the real fixture
            self.assertEqual((score["ipsae"]["a_b"], score["ipsae"]["b_a"], score["ipsae"]["max"]), (0.111551, 0.439671, 0.439671))
        self.assertEqual(self.members(response),
                         sorted(f"r1/{i}/{n}" for i in self.ids[:3] for n in ("model.cif", "pae.json", "metrics.json")))
        self.assertEqual((response["gpu"], response["run_id"]), ("Stub GPU", "r1"))
        self.assertGreater(response["worker_seconds"], 0)

    def test_esm_backend_and_seed_reach_the_model(self):
        self.module.esmfold2_evaluate.local(self.request(kernel_backend="none", seed=6))
        self.assertIn(("set_kernel_backend", None), self.esm_calls())
        fold = next(c for c in self.esm_calls() if c[0] == "fold")
        self.assertEqual((fold[2]["seed"], "kernel_backend" in fold[2]), (6, False))

    def test_one_failing_candidate_does_not_cost_the_others(self):
        os.environ["STUB_FAIL_SEQ"] = self.seqs[1]
        response = self.module.esmfold2_evaluate.local(self.request())
        self.assertEqual([r["status"] for r in response["results"]], ["ok", "failed", "ok"])
        self.assertIn("out of memory", response["results"][1]["error"])
        self.assertFalse(any(m.startswith(f"r1/{self.ids[1]}/") for m in self.members(response)))

    def test_a_prediction_with_the_wrong_chain_order_is_rejected_inside_the_container(self):
        os.environ["STUB_BAD_ORDER_SEQ"] = self.seqs[0]
        response = self.module.esmfold2_evaluate.local(self.request(n=2))
        self.assertEqual([r["status"] for r in response["results"]], ["failed", "ok"])
        self.assertIn("chain order", response["results"][0]["error"])
        self.assertFalse(any(m.startswith(f"r1/{self.ids[0]}/") for m in self.members(response)))        # nothing of it is shipped

    def test_esm_refusals_happen_before_the_model_is_loaded(self):
        bad = {"wrong predictor": self.request("boltz2"), "bad run id": self.request(run_id="bad id"),
               "no candidates": {**self.request(), "candidates": []},
               "too many candidates": {**self.request(), "candidates": [{"id": f"c{i}", "sequence": self.seqs[0]} for i in range(21)]},
               "duplicate ids": {**self.request(), "candidates": [{"id": "x", "sequence": self.seqs[0]}] * 2},
               "unsafe id": {**self.request(), "candidates": [{"id": "../x", "sequence": self.seqs[0]}]},
               "invalid sequence": {**self.request(), "candidates": [{"id": "x", "sequence": "ACDX"}]},
               "bad seed": self.request(seed=-1), "bad backend": self.request(kernel_backend="turbo")}
        for label, request in bad.items():
            with self.assertRaises(ValueError, msg=label):
                self.module.esmfold2_evaluate.local(request)
        (self.folder / "weights.safetensors").write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "wrong size"):
            self.module.esmfold2_evaluate.local(self.request(run_id="r2"))
        self.assertEqual(self.esm_calls(), [])

    def test_a_missing_scorer_is_refused_and_a_used_run_id_is_refused(self):
        self.module.esmfold2_evaluate.local(self.request(run_id="used", n=1))
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.module.esmfold2_evaluate.local(self.request(run_id="used", n=1))
        (self.opt / "IPSAE/ipsae.py").unlink()
        with self.assertRaisesRegex(ValueError, "scorer is missing"):
            self.module.esmfold2_evaluate.local(self.request(run_id="r3"))

    # ---- Boltz-2
    def test_boltz_runs_the_teams_script_once_per_candidate_with_the_current_target(self):
        response = self.module.boltz_evaluate.local(self.request("boltz2", n=2, run_id="b1", seed=5))
        self.assertEqual([r["status"] for r in response["results"]], ["ok", "ok"])
        calls = [json.loads(l) for l in (self.dir / "boltz-calls.jsonl").read_text().splitlines()]
        self.assertEqual(len(calls), 2)
        spec = calls[0]["request"]
        self.assertEqual(len(spec["target"]["sequence"]), 219)                                    # the current 21-239 target
        self.assertEqual((spec["partner"]["msa"], spec["settings"], spec["partner"]["sequence"]), (None, {"seed": 5}, self.seqs[0]))
        resolved = (Path(calls[0]["script_dir"]).parent / spec["target"]["msa"]).resolve()       # how the team's script resolves it
        self.assertEqual(resolved, (PHASE1 / "target.a3m").resolve())
        score = response["results"][0]["score"]
        self.assertEqual((score["predictor"], score["iptm"], score["gpu"], score["plddt_raw_mean"]), ("boltz2", 0.8, "Stub GPU", None))
        self.assertEqual(self.members(response), sorted(f"b1/{i}/{n}" for i in self.ids[:2] for n in ("model.cif", "pae.json", "metrics.json")))
        self.assertTrue((self.cache / "runs/b1/env/boltz-checkpoints.json").is_file())             # what the team's script reads
        self.assertTrue((self.cache / "runs/b1/src/IPSAE/ipsae.py").is_file())

    def test_a_failing_boltz_candidate_is_recorded_with_the_scripts_error_and_the_rest_continue(self):
        os.environ["STUB_FAIL_SEQ"] = self.seqs[0]
        response = self.module.boltz_evaluate.local(self.request("boltz2", n=2, run_id="b2"))
        self.assertEqual([r["status"] for r in response["results"]], ["failed", "ok"])
        self.assertIn("exited 3", response["results"][0]["error"])
        self.assertIn("could not fold", response["results"][0]["error"])

    def test_boltz_refuses_unprepared_weights_and_the_wrong_predictor(self):
        (self.cache / "env/boltz-checkpoints.json").unlink()
        with self.assertRaisesRegex(ValueError, "modal_app.py --stage prepare"):
            self.module.boltz_evaluate.local(self.request("boltz2", run_id="b3"))
        with self.assertRaises(ValueError):
            self.module.boltz_evaluate.local(self.request("esmfold2", run_id="b4"))

    # ---- whole chain
    def candidates_file(self, flagged_last=True):
        records = [{"candidate_id": i, "sequence": s, "seed": k, "qc_reasons": [], "screen_status": "not_run",
                    "independent_prediction_status": "not_run", "novelty_review": "pending", "submission_eligible": False,
                    "target_id": "il7ra"} for k, (i, s) in enumerate(zip(self.ids, self.seqs), 1)]
        if flagged_last:
            records[-1]["qc_reasons"] = ["near_duplicate:exact_duplicate"]
        path = self.dir / "candidates.json"
        path.write_text(json.dumps({"candidates": records}))
        return path

    def raw_main(self, **kwargs):
        import types
        self.module.esmfold2_evaluate = types.SimpleNamespace(remote=self.module.esmfold2_evaluate.local)
        self.module.boltz_evaluate = types.SimpleNamespace(remote=self.module.boltz_evaluate.local)
        args = dict(stage="evaluate", candidates=str(self.candidates_file()), eval_id="ev1", predictors="esmfold2,boltz2",
                    chunk_size=2, max_calls=10, max_gpu_seconds=1e6, out=str(self.dir / "eval-out"))
        args.update(kwargs)
        return self.module.main.info.raw_f(**args)

    def test_the_whole_chain_from_candidates_file_to_attached_scores(self):
        self.raw_main()
        out = self.dir / "eval-out"
        report = json.loads((out / "evaluation-report.json").read_text())
        self.assertEqual((report["stop_reason"], report["candidates_evaluated"], report["calls"]), ("completed", 3, 4))
        self.assertEqual(report["by_predictor"]["esmfold2"], {"done": 3, "failed": 0, "started": 0, "not_started": 0})
        records = {r["candidate_id"]: r for r in json.loads((out / "evaluations.json").read_text())["candidates"]}
        for cid in self.ids[:3]:
            e = records[cid]["evaluations"]
            self.assertEqual((e["esmfold2"]["status"], e["boltz2"]["status"]), ("done", "done"))
            self.assertEqual(records[cid]["screen_status"], "not_run")                           # never promoted automatically
        self.assertEqual(records[self.ids[3]]["evaluation_skipped"], ["near_duplicate:exact_duplicate"])
        self.assertEqual(sum(c[0] == "from_pretrained" for c in self.esm_calls()), 2)             # one model load per call (2 + 1)
        self.assertTrue((out / "evals" / self.ids[0] / "esmfold2" / "model.cif").is_file())
        self.assertEqual(len((out / "scores.csv").read_text().strip().splitlines()), 7)           # header + 3 x 2

    def test_a_failure_in_the_container_ends_up_in_the_report_and_the_others_are_kept(self):
        os.environ["STUB_FAIL_SEQ"] = self.seqs[1]
        self.raw_main(max_failures=10)
        report = json.loads((self.dir / "eval-out/evaluation-report.json").read_text())
        self.assertEqual(sorted((f["candidate_id"], f["predictor"]) for f in report["failures"]),
                         sorted([(self.ids[1], "esmfold2"), (self.ids[1], "boltz2")]))
        self.assertEqual(report["by_predictor"]["esmfold2"]["done"], 2)

    def test_the_entry_point_refuses_bad_requests_before_launching_anything(self):
        import types
        launched = []
        self.module.esmfold2_evaluate = types.SimpleNamespace(remote=lambda r: launched.append(r))
        self.module.boltz_evaluate = types.SimpleNamespace(remote=lambda r: launched.append(r))
        raw = self.module.main.info.raw_f
        good = dict(candidates=str(self.candidates_file()), eval_id="x1", out=str(self.dir / "x"))
        for label, kwargs in {"no call budget": {"max_calls": 0, "max_gpu_seconds": 10.0},
                              "no gpu-second budget": {"max_calls": 2, "max_gpu_seconds": 0.0},
                              "other stage": {"stage": "qualify", "max_calls": 2, "max_gpu_seconds": 10.0},
                              "unknown predictor": {"predictors": "alphafold", "max_calls": 2, "max_gpu_seconds": 10.0},
                              "bad backend": {"kernel_backend": "turbo", "max_calls": 2, "max_gpu_seconds": 10.0}}.items():
            with self.assertRaises(ValueError, msg=label):
                raw(**{**good, **kwargs})
        only_flagged = self.dir / "flagged.json"
        only_flagged.write_text(json.dumps({"candidates": [{**json.loads(self.candidates_file().read_text())["candidates"][3]}]}))
        with self.assertRaises(ValueError):
            raw(**{**good, "candidates": str(only_flagged), "max_calls": 2, "max_gpu_seconds": 10.0})
        self.assertEqual(launched, [])


if __name__ == "__main__":
    unittest.main()
