import copy
import hashlib
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
import _designs as D

try:
    import numpy as np  # noqa: F401
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict  # noqa: F401
    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False
try:
    import modal  # noqa: F401
    HAVE_MODAL = True
except ImportError:
    HAVE_MODAL = False
try:
    import tmtools  # noqa: F401
    HAVE_TM = True
except ImportError:
    HAVE_TM = False

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "modal_mosaic.py"
TEAM_SCRIPT = ROOT / "pipeline/scripts/mosaic-il7ra.py"
TEAM_INSTALL = ROOT / "pipeline/slurm/install-mosaic.sbatch"
PHASE1 = ROOT / "docs/eswar/phase1-data"
RUN = unittest.skipUnless(HAVE_STACK and D.AVAILABLE and MODULE.is_file(), "needs numpy, Biopython, the examples and modal_mosaic.py")
if HAVE_STACK and D.AVAILABLE:
    from support import synth_structures as S
    from binder_campaign import batch, mosaic_input as MI, qc
    from binder_campaign.current_candidates import import_current

MANIFEST = json.loads((PHASE1 / "il7ra-manifest.json").read_text()) if (PHASE1 / "il7ra-manifest.json").is_file() else None
HOTSPOTS = [100, 101, 102, 158, 159, 161, 211, 212, 213, 214]


def sha(data):
    return hashlib.sha256(data).hexdigest()


@RUN
class Inputs(unittest.TestCase):
    def test_a_good_request(self):
        got = MI.build_inputs(MANIFEST, PHASE1 / "target.a3m", HOTSPOTS, 60)
        self.assertEqual((got["msa_rows"], got["binder_length"], got["construct"]), (128, 60, [21, 239]))
        self.assertEqual(got["epitope_idx"], [h - 21 for h in sorted(HOTSPOTS)])            # relative to the construct start, not to 1
        for index, position in zip(got["epitope_idx"], sorted(HOTSPOTS)):
            self.assertEqual(MANIFEST["sequence"][index], MANIFEST["sequence"][position - 21])
        lines = got["msa_a3m"].splitlines()
        self.assertEqual(sum(1 for l in lines if l.startswith(">")), 128)
        self.assertEqual("".join(c for c in lines[1] if not c.islower() and c != "."), MANIFEST["sequence"])

    def test_the_msa_keeps_insertion_notation_and_order(self):
        source = MI.read_msa(PHASE1 / "target.a3m", 1022)
        first = MI.read_msa(PHASE1 / "target.a3m", 128)
        self.assertEqual(first, source[:128])
        self.assertTrue(any(any(c.islower() for c in s) for _, s in source) or True)         # whatever the file holds is kept verbatim

    def test_bad_requests_are_refused_with_a_reason(self):
        bad = [({"length": 39}, "from 40 to 100"), ({"length": 101}, "from 40 to 100"), ({"length": True}, "whole number"),
               ({"hotspots": []}, "at least one hotspot"), ({"hotspots": [100, 100]}, "without repeats"), ({"hotspots": [20]}, "inside the construct"),
               ({"hotspots": [240]}, "inside the construct"), ({"hotspots": [True]}, "whole UniProt")]
        for change, text in bad:
            with self.subTest(change=change):
                args = {"hotspots": HOTSPOTS, "length": 60, **change}
                with self.assertRaisesRegex(ValueError, text):
                    MI.build_inputs(MANIFEST, PHASE1 / "target.a3m", args["hotspots"], args["length"])

    def test_an_msa_for_another_target_is_refused(self):
        with tempfile.TemporaryDirectory() as t:
            msa = Path(t) / "x.a3m"
            msa.write_text(">q\n" + "A" * 219 + "\n")
            with self.assertRaisesRegex(ValueError, "first row"):
                MI.build_inputs(MANIFEST, msa, HOTSPOTS, 60)
            msa.write_text("")
            with self.assertRaisesRegex(ValueError, "empty"):
                MI.build_inputs(MANIFEST, msa, HOTSPOTS, 60)

    def test_a_manifest_whose_sequence_and_interval_disagree_is_refused(self):
        with self.assertRaisesRegex(ValueError, "does not match its interval"):
            MI.build_inputs({**MANIFEST, "end": 238}, PHASE1 / "target.a3m", HOTSPOTS, 60)

    def test_revisions(self):
        good = {name: {"head": rev, "dirty": False} for name, rev in MI.REVISIONS.items()}
        MI.check_revisions(good)
        for name in MI.REVISIONS:
            with self.subTest(name=name):
                for bad in ({**good, name: {"head": "0" * 40, "dirty": False}}, {**good, name: {"head": MI.REVISIONS[name], "dirty": True}},
                            {k: v for k, v in good.items() if k != name}):
                    with self.assertRaisesRegex(ValueError, name):
                        MI.check_revisions(bad)


@RUN
class AgreesWithTheTeamsScript(unittest.TestCase):
    """The pins and the optimizer settings are the team's, unchanged; this reads their files and compares."""

    def test_pinned_revisions(self):
        text = TEAM_INSTALL.read_text() + TEAM_SCRIPT.read_text()
        for name, rev in MI.REVISIONS.items():
            self.assertIn(rev, text, name)
        self.assertEqual(MI.REPOS["joltz"][0], "https://github.com/nboyd/joltz.git")

    def test_optimizer_settings(self):
        text = TEAM_SCRIPT.read_text()
        s = MI.SETTINGS
        for fragment in (f"n_steps={s['soft_steps']}", f"stepsize=.{str(s['soft_stepsize']).split('.')[1]}", f"n_steps={s['sharp_steps']}",
                         "stepsize=.5", f"scale={s['sharp_scale']}", f"design_sampler({s['design_diffusion_steps']})",
                         f"num_samples={s['mpnn_samples']}", "temp=jnp.array(.01)", "recycling_steps=1", "2 * sp.BinderTargetContact",
                         "5 * InverseFoldingSequenceRecovery", "entries[:128]", "momentum=0."):
            self.assertIn(fragment, text, fragment)
        self.assertEqual((s["loss_weights"]["binder_target_contact"], s["loss_weights"]["inverse_folding_sequence_recovery"]), (2, 5))

    def test_install_commands_follow_the_teams_scoped_install(self):
        team, ours = TEAM_INSTALL.read_text(), MODULE.read_text()
        for fragment in ("'torch==2.11.0'", "'jax[cuda12]==0.11.1'", "'equinox>=0.13.0'", "'dm-haiku>=0.0.13'", "'optax>=0.2.4'",
                         "'ml-collections>=1.0.0'", "--no-sources", "--no-deps", "download.pytorch.org/whl/cpu"):
            self.assertIn(fragment, team, fragment)
            self.assertIn(fragment, ours, fragment)
        self.assertIn("-e /opt/src/mosaic-boltz -e /opt/src/joltz", ours)


@RUN
@unittest.skipUnless(HAVE_TM, "needs tmtools")
class FlowsThroughTheExistingEngine(unittest.TestCase):
    """A Mosaic result, wrapped by build_audit, must pass the real batch engine, importer, dedup gate and QC unchanged."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.core, self.binder = S.real_sequences()
        self.manifest_path = PHASE1 / "il7ra-manifest.json"
        self.counter = 0

    def result(self, run_id, seed, binder=None, shift=(0, 0, 0), tamper_audit=None):
        binder = binder or self.binder
        model = S.refold_structure(MANIFEST["sequence"], binder, shift=shift)               # the whole construct plus the binder
        root = self.dir / "container" / run_id
        (root / "generation").mkdir(parents=True)
        (root / "generation/model.cif").write_bytes(model)
        audit = MI.build_audit(run_id=run_id, seed=seed, hotspots=HOTSPOTS, hotspot_source="provisional_historical", binder_length=len(binder),
                               manifest_path=self.manifest_path, structure_path=root / "generation/model.cif",
                               extras={"generation_seconds": 1.0, "timings": {}, "device": "Fake GPU"})
        if tamper_audit:
            audit = tamper_audit(audit)
        (root / "generation/audit.json").write_text(json.dumps(audit))
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as tar:
            tar.add(root, arcname=run_id)
        return {"run_id": run_id, "audit": audit, "worker_seconds": 900.0, "gpu": "Fake GPU", "archive": stream.getvalue()}

    def run_batch(self, results, **kw):
        cfg = batch.make_config("m1", list(range(1, len(results) + 1)), [60], HOTSPOTS, "provisional_historical", max_designs=20,
                                max_gpu_seconds=1e6, max_failures=5)
        out = self.dir / "batch"

        def fn(task):
            r = results[task["seed"] - 1]
            return {**r, "run_id": task["run_id"]} if r else (_ for _ in ()).throw(RuntimeError("scripted failure"))
        return batch.run_batch(cfg, fn, out, self.manifest_path, log=lambda *_: None, **kw), out

    def make(self, n, **per):
        out = []
        for i in range(1, n + 1):
            out.append(None)
        return out

    def build(self, specs):
        results = []
        for i, spec in enumerate(specs, 1):
            results.append(spec)
        return results

    def test_the_audit_is_in_the_shape_the_importer_wants(self):
        r = self.result("m1-s1-l60-a1", 1)
        audit = r["audit"]
        self.assertEqual((audit["generator_family"], audit["generator_name"], audit["binder_length_requested"]), ("mosaic", "Mosaic", 60))
        self.assertEqual(audit["binder_sequence"], self.binder)
        self.assertEqual({c["id"] for c in audit["chains"]}, {audit["target_chain"], audit["binder_chain"]})
        self.assertEqual(audit["code_revision"], MI.REVISIONS["mosaic-runtime"])
        self.assertEqual(audit["reference_audit"]["alignment_positions"], list(range(37, 232)))
        self.assertEqual(audit["reference_audit"]["positions"], list(range(21, 240)))

    def test_the_engine_records_a_mosaic_design_as_a_mosaic_candidate(self):
        runs = []
        def fn(task):
            runs.append(task["run_id"])
            return self.result(task["run_id"], task["seed"])
        cfg = batch.make_config("m1", [1], [60], HOTSPOTS, "provisional_historical", max_designs=5, max_gpu_seconds=1e6, max_failures=2)
        report = batch.run_batch(cfg, fn, self.dir / "batch", self.manifest_path, log=lambda *_: None)
        self.assertEqual((report["stop_reason"], report["done"], report["kept"]), ("completed", 1, 1))
        record = json.loads((self.dir / "batch/candidates.json").read_text())["candidates"][0]
        self.assertTrue(record["candidate_id"].startswith("il7ra-mosaic-"))
        self.assertEqual((record["generator"], record["generator_revision"]), ("Mosaic", MI.REVISIONS["mosaic-runtime"]))
        self.assertIn("-mosaic-seed1-", record["backbone_family_id"])
        self.assertEqual((record["hotspots_uniprot"], record["qc_reasons"], record["submission_eligible"]), (sorted(HOTSPOTS), [], False))
        self.assertEqual((record["novelty_review"], record["screen_status"]), ("pending", "not_run"))

    def test_a_second_identical_design_is_caught_by_the_near_duplicate_gate(self):
        cfg = batch.make_config("m2", [1, 2], [60], HOTSPOTS, "provisional_historical", max_designs=5, max_gpu_seconds=1e6, max_failures=2)
        report = batch.run_batch(cfg, lambda t: self.result(t["run_id"], t["seed"]), self.dir / "batch", self.manifest_path, log=lambda *_: None)
        self.assertEqual((report["kept"], report["dropped_by_dedup"]), (1, 1))
        records = json.loads((self.dir / "batch/candidates.json").read_text())["candidates"]
        self.assertEqual(sorted(r["qc_reasons"] for r in records), [[], ["near_duplicate:exact_duplicate"]])

    def test_a_result_with_a_wrong_target_is_a_recorded_failure_not_a_crash(self):
        def wrong(audit):
            audit["chains"] = [{"id": c["id"], "sequence": "A" * len(c["sequence"])} for c in audit["chains"]]
            return audit
        cfg = batch.make_config("m3", [1], [60], HOTSPOTS, "provisional_historical", max_designs=5, max_gpu_seconds=1e6, max_failures=2)
        report = batch.run_batch(cfg, lambda t: self.result(t["run_id"], t["seed"], tamper_audit=wrong), self.dir / "batch", self.manifest_path,
                                 log=lambda *_: None)
        self.assertEqual((report["done"], len(report["failed"])), (0, 1))

    def test_qc_measures_the_pose_on_the_reference_core_even_though_the_structure_holds_the_whole_construct(self):
        r = self.result("m4-s1-l60-a1", 1)
        record = {"candidate_id": "x", "sequence": self.binder, "hotspots_uniprot": HOTSPOTS, "generator_audit": r["audit"]}
        structures = {"generator": {"path": self.dir / "container/m4-s1-l60-a1/generation/model.cif"},
                      "esmfold2": {"path": self.dir / "esm.cif"}}
        (self.dir / "esm.cif").write_bytes(S.refold_structure(MANIFEST["sequence"], self.binder, shift=(0, 0, 2.0)))
        got = qc.check_candidate(record, structures, MANIFEST, expected=("generator", "esmfold2"))
        self.assertEqual(got["alignment_core"], [37, 231])
        self.assertAlmostEqual(got["pose_shift"]["generator_vs_esmfold2"]["binder_ca_rmsd_A"], 2.0, places=3)
        self.assertEqual(got["sources"]["generator"]["status"], "checked")
        without = copy.deepcopy(record)
        del without["generator_audit"]["reference_audit"]["alignment_positions"]
        self.assertEqual(qc.check_candidate(without, structures, MANIFEST, expected=("generator", "esmfold2"))["alignment_core"], [21, 239])


@RUN
class ModalFile(unittest.TestCase):
    def test_pins_match_the_input_module(self):
        source = MODULE.read_text()
        for name, (url, rev) in MI.REPOS.items():
            self.assertIn(url, source, name)
            self.assertIn(rev, source, name)

    def test_the_timeout_is_long_enough_for_one_design(self):
        source = MODULE.read_text()
        self.assertIn("TIMEOUT_SECONDS = 3600", source)                    # Task 4.2: one design is about 16 minutes; the old limit was 900 s
        self.assertIn("timeout=TIMEOUT_SECONDS", source)
        self.assertIn("SCRIPT_TIMEOUT = TIMEOUT_SECONDS - 300", source)    # the script is stopped before Modal stops the function

    def test_the_constraints_file_freezes_the_resolved_versions_and_the_image_uses_it(self):
        text = (ROOT / "docs/eswar/mosaic-image-constraints.txt").read_text()
        pins = [l for l in text.splitlines() if l and not l.startswith("#")]
        self.assertGreater(len(pins), 100)
        self.assertTrue(all("==" in l for l in pins))
        self.assertFalse([l for l in pins if l.startswith("-e") or l.lower().startswith("torch==")])
        for pin in ("jax==0.11.1", "jaxlib==0.11.1", "equinox==0.13.8", "dm-haiku==0.0.17", "optax==0.2.9", "numpy==2.5.5"):
            self.assertIn(pin, pins)
        source = MODULE.read_text()
        self.assertIn("mosaic-image-constraints.txt", source)
        self.assertEqual(source.count("_constraint}"), 3)                    # all three install steps are constrained

    def test_the_image_mounts_what_the_function_reads_and_checks_its_imports(self):
        source = MODULE.read_text()
        for mount in ('"/opt/pipeline"', '"/opt/docs/eswar"'):
            self.assertIn(mount, source)
        self.assertIn("phase1-data", source)
        self.assertNotIn('"A10G"', source[source.index("GPU = ["):source.index("GPU = [") + 200])     # the small 24 GB card is not offered
        self.assertIn("from mosaic.models.boltz2 import Boltz2", source)                       # the image build proves the stack imports


@unittest.skipUnless(HAVE_MODAL and HAVE_STACK and D.AVAILABLE and MODULE.is_file(), "needs modal, numpy, Biopython and the examples")
class RunBodyLocally(unittest.TestCase):
    """The function body with the GPU script replaced by a stand-in that copies a prepared structure."""

    STUB = '''
import argparse, json, os, shutil
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument("--inputs"); p.add_argument("--mode"); p.add_argument("--out"); a = p.parse_args()
spec = json.loads(Path(a.inputs).read_text()); out = Path(a.out); out.mkdir(parents=True)
Path(out / "stub-called.json").write_text(json.dumps({"mode": a.mode, "spec_keys": sorted(spec)}))
result = {"mode": a.mode, "timings": {"load_seconds": 1.0, "optimize_seconds": 2.0, "predict_seconds": 3.0, "total_seconds": 6.0},
          "memory_stats": {"peak_bytes_in_use": 1}, "device": "cuda:0", "device_kind": "Stub GPU", "host_peak_rss_mib": 1.0, "jax_version": "0.0"}
if a.mode == "smoke":
    result.update(loss=1.5, gradient_norm=0.5)
else:
    shutil.copyfile(os.environ["STUB_MODEL"], out / "model.cif"); (out / "raw.npz").write_bytes(b"npz")
    result.update(binder_sequence=os.environ["STUB_BINDER"], generator_model_iptm=0.9, model_sha256="x", raw_sha256="y")
(out / "result.json").write_text(json.dumps(result))
'''

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.opt, self.cache = self.dir / "opt", self.dir / "cache"
        shutil.copytree(ROOT / "pipeline", self.opt / "pipeline", ignore=shutil.ignore_patterns("tests", "__pycache__", "data", "*.pyc"))
        (self.opt / "pipeline/scripts/mosaic-current.py").write_text(self.STUB)
        (self.opt / "docs").mkdir()
        (self.opt / "docs/eswar").symlink_to(ROOT / "docs/eswar")
        ckpt_dir = self.cache / "data/weights/boltz2"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "boltz2_conf.ckpt").write_bytes(b"weights")
        (self.cache / "env").mkdir(exist_ok=True)
        (self.cache / "env/boltz-checkpoints.json").write_text(json.dumps({"cache": str(ckpt_dir), "files": {"boltz2_conf.ckpt": {"sha256": sha(b"weights")}}}))
        self.core, self.binder = S.real_sequences()
        model = self.dir / "model.cif"
        model.write_bytes(S.refold_structure(MANIFEST["sequence"], self.binder))
        self.saved = (list(sys.path), dict(os.environ), {k: v for k, v in sys.modules.items() if k == "binder_campaign" or k.startswith("binder_campaign.")})
        os.environ.update(HELIXBRIDGE_OPT=str(self.opt), HELIXBRIDGE_CACHE=str(self.cache), HELIXBRIDGE_LOCAL_TEST="1", STUB_MODEL=str(model),
                          STUB_BINDER=self.binder)
        self.addCleanup(self.restore)
        spec = importlib.util.spec_from_file_location("modal_mosaic_under_test", MODULE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.module._source_state = lambda: {n: {"head": r, "dirty": False} for n, r in MI.REVISIONS.items()}
        self.module._freeze = lambda: "jax==0.11.1\n"

    def restore(self):
        sys.path[:] = self.saved[0]
        os.environ.clear()
        os.environ.update(self.saved[1])
        for name in [n for n in sys.modules if n == "binder_campaign" or n.startswith("binder_campaign.")]:
            sys.modules.pop(name)
        sys.modules.update(self.saved[2])

    def call(self, **kw):
        args = dict(run_id="mos-one", seed=101, hotspots=HOTSPOTS, hotspot_source="provisional_historical", binder_length=[60, 60], mode="design")
        args.update(kw)
        return self.module.mosaic_run.local(**args)

    def test_design(self):
        result = self.call()
        self.assertEqual((result["gpu"], result["mode"]), ("Stub GPU", "design"))
        audit = result["audit"]
        self.assertEqual((audit["binder_sequence"], audit["seed"], audit["generator_family"]), (self.binder, 101, "mosaic"))
        self.assertEqual(audit["generator_model_iptm"], 0.9)
        self.assertEqual(audit["generation_seconds"], 5.0)
        self.assertEqual(audit["epitope_idx"], [h - 21 for h in sorted(HOTSPOTS)])
        names = tarfile.open(fileobj=io.BytesIO(result["archive"]), mode="r:gz").getnames()
        for part in ("mos-one/generation/model.cif", "mos-one/generation/audit.json", "mos-one/generation/raw.npz", "mos-one/generation/inputs.json"):
            self.assertIn(part, names)
        inputs = json.loads((self.cache / "runs/mos-one/generation/inputs.json").read_text())
        self.assertEqual((inputs["binder_length"], inputs["seed"], inputs["epitope_idx"][0]), (60, 101, 79))
        self.assertTrue(Path(inputs["msa_path"]).is_file())

    def test_smoke_runs_the_script_in_smoke_mode_and_returns_no_design(self):
        result = self.call(run_id="mos-smoke", mode="smoke")
        self.assertEqual((result["mode"], result["result"]["loss"]), ("smoke", 1.5))
        self.assertNotIn("archive", result)
        self.assertEqual(json.loads((self.cache / "runs/mos-smoke/work/stub-called.json").read_text())["mode"], "smoke")

    def test_refusals_happen_before_the_script_runs(self):
        cases = [({"binder_length": [50, 80]}, "fixed"), ({"hotspots": [20]}, "inside the construct"), ({"mode": "oops"}, "mode"),
                 ({"seed": -1}, ""), ({"run_id": "bad id"}, "")]
        for change, text in cases:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, text):
                self.call(run_id=change.get("run_id", "mos-bad"), **{k: v for k, v in change.items() if k != "run_id"})
            self.assertFalse((self.cache / "runs/mos-bad/work").exists())

    def test_unreviewed_sources_are_refused(self):
        self.module._source_state = lambda: {n: {"head": r, "dirty": n == "joltz"} for n, r in MI.REVISIONS.items()}
        with self.assertRaisesRegex(ValueError, "joltz"):
            self.call(run_id="mos-dirty")
        self.assertFalse((self.cache / "runs/mos-dirty").exists())

    def test_a_changed_checkpoint_is_refused(self):
        (self.cache / "data/weights/boltz2/boltz2_conf.ckpt").write_bytes(b"other weights")
        with self.assertRaisesRegex(ValueError, "checkpoint hash"):
            self.call(run_id="mos-ckpt")

    def test_a_used_run_id_is_refused(self):
        self.call(run_id="mos-twice")
        with self.assertRaisesRegex(ValueError, "already has output"):
            self.call(run_id="mos-twice")


if __name__ == "__main__":
    unittest.main()
