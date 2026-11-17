import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign import esm_eval as E

try:
    import biotite  # noqa: F401
    HAVE_BIOTITE = True
except ImportError:
    HAVE_BIOTITE = False

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/eswar/esmfold2-checkpoint-manifest.json"
TEAM_REPORT = ROOT / "outputs/pipeline-execution-status.md"
TARGET = "ACDEFGHIKLMNPQRSTVWY" * 3          # 60 residues, any fixed target for the shape checks
BINDER = "MKTAYIAKQR" * 4                    # 40 residues


def make_files(directory, files):
    manifest = {"repo": E.REPO, "revision": E.REVISION, "esmc_backbone_bundled": True, "allow_patterns": ["*.json"], "files": {}}
    for name, data in files.items():
        (directory / name).write_bytes(data)
        manifest["files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return manifest


class ManifestFileTests(unittest.TestCase):
    @unittest.skipUnless(MANIFEST.is_file(), "needs docs/eswar")
    def test_the_committed_manifest_is_the_audited_checkpoint(self):
        manifest = E.load_manifest(MANIFEST)
        self.assertEqual(len(manifest["files"]), 11)                                  # the team's audit: 11 files
        self.assertEqual(sorted(n for n in manifest["files"] if n.endswith(".safetensors")),
                         [f"model-0000{i}-of-00006.safetensors" for i in range(1, 7)])
        self.assertTrue({"config.json", "ccd.pkl", "model.safetensors.index.json"} <= set(manifest["files"]))
        self.assertAlmostEqual(manifest["total_bytes"] / 1024 ** 3, 24.9, delta=0.05)  # the team's audit: about 24.9 GiB
        self.assertEqual(manifest["total_bytes"], sum(f["bytes"] for f in manifest["files"].values()))
        self.assertTrue(all(len(f["sha256"]) == 64 for f in manifest["files"].values()))

    @unittest.skipUnless(MANIFEST.is_file() and TEAM_REPORT.is_file(), "needs docs/eswar and the team's reports")
    def test_the_pinned_revision_is_the_one_the_team_audited(self):
        self.assertIn(E.REVISION, TEAM_REPORT.read_text())

    def test_loader_refuses_other_revisions_and_malformed_entries(self):
        with tempfile.TemporaryDirectory() as temp:
            def write(**changes):
                base = {"repo": E.REPO, "revision": E.REVISION, "esmc_backbone_bundled": True,
                        "files": {"a.json": {"bytes": 1, "sha256": "0" * 64}}}
                path = Path(temp) / "m.json"
                path.write_text(json.dumps({**base, **changes}))
                return path
            self.assertEqual(E.load_manifest(write())["revision"], E.REVISION)
            for changes in ({"revision": "0" * 40}, {"repo": "someone/else"}, {"esmc_backbone_bundled": False}, {"files": {}},
                            {"files": {"a.json": {"bytes": "1", "sha256": "0" * 64}}},
                            {"files": {"a.json": {"bytes": 1, "sha256": "short"}}}):
                with self.assertRaises(ValueError, msg=repr(changes)[:50]):
                    E.load_manifest(write(**changes))


class VerifyCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.manifest = make_files(self.dir, {"config.json": b'{"a": 1}', "weights.safetensors": b"x" * 5000})

    def test_a_correct_folder_passes_with_and_without_hashing(self):
        self.assertEqual(E.verify_checkpoint(self.dir, self.manifest), {"files": 2, "bytes": 5008, "hashed": True})
        self.assertFalse(E.verify_checkpoint(self.dir, self.manifest, hash_files=False)["hashed"])

    def test_extra_files_are_ignored(self):
        (self.dir / "README.md").write_text("not listed")
        self.assertEqual(E.verify_checkpoint(self.dir, self.manifest)["files"], 2)

    def test_each_kind_of_damage_is_reported(self):
        (self.dir / "weights.safetensors").write_bytes(b"y" * 5000)                  # same size, different content
        with self.assertRaisesRegex(ValueError, "hash mismatch: weights.safetensors"):
            E.verify_checkpoint(self.dir, self.manifest)
        E.verify_checkpoint(self.dir, self.manifest, hash_files=False)                # a size-only check cannot see it
        (self.dir / "weights.safetensors").write_bytes(b"y" * 4999)
        with self.assertRaisesRegex(ValueError, "wrong size: weights.safetensors"):
            E.verify_checkpoint(self.dir, self.manifest, hash_files=False)
        (self.dir / "config.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing or not a regular file: config.json"):
            E.verify_checkpoint(self.dir, self.manifest, hash_files=False)

    def test_all_problems_are_listed_together(self):
        (self.dir / "config.json").unlink()
        (self.dir / "weights.safetensors").write_bytes(b"short")
        with self.assertRaises(ValueError) as caught:
            E.verify_checkpoint(self.dir, self.manifest)
        self.assertIn("config.json", str(caught.exception))
        self.assertIn("weights.safetensors", str(caught.exception))

    def test_a_symlink_is_not_accepted_as_a_file(self):
        real = self.dir / "elsewhere.bin"
        real.write_bytes(b"x" * 5000)
        (self.dir / "weights.safetensors").unlink()
        (self.dir / "weights.safetensors").symlink_to(real)
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            E.verify_checkpoint(self.dir, self.manifest, hash_files=False)

    def test_the_record_has_the_shape_the_teams_scripts_read(self):
        record = E.checkpoint_record(self.dir, self.manifest)
        self.assertEqual(len(record), 1)
        self.assertEqual((record[0]["repo"], record[0]["revision"], record[0]["path"]), (E.REPO, E.REVISION, str(self.dir)))
        self.assertEqual(set(record[0]["files"]["config.json"]), {"sha256", "bytes"})


class RequestTests(unittest.TestCase):
    def test_request_matches_the_teams_screen_approximation(self):
        request = E.build_request(TARGET, BINDER, "/x/target.a3m", "c1", role="r")
        self.assertEqual(request["settings"], E.SCREEN_SETTINGS)
        self.assertEqual((request["settings"]["num_loops"], request["settings"]["num_sampling_steps"],
                          request["settings"]["num_diffusion_samples"], request["settings"]["msa_max_depth"]), (20, 200, 1, 1024))
        self.assertEqual((request["target"]["msa"], request["partner"]["msa"], request["partner"]["id"]), ("/x/target.a3m", None, "B"))
        self.assertIn("unconfirmed", request["protocol_status"])

    def test_the_shared_settings_are_never_modified(self):
        before = dict(E.SCREEN_SETTINGS)
        request = E.build_request(TARGET, BINDER, "m", "c", role="r", seed=9, kernel_backend=None)
        request["settings"]["num_loops"] = 1
        self.assertEqual(E.SCREEN_SETTINGS, before)
        self.assertEqual(request["settings"]["seed"], 9)

    def test_backends_seeds_and_sequences_are_checked(self):
        for backend in ("fused", "cuequivariance", None):
            E.build_request(TARGET, BINDER, "m", "c", role="r", kernel_backend=backend)
        for kwargs in ({"kernel_backend": "turbo"}, {"seed": -1}, {"seed": True}, {"seed": 1.5}):
            with self.assertRaises(ValueError, msg=repr(kwargs)):
                E.build_request(TARGET, BINDER, "m", "c", role="r", **kwargs)
        for bad in ("", "ACDX", "acd", None):
            with self.assertRaises(ValueError):
                E.build_request(bad, BINDER, "m", "c", role="r")
            with self.assertRaises(ValueError):
                E.build_request(TARGET, bad, "m", "c", role="r")

    def test_more_than_one_sample_is_refused(self):
        with self.assertRaises(ValueError):
            E.check_settings({**E.SCREEN_SETTINGS, "num_diffusion_samples": 5})


COLUMNS = ["group_PDB", "id", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id", "label_asym_id",
           "label_entity_id", "label_seq_id", "pdbx_PDB_ins_code", "Cartn_x", "Cartn_y", "Cartn_z", "occupancy",
           "B_iso_or_equiv", "auth_seq_id", "auth_comp_id", "auth_asym_id", "auth_atom_id", "pdbx_PDB_model_num"]


def full_cif(chains, nan_at=None):
    """CA-only mmCIF with every column biotite reads. chains: {id: (sequence, coords)}."""
    import _designs as D
    rows = ["data_test", "loop_"] + [f"_atom_site.{c}" for c in COLUMNS]
    n = 0
    for entity, (chain, (sequence, coords)) in enumerate(chains.items(), 1):
        for i, (aa, (x, y, z)) in enumerate(zip(sequence, coords), 1):
            n += 1
            xs = "nan" if nan_at == n else f"{x:.3f}"
            res = D.ONE_TO_THREE[aa]
            rows.append(f"ATOM {n} C CA . {res} {chain} {entity} {i} ? {xs} {y:.3f} {z:.3f} 1.00 50.00 {i} {res} {chain} CA 1")
    return "\n".join(rows) + "\n"


@unittest.skipUnless(HAVE_BIOTITE, "needs biotite")
class PredictionTests(unittest.TestCase):
    def cif(self, target=TARGET, binder=BINDER, order=("A", "B"), nan_at=None):
        rng = np.random.default_rng(0)
        chains = {"A": (target, np.cumsum(rng.normal(size=(len(target), 3)), axis=0) * 3.0),
                  "B": (binder, np.cumsum(rng.normal(size=(len(binder), 3)), axis=0) * 3.0)}
        return full_cif({k: chains[k] for k in order}, nan_at)

    def pae(self, n=len(TARGET) + len(BINDER)):
        return np.full((n, n), 5.0)

    def test_a_correct_prediction_passes(self):
        result = E.validate_prediction(self.pae(), self.cif(), TARGET, BINDER)
        self.assertEqual(result, {"chains": [{"id": "A", "length": 60}, {"id": "B", "length": 40}], "residues": 100})

    def test_each_kind_of_wrong_output_is_refused(self):
        bad_nan = self.pae()
        bad_nan[3, 4] = np.nan
        cases = {"pae too small": (self.pae(99), self.cif()), "pae too large": (self.pae(101), self.cif()),
                 "pae not square": (np.zeros((100, 99)), self.cif()), "pae has NaN": (bad_nan, self.cif()),
                 "chains swapped": (self.pae(), self.cif(order=("B", "A"))),
                 "different residues": (self.pae(), self.cif(binder="W" * 40)),
                 "one residue missing": (self.pae(), self.cif(binder=BINDER[:-1]))}
        for label, (pae, cif) in cases.items():
            with self.assertRaises(ValueError, msg=label):
                E.validate_prediction(pae, cif, TARGET, BINDER)

    def test_nonfinite_coordinates_are_refused_for_that_reason(self):
        with self.assertRaisesRegex(ValueError, "Invalid coordinates"):
            E.validate_prediction(self.pae(), self.cif(nan_at=7), TARGET, BINDER)

    def test_the_wrong_output_cases_fail_for_their_own_reasons(self):
        reasons = {"pae too small": (self.pae(99), self.cif(), "one finite token"),
                   "chains swapped": (self.pae(), self.cif(order=("B", "A")), "chain order or sequence"),
                   "one residue missing": (self.pae(), self.cif(binder=BINDER[:-1]), "residue counts differ")}
        for label, (pae, cif, message) in reasons.items():
            with self.assertRaisesRegex(ValueError, message, msg=label):
                E.validate_prediction(pae, cif, TARGET, BINDER)


class ConfidenceTests(unittest.TestCase):
    def test_finite_values_are_returned_as_floats(self):
        out = E.confidence_summary(np.array([0.5, 0.7]), 0.8, np.float32(0.9))
        self.assertAlmostEqual(out["plddt_raw_mean"], 0.6)
        self.assertEqual((out["ptm"], round(out["iptm"], 2)), (0.8, 0.9))

    def test_nonfinite_values_are_refused(self):
        for args in ((np.array([np.nan]), 0.5, 0.5), (np.array([0.5]), float("inf"), 0.5), (np.array([0.5]), 0.5, float("nan"))):
            with self.assertRaises(ValueError):
                E.confidence_summary(*args)


if __name__ == "__main__":
    unittest.main()
