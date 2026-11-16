import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.conditioning import CONDITIONING_ROLE, write_conditioning
from binder_campaign.generation_input import (build_overrides, check_generated_chains, check_seed, length_allowed,
                                              load_reference, read_reference_ca)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/il7ra-rehearsal"
CURRENT = ROOT / "docs/eswar/phase1-data/il7ra-manifest.json"
HISTORICAL = EXAMPLE / "historical-target.json"
HOTSPOTS = [100, 101, 102, 158, 159, 161, 211, 212, 213, 214]
TARGET = "ACDEFGHIKL" * 19  # any fixed target sequence for the chain-level checks
BINDER = "MKTAYIAKQR" * 6


class SeedTests(unittest.TestCase):
    def test_historical_whitelist_is_gone(self):
        for ok in (0, 1, 42, 43, 44, 999, 123456):
            self.assertEqual(check_seed(ok), ok)

    def test_bad_seeds_are_rejected(self):
        for bad in (-1, True, False, 1.5, "43", None, [43]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                check_seed(bad)


class LengthTests(unittest.TestCase):
    def test_exact_and_range(self):
        self.assertTrue(length_allowed(60, 60))
        self.assertFalse(length_allowed(59, 60))
        for n, expected in ((49, False), (50, True), (80, True), (81, False)):
            self.assertEqual(length_allowed(n, [50, 80]), expected)

    def test_malformed_spec_is_an_error(self):
        for bad in ("60", None, [50], [50, 60, 70], [50.0, 60], True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                length_allowed(60, bad)


class ChainTests(unittest.TestCase):
    def chains(self, *sequences):
        return [{"id": chr(65 + i), "sequence": s} for i, s in enumerate(sequences)]

    def test_valid_pair_in_either_order(self):
        for chains in (self.chains(TARGET, BINDER), self.chains(BINDER, TARGET)):
            target, binder = check_generated_chains(chains, TARGET, 60)
            self.assertEqual((target["sequence"], binder["sequence"]), (TARGET, BINDER))

    def test_range_boundaries(self):
        check_generated_chains(self.chains(TARGET, BINDER[:50]), TARGET, [50, 80])
        with self.assertRaises(ValueError):
            check_generated_chains(self.chains(TARGET, BINDER[:49]), TARGET, [50, 80])

    def test_structural_failures(self):
        cases = {"three chains": self.chains(TARGET, BINDER, BINDER),
                 "one chain": self.chains(TARGET),
                 "no target chain": self.chains(BINDER, BINDER[::-1]),
                 "two target chains": self.chains(TARGET, TARGET),
                 "wrong length": self.chains(TARGET, BINDER[:30]),
                 "invalid residue": self.chains(TARGET, "X" * 60),
                 "empty binder": self.chains(TARGET, "")}
        for name, chains in cases.items():
            with self.assertRaises(ValueError, msg=name):
                check_generated_chains(chains, TARGET, 60)


@unittest.skipUnless(EXAMPLE.is_dir() and CURRENT.is_file(), "needs examples/ and docs/eswar/phase1-data")
class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)

    def conditioned(self, name="cond", **kwargs):
        out = self.dir / name
        write_conditioning(EXAMPLE, out, HOTSPOTS, 60, hotspot_source="provisional_historical")
        return out

    def copy_reference(self):
        out = self.dir / "ref"
        shutil.copytree(EXAMPLE, out, ignore=shutil.ignore_patterns("*.a3m", "summary.json", "historical-target.json"))
        return out

    def test_historical_reference_with_historical_manifest(self):
        ref = load_reference(EXAMPLE, HISTORICAL)
        self.assertEqual(ref["role"], "reference_rehearsal_only")
        self.assertEqual(len(ref["target_sequence"]), 195)

    def test_phase1_conditioning_with_the_current_manifest(self):
        ref = load_reference(self.conditioned(), CURRENT)
        self.assertEqual(ref["role"], CONDITIONING_ROLE)
        self.assertEqual(ref["binder_length"], 60)
        self.assertEqual(ref["hotspots_uniprot"], HOTSPOTS)
        self.assertEqual(ref["target_sequence"], json.loads(CURRENT.read_text())["sequence"][16:211])

    def test_unrecognized_role_is_rejected(self):
        ref = self.copy_reference()
        audit = json.loads((ref / "audit.json").read_text())
        audit["role"] = "something_else"
        (ref / "audit.json").write_text(json.dumps(audit))
        with self.assertRaises(ValueError):
            load_reference(ref, CURRENT)

    def test_changed_coordinates_file_is_rejected(self):
        ref = self.conditioned()
        (ref / "target.cif").write_bytes((ref / "target.cif").read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            load_reference(ref, CURRENT)

    def test_manifest_for_another_target_is_rejected(self):
        manifest = json.loads(CURRENT.read_text())
        manifest["id"] = "tl1a"
        other = self.dir / "other.json"
        other.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            load_reference(self.conditioned(), other)

    def test_manifest_sequence_that_disagrees_is_rejected(self):
        manifest = json.loads(CURRENT.read_text())
        i = 66 - manifest["start"]
        manifest["sequence"] = manifest["sequence"][:i] + ("T" if manifest["sequence"][i] != "T" else "A") + manifest["sequence"][i + 1:]
        other = self.dir / "other.json"
        other.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            load_reference(self.conditioned(), other)

    def test_interval_that_excludes_the_reference_is_rejected(self):
        manifest = json.loads(CURRENT.read_text())
        manifest.update(start=40, end=239, sequence=manifest["sequence"][19:])
        other = self.dir / "other.json"
        other.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            load_reference(self.conditioned(), other)

    def test_hotspot_not_in_the_reference_is_rejected(self):
        ref = self.copy_reference()
        audit = json.loads((ref / "audit.json").read_text())
        audit["hotspots_uniprot"] = audit["hotspots_uniprot"] + [25]
        (ref / "audit.json").write_text(json.dumps(audit))
        with self.assertRaises(ValueError):
            load_reference(ref, CURRENT)

    def test_reference_reader_returns_the_audited_run(self):
        positions, sequence = read_reference_ca(EXAMPLE / "target.cif")
        self.assertEqual((positions[0], positions[-1], len(positions)), (37, 231, 195))
        self.assertEqual(sequence, json.loads((EXAMPLE / "audit.json").read_text())["reference_sequence"])


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
RECORD = {"path": "/ckpt", "revision": "abc123"}


class OverrideTests(unittest.TestCase):
    def test_settings_for_one_design(self):
        overrides = build_overrides("/in/conditions.json", "/out", 7, RECORD)
        for expected in ("inputs=/in/conditions.json", "out_dir=/out", "hf_ckpt_path=/ckpt", "hf_ckpt_revision=abc123",
                         "seed=7", "num_replicates=1", "batch_size=1", "recycle_strategy=thinking_high",
                         "prevalidate_inputs=true", "skip_existing=false"):
            self.assertIn(expected, overrides)

    def test_bad_seed_is_refused_here_too(self):
        with self.assertRaises(ValueError):
            build_overrides("/in", "/out", -1, RECORD)

    @unittest.skipUnless((SCRIPTS / "rfd4-il7ra-seed.py").is_file(), "needs the original qualified script")
    def test_recipe_matches_the_original_qualified_script(self):
        tree = ast.parse((SCRIPTS / "rfd4-il7ra-seed.py").read_text())
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "generate")
        items = next(k.value for k in call.keywords if k.arg == "overrides").elts
        original = {}
        for item in items:
            text = item.value if isinstance(item, ast.Constant) else item.values[0].value
            key, _, value = text.partition("=")
            original[key] = value if isinstance(item, ast.Constant) else None  # None = value filled in at run time
        ours = {}
        for item in build_overrides("/in", "/out", 1, RECORD):
            key, _, value = item.partition("=")
            ours[key] = value
        self.assertEqual(set(ours), set(original))
        for key, value in original.items():
            if value is not None:
                self.assertEqual(ours[key], value, key)


@unittest.skipUnless(EXAMPLE.is_dir() and CURRENT.is_file(), "needs examples/ and docs/eswar/phase1-data")
class DryRunScriptTests(unittest.TestCase):
    def run_script(self, reference_dir, out, seed, manifest=CURRENT):
        return subprocess.run([sys.executable, str(SCRIPTS / "rfd4-generate.py"), "--dry-run", "--reference-dir",
                               str(reference_dir), "--manifest", str(manifest), "--out", str(out), "--seed", str(seed)],
                              capture_output=True, text=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.ref = self.dir / "cond"
        write_conditioning(EXAMPLE, self.ref, HOTSPOTS, 60, hotspot_source="provisional_historical")

    def test_dry_run_reports_the_plan_for_a_seed_the_old_script_refused(self):
        done = self.run_script(self.ref, self.dir / "out", 7)
        self.assertEqual(done.returncode, 0, done.stderr)
        plan = json.loads(done.stdout)
        self.assertEqual((plan["seed"], plan["binder_length"], plan["target_residues"]), (7, 60, 195))
        self.assertEqual(plan["hotspot_source"], "provisional_historical")
        self.assertIn("seed=7", plan["overrides"])
        self.assertFalse((self.dir / "out").exists())  # a dry run writes nothing

    def test_dry_run_refuses_bad_input(self):
        self.assertNotEqual(self.run_script(self.ref, self.dir / "o1", -3).returncode, 0)
        (self.ref / "target.cif").write_bytes((self.ref / "target.cif").read_bytes() + b"\n")
        self.assertNotEqual(self.run_script(self.ref, self.dir / "o2", 7).returncode, 0)

    def test_dry_run_refuses_a_used_output_directory(self):
        used = self.dir / "used"
        used.mkdir()
        (used / "x").write_text("x")
        self.assertNotEqual(self.run_script(self.ref, used, 7).returncode, 0)


if __name__ == "__main__":
    unittest.main()
