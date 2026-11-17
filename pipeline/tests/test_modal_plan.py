import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.modal_plan import build_plan, check_run_id, local_dry_run, resolve_hotspots, resolve_length

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/plan-modal-run.py"
HAVE_FILES = ((ROOT / "docs/eswar/hotspot-options/A/audit.json").is_file() and (ROOT / "examples/il7ra-rehearsal").is_dir()
              and (ROOT / "docs/eswar/phase1-data/il7ra-manifest.json").is_file())
A = [100, 101, 102, 158, 159, 161, 211, 212, 213, 214]
B = [78, 79, 98, 100, 102, 103, 158, 159, 211, 212]
C = [170, 172, 173, 174, 175, 176, 177, 178, 179, 200]


class RunIdAndLengthTests(unittest.TestCase):
    def test_run_ids(self):
        for ok in ("eswar-cur-001", "a", "A_1", "x" * 64):
            self.assertEqual(check_run_id(ok), ok)
        for bad in ("", "-start", "has space", "../escape", "x" * 65, "semi;colon", None, 5):
            with self.assertRaises(ValueError, msg=repr(bad)):
                check_run_id(bad)

    def test_exact_and_range_lengths(self):
        self.assertEqual(resolve_length(60), [60, 60])
        self.assertEqual(resolve_length(length_min=50, length_max=80), [50, 80])
        self.assertEqual(resolve_length(40), [40, 40])
        self.assertEqual(resolve_length(100), [100, 100])

    def test_bad_lengths(self):
        for args in ({"length": 39}, {"length": 101}, {"length_min": 50}, {"length_max": 80},
                     {"length_min": 80, "length_max": 50}, {"length_min": 30, "length_max": 60},
                     {"length_min": 60, "length_max": 120}, {"length": True}, {"length": 60.5}):
            with self.assertRaises(ValueError, msg=repr(args)):
                resolve_length(**args)


@unittest.skipUnless(HAVE_FILES, "needs docs/eswar and examples/")
class PlanTests(unittest.TestCase):
    def test_default_is_the_provisional_set(self):
        plan = build_plan(ROOT)
        self.assertEqual((plan["hotspots"], plan["hotspot_source"], plan["seed"], plan["binder_length"]),
                         (A, "provisional_historical", 43, [60, 60]))

    def test_options_b_and_c(self):
        self.assertEqual(build_plan(ROOT, option="b")["hotspots"], B)
        plan = build_plan(ROOT, option="C")
        self.assertEqual(plan["hotspots"], C)
        self.assertTrue(plan["hotspot_source"].startswith("proposed"))

    def test_custom_hotspots_need_a_source_and_exclude_an_option(self):
        plan = build_plan(ROOT, hotspots="102, 100,101", hotspot_source="reviewed_by_someone")
        self.assertEqual(plan["hotspots"], [100, 101, 102])
        for kwargs in ({"hotspots": "100,101"}, {"hotspots": "100", "option": "A", "hotspot_source": "s"},
                       {"hotspots": "a,b", "hotspot_source": "s"}, {"hotspots": " , ", "hotspot_source": "s"},
                       {"option": "D"}):
            with self.assertRaises(ValueError, msg=repr(kwargs)):
                build_plan(ROOT, **kwargs)

    def test_seed_and_run_id_are_checked(self):
        self.assertEqual(build_plan(ROOT, seed=0, run_id="r1")["run_id"], "r1")
        for kwargs in ({"seed": -1}, {"seed": True}, {"run_id": "bad id"}):
            with self.assertRaises(ValueError, msg=repr(kwargs)):
                build_plan(ROOT, **kwargs)

    def test_dry_run_passes_for_every_option_and_a_seed_the_old_script_refused(self):
        for option, expected in (("A", A), ("B", B), ("C", C)):
            checked = local_dry_run(ROOT, build_plan(ROOT, option=option, seed=7))
            self.assertEqual((checked["seed"], checked["hotspots"], checked["binder_length"]), (7, expected, 60))
            self.assertIn("seed=7", checked["overrides"])

    def test_dry_run_with_a_length_range(self):
        checked = local_dry_run(ROOT, build_plan(ROOT, length_min=50, length_max=80))
        self.assertEqual(checked["binder_length"], [50, 80])

    def test_dry_run_refuses_a_hotspot_without_coordinates(self):
        plan = build_plan(ROOT, hotspots="25,100", hotspot_source="test")        # 25 is inside 21-239 but unresolved
        with self.assertRaises(ValueError):
            local_dry_run(ROOT, plan)

    def test_command_line_wrapper(self):
        ok = subprocess.run([sys.executable, str(SCRIPT), "--root", str(ROOT), "--option", "B", "--seed", "9"],
                            capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertEqual(json.loads(ok.stdout)["plan"]["hotspots"], B)
        bad = subprocess.run([sys.executable, str(SCRIPT), "--root", str(ROOT), "--seed", "-4"], capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("REFUSED", bad.stderr)


if __name__ == "__main__":
    unittest.main()
