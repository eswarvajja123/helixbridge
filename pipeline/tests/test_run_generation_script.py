import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _designs as D
from binder_campaign.batch import make_config, run_batch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run-il7ra-generation.py"
ARGS = ["--batch-id", "b1", "--seeds", "43,44", "--lengths", "50,60", "--option", "A", "--max-designs", "3",
        "--max-gpu-seconds", "600"]


@unittest.skipUnless(D.AVAILABLE, "needs the examples and docs/eswar")
class RunGenerationScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)

    def script(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)

    def test_plan_describes_the_batch_and_gives_the_exact_modal_command(self):
        done = self.script("plan", *ARGS, "--out", str(self.dir / "out"))
        self.assertEqual(done.returncode, 0, done.stderr)
        plan = json.loads(done.stdout)
        self.assertEqual((plan["input_checks"], plan["designs_planned"], plan["designs_that_can_launch_under_the_cap"]), ("passed", 4, 3))
        self.assertEqual([t["task"] for t in plan["tasks"]], ["s43-l50", "s43-l60", "s44-l50", "s44-l60"])
        self.assertEqual(plan["tasks"][0]["first_run_id"], "b1-s43-l50-a1")
        self.assertEqual(plan["hotspot_source"], "provisional_historical")
        for part in ("modal run modal_current.py --stage batch", "--max-designs 3", "--max-gpu-seconds 600", "--seeds 43,44",
                     "--option A"):
            self.assertIn(part, plan["command"])
        self.assertIsNone(plan["resume"])

    def test_plan_refuses_a_batch_without_a_budget(self):
        no_budget = [a for a in ARGS]
        no_budget[no_budget.index("--max-designs") + 1] = "0"
        for args in (no_budget, ["--batch-id", "b1", "--seeds", "43", "--lengths", "60", "--option", "A"]):
            done = self.script("plan", *args)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("REFUSED", done.stderr)

    def test_plan_refuses_bad_inputs(self):
        for change in ({"--seeds": "43,43"}, {"--lengths": "30"}, {"--option": "Z"}, {"--batch-id": "bad id"}):
            args = list(ARGS)
            for flag, value in change.items():
                args[args.index(flag) + 1] = value
            done = self.script("plan", *args)
            self.assertNotEqual(done.returncode, 0, change)
            self.assertIn("REFUSED", done.stderr, change)
        bad_hotspot = self.script("plan", "--batch-id", "b1", "--seeds", "43", "--lengths", "60", "--hotspots", "25,100",
                                  "--hotspot-source", "test", "--max-designs", "1", "--max-gpu-seconds", "60")
        self.assertIn("not resolved", bad_hotspot.stderr)

    def test_plan_with_custom_hotspots_puts_them_in_the_command(self):
        done = self.script("plan", "--batch-id", "b2", "--seeds", "1", "--lengths", "60", "--hotspots", "100,101",
                           "--hotspot-source", "my_note", "--max-designs", "1", "--max-gpu-seconds", "60")
        self.assertEqual(done.returncode, 0, done.stderr)
        command = json.loads(done.stdout)["command"]
        self.assertIn("--hotspots 100,101 --hotspot-source my_note", command)
        self.assertNotIn("--option", command)

    def test_plan_reads_an_existing_folder_and_says_whether_it_can_resume(self):
        designs, out = D.Designs(self.dir), self.dir / "out"
        cfg = make_config("b1", [43, 44], [50, 60], [100, 101, 102, 158, 159, 161, 211, 212, 213, 214],
                          "provisional_historical", max_designs=1, max_gpu_seconds=600)
        run_batch(cfg, lambda t: designs.result(t["run_id"], t["seed"], "base", length=t["binder_length"][0]), out,
                  D.MANIFEST, log=lambda *_: None)
        same = json.loads(self.script("plan", *ARGS, "--out", str(out)).stdout)["resume"]
        self.assertEqual((same["same_configuration"], same["designs_by_status"]), (True, {"done": 1}))
        other = [a for a in ARGS]
        other[other.index("--seeds") + 1] = "43,45"
        self.assertFalse(json.loads(self.script("plan", *other, "--out", str(out)).stdout)["resume"]["same_configuration"])

    def test_report_summarises_a_finished_batch_and_refuses_a_missing_folder(self):
        designs, out = D.Designs(self.dir), self.dir / "out"
        cfg = make_config("b1", [43, 44, 45], [60], [100, 101, 102, 158, 159, 161, 211, 212, 213, 214],
                          "provisional_historical", max_designs=5, max_gpu_seconds=600)
        kinds = {43: "base", 44: "exact", 45: "fold1"}
        run_batch(cfg, lambda t: designs.result(t["run_id"], t["seed"], kinds[t["seed"]]), out, D.MANIFEST, log=lambda *_: None)
        done = self.script("report", "--out", str(out))
        self.assertEqual(done.returncode, 0, done.stderr)
        report = json.loads(done.stdout)
        self.assertEqual((report["done"], report["kept"], report["dropped_by_dedup"], report["stop_reason"]), (3, 2, 1, "completed"))
        missing = self.script("report", "--out", str(self.dir / "nowhere"))
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("REFUSED", missing.stderr)


if __name__ == "__main__":
    unittest.main()
