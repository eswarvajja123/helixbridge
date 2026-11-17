import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _designs as D
import test_evaluate as T
from binder_campaign.evaluate import make_config, run_evaluation

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate-candidates.py"
ROOT = Path(__file__).resolve().parents[2]
AVAILABLE = (ROOT / "docs/eswar/esmfold2-checkpoint-manifest.json").is_file() and (ROOT / "docs/eswar/phase1-data/target.a3m").is_file()


@unittest.skipUnless(AVAILABLE, "needs docs/eswar")
class EvaluateScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.records = [T.record(i) for i in range(1, 6)] + [T.record(9, flagged=True)]
        self.candidates = self.dir / "candidates.json"
        self.candidates.write_text(json.dumps({"candidates": self.records}))
        self.args = ["--candidates", str(self.candidates), "--eval-id", "e1", "--max-calls", "6", "--max-gpu-seconds", "1800",
                     "--chunk-size", "2"]

    def script(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)

    def test_plan_describes_the_evaluation_and_gives_the_exact_modal_command(self):
        done = self.script("plan", *self.args, "--out", str(self.dir / "out"))
        self.assertEqual(done.returncode, 0, done.stderr)
        plan = json.loads(done.stdout)
        self.assertEqual((plan["input_checks"], plan["candidates_to_evaluate"], plan["calls_needed_if_nothing_fails"],
                          plan["calls_that_can_launch_under_the_cap"]), ("passed", 5, 6, 6))                  # ceil(5/2) = 3 per predictor
        self.assertEqual([s["reasons"] for s in plan["candidates_skipped"]], [["near_duplicate:exact_duplicate"]])
        for part in ("modal run modal_evaluate.py --stage evaluate", "--max-calls 6", "--max-gpu-seconds 1800", "--chunk-size 2",
                     "--predictors esmfold2,boltz2", "--kernel-backend fused"):
            self.assertIn(part, plan["command"])
        self.assertIn("once per call", plan["model_loads_note"])
        self.assertIsNone(plan["resume"])

    def test_plan_refuses_a_missing_budget_and_other_bad_input(self):
        base = ["--candidates", str(self.candidates), "--eval-id", "e1"]
        cases = {"no budget": base, "no gpu-second cap": base + ["--max-calls", "2"], "no call cap": base + ["--max-gpu-seconds", "60"],
                 "unknown predictor": self.args + ["--predictors", "alphafold"], "bad backend": self.args + ["--kernel-backend", "turbo"],
                 "chunk too big": self.args + ["--chunk-size", "50"], "missing file": ["--candidates", str(self.dir / "none.json"),
                 "--eval-id", "e1", "--max-calls", "2", "--max-gpu-seconds", "60"]}
        for label, args in cases.items():
            done = self.script("plan", *args)
            self.assertNotEqual(done.returncode, 0, label)
            self.assertIn("REFUSED", done.stderr, label)

    def test_plan_refuses_when_everything_is_flagged_but_not_with_include_flagged(self):
        flagged = self.dir / "flagged.json"
        flagged.write_text(json.dumps({"candidates": [T.record(1, flagged=True)]}))
        args = ["--candidates", str(flagged), "--eval-id", "e1", "--max-calls", "2", "--max-gpu-seconds", "60"]
        self.assertIn("No candidates", self.script("plan", *args).stderr)
        self.assertEqual(self.script("plan", *args, "--include-flagged").returncode, 0)

    def test_plan_refuses_an_invalid_sequence_before_any_spending(self):
        bad = self.dir / "bad.json"
        bad.write_text(json.dumps({"candidates": [{**T.record(1), "sequence": "ACDXZ"}]}))
        done = self.script("plan", "--candidates", str(bad), "--eval-id", "e1", "--max-calls", "2", "--max-gpu-seconds", "60")
        self.assertIn("REFUSED", done.stderr)

    def test_plan_and_report_read_an_existing_evaluation_folder(self):
        out = self.dir / "out"
        engine = T.EngineTests("test_scores_csv")
        engine.setUp()
        self.addCleanup(engine.temp.cleanup)
        cfg = make_config("e1", ["esmfold2", "boltz2"], 2, max_calls=2, max_gpu_seconds=1800)
        run_evaluation(cfg, self.candidates, engine.fn(), out, log=lambda *_: None)
        plan = json.loads(self.script("plan", *self.args, "--out", str(out)).stdout)
        self.assertEqual((plan["resume"]["same_settings"], plan["resume"]["calls_so_far"]), (True, 2))
        other = json.loads(self.script("plan", *[a if a != "2" else "3" for a in self.args], "--out", str(out)).stdout)
        self.assertFalse(other["resume"]["same_settings"])                                   # a different chunk size
        done = self.script("report", "--out", str(out))
        self.assertEqual(done.returncode, 0, done.stderr)
        report = json.loads(done.stdout)
        self.assertEqual((report["stop_reason"], report["candidates_evaluated"], report["calls"]), ("max_calls", 5, 2))
        self.assertEqual(report["by_predictor"]["esmfold2"]["done"], 4)
        missing = self.script("report", "--out", str(self.dir / "nowhere"))
        self.assertIn("REFUSED", missing.stderr)


if __name__ == "__main__":
    unittest.main()
