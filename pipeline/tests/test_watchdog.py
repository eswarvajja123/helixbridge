from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.watchdog import Stalled, run_watched  # noqa: E402

PY = sys.executable


class Watch(unittest.TestCase):
    def test_returns_code_and_output_and_echoes_live(self):
        seen = []
        code, out = run_watched([PY, "-c", "print('a');print('b')"], 5, 10, echo=seen.append, poll=0.05)
        self.assertEqual((code, out, seen), (0, "a\nb\n", ["a", "b"]))

    def test_nonzero_exit_is_returned_not_raised(self):
        code, out = run_watched([PY, "-c", "import sys;print('x',file=sys.stderr);sys.exit(3)"], 5, 10, echo=lambda l: None, poll=0.05)
        self.assertEqual(code, 3)
        self.assertIn("x", out)

    def test_a_silent_job_is_killed_with_its_last_output(self):
        with self.assertRaisesRegex(Stalled, "no output for 1 s.*started"):
            run_watched([PY, "-u", "-c", "import time;print('started',flush=True);time.sleep(30)"], 1, 20, echo=lambda l: None, poll=0.05)

    def test_a_chatty_job_over_the_total_limit_is_killed(self):
        with self.assertRaisesRegex(Stalled, "over the 1 s limit"):
            run_watched([PY, "-u", "-c", "import time\nwhile True:\n print('.',flush=True);time.sleep(0.1)"], 5, 1, echo=lambda l: None, poll=0.05)


if __name__ == "__main__":
    unittest.main()
