"""The optional one-line patch that makes the importer reachable as candidates.import_current."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1] / "binder_campaign"
PATCH = Path(__file__).resolve().parents[2] / "docs/eswar/candidates-reexport.patch"
HAVE_GIT = shutil.which("git") is not None


@unittest.skipUnless((PACKAGE / "candidates.py").is_file() and PATCH.is_file() and HAVE_GIT,
                     "needs the team's candidates.py, the patch file and git")
class ReexportPatchTests(unittest.TestCase):
    def test_patch_applies_and_the_function_is_the_same_object(self):
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "pipeline"
            shutil.copytree(PACKAGE, copy / "binder_campaign", ignore=shutil.ignore_patterns("__pycache__"))
            applied = subprocess.run(["git", "apply", str(PATCH)], cwd=temp, capture_output=True, text=True)
            self.assertEqual(applied.returncode, 0, applied.stderr)
            check = ("import binder_campaign.candidates as c, binder_campaign.current_candidates as cc;"
                     "assert c.import_current is cc.import_current; assert callable(c.import_rfd4); print('ok')")
            done = subprocess.run([sys.executable, "-c", check], cwd=copy, capture_output=True, text=True)
            self.assertEqual((done.returncode, done.stdout.strip()), (0, "ok"), done.stderr)

    def test_the_team_file_itself_is_not_modified_by_the_tests(self):
        self.assertNotIn("current_candidates", (PACKAGE / "candidates.py").read_text())


if __name__ == "__main__":
    unittest.main()
