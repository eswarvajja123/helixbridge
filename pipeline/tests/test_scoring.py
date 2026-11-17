import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign import scoring as S

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures/ipsae_model_10_10.txt"            # real output of the pinned scorer on the real Boltz structure
REAL = Path(__file__).resolve().parents[2] / "examples/verified-modal-run"
IPSAE = os.environ.get("HELIXBRIDGE_IPSAE", "")              # path to ipsae.py at the pinned commit (optional)
ROWS = [{"chain1": "A", "chain2": "B", "type": "asym", "ipsae": 0.111551},
        {"chain1": "B", "chain2": "A", "type": "asym", "ipsae": 0.439671},
        {"chain1": "A", "chain2": "B", "type": "max", "ipsae": 0.439671}]


def metrics_like_the_teams_boltz_script():
    """The keys scripts/boltz-predict.py writes into metrics.json, with values from the verified run."""
    summary = json.loads((REAL / "summary.json").read_text()) if (REAL / "summary.json").is_file() else None
    iptm = summary["iptm"] if summary else 0.945
    rows = summary["ipsae"]["pairwise"] if summary else ROWS
    return {"iptm": iptm, "ptm": 0.9, "ipsae": {"pairwise": rows}, "prediction_seconds": 61.5, "peak_reserved_mib": 9000.0,
            "seed": 0, "samples": 1, "target_only_precomputed_msa": True}


@unittest.skipUnless(FIXTURE.is_file(), "needs the fixture")
class ParseTests(unittest.TestCase):
    def test_the_real_output_parses_to_the_three_rows(self):
        self.assertEqual(S.parse_ipsae(FIXTURE.read_text()), ROWS)

    def test_the_header_and_other_lines_are_ignored(self):
        self.assertEqual(S.parse_ipsae("\nChn1 Chn2 PAE Dist Type ipSAE\nnot a row\n"), [])

    def test_a_nonfinite_value_is_refused(self):
        with self.assertRaises(ValueError):
            S.parse_ipsae("A B 10 10 asym nan 0 0")


class RowChecks(unittest.TestCase):
    def test_a_correct_set_passes_in_either_chain_order(self):
        S.check_rows(ROWS)
        reversed_rows = [{**r, "chain1": r["chain2"], "chain2": r["chain1"]} if r["type"] == "asym" else r for r in ROWS]
        S.check_rows(reversed_rows, ("B", "A"))

    def test_each_kind_of_wrong_set_is_refused(self):
        cases = {"only one direction": ROWS[:1] + ROWS[2:], "no maximum": ROWS[:2],
                 "two maxima": ROWS + [ROWS[2]], "a repeated direction": [ROWS[0], ROWS[0], ROWS[2]],
                 "other chains": [{**r, "chain1": "C"} if r["type"] == "asym" else r for r in ROWS], "empty": []}
        for label, rows in cases.items():
            with self.assertRaises(ValueError, msg=label):
                S.check_rows(rows)

    def test_nonfinite_or_non_float_values_are_refused(self):
        for bad in (float("nan"), float("inf"), 1, "0.5"):
            rows = [dict(r) for r in ROWS]
            rows[0]["ipsae"] = bad
            with self.assertRaises(ValueError, msg=repr(bad)):
                S.check_rows(rows)

    def test_the_summary_names_and_values(self):
        self.assertEqual(S.summarise_rows(ROWS), {"a_b": 0.111551, "b_a": 0.439671, "max": 0.439671, "min_directional": 0.111551})


class RunIpsaeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        (self.dir / "model.cif").write_text("data_x\n")
        (self.dir / "pae.json").write_text("{}")

    def script(self, body):
        path = self.dir / "fake_ipsae.py"
        path.write_text(body)
        return path

    def test_it_runs_the_script_with_the_team_cutoffs_and_reads_the_file_next_to_the_structure(self):
        fake = self.script("import sys, pathlib\n"
                           "pae, cif, pae_cut, dist_cut = sys.argv[1:5]\n"
                           "assert (pae_cut, dist_cut) == ('10', '10'), sys.argv\n"
                           "out = pathlib.Path(cif).with_name(pathlib.Path(cif).stem + '_' + pae_cut + '_' + dist_cut + '.txt')\n"
                           "out.write_text(open(%r).read())\n" % str(FIXTURE))
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.dir.parent)                                     # a different working directory must not matter
        self.assertEqual(S.run_ipsae(fake, self.dir / "pae.json", self.dir / "model.cif"), ROWS)
        self.assertTrue((self.dir / "model.ipsae.log").is_file())

    def test_a_failing_script_a_silent_script_and_bad_output_are_all_errors(self):
        with self.assertRaises(subprocess.CalledProcessError):
            S.run_ipsae(self.script("raise SystemExit(3)"), self.dir / "pae.json", self.dir / "model.cif")
        with self.assertRaisesRegex(ValueError, "no score file"):
            S.run_ipsae(self.script("pass"), self.dir / "pae.json", self.dir / "model.cif")
        incomplete = self.script("import sys, pathlib\ncif = pathlib.Path(sys.argv[2])\n"
                                 "cif.with_name(cif.stem + '_10_10.txt').write_text('A B 10 10 asym 0.5 0\\n')\n")
        with self.assertRaises(ValueError):
            S.run_ipsae(incomplete, self.dir / "pae.json", self.dir / "model.cif")


@unittest.skipUnless(IPSAE and Path(IPSAE).is_file() and (REAL / "boltz.cif").is_file(),
                     "set HELIXBRIDGE_IPSAE to ipsae.py (pinned commit) to run the real scorer")
class RealScorerTests(unittest.TestCase):
    def test_the_real_scorer_reproduces_the_fixture_exactly(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            shutil.copyfile(REAL / "boltz.cif", temp / "model.cif")
            n, rng = 296, np.random.default_rng(0)               # the recipe used to make the fixture
            pae = np.full((n, n), 3.0)
            pae[:236, 236:], pae[236:, :236] = 7.0, 6.0
            pae += rng.uniform(0, 1, size=(n, n))
            (temp / "pae.json").write_text(json.dumps({"pae": pae.tolist()}))
            self.assertEqual(S.run_ipsae(IPSAE, temp / "pae.json", temp / "model.cif"), ROWS)

    def test_the_scorer_is_the_pinned_commit(self):
        done = subprocess.run(["git", "-C", str(Path(IPSAE).parent), "rev-parse", "HEAD"], capture_output=True, text=True)
        if done.returncode != 0:
            self.skipTest("not a git checkout")
        self.assertEqual(done.stdout.strip(), S.IPSAE_COMMIT)


class BuildScoreTests(unittest.TestCase):
    def good(self, **changes):
        args = dict(iptm=0.9, ptm=0.8, rows=ROWS, seconds=12.0, peak_reserved_mib=2000.0, plddt_raw_mean=0.7, gpu="X",
                    settings={"seed": 0})
        args.update(changes)
        return S.build_score("esmfold2", **args)

    def test_shape_and_values(self):
        score = self.good()
        self.assertEqual((score["predictor"], score["iptm"], score["ipsae"]["min_directional"], score["ipsae"]["pae_cutoff"]),
                         ("esmfold2", 0.9, 0.111551, 10))
        self.assertEqual(len(score["ipsae"]["pairwise"]), 3)
        self.assertIn("No threshold", score["note"])
        self.assertIsNone(self.good(plddt_raw_mean=None)["plddt_raw_mean"])

    def test_bad_inputs_are_refused(self):
        for changes in ({"iptm": float("nan")}, {"ptm": float("inf")}, {"seconds": None}, {"peak_reserved_mib": True},
                        {"plddt_raw_mean": float("nan")}, {"rows": ROWS[:2]}):
            with self.assertRaises(ValueError, msg=repr(changes)):
                self.good(**changes)
        with self.assertRaises(ValueError):
            S.build_score("alphafold", iptm=1, ptm=1, rows=ROWS, seconds=1, peak_reserved_mib=1)

    def test_a_received_score_is_revalidated(self):
        score = self.good()
        self.assertEqual(S.check_score(json.loads(json.dumps(score)), "esmfold2"), score)
        broken = {"another predictor": ({**score, "predictor": "boltz2"}, "esmfold2"), "not a dict": ("x", "esmfold2"),
                  "NaN iptm": ({**score, "iptm": float("nan")}, "esmfold2"),
                  "missing ipsae": ({k: v for k, v in score.items() if k != "ipsae"}, "esmfold2")}
        for label, (value, predictor) in broken.items():
            with self.assertRaises((ValueError, KeyError, TypeError), msg=label):
                S.check_score(value, predictor)

    def test_boltz_metrics_become_a_score(self):
        score = S.boltz_score(metrics_like_the_teams_boltz_script(), gpu="A10G")
        self.assertEqual((score["predictor"], score["gpu"], score["plddt_raw_mean"]), ("boltz2", "A10G", None))
        self.assertAlmostEqual(score["iptm"], 0.9454354643821716)
        self.assertEqual(score["ipsae"]["max"], 0.784061)                      # the verified run's symmetric maximum


if __name__ == "__main__":
    unittest.main()
