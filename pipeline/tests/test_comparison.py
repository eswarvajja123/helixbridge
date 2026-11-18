import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign import comparison as C


def batch(done, failed=0, interrupted=0, seconds=0.0, attempts=None):
    r = {"done": done, "failed": [{"x": 1}] * failed, "interrupted": [{"x": 1}] * interrupted, "gpu_seconds_reported": seconds}
    if attempts is not None:
        r["attempts"] = attempts
    return r


def evaluation(done, seconds):
    return {"by_predictor": {"esmfold2": {"done": done}, "boltz2": {"done": done}}, "gpu_seconds_reported": seconds}


def rec(status, reasons=(), esm=None, boltz=None):
    r = {"qc": {"status": status, "reasons": list(reasons)}}
    if esm is not None:
        r["evaluations"] = {"esmfold2": {"status": "done", "score": {"iptm": esm}}, "boltz2": {"status": "done", "score": {"iptm": boltz}}}
    return r


def diversity(fold_groups, site_groups, members):
    return {"sources": {"generator": {"summary": {"fold": {"0.5": {"complete_link_greedy": {"groups": fold_groups}}},
                                                  "site": {"0.5": {"complete_link_greedy": {"groups": site_groups}}}},
                                      "members": list(range(members))}}}


class Summaries(unittest.TestCase):
    def setUp(self):
        self.records = ([rec("no_flags")] * 4 + [rec("flagged", ["pose_shift:generator_vs_boltz2:9A>2.5A", "pose_shift:esmfold2_vs_boltz2:9A>3.0A"])] * 3
                        + [rec("flagged", ["pose_shift:generator_vs_esmfold2:9A>2.5A"])] * 2 + [rec("incomplete")])

    def row(self, **kw):
        args = dict(label="X", batch_reports=[batch(10, 1, 0, 600.0)], evaluation_reports=[evaluation(10, 1200.0)], qc_records=self.records,
                    diversity=diversity(3, 2, 4))
        args.update(kw)
        return C.summarize(**args)

    def test_counts_and_rates_are_plain_arithmetic(self):
        r = self.row()
        self.assertEqual((r["designs_generated"], r["attempts_failed"], r["designs_scored"]), (10, 1, 10))
        self.assertEqual((r["gpu_seconds_generation"], r["gpu_seconds_scoring"], r["gpu_seconds_total"]), (600.0, 1200.0, 1800.0))
        self.assertEqual((r["gpu_seconds_per_design_generation"], r["gpu_seconds_per_design_total"]), (60.0, 180.0))
        self.assertEqual((r["qc_clean"], r["qc_flagged"], r["qc_incomplete"], r["qc_clean_share"]), (4, 5, 1, round(4 / 9, 3)))
        self.assertEqual(r["per_gpu_hour"], {"designs": 20.0, "qc_clean": 8.0, "distinct_fold_groups_among_clean": 6.0})
        self.assertEqual((r["distinct_fold_groups_among_clean"], r["distinct_site_groups_among_clean"], r["designs_in_diversity_report"]), (3, 2, 4))

    def test_the_boltz2_independent_reading(self):
        r = self.row()
        # 4 clean + the 3 flagged only by Boltz-2-related flags; the 2 flagged by an ESMFold2 pose shift stay out; incomplete never counts
        self.assertEqual(r["clean_if_boltz2_flags_are_ignored"], 7)

    def test_failures_are_kept_and_their_seconds_stay_in_the_total(self):
        r = self.row(batch_reports=[batch(5, 2, 1, 300.0), batch(5, 0, 0, 300.0)])
        self.assertEqual((r["designs_generated"], r["attempts_failed"], r["attempts_interrupted"], r["gpu_seconds_generation"]), (10, 2, 1, 600.0))

    def test_attempts_that_made_no_design_are_shown(self):
        r = self.row(batch_reports=[batch(4, 0, 0, 300.0, attempts=5), batch(4, 0, 0, 300.0, attempts=4)])
        self.assertEqual((r["designs_generated"], r["attempts_total"], r["attempts_not_producing_a_design"]), (8, 9, 1))
        old = self.row(batch_reports=[batch(5, 2, 1, 300.0)])                      # a report without an attempts count still adds up
        self.assertEqual((old["attempts_total"], old["attempts_not_producing_a_design"]), (8, 3))

    def test_score_spreads_are_described_per_predictor(self):
        recs = [rec("no_flags", esm=0.9, boltz=0.8), rec("flagged", esm=0.1, boltz=0.7), rec("flagged", esm=0.5, boltz=0.9), rec("incomplete")]
        r = self.row(qc_records=recs)
        self.assertEqual(r["esmfold2_iptm"], {"n": 3, "min": 0.1, "median": 0.5, "max": 0.9})
        self.assertEqual(r["boltz2_iptm"], {"n": 3, "min": 0.7, "median": 0.8, "max": 0.9})
        self.assertIsNone(self.row(qc_records=[rec("flagged")])["esmfold2_iptm"])

    def test_no_clean_design_means_zero_groups_not_unknown(self):
        r = self.row(qc_records=[rec("flagged")] * 3, diversity=None)
        self.assertEqual((r["distinct_fold_groups_among_clean"], r["distinct_site_groups_among_clean"]), (0, 0))
        self.assertIsNone(self.row(diversity=None)["distinct_fold_groups_among_clean"])          # clean designs exist but no report was given

    def test_no_diversity_report_gives_none_not_zero(self):
        r = self.row(diversity=None)
        self.assertIsNone(r["distinct_fold_groups_among_clean"])
        self.assertIsNone(r["per_gpu_hour"]["distinct_fold_groups_among_clean"])

    def test_nothing_generated_does_not_divide_by_zero(self):
        r = self.row(batch_reports=[batch(0, 3, 0, 0.0)], evaluation_reports=[evaluation(0, 0.0)], qc_records=[], diversity=None)
        self.assertEqual((r["designs_generated"], r["attempts_failed"]), (0, 3))
        self.assertIsNone(r["per_gpu_hour"])
        self.assertIsNone(r["gpu_seconds_per_design_total"])

    def test_scaling_to_another_budget_is_proportional(self):
        r = self.row()
        scaled = C.at_budget(r, 3600.0)
        self.assertEqual((scaled["designs"], scaled["qc_clean"], scaled["distinct_fold_groups_among_clean"]), (20.0, 8.0, 6.0))
        self.assertIsNone(C.at_budget(self.row(batch_reports=[batch(0, 1, 0, 0.0)], evaluation_reports=[evaluation(0, 0.0)]), 3600.0))
        self.assertIsNone(C.at_budget(r, 0))


class Compare(unittest.TestCase):
    def test_outputs(self):
        a = C.summarize("RFD4", [batch(72, 0, 0, 5000.0)], [evaluation(71, 4400.0)], [rec("no_flags")] * 27 + [rec("flagged")] * 44, diversity(15, 3, 27))
        b = C.summarize("Mosaic", [batch(8, 1, 0, 6000.0)], [evaluation(8, 500.0)], [rec("no_flags")] * 2 + [rec("flagged")] * 6, diversity(2, 1, 2))
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "c"
            result = C.compare([a, b], out)
            data = json.loads((out / "comparison.json").read_text())
            self.assertEqual([m["label"] for m in data["methods"]], ["RFD4", "Mosaic"])
            self.assertEqual(data["at_the_other_methods_budget"]["RFD4"]["Mosaic"]["gpu_seconds"], b["gpu_seconds_total"])
            self.assertIn("not a ranking of designs", data["scope"])
            rows = list(csv.DictReader(io.StringIO((out / "comparison.csv").read_text())))
            self.assertEqual([r["label"] for r in rows], ["RFD4", "Mosaic"])
            self.assertEqual(rows[1]["attempts_failed"], "1")
            self.assertIn("| RFD4 | 72 |", (out / "comparison.md").read_text())
            with self.assertRaisesRegex(ValueError, "fresh output"):
                C.compare([a, b], out)
            with self.assertRaisesRegex(ValueError, "two or more"):
                C.compare([a], Path(t) / "c2")
            with self.assertRaisesRegex(ValueError, "distinct labels"):
                C.compare([a, a], Path(t) / "c3")

    def test_script(self):
        script = Path(__file__).resolve().parents[1] / "scripts/compare-generators.py"
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            def write(name, obj):
                (t / name).write_text(json.dumps(obj))
                return str(t / name)
            spec = lambda label, secs: (f"{label}=" + write(f"{label}-b.json", batch(4, 0, 0, secs)) + "|" + write(f"{label}-e.json", evaluation(4, 100.0))
                                        + "|" + write(f"{label}-q.json", {"candidates": [rec("no_flags")] * 2 + [rec("flagged")] * 2}) + "|")
            run = lambda *a: subprocess.run([sys.executable, "-I", str(script), *a], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
            done = run("--out", str(t / "o"), "--method", spec("A", 200.0), "--method", spec("B", 900.0))
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual([m["label"] for m in json.loads(done.stdout)["methods"]], ["A", "B"])
            bad = run("--out", str(t / "o2"), "--method", "oops", "--method", "x=y")
            self.assertTrue(bad.returncode != 0 and "REFUSED" in bad.stderr and "Traceback" not in bad.stderr, bad.stderr)


if __name__ == "__main__":
    unittest.main()
