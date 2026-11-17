import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _designs as D
from binder_campaign import batch
from binder_campaign.batch import build_tasks, make_config, parse_lengths, parse_seeds, run_batch

try:
    import tmtools  # noqa: F401
    HAVE_TM = True
except ImportError:
    HAVE_TM = False

BASE = dict(batch_id="b1", seeds=[1, 2], lengths=[60], hotspots=[100, 101, 102], hotspot_source="provisional_historical",
            max_designs=10, max_gpu_seconds=10000)


def config(**changes):
    return make_config(**{**BASE, **changes})


class ConfigTests(unittest.TestCase):
    def test_budget_has_no_defaults(self):
        for missing in ("max_designs", "max_gpu_seconds"):
            args = {k: v for k, v in BASE.items() if k != missing}
            with self.assertRaises(TypeError, msg=missing):
                make_config(**args)

    def test_bad_values_are_refused(self):
        bad = [{"batch_id": "has space"}, {"batch_id": "x" * 37}, {"seeds": []}, {"seeds": [1, 1]}, {"seeds": [-1]},
               {"seeds": [True]}, {"lengths": []}, {"lengths": [39]}, {"lengths": [101]}, {"lengths": [[80, 50]]},
               {"lengths": [[1, 2, 3]]}, {"hotspots": []}, {"hotspot_source": " "}, {"max_designs": 0},
               {"max_designs": True}, {"max_gpu_seconds": 0}, {"max_gpu_seconds": -5}, {"max_gpu_seconds": True},
               {"max_failures": 0}, {"batch_id": "b" * 36, "seeds": [10 ** 25]}, {"seeds": list(range(600))}]
        for changes in bad:
            with self.assertRaises(ValueError, msg=repr(changes)[:60]):
                config(**changes)

    def test_lengths_are_normalised_and_deduplicated(self):
        self.assertEqual(config(lengths=[60, [60, 60], [70, 80], 70])["lengths"], [[60, 60], [70, 80], [70, 70]])

    def test_parsers(self):
        self.assertEqual(parse_seeds("43, 44,45"), [43, 44, 45])
        self.assertEqual(parse_lengths("50,60,70-80"), [[50, 50], [60, 60], [70, 80]])
        for bad in ("a,b", "1;2"):
            with self.assertRaises(ValueError):
                parse_seeds(bad)
        with self.assertRaises(ValueError):
            parse_lengths("x-y")

    def test_task_grid_is_seed_major_with_stable_ids(self):
        tasks = build_tasks(config(seeds=[43, 44], lengths=[50, [70, 80]]))
        self.assertEqual([t["task_id"] for t in tasks], ["s43-l50", "s43-l70x80", "s44-l50", "s44-l70x80"])
        self.assertEqual(tasks[1]["binder_length"], [70, 80])

    def test_resume_identity_ignores_the_budget_only(self):
        a, b = config(), config(max_designs=99, max_gpu_seconds=5, max_failures=9)
        self.assertEqual(batch.identity_of(a), batch.identity_of(b))
        self.assertNotEqual(batch.identity_of(a), batch.identity_of(config(seeds=[1, 3])))


@unittest.skipUnless(D.AVAILABLE and HAVE_TM, "needs the examples, docs/eswar and tmtools")
class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.designs = D.Designs(self.dir)
        self.out = self.dir / "out"
        self.calls = []

    def scripted(self, kinds, fail=(), seconds=60.0, **flags):
        """generate_fn that returns the real-format result for kinds[seed]; seeds in `fail` raise instead."""
        def generate(task):
            self.calls.append(task["run_id"])
            if task["seed"] in fail:
                raise RuntimeError(f"container failed for seed {task['seed']}")
            return self.designs.result(task["run_id"], task["seed"], kinds[task["seed"]], length=task["binder_length"][0],
                                       worker_seconds=seconds, **flags)
        return generate

    def go(self, cfg, fn, **kwargs):
        return run_batch(cfg, fn, self.out, D.MANIFEST, log=lambda *_: None, **kwargs)

    def records(self):
        return json.loads((self.out / "candidates.json").read_text())["candidates"]

    def test_diverse_designs_are_kept_and_each_kind_of_copy_is_dropped_for_its_own_reason(self):
        kinds = {1: "base", 2: "fold1", 3: "exact", 4: "twin", 5: "mutant", 6: "fold2"}
        report = self.go(config(seeds=list(kinds)), self.scripted(kinds))
        self.assertEqual((report["done"], report["kept"], report["dropped_by_dedup"], report["stop_reason"]), (6, 3, 3, "completed"))
        self.assertEqual(report["dropped_by_reason"], {"exact_duplicate": 1, "backbone_tm": 1, "sequence_identity": 1})
        by_seed = {r["seed"]: r for r in self.records()}
        for seed in (1, 2, 6):
            self.assertEqual(by_seed[seed]["qc_reasons"], [], seed)
        self.assertEqual(by_seed[3]["qc_reasons"], ["near_duplicate:exact_duplicate"])
        self.assertEqual(by_seed[4]["qc_reasons"], ["near_duplicate:backbone_tm"])
        self.assertEqual(by_seed[5]["qc_reasons"], ["near_duplicate:sequence_identity"])
        self.assertEqual(len(self.records()), 6)                      # dropped designs stay on record

    def test_every_record_is_unreviewed_and_the_export_guard_refuses_a_dropped_one(self):
        from binder_campaign.candidates import export_draft
        kinds = {1: "base", 2: "exact"}
        self.go(config(seeds=[1, 2]), self.scripted(kinds))
        records = self.records()
        self.assertTrue(all(r["submission_eligible"] is False and r["novelty_review"] == "pending" for r in records))
        data = json.loads((self.out / "candidates.json").read_text())
        for r in data["candidates"]:
            r.update(novelty_review="passed", independent_prediction_status="reviewed", screen_status="approximation_reviewed")
        (self.out / "reviewed.json").write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            export_draft(self.out / "reviewed.json", self.out / "d.csv", [r["candidate_id"] for r in records if r["seed"] == 2])
        export_draft(self.out / "reviewed.json", self.out / "ok.csv", [r["candidate_id"] for r in records if r["seed"] == 1])

    def test_max_designs_stops_the_batch_before_the_next_launch(self):
        kinds = {1: "base", 2: "fold1", 3: "fold2", 4: "fold3"}
        report = self.go(config(seeds=[1, 2, 3, 4], max_designs=2), self.scripted(kinds))
        self.assertEqual((report["stop_reason"], report["done"], len(self.calls)), ("max_designs", 2, 2))
        self.assertEqual(report["not_started"], ["s3-l60", "s4-l60"])

    def test_max_gpu_seconds_stops_with_at_most_one_design_of_overshoot(self):
        kinds = {1: "base", 2: "fold1", 3: "fold2", 4: "fold3"}
        report = self.go(config(seeds=[1, 2, 3, 4], max_gpu_seconds=150), self.scripted(kinds, seconds=100.0))
        self.assertEqual((report["stop_reason"], report["done"], report["gpu_seconds_reported"]), ("max_gpu_seconds", 2, 200.0))

    def test_failures_are_recorded_and_max_failures_stops_the_batch(self):
        kinds = {1: "base", 2: "fold1", 3: "fold2"}
        report = self.go(config(seeds=[1, 2, 3], max_failures=1), self.scripted(kinds, fail={2}))
        self.assertEqual((report["stop_reason"], report["done"], len(report["failed"])), ("max_failures", 1, 1))
        self.assertIn("container failed for seed 2", report["failed"][0]["error"])
        self.assertEqual(report["not_started"], ["s3-l60"])

    def test_other_designs_continue_after_a_failure_when_the_cap_allows(self):
        kinds = {1: "base", 2: "fold1", 3: "fold2"}
        report = self.go(config(seeds=[1, 2, 3], max_failures=3), self.scripted(kinds, fail={2}))
        self.assertEqual((report["done"], report["stop_reason"], [f["task"] for f in report["failed"]]),
                         (2, "finished_with_open_tasks", ["s2-l60"]))

    def test_a_failed_design_is_not_retried_unless_asked_and_the_failure_stays_on_record(self):
        kinds = {1: "base", 2: "fold1"}
        self.go(config(seeds=[1, 2], max_failures=3), self.scripted(kinds, fail={2}))
        again = self.go(config(seeds=[1, 2], max_failures=3), self.scripted(kinds))                 # not asked: no retry
        self.assertEqual((len(self.calls), again["done"]), (2, 1))
        final = self.go(config(seeds=[1, 2], max_failures=3), self.scripted(kinds), retry_failed=True)
        self.assertEqual((final["done"], final["stop_reason"]), (2, "completed"))
        design = final["designs"]["s2-l60"]
        self.assertEqual((design["attempts"], design["run_id"].endswith("-a2")), (2, True))
        self.assertIn("container failed for seed 2", design["earlier_attempts"][0]["error"])
        self.assertEqual(final["failed_attempts"], 1)

    def test_resuming_gives_exactly_the_result_of_one_uninterrupted_run(self):
        kinds = {1: "base", 2: "fold1", 3: "exact", 4: "twin", 5: "mutant", 6: "fold2"}
        cfg_all = config(seeds=list(kinds))
        first = self.go(config(seeds=list(kinds), max_designs=3), self.scripted(kinds))
        self.assertEqual(first["done"], 3)
        resumed = self.go(cfg_all, self.scripted(kinds))
        self.assertEqual((resumed["done"], len(self.calls)), (6, 6))                      # 3 + 3, nothing launched twice
        split = [(r["seed"], tuple(r["qc_reasons"])) for r in self.records()]
        self.out = self.dir / "single"
        self.calls.clear()
        single = self.go(cfg_all, self.scripted(kinds))
        self.assertEqual(single["done"], 6)
        self.assertEqual(split, [(r["seed"], tuple(r["qc_reasons"])) for r in self.records()])

    def test_a_changed_configuration_is_refused_but_a_larger_budget_is_allowed(self):
        kinds = {1: "base", 2: "fold1", 3: "fold2"}
        self.go(config(seeds=[1, 2], max_designs=1), self.scripted(kinds))
        with self.assertRaises(ValueError):
            self.go(config(seeds=[1, 3]), self.scripted(kinds))
        with self.assertRaises(ValueError):
            self.go(config(seeds=[1, 2], hotspot_source="other"), self.scripted(kinds))
        report = self.go(config(seeds=[1, 2], max_designs=5), self.scripted(kinds))
        self.assertEqual(report["done"], 2)

    def test_an_interrupted_design_is_reported_and_never_relaunched_silently(self):
        kinds = {1: "base", 2: "fold1"}

        def generate(task):
            self.calls.append(task["run_id"])
            if task["seed"] == 2:
                raise KeyboardInterrupt
            return self.designs.result(task["run_id"], task["seed"], kinds[task["seed"]])
        with self.assertRaises(KeyboardInterrupt):
            self.go(config(seeds=[1, 2]), generate)
        report = json.loads((self.out / "batch-report.json").read_text())      # written even though it was interrupted
        self.assertEqual((report["done"], [i["task"] for i in report["interrupted"]], report["stop_reason"]),
                         (1, ["s2-l60"], "interrupted_by_user"))
        calls_before = len(self.calls)
        quiet = self.go(config(seeds=[1, 2]), self.scripted(kinds))
        self.assertEqual((len(self.calls), [i["task"] for i in quiet["interrupted"]]), (calls_before, ["s2-l60"]))
        retried = self.go(config(seeds=[1, 2]), self.scripted(kinds), retry_interrupted=True)
        self.assertEqual((retried["done"], retried["designs"]["s2-l60"]["attempts"]), (2, 2))

    def test_unusable_results_become_failures_and_the_batch_carries_on(self):
        ok = self.designs.result("x", 9, "base")
        archive_with = {
            "path traversal": D.archive({"../evil.txt": b"x", "b1-s1-l60-a1/generation/model.cif": b"x"}),
            "no model file": D.archive({"b1-s1-l60-a1/generation/audit.json": b"{}"}),
            "not an archive": b"this is not gzip data",
        }
        for label, data in archive_with.items():
            self.out = self.dir / label.replace(" ", "-")
            report = self.go(config(seeds=[1], max_failures=3), lambda task, data=data: {**ok, "archive": data})
            self.assertEqual((report["done"], len(report["failed"])), (0, 1), label)
        for label, flags in {"hash mismatch": {"break_hash": True}, "wrong manifest": {"wrong_manifest": True}}.items():
            self.out = self.dir / label.replace(" ", "-")
            report = self.go(config(seeds=[1], max_failures=3), self.scripted({1: "base"}, **flags))
            self.assertEqual((report["done"], len(report["failed"])), (0, 1), label)
            self.assertIn("ValueError", report["failed"][0]["error"])
        self.out = self.dir / "missing-key"
        report = self.go(config(seeds=[1], max_failures=3), lambda task: {"archive": ok["archive"]})   # no worker_seconds
        self.assertEqual((report["done"], len(report["failed"])), (0, 1))

    def test_an_oversize_member_is_refused(self):
        original = batch.MAX_MEMBER_BYTES
        batch.MAX_MEMBER_BYTES = 10
        self.addCleanup(setattr, batch, "MAX_MEMBER_BYTES", original)
        report = self.go(config(seeds=[1], max_failures=2), self.scripted({1: "base"}))
        self.assertEqual((report["done"], len(report["failed"])), (0, 1))
        self.assertIn("too large", report["failed"][0]["error"])

    def test_a_failed_import_does_not_poison_the_dedup_gate(self):
        # design X first arrives with a wrong manifest hash (rejected), then correctly: it must be KEPT, not "duplicate"
        def generate(task):
            self.calls.append(task["run_id"])
            return self.designs.result(task["run_id"], task["seed"], "base", wrong_manifest=(task["seed"] == 1))
        report = self.go(config(seeds=[1, 2], max_failures=3), generate)
        self.assertEqual((report["done"], report["kept"], len(report["failed"])), (1, 1, 1))
        self.assertEqual(self.records()[0]["qc_reasons"], [])

    def test_state_is_written_before_the_call(self):
        seen = {}

        def generate(task):
            seen["state"] = json.loads((self.out / "state.json").read_text())["tasks"]["s1-l60"]["status"]
            return self.designs.result(task["run_id"], task["seed"], "base")
        self.go(config(seeds=[1]), generate)
        self.assertEqual(seen["state"], "started")

    def test_range_lengths_pass_through_and_short_designs_work(self):
        kinds = {1: "base", 2: "fold1"}
        report = self.go(config(seeds=[1, 2], lengths=[50]), self.scripted(kinds))
        self.assertEqual(report["done"], 2)
        self.assertEqual({len(r["sequence"]) for r in self.records()}, {50})

    def test_task_ids_and_run_ids_stay_within_limits(self):
        kinds = {1: "base"}
        self.go(config(seeds=[1], lengths=[[50, 80]]), self.scripted(kinds))
        self.assertTrue(all(len(c) <= 64 for c in self.calls))


if __name__ == "__main__":
    unittest.main()
