import csv
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign import selection as S

AA = "ACDEFGHIKLMNPQRSTVWY"


def seq(seed, n=60):
    rng = random.Random(seed)
    return "".join(rng.choice(AA) for _ in range(n))


def score(iptm, ipsae):
    return {"status": "done", "score": {"iptm": iptm, "ipsae": {"max": ipsae}}}


def rec(cid, n=1, esm=(0.9, 0.8), boltz=(0.9, 0.8), flags=(), qc="no_flags", novelty="no_public_hit", reasons=(), sequence=None):
    return {"candidate_id": cid, "target_id": "il7ra", "sequence": sequence or seq(int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16)),
            "seed": n, "hotspot_source": "x", "campaign_kind": "competition_campaign", "qc_reasons": list(reasons),
            "screen_status": "not_run", "novelty_review": "pending", "independent_prediction_status": "not_run", "submission_eligible": False,
            "qc": {"status": qc}, "novelty_prescreen": {"status": novelty},
            "liabilities": {"diagnostic": {"length": 60}, "flags": list(flags)},
            "evaluations": {"esmfold2": score(*esm), "boltz2": score(*boltz)}}


def diversity(fold, site, source="generator", ft="0.5", st="0.5"):
    return {"sources": {source: {"groups": {"fold": {ft: {"complete_link_greedy": fold}}, "site": {st: {"complete_link_greedy": site}}}}}}


class Eligibility(unittest.TestCase):
    def test_each_reason(self):
        self.assertEqual(S.eligibility(rec("a")), [])
        self.assertEqual(S.eligibility(rec("a", reasons=["near_duplicate:x"])), ["qc_reason:near_duplicate:x"])
        self.assertEqual(S.eligibility(rec("a", qc="flagged")), ["structure_qc_flagged"])
        self.assertEqual(S.eligibility(rec("a", qc="incomplete")), ["structure_qc_incomplete"])
        self.assertEqual(S.eligibility(rec("a", novelty="flagged")), ["novelty_prescreen_flagged"])
        self.assertEqual(S.eligibility(rec("a", novelty="not_searched")), ["novelty_prescreen_not_searched"])
        r = rec("a"); del r["qc"]
        self.assertEqual(S.eligibility(r), ["structure_qc_missing"])
        r = rec("a"); r["evaluations"]["boltz2"] = {"status": "failed"}
        self.assertEqual(S.eligibility(r), ["not_scored_by_both_predictors"])
        r = rec("a"); r["liabilities"]["diagnostic"] = None
        self.assertEqual(S.eligibility(r), ["liabilities_not_reviewed"])

    def test_disagreement_is_the_largest_gap(self):
        self.assertEqual(S.disagreement(rec("a", esm=(0.9, 0.8), boltz=(0.85, 0.5))), 0.3)
        self.assertEqual(S.disagreement(rec("a", esm=(0.5, 0.5), boltz=(0.9, 0.5))), 0.4)


class Picking(unittest.TestCase):
    def ids(self, selected):
        return [s["candidate_id"] for s in selected]

    def test_round_robin_over_fold_groups(self):
        recs = [rec(c) for c in "abcdefghi"]
        fold = [list("abcde"), list("fgh"), ["i"]]
        site = [[c] for c in "abcdefghi"]                       # every design its own site, so only the fold rule is visible
        selected, report = S.select(recs, diversity(fold, site), limit=6)
        groups = [s["fold_group"] for s in selected]
        self.assertEqual(groups[:3], ["1", "2", "3"])           # one from each fold group first
        self.assertEqual(sorted(groups), ["1", "1", "1", "2", "2", "3"])
        self.assertEqual([s["pick_order"] for s in selected], [1, 2, 3, 4, 5, 6])
        self.assertEqual((report["fold_groups_selected"], report["fold_groups_available"]), (3, 3))

    def test_a_group_of_one_is_never_starved_by_a_big_group(self):
        recs = [rec(c) for c in "abcdefg"]
        selected, _ = S.select(recs, diversity([list("abcdef"), ["g"]], [[c] for c in "abcdefg"]), limit=2)
        self.assertEqual(sorted(s["fold_group"] for s in selected), ["1", "2"])

    def test_inside_a_group_an_unused_site_comes_first(self):
        recs = [rec(c) for c in "abc"]
        fold = [list("abc")]
        site = [["a", "b"], ["c"]]                              # a and b share a site; c is elsewhere
        selected, _ = S.select(recs, diversity(fold, site), limit=2)
        self.assertEqual(self.ids(selected), ["a", "c"])        # a first (alphabetical), then c because a's site is taken

    def test_then_fewer_liability_flags_then_better_agreement_then_id(self):
        fold, site = [list("abcd")], [[c] for c in "abcd"]
        recs = [rec("a", flags=["asn_gly_motif"]), rec("b"), rec("c", esm=(0.9, 0.8), boltz=(0.5, 0.8)), rec("d")]
        selected, _ = S.select(recs, diversity(fold, site), limit=4)
        self.assertEqual(self.ids(selected), ["b", "d", "c", "a"])   # b,d: no flags and agree; c: flagged by nothing but disagrees; a: has a flag

    def test_strength_does_not_decide_selection(self):
        fold, site = [list("abcdef")], [[c] for c in "abcdef"]
        weak = [rec(c, esm=(0.6, 0.4), boltz=(0.6, 0.4)) for c in "abc"]
        strong = [rec(c, esm=(0.97, 0.9), boltz=(0.97, 0.9)) for c in "def"]
        selected, _ = S.select(weak + strong, diversity(fold, site), limit=3)
        self.assertEqual(self.ids(selected), ["a", "b", "c"])        # equal agreement, no flags: id order, not score

    def test_ineligible_designs_are_listed_with_reasons_and_never_picked(self):
        recs = [rec("a"), rec("b", qc="flagged"), rec("c", novelty="flagged"), rec("d", reasons=["x"]), rec("e")]
        selected, report = S.select(recs, diversity([list("abcde")], [[c] for c in "abcde"]), limit=10)
        self.assertEqual(self.ids(selected), ["a", "e"])
        self.assertEqual({e["candidate_id"] for e in report["excluded"]}, {"b", "c", "d"})
        self.assertEqual((report["pool"], report["eligible"]), (5, 2))

    def test_a_design_missing_from_the_diversity_report_is_excluded(self):
        selected, report = S.select([rec("a"), rec("b")], diversity([["a"]], [["a"]]), limit=5)
        self.assertEqual(self.ids(selected), ["a"])
        self.assertEqual(report["excluded"], [{"candidate_id": "b", "reasons": ["not_in_the_diversity_report"]}])

    def test_an_exact_duplicate_sequence_is_picked_once(self):
        s = seq(5)
        recs = [rec("a", sequence=s), rec("b", sequence=s), rec("c")]
        selected, report = S.select(recs, diversity([list("abc")], [[c] for c in "abc"]), limit=5)
        self.assertEqual(self.ids(selected), ["a", "c"])
        self.assertIn({"candidate_id": "b", "reasons": ["exact_duplicate_sequence_of_an_earlier_design"]}, report["excluded"])

    def test_the_shortlist_carries_each_designs_sequence_and_the_csv_matches(self):
        recs = [rec(c) for c in "abc"]
        selected, report = S.select(recs, diversity([list("abc")], [list("abc")]), limit=3)
        self.assertEqual({s["candidate_id"]: s["sequence"] for s in selected}, {r["candidate_id"]: r["sequence"] for r in recs})
        with tempfile.TemporaryDirectory() as t:
            S.write_outputs(Path(t), selected, report, recs, {"check": {"passes": False, "message": "x"}, "gaps": {}})
            rows = list(csv.DictReader(io.StringIO((Path(t) / "selection.csv").read_text())))
        self.assertEqual({r["candidate_id"]: r["sequence"] for r in rows}, {r["candidate_id"]: r["sequence"] for r in recs})
        self.assertEqual(list(rows[0])[:3], ["pick_order", "candidate_id", "sequence"])

    def test_limits_and_inputs_are_validated(self):
        d = diversity([["a"]], [["a"]])
        for bad in (0, 21, True, 2.5, -1):
            with self.subTest(limit=bad), self.assertRaises(ValueError):
                S.select([rec("a")], d, limit=bad)
        with self.assertRaisesRegex(ValueError, "no groups for source"):
            S.select([rec("a")], d, source="boltz2")
        with self.assertRaisesRegex(ValueError, "threshold"):
            S.select([rec("a")], d, fold_threshold=0.9)
        with self.assertRaisesRegex(ValueError, "Duplicate candidate_id"):
            S.select([rec("a"), rec("a")], d)

    def test_never_more_than_twenty_and_deterministic(self):
        recs = [rec(f"c{i:02d}") for i in range(30)]
        fold = [[f"c{i:02d}" for i in range(0, 30, 3)], [f"c{i:02d}" for i in range(1, 30, 3)], [f"c{i:02d}" for i in range(2, 30, 3)]]
        site = [[f"c{i:02d}" for i in range(15)], [f"c{i:02d}" for i in range(15, 30)]]
        first, _ = S.select(recs, diversity(fold, site))
        shuffled = recs[:]
        random.Random(3).shuffle(shuffled)
        second, _ = S.select(shuffled, diversity(fold, site))
        self.assertEqual(len(first), 20)
        self.assertEqual([s["candidate_id"] for s in first], [s["candidate_id"] for s in second])


class Reviews(unittest.TestCase):
    def setUp(self):
        self.recs = [rec("a"), rec("b"), rec("c")]
        self.ids = ["a", "b"]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "draft.csv"
        self.full = {"reviewer": "Eswar", "date": "2026-11-18",
                     "candidates": {c: {"novelty_review": "passed", "independent_prediction_status": "reviewed",
                                        "screen_status": "approximation_reviewed"} for c in self.ids}}

    def test_the_guard_refuses_until_a_person_has_reviewed(self):
        check = S.export_check(self.recs, self.ids, self.out)
        self.assertFalse(check["passes"])
        self.assertIn("review is incomplete", check["message"])
        self.assertFalse(self.out.exists())
        gaps = S.review_gaps(self.recs, self.ids)
        self.assertEqual(len(gaps["a"]), 3)
        self.assertTrue(gaps["a"][1].startswith("independent_prediction_status"))

    def test_recorded_reviews_let_the_guard_pass_and_leave_provenance(self):
        reviewed = S.apply_reviews(self.recs, self.ids, self.full)
        self.assertEqual([r["candidate_id"] for r in reviewed], ["a", "b"])
        self.assertEqual(reviewed[0]["review_provenance"], {"reviewer": "Eswar", "date": "2026-11-18",
                         "fields": ["independent_prediction_status", "novelty_review", "screen_status"]})
        check = S.export_check(reviewed, self.ids, self.out)
        self.assertTrue(check["passes"], check)
        rows = list(csv.DictReader(io.StringIO(self.out.read_text())))
        self.assertEqual([r["candidate_id"] for r in rows], ["a", "b"])
        self.assertEqual(S.review_gaps(reviewed, self.ids), {"a": [], "b": []})
        self.assertEqual(self.recs[0]["novelty_review"], "pending")              # the input is not modified

    def test_a_partial_review_still_refuses(self):
        partial = {**self.full, "candidates": {"a": self.full["candidates"]["a"], "b": {"novelty_review": "passed"}}}
        reviewed = S.apply_reviews(self.recs, self.ids, partial)
        check = S.export_check(reviewed, self.ids, self.out)
        self.assertFalse(check["passes"])
        self.assertEqual(len(S.review_gaps(reviewed, self.ids)["b"]), 2)

    def test_a_review_that_leaves_novelty_unreviewed_leaves_it_pending(self):
        """No code path may mark novelty as passed on a person's behalf."""
        only = {**self.full, "candidates": {c: {"independent_prediction_status": "reviewed", "screen_status": "approximation_reviewed"}
                                            for c in self.ids}}
        reviewed = S.apply_reviews(self.recs, self.ids, only)
        self.assertEqual({r["novelty_review"] for r in reviewed}, {"pending"})
        self.assertEqual([len(v) for v in S.review_gaps(reviewed, self.ids).values()], [1, 1])
        self.assertFalse(S.export_check(reviewed, self.ids, self.out)["passes"])
        for r in S.apply_reviews(self.recs, self.ids, {**self.full, "candidates": {}}):          # an empty review changes nothing
            self.assertEqual((r["novelty_review"], r["independent_prediction_status"], r["screen_status"]), ("pending", "not_run", "not_run"))
            self.assertNotIn("review_provenance", r)

    def test_bad_review_files_are_refused(self):
        for bad in ({}, {"reviewer": "x", "date": "d"}, {**self.full, "reviewer": ""},
                    {**self.full, "candidates": {"zzz": {"novelty_review": "passed"}}},
                    {**self.full, "candidates": {"a": {"novelty_review": "maybe"}}},
                    {**self.full, "candidates": {"a": {"submission_eligible": True}}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                S.apply_reviews(self.recs, self.ids, bad)

    def test_a_design_with_qc_reasons_cannot_be_exported_even_if_reviewed(self):
        recs = [rec("a", reasons=["public_similarity:x"])]
        reviewed = S.apply_reviews(recs, ["a"], {**self.full, "candidates": {"a": self.full["candidates"]["a"]}})
        check = S.export_check(reviewed, ["a"], self.out)
        self.assertFalse(check["passes"])
        self.assertIn("cannot enter a submission draft", check["message"])


class Script(unittest.TestCase):
    SCRIPT = Path(__file__).resolve().parents[1] / "scripts/select-finalists.py"

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            recs = [rec(c) for c in "abcdef"]
            (t / "r.json").write_text(json.dumps({"candidates": recs}))
            (t / "d.json").write_text(json.dumps(diversity([list("abc"), list("def")], [list("abcdef")])))
            run = lambda *a: subprocess.run([sys.executable, "-I", str(self.SCRIPT), *a], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
            done = run("--records", str(t / "r.json"), "--diversity", str(t / "d.json"), "--out", str(t / "o"), "--limit", "4")
            self.assertEqual(done.returncode, 0, done.stderr)
            summary = json.loads(done.stdout)
            self.assertEqual((summary["selected"], summary["fold_groups"], summary["export_guard_passes"]), (4, "2 of 2", False))
            self.assertFalse((t / "o" / "draft-submission.csv").exists())
            sel = json.loads((t / "o" / "selection.json").read_text())
            self.assertEqual(len(sel["review_gaps"]["a"]), 3)
            reviews = {"reviewer": "E", "date": "d", "candidates": {s["candidate_id"]: {"novelty_review": "passed",
                       "independent_prediction_status": "reviewed", "screen_status": "approximation_reviewed"} for s in sel["selected"]}}
            (t / "rev.json").write_text(json.dumps(reviews))
            again = run("--records", str(t / "r.json"), "--diversity", str(t / "d.json"), "--out", str(t / "o2"), "--limit", "4", "--reviews", str(t / "rev.json"))
            self.assertEqual(json.loads(again.stdout)["export_guard_passes"], True, again.stderr)
            self.assertTrue((t / "o2" / "draft-submission.csv").is_file())
            self.assertTrue(run("--records", str(t / "r.json"), "--diversity", str(t / "d.json"), "--out", str(t / "o")).returncode != 0)
            bad = run("--records", str(t / "r.json"), "--diversity", str(t / "d.json"), "--out", str(t / "o3"), "--limit", "99")
            self.assertTrue(bad.returncode != 0 and "REFUSED" in bad.stderr and "Traceback" not in bad.stderr, bad.stderr)


if __name__ == "__main__":
    unittest.main()
