import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _designs as D
from binder_campaign import batch, evaluate as EV
from binder_campaign.evaluate import make_config, run_evaluation, select_candidates
from binder_campaign.scoring import build_score

ROWS = [{"chain1": "A", "chain2": "B", "type": "asym", "ipsae": 0.5},
        {"chain1": "B", "chain2": "A", "type": "asym", "ipsae": 0.6},
        {"chain1": "A", "chain2": "B", "type": "max", "ipsae": 0.6}]


def record(index, flagged=False):
    return {"candidate_id": f"il7ra-rfd4-{index:012x}", "target_id": "il7ra", "sequence": D.random_sequence(100 + index),
            "seed": index, "qc_reasons": ["near_duplicate:exact_duplicate"] if flagged else [], "screen_status": "not_run",
            "independent_prediction_status": "not_run", "novelty_review": "pending", "submission_eligible": False}


class ConfigAndSelection(unittest.TestCase):
    def test_budget_has_no_defaults(self):
        for missing in ("max_calls", "max_gpu_seconds"):
            args = {"eval_id": "e1", "predictors": ["esmfold2"], "chunk_size": 2, "max_calls": 3, "max_gpu_seconds": 100}
            del args[missing]
            with self.assertRaises(TypeError, msg=missing):
                make_config(**args)

    def test_bad_values_are_refused(self):
        base = dict(eval_id="e1", predictors=["esmfold2"], chunk_size=2, max_calls=3, max_gpu_seconds=100)
        for changes in ({"eval_id": "bad id"}, {"eval_id": "x" * 31}, {"predictors": []}, {"predictors": ["alphafold"]},
                        {"predictors": ["boltz2", "boltz2"]}, {"chunk_size": 0}, {"chunk_size": 21}, {"chunk_size": True},
                        {"max_calls": 0}, {"max_gpu_seconds": 0}, {"max_gpu_seconds": True}, {"max_failures": 0},
                        {"seed": -1}, {"seed": 2 ** 31}, {"seed": True}, {"kernel_backend": "turbo"}, {"kernel_backend": None}):
            with self.assertRaises(ValueError, msg=repr(changes)):
                make_config(**{**base, **changes})
        self.assertEqual(make_config(**base)["predictors"], ["esmfold2"])

    def test_flagged_candidates_are_skipped_with_their_reasons(self):
        records = [record(1), record(2, flagged=True), record(3)]
        chosen, skipped = select_candidates(records)
        self.assertEqual([r["seed"] for r in chosen], [1, 3])
        self.assertEqual(skipped, [{"candidate_id": records[1]["candidate_id"], "reasons": ["near_duplicate:exact_duplicate"]}])
        self.assertEqual(len(select_candidates(records, include_flagged=True)[0]), 3)

    def test_unsafe_duplicate_or_invalid_records_are_refused(self):
        for bad in ({"candidate_id": "../escape"}, {"candidate_id": ""}, {"candidate_id": None}, {"sequence": "ACDX"},
                    {"sequence": ""}, {"sequence": None}):
            with self.assertRaises(ValueError, msg=repr(bad)):
                select_candidates([{**record(1), **bad}])
        with self.assertRaises(ValueError):
            select_candidates([record(1), record(1)])


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.records = [record(i) for i in range(1, 6)] + [record(9, flagged=True)]
        self.candidates = self.dir / "candidates.json"
        self.candidates.write_text(json.dumps({"candidates": self.records}))
        self.out = self.dir / "out"
        self.requests = []

    def config(self, **changes):
        base = dict(eval_id="e1", predictors=["esmfold2", "boltz2"], chunk_size=2, max_calls=50, max_gpu_seconds=100000)
        base.update(changes)
        return make_config(**base)

    def fn(self, fail_ids=(), crash_calls=(), seconds=30.0, bad=None):
        """evaluate_fn returning real-format responses; `bad(kind)` can corrupt one part."""
        def run(request):
            self.requests.append(request)
            if len(self.requests) in crash_calls:
                raise RuntimeError("container crashed")
            results, members = [], {}
            for c in request["candidates"]:
                cid = c["id"]
                if cid in fail_ids:
                    results.append({"id": cid, "status": "failed", "error": "model said no"})
                    continue
                shift = int(hashlib.sha256(cid.encode()).hexdigest()[:2], 16) / 1000
                rows = [{**r, "ipsae": round(r["ipsae"] * 0.5 + shift, 6)} for r in ROWS]
                score = build_score(request["predictor"], iptm=0.6 + shift, ptm=0.7, rows=rows, seconds=10.0,
                                    peak_reserved_mib=2000.0, plddt_raw_mean=0.8 if request["predictor"] == "esmfold2" else None, gpu="Fake")
                results.append({"id": cid, "status": "ok", "sequence": c["sequence"], "score": score})
                for name in EV.FILES:
                    members[f"{request['run_id']}/{cid}/{name}"] = f"{name} of {cid}".encode()
            response = {"run_id": request["run_id"], "gpu": "Fake GPU", "worker_seconds": seconds, "results": results,
                        "archive": D.archive(members)}
            return bad(response, request) if bad else response
        return run

    def go(self, cfg=None, fn=None, **kwargs):
        return run_evaluation(cfg or self.config(), self.candidates, fn or self.fn(), self.out, log=lambda *_: None, **kwargs)

    def evaluated(self):
        return {r["candidate_id"]: r for r in json.loads((self.out / "evaluations.json").read_text())["candidates"]}

    def test_every_eligible_candidate_gets_both_predictors_and_the_flagged_one_is_skipped(self):
        report = self.go()
        self.assertEqual((report["stop_reason"], report["candidates_evaluated"], len(report["candidates_skipped"])), ("completed", 5, 1))
        self.assertEqual(report["by_predictor"]["esmfold2"], {"done": 5, "failed": 0, "started": 0, "not_started": 0})
        self.assertEqual(len(self.requests), 6)                                   # 3 chunks per predictor (2 + 2 + 1)
        evaluated = self.evaluated()
        for r in self.records[:5]:
            e = evaluated[r["candidate_id"]]["evaluations"]
            self.assertEqual({p: e[p]["status"] for p in e}, {"esmfold2": "done", "boltz2": "done"})
            self.assertEqual(e["esmfold2"]["score"]["predictor"], "esmfold2")
            self.assertIsNotNone(e["esmfold2"]["score"]["plddt_raw_mean"])
            self.assertIsNone(e["boltz2"]["score"]["plddt_raw_mean"])
        flagged = evaluated[self.records[5]["candidate_id"]]
        self.assertEqual((flagged["evaluations"], flagged["evaluation_skipped"]), ({}, ["near_duplicate:exact_duplicate"]))

    def test_requests_carry_exactly_what_is_needed(self):
        self.go(self.config(seed=4, kernel_backend="none"))
        first = self.requests[0]
        self.assertEqual((first["predictor"], first["seed"], first["run_id"], first["kernel_backend"]), ("esmfold2", 4, "e1-esmfold2-c1", "none"))
        self.assertEqual(first["candidates"], [{"id": r["candidate_id"], "sequence": r["sequence"]} for r in self.records[:2]])
        self.assertTrue(all(len(r["run_id"]) <= 64 for r in self.requests))
        self.assertNotIn(self.records[5]["candidate_id"], json.dumps(self.requests))      # the flagged one is never sent

    def test_records_are_copied_never_altered_and_the_input_file_is_untouched(self):
        before = hashlib.sha256(self.candidates.read_bytes()).hexdigest()
        self.go()
        self.assertEqual(hashlib.sha256(self.candidates.read_bytes()).hexdigest(), before)
        for r in self.records:
            item = self.evaluated()[r["candidate_id"]]
            self.assertEqual({k: v for k, v in item.items() if k not in ("evaluations", "evaluation_skipped")}, r)
            self.assertEqual((item["screen_status"], item["novelty_review"], item["submission_eligible"]), ("not_run", "pending", False))

    def test_scores_are_stored_raw_with_no_ranking_threshold_or_combination(self):
        self.go()
        text = (self.out / "evaluations.json").read_text()
        for word in ("rank", "pass", "threshold_met", "combined", "consensus"):
            self.assertNotIn(f'"{word}', text)
        self.assertIn("No threshold", text)

    def test_artifacts_are_saved_with_matching_hashes(self):
        self.go()
        cid = self.records[0]["candidate_id"]
        entry = self.evaluated()[cid]["evaluations"]["boltz2"]
        folder = self.out / "evals" / cid / "boltz2"
        for name, sha in entry["artifacts"].items():
            self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), sha)
        self.assertEqual((folder / "model.cif").read_bytes(), f"model.cif of {cid}".encode())
        self.assertTrue((self.out / "evals" / entry["run_id"] / "result.tar.gz").is_file())

    def test_scores_csv(self):
        self.go()
        rows = list(csv.DictReader((self.out / "scores.csv").open()))
        self.assertEqual(len(rows), 10)                                           # 5 candidates x 2 predictors
        row = next(r for r in rows if r["predictor"] == "esmfold2")
        self.assertTrue(float(row["iptm"]) > 0.6 and row["status"] == "done" and float(row["ipsae_min_directional"]) > 0)
        boltz = next(r for r in rows if r["predictor"] == "boltz2")
        self.assertEqual(boltz["plddt_raw_mean"], "")

    # ---- budget
    def test_max_calls_stops_before_the_next_launch(self):
        report = self.go(self.config(max_calls=2))
        self.assertEqual((report["stop_reason"], report["calls"], len(self.requests)), ("max_calls", 2, 2))
        self.assertEqual(report["by_predictor"]["esmfold2"]["done"], 4)
        self.assertEqual(report["by_predictor"]["esmfold2"]["not_started"], 1)
        self.assertEqual(report["by_predictor"]["boltz2"]["not_started"], 5)

    def test_gpu_seconds_cap_overshoots_by_at_most_one_call(self):
        report = self.go(self.config(max_gpu_seconds=100), self.fn(seconds=80.0))
        self.assertEqual((report["stop_reason"], report["calls"], report["gpu_seconds_reported"]), ("max_gpu_seconds", 2, 160.0))

    def test_failure_cap_stops_after_enough_failed_evaluations(self):
        ids = [r["candidate_id"] for r in self.records[:5]]
        report = self.go(self.config(max_failures=2), self.fn(fail_ids=ids[:2]))
        self.assertEqual((report["stop_reason"], report["failed_evaluations"], report["calls"]), ("max_failures", 2, 1))

    # ---- failures
    def test_one_bad_candidate_does_not_discard_the_others_in_its_call(self):
        victim = self.records[0]["candidate_id"]
        report = self.go(self.config(max_failures=5), self.fn(fail_ids=[victim]))
        self.assertEqual(report["by_predictor"]["esmfold2"], {"done": 4, "failed": 1, "started": 0, "not_started": 0})
        self.assertIn("model said no", [f for f in report["failures"] if f["predictor"] == "esmfold2"][0]["error"])
        self.assertEqual(self.evaluated()[self.records[1]["candidate_id"]]["evaluations"]["esmfold2"]["status"], "done")
        self.assertEqual(self.evaluated()[victim]["evaluations"]["esmfold2"]["status"], "failed")

    def test_a_crashed_call_fails_its_candidates_and_the_run_carries_on(self):
        report = self.go(self.config(max_failures=10), self.fn(crash_calls={1}))
        self.assertEqual(report["by_predictor"]["esmfold2"], {"done": 3, "failed": 2, "started": 0, "not_started": 0})
        self.assertIn("container crashed", report["failures"][0]["error"])

    def test_every_kind_of_malformed_result_becomes_a_failure_not_a_crash(self):
        victim = self.records[0]["candidate_id"]

        def corrupt(label):
            def change(response, request):
                if request["predictor"] != "esmfold2":
                    return response
                if label == "traversal":
                    response["archive"] = D.archive({"../evil": b"x"})
                    return response
                if label == "bad archive":
                    response["archive"] = b"not gzip"
                    return response
                item = next((r for r in response["results"] if r["id"] == victim), None)
                if item is None:                                  # this chunk does not hold the victim
                    return response
                if label == "missing result":
                    response["results"].remove(item)
                elif label == "wrong sequence":
                    item["sequence"] = "A" * len(item["sequence"])
                elif label == "NaN score":
                    item["score"]["iptm"] = float("nan")
                elif label == "other predictor":
                    item["score"]["predictor"] = "boltz2"
                elif label == "missing member":
                    members = {f"{request['run_id']}/{r['id']}/{n}": b"x" for r in response["results"] if r["status"] == "ok"
                               for n in EV.FILES if not (r["id"] == victim and n == "pae.json")}
                    response["archive"] = D.archive(members)
                return response
            return change
        for label in ("missing result", "wrong sequence", "NaN score", "other predictor", "missing member", "traversal", "bad archive"):
            self.out, self.requests = self.dir / label.replace(" ", "-"), []
            report = self.go(self.config(predictors=["esmfold2"], max_failures=50), self.fn(bad=corrupt(label)))
            states = report["by_predictor"]["esmfold2"]
            if label in ("traversal", "bad archive"):
                self.assertEqual((states["failed"], states["done"]), (5, 0), label)    # no archive is usable: every chunk fails
            else:
                self.assertEqual((states["failed"], states["done"]), (1, 4), label)    # only the victim fails

    def test_an_oversize_member_is_refused(self):
        original = batch.MAX_MEMBER_BYTES
        batch.MAX_MEMBER_BYTES = 5
        self.addCleanup(setattr, batch, "MAX_MEMBER_BYTES", original)
        report = self.go(self.config(predictors=["esmfold2"], max_failures=50))
        self.assertEqual(report["by_predictor"]["esmfold2"]["done"], 0)
        self.assertIn("too large", report["failures"][0]["error"])

    # ---- resume
    def test_resuming_gives_the_result_of_one_uninterrupted_run(self):
        first = self.go(self.config(max_calls=3))
        self.assertEqual(first["stop_reason"], "max_calls")
        calls_after_first = len(self.requests)
        resumed = self.go(self.config(max_calls=50))
        self.assertEqual((resumed["stop_reason"], len(self.requests)), ("completed", 6))      # 3 + 3, nothing sent twice
        self.assertEqual(calls_after_first, 3)
        split = {cid: json.dumps(r["evaluations"], sort_keys=True, default=str) for cid, r in self.evaluated().items()
                 if "run_id" not in json.dumps(r["evaluations"])}
        self.assertEqual(len(self.evaluated()), 6)
        scores_split = {cid: {p: e["score"]["ipsae"]["min_directional"] for p, e in r["evaluations"].items()}
                        for cid, r in self.evaluated().items() if r["evaluations"]}
        self.out, self.requests = self.dir / "single", []
        self.go()
        scores_single = {cid: {p: e["score"]["ipsae"]["min_directional"] for p, e in r["evaluations"].items()}
                         for cid, r in self.evaluated().items() if r["evaluations"]}
        self.assertEqual(scores_split, scores_single)

    def test_a_changed_input_or_setting_is_refused_but_a_larger_budget_is_allowed(self):
        self.go(self.config(max_calls=1))
        changed = [dict(r) for r in self.records]
        changed[0]["sequence"] = D.random_sequence(777)
        other = self.dir / "other.json"
        other.write_text(json.dumps({"candidates": changed}))
        with self.assertRaises(ValueError):
            run_evaluation(self.config(), other, self.fn(), self.out, log=lambda *_: None)
        with self.assertRaises(ValueError):
            self.go(self.config(chunk_size=3))
        with self.assertRaises(ValueError):
            self.go(self.config(seed=9))
        with self.assertRaises(ValueError):
            self.go(self.config(kernel_backend="none"))                       # a different backend is a different evaluation
        self.assertEqual(self.go(self.config(max_calls=99))["stop_reason"], "completed")

    def test_failed_pairs_are_not_retried_unless_asked_and_history_is_kept(self):
        victim = self.records[0]["candidate_id"]
        self.go(self.config(max_failures=5), self.fn(fail_ids=[victim]))
        again = self.go(self.config(max_failures=5), self.fn())
        self.assertEqual(again["by_predictor"]["esmfold2"]["failed"], 1)             # not asked, not retried
        final = self.go(self.config(max_failures=5), self.fn(), retry_failed=True)
        self.assertEqual((final["stop_reason"], final["by_predictor"]["esmfold2"]["failed"]), ("completed", 0))
        entry = self.evaluated()[victim]["evaluations"]["esmfold2"]
        self.assertEqual((entry["status"], entry["attempts"]), ("done", 2))
        self.assertIn("model said no", entry["earlier_attempts"][0]["error"])

    def test_an_interrupted_call_is_reported_and_not_silently_relaunched(self):
        def interrupt(request):
            self.requests.append(request)
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.go(fn=interrupt)
        report = json.loads((self.out / "evaluation-report.json").read_text())
        self.assertEqual((report["stop_reason"], len(report["interrupted"])), ("interrupted_by_user", 2))
        sent = len(self.requests)
        quiet = self.go()
        self.assertEqual(len(self.requests), sent + 5)                       # esmfold2: 3 left = 2 calls; boltz2: all 5 = 3 calls
        self.assertEqual(len(quiet["interrupted"]), 2)
        retried = self.go(retry_interrupted=True)
        self.assertEqual(len(retried["interrupted"]), 0)

    def test_state_is_written_before_the_call(self):
        seen = {}

        def probe(request):
            state = json.loads((self.out / "state.json").read_text())
            seen["status"] = state["results"][request["candidates"][0]["id"]][request["predictor"]]["status"]
            return self.fn()(request)
        self.go(self.config(predictors=["esmfold2"], max_calls=1), probe)
        self.assertEqual(seen["status"], "started")

    @unittest.skipUnless(D.AVAILABLE, "needs the examples and docs/eswar")
    def test_it_reads_the_real_output_of_the_generation_batch(self):
        """Task 2.5's candidates.json (real records from import_current, with dedup decisions) feeds this evaluator as it is."""
        designs = D.Designs(self.dir / "designs")
        kinds = {1: "base", 2: "fold1", 3: "exact", 4: "twin", 5: "fold2"}
        cfg = batch.make_config("g1", list(kinds), [60], D.HOTSPOTS, "provisional_historical", max_designs=10, max_gpu_seconds=10000)
        gen = batch.run_batch(cfg, lambda t: designs.result(t["run_id"], t["seed"], kinds[t["seed"]]), self.dir / "gen", D.MANIFEST,
                              log=lambda *_: None)
        self.assertEqual((gen["kept"], gen["dropped_by_dedup"]), (3, 2))
        out = self.dir / "eval-from-gen"
        report = run_evaluation(self.config(predictors=["esmfold2"], chunk_size=5), self.dir / "gen/candidates.json", self.fn(), out,
                                log=lambda *_: None)
        self.assertEqual((report["candidates_total"], report["candidates_evaluated"], len(report["candidates_skipped"])), (5, 3, 2))
        self.assertEqual(sorted(r for s in report["candidates_skipped"] for r in s["reasons"]),
                         ["near_duplicate:backbone_tm", "near_duplicate:exact_duplicate"])
        self.assertEqual(len(self.requests), 1)                                   # the three kept designs go in one call
        self.assertEqual(len(self.requests[0]["candidates"]), 3)
        sent = {c["id"] for c in self.requests[0]["candidates"]}
        generated = json.loads((self.dir / "gen/candidates.json").read_text())["candidates"]
        self.assertEqual(sent, {r["candidate_id"] for r in generated if not r["qc_reasons"]})

    def test_nothing_to_evaluate_is_an_error(self):
        self.candidates.write_text(json.dumps({"candidates": [record(1, flagged=True)]}))
        with self.assertRaisesRegex(ValueError, "No candidates"):
            self.go()


class ComparisonTests(unittest.TestCase):
    """comparison.csv and the pool summary: descriptive only, in file order, no ranking, no verdict."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.records = [record(i) for i in range(1, 5)] + [record(9, flagged=True)]
        self.candidates = self.dir / "candidates.json"
        self.candidates.write_text(json.dumps({"candidates": self.records}))
        self.out = self.dir / "out"
        self.iptm = {"esmfold2": {1: 0.5, 2: 0.9, 3: 0.3, 4: 0.7}, "boltz2": {1: 0.8, 2: 0.2, 3: 0.6, 4: 0.4}}   # neither is in order, on purpose

    def fn(self, fail=()):
        def run(request):
            results, members = [], {}
            for c in request["candidates"]:
                index = int(c["id"].rsplit("-", 1)[1], 16)
                if (request["predictor"], index) in fail:
                    results.append({"id": c["id"], "status": "failed", "error": "scripted"})
                    continue
                value = self.iptm[request["predictor"]][index]
                rows = [{**r, "ipsae": round(value / 2, 6) if r["type"] == "max" else r["ipsae"]} for r in ROWS]
                score = build_score(request["predictor"], iptm=value, ptm=0.7, rows=rows, seconds=5.0, peak_reserved_mib=1.0,
                                    plddt_raw_mean=0.8 if request["predictor"] == "esmfold2" else None, gpu="Fake")
                results.append({"id": c["id"], "status": "ok", "sequence": c["sequence"], "score": score})
                for name in EV.FILES:
                    members[f"{request['run_id']}/{c['id']}/{name}"] = b"x"
            return {"run_id": request["run_id"], "gpu": "Fake", "worker_seconds": 5.0, "results": results, "archive": D.archive(members)}
        return run

    def go(self, predictors=("esmfold2", "boltz2"), fail=()):
        cfg = make_config("e1", list(predictors), 2, max_calls=50, max_gpu_seconds=1e6)
        return run_evaluation(cfg, self.candidates, self.fn(fail), self.out, log=lambda *_: None)

    def rows(self):
        return list(csv.DictReader(io.StringIO((self.out / "comparison.csv").read_text())))

    def test_rows_stay_in_file_order_whatever_the_scores_are(self):
        self.go()
        rows = self.rows()
        self.assertEqual([r["candidate_id"] for r in rows], [r["candidate_id"] for r in self.records[:4]])      # flagged one is absent
        values = [float(r["esmfold2_iptm"]) for r in rows]
        self.assertEqual(values, [0.5, 0.9, 0.3, 0.7])
        self.assertNotIn(values, (sorted(values), sorted(values, reverse=True)))     # in neither score order
        self.assertEqual([float(r["boltz2_iptm"]) for r in rows], [0.8, 0.2, 0.6, 0.4])

    def test_difference_is_plain_arithmetic(self):
        self.go()
        first = self.rows()[0]
        self.assertAlmostEqual(float(first["esmfold2_minus_boltz2_iptm"]), -0.3, places=6)
        self.assertAlmostEqual(float(first["esmfold2_minus_boltz2_ipsae_max"]), -0.15, places=6)

    def test_a_failed_predictor_leaves_blanks_not_zeros(self):
        self.go(fail={("boltz2", 2)})
        row = self.rows()[1]
        self.assertEqual((row["boltz2_iptm"], row["esmfold2_minus_boltz2_iptm"]), ("", ""))
        self.assertEqual(float(row["esmfold2_iptm"]), 0.9)

    def test_pool_summary_is_the_spread_and_carries_no_verdict(self):
        report = self.go()
        summary = report["pool_summary"]
        esm = summary["by_predictor"]["esmfold2"]["iptm"]
        self.assertEqual((esm["n"], esm["min"], esm["max"]), (4, 0.3, 0.9))
        self.assertAlmostEqual(esm["median"], 0.6)
        paired = summary["paired_difference"]["esmfold2_minus_boltz2_iptm"]
        self.assertEqual(paired["n"], 4)
        self.assertAlmostEqual(paired["min"], -0.3)
        self.assertAlmostEqual(paired["max"], 0.7)
        banned = {"rank", "ranking", "pass", "fail", "threshold_met", "recommended", "best", "good", "score_total", "combined"}
        def keys(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    yield k.lower()
                    yield from keys(v)
        self.assertFalse(banned & set(keys(report)))
        self.assertFalse(banned & {c.lower() for c in self.rows()[0]})
        self.assertIn("no supportable pass/fail threshold", summary["note"])

    def test_one_predictor_gives_no_difference_columns(self):
        report = self.go(predictors=("boltz2",))
        self.assertEqual(report["pool_summary"]["paired_difference"], {})
        self.assertNotIn("esmfold2_minus_boltz2_iptm", self.rows()[0])

    def test_empty_pool_summary_when_nothing_finished(self):
        report = self.go(fail={("esmfold2", i) for i in range(1, 5)} | {("boltz2", i) for i in range(1, 5)})
        self.assertIsNone(report["pool_summary"]["by_predictor"]["esmfold2"]["iptm"])
        self.assertTrue(all(r["esmfold2_iptm"] == "" for r in self.rows()))


if __name__ == "__main__":
    unittest.main()
