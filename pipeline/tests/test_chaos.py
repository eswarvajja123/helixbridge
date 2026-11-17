"""Randomised stress tests for the generation batch engine and the evaluation engine.

Each scenario is built from a fixed seed (so a failure is reproducible: the seed is in the message). A scenario injects random
failures and Ctrl-C interruptions, changes the budget between runs, resumes several times, and then checks invariants that must
hold whatever happened. This is the kind of test that finds the bug nobody thought to write a case for.
"""
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _designs as D
from binder_campaign import batch, evaluate as EV
from binder_campaign.dedup import DedupGate, Thresholds, read_chain_ca
from binder_campaign.scoring import build_score

try:
    import tmtools  # noqa: F401
    HAVE_TM = True
except ImportError:
    HAVE_TM = False

BATCH_SCENARIOS = range(40)
EVAL_SCENARIOS = range(80)
KINDS = ["base", "fold1", "fold2", "fold3", "exact", "twin", "mutant"]
ROWS = [{"chain1": "A", "chain2": "B", "type": "asym", "ipsae": 0.5}, {"chain1": "B", "chain2": "A", "type": "asym", "ipsae": 0.6},
        {"chain1": "A", "chain2": "B", "type": "max", "ipsae": 0.6}]


@unittest.skipUnless(D.AVAILABLE and HAVE_TM, "needs the examples, docs/eswar and tmtools")
class BatchChaos(unittest.TestCase):
    def scenario(self, number):
        rng = random.Random(1000 + number)
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            designs = D.Designs(temp / "designs")
            seeds = list(range(1, rng.randint(2, 6) + 1))
            lengths = rng.choice([[60], [50, 60]])
            kinds = {s: rng.choice(KINDS) for s in seeds}
            tasks = len(seeds) * len(lengths)
            calls, outcomes = [], {}

            def outcome(index):                       # a fixed script per call number, so a scenario replays identically
                return outcomes.setdefault(index, rng.choices(["ok", "error", "interrupt"], [70, 22, 8])[0])

            def fn(task):
                calls.append(task["run_id"])
                result = outcome(len(calls))
                if result == "error":
                    raise RuntimeError("scripted failure")
                if result == "interrupt":
                    raise KeyboardInterrupt
                return designs.result(task["run_id"], task["seed"], kinds[task["seed"]], length=task["binder_length"][0], worker_seconds=40.0)

            out = temp / "out"
            cap_calls, cap_seconds = rng.randint(1, tasks * 2), rng.choice([60.0, 150.0, 1e6])
            history = []
            for phase in range(rng.randint(2, 6)):
                cfg = batch.make_config("c1", seeds, lengths, D.HOTSPOTS, "provisional_historical", max_designs=cap_calls,
                                        max_gpu_seconds=cap_seconds, max_failures=rng.randint(1, 4))
                before = len(calls)
                try:
                    report = batch.run_batch(cfg, fn, out, D.MANIFEST, retry_failed=rng.random() < 0.5,
                                             retry_interrupted=rng.random() < 0.5, log=lambda *_: None)
                except KeyboardInterrupt:
                    report = json.loads((out / "batch-report.json").read_text())
                    self.assertEqual(report["stop_reason"], "interrupted_by_user", f"scenario {number}")
                self.check_invariants(number, report, out, calls, cap_calls, cap_seconds, tasks, kinds, before)
                history.append(report["stop_reason"])
                cap_calls += rng.randint(0, tasks)                       # budgets may only be raised when resuming
                cap_seconds += rng.choice([0.0, 100.0, 1e6])
            # a final, generous, failure-free run that retries everything must finish every design
            return self.finish(number, seeds, lengths, kinds, temp, out, designs, calls, tasks)

    def finish(self, number, seeds, lengths, kinds, temp, out, designs, calls, tasks):
        def fn(task):
            calls.append(task["run_id"])
            return designs.result(task["run_id"], task["seed"], kinds[task["seed"]], length=task["binder_length"][0], worker_seconds=40.0)
        cfg = batch.make_config("c1", seeds, lengths, D.HOTSPOTS, "provisional_historical", max_designs=10 ** 6, max_gpu_seconds=1e9,
                                max_failures=10 ** 6)
        report = batch.run_batch(cfg, fn, out, D.MANIFEST, retry_failed=True, retry_interrupted=True, log=lambda *_: None)
        self.assertEqual((report["stop_reason"], report["done"], report["planned"]), ("completed", tasks, tasks), f"scenario {number}")
        return report

    def check_invariants(self, number, report, out, calls, cap_calls, cap_seconds, tasks, kinds, before):
        tag = f"scenario {number}"
        state = json.loads((out / "state.json").read_text())
        entries = state["tasks"]
        # 1. every launch is accounted for, and the attempts cap is respected
        self.assertEqual(sum(e["attempts"] for e in entries.values()), len(calls), tag)
        self.assertLessEqual(len(calls), max(cap_calls, before), tag)
        # 2. the reported partition adds up
        self.assertEqual(report["done"] + len(report["failed"]) + len(report["interrupted"]) + len(report["not_started"]), tasks, tag)
        # 3. a design is never launched twice with the same run id, and never again once it is done
        self.assertEqual(len(calls), len(set(calls)), tag)
        # 4. the GPU-seconds cap is overshot by at most one design (40 s each here)
        self.assertLessEqual(report["gpu_seconds_reported"], max(cap_seconds, 0) + 40.0 + (0 if report["gpu_seconds_reported"] <= cap_seconds else 0), tag)
        # 5. candidates.json holds exactly the finished designs, in offer order
        records = json.loads((out / "candidates.json").read_text())["candidates"]
        done = sorted((e for e in entries.values() if e["status"] == "done"), key=lambda e: e["offer_index"])
        self.assertEqual(len(records), len(done), tag)
        self.assertEqual([r["seed"] for r in records], [e["seed"] for e in done], tag)
        self.assertEqual(len({e["offer_index"] for e in done}), len(done), tag)
        # 6. replay determinism: offering the finished designs again, in the recorded order, reproduces every recorded decision
        gate = DedupGate(Thresholds(**state["identity"]["thresholds"]))
        for entry in done:
            folder = out / "designs" / entry["run_id"]
            audit = json.loads((folder / "audit.json").read_text())
            sequence, coords = read_chain_ca(folder / "model.cif", audit["binder_chain"])
            decision = gate.offer(entry["run_id"], sequence, coords)
            self.assertEqual((decision.kept, decision.reason), (entry["decision"]["kept"], entry["decision"]["reason"]), tag)
        # 7. a dropped design is never exportable; a kept one has no QC reasons
        for record, entry in zip(records, done):
            self.assertEqual(bool(record["qc_reasons"]), not entry["decision"]["kept"], tag)

    def test_random_scenarios(self):
        for number in BATCH_SCENARIOS:
            self.scenario(number)


@unittest.skipUnless(D.AVAILABLE, "needs the examples and docs/eswar")
class EvaluateChaos(unittest.TestCase):
    @staticmethod
    def score(predictor, cid):
        shift = int(hashlib.sha256((predictor + cid).encode()).hexdigest()[:2], 16) / 1000
        return build_score(predictor, iptm=0.5 + shift, ptm=0.7, rows=[{**r, "ipsae": round(r["ipsae"] * 0.5 + shift, 6)} for r in ROWS],
                           seconds=5.0, peak_reserved_mib=1000.0, plddt_raw_mean=0.8 if predictor == "esmfold2" else None, gpu="Fake")

    def response(self, request, fail_ids=()):
        results, members = [], {}
        for c in request["candidates"]:
            if c["id"] in fail_ids:
                results.append({"id": c["id"], "status": "failed", "error": "scripted"})
                continue
            results.append({"id": c["id"], "status": "ok", "sequence": c["sequence"], "score": self.score(request["predictor"], c["id"])})
            for name in EV.FILES:
                members[f"{request['run_id']}/{c['id']}/{name}"] = f"{name}".encode()
        return {"run_id": request["run_id"], "gpu": "Fake", "worker_seconds": 30.0, "results": results, "archive": D.archive(members)}

    def scenario(self, number):
        rng = random.Random(5000 + number)
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            n = rng.randint(3, 9)
            records = [{"candidate_id": f"il7ra-rfd4-{i:012x}", "sequence": D.random_sequence(900 + i), "seed": i, "qc_reasons":
                        (["near_duplicate:x"] if rng.random() < 0.2 else []), "screen_status": "not_run", "novelty_review": "pending",
                        "independent_prediction_status": "not_run", "submission_eligible": False} for i in range(1, n + 1)]
            if all(r["qc_reasons"] for r in records):
                records[0]["qc_reasons"] = []
            flagged = {r["candidate_id"] for r in records if r["qc_reasons"]}
            eligible = [r for r in records if not r["qc_reasons"]]
            candidates = temp / "candidates.json"
            candidates.write_text(json.dumps({"candidates": records}))
            predictors = rng.choice([["esmfold2"], ["boltz2"], ["esmfold2", "boltz2"]])
            chunk = rng.randint(1, 4)
            sent, outcomes = [], {}

            def fn(request):
                sent.append(request)
                index = len(sent)
                kind = outcomes.setdefault(index, rng.choices(["ok", "partial", "crash", "interrupt"], [55, 25, 12, 8])[0])
                if kind == "crash":
                    raise RuntimeError("scripted crash")
                if kind == "interrupt":
                    raise KeyboardInterrupt
                fail = {c["id"] for c in request["candidates"] if rng.random() < 0.4} if kind == "partial" else set()
                return self.response(request, fail)

            out = temp / "out"
            cap_calls, cap_seconds = rng.randint(1, 8), rng.choice([60.0, 200.0, 1e6])
            finished = set()
            for phase in range(rng.randint(2, 6)):
                cfg = EV.make_config("c1", predictors, chunk, max_calls=cap_calls, max_gpu_seconds=cap_seconds, max_failures=rng.randint(1, 6))
                before = len(sent)
                try:
                    report = EV.run_evaluation(cfg, candidates, fn, out, retry_failed=rng.random() < 0.5,
                                               retry_interrupted=rng.random() < 0.5, log=lambda *_: None)
                except KeyboardInterrupt:
                    report = json.loads((out / "evaluation-report.json").read_text())
                tag = f"scenario {number}"
                state = json.loads((out / "state.json").read_text())
                self.assertEqual(state["calls"], len(sent), tag)
                self.assertLessEqual(len(sent), max(cap_calls, before), tag)
                self.assertTrue(all(c["id"] not in flagged for r in sent for c in r["candidates"]), tag)       # flagged are never sent
                self.assertEqual(len({r["run_id"] for r in sent}), len(sent), tag)
                for predictor in predictors:
                    counts = report["by_predictor"][predictor]
                    self.assertEqual(sum(counts.values()), len(eligible), tag)
                self.assertLessEqual(report["gpu_seconds_reported"], cap_seconds + 30.0 if cap_seconds < 1e6 else 1e7, tag)
                self.assertEqual(len(report["candidates_skipped"]), len(flagged), tag)
                for request in sent[before:]:                      # nothing finished in an earlier phase is ever sent again
                    for c in request["candidates"]:
                        self.assertNotIn((c["id"], request["predictor"]), finished, f"{tag}: finished pair sent again")
                finished |= {(cid, p) for cid, by in state["results"].items() for p, e in by.items() if e["status"] == "done"}
                cap_calls += rng.randint(0, 6)
                cap_seconds += rng.choice([0.0, 100.0, 1e6])
            # a final generous failure-free run that retries everything must finish every pair, with the right scores
            final = EV.make_config("c1", predictors, chunk, max_calls=10 ** 6, max_gpu_seconds=1e9, max_failures=10 ** 6)
            report = EV.run_evaluation(final, candidates, lambda r: self.response(r), out, retry_failed=True, retry_interrupted=True,
                                       log=lambda *_: None)
            tag = f"scenario {number}"
            self.assertEqual(report["stop_reason"], "completed", tag)
            for item in json.loads((out / "evaluations.json").read_text())["candidates"]:
                if item["candidate_id"] in flagged:
                    self.assertEqual(item["evaluations"], {}, tag)
                    continue
                for predictor in predictors:
                    got = item["evaluations"][predictor]
                    self.assertEqual(got["status"], "done", tag)
                    self.assertEqual(got["score"]["iptm"], self.score(predictor, item["candidate_id"])["iptm"], tag)   # order-independent

    def test_random_scenarios(self):
        for number in EVAL_SCENARIOS:
            self.scenario(number)


if __name__ == "__main__":
    unittest.main()
