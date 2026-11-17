import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _designs as D
from binder_campaign import batch

try:
    import tmtools  # noqa: F401
    HAVE_TM = True
except ImportError:
    HAVE_TM = False

RUN = unittest.skipUnless(D.AVAILABLE and HAVE_TM, "needs the examples, docs/eswar and tmtools")
if D.AVAILABLE and HAVE_TM:
    from binder_campaign import evaluate as EV, pool


def make_batch(root, designs, batch_id, kinds, hotspots=None, source="provisional_historical"):
    """Run the REAL batch engine with fake generation; `kinds` maps seed -> design kind."""
    seeds = list(kinds)

    def fn(task):
        return designs.result(task["run_id"], task["seed"], kinds[task["seed"]], length=60, worker_seconds=40.0)
    cfg = batch.make_config(batch_id, seeds, [60], hotspots or D.HOTSPOTS, source, max_designs=50, max_gpu_seconds=1e6, max_failures=5)
    out = Path(root) / batch_id
    batch.run_batch(cfg, fn, out, D.MANIFEST, log=lambda *_: None)
    return out


@RUN
class PoolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.designs = D.Designs(self.dir / "designs")
        # batch A: base + two different folds. batch B: an exact copy of A's base (a cross-batch duplicate) + a new fold + a twin
        self.a = make_batch(self.dir, self.designs, "pa", {1: "base", 2: "fold1", 3: "fold2"})
        self.b = make_batch(self.dir, self.designs, "pb", {11: "exact", 12: "fold3", 13: "twin"}, hotspots=[78, 79, 98], source="proposed")

    def merge(self, *batches, out="pool"):
        return pool.merge_batches(list(batches), self.dir / out, log=lambda *_: None)

    def records(self, out="pool"):
        return json.loads((self.dir / out / "candidates.json").read_text())["candidates"]

    def test_designs_are_copied_with_their_files_and_hashes_hold(self):
        report = self.merge(self.a, self.b)
        recs = self.records()
        self.assertEqual((report["designs"], len(recs)), (6, 6))
        for r in recs:
            folder = self.dir / "pool" / "designs" / r["pool"]["run_id"]
            self.assertTrue((folder / "model.cif").is_file() and (folder / "audit.json").is_file())
            self.assertEqual(r["generator_structure"], str(folder / "model.cif"))
            self.assertEqual(r["source_audit"], str(folder / "audit.json"))
            self.assertEqual(r["generator_audit"]["generated_structure"], str(folder / "model.cif"))
            self.assertEqual(batch.digest(folder / "model.cif"), r["generator_structure_sha256"])

    def test_a_cross_batch_duplicate_is_flagged_and_the_earlier_batch_wins(self):
        report = self.merge(self.a, self.b)
        recs = {r["seed"]: r for r in self.records()}
        self.assertEqual(recs[1]["qc_reasons"], [])                                     # batch A's base stays
        self.assertEqual(recs[11]["qc_reasons"], ["near_duplicate:exact_duplicate"])     # batch B's copy of it is dropped
        self.assertTrue(recs[11]["dedup_in_batch"]["kept"])                             # batch B alone had kept it
        self.assertFalse(recs[11]["dedup"]["kept"])
        self.assertEqual(recs[11]["dedup"]["nearest_id"], recs[1]["candidate_id"])
        self.assertEqual({c["candidate_id"]: c["reason"] for c in report["dropped_only_across_batches"]},
                         {recs[11]["candidate_id"]: "exact_duplicate", recs[13]["candidate_id"]: "backbone_tm"})
        self.assertEqual(recs[13]["qc_reasons"], ["near_duplicate:backbone_tm"])         # same backbone, new sequence: caught by the TM check
        self.assertEqual(recs[12]["qc_reasons"], [])                                    # a genuinely new fold survives
        self.assertEqual((report["kept_by_gate"], report["dropped_by_gate"]), (4, 2))

    def test_the_order_given_decides_who_wins(self):
        self.merge(self.b, self.a, out="pool_ba")
        recs = {r["seed"]: r for r in self.records("pool_ba")}
        self.assertEqual(recs[11]["qc_reasons"], [])                                    # B first: its copy is the kept one
        self.assertEqual(recs[1]["qc_reasons"], ["near_duplicate:exact_duplicate"])

    def test_the_hotspot_set_of_each_design_is_kept(self):
        recs = {r["seed"]: r for r in self.records_after_merge()}
        for path in (self.a, self.b):                                          # each design keeps exactly what its own batch recorded
            for own in json.loads((path / "candidates.json").read_text())["candidates"]:
                merged = recs[own["seed"]]
                self.assertEqual((merged["hotspots_uniprot"], merged["hotspot_source"]), (own["hotspots_uniprot"], own["hotspot_source"]))
        self.assertEqual((recs[1]["pool"]["batch_id"], recs[12]["pool"]["batch_id"]), ("pa", "pb"))

    def records_after_merge(self):
        self.merge(self.a, self.b)
        return self.records()

    def test_statuses_are_never_promoted_and_inputs_are_untouched(self):
        before = {p.name: (p / "candidates.json").read_bytes() for p in (self.a, self.b)}
        self.merge(self.a, self.b)
        for p in (self.a, self.b):
            self.assertEqual((p / "candidates.json").read_bytes(), before[p.name])
        for r in self.records():
            self.assertEqual((r["screen_status"], r["novelty_review"], r["independent_prediction_status"], r["submission_eligible"]),
                             ("not_run", "pending", "not_run", False))

    def test_reasons_other_than_near_duplicates_are_carried_over(self):
        marked = self.dir / "pm"
        shutil.copytree(self.a, marked)
        data = json.loads((marked / "candidates.json").read_text())
        data["candidates"][1]["qc_reasons"] = ["hotspots_not_reviewed"]
        (marked / "candidates.json").write_text(json.dumps(data))
        self.merge(marked, out="pool_m")
        recs = {r["seed"]: r for r in self.records("pool_m")}
        self.assertEqual(recs[2]["qc_reasons"], ["hotspots_not_reviewed"])
        self.assertEqual(recs[1]["qc_reasons"], [])

    def test_merging_one_batch_reproduces_its_own_decisions(self):
        self.merge(self.b, out="solo")
        own = {r["candidate_id"]: r for r in json.loads((self.b / "candidates.json").read_text())["candidates"]}
        for r in self.records("solo"):
            self.assertEqual((r["qc_reasons"], r["dedup"]["kept"], r["dedup"]["reason"]),
                             (own[r["candidate_id"]]["qc_reasons"], own[r["candidate_id"]]["dedup"]["kept"],
                              own[r["candidate_id"]]["dedup"]["reason"]))

    def test_the_merge_is_deterministic(self):
        self.merge(self.a, self.b, out="p1")
        self.merge(self.a, self.b, out="p2")
        strip = lambda recs: [{k: v for k, v in r.items() if k not in ("generator_structure", "source_audit", "generator_audit")} for r in recs]
        self.assertEqual(strip(self.records("p1")), strip(self.records("p2")))

    def test_the_evaluation_engine_accepts_the_pool_and_skips_the_flagged(self):
        self.merge(self.a, self.b)
        records = self.records()
        chosen, skipped = EV.select_candidates(records)
        self.assertEqual((len(chosen), len(skipped)), (4, 2))
        self.assertEqual(sorted(s["reasons"][0] for s in skipped), ["near_duplicate:backbone_tm", "near_duplicate:exact_duplicate"])

    # ---- refusals
    def test_refusals(self):
        self.merge(self.a)
        with self.assertRaisesRegex(ValueError, "fresh output"):
            self.merge(self.a)
        with self.assertRaisesRegex(ValueError, "twice"):
            self.merge(self.a, self.a, out="x1")
        with self.assertRaisesRegex(ValueError, "at least one"):
            pool.merge_batches([], self.dir / "x2")
        self.assertFalse((self.dir / "x1").exists())

    def test_a_batch_for_a_different_target_or_thresholds_is_refused(self):
        other = self.dir / "pc"
        shutil.copytree(self.b, other)
        data = json.loads((other / "candidates.json").read_text())
        data["candidates"][0]["target_manifest_sha256"] = "0" * 64
        (other / "candidates.json").write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "different target manifest sha256"):
            self.merge(self.a, other, out="x3")
        third = self.dir / "pd"
        shutil.copytree(self.b, third)
        state = json.loads((third / "state.json").read_text())
        state["identity"]["thresholds"]["tm"] = 0.5
        (third / "state.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "different near-duplicate thresholds"):
            self.merge(self.a, third, out="x4")

    def test_duplicate_ids_across_batches_are_refused(self):
        clone = self.dir / "pe"
        shutil.copytree(self.a, clone)
        with self.assertRaisesRegex(ValueError, "Duplicate candidate id"):
            self.merge(self.a, clone, out="x5")

    def test_a_tampered_structure_is_refused(self):
        run = next(p for p in (self.b / "designs").iterdir())
        (run / "model.cif").write_bytes((run / "model.cif").read_bytes() + b"\n# changed\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.merge(self.a, self.b, out="x6")


@RUN
class MergeScript(unittest.TestCase):
    SCRIPT = Path(__file__).resolve().parents[1] / "scripts/merge-pool.py"

    def test_end_to_end_and_refusals(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            designs = D.Designs(temp / "designs")
            a = make_batch(temp, designs, "pa", {1: "base", 2: "fold1"})
            b = make_batch(temp, designs, "pb", {11: "exact", 12: "fold2"})
            run = lambda *args: subprocess.run([sys.executable, "-I", str(self.SCRIPT), *args], capture_output=True, text=True,
                                               env={"PATH": "/usr/bin:/bin"})
            done = run("--batch", str(a), "--batch", str(b), "--out", str(temp / "pool"))
            self.assertEqual(done.returncode, 0, done.stderr)
            summary = json.loads(done.stdout)
            self.assertEqual((summary["designs"], summary["kept_by_gate"], summary["dropped_by_gate"]), (4, 3, 1))
            again = run("--batch", str(a), "--out", str(temp / "pool"))
            self.assertTrue(again.returncode != 0 and "REFUSED: Use a fresh output directory" in again.stderr and "Traceback" not in again.stderr)
            missing = run("--batch", str(temp / "nope"), "--out", str(temp / "p9"))
            self.assertTrue(missing.returncode != 0 and "REFUSED" in missing.stderr and "Traceback" not in missing.stderr, missing.stderr)


if __name__ == "__main__":
    unittest.main()
