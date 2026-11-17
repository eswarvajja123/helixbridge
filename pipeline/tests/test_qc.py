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

try:
    import numpy as np
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict  # noqa: F401
    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False

if HAVE_STACK and D.AVAILABLE:
    from support import synth_structures as S
    from binder_campaign import evaluate as EV, qc
    from binder_campaign.evaluate import make_config, run_evaluation
    from binder_campaign.scoring import build_score

TARGET = json.loads(D.MANIFEST.read_text()) if D.AVAILABLE else None
ROWS = [{"chain1": "A", "chain2": "B", "type": "asym", "ipsae": 0.5}, {"chain1": "B", "chain2": "A", "type": "asym", "ipsae": 0.6},
        {"chain1": "A", "chain2": "B", "type": "max", "ipsae": 0.6}]
HOTSPOTS = D.HOTSPOTS
RUN = unittest.skipUnless(HAVE_STACK and D.AVAILABLE, "needs numpy, Biopython and the examples")


def sha(data):
    return hashlib.sha256(data).hexdigest()


class World:
    """Builds records, runs the REAL evaluation engine with a fake predictor that writes chosen structures, then runs QC."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.core, self.real_binder = S.real_sequences()
        self.records, self.refolds, self.shifts = [], {}, {}

    def add(self, name, sequence=None, shift=None, hotspots=HOTSPOTS, qc_reasons=(), generator=True, esm=(0, 0, 0), boltz=(0, 0, 0),
            esm_extra=(), esm_third_chain=False, esm_binder=None, boltz_bytes=None, skip=()):
        sequence = sequence or D.DESIGN
        folder = self.dir / "batch" / "designs" / name
        folder.mkdir(parents=True, exist_ok=True)
        has_structures = len(sequence) == len(self.real_binder)         # the real binder has 60 residues
        data = S.generator_structure(sequence) if has_structures else b""
        (folder / "model.cif").write_bytes(data)
        record = {"candidate_id": f"il7ra-rfd4-{name}", "target_id": "il7ra", "sequence": sequence, "seed": len(self.records) + 1,
                  "qc_reasons": list(qc_reasons), "screen_status": "not_run", "novelty_review": "pending",
                  "independent_prediction_status": "not_run", "submission_eligible": False,
                  "hotspots_uniprot": hotspots,
                  "generator_structure": str(folder / "model.cif") if generator else "",
                  "generator_audit": {"generated_sha256": sha(data), "reference_audit": {
                      "reference_sequence": self.core, "positions": list(range(S.CORE_START, S.CORE_END + 1))}}}
        self.records.append(record)
        if not has_structures:
            record["generator_structure"] = ""
            self.refolds[record["candidate_id"]] = {"esmfold2": None, "boltz2": None}
            return record
        self.refolds[record["candidate_id"]] = {
            "esmfold2": S.refold_structure(TARGET["sequence"], sequence, esm, esm_extra, esm_third_chain, esm_binder),
            "boltz2": boltz_bytes if boltz_bytes is not None else S.refold_structure(TARGET["sequence"], sequence, boltz)}
        for predictor in skip:
            self.refolds[record["candidate_id"]][predictor] = None
        return record

    def evaluate(self, predictors=("esmfold2", "boltz2")):
        self.candidates = self.dir / "candidates.json"
        self.candidates.write_text(json.dumps({"candidates": self.records}))
        self.eval = self.dir / "eval"
        refolds = self.refolds

        def fn(request):
            results, members = [], {}
            for c in request["candidates"]:
                data = refolds[c["id"]][request["predictor"]]
                if data is None:
                    results.append({"id": c["id"], "status": "failed", "error": "scripted"})
                    continue
                score = build_score(request["predictor"], iptm=0.8, ptm=0.8, rows=ROWS, seconds=5.0, peak_reserved_mib=1.0,
                                    plddt_raw_mean=0.8 if request["predictor"] == "esmfold2" else None, gpu="Fake")
                results.append({"id": c["id"], "status": "ok", "sequence": c["sequence"], "score": score})
                for name in EV.FILES:
                    members[f"{request['run_id']}/{c['id']}/{name}"] = data if name == "model.cif" else b"{}"
            return {"run_id": request["run_id"], "gpu": "Fake", "worker_seconds": 5.0, "results": results, "archive": D.archive(members)}

        cfg = make_config("e1", list(predictors), 3, max_calls=50, max_gpu_seconds=1e6, include_flagged=True)
        run_evaluation(cfg, self.candidates, fn, self.eval, log=lambda *_: None)
        return self

    def qc(self, thresholds=None, out="qc", expected=qc.SOURCES if (HAVE_STACK and D.AVAILABLE) else ()):
        report = qc.run_qc(self.eval, D.MANIFEST, self.dir / out, thresholds, self.dir / "batch", expected, log=lambda *_: None)
        self.out = self.dir / out
        self.result = {r["candidate_id"]: r for r in json.loads((self.out / "qc-candidates.json").read_text())["candidates"]}
        return report

    def of(self, name):
        return self.result[f"il7ra-rfd4-{name}"]


@RUN
class QcCases(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.w = World(self.temp.name)

    def run_one(self, thresholds=None, **kwargs):
        self.w.add("a", **kwargs)
        self.w.evaluate()
        self.w.qc(thresholds)
        return self.w.of("a")

    # ---- the clean case, checked against an independent calculation
    def test_a_design_with_no_problems_gets_no_flags_and_real_measurements(self):
        item = self.run_one()
        self.assertEqual(item["qc"]["status"], "no_flags")
        self.assertEqual(item["qc_reasons"], [])
        close, minimum, contacts = S.brute_force(self.w.refolds[item["candidate_id"]]["esmfold2"])
        # the planted brute-force reads the file's label_seq_id (1..219); pose_review reports UniProt numbers
        uniprot = [c + S.TARGET_START - 1 for c in contacts]
        esm = item["qc"]["sources"]["esmfold2"]
        self.assertEqual(esm["target_contacts"], uniprot)
        self.assertEqual((esm["heavy_atom_pairs_lt_2A"], round(esm["min_heavy_distance_A"], 3)), (close, round(minimum, 3)))
        self.assertTrue(set(esm["hotspot_contacts"]) and set(esm["hotspot_contacts"]) <= set(HOTSPOTS))
        for name in ("generator_vs_esmfold2", "generator_vs_boltz2", "esmfold2_vs_boltz2"):
            self.assertAlmostEqual(item["qc"]["pose_shift"][name]["binder_ca_rmsd_A"], 0.0, places=3)
        self.assertEqual(item["qc"]["alignment_core"], [S.CORE_START, S.CORE_END])

    # ---- length
    def test_length_outside_40_to_100_is_flagged(self):
        for length, expect in ((39, True), (40, False), (100, False), (101, True)):
            with self.subTest(length=length):
                d = World(tempfile.mkdtemp(dir=self.temp.name))
                d.add("a", sequence=D.random_sequence(5, length))
                d.evaluate(); d.qc(expected=())
                reasons = d.of("a")["qc"]["reasons"]
                self.assertEqual(any(r.startswith("length_out_of_range") for r in reasons), expect)

    # ---- chains
    def test_a_third_chain_is_not_a_single_binder_chain(self):
        item = self.run_one(esm_third_chain=True)
        self.assertIn("unexpected_chains:esmfold2:found_3_expected_2", item["qc_reasons"])
        self.assertEqual(item["qc"]["sources"]["esmfold2"]["status"], "unreadable")
        self.assertEqual(item["qc"]["status"], "flagged")

    def test_a_binder_with_a_different_sequence_is_flagged(self):
        other = list(D.DESIGN)
        other[7] = "G" if other[7] != "G" else "A"
        item = self.run_one(esm_binder="".join(other))
        self.assertIn("binder_sequence_mismatch:esmfold2", item["qc_reasons"])

    def test_unreadable_and_altered_structures_are_flagged_not_crashed_on(self):
        self.w.add("a", boltz_bytes=b"this is not a structure")
        self.w.evaluate()
        self.w.qc()
        reasons = self.w.of("a")["qc_reasons"]
        self.assertTrue(any(r.startswith("structure_unreadable:boltz2") for r in reasons), reasons)
        # a file changed after the evaluation stored its hash
        victim = self.w.eval / "evals" / "il7ra-rfd4-a" / "esmfold2" / "model.cif"
        victim.write_bytes(victim.read_bytes() + b"\n# edited\n")
        self.w.qc(out="qc2")
        self.assertIn("structure_hash_mismatch:esmfold2", self.w.of("a")["qc_reasons"])

    # ---- clash
    def test_a_planted_close_pair_is_counted_exactly(self):
        point = S.clash_atom()
        item = self.run_one(esm_extra=[(3, "OX", "O", point)])
        close, _, _ = S.brute_force(self.w.refolds[item["candidate_id"]]["esmfold2"])
        self.assertGreaterEqual(close, 1)
        self.assertIn(f"clash:esmfold2:{close}_heavy_atom_pairs_lt_2A", item["qc_reasons"])
        self.assertEqual(item["qc"]["sources"]["boltz2"]["heavy_atom_pairs_lt_2A"], 0)
        self.assertFalse([r for r in item["qc_reasons"] if r.startswith("clash:boltz2")])

    def test_the_clash_limit_is_a_parameter(self):
        point = S.clash_atom()
        item = self.run_one(thresholds=qc.Thresholds(max_close_pairs=50), esm_extra=[(3, "OX", "O", point)])
        self.assertFalse([r for r in item["qc_reasons"] if r.startswith("clash")])

    # ---- contact
    def test_a_binder_far_from_the_target_has_no_contact(self):
        item = self.run_one(esm=(0, 0, 80))
        self.assertIn("no_target_contact:esmfold2", item["qc_reasons"])
        self.assertEqual(item["qc"]["sources"]["esmfold2"]["target_contact_residues"], 0)
        self.assertIn("pose_shift:generator_vs_esmfold2:80.00A>2.5A", item["qc_reasons"])

    def test_contact_that_misses_every_hotspot_is_flagged_and_can_be_switched_off(self):
        item = self.run_one(hotspots=[60])
        self.assertIn("no_hotspot_contact:generator", item["qc_reasons"])
        self.assertIn("no_hotspot_contact:esmfold2", item["qc_reasons"])
        self.assertIn("no_hotspot_contact:boltz2", item["qc_reasons"])
        d = World(tempfile.mkdtemp(dir=self.temp.name))
        d.add("a", hotspots=[60]); d.evaluate(); d.qc(qc.Thresholds(require_hotspot_contact=False))
        self.assertEqual(d.of("a")["qc"]["status"], "no_flags")
        d2 = World(tempfile.mkdtemp(dir=self.temp.name))
        d2.add("a", hotspots=None); d2.evaluate(); d2.qc()
        self.assertEqual(d2.of("a")["qc"]["status"], "no_flags")                     # no hotspots recorded: nothing to test

    # ---- pose shift
    def test_pose_shift_is_the_rigid_displacement_and_the_limits_are_applied(self):
        item = self.run_one(esm=(0, 0, 2.0), boltz=(0, 0, -1.0))
        shifts = item["qc"]["pose_shift"]
        self.assertAlmostEqual(shifts["generator_vs_esmfold2"]["binder_ca_rmsd_A"], 2.0, places=3)
        self.assertAlmostEqual(shifts["generator_vs_boltz2"]["binder_ca_rmsd_A"], 1.0, places=3)
        self.assertAlmostEqual(shifts["esmfold2_vs_boltz2"]["binder_ca_rmsd_A"], 3.0, places=3)
        self.assertFalse([r for r in item["qc_reasons"] if r.startswith("pose_shift")])        # 2.0 <= 2.5, 3.0 <= 3.0 (limit is inclusive)

    def test_each_pose_shift_limit_is_enforced_separately(self):
        item = self.run_one(esm=(0, 0, 1.8), boltz=(0, 0, -1.8))                     # each within 2.5, but 3.6 apart (> 3.0)
        reasons = [r for r in item["qc_reasons"] if r.startswith("pose_shift")]
        self.assertEqual(len(reasons), 1, reasons)
        self.assertTrue(reasons[0].startswith("pose_shift:esmfold2_vs_boltz2:3.60A>3.0A"))
        w2 = World(tempfile.mkdtemp(dir=self.temp.name))
        w2.add("a", esm=(0, 0, 3.0), boltz=(0, 0, 3.0)); w2.evaluate(); w2.qc()           # together, but 3.0 from the generated pose
        reasons = [r for r in w2.of("a")["qc_reasons"] if r.startswith("pose_shift")]
        self.assertEqual(len(reasons), 2, reasons)
        self.assertTrue(all("generator_vs" in r for r in reasons))

    # ---- incompleteness is never reported as clean
    def test_a_missing_predictor_makes_the_result_incomplete_not_clean(self):
        item = self.run_one(skip=("boltz2",))
        self.assertEqual(item["qc"]["status"], "incomplete")
        self.assertEqual(item["qc"]["unchecked"], ["boltz2"])
        self.assertEqual(item["qc"]["sources"]["boltz2"], {"status": "not_checked"})
        self.assertNotIn("esmfold2_vs_boltz2", item["qc"]["pose_shift"])

    def test_a_candidate_with_no_structures_at_all_is_incomplete(self):
        self.w.add("a", generator=False, skip=("esmfold2", "boltz2"))
        self.w.evaluate()
        self.w.qc()
        self.assertEqual(self.w.of("a")["qc"]["status"], "incomplete")
        self.assertEqual(self.w.of("a")["qc"]["unchecked"], ["generator", "esmfold2", "boltz2"])

    # ---- what the output promises
    def test_originals_are_untouched_and_existing_reasons_are_kept(self):
        self.w.add("a", qc_reasons=["near_duplicate:exact_duplicate"], esm=(0, 0, 80))
        self.w.add("b")
        self.w.evaluate()
        before = (self.w.eval / "evaluations.json").read_bytes()
        stored = json.loads(before)["candidates"]
        self.w.qc()
        self.assertEqual((self.w.eval / "evaluations.json").read_bytes(), before)
        item = self.w.of("a")
        self.assertEqual(item["qc_reasons"][0], "near_duplicate:exact_duplicate")
        self.assertIn("no_target_contact:esmfold2", item["qc_reasons"])
        for original in stored:
            changed = {k for k in item if k in original and item[k] != original[k]} if item["candidate_id"] == original["candidate_id"] else set()
            self.assertLessEqual(changed, {"qc_reasons"})
        clean = self.w.of("b")
        for key in ("screen_status", "novelty_review", "independent_prediction_status", "submission_eligible"):
            self.assertEqual(clean[key], self.w.records[1][key])
        self.assertIs(clean["submission_eligible"], False)

    def test_a_qc_output_cannot_be_qcd_again_and_the_folder_must_be_fresh(self):
        self.w.add("a")
        self.w.evaluate()
        self.w.qc()
        with self.assertRaisesRegex(ValueError, "fresh output"):
            self.w.qc()
        shutil_dir = self.w.out
        (self.w.dir / "again").mkdir()
        (self.w.dir / "again" / "evaluations.json").write_bytes((shutil_dir / "qc-candidates.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "already holds a QC result"):
            qc.run_qc(self.w.dir / "again", D.MANIFEST, self.w.dir / "qc3", log=lambda *_: None)

    def test_csv_and_report(self):
        self.w.add("a")
        self.w.add("b", esm=(0, 0, 80))
        self.w.add("c", skip=("boltz2",))
        self.w.evaluate()
        report = self.w.qc()
        self.assertEqual(report["status_counts"], {"flagged": 1, "no_flags": 1, "incomplete": 1})
        self.assertEqual(report["reason_counts"]["no_target_contact:esmfold2"], 1)
        self.assertIn("not evidence that a design is good", report["scope"])
        rows = list(csv.DictReader(io.StringIO((self.w.out / "qc.csv").read_text())))
        self.assertEqual([r["candidate_id"] for r in rows], ["il7ra-rfd4-a", "il7ra-rfd4-b", "il7ra-rfd4-c"])
        self.assertEqual([r["qc_status"] for r in rows], ["no_flags", "flagged", "incomplete"])
        self.assertEqual(rows[1]["esmfold2_target_contact_residues"], "0")
        self.assertEqual(rows[2]["boltz2_status"], "not_checked")
        self.assertEqual(float(rows[1]["generator_vs_esmfold2_A"]), 80.0)

    def test_thresholds_are_validated(self):
        for bad in ({"max_close_pairs": -1}, {"max_close_pairs": 1.5}, {"min_target_contact_residues": True},
                    {"max_pose_shift_generator_A": 0}, {"max_pose_shift_between_predictors_A": float("nan")},
                    {"require_hotspot_contact": "yes"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                qc.Thresholds(**bad)


@RUN
class DamagedFiles(unittest.TestCase):
    """Whatever is wrong with a structure file, QC reports a reason for that candidate and carries on; it never crashes."""

    def test_random_damage_never_escapes(self):
        import random
        core, binder = S.real_sequences()
        good = S.refold_structure(TARGET["sequence"], binder).decode()
        record = {"candidate_id": "x", "sequence": binder, "hotspots_uniprot": HOTSPOTS,
                  "generator_audit": {"reference_audit": {"reference_sequence": core, "positions": list(range(37, 232))}}}
        rng = random.Random(7)
        outcomes = set()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.cif"
            for case in range(150):
                lines = good.splitlines()
                kind = rng.choice(["truncate", "drop_line", "garble", "drop_header", "empty", "binary", "nan", "swap_chain", "reorder"])
                if kind == "truncate":
                    text = good[:rng.randrange(0, len(good))]
                elif kind == "drop_line":
                    del lines[rng.randrange(len(lines))]
                    text = "\n".join(lines)
                elif kind == "garble":
                    for _ in range(20):
                        i = rng.randrange(len(lines))
                        lines[i] = "".join(rng.choice("x9.-# ") if rng.random() < 0.15 else c for c in lines[i])
                    text = "\n".join(lines)
                elif kind == "drop_header":
                    text = "\n".join(l for l in lines if not l.startswith("_atom_site.label_seq_id"))
                elif kind == "empty":
                    text = ""
                elif kind == "binary":
                    text = "".join(chr(rng.randrange(0, 255)) for _ in range(400))
                elif kind == "nan":
                    text = good.replace(" 1 ", " nan ", 1) if rng.random() < 0.5 else good.replace(".", "nan", 5)
                elif kind == "swap_chain":
                    text = good.replace(" B ", " A ")
                else:
                    rng.shuffle(lines)
                    text = "\n".join(lines)
                path.write_text(text, errors="replace")
                try:
                    got = qc.check_candidate(record, {"esmfold2": {"path": path}}, TARGET, expected=("esmfold2",))
                except Exception as error:           # noqa: BLE001  (this is exactly what the test forbids)
                    self.fail(f"case {case} ({kind}) escaped with {type(error).__name__}: {error}")
                outcomes.add(got["status"])
                self.assertIn(got["status"], {"flagged", "no_flags", "incomplete"})
                if got["status"] == "flagged":
                    self.assertTrue(got["reasons"])
        self.assertIn("flagged", outcomes)


@RUN
class RealStructures(unittest.TestCase):
    """The verified RFD4 design and its Boltz-2 refold, both REAL files, run through the same code (historical target numbering)."""

    def test_real_generator_and_real_boltz_refold(self):
        real = Path(__file__).resolve().parents[2] / "examples/verified-modal-run"
        core, binder = S.real_sequences()
        from binder_campaign.dedup import read_chain_ca
        target_seq, _ = read_chain_ca(real / "boltz.cif", "A")
        historical = {"sequence": target_seq, "start": 1, "end": 236, "min_length": 40, "max_length": 100}
        record = {"candidate_id": "real", "sequence": binder, "hotspots_uniprot": HOTSPOTS,
                  "generator_audit": {"reference_audit": {"reference_sequence": core, "positions": list(range(37, 232))}}}
        got = qc.check_candidate(record, {"generator": {"path": real / "generator.cif"}, "boltz2": {"path": real / "boltz.cif"}},
                                 historical, expected=("generator", "boltz2"))
        self.assertEqual(got["status"], "no_flags", got["reasons"])
        self.assertEqual(got["sources"]["generator"]["target_contact_residues"], 21)
        self.assertEqual(got["sources"]["boltz2"]["target_contact_residues"], 20)
        self.assertEqual(got["sources"]["generator"]["hotspot_contacts"], [100, 101, 102, 158, 159, 161, 211, 212, 213])
        self.assertEqual(got["sources"]["boltz2"]["heavy_atom_pairs_lt_2A"], 0)
        self.assertAlmostEqual(got["pose_shift"]["generator_vs_boltz2"]["binder_ca_rmsd_A"], 0.8925, places=3)
        close, minimum, contacts = S.brute_force(S.write_cif(S.real_atoms()))
        self.assertEqual((close, round(minimum, 2)), (0, 2.24))
        self.assertEqual(len(contacts), 21)


@RUN
class QcScript(unittest.TestCase):
    SCRIPT = Path(__file__).resolve().parents[1] / "scripts/qc-candidates.py"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.w = World(self.temp.name)
        self.w.add("a")
        self.w.add("b", esm=(0, 0, 80))
        self.w.evaluate()

    def call(self, *extra, out="out"):
        import subprocess
        return subprocess.run([sys.executable, "-I", str(self.SCRIPT), "--evaluation", str(self.w.eval), "--batch-dir",
                               str(self.w.dir / "batch"), "--target-manifest", str(D.MANIFEST), "--out", str(self.w.dir / out), *extra],
                              capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})

    def test_runs_end_to_end_and_prints_a_summary(self):
        done = self.call()
        self.assertEqual(done.returncode, 0, done.stderr)
        summary = json.loads(done.stdout)
        self.assertEqual(summary["status_counts"], {"flagged": 1, "no_flags": 1, "incomplete": 0})
        self.assertTrue((self.w.dir / "out" / "qc-candidates.json").is_file())
        self.assertIn("il7ra-rfd4-b: flagged", done.stderr)

    def test_options_reach_the_engine(self):
        done = self.call("--max-shift-generator", "100", "--min-contact-residues", "0", "--no-hotspot-requirement", "--sources", "generator,esmfold2")
        self.assertEqual(done.returncode, 0, done.stderr)
        report = json.loads(done.stdout)
        self.assertEqual(report["thresholds"]["max_pose_shift_generator_A"], 100.0)
        self.assertEqual(report["status_counts"]["flagged"], 0)

    def test_refusals_are_clean(self):
        self.assertEqual(self.call().returncode, 0)
        again = self.call()
        self.assertTrue(again.returncode != 0 and "REFUSED: Use a fresh output directory" in again.stderr, again.stderr)
        bad = self.call("--sources", "nonsense", out="o2")
        self.assertTrue(bad.returncode != 0 and "REFUSED" in bad.stderr and "Traceback" not in bad.stderr, bad.stderr)
        worse = self.call("--max-close-pairs", "-4", out="o3")
        self.assertTrue(worse.returncode != 0 and "REFUSED" in worse.stderr and "Traceback" not in worse.stderr, worse.stderr)


if __name__ == "__main__":
    unittest.main()
