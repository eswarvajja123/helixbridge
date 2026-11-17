import csv
import io
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _designs as D

try:
    import numpy as np
    import tmtools  # noqa: F401
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict  # noqa: F401
    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False

RUN = unittest.skipUnless(HAVE_STACK and D.AVAILABLE, "needs numpy, Biopython, tmtools and the examples")
if HAVE_STACK and D.AVAILABLE:
    from support import synth_structures as S
    from binder_campaign import diversity as DV
    from binder_campaign.dedup import read_chain_ca
    from test_qc import World

TARGET = json.loads(D.MANIFEST.read_text()) if D.AVAILABLE else None
FIXTURE = Path(__file__).resolve().parent / "fixtures/usalign_reference.json"
REAL = Path(__file__).resolve().parents[2] / "examples/verified-modal-run"


def team_groups(ids, rows, field, threshold, coverage=False):
    """Verbatim copy of `groups` from the team's scripts/review-il7ra-diversity.py (the part that decides the groups); `records`
    is replaced by `ids`. Used only to check that the extracted version decides the same."""
    records = ids
    lookup = {tuple(sorted((r['left'], r['right']))): r for r in rows}

    def related(a, b):
        r = lookup[tuple(sorted((a, b)))]
        score = min(r['tm_left'], r['tm_right']) if field == 'tm' else r['site_jaccard']
        return score >= threshold and (not coverage or r['aligned_length'] >= 48)
    complete = []
    for identifier in sorted(records):
        for group in complete:
            if all(related(identifier, x) for x in group): group.append(identifier); break
        else: complete.append([identifier])
    connected = []; remaining = set(records)
    while remaining:
        group = {min(remaining)}
        while True:
            expanded = group | {b for b in remaining if any(a != b and related(a, b) for a in group)}
            if expanded == group: break
            group = expanded
        connected.append(sorted(group)); remaining -= group
    return {'complete_link_greedy': complete, 'connected_components': connected}


@RUN
class TmEngineMatchesRealUsalign(unittest.TestCase):
    """tmtools (TM-align) reproduces the real USalign binary on real backbones, so the diversity report needs no USalign build."""

    def test_28_real_pairs(self):
        doc = json.loads(FIXTURE.read_text())
        self.assertEqual(doc["usalign"]["commit"], "23f14ffcf8647f2dc1c0e01043db3a7f3a661c28")
        backbones = {}
        for name, o in doc["objects"].items():
            seq, xyz = read_chain_ca(REAL / o["file"], o["chain"])
            a, b = o["slice"]
            backbones[name] = (seq[a:b], xyz[a:b])
        self.assertEqual(len(doc["pairs"]), 28)
        for p in doc["pairs"]:
            got = DV.tm_pair(backbones[p["a"]][1], backbones[p["a"]][0], backbones[p["b"]][1], backbones[p["b"]][0])
            with self.subTest(pair=(p["a"], p["b"])):
                self.assertAlmostEqual(got["tm_left"], p["tm_a"], delta=2e-5)
                self.assertAlmostEqual(got["tm_right"], p["tm_b"], delta=2e-5)
                self.assertEqual(got["aligned_length"], p["aligned_length"])
                self.assertAlmostEqual(got["rmsd_A"], p["rmsd_A"], delta=0.006)

    def test_the_real_design_and_its_boltz_refold_are_the_same_fold_and_unrelated_stretches_are_not(self):
        doc = {(p["a"], p["b"]): p for p in json.loads(FIXTURE.read_text())["pairs"]}
        self.assertGreater(doc[("gen", "boltz")]["tm_a"], 0.99)
        self.assertEqual(doc[("gen", "boltz")]["aligned_length"], 60)
        self.assertLess(max(doc[("gen", t)]["tm_a"] for t in ("t0", "t30", "t60", "t90", "t120")), 0.4)


@RUN
class GroupingLogic(unittest.TestCase):
    def pairs(self, rng, n, lengths=None):
        ids = [f"d{i:02d}" for i in range(n)]
        rows = []
        for a, b in itertools.combinations(ids, 2):
            tm1, tm2 = rng.random(), rng.random()
            rows.append({"left": a, "right": b, "tm_left": tm1, "tm_right": tm2, "aligned_length": rng.choice([30, 47, 48, 49, 60]),
                         "site_jaccard": rng.choice([0.0, 0.2, 0.3, 0.5, 0.7, 1.0, rng.random()])})
        return ids, rows

    def test_decides_exactly_like_the_teams_code_on_300_random_pools(self):
        rng = random.Random(3)
        for trial in range(300):
            ids, rows = self.pairs(rng, rng.randint(2, 14))
            lengths = {i: 60 for i in ids}
            shuffled = ids[:]
            rng.shuffle(shuffled)                                       # member order must not matter: ids are sorted inside
            for threshold in (0.5, 0.6, 0.7):
                self.assertEqual(DV.group_pool(shuffled, rows, "tm", threshold, lengths),
                                 team_groups(ids, rows, "tm", threshold, coverage=True), f"trial {trial} tm {threshold}")
            for threshold in (0.3, 0.5, 0.7):
                self.assertEqual(DV.group_pool(shuffled, rows, "site", threshold), team_groups(ids, rows, "site", threshold),
                                 f"trial {trial} site {threshold}")

    def test_connected_components_agree_with_an_independent_union_find(self):
        rng = random.Random(9)
        for _ in range(200):
            ids, rows = self.pairs(rng, rng.randint(2, 16))
            parent = {i: i for i in ids}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for r in rows:
                if r["site_jaccard"] >= 0.5:
                    parent[find(r["left"])] = find(r["right"])
            expected = sorted(sorted(i for i in ids if find(i) == root) for root in {find(i) for i in ids})
            got = sorted(DV.group_pool(ids, rows, "site", 0.5)["connected_components"])
            self.assertEqual(got, expected)

    def test_every_member_lands_in_exactly_one_group_and_complete_link_groups_are_complete(self):
        rng = random.Random(5)
        for _ in range(100):
            ids, rows = self.pairs(rng, rng.randint(2, 14))
            groups = DV.group_pool(ids, rows, "site", 0.5)
            lookup = {(r["left"], r["right"]): r["site_jaccard"] for r in rows}
            for method in groups.values():
                self.assertEqual(sorted(i for g in method for i in g), sorted(ids))
            for g in groups["complete_link_greedy"]:
                for a, b in itertools.combinations(sorted(g), 2):
                    self.assertGreaterEqual(lookup[(a, b)], 0.5)
            for g in groups["complete_link_greedy"]:                       # each complete-link group sits inside one component
                self.assertTrue(any(set(g) <= set(c) for c in groups["connected_components"]))

    def test_coverage_rule_is_80_percent_of_the_shorter_binder(self):
        self.assertTrue(DV.covered(48, 60, 60))
        self.assertFalse(DV.covered(47, 60, 60))
        self.assertTrue(DV.covered(40, 50, 60))
        self.assertFalse(DV.covered(39, 50, 60))
        self.assertTrue(DV.covered(32, 40, 90))

    def test_coverage_blocks_a_high_tm_with_a_short_alignment(self):
        rows = [{"left": "a", "right": "b", "tm_left": 0.9, "tm_right": 0.9, "aligned_length": 47, "site_jaccard": 0.0}]
        self.assertEqual(DV.group_pool(["a", "b"], rows, "tm", 0.5, {"a": 60, "b": 60})["connected_components"], [["a"], ["b"]])
        rows[0]["aligned_length"] = 48
        self.assertEqual(DV.group_pool(["a", "b"], rows, "tm", 0.5, {"a": 60, "b": 60})["connected_components"], [["a", "b"]])

    def test_jaccard(self):
        self.assertEqual(DV.jaccard([1, 2, 3], [2, 3, 4]), 0.5)
        self.assertEqual(DV.jaccard([], []), 0.0)
        self.assertEqual(DV.jaccard([1], []), 0.0)
        self.assertEqual(DV.jaccard([5, 5, 6], [5, 6]), 1.0)

    def test_threshold_validation(self):
        for bad in ([], [0], [1.5], [float("nan")], [True], ["0.5"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                DV._check_thresholds(bad, "x")


@RUN
class PoolOfComplexes(unittest.TestCase):
    """Real geometry: the verified binder (same fold, moved or not) and a real different fold, all against the real target core."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.core, self.binder = S.real_sequences()
        other_seq, other_atoms = S.other_fold_binder()
        self.members = []
        self.cifs = {}
        for name, sequence, data in (
                ("m1_base", self.binder, S.refold_structure(TARGET["sequence"], self.binder)),
                ("m2_nudged", self.binder, S.refold_structure(TARGET["sequence"], self.binder, shift=(0, 0, 0.5))),
                ("m3_far", self.binder, S.refold_structure(TARGET["sequence"], self.binder, shift=(0, 0, 90))),
                ("m4_other_fold", other_seq, S.refold_custom(TARGET["sequence"], other_atoms))):
            path = self.dir / f"{name}.cif"
            path.write_bytes(data)
            self.cifs[name] = data
            self.members.append({"id": name, "binder_sequence": sequence, "structure": path})
        self.positions = list(range(21, 240))

    def analyse(self, **kw):
        return DV.analyse_source(self.members, TARGET["sequence"], self.positions, "esmfold2", log=lambda *_: None, **kw)

    def pair(self, result, a, b):
        return next(p for p in result["pairs"] if (p["left"], p["right"]) == tuple(sorted((a, b))))

    def test_pairs_are_all_unordered_pairs_with_real_measurements(self):
        r = self.analyse()
        self.assertEqual(len(r["pairs"]), 6)
        self.assertEqual(r["members"], sorted(m["id"] for m in self.members))
        same = self.pair(r, "m1_base", "m2_nudged")
        self.assertGreater(min(same["tm_left"], same["tm_right"]), 0.99)
        self.assertEqual(same["aligned_length"], 60)
        self.assertEqual(same["sequence_identity"], 1.0)
        self.assertGreater(same["site_jaccard"], 0.6)
        other = self.pair(r, "m1_base", "m4_other_fold")
        self.assertLess(min(other["tm_left"], other["tm_right"]), 0.5)

    def test_site_jaccard_matches_an_independent_calculation_from_the_files(self):
        r = self.analyse()
        contacts = {n: {c + 20 for c in S.brute_force(d)[2]} for n, d in self.cifs.items()}
        for p in r["pairs"]:
            a, b = contacts[p["left"]], contacts[p["right"]]
            self.assertAlmostEqual(p["site_jaccard"], len(a & b) / len(a | b) if a | b else 0.0, places=9)
            self.assertEqual(p["shared_contact_residues"], len(a & b))
        self.assertEqual(contacts["m3_far"], set())                            # 90 A away (checked above, from the file): touches nothing
        self.assertEqual(r["target_contacts"]["m3_far"], [])

    def test_a_same_fold_binder_in_another_place_shares_a_fold_group_but_not_a_site_group(self):
        r = self.analyse()
        far = self.pair(r, "m1_base", "m3_far")
        self.assertGreater(min(far["tm_left"], far["tm_right"]), 0.99)
        self.assertEqual(far["site_jaccard"], 0.0)
        fold = r["groups"]["fold"]["0.7"]["complete_link_greedy"]
        self.assertIn(["m1_base", "m2_nudged", "m3_far"], fold)
        self.assertIn(["m4_other_fold"], fold)
        site = r["groups"]["site"]["0.3"]["connected_components"]
        self.assertIn(["m1_base", "m2_nudged"], site)
        self.assertIn(["m3_far"], site)
        self.assertEqual(r["summary"]["fold"]["0.5"]["complete_link_greedy"], {"groups": 2, "largest": 3, "singletons": 1})

    def test_a_member_that_cannot_be_read_is_left_out_with_a_reason_not_a_crash(self):
        bad = self.dir / "bad.cif"
        bad.write_bytes(b"not a structure")
        self.members.append({"id": "m5_bad", "binder_sequence": self.binder, "structure": bad})
        self.members.append({"id": "m6_wrong_hash", "binder_sequence": self.binder, "structure": self.dir / "m1_base.cif", "sha256": "0" * 64})
        r = self.analyse()
        self.assertEqual([e["id"] for e in r["excluded"]], ["m5_bad", "m6_wrong_hash"])
        self.assertTrue(r["excluded"][1]["reasons"][0].startswith("structure_hash_mismatch"))
        self.assertEqual(len(r["pairs"]), 6)

    def test_refusals(self):
        self.members.append(dict(self.members[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate member ids"):
            self.analyse()
        self.members.pop()
        with self.assertRaisesRegex(ValueError, "limit is 3"):
            self.analyse(max_pool=3)
        with self.assertRaises(ValueError):
            self.analyse(fold_thresholds=[2.0])


@RUN
class PipelineFolders(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.w = World(self.temp.name)
        self.w.add("a")
        self.w.add("b", sequence=D.random_sequence(1, 60), esm=(0, 0, 0.4))
        self.w.add("c", sequence=D.random_sequence(2, 60), esm=(0, 0, 60))          # esmfold2 pose far from the target
        self.w.add("d", sequence=D.random_sequence(3, 60), qc_reasons=["near_duplicate:exact_duplicate"])
        self.w.evaluate()
        self.records = json.loads((self.w.eval / "evaluations.json").read_text())["candidates"]

    def run_it(self, **kw):
        out = Path(self.temp.name) / kw.pop("out", "div")
        return DV.run_diversity(self.records, TARGET, out, self.w.eval, self.w.dir / "batch", log=lambda *_: None, **kw), out

    def test_one_analysis_per_source_and_flagged_records_are_left_out(self):
        report, out = self.run_it()
        self.assertEqual(sorted(report["sources"]), ["boltz2", "esmfold2", "generator"])
        self.assertEqual(report["left_out_as_flagged"], [{"id": "il7ra-rfd4-d", "reasons": ["near_duplicate:exact_duplicate"]}])
        for source in ("generator", "esmfold2", "boltz2"):
            self.assertEqual(report["sources"][source]["members"], ["il7ra-rfd4-a", "il7ra-rfd4-b", "il7ra-rfd4-c"])
            self.assertEqual(len(report["sources"][source]["pairs"]), 3)
        again, _ = self.run_it(out="div2", include_flagged=True)
        self.assertEqual(len(again["sources"]["esmfold2"]["members"]), 4)
        self.assertEqual(again["left_out_as_flagged"], [])

    def test_sources_differ_because_the_structures_differ(self):
        report, _ = self.run_it()
        ac_gen = next(p for p in report["sources"]["generator"]["pairs"] if (p["left"], p["right"]) == ("il7ra-rfd4-a", "il7ra-rfd4-c"))
        ac_esm = next(p for p in report["sources"]["esmfold2"]["pairs"] if (p["left"], p["right"]) == ("il7ra-rfd4-a", "il7ra-rfd4-c"))
        self.assertEqual(ac_gen["site_jaccard"], 1.0)        # the generator poses are identical
        self.assertEqual(ac_esm["site_jaccard"], 0.0)        # the ESMFold2 refold of c sits far away
        self.assertEqual(ac_gen["sequence_identity"], ac_esm["sequence_identity"])

    def test_files_are_written_and_consistent(self):
        report, out = self.run_it()
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["diversity.json", "groups.csv", "pairs.csv"])
        pairs = list(csv.DictReader(io.StringIO((out / "pairs.csv").read_text())))
        self.assertEqual(len(pairs), 9)
        groups = list(csv.DictReader(io.StringIO((out / "groups.csv").read_text())))
        per = {(g["source"], g["kind"], g["threshold"], g["method"]): 0 for g in groups}
        for g in groups:
            per[(g["source"], g["kind"], g["threshold"], g["method"])] += 1
        self.assertTrue(all(count == 3 for count in per.values()), per)       # every member appears once per source, kind, threshold, method
        self.assertEqual(len(per), 3 * (3 + 3) * 2)
        self.assertEqual(json.loads((out / "diversity.json").read_text())["sources"]["generator"]["members"], report["sources"]["generator"]["members"])

    def test_a_source_with_fewer_than_two_structures_is_skipped_not_invented(self):
        d = World(tempfile.mkdtemp(dir=self.temp.name))
        d.add("a")
        d.add("b", skip=("boltz2",))
        d.evaluate()
        records = json.loads((d.eval / "evaluations.json").read_text())["candidates"]
        report = DV.run_diversity(records, TARGET, Path(self.temp.name) / "div3", d.eval, d.dir / "batch", log=lambda *_: None)
        self.assertIn("skipped", report["sources"]["boltz2"])
        self.assertEqual(len(report["sources"]["esmfold2"]["pairs"]), 1)

    def test_without_an_evaluation_folder_only_the_generator_source_is_analysed(self):
        report = DV.run_diversity(self.records, TARGET, Path(self.temp.name) / "div5", None, self.w.dir / "batch", log=lambda *_: None)
        self.assertEqual(len(report["sources"]["generator"]["pairs"]), 3)
        self.assertIn("no --evaluation folder", report["sources"]["esmfold2"]["skipped"])
        self.assertIn("no --evaluation folder", report["sources"]["boltz2"]["skipped"])

    def test_bad_thresholds_are_refused_before_any_work(self):
        with self.assertRaisesRegex(ValueError, "fold thresholds"):
            DV.run_diversity(self.records, TARGET, Path(self.temp.name) / "div6", self.w.eval, self.w.dir / "batch",
                             fold_thresholds=[7], log=lambda *_: None)
        self.assertFalse((Path(self.temp.name) / "div6").exists())

    def test_refuses_an_existing_folder_and_a_pool_without_a_shared_numbering(self):
        _, out = self.run_it()
        with self.assertRaisesRegex(ValueError, "fresh output"):
            DV.run_diversity(self.records, TARGET, out, self.w.eval, self.w.dir / "batch", log=lambda *_: None)
        records = json.loads(json.dumps(self.records))
        records[1]["generator_audit"]["reference_audit"]["positions"] = list(range(38, 233))
        with self.assertRaisesRegex(ValueError, "do not share one target numbering"):
            DV.run_diversity(records, TARGET, Path(self.temp.name) / "div4", self.w.eval, self.w.dir / "batch", log=lambda *_: None)


@RUN
class Script(unittest.TestCase):
    SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diversity-report.py"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.w = World(self.temp.name)
        self.w.add("a")
        self.w.add("b", sequence=D.random_sequence(1, 60), esm=(0, 0, 0.4))
        self.w.evaluate()

    def call(self, *args):
        return subprocess.run([sys.executable, "-I", str(self.SCRIPT), "--target-manifest", str(D.MANIFEST), *args],
                              capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})

    def test_records_mode(self):
        done = self.call("--records", str(self.w.eval / "evaluations.json"), "--evaluation", str(self.w.eval), "--batch-dir",
                         str(self.w.dir / "batch"), "--out", str(self.w.dir / "o1"))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(sorted(json.loads(done.stdout)["sources"]), ["boltz2", "esmfold2", "generator"])
        self.assertTrue((self.w.dir / "o1" / "pairs.csv").is_file())

    def test_pool_mode_with_any_pool(self):
        real_seq, other = S.other_fold_binder()
        (self.w.dir / "p1.cif").write_bytes(S.refold_structure(TARGET["sequence"], self.w.real_binder))
        (self.w.dir / "p2.cif").write_bytes(S.refold_custom(TARGET["sequence"], other))
        pool = {"label": "mine", "target_sequence": TARGET["sequence"], "positions": list(range(21, 240)),
                "members": [{"id": "x1", "binder_sequence": self.w.real_binder, "structure": str(self.w.dir / "p1.cif")},
                            {"id": "x2", "binder_sequence": real_seq, "structure": str(self.w.dir / "p2.cif")}]}
        (self.w.dir / "pool.json").write_text(json.dumps(pool))
        done = self.call("--pool", str(self.w.dir / "pool.json"), "--out", str(self.w.dir / "o2"))
        self.assertEqual(done.returncode, 0, done.stderr)
        report = json.loads((self.w.dir / "o2" / "diversity.json").read_text())
        self.assertEqual(report["sources"]["mine"]["members"], ["x1", "x2"])
        self.assertLess(min(report["sources"]["mine"]["pairs"][0]["tm_left"], report["sources"]["mine"]["pairs"][0]["tm_right"]), 0.5)

    def test_refusals_are_clean(self):
        for args in (["--out", str(self.w.dir / "o3")],
                     ["--records", "x", "--pool", "y", "--out", str(self.w.dir / "o3")],
                     ["--records", str(self.w.eval / "evaluations.json"), "--sources", "bogus", "--out", str(self.w.dir / "o3")],
                     ["--records", str(self.w.eval / "evaluations.json"), "--fold-thresholds", "7", "--out", str(self.w.dir / "o3")],
                     ["--records", str(self.w.dir / "missing.json"), "--out", str(self.w.dir / "o3")]):
            with self.subTest(args=args):
                done = self.call(*args)
                self.assertTrue(done.returncode != 0 and "REFUSED" in done.stderr and "Traceback" not in done.stderr, done.stderr)


if __name__ == "__main__":
    unittest.main()
