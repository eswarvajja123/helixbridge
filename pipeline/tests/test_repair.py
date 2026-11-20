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
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False

RUN = unittest.skipUnless(HAVE and D.AVAILABLE, "needs numpy, Biopython and the examples")
if HAVE and D.AVAILABLE:
    from support import synth_structures as S
    from binder_campaign import repair, qc
    from binder_campaign.liabilities import flags_for
    from binder_campaign.sequence_review import sequence_diagnostic

REAL = Path(__file__).resolve().parents[2] / "examples/verified-modal-run/generator.cif"
AA = "ACDEFGHIKLMNPQRSTVWY"
CLEAN = "ADEFGHIKLMQRSTVWY"


def clean(seed, n=60):
    rng = random.Random(seed)
    return "".join(rng.choice(CLEAN) for _ in range(n))


class Edits(unittest.TestCase):
    def test_each_liability(self):
        b = clean(1)
        self.assertEqual(repair.liability_edits(b), ({}, []))
        self.assertEqual(repair.liability_edits(b[:9] + "C" + b[10:]), ({10: "C"}, ["cysteine"]))
        self.assertEqual(repair.liability_edits(b[:9] + "CC" + b[11:]), ({}, []))                 # an even number is left alone
        self.assertEqual(repair.liability_edits(b[:4] + "C" + b[5:20] + "CC" + b[22:]), ({5: "C", 21: "C", 22: "C"}, ["cysteine"]))
        self.assertEqual(repair.liability_edits(b[:9] + "NAS" + b[12:]), ({10: "N"}, ["sequon"]))
        self.assertEqual(repair.liability_edits(b[:9] + "NPS" + b[12:]), ({}, []))
        self.assertEqual(repair.liability_edits(b[:9] + "NGA" + b[12:]), ({10: "N"}, ["asn_gly"]))
        self.assertEqual(repair.liability_edits(b[:9] + "DPA" + b[12:]), ({10: "D"}, ["asp_pro"]))

    def test_overlapping_motifs_share_one_edit_and_all_types_are_listed(self):
        b = clean(2)
        seq = b[:9] + "NGSA" + b[13:]                    # N-G-S is both a sequon and an Asn-Gly motif, at one position
        self.assertEqual(repair.liability_edits(seq), ({10: "N"}, ["asn_gly", "sequon"]))

    def test_edits_always_point_at_the_offending_residue(self):
        rng = random.Random(4)
        for _ in range(300):
            seq = "".join(rng.choice(AA) for _ in range(rng.randint(40, 100)))
            edits, types = repair.liability_edits(seq)
            for position, letters in edits.items():
                self.assertIn(seq[position - 1], letters)
            self.assertEqual(bool(edits), bool(flags_for(sequence_diagnostic(seq))) and bool(types))


@RUN
class Pdb(unittest.TestCase):
    def test_real_structure_converts_with_exact_coordinates_and_sequences(self):
        pdb, chains = repair.cif_to_pdb(REAL)
        core, binder = S.real_sequences()
        self.assertEqual(chains, {"B": binder, "A": core})                          # this file lists the binder chain first
        atoms = S.real_atoms()
        lines = [l for l in pdb.splitlines() if l.startswith("ATOM")]
        heavy = [a for a in atoms if a[4] not in ("H", "D")]
        self.assertEqual(len(lines), len(heavy))
        for line, (chain, rid, comp, name, element, xyz) in zip(lines, heavy):
            self.assertEqual((line[21], int(line[22:26]), line[17:20].strip(), line[12:16].strip()), (chain, rid, comp, name))
            self.assertEqual([float(line[30:38]), float(line[38:46]), float(line[46:54])], [round(c, 3) for c in xyz])
            self.assertEqual(line[76:78].strip(), element)
        self.assertTrue(pdb.rstrip().endswith("END"))

    def test_the_native_output_order_is_alphabetical_not_file_order(self):
        _, chains = repair.cif_to_pdb(REAL)
        self.assertEqual(list(chains), ["B", "A"])
        self.assertEqual(repair.binder_chain_index(chains, "B"), 1)                  # upstream joins chains A:B whatever the file order
        with self.assertRaises(ValueError):
            repair.binder_chain_index(chains, "Z")


class Outputs(unittest.TestCase):
    def fasta(self, samples, header_extra=""):
        lines = [">design, T=0.1, seed=401, num_res=255, num_ligand_res=0, use_ligand_context=False, ligand_cutoff_distance=8.0, "
                 "batch_size=16, number_of_batches=1, model_path=/w/solublempnn_v_48_020.pt", "TARGET:" + "P" * 5]
        for i, (seq, conf) in enumerate(samples):
            lines += [f">design, id={i}, T=0.1, seed=401, overall_confidence={conf}, ligand_confidence=0.0, seq_rec=0.9{header_extra}", "TARGET:" + seq]
        return "\n".join(lines)

    def parse(self, text, index=1):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "d.fa"
            p.write_text(text)
            return repair.parse_outputs(p, index)

    def test_reads_the_upstream_format(self):
        out = self.parse(self.fasta([("AAAA", 0.61), ("CCCC", 0.7)]))
        self.assertEqual([(o["sequence"], o["overall_confidence"], o["sample"]) for o in out], [("AAAA", 0.61, 0), ("CCCC", 0.7, 1)])

    def test_bad_outputs_are_refused(self):
        with self.assertRaises(ValueError):
            self.parse(self.fasta([("AAAA", 0.6)]).replace("overall_confidence=0.6", "x=1"))
        with self.assertRaises(ValueError):
            self.parse(self.fasta([("AAAA", 0.6)]), index=3)


class Choosing(unittest.TestCase):
    def setUp(self):
        self.parent = clean(5)[:9] + "C" + clean(5)[10:]
        self.edits, self.types = repair.liability_edits(self.parent)
        self.assertEqual((self.edits, self.types), ({10: "C"}, ["cysteine"]))

    def variant(self, letter, position=10):
        return self.parent[:position - 1] + letter + self.parent[position:]

    def sample(self, seq, conf, n):
        return {"sequence": seq, "overall_confidence": conf, "sample": n, "header": ""}

    def test_ranking_follows_the_team_rule(self):
        s = [self.sample(self.variant("A"), 0.60, 0), self.sample(self.variant("L"), 0.80, 1), self.sample(self.variant("K"), 0.80, 2),
             self.sample(self.variant("L"), 0.55, 3)]                                   # a duplicate of sample 1 with lower confidence
        chosen, rejected, shortfall = repair.choose(self.parent, s, self.edits, self.types)
        self.assertEqual([c["sequence"] for c in chosen], sorted([self.variant("L"), self.variant("K")], key=lambda x: x)[:2])
        self.assertEqual((rejected, shortfall), ([], 0))
        self.assertEqual({c["overall_confidence"] for c in chosen}, {0.8})              # tie on confidence broken by the sequence

    def test_higher_confidence_wins_over_sequence_order(self):
        s = [self.sample(self.variant("A"), 0.5, 0), self.sample(self.variant("W"), 0.9, 1), self.sample(self.variant("D"), 0.7, 2)]
        chosen, _, _ = repair.choose(self.parent, s, self.edits, self.types)
        self.assertEqual([c["sequence"][9] for c in chosen], ["W", "D"])

    def test_invalid_samples_are_rejected_with_the_reason(self):
        bad_len = self.variant("A") + "A"
        changed = self.variant("A", 10)[:20] + ("G" if self.parent[20] != "G" else "A") + self.variant("A", 10)[21:]
        s = [self.sample(self.variant("C"), 0.9, 0),                  # forbidden residue
             self.sample(bad_len, 0.9, 1),                            # wrong length
             self.sample(changed, 0.9, 2),                            # a fixed position changed
             self.sample(self.parent, 0.9, 3),                        # identical to the parent
             self.sample(self.variant("X"), 0.9, 4),                  # not a canonical letter
             self.sample(self.variant("L"), 0.1, 5)]
        chosen, rejected, shortfall = repair.choose(self.parent, s, self.edits, self.types)
        why = {r["sample"]: r["problems"] for r in rejected}
        self.assertIn("forbidden_residue_at_10", why[0])
        self.assertIn("format", why[1])
        self.assertTrue(any(p.startswith("changed_fixed_position") for p in why[2]))
        self.assertIn("identical_to_parent", why[3])
        self.assertIn("format", why[4])
        self.assertEqual([c["sample"] for c in chosen], [5])
        self.assertEqual(shortfall, 1)                                  # recorded, never filled by resampling

    def test_a_variant_that_keeps_the_liability_is_rejected(self):
        b = clean(6)
        parent = b[:9] + "C" + b[10:30] + "NAS" + b[33:]
        edits, types = repair.liability_edits(parent)
        self.assertEqual(types, ["cysteine", "sequon"])
        fixes_only_cys = parent[:9] + "L" + parent[10:]                  # position 30 (the sequon N) left as N
        chosen, rejected, _ = repair.choose(parent, [self.sample(fixes_only_cys, 0.9, 0)], edits, types)
        self.assertEqual(chosen, [])
        self.assertIn("liability_remains:n_glycosylation_sequon", rejected[0]["problems"])

    def test_variant_records_carry_no_evidence_from_the_parent(self):
        parent = {"candidate_id": "il7ra-rfd4-aaa", "sequence": self.parent, "target_id": "il7ra", "campaign_kind": "competition_campaign",
                  "generator_structure": "x.cif", "generator_binder_chain": "B", "seed": 1, "hotspots_uniprot": [1],
                  "evaluations": {"esmfold2": {"status": "done"}}, "qc": {"status": "no_flags"}, "liabilities": {"flags": []},
                  "novelty_prescreen": {"status": "no_public_hit"}, "novelty_review": "passed", "screen_status": "approximation_reviewed",
                  "independent_prediction_status": "reviewed", "submission_eligible": True, "qc_reasons": ["x"]}
        v = repair.variant_record(parent, self.sample(self.variant("L"), 0.8, 3), 2, self.edits, self.types)
        self.assertEqual((v["candidate_id"], v["parent_candidate_id"], v["generator_structure_sequence"]), ("il7ra-rfd4-aaa-r2", "il7ra-rfd4-aaa", self.parent))
        for key in ("evaluations", "qc", "liabilities", "novelty_prescreen"):
            self.assertNotIn(key, v)
        self.assertEqual((v["novelty_review"], v["screen_status"], v["independent_prediction_status"], v["submission_eligible"], v["qc_reasons"]),
                         ("pending", "not_run", "not_run", False, []))
        self.assertEqual(v["repair"]["rank_within_parent"], 2)
        self.assertEqual(v["repair"]["forbidden"], {10: "C"})


class NativeWrapper(unittest.TestCase):
    """The wrapper around the upstream command, tested with a stand-in run.py (the real code is third-party and is not run in tests)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        (self.dir / "code").mkdir()
        (self.dir / "code" / "run.py").write_text(
            "import argparse, json, os, sys\n"
            "p = argparse.ArgumentParser()\n"
            "for k in ['model_type','checkpoint_soluble_mpnn','pdb_path','out_folder','redesigned_residues','omit_AA_per_residue','temperature','seed',"
            "'batch_size','number_of_batches','save_stats','pack_side_chains']:\n"
            "    p.add_argument('--' + k)\n"
            "a = p.parse_args()\n"
            "if os.environ.get('FAKE_FAIL'): sys.exit('boom')\n"
            "os.makedirs(a.out_folder + '/seqs', exist_ok=True)\n"
            "json.dump(vars(a), open(a.out_folder + '/args.json', 'w'))\n"
            "open(a.out_folder + '/seqs/design.fa', 'w').write('>design, T=0.1\\nAAA:CCC\\n>design, id=0, T=0.1, seed=401, overall_confidence=0.5\\nAAA:CLC\\n')\n")
        (self.dir / "d.pdb").write_text("END\n")
        (self.dir / "w.pt").write_text("w")

    UPSTREAM_FLAGS = ['--batch_size', '--bias_AA', '--bias_AA_per_residue', '--bias_AA_per_residue_multi', '--chains_to_design', '--checkpoint_global_label_membrane_mpnn', '--checkpoint_ligand_mpnn', '--checkpoint_path_sc', '--checkpoint_per_residue_label_membrane_mpnn', '--checkpoint_protein_mpnn', '--checkpoint_soluble_mpnn', '--fasta_seq_separation', '--file_ending', '--fixed_residues', '--fixed_residues_multi', '--force_hetatm', '--global_transmembrane_label', '--homo_oligomer', '--ligand_mpnn_cutoff_for_score', '--ligand_mpnn_use_atom_context', '--ligand_mpnn_use_side_chain_context', '--model_type', '--number_of_batches', '--number_of_packs_per_design', '--omit_AA', '--omit_AA_per_residue', '--omit_AA_per_residue_multi', '--out_folder', '--pack_side_chains', '--pack_with_ligand_context', '--packed_suffix', '--parse_atoms_with_zero_occupancy', '--parse_these_chains_only', '--pdb_path', '--pdb_path_multi', '--redesigned_residues', '--redesigned_residues_multi', '--repack_everything', '--save_stats', '--sc_num_denoising_steps', '--sc_num_samples', '--seed', '--symmetry_residues', '--symmetry_weights', '--temperature', '--transmembrane_buried', '--transmembrane_interface', '--verbose', '--zero_indexed']     # every option of the pinned upstream run.py (LigandMPNN 26ec57ac)

    def test_every_option_we_pass_exists_in_the_pinned_upstream_run_py(self):
        _, command = repair.run_native(self.dir / "d.pdb", {28: "C", 30: "N"}, self.dir / "outx", self.dir / "code", self.dir / "w.pt", sys.executable)
        used = [c for c in command if isinstance(c, str) and c.startswith("--")]
        self.assertTrue(used)
        self.assertEqual([c for c in used if c not in self.UPSTREAM_FLAGS], [])          # a made-up option (--device) failed on the real GPU run

    def test_the_team_settings_are_passed_to_the_upstream_command(self):
        fasta, command = repair.run_native(self.dir / "d.pdb", {28: "C", 30: "N"}, self.dir / "out", self.dir / "code", self.dir / "w.pt", sys.executable)
        args = json.loads((self.dir / "out" / "args.json").read_text())
        self.assertEqual((args["model_type"], args["temperature"], args["seed"], args["batch_size"], args["number_of_batches"]),
                         ("soluble_mpnn", "0.1", "401", "16", "1"))
        self.assertEqual(args["redesigned_residues"], "B28 B30")
        self.assertEqual(json.loads((self.dir / "out" / "omit.json").read_text()), {"B28": "C", "B30": "N"})
        self.assertEqual((args["save_stats"], args["pack_side_chains"], "device" in args), ("1", "0", False))
        self.assertEqual(repair.parse_outputs(fasta, 1)[0]["sequence"], "CLC")

    def test_a_failing_run_is_an_error(self):
        import os
        os.environ["FAKE_FAIL"] = "1"
        self.addCleanup(os.environ.pop, "FAKE_FAIL", None)
        with self.assertRaisesRegex(RuntimeError, "SolubleMPNN failed"):
            repair.run_native(self.dir / "d.pdb", {1: "C"}, self.dir / "out2", self.dir / "code", self.dir / "w.pt", sys.executable)


@RUN
class QcOfVariants(unittest.TestCase):
    """QC must compare a variant's refolds with its parent's backbone, whose structure file still holds the parent's sequence."""

    def setUp(self):
        self.target = json.loads(D.MANIFEST.read_text())
        self.core, self.binder = S.real_sequences()
        self.variant = self.binder[:9] + ("G" if self.binder[9] != "G" else "A") + self.binder[10:]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = {}
        for name, data in (("generator", REAL.read_bytes()), ("esmfold2", S.refold_structure(self.target["sequence"], self.variant))):
            p = Path(self.tmp.name) / f"{name}.cif"
            p.write_bytes(data)
            self.paths[name] = {"path": p}
        self.record = {"candidate_id": "v", "sequence": self.variant, "hotspots_uniprot": D.HOTSPOTS,
                       "generator_audit": {"reference_audit": {"reference_sequence": self.core, "positions": list(range(37, 232))}}}

    def check(self, record):
        return qc.check_candidate(record, self.paths, self.target, expected=("generator", "esmfold2"))

    def test_without_the_parent_sequence_the_generator_structure_looks_wrong(self):
        got = self.check(self.record)
        self.assertIn("binder_sequence_mismatch:generator", got["reasons"])

    def test_with_the_parent_sequence_the_variant_is_checked_against_the_parents_backbone(self):
        got = self.check({**self.record, "generator_structure_sequence": self.binder})
        self.assertEqual(got["reasons"], [], got["reasons"])
        self.assertEqual(got["status"], "no_flags")
        self.assertAlmostEqual(got["pose_shift"]["generator_vs_esmfold2"]["binder_ca_rmsd_A"], 0.0, places=3)

    def test_the_refold_is_still_checked_against_the_variants_own_sequence(self):
        got = self.check({**self.record, "sequence": self.binder, "generator_structure_sequence": self.binder})     # refold holds the variant
        self.assertIn("binder_sequence_mismatch:esmfold2", got["reasons"])


@RUN
class Script(unittest.TestCase):
    SCRIPT = Path(__file__).resolve().parents[1] / "scripts/repair-liabilities.py"

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        core, binder = S.real_sequences()
        self.sequence = binder[:9] + "C" + binder[10:]                        # one cysteine: odd
        gen = self.dir / "gen.cif"
        gen.write_bytes(S.generator_structure(self.sequence))
        clean_seq = binder[:9] + "L" + binder[10:]
        gen2 = self.dir / "gen2.cif"
        gen2.write_bytes(S.generator_structure(clean_seq))
        def rec(cid, seq, path):
            return {"candidate_id": cid, "sequence": seq, "generator_structure": str(path), "generator_binder_chain": "B", "target_id": "il7ra",
                    "qc_reasons": [], "seed": 1}
        self.records = [rec("a", self.sequence, gen), rec("b", clean_seq, gen2)]
        (self.dir / "r.json").write_text(json.dumps({"candidates": self.records}))
        (self.dir / "sel.json").write_text(json.dumps({"selected": [{"candidate_id": "a"}, {"candidate_id": "b"}]}))

    def call(self, *a):
        return subprocess.run([sys.executable, "-I", str(self.SCRIPT), *a], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})

    def test_plan_lists_only_designs_with_liabilities_and_writes_the_native_inputs(self):
        done = self.call("plan", "--records", str(self.dir / "r.json"), "--selection", str(self.dir / "sel.json"), "--out", str(self.dir / "o"))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout)["designs_with_liabilities"], 1)
        plan = json.loads((self.dir / "o" / "repair-plan.json").read_text())
        self.assertEqual(plan["designs"][0]["redesign_positions"], {"10": "C"})
        self.assertEqual((plan["designs"][0]["binder_chain_index"], plan["settings"]["seed"], plan["settings"]["temperature"]), (1, 401, 0.1))
        self.assertTrue((self.dir / "o" / "a" / "design.pdb").is_file())
        self.assertFalse((self.dir / "o" / "b").exists())

    def test_ingest_builds_variant_records_from_a_native_run_done_elsewhere(self):
        native = self.dir / "native" / "a" / "seqs"
        native.mkdir(parents=True)
        v1, v2 = self.sequence[:9] + "L" + self.sequence[10:], self.sequence[:9] + "K" + self.sequence[10:]
        keep = self.sequence[:9] + "C" + self.sequence[10:]
        (native / "a.fa").write_text("\n".join([">a, T=0.1, seed=401", "T:" + self.sequence,
            f">a, id=0, T=0.1, seed=401, overall_confidence=0.55, ligand_confidence=0.0, seq_rec=0.9", "T:" + v1,
            f">a, id=1, T=0.1, seed=401, overall_confidence=0.75, ligand_confidence=0.0, seq_rec=0.9", "T:" + v2,
            f">a, id=2, T=0.1, seed=401, overall_confidence=0.95, ligand_confidence=0.0, seq_rec=0.9", "T:" + keep]) + "\n")
        done = self.call("ingest", "--records", str(self.dir / "r.json"), "--native-dir", str(self.dir / "native"), "--out", str(self.dir / "o2"))
        self.assertEqual(done.returncode, 0, done.stderr)
        out = json.loads((self.dir / "o2" / "repair-candidates.json").read_text())["candidates"]
        self.assertEqual([(v["candidate_id"], v["sequence"][9]) for v in out], [("a-r1", "K"), ("a-r2", "L")])      # higher confidence first
        self.assertEqual({v["generator_structure_sequence"] for v in out}, {self.sequence})
        summary = json.loads(done.stdout)["results"][0]
        self.assertEqual((summary["status"], summary["samples"], summary["rejected"], summary["variants"]), ("repaired", 3, 1, 2))

    def test_max_variants_widens_the_screen_and_is_recorded(self):
        native = self.dir / "native" / "a" / "seqs"
        native.mkdir(parents=True)
        lines = [">a, T=0.1, seed=401", "T:" + self.sequence]
        for i, letter in enumerate("LKMIVAQ"):
            lines += [f">a, id={i}, T=0.1, seed=401, overall_confidence=0.{50 + i}, ligand_confidence=0.0, seq_rec=0.9", "T:" + self.sequence[:9] + letter + self.sequence[10:]]
        (native / "a.fa").write_text("\n".join(lines) + "\n")
        done = self.call("ingest", "--records", str(self.dir / "r.json"), "--native-dir", str(self.dir / "native"), "--max-variants", "5", "--out", str(self.dir / "o6"))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(len(json.loads((self.dir / "o6" / "repair-candidates.json").read_text())["candidates"]), 5)
        self.assertEqual(json.loads((self.dir / "o6" / "repair-plan.json").read_text())["settings"]["max_variants_per_design"], 5)
        default = self.call("ingest", "--records", str(self.dir / "r.json"), "--native-dir", str(self.dir / "native"), "--out", str(self.dir / "o7"))
        self.assertEqual(len(json.loads((self.dir / "o7" / "repair-candidates.json").read_text())["candidates"]), 2)     # the team's default is unchanged
        for bad in ("0", "9", "x"):
            refused = self.call("ingest", "--records", str(self.dir / "r.json"), "--native-dir", str(self.dir / "native"), "--max-variants", bad, "--out", str(self.dir / ("b" + bad)))
            self.assertNotEqual(refused.returncode, 0)

    def test_a_tag_and_the_recorded_run_settings_reach_the_variant_records(self):
        run = self.dir / "run2"
        native = run / "native" / "a" / "seqs"
        native.mkdir(parents=True)
        v = self.sequence[:9] + "L" + self.sequence[10:]
        (native / "a.fa").write_text("\n".join([">a, T=0.3, seed=402", "T:" + self.sequence,
            ">a, id=0, T=0.3, seed=402, overall_confidence=0.6, ligand_confidence=0.0, seq_rec=0.9", "T:" + v]) + "\n")
        (run / "native-run.json").write_text(json.dumps({"temperature": 0.3, "seed": 402, "designs": []}))
        done = self.call("ingest", "--records", str(self.dir / "r.json"), "--native-dir", str(run / "native"), "--tag", "t03", "--out", str(self.dir / "o9"))
        self.assertEqual(done.returncode, 0, done.stderr)
        out = json.loads((self.dir / "o9" / "repair-candidates.json").read_text())["candidates"]
        self.assertEqual([c["candidate_id"] for c in out], ["a-t03r1"])
        self.assertEqual((out[0]["repair"]["temperature"], out[0]["repair"]["seed"]), (0.3, 402))
        self.assertEqual(json.loads((self.dir / "o9" / "repair-plan.json").read_text())["settings"]["temperature"], 0.3)
        bad = self.call("ingest", "--records", str(self.dir / "r.json"), "--native-dir", str(run / "native"), "--tag", "bad tag!", "--out", str(self.dir / "o10"))
        self.assertNotEqual(bad.returncode, 0)

    def test_only_types_repairs_just_those_kinds(self):
        base = self.sequence.replace("C", "A")                                                       # no cysteine, so only the motifs added below count
        seq = base[:4] + "NVT" + base[7:20] + "NG" + base[22:30] + "DP" + base[32:]
        self.assertEqual(repair.liability_edits(seq)[1], ["asn_gly", "asp_pro", "sequon"])
        self.assertEqual(repair.liability_edits(seq, ["sequon"]), ({5: "N"}, ["sequon"]))
        self.assertEqual(repair.liability_edits(seq, ["asp_pro", "asn_gly"])[0], {21: "N", 31: "D"})
        self.assertEqual(repair.liability_edits(seq, ["cysteine"]), ({}, []))                       # a kind the design does not have: nothing to do
        for bad in ([], ["glycine"], ["sequon", "x"]):
            with self.assertRaises(ValueError):
                repair.liability_edits(seq, bad)

    def test_the_script_passes_only_types_to_plan(self):
        sequence = self.sequence
        done = self.call("plan", "--records", str(self.dir / "r.json"), "--selection", str(self.dir / "sel.json"), "--only-types", "cysteine", "--out", str(self.dir / "o11"))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout)["designs_with_liabilities"], 1)
        none = self.call("plan", "--records", str(self.dir / "r.json"), "--selection", str(self.dir / "sel.json"), "--only-types", "sequon", "--out", str(self.dir / "o12"))
        self.assertEqual(json.loads(none.stdout)["designs_with_liabilities"], 0)                   # the test design has only an odd cysteine
        bad = self.call("plan", "--records", str(self.dir / "r.json"), "--only-types", "nonsense", "--out", str(self.dir / "o13"))
        self.assertNotEqual(bad.returncode, 0)

    def test_the_local_run_mode_works_with_a_stand_in_for_the_upstream_program(self):
        code = self.dir / "code"
        code.mkdir()
        (code / "run.py").write_text(
            "import argparse, os\n"
            "p = argparse.ArgumentParser()\n"
            "for k in ['model_type','checkpoint_soluble_mpnn','pdb_path','out_folder','redesigned_residues','omit_AA_per_residue','temperature','seed',"
            "'batch_size','number_of_batches','save_stats','pack_side_chains']:\n"
            "    p.add_argument('--' + k)\n"
            "a = p.parse_args()\n"
            "os.makedirs(a.out_folder + '/seqs', exist_ok=True)\n"
            "open(a.out_folder + '/seqs/design.fa', 'w').write('>design, T=0.1\\nAAA:CCC\\n')\n")
        (self.dir / "w.pt").write_text("w")
        done = self.call("run", "--records", str(self.dir / "r.json"), "--selection", str(self.dir / "sel.json"), "--mpnn-code", str(code),
                         "--weights", str(self.dir / "w.pt"), "--python", sys.executable, "--out", str(self.dir / "o8"))
        self.assertEqual(done.returncode, 0, done.stderr)                       # this used to fail with an unexpected 'device' argument
        self.assertNotIn("Traceback", done.stderr)

    def test_missing_native_output_is_reported_not_invented(self):
        done = self.call("ingest", "--records", str(self.dir / "r.json"), "--native-dir", str(self.dir / "nothing"), "--out", str(self.dir / "o3"))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(done.stdout)["results"][0]["status"], "no_native_output")
        self.assertFalse((self.dir / "o3" / "repair-candidates.json").exists())

    def test_refusals(self):
        self.call("plan", "--records", str(self.dir / "r.json"), "--out", str(self.dir / "o4"))
        again = self.call("plan", "--records", str(self.dir / "r.json"), "--out", str(self.dir / "o4"))
        self.assertTrue(again.returncode != 0 and "REFUSED" in again.stderr and "Traceback" not in again.stderr)
        no_args = self.call("run", "--records", str(self.dir / "r.json"), "--out", str(self.dir / "o5"))
        self.assertNotEqual(no_args.returncode, 0)


if __name__ == "__main__":
    unittest.main()
