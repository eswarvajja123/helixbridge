import ast
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.conditioning import write_conditioning
from binder_campaign.current_candidates import import_current
from binder_campaign.dedup import DedupGate, read_chain_ca
from binder_campaign.generation_input import load_reference
from binder_campaign.runner import digest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/il7ra-rehearsal"
REAL = ROOT / "examples/verified-modal-run"
MANIFEST = ROOT / "docs/eswar/phase1-data/il7ra-manifest.json"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rfd4-generate.py"
HOTSPOTS = [100, 101, 102, 158, 159, 161, 211, 212, 213, 214]
NEEDED = EXAMPLE.is_dir() and REAL.is_dir() and MANIFEST.is_file()


def script_audit_keys():
    """The keys rfd4-generate.py really writes into audit.json (read from its source, not from memory)."""
    tree = ast.parse(SCRIPT.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "write_json" and len(node.args) == 2
                and isinstance(node.args[1], ast.Dict)):
            return {k.value for k in node.args[1].keys if isinstance(k, ast.Constant)}
    raise AssertionError("audit dict not found in rfd4-generate.py")


@unittest.skipUnless(NEEDED, "needs examples/ and docs/eswar/phase1-data")
class ImportCurrentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.make_audit("one")

    def make_audit(self, name, source="provisional_historical", seed=7, requested=60):
        """An audit shaped exactly like the one rfd4-generate.py writes, built on the real verified design."""
        ref_dir = self.dir / f"{name}-ref"
        write_conditioning(EXAMPLE, ref_dir, HOTSPOTS, requested, hotspot_source=source)
        reference = load_reference(ref_dir, MANIFEST)
        out = self.dir / f"{name}-out"
        out.mkdir()
        shutil.copyfile(REAL / "generator.cif", out / "model.cif")
        chains = [{"id": c, "sequence": read_chain_ca(out / "model.cif", c)[0]} for c in ("B", "A")]
        binder = chains[0]["sequence"]
        audit = {
            "status": "conditioned_generation; not a submission-eligible candidate or binding result",
            "reference_audit": {k: v for k, v in reference.items() if k != "target_sequence"},
            "reference_audit_sha256": digest(ref_dir / "audit.json"), "target_manifest_sha256": digest(MANIFEST),
            "spec_sha256": digest(ref_dir / "conditions.json"), "reference_sha256": digest(ref_dir / "target.cif"),
            "target_chain": "A", "binder_chain": "B", "binder_sequence": binder, "binder_length_requested": requested,
            "model": {"path": "/ckpt", "revision": "r"}, "code_revision": "ddd934f7e6b86f7582ce487d5b19f6c526c644b5",
            "seed": seed, "replicates": 1, "batch_size": 1, "compilation": False, "external_metrics": False,
            "recycle_strategy": "thinking_high", "generated_structure": str(out / "model.cif"),
            "generated_sha256": digest(out / "model.cif"), "chains": chains, "raw_structure_hashes": {},
            "generation_seconds": 1.0, "peak_reserved_mib": 1.0, "peak_allocated_mib": 1.0, "host_peak_rss_mib": 1.0}
        (out / "audit.json").write_text(json.dumps(audit))
        return out / "audit.json"

    def run_import(self, name="one", **kwargs):
        return import_current(self.dir / f"{name}-out/audit.json", MANIFEST, self.dir / f"{name}-records.json", **kwargs)[0]

    def rewrite(self, name, **changes):
        path = self.dir / f"{name}-out/audit.json"
        audit = json.loads(path.read_text())
        audit.update(changes)
        path.write_text(json.dumps(audit))

    def test_valid_record(self):
        record = self.run_import()
        self.assertEqual(record["target_id"], "il7ra")
        self.assertEqual(record["format"], "minibinder")
        self.assertEqual(record["campaign_kind"], "competition_campaign")
        self.assertEqual(len(record["sequence"]), 60)
        self.assertEqual(record["qc_reasons"], [])
        self.assertEqual((record["construct"]["start"], record["construct"]["end"]), (21, 239))
        self.assertEqual(record["hotspots_uniprot"], HOTSPOTS)
        self.assertEqual((record["hotspot_source"], record["hotspot_review"]), ("provisional_historical", "unreviewed"))
        self.assertTrue(record["candidate_id"].startswith("il7ra-rfd4-"))
        self.assertIn("seed7", record["backbone_family_id"])

    def test_every_record_starts_unreviewed_and_not_eligible(self):
        record = self.run_import()
        self.assertIs(record["submission_eligible"], False)
        self.assertEqual((record["screen_status"], record["independent_prediction_status"], record["novelty_review"]),
                         ("not_run", "not_run", "pending"))

    def test_reviewed_hotspots_are_marked(self):
        self.make_audit("two", source="reviewed_by_samyuktha_2026-11-18")
        self.assertEqual(self.run_import("two")["hotspot_review"], "reviewed")

    def test_output_file_has_the_candidate_list(self):
        self.run_import()
        written = json.loads((self.dir / "one-records.json").read_text())
        self.assertEqual((written["count"], len(written["candidates"])), (1, 1))

    def test_same_audit_gives_the_same_id_and_different_audits_differ(self):
        self.make_audit("two", seed=8)
        a = self.run_import()
        again = import_current(self.dir / "one-out/audit.json", MANIFEST, self.dir / "again.json")[0]
        b = self.run_import("two")
        self.assertEqual(a["candidate_id"], again["candidate_id"])
        self.assertNotEqual(a["candidate_id"], b["candidate_id"])

    def test_existing_output_is_refused(self):
        self.run_import()
        with self.assertRaises(ValueError):
            self.run_import()

    def test_tampering_and_mismatches_are_refused(self):
        cases = {
            "rehearsal audit": {"status": "IL7RA_reference_rehearsal; not a submission-eligible candidate"},
            "wrong manifest": {"target_manifest_sha256": "0" * 64},
            "changed structure": {"generated_sha256": "0" * 64},
            "binder differs from chains": {"binder_sequence": "A" * 60},
            "wrong binder chain id": {"binder_chain": "A", "target_chain": "B"},
        }
        for label, change in cases.items():
            self.make_audit("t")
            self.rewrite("t", **change)
            with self.assertRaises(ValueError, msg=label):
                self.run_import("t")
            for leftover in self.dir.glob("t-*"):
                shutil.rmtree(leftover) if leftover.is_dir() else leftover.unlink()

    def test_requested_length_that_the_design_does_not_have_is_refused(self):
        self.rewrite("one", binder_length_requested=80)
        with self.assertRaises(ValueError):
            self.run_import()

    def test_length_range_is_accepted_when_the_design_fits(self):
        self.make_audit("r", requested=60)
        self.rewrite("r", binder_length_requested=[50, 70])
        self.assertEqual(self.run_import("r")["binder_length_requested"], [50, 70])

    def test_wrong_format_manifest_adds_a_review_reason(self):
        vhh = json.loads(MANIFEST.read_text())
        vhh.update(format="VHH", min_length=100, max_length=150)
        other = self.dir / "vhh.json"
        other.write_text(json.dumps(vhh))
        self.rewrite("one", target_manifest_sha256=digest(other))
        record = import_current(self.dir / "one-out/audit.json", other, self.dir / "v.json")[0]
        self.assertIn("format_and_framework_review_pending", record["qc_reasons"])
        self.assertIn("wrong_length", record["qc_reasons"])

    def test_dedup_decisions_are_recorded_and_a_dropped_design_is_blocked(self):
        gate = DedupGate()
        kept = gate.offer("first", self.run_import()["sequence"])
        record_kept = import_current(self.dir / "one-out/audit.json", MANIFEST, self.dir / "k.json", dedup_decision=kept)[0]
        self.assertEqual(record_kept["qc_reasons"], [])
        self.assertTrue(record_kept["dedup"]["kept"])
        dropped = gate.offer("second", record_kept["sequence"])
        record_dropped = import_current(self.dir / "one-out/audit.json", MANIFEST, self.dir / "d.json", dedup_decision=dropped)[0]
        self.assertEqual(record_dropped["qc_reasons"], ["near_duplicate:exact_duplicate"])
        self.assertEqual(record_dropped["dedup"]["nearest_id"], "first")

    def test_the_export_guard_still_holds_and_a_fully_reviewed_record_passes(self):
        from binder_campaign.candidates import export_draft
        self.run_import()
        records = self.dir / "one-records.json"
        record_id = json.loads(records.read_text())["candidates"][0]["candidate_id"]
        with self.assertRaises(ValueError):                 # fresh import: reviews missing
            export_draft(records, self.dir / "draft1.csv", [record_id])
        data = json.loads(records.read_text())
        data["candidates"][0].update(novelty_review="passed", independent_prediction_status="reviewed",
                                     screen_status="approximation_reviewed")
        records.write_text(json.dumps(data))
        export_draft(records, self.dir / "draft2.csv", [record_id])   # all reviews filled in by a person
        self.assertIn(data["candidates"][0]["sequence"], (self.dir / "draft2.csv").read_text())


class SchemaCouplingTests(unittest.TestCase):
    @unittest.skipUnless(SCRIPT.is_file(), "needs rfd4-generate.py")
    def test_importer_only_needs_fields_the_generation_script_writes(self):
        written = script_audit_keys()
        source = (Path(__file__).resolve().parents[1] / "binder_campaign/current_candidates.py").read_text()
        needed = {node.slice.value for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "audit"
                  and isinstance(node.slice, ast.Constant)}
        self.assertTrue(needed, "no audit fields found in the importer")
        self.assertEqual(needed - written, set())


if __name__ == "__main__":
    unittest.main()
