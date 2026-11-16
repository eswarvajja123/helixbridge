"""Lock the Phase 1 claims about the current IL-7Ra target (21-239) to the saved data."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/eswar/phase1-data"
EXAMPLE = ROOT / "examples/il7ra-rehearsal"
HOTSPOTS = {100: "L", 101: "L", 102: "I", 158: "K", 159: "Y", 161: "K", 211: "H", 212: "Y", 213: "F", 214: "K"}


@unittest.skipUnless((DATA / "il7ra-manifest.json").is_file(), "needs docs/eswar/phase1-data")
class CurrentTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((DATA / "il7ra-manifest.json").read_text())
        cls.sequence = cls.manifest["sequence"]

    def test_official_interval_and_endpoints(self):
        m = self.manifest
        self.assertEqual((m["uniprot"], m["start"], m["end"]), ("P16871-1", 21, 239))
        self.assertEqual(len(self.sequence), 219)
        self.assertEqual((self.sequence[0], self.sequence[-1]), ("E", "D"))
        self.assertEqual(hashlib.sha256(self.sequence.encode()).hexdigest(), m["sequence_sha256"])

    @unittest.skipUnless(EXAMPLE.is_dir(), "needs examples/il7ra-rehearsal")
    def test_frozen_reference_keeps_uniprot_numbering(self):
        audit = json.loads((EXAMPLE / "audit.json").read_text())
        start = self.manifest["start"]
        self.assertEqual(self.sequence[audit["positions"][0] - start: audit["positions"][-1] - start + 1],
                         audit["reference_sequence"])
        for position, residue in HOTSPOTS.items():
            self.assertEqual(self.sequence[position - start], residue)

    def test_msa_matches_the_current_sequence(self):
        manifest = json.loads((DATA / "msa-manifest.json").read_text())
        raw = (DATA / "target.a3m").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest["msa_sha256"])
        names, seqs = [], []
        for line in raw.decode().splitlines():
            if line.startswith(">"):
                names.append(line)
                seqs.append("")
            else:
                seqs[-1] += line.strip()
        self.assertEqual(len(seqs), manifest["rows"])
        self.assertEqual(seqs[0], self.sequence)
        self.assertEqual(manifest["sequence_sha256"], self.manifest["sequence_sha256"])
        for seq in seqs:  # aligned width (ignoring lowercase insertions) must equal the query length
            self.assertEqual(sum(1 for c in seq if not c.islower()), 219)
        self.assertTrue(manifest["public_target_only"])

    def test_comparison_report_is_consistent_with_the_notes(self):
        report = json.loads((DATA / "reference-vs-3DI3.json").read_text())
        self.assertEqual(report["pdb_uniprot_range"], [37, 229])
        self.assertEqual(report["pdb_residues"], 193)
        self.assertEqual(report["pdb_vs_uniprot_identity_differences"], [{"position": 66, "pdb": "T", "uniprot": "I"}])
        self.assertEqual(report["only_in_reference"], [230, 231])
        self.assertEqual(report["reference_vs_uniprot_identity_differences"], [])
        self.assertLess(report["ca_rmsd_angstrom_over_shared"], 1.5)

    def test_natural_il7_interface_and_residue_66(self):
        report = json.loads((DATA / "reference-vs-3DI3.json").read_text())
        self.assertEqual(report["partner_chain"], "A")
        self.assertEqual(report["hotspots_in_partner_interface"], [100, 101, 102, 158, 159, 211, 212, 213])
        self.assertEqual(report["hotspots_not_in_partner_interface"], [161, 214])
        self.assertGreater(report["residue_66_min_heavy_atom_distance_to_partner_angstrom"], 20)
        self.assertEqual(report["interface_count"], len(report["interface_residues_uniprot"]))

    def test_glycan_context(self):
        report = json.loads((DATA / "glycan-context.json").read_text())
        self.assertEqual(report["sequons_in_target_sequence"], [49, 65, 151, 182, 232, 233])
        self.assertEqual(report["sequons_with_observed_glycan"], [49, 65, 151])
        self.assertEqual(report["sequons_resolved_without_glycan"], [182])
        self.assertEqual(report["sequons_outside_resolved_range"], [232, 233])
        self.assertEqual(report["hotspots_near_any_glycan"], [])
        closest = min(d for g in report["glycans"] for d in g["min_distance_to_hotspots_angstrom"].values())
        self.assertGreater(closest, report["screen_cutoff_angstrom"])

    def test_hotspot_options(self):
        report = json.loads((DATA.parent / "hotspot-options/hotspot-options.json").read_text())
        a, b, c = (report["options"][k] for k in ("A_current_provisional", "B_exposed_natural_site", "C_offsite_exploratory"))
        self.assertEqual(a["hotspots"], [100, 101, 102, 158, 159, 161, 211, 212, 213, 214])
        self.assertEqual(len(a["in_natural_interface"]), 8)
        self.assertEqual(a["buried_below_30"], [101])
        self.assertEqual(b["hotspots"], [78, 79, 98, 100, 102, 103, 158, 159, 211, 212])
        self.assertEqual(len(b["in_natural_interface"]), 10)
        self.assertEqual(c["hotspots"], [170, 172, 173, 174, 175, 176, 177, 178, 179, 200])
        self.assertEqual(c["in_natural_interface"], [])
        self.assertTrue(all(h < 220 for h in c["hotspots"]))  # never next to the C-terminal tag
        for option in (b, c):
            self.assertGreaterEqual(option["min_glycan_distance"], 10)
            self.assertEqual(option["count"], 10)

    def test_option_conditioning_files_match_the_options(self):
        options = DATA.parent / "hotspot-options"
        report = json.loads((options / "hotspot-options.json").read_text())
        for letter, key in (("A", "A_current_provisional"), ("B", "B_exposed_natural_site"), ("C", "C_offsite_exploratory")):
            audit = json.loads((options / letter / "audit.json").read_text())
            self.assertEqual(audit["hotspots_uniprot"], report["options"][key]["hotspots"])
            self.assertEqual(audit["binder_length"], 60)
            self.assertEqual(audit["hotspot_source"].startswith("proposed"), letter != "A")
            selection = json.loads((options / letter / "conditions.json").read_text())["il7ra"]["conditions"]["C_HOT"]["true"][0]["select"]
            self.assertEqual(selection, "sel('A/*/[" + ",".join(map(str, audit["hotspots_uniprot"])) + "]')")
        if EXAMPLE.is_dir():  # Option A must equal the historical conditioning file exactly
            self.assertEqual((options / "A/conditions.json").read_bytes(), (EXAMPLE / "conditions.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
