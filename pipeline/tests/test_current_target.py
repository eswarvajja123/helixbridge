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


if __name__ == "__main__":
    unittest.main()
