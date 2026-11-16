import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.conditioning import build_conditions, write_conditioning
from binder_campaign.runner import digest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/il7ra-rehearsal"
HISTORICAL_HOTSPOTS = [100, 101, 102, 158, 159, 161, 211, 212, 213, 214]
RESOLVED = list(range(37, 232))


class BuildConditionsTests(unittest.TestCase):
    def test_hotspot_outside_resolved_structure_is_rejected(self):
        # 25 is inside the official 21-239 interval but has no coordinates in the reference.
        with self.assertRaises(ValueError):
            build_conditions(RESOLVED, [25, 100], 60)
        with self.assertRaises(ValueError):
            build_conditions(RESOLVED, [100, 235], 60)

    def test_bad_hotspots_are_rejected(self):
        for bad in ([], [100, 100], [True], ["100"], [100.0]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                build_conditions(RESOLVED, bad, 60)

    def test_length_boundaries(self):
        self.assertIn("40, /0", build_conditions(RESOLVED, [100], 40)["il7ra"]["contig"])
        self.assertIn("100, /0", build_conditions(RESOLVED, [100], 100)["il7ra"]["contig"])
        for bad in (39, 101, (50, 40), (30, 60), (60, 120), (50,), "60", True, None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                build_conditions(RESOLVED, [100], bad)

    def test_length_range_makes_a_variable_contig(self):
        contig = build_conditions(RESOLVED, [100], (50, 80))["il7ra"]["contig"]
        self.assertEqual(contig, "50-80, /0, A37-231")

    def test_non_contiguous_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            build_conditions([37, 38, 40], [37], 60)
        with self.assertRaises(ValueError):
            build_conditions([], [37], 60)

    def test_hotspots_are_sorted(self):
        select = build_conditions(RESOLVED, [214, 100, 150], 60)["il7ra"]["conditions"]["C_HOT"]["true"][0]["select"]
        self.assertEqual(select, "sel('A/*/[100,150,214]')")


@unittest.skipUnless(EXAMPLE.is_dir(), "needs examples/il7ra-rehearsal from the full project")
class HistoricalReproductionTests(unittest.TestCase):
    def test_rebuilds_the_historical_conditions_file_exactly(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out"
            audit = write_conditioning(EXAMPLE, out, HISTORICAL_HOTSPOTS, 60,
                                       hotspot_source="provisional_historical")
            self.assertEqual((out / "conditions.json").read_bytes(), (EXAMPLE / "conditions.json").read_bytes())
            self.assertEqual(digest(out / "target.cif"), digest(EXAMPLE / "target.cif"))
            self.assertEqual(audit["hotspots_uniprot"], HISTORICAL_HOTSPOTS)
            self.assertEqual(audit["binder_length"], 60)
            self.assertIsInstance(audit["binder_length"], int)
            self.assertEqual(audit["hotspot_source"], "provisional_historical")
            self.assertEqual(audit["files"], {"target.cif": digest(out / "target.cif"),
                                              "conditions.json": digest(out / "conditions.json")})
            self.assertNotEqual(audit["role"], "reference_rehearsal_only")

    def test_range_is_recorded_as_a_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            audit = write_conditioning(EXAMPLE, Path(temp) / "o", HISTORICAL_HOTSPOTS, (50, 80),
                                       hotspot_source="provisional_historical")
            self.assertEqual(audit["binder_length"], [50, 80])

    def test_source_must_be_stated_and_output_must_be_fresh(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                write_conditioning(EXAMPLE, Path(temp) / "a", HISTORICAL_HOTSPOTS, 60, hotspot_source=" ")
            used = Path(temp) / "used"
            used.mkdir()
            (used / "x").write_text("x")
            with self.assertRaises(ValueError):
                write_conditioning(EXAMPLE, used, HISTORICAL_HOTSPOTS, 60, hotspot_source="s")

    def test_changed_reference_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "ref"
            copy.mkdir()
            for name in ("target.cif", "conditions.json", "audit.json"):
                (copy / name).write_bytes((EXAMPLE / name).read_bytes())
            (copy / "target.cif").write_bytes((copy / "target.cif").read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                write_conditioning(copy, Path(temp) / "out", HISTORICAL_HOTSPOTS, 60, hotspot_source="s")


if __name__ == "__main__":
    unittest.main()
