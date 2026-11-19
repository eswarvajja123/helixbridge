from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.alphafold_msa import alignment_for, complex_a3m, read_rows, single_a3m  # noqa: E402

TARGET = "ACDEFGHIKL"
MSA = f">q\n{TARGET}\n>h1 12 0.9\nACDEFGHIKL\n>h2\nACD-FGHaaIKL\n>h3\n----------\n>dup\nACDEFGHIKL\n>h4\nAC-EFGHIKL\n"
BINDER = "MNPQRSTV"


class Complex(unittest.TestCase):
    def lines(self, **kw):
        return complex_a3m(MSA, kw.get("target", TARGET), kw.get("binder", BINDER)).splitlines()

    def test_header_query_and_every_row_has_the_full_length(self):
        out = self.lines()
        self.assertEqual(out[0], "#10,8\\t1,1".replace("\\t", "\t"))
        self.assertEqual(out[2], TARGET + BINDER)
        self.assertTrue(all(len(s) == 18 for s in out[2::2]))

    def test_homolog_rows_are_padded_over_the_binder_and_the_binder_has_its_own_row(self):
        out = self.lines()
        self.assertIn("ACD-FGHIKL" + "-" * 8, out)
        self.assertEqual(out[-2:], [">102", "-" * 10 + BINDER])

    def test_repeats_empty_rows_and_the_query_copy_are_dropped_and_headers_have_no_tabs(self):
        out = self.lines()
        rows = out[3:-2:2]
        self.assertEqual(len(rows), 2)                                   # h2 and h4 only (h1/dup repeat the query, h3 is all gaps)
        self.assertTrue(all("\t" not in h and " " not in h for h in out[3::2][:-1]))

    def test_a_single_sequence_target_gives_a_complex_with_only_the_query_and_the_binder(self):
        out = complex_a3m(single_a3m(TARGET), TARGET, BINDER).splitlines()
        self.assertEqual(out, ["#10,8\t1,1", ">101\t102", TARGET + BINDER, ">101", TARGET + "-" * 8, ">102", "-" * 10 + BINDER])      # each chain has an unpaired row
        with self.assertRaises(ValueError):
            single_a3m("ACDX")
        with self.assertRaises(ValueError):
            single_a3m("")

    def test_alignment_choice(self):
        self.assertEqual(alignment_for(MSA, TARGET, "deep"), MSA)                      # the target's own alignment
        self.assertEqual(alignment_for(MSA, TARGET, "single"), single_a3m(TARGET))      # like-for-like mode
        self.assertEqual(alignment_for(MSA, "GGGGGGGGGG", "deep"), single_a3m("GGGGGGGGGG"))   # a decoy never gets the real target's alignment
        other = ">q\nGGGGGGGGGG\n>h\nGGGGGGGGGG\n"
        mapping = {TARGET: MSA, "GGGGGGGGGG": other}
        self.assertEqual(alignment_for(mapping, "GGGGGGGGGG", "deep"), other)             # a decoy with its own alignment gets it
        self.assertEqual(alignment_for(mapping, "GGGGGGGGGG", "single"), single_a3m("GGGGGGGGGG"))
        self.assertEqual(alignment_for(mapping, "AAAAAAAAAA", "deep"), single_a3m("AAAAAAAAAA"))
        with self.assertRaises(ValueError):
            alignment_for(MSA, TARGET, "other")

    def test_a_binder_with_its_own_alignment_gets_its_homolog_rows_padded_over_the_target(self):
        binder_msa = f">q\n{BINDER}\n>b1 9 0.5\nMNPQRSTV\n>b2\nMNPQ-STV\n>b3\n--------\n"
        out = complex_a3m(MSA, TARGET, BINDER, binder_msa).splitlines()
        self.assertEqual(out[-2:], [">b2", "-" * 10 + "MNPQ-STV"])
        self.assertIn(">102", out)
        self.assertTrue(all(len(s) == 18 for s in out[2::2]))
        self.assertEqual(len([h for h in out if h in (">b1", ">b3")]), 0)               # repeats of the query and empty rows are dropped
        with self.assertRaisesRegex(ValueError, "not the sequence"):
            complex_a3m(MSA, TARGET, BINDER, ">q\nAAAAAAAA\n>x\nAAAAAAAA\n")
        self.assertEqual(complex_a3m(MSA, TARGET, BINDER), complex_a3m(MSA, TARGET, BINDER, None))

    def test_insertions_are_removed(self):
        self.assertEqual(read_rows(MSA)[2][1], "ACD-FGHIKL")

    def test_wrong_alignment_or_sequences_are_refused(self):
        with self.assertRaisesRegex(ValueError, "first row"):
            complex_a3m(MSA, "CCCCCCCCCC", BINDER)
        with self.assertRaisesRegex(ValueError, "binder"):
            complex_a3m(MSA, TARGET, "MNPX")
        with self.assertRaisesRegex(ValueError, "binder"):
            complex_a3m(MSA, TARGET, "")
        with self.assertRaisesRegex(ValueError, "target"):
            complex_a3m(MSA, "ACDXFGHIKL", BINDER)
        with self.assertRaisesRegex(ValueError, "length"):
            complex_a3m(f">q\n{TARGET}\n>s\nACD\n", TARGET, BINDER)

    def test_the_real_target_alignment_works(self):
        path = Path(__file__).resolve().parents[2] / "docs/eswar/phase1-data/target.a3m"
        if not path.is_file():
            self.skipTest("target alignment not present")
        text = path.read_text()
        target = read_rows(text)[0][1]
        out = complex_a3m(text, target, "A" * 69).splitlines()
        self.assertEqual(out[0], "#219,69\t1,1")
        self.assertTrue(all(len(s) == 288 for s in out[2::2]))
        self.assertGreater(len(out) // 2, 900)


if __name__ == "__main__":
    unittest.main()
