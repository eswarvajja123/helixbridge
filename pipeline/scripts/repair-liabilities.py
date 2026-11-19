"""Optional sequence repair of liabilities with SolubleMPNN (Task 5.2, optional part). Follows the team's lead101 protocol.

For each chosen design with a liability (odd cysteine count, N-glycosylation sequon, Asn-Gly, Asp-Pro): redesign only the positions
involved, forbid the offending residue there, keep everything else fixed, sample 16 sequences at temperature 0.1 (seed 401) from the
design's own backbone, and keep the best two unique valid ones by MPNN's own confidence. A repaired variant shares its parent's
backbone and must be re-folded and re-checked (ESMFold2 + Boltz-2 + QC) before it is trusted.

Three ways to run it:
  plan   free: show, per design, the positions to redesign and the exact native command
  run    run the native SolubleMPNN yourself (needs the upstream code, the weights, openfold and prody in the python you give)
  ingest read outputs of a native run you did elsewhere (--native-dir with <candidate_id>/seqs/*.fa); use the design.pdb files that
         `plan` wrote as the inputs of that run

Example:
  python pipeline/scripts/repair-liabilities.py plan --records results/liabilities-eswar-l2/liabilities-candidates.json \
        --selection results/selection-eswar-s2/selection.json --out results/repair-eswar-r1
Upstream: github.com/dauparas/LigandMPNN, weights solublempnn_v_48_020.pt from files.ipd.uw.edu. Running them is running third-party
code and loading third-party pickled weights: do it only in an environment you are comfortable with.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign import repair
from binder_campaign.repair import MAX_VARIANTS, SAMPLES, SEED, TEMPERATURE
from binder_campaign.runner import write_json

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = parser.add_subparsers(dest="mode", required=True)
for name in ("plan", "run", "ingest"):
    p = sub.add_parser(name)
    p.add_argument("--records", type=Path, required=True)
    p.add_argument("--selection", type=Path, default=None, help="only designs in this shortlist (selection.json); default: every design")
    p.add_argument("--out", type=Path, required=True)
    if name == "ingest":
        p.add_argument("--native-dir", type=Path, required=True, help="folder holding <candidate_id>/seqs/*.fa from your own native run")
        p.add_argument("--tag", default="", help="1 to 6 letters or digits put in the variant ids (`<parent>-<tag>r1`), so variants from a second "
                       "run with other settings never collide with the first run's ids")
        p.add_argument("--max-variants", type=int, default=MAX_VARIANTS,
                       help=f"variants kept per design (default {MAX_VARIANTS}, the team's protocol); a larger number is a wider screen and every "
                            "variant must then be re-checked, AlphaFold included")
    if name == "run":
        p.add_argument("--mpnn-code", type=Path, required=True)
        p.add_argument("--weights", type=Path, required=True)
        p.add_argument("--python", type=Path, required=True, help="a python with torch, numpy, prody and openfold")
args = parser.parse_args()
keep = getattr(args, "max_variants", MAX_VARIANTS)
try:
    if not isinstance(keep, int) or not 1 <= keep <= 8:
        raise ValueError("--max-variants must be a whole number from 1 to 8")
    tag = getattr(args, "tag", "")
    if tag and not (tag.isalnum() and tag.isascii() and len(tag) <= 6):
        raise ValueError("--tag must be 1 to 6 letters or digits")
    run_settings = {}
    if args.mode == "ingest":                                   # the temperature and seed the native run really used, when it recorded them
        recorded = args.native_dir.parent / "native-run.json"
        if recorded.is_file():
            run_settings = {k: v for k, v in json.loads(recorded.read_text()).items() if k in ("temperature", "seed")}
    if args.out.exists():
        raise ValueError("Use a fresh output directory")
    doc = json.loads(args.records.read_text())
    records = doc["candidates"]
    wanted = None
    if args.selection:
        wanted = {s["candidate_id"] for s in json.loads(args.selection.read_text())["selected"]}
    parents = [r for r in records if (wanted is None or r["candidate_id"] in wanted) and not r.get("parent_candidate_id")]
    args.out.mkdir(parents=True)
    plan, variants, summary = [], [], []
    for parent in parents:
        edits, types = repair.liability_edits(parent["sequence"])
        if not edits:
            continue
        folder = args.out / parent["candidate_id"]
        folder.mkdir()
        pdb, chains = repair.cif_to_pdb(parent["generator_structure"])
        (folder / "design.pdb").write_text(pdb)
        index = repair.binder_chain_index(chains, parent["generator_binder_chain"])
        entry = {"candidate_id": parent["candidate_id"], "types": types, "redesign_positions": edits,
                 "binder_chain": parent["generator_binder_chain"], "binder_chain_index": index}
        plan.append(entry)
        if args.mode == "plan":
            continue
        if args.mode == "run":
            fasta, command = repair.run_native(folder / "design.pdb", edits, folder / "native", args.mpnn_code, args.weights, args.python,
                                               parent["generator_binder_chain"])
            entry["command"] = command
        else:
            base = args.native_dir / parent["candidate_id"]
            found = sorted((base / "seqs").glob("*.fa")) or sorted((base / "native" / "seqs").glob("*.fa"))
            if len(found) != 1:
                summary.append({"candidate_id": parent["candidate_id"], "status": "no_native_output"})
                continue
            fasta = found[0]
        samples = repair.parse_outputs(fasta, index)
        chosen, rejected, shortfall = repair.choose(parent["sequence"], samples, edits, types, keep)
        for n, c in enumerate(chosen, 1):
            variants.append(repair.variant_record(parent, c, n, edits, types, run_settings.get("temperature", TEMPERATURE),
                                                  run_settings.get("seed", SEED), tag))
        summary.append({"candidate_id": parent["candidate_id"], "status": "repaired" if chosen else "no_valid_variant", "types": types,
                        "samples": len(samples), "rejected": len(rejected), "variants": len(chosen), "shortfall": shortfall})
    write_json(args.out / "repair-plan.json", {"mode": args.mode, "settings": {"temperature": run_settings.get("temperature", TEMPERATURE),
                                                                                 "seed": run_settings.get("seed", SEED), "samples": SAMPLES,
                                              "max_variants_per_design": keep}, "designs": plan, "results": summary})
    if variants:
        write_json(args.out / "repair-candidates.json", {"candidates": variants, "count": len(variants),
                   "note": "Repaired variants share their parent's backbone. Re-fold and re-check them before trusting them."})
    print(json.dumps({"output_folder": str(args.out), "designs_with_liabilities": len(plan), "variants": len(variants),
                      "results": summary if args.mode != "plan" else None}, indent=2))
except (ValueError, RuntimeError, FileNotFoundError, KeyError) as error:
    sys.exit(f"REFUSED: {error}")
