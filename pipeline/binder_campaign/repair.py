"""Sequence repair of liabilities with native SolubleMPNN, following the team's lead101 protocol (Phase 5, Task 5.2, optional part).

For a design whose sequence carries a liability (odd cysteine count, N-glycosylation sequon, Asn-Gly, Asp-Pro), redesign ONLY the
positions involved, forbid the offending residue there, keep every other binder residue and the whole target fixed, and let the
unmodified upstream SolubleMPNN (github.com/dauparas/LigandMPNN with the original solublempnn_v_48_020 weights) propose sequences on the
design's own generated backbone. Settings follow the team's protocol: temperature 0.1, seed 401, 16 samples. Unique, format-valid
variants are ranked WITHIN a design by MPNN's own overall_confidence (highest first, then sequence); the first two are kept for common
re-evaluation. MPNN confidence only chooses among samples: it is not affinity and not the screen score. A shortfall is recorded, never
filled by resampling.

A repaired variant is not a new design: it shares its parent's backbone. It must be re-folded and re-checked (ESMFold2 + Boltz-2 + QC)
before anyone trusts it, and a repair does not show the design binds. Nothing here changes a record's review fields.
"""
import json
from pathlib import Path
import re
import subprocess
import tempfile

from .liabilities import flags_for
from .sequence_review import sequence_diagnostic
from .structure import AA3

SCHEMA_VERSION = 1
TEMPERATURE = 0.1
SEED = 401
SAMPLES = 16
MAX_VARIANTS = 2
CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
FLAG_FOR_TYPE = {"cysteine": "odd_cysteine_count", "sequon": "n_glycosylation_sequon", "asn_gly": "asn_gly_motif", "asp_pro": "asp_pro_motif"}


def cif_to_pdb(cif_path):
    """PDB text and {chain: sequence} from an mmCIF, using label_asym_id / label_seq_id. Heavy atoms only; model 1."""
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    d = MMCIF2Dict(str(cif_path))
    models = d.get("_atom_site.pdbx_PDB_model_num")
    first = models[0] if models else None
    lines, chains, last = [], {}, set()
    serial = 0
    for i, name in enumerate(d["_atom_site.label_atom_id"]):
        if models and models[i] != first:
            continue
        element = d["_atom_site.type_symbol"][i].strip()
        if element in ("H", "D"):
            continue
        chain, residue, comp = d["_atom_site.label_asym_id"][i], int(d["_atom_site.label_seq_id"][i]), d["_atom_site.label_comp_id"][i]
        if len(chain) != 1 or comp not in AA3:
            raise ValueError(f"Cannot write a PDB for chain '{chain}' residue '{comp}'")
        x, y, z = (float(d[k][i]) for k in ("_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z"))
        serial += 1
        atom = f" {name:<3s}" if len(name) < 4 and len(element) == 1 else f"{name:<4s}"
        lines.append(f"ATOM  {serial:5d} {atom} {comp:>3s} {chain}{residue:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}")
        if name == "CA" and (chain, residue) not in last:
            last.add((chain, residue))
            chains[chain] = chains.get(chain, "") + AA3[comp]
    if not lines:
        raise ValueError("The structure holds no atoms")
    return "\n".join(lines) + "\nTER\nEND\n", chains


def liability_edits(sequence):
    """{position (1-based): residue letters to forbid there} and the liability types addressed. Empty if nothing needs repair."""
    edits, types = {}, set()
    cysteines = [i + 1 for i, a in enumerate(sequence) if a == "C"]
    if len(cysteines) % 2 == 1:
        types.add("cysteine")
        for p in cysteines:
            edits.setdefault(p, set()).add("C")
    for i in range(len(sequence) - 2):
        if sequence[i] == "N" and sequence[i + 1] != "P" and sequence[i + 2] in "ST":
            types.add("sequon")
            edits.setdefault(i + 1, set()).add("N")
    for i in range(len(sequence) - 1):
        if sequence[i:i + 2] == "NG":
            types.add("asn_gly")
            edits.setdefault(i + 1, set()).add("N")
        if sequence[i:i + 2] == "DP":
            types.add("asp_pro")
            edits.setdefault(i + 1, set()).add("D")
    return {p: "".join(sorted(v)) for p, v in sorted(edits.items())}, sorted(types)


def binder_chain_index(chains, binder_chain="B"):
    """Position of the binder in the native output line: the upstream joins chains with ':' in ALPHABETICAL chain order."""
    ordered = sorted(chains)
    if binder_chain not in ordered:
        raise ValueError(f"Chain {binder_chain} is not in the structure")
    return ordered.index(binder_chain)


def parse_outputs(fasta_path, binder_chain_index):
    """The samples of a native run: [{sequence, overall_confidence, sample, header}]. The first record is the input and is skipped.

    Format read from the upstream run.py: '>name, T=..., seed=...' for the input, then '>name, id=N, T=..., overall_confidence=X, ...'
    per sample (id counts from 0); several chains are joined with ':'."""
    records, header = [], None
    for line in Path(fasta_path).read_text().splitlines():
        if line.startswith(">"):
            header = line[1:]
        elif header is not None:
            records.append((header, line.strip()))
            header = None
    out = []
    for header, sequence in records[1:]:
        fields = dict(f.strip().split("=", 1) for f in header.split(",") if "=" in f)
        parts = sequence.split(":")
        if "overall_confidence" not in fields or binder_chain_index >= len(parts):
            raise ValueError("Unexpected native output format: " + header[:80])
        out.append({"sequence": parts[binder_chain_index], "overall_confidence": float(fields["overall_confidence"]),
                    "sample": int(fields.get("id", len(out))), "header": header})
    return out


def choose(parent, samples, edits, types, max_variants=MAX_VARIANTS):
    """Apply the team's selection rule. Returns (variants, rejected, shortfall)."""
    allowed = {}
    for sample in samples:
        s = sample["sequence"]
        problems = []
        if len(s) != len(parent) or set(s) - CANONICAL:
            problems.append("format")
        else:
            for i, (a, b) in enumerate(zip(parent, s), 1):
                if i in edits:
                    if b in edits[i]:
                        problems.append(f"forbidden_residue_at_{i}")
                elif a != b:
                    problems.append(f"changed_fixed_position_{i}")
            try:
                remaining = set(flags_for(sequence_diagnostic(s)))
            except ValueError:
                remaining = {"not_diagnosable"}
            left = remaining & {FLAG_FOR_TYPE[t] for t in types}
            if left:
                problems.append("liability_remains:" + ",".join(sorted(left)))
        if s == parent:
            problems.append("identical_to_parent")
        allowed[sample["sample"]] = problems
    rejected = [{"sample": k, "problems": v} for k, v in allowed.items() if v]
    valid = [s for s in samples if not allowed[s["sample"]]]
    best = {}
    for s in valid:                                                   # exact duplicates collapse to their highest-confidence sample
        if s["sequence"] not in best or s["overall_confidence"] > best[s["sequence"]]["overall_confidence"]:
            best[s["sequence"]] = s
    ranked = sorted(best.values(), key=lambda s: (-s["overall_confidence"], s["sequence"]))
    chosen = ranked[:max_variants]
    shortfall = max_variants - len(chosen) if len(chosen) < max_variants else 0
    return chosen, rejected, shortfall


def run_native(pdb_path, edits, out_dir, code_dir, weights, python, binder_chain="B", seed=SEED, samples=SAMPLES,
               temperature=TEMPERATURE):
    """Run the unmodified upstream run.py. Returns the path of the sequences FASTA."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    omit = out_dir / "omit.json"
    omit.write_text(json.dumps({f"{binder_chain}{p}": letters for p, letters in edits.items()}))
    command = [str(python), str(Path(code_dir) / "run.py"), "--model_type", "soluble_mpnn", "--checkpoint_soluble_mpnn", str(weights),
               "--pdb_path", str(pdb_path), "--out_folder", str(out_dir), "--redesigned_residues",
               " ".join(f"{binder_chain}{p}" for p in edits), "--omit_AA_per_residue", str(omit), "--temperature", str(temperature),
               "--seed", str(seed), "--batch_size", str(samples), "--number_of_batches", "1", "--save_stats", "1",
               "--pack_side_chains", "0"]                   # upstream picks CUDA if present, else CPU: it has no device flag
    done = subprocess.run(command, capture_output=True, text=True, cwd=str(code_dir))
    (out_dir / "run.log").write_text(done.stdout + "\n" + done.stderr)
    if done.returncode != 0:
        raise RuntimeError("SolubleMPNN failed: " + (done.stderr or done.stdout).strip()[-400:])
    fastas = sorted((out_dir / "seqs").glob("*.fa"))
    if len(fastas) != 1:
        raise RuntimeError("SolubleMPNN wrote no sequences file")
    return fastas[0], command


def variant_record(parent, sample, number, edits, types):
    """A candidate record for a repaired variant. Evidence about the parent (scores, QC, novelty) is NOT carried over: it must be redone."""
    keep = ("schema_version", "target_id", "format", "generator", "generator_revision", "campaign_kind", "seed", "binder_length_requested",
            "hotspots_uniprot", "hotspot_source", "hotspot_review", "construct", "target_manifest_sha256", "generator_audit",
            "generator_structure", "generator_structure_sha256", "generator_target_chain", "generator_binder_chain", "source_audit")
    record = {k: parent[k] for k in keep if k in parent}
    record.update({"candidate_id": f"{parent['candidate_id']}-r{number}", "sequence": sample["sequence"],
                   "parent_candidate_id": parent["candidate_id"], "generator_structure_sequence": parent["sequence"],
                   "backbone_family_id": parent.get("backbone_family_id"), "qc_reasons": [],
                   "repair": {"method": "SolubleMPNN native, solublempnn_v_48_020", "temperature": TEMPERATURE, "seed": SEED,
                              "samples": SAMPLES, "redesigned_positions": sorted(edits), "forbidden": edits, "addressed": types,
                              "overall_confidence": sample["overall_confidence"], "sample": sample["sample"],
                              "rank_within_parent": number,
                              "note": "MPNN confidence only chose among samples; it is not affinity and not the screen score."},
                   "screen_status": "not_run", "independent_prediction_status": "not_run", "novelty_review": "pending",
                   "submission_eligible": False})
    return record
