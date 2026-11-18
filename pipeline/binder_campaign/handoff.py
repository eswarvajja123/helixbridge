"""Hand-off package for the independent AlphaFold cross-check (Phase 5, Task 5.4).

5.4 is a person's step (Stefanos runs AlphaFold on the finalists). This writes what that person needs and nothing more: for each
shortlisted design the target and binder sequences in the formats AlphaFold tools take, and a BLANK results sheet. It never fills
in a result and never sets independent_prediction_status: that field is the person's to set after reading the AlphaFold output.
"""
import csv
import json
from pathlib import Path

SCHEMA_VERSION = 1
SHEET_COLUMNS = ["candidate_id", "tool_and_version", "ptm", "iptm", "ipsae_max", "binder_ca_rmsd_to_generator_A", "interface_looks_as_designed",
                 "notes", "reviewer", "date"]


def export(records, selected_ids, target, out_dir):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise ValueError("Use a fresh output directory")
    selected_ids = list(selected_ids)
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("Give a non-empty list of distinct candidate ids")
    by_id = {r["candidate_id"]: r for r in records}
    missing = [c for c in selected_ids if c not in by_id]
    if missing:
        raise ValueError("Not in the records: " + ", ".join(missing))
    sequence = target["sequence"]
    if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
        raise ValueError("The target sequence is missing or not canonical")
    for cid in selected_ids:
        seq = by_id[cid]["sequence"]
        if not seq or set(seq) - set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError("A design has a non-canonical sequence: " + cid)
    out_dir.mkdir(parents=True)
    with (out_dir / "alphafold-inputs.fasta").open("w") as f:                      # one complex per entry, chains joined with ':'
        for cid in selected_ids:
            f.write(f">{cid}\n{sequence}:{by_id[cid]['sequence']}\n")
    with (out_dir / "colabfold-batch.csv").open("w", newline="") as f:             # ColabFold batch format: id,sequence
        w = csv.writer(f)
        w.writerow(["id", "sequence"])
        for cid in selected_ids:
            w.writerow([cid, f"{sequence}:{by_id[cid]['sequence']}"])
    with (out_dir / "designs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "binder_sequence", "binder_length", "hotspot_source", "generator_seed"])
        for cid in selected_ids:
            r = by_id[cid]
            w.writerow([cid, r["sequence"], len(r["sequence"]), r.get("hotspot_source", ""), r.get("seed", "")])
    with (out_dir / "alphafold-results-template.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SHEET_COLUMNS)
        for cid in selected_ids:
            w.writerow([cid] + [""] * (len(SHEET_COLUMNS) - 1))
    (out_dir / "README.md").write_text(README.format(n=len(selected_ids), tlen=len(sequence), start=target.get("start", "?"),
                                                      end=target.get("end", "?"), uniprot=target.get("uniprot", "?")))
    manifest = {"schema": SCHEMA_VERSION, "designs": selected_ids, "target": {"uniprot": target.get("uniprot"), "start": target.get("start"),
                "end": target.get("end"), "length": len(sequence)}, "status": "inputs_only_no_results",
                "note": "No AlphaFold result is recorded here and no review field was set."}
    (out_dir / "handoff.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


README = """# AlphaFold cross-check: inputs for {n} shortlisted designs

Task 5.4 of the plan. A person runs AlphaFold on these; nothing here has been run, and no review has been recorded.

- Target: IL-7R alpha, UniProt {uniprot} residues {start}-{end} ({tlen} aa), chain A. Binder: chain B.
- `alphafold-inputs.fasta` has one complex per entry (`TARGET:BINDER`). `colabfold-batch.csv` is the same in ColabFold's batch format.
- Fill in `alphafold-results-template.csv` yourself (tool and version, pTM, ipTM, ipSAE, binder RMSD to the generated pose, whether the
  interface looks as designed). Please also keep the raw AlphaFold outputs.
- Compare against what the two earlier predictors (ESMFold2, Boltz-2) said, but record what AlphaFold says, not what the others said.
- When you have read the results and decided, a person sets `independent_prediction_status` to `reviewed` for the design (see
  `docs/eswar/RUN-GPU.md`, Stage L). Nothing here sets it.
"""
