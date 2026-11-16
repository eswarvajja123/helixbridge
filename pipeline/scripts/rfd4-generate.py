"""Generate ONE RFD4-Proteina binder against a conditioned reference (Phase 2, Task 2.1).

Generalizes rfd4-il7ra-seed.py: any non-negative seed, any prepared conditioning directory
(binder_campaign.conditioning), an explicit target manifest, and a binder length that may be an
exact number or a [min, max] range. All identity checks run BEFORE any GPU work and live in
binder_campaign.generation_input, where they are unit-tested.

  GPU run : TASK_ROOT=<root> python scripts/rfd4-generate.py --reference-dir D --manifest M --out O --seed N
  Dry run : add --dry-run to run every input check and print the exact settings without importing
            torch, biotite or RFD4 (works on a CPU-only machine)

TASK_ROOT must contain env/rfd4-checkpoints.json and data/rfd4 (as in the historical adapter).
Nothing here is a binding result or a submission-eligible candidate.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.generation_input import build_overrides, check_generated_chains, check_seed, load_reference
from binder_campaign.runner import digest, write_json

RFD4_CODE_REVISION = "ddd934f7e6b86f7582ce487d5b19f6c526c644b5"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--reference-dir", type=Path, required=True, help="directory with audit.json, target.cif, conditions.json")
parser.add_argument("--manifest", type=Path, required=True, help="target manifest (id, start, end, sequence)")
parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--dry-run", action="store_true", help="check inputs and print the settings; no GPU, no model")
args = parser.parse_args()

reference_dir, out = args.reference_dir.resolve(), args.out.resolve()
seed = check_seed(args.seed)
if out.exists() and any(out.iterdir()):
    raise ValueError("Use a fresh native output directory")
reference = load_reference(reference_dir, args.manifest)        # every identity check, before the GPU
spec = reference_dir / "conditions.json"

if args.dry_run:
    placeholder = {"path": "<checkpoint path from env/rfd4-checkpoints.json>", "revision": "<checkpoint revision>"}
    print(json.dumps({"dry_run": True, "seed": seed, "target_id": reference["target_id"], "role": reference["role"],
                      "binder_length": reference["binder_length"], "hotspots": reference["hotspots_uniprot"],
                      "hotspot_source": reference.get("hotspot_source"),
                      "target_residues": len(reference["target_sequence"]),
                      "overrides": build_overrides(spec, out, seed, placeholder)}, indent=2))
    sys.exit(0)

# GPU-only imports start here
root = Path(os.environ["TASK_ROOT"])
os.environ["DATA_PATH"] = str(root / "data/rfd4")
os.environ["WANDB_MODE"] = "disabled"
import numpy as np
import torch
from biotite.structure.io.pdbx import CIFFile, get_structure
from binder_campaign.structure import AA3
import rfd4_proteina as rfd4

model_record = json.loads((root / "env/rfd4-checkpoints.json").read_text())
if not torch.cuda.is_available():
    raise RuntimeError("A CUDA GPU is required (use --dry-run to check inputs without one)")
torch.cuda.reset_peak_memory_stats()
started = time.monotonic()
rfd4.generate(overrides=build_overrides(spec, out, seed, model_record))
torch.cuda.synchronize()
generated = sorted(out.rglob("*_generated.cif"))
if len(generated) != 1:
    raise ValueError(f"Expected one generated structure, found {len(generated)}")
design = get_structure(CIFFile.read(generated[0]), model=1, use_author_fields=False)
ca = design[design.atom_name == "CA"]
if not np.isfinite(design.coord).all():
    raise ValueError("Nonfinite generated coordinates")
chains = [{"id": str(chain), "sequence": "".join(AA3[x] for x in ca[ca.chain_id == chain].res_name)}
          for chain in dict.fromkeys(ca.chain_id.tolist())]
target_chain, binder_chain = check_generated_chains(chains, reference["target_sequence"], reference["binder_length"])
(out / "model.cif").write_bytes(generated[0].read_bytes())
write_json(out / "audit.json", {
    "status": "conditioned_generation; not a submission-eligible candidate or binding result",
    "reference_audit": {k: v for k, v in reference.items() if k != "target_sequence"},
    "reference_audit_sha256": digest(reference_dir / "audit.json"),
    "target_manifest_sha256": digest(args.manifest), "spec_sha256": digest(spec),
    "reference_sha256": digest(reference_dir / "target.cif"),
    "target_chain": target_chain["id"], "binder_chain": binder_chain["id"],
    "binder_sequence": binder_chain["sequence"], "binder_length_requested": reference["binder_length"],
    "model": model_record, "code_revision": RFD4_CODE_REVISION,
    "seed": seed, "replicates": 1, "batch_size": 1, "compilation": False,
    "external_metrics": False, "recycle_strategy": "thinking_high",
    "generated_structure": str(out / "model.cif"), "generated_sha256": digest(out / "model.cif"),
    "chains": chains, "raw_structure_hashes": {str(p.relative_to(out)): digest(p) for p in out.rglob("*.cif")},
    "generation_seconds": time.monotonic() - started,
    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
    "host_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
})
print(f"Seed {seed}: generated structure and target/binder identities validated", flush=True)
