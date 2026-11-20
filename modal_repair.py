"""Native SolubleMPNN on Modal for the optional sequence repair (Phase 5, Task 5.2), run in a container, not on your machine.

Runs the UNMODIFIED upstream code (github.com/dauparas/LigandMPNN, pinned commit) with the original solublempnn_v_48_020 weights (hash
checked at image build, and checked to load as plain tensors with torch.load(weights_only=True)) on a CPU container, through the tested
wrapper binder_campaign.repair.run_native with the team's lead101 settings (temperature 0.1, seed 401, 16 samples). It only writes the
native outputs; choosing and recording variants is `pipeline/scripts/repair-liabilities.py ingest`.

  modal run modal_repair.py --records results/liabilities-x/liabilities-candidates.json --selection results/selection-x/selection.json \
        --out results/repair-native-1
  python pipeline/scripts/repair-liabilities.py ingest --records ... --selection ... --native-dir results/repair-native-1/native --out results/repair-1

Third-party code and weights run only inside the Modal container. Nothing here marks a design as better, reviewed or eligible.
"""
from pathlib import Path
import json
import os
import sys

import modal

HERE = Path(__file__).resolve().parent
OPT = Path(os.environ.get("HELIXBRIDGE_OPT", "/opt"))
MPNN_CODE = Path(os.environ.get("HELIXBRIDGE_MPNN_CODE", "/opt/LigandMPNN"))
MPNN_WEIGHTS = Path(os.environ.get("HELIXBRIDGE_MPNN_WEIGHTS", "/opt/weights/solublempnn_v_48_020.pt"))
REVISION = "26ec57ac976ade5379920dbd43c7f97a91cf82de"
WEIGHTS_URL = "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt"
WEIGHTS_SHA256 = "7af52d090172c230c7f0e9d21e02203f6b3a38b16db58d3c7a3960e0a9a6e31a"      # as downloaded on 2026-11-17; recorded in docs/eswar/gpu-runs.md

app = modal.App("helixbridge-repair")
base = modal.Image.debian_slim(python_version="3.11").apt_install("git", "curl", "build-essential").pip_install("uv==0.12.9")
image = (
    base.run_commands(
        f"git clone https://github.com/dauparas/LigandMPNN.git /opt/LigandMPNN && git -C /opt/LigandMPNN checkout {REVISION} && "
        f"test \"$(git -C /opt/LigandMPNN rev-parse HEAD)\" = {REVISION}",
        # the upstream's own pinned requirements for the CPU path (its requirements.txt lists CUDA wheels we do not need here)
        "uv pip install --system --index-url https://download.pytorch.org/whl/cpu 'torch==2.2.1'",
        "uv pip install --system 'numpy==1.23.5' 'ProDy==2.4.1' 'biopython==1.79' 'scipy==1.12.0' 'ml-collections==0.1.1' 'dm-tree==0.1.8' "
        "'pyparsing==3.1.1' 'filelock==3.13.1' 'typing_extensions==4.10.0'",
        f"mkdir -p /opt/weights && curl -sSfL -o /opt/weights/solublempnn_v_48_020.pt {WEIGHTS_URL} && "
        f"echo '{WEIGHTS_SHA256}  /opt/weights/solublempnn_v_48_020.pt' | sha256sum -c -",
        # the weights must load in torch's safe mode: that proves the file holds plain tensors and no executable objects
        "python -c \"import torch; d = torch.load('/opt/weights/solublempnn_v_48_020.pt', map_location='cpu', weights_only=True); "
        "print('weights load safely:', sorted(d)[:4])\"",
    )
    .add_local_dir(HERE / "pipeline", "/opt/pipeline", ignore=["data/", "__pycache__/", "*.pyc"])
)


@app.function(image=image, cpu=4, memory=8192, timeout=1800, retries=0, max_containers=2, scaledown_window=2)
def native_repair(parent_id: str, pdb_text: str, edits: dict, binder_chain: str = "B", temperature: float = 0.1, seed: int = 401):
    """Run native SolubleMPNN on one design's backbone. `edits` maps a 1-based binder position to the residue letters to forbid there."""
    import subprocess
    import tempfile
    import time
    sys.path.insert(0, str(OPT / "pipeline"))
    from binder_campaign import repair
    edits = {int(k): str(v) for k, v in edits.items()}
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 < temperature <= 1.0:
        raise ValueError("temperature must be above 0 and at most 1.0")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a whole number, 0 or more")
    if not edits or any(not v or set(v) - set("ACDEFGHIKLMNPQRSTVWY") for v in edits.values()):
        raise ValueError("Give at least one position and the residue letters to forbid there")
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "design.pdb").write_text(pdb_text)
        fasta, command = repair.run_native(tmp / "design.pdb", edits, tmp / "native", MPNN_CODE, MPNN_WEIGHTS, sys.executable, binder_chain, seed=seed,
                                        temperature=temperature)
        versions = subprocess.run([sys.executable, "-c", "import torch, numpy, prody; print(torch.__version__, numpy.__version__, prody.__version__)"],
                                  capture_output=True, text=True).stdout.strip()
        return {"parent": parent_id, "fasta_name": fasta.name, "fasta": fasta.read_text(), "command": [str(c) for c in command],
                "log_tail": (tmp / "native" / "run.log").read_text()[-2000:], "versions": versions, "revision": REVISION,
                "weights_sha256": WEIGHTS_SHA256, "seconds": time.monotonic() - started}


@app.local_entrypoint()
def main(records: str, selection: str = "", out: str = "", temperature: float = 0.1, seed: int = 401, only_types: str = ""):
    """Run native SolubleMPNN for every chosen design that has a liability; writes native outputs for repair-liabilities.py ingest."""
    sys.path.insert(0, str(HERE / "pipeline"))
    from binder_campaign import repair
    folder = Path(out)
    if not out or folder.exists():
        raise ValueError("Give a fresh --out folder")
    doc = json.loads(Path(records).read_text())["candidates"]
    wanted = {s["candidate_id"] for s in json.loads(Path(selection).read_text())["selected"]} if selection else None
    jobs = []
    for parent in doc:
        if (wanted is not None and parent["candidate_id"] not in wanted) or parent.get("parent_candidate_id"):
            continue
        edits, types = repair.liability_edits(parent["sequence"], [t for t in only_types.split(",") if t] or None)
        if edits:
            pdb, chains = repair.cif_to_pdb(parent["generator_structure"])
            jobs.append((parent, edits, types, pdb, chains))
    folder.mkdir(parents=True)
    summary = []
    for parent, edits, types, pdb, chains in jobs:
        try:
            result = native_repair.remote(parent["candidate_id"], pdb, {str(k): v for k, v in edits.items()}, parent["generator_binder_chain"], temperature, seed)
        except Exception as error:                       # one failed design must not hide the others
            summary.append({"candidate_id": parent["candidate_id"], "status": "failed", "error": f"{type(error).__name__}: {error}"[:300]})
            continue
        target = folder / "native" / parent["candidate_id"] / "seqs"
        target.mkdir(parents=True)
        (target / result["fasta_name"]).write_text(result["fasta"])
        (folder / "native" / parent["candidate_id"] / "run.json").write_text(json.dumps({k: v for k, v in result.items() if k != "fasta"}, indent=2))
        summary.append({"candidate_id": parent["candidate_id"], "status": "ran", "types": types, "positions": edits, "seconds": round(result["seconds"], 1)})
    (folder / "native-run.json").write_text(json.dumps({"revision": REVISION, "weights_sha256": WEIGHTS_SHA256, "temperature": temperature, "seed": seed, "only_types": only_types or "all", "designs": summary}, indent=2) + "\n")
    print(json.dumps({"output_folder": str(folder), "designs_with_liabilities": len(jobs), "results": summary}, indent=2))
