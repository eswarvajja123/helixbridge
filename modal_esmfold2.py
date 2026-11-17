"""Qualify ESMFold2 on Modal (Phase 3, Task 3.1): can it load, fold one known complex, and report what it needs?

Two steps, each its own Modal function:
  prepare  (CPU)  download the pinned checkpoint (biohub/ESMFold2 at one fixed revision, ~24.9 GiB) into the shared model
                  Volume and check every file against docs/eswar/esmfold2-checkpoint-manifest.json (sizes and SHA-256)
  qualify  (GPU)  fold the verified 60-residue RFD4 design against the CURRENT IL-7Ra target (21-239) with its target-only
                  MSA, using the team's local approximation of the organizer screen, and validate the output the way
                  the team's own script does

  modal run modal_esmfold2.py --stage prepare
  modal run modal_esmfold2.py --stage qualify --run-id eswar-esm-001

This only shows the software runs on a Modal GPU and how much memory and time it takes. It says nothing about how
accurate ESMFold2 is, and no result is a binding prediction. Same cache Volume as modal_app.py / modal_current.py.
"""
from pathlib import Path
import io
import json
import os
import sys

import modal

HERE = Path(__file__).resolve().parent
VOLUME_NAME = "helixbridge-model-cache"
# Where things live inside the container; the environment variables exist only so tests can run the bodies locally.
OPT = Path(os.environ.get("HELIXBRIDGE_OPT", "/opt"))
CACHE = Path(os.environ.get("HELIXBRIDGE_CACHE", "/cache"))
MANIFEST = Path(os.environ.get("HELIXBRIDGE_ESM_MANIFEST", str(OPT / "docs/eswar/esmfold2-checkpoint-manifest.json")))
RESULTS = Path(os.environ.get("HELIXBRIDGE_RESULTS", str(HERE / "results")))
# ESMFold2 reached about 26 GiB on the team's 46 GB card, so a 24 GB A10G is left out on purpose.
GPU = [g.strip() for g in os.environ.get("HELIXBRIDGE_GPU", "").split(",") if g.strip()] or ["A100-80GB", "L40S"]

app = modal.App("helixbridge-esmfold2")
cache = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
base = modal.Image.debian_slim(python_version="3.12").apt_install("git", "build-essential").pip_install("uv==0.12.9")
image = (
    base.add_local_file(HERE / "docs" / "eswar" / "esm-image-constraints.txt", "/tmp/esm-constraints.txt", copy=True).run_commands(
        # esm 3.4.1.post1 needs torch >= 2.11.0 and < 2.12.0; 2.11.0 is the only release in that range (CUDA 13.0 libraries)
        "uv pip install --system -c /tmp/esm-constraints.txt 'torch==2.11.0' 'esm==3.4.1.post1' 'biopython==1.88' 'biotite==1.6.0' 'huggingface-hub'",
        # The fused kernels are compiled by Triton at run time. The team's first cluster run failed on a missing Python.h,
        # so the build itself proves the headers exist and a C compiler can use them.
        "python -c \"import os, sysconfig; p = os.path.join(sysconfig.get_path('include'), 'Python.h'); assert os.path.exists(p), p\"",
        "printf '#include <Python.h>\\nint main(void){return 0;}\\n' > /tmp/header-check.c && "
        "gcc -fsyntax-only -I$(python -c \"import sysconfig; print(sysconfig.get_path('include'))\") /tmp/header-check.c",
        "python -c 'from esm.models.esmfold2 import EsmFold2Model, ESMFold2InputBuilder, ProteinInput, StructurePredictionInput; "
        "from esm.utils.msa import MSA; import triton'",
    )
    .add_local_dir(HERE / "pipeline", "/opt/pipeline", ignore=["data/", "__pycache__/", "*.pyc"])
    .add_local_dir(HERE / "examples", "/opt/examples")
    .add_local_dir(HERE / "docs" / "eswar", "/opt/docs/eswar")
)


def _commit():
    if not os.environ.get("HELIXBRIDGE_LOCAL_TEST"):
        cache.commit()


def _version(package):
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(package)
    except PackageNotFoundError:
        return "unavailable"


def _archive(directory):
    import tarfile
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        tar.add(directory, arcname=directory.name)
    return stream.getvalue()


@app.function(image=image, cpu=4, memory=16384, volumes={str(CACHE): cache}, timeout=3600, retries=0, max_containers=1,
              scaledown_window=2)
def prepare_esmfold2(verify_hashes: bool = True):
    """Download the pinned checkpoint into the shared Volume and verify it against the committed manifest."""
    import time
    sys.path.insert(0, str(OPT / "pipeline"))
    from binder_campaign.esm_eval import REPO, REVISION, checkpoint_record, load_manifest, verify_checkpoint
    started = time.monotonic()
    manifest = load_manifest(MANIFEST)
    folder = CACHE / "esmfold2" / REVISION
    folder.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HOME=str(CACHE / "hf"), HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=REPO, revision=REVISION, local_dir=str(folder), allow_patterns=manifest["allow_patterns"],
                      max_workers=4, token=False)
    report = verify_checkpoint(folder, manifest, hash_files=verify_hashes)      # raises if anything differs
    (CACHE / "env").mkdir(parents=True, exist_ok=True)
    (CACHE / "env" / "esmfold2-checkpoints.json").write_text(json.dumps(checkpoint_record(folder, manifest), indent=2) + "\n")
    _commit()
    return {"status": "checkpoint verified", "repo": REPO, "revision": REVISION, "path": str(folder), **report,
            "seconds": round(time.monotonic() - started, 1)}


@app.function(image=image, gpu=GPU, cpu=4, memory=65536, volumes={str(CACHE): cache}, timeout=1800, retries=0,
              max_containers=1, scaledown_window=2)
def qualify_esmfold2(run_id: str, seed: int = 0, kernel_backend: str = "fused"):
    """Fold the verified design against the current target on one GPU and report what it took."""
    import time
    sys.path.insert(0, str(OPT / "pipeline"))
    from binder_campaign.esm_eval import (REVISION, build_request, checkpoint_record, confidence_summary, load_manifest,
                                          validate_prediction, verify_checkpoint)
    from binder_campaign.modal_plan import check_run_id
    run_id = check_run_id(run_id)
    backend = None if str(kernel_backend).lower() in ("none", "") else kernel_backend
    root = CACHE / "runs" / run_id
    if root.exists():
        raise ValueError("This run ID already exists; use a new --run-id")
    manifest = load_manifest(MANIFEST)
    folder = CACHE / "esmfold2" / REVISION
    verify_checkpoint(folder, manifest, hash_files=False)          # sizes only here; prepare did the full hash check

    phase1 = OPT / "docs/eswar/phase1-data"
    target = json.loads((phase1 / "il7ra-manifest.json").read_text())["sequence"]
    binder = json.loads((OPT / "examples/verified-modal-run/summary.json").read_text())["binder_sequence"]
    msa_path = phase1 / "target.a3m"
    first_row = [line.strip() for line in msa_path.read_text().splitlines() if line and not line.startswith(">")][0]
    if first_row != target:
        raise ValueError("The MSA's query row is not the current target sequence")
    request = build_request(target, binder, msa_path, "esmfold2-qualification-" + run_id, role="esmfold2_qualification",
                            seed=seed, kernel_backend=backend)

    os.environ.update(HF_HOME=str(CACHE / "hf"), HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    import numpy as np
    import torch
    from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model, ProteinInput, StructurePredictionInput
    from esm.utils.msa import MSA
    msa = MSA.from_a3m(msa_path, remove_insertions=True, max_sequences=request["settings"]["msa_max_depth"])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = EsmFold2Model.from_pretrained(str(folder), load_esmc=False, device="cuda").eval()
    if model.esmc is None or model.config.esmc_config is None:
        raise ValueError("The checkpoint did not supply its expected bundled ESMC backbone")
    os.environ["ESMCFOLD_CCD_PATH"] = str(folder / "ccd.pkl")      # a FILE: esm opens it as one if this is set before it is imported
    settings = dict(request["settings"])
    settings.pop("kernel_backend")
    model.set_kernel_backend(backend)
    loaded = time.monotonic()
    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=target, msa=msa),
                                              ProteinInput(id="B", sequence=binder, msa=None)])
    result = ESMFold2InputBuilder(ccd_cache=Path(folder)).fold(model, spi, **settings)
    torch.cuda.synchronize()
    folded = time.monotonic()

    pae = np.asarray(result.pae)
    cif = result.complex.to_mmcif()
    checks = validate_prediction(pae, cif, target, binder)          # raises if the output is not what we asked for
    confidence = confidence_summary(result.plddt, result.ptm, result.iptm)
    out = root / "esmfold2"
    out.mkdir(parents=True)
    (out / "model.cif").write_text(cif)
    (out / "pae.json").write_text(json.dumps({"pae": pae.tolist()}))
    (out / "request.json").write_text(json.dumps(request, indent=2))
    metrics = {
        "status": "ESMFold2 qualification on Modal; software check only, not accuracy, not a binding result",
        "run_id": run_id, "repo": manifest["repo"], "revision": manifest["revision"], "checks": checks, **confidence,
        "settings": request["settings"], "kernel_backend": backend, "gpu": torch.cuda.get_device_name(),
        "load_seconds": round(loaded - started, 2), "fold_seconds": round(folded - loaded, 2),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024 ** 2,
        "versions": {"torch": getattr(torch, "__version__", "unavailable"), "esm": _version("esm"), "triton": _version("triton")},
        "target_residues": len(target), "binder_residues": len(binder), "msa_rows_requested": request["settings"]["msa_max_depth"],
        "protocol_status": request["protocol_status"]}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _commit()
    return {"run_id": run_id, "metrics": metrics, "archive": _archive(root)}


@app.local_entrypoint()
def main(stage: str = "prepare", run_id: str = "", seed: int = 0, kernel_backend: str = "fused", skip_hash: bool = False):
    """Stages: prepare (download and verify the checkpoint) or qualify (one fold on a GPU)."""
    if stage not in {"prepare", "qualify"}:
        raise ValueError("Choose prepare or qualify")
    sys.path.insert(0, str(HERE / "pipeline"))
    if stage == "prepare":
        print(json.dumps(prepare_esmfold2.remote(not skip_hash), indent=2))
        return
    from binder_campaign.esm_eval import KERNEL_BACKENDS, build_request, load_manifest
    from binder_campaign.modal_plan import check_run_id
    check_run_id(run_id)
    backend = None if str(kernel_backend).lower() in ("none", "") else kernel_backend
    if backend not in KERNEL_BACKENDS:
        raise ValueError("--kernel-backend must be fused, cuequivariance or none")
    load_manifest(HERE / "docs/eswar/esmfold2-checkpoint-manifest.json")          # free local checks before any GPU spending
    phase1 = HERE / "docs/eswar/phase1-data"
    build_request(json.loads((phase1 / "il7ra-manifest.json").read_text())["sequence"],
                  json.loads((HERE / "examples/verified-modal-run/summary.json").read_text())["binder_sequence"],
                  phase1 / "target.a3m", "check", role="check", seed=seed, kernel_backend=backend)
    folder = RESULTS / run_id
    archive_path = folder / "esmfold2-qualify.tar.gz"
    if archive_path.exists():
        raise ValueError("Refusing to overwrite an existing local result")
    folder.mkdir(parents=True, exist_ok=True)
    result = qualify_esmfold2.remote(run_id, seed, kernel_backend)
    archive_path.write_bytes(result.pop("archive"))
    result["modal_profile"] = os.environ.get("MODAL_PROFILE", "active CLI profile")
    (folder / "esmfold2-qualify.json").write_text(json.dumps(result, indent=2) + "\n")
    m = result["metrics"]
    print(json.dumps({"saved": str(archive_path), "gpu": m["gpu"], "load_seconds": m["load_seconds"], "fold_seconds": m["fold_seconds"],
                      "peak_reserved_mib": m["peak_reserved_mib"], "ptm": m["ptm"], "iptm": m["iptm"],
                      "note": "software check only; not an accuracy or binding result"}, indent=2))
