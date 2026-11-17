"""Evaluate many candidates with ESMFold2 and Boltz-2 on Modal (Phase 3, Task 3.2).

Takes the candidates.json written by `modal_current.py --stage batch` and attaches RAW scores (ipTM, pTM, ipSAE) from both
predictors to each candidate that has no QC reasons. The engine (binder_campaign/evaluate.py) is budget-capped, resumable and
keeps failures; this file only supplies the two GPU functions and the entry point.

  modal run modal_evaluate.py --stage evaluate --candidates results/batch-b1/candidates.json --eval-id eswar-e1 \\
        --predictors esmfold2,boltz2 --chunk-size 4 --max-calls 6 --max-gpu-seconds 1800

Before the first run, in the same workspace: `modal_app.py --stage prepare` (Boltz-2 weights) and
`modal_esmfold2.py --stage prepare` (ESMFold2 weights). Both are checked at the start of each call and fail with a clear message.

Candidates are scored against the CURRENT target (UniProt 21-239) with its target-only MSA. ESMFold2 runs the team's local
approximation of the organizer screen (the organizer's exact implementation is unconfirmed). Scores are raw model outputs: not
affinities, not probabilities, not comparable across predictors, and never a basis by themselves for calling a design a binder.
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
ESM_MANIFEST = Path(os.environ.get("HELIXBRIDGE_ESM_MANIFEST", str(OPT / "docs/eswar/esmfold2-checkpoint-manifest.json")))
BOLTZ_SCRIPT = Path(os.environ.get("HELIXBRIDGE_BOLTZ_SCRIPT", str(OPT / "pipeline/scripts/boltz-predict.py")))
IPSAE = Path(os.environ.get("HELIXBRIDGE_IPSAE_DIR", str(OPT / "IPSAE")))
RESULTS = Path(os.environ.get("HELIXBRIDGE_RESULTS", str(HERE / "results")))
ESM_GPU = [g.strip() for g in os.environ.get("HELIXBRIDGE_ESM_GPU", "").split(",") if g.strip()] or ["A100-80GB", "L40S"]
BOLTZ_GPU = [g.strip() for g in os.environ.get("HELIXBRIDGE_BOLTZ_GPU", "").split(",") if g.strip()] or ["A100-80GB", "L40S", "A10G"]
IPSAE_COMMIT = "6174cf9e71cb1bd660cc805856a18c4871a6dec3"
MAX_CHUNK = 20

app = modal.App("helixbridge-evaluate")
cache = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
base = modal.Image.debian_slim(python_version="3.12").apt_install("git", "build-essential").pip_install("uv==0.12.9")
_ipsae = [f"git clone https://github.com/DunbrackLab/IPSAE.git {OPT}/IPSAE", f"git -C {OPT}/IPSAE checkout {IPSAE_COMMIT}"]
_local = lambda image: (image.add_local_dir(HERE / "pipeline", "/opt/pipeline", ignore=["data/", "__pycache__/", "*.pyc"])
                        .add_local_dir(HERE / "docs" / "eswar", "/opt/docs/eswar"))
# Same recipe as modal_esmfold2.py (esm 3.4.1.post1 needs torch 2.11.0; Triton compiles C at run time, so the headers are checked)
esm_image = _local(base.add_local_file(HERE / "docs" / "eswar" / "esm-image-constraints.txt", "/tmp/esm-constraints.txt", copy=True).run_commands(
    "uv pip install --system -c /tmp/esm-constraints.txt 'torch==2.11.0' 'esm==3.4.1.post1' 'biopython==1.88' 'biotite==1.6.0' 'huggingface-hub'",
    "python -c \"import os, sysconfig; p = os.path.join(sysconfig.get_path('include'), 'Python.h'); assert os.path.exists(p), p\"",
    "printf '#include <Python.h>\\nint main(void){return 0;}\\n' > /tmp/header-check.c && "
    "gcc -fsyntax-only -I$(python -c \"import sysconfig; print(sysconfig.get_path('include'))\") /tmp/header-check.c",
    "python -c 'from esm.models.esmfold2 import EsmFold2Model, ESMFold2InputBuilder, ProteinInput, StructurePredictionInput; "
    "from esm.utils.msa import MSA; import triton'",
    *_ipsae))
# Same recipe as the team's modal_app.py boltz_image
boltz_image = _local(base.uv_pip_install("boltz==2.2.1", "huggingface-hub", "biopython").run_commands(*_ipsae))


def _commit():
    if not os.environ.get("HELIXBRIDGE_LOCAL_TEST"):
        cache.commit()


def _short(error, limit=400):
    text = f"{type(error).__name__}: {error}".strip()
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _archive(run_id, files):
    """A tar.gz holding exactly `files` ({member name relative to run_id: Path}), nothing else."""
    import tarfile
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for name, path in files.items():
            tar.add(path, arcname=f"{run_id}/{name}")
    return stream.getvalue()


def _check_request(request, predictor):
    sys.path.insert(0, str(OPT / "pipeline"))
    from binder_campaign.modal_plan import check_run_id
    if request.get("predictor") != predictor:
        raise ValueError(f"This function evaluates with {predictor}, not {request.get('predictor')!r}")
    run_id = check_run_id(request["run_id"])
    seed = request.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 31:
        raise ValueError("seed must be a non-negative 31-bit integer")
    candidates = request["candidates"]
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= MAX_CHUNK:
        raise ValueError(f"Send between 1 and {MAX_CHUNK} candidates per call")
    ids = [c["id"] for c in candidates]
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i.replace("-", "").replace("_", "").replace(".", "").isalnum() for i in ids):
        raise ValueError("Candidate ids must be unique and plain (letters, digits, - _ .)")
    root = CACHE / "runs" / run_id
    if root.exists():
        raise ValueError("This run ID already exists; the engine uses a new one for every call")
    return run_id, seed, candidates, root


def _target():
    phase1 = OPT / "docs/eswar/phase1-data"
    target = json.loads((phase1 / "il7ra-manifest.json").read_text())["sequence"]
    msa = phase1 / "target.a3m"
    first_row = [line.strip() for line in msa.read_text().splitlines() if line and not line.startswith(">")][0]
    if first_row != target:
        raise ValueError("The MSA's query row is not the current target sequence")
    return target, msa


@app.function(image=esm_image, gpu=ESM_GPU, cpu=4, memory=65536, volumes={str(CACHE): cache}, timeout=3600, retries=0,
              max_containers=1, scaledown_window=2)
def esmfold2_evaluate(request: dict):
    """Fold every candidate with ESMFold2 (the model is loaded ONCE per call) and score it with ipSAE."""
    import time
    started_call = time.monotonic()
    run_id, seed, candidates, root = _check_request(request, "esmfold2")
    from binder_campaign.esm_eval import (REVISION, build_request, confidence_summary, load_manifest, validate_prediction,
                                          verify_checkpoint)
    from binder_campaign.scoring import build_score, run_ipsae
    backend = None if request.get("kernel_backend", "fused") in ("none", "", None) else request["kernel_backend"]
    manifest = load_manifest(ESM_MANIFEST)
    folder = CACHE / "esmfold2" / REVISION
    verify_checkpoint(folder, manifest, hash_files=False)       # sizes only; modal_esmfold2.py --stage prepare did the full check
    if not (IPSAE / "ipsae.py").is_file():
        raise ValueError("The ipSAE scorer is missing from the image")
    target, msa_path = _target()
    for c in candidates:
        build_request(target, c["sequence"], msa_path, c["id"], role="esmfold2_evaluation", seed=seed, kernel_backend=backend)

    os.environ.update(HF_HOME=str(CACHE / "hf"), HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    import numpy as np
    import torch
    from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model, ProteinInput, StructurePredictionInput
    from esm.utils.msa import MSA
    msa = MSA.from_a3m(msa_path, remove_insertions=True, max_sequences=1024)
    load_started = time.monotonic()
    model = EsmFold2Model.from_pretrained(str(folder), load_esmc=False, device="cuda").eval()
    if model.esmc is None or model.config.esmc_config is None:
        raise ValueError("The checkpoint did not supply its expected bundled ESMC backbone")
    os.environ["ESMCFOLD_CCD_PATH"] = str(folder / "ccd.pkl")      # a FILE: esm opens it as one if this is set before it is imported
    model.set_kernel_backend(backend)
    load_seconds = time.monotonic() - load_started
    builder = ESMFold2InputBuilder(ccd_cache=Path(folder))
    gpu = torch.cuda.get_device_name()
    results, files = [], {}
    for c in candidates:
        cid, binder = c["id"], c["sequence"]
        try:
            one = build_request(target, binder, msa_path, cid, role="esmfold2_evaluation", seed=seed, kernel_backend=backend)
            settings = dict(one["settings"])
            settings.pop("kernel_backend")
            spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=target, msa=msa),
                                                      ProteinInput(id="B", sequence=binder, msa=None)])
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            result = builder.fold(model, spi, **settings)
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            pae, cif = np.asarray(result.pae), result.complex.to_mmcif()
            validate_prediction(pae, cif, target, binder)
            confidence = confidence_summary(result.plddt, result.ptm, result.iptm)
            out = root / cid
            out.mkdir(parents=True)
            (out / "model.cif").write_text(cif)
            (out / "pae.json").write_text(json.dumps({"pae": pae.tolist()}))
            rows = run_ipsae(IPSAE / "ipsae.py", out / "pae.json", out / "model.cif")
            score = build_score("esmfold2", iptm=confidence["iptm"], ptm=confidence["ptm"], rows=rows, seconds=seconds,
                                peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024 ** 2,
                                plddt_raw_mean=confidence["plddt_raw_mean"], gpu=gpu,
                                settings={**one["settings"], "revision": REVISION})
            (out / "metrics.json").write_text(json.dumps({"candidate_id": cid, "score": score, "request": one}, indent=2))
            files.update({f"{cid}/{name}": out / name for name in ("model.cif", "pae.json", "metrics.json")})
            results.append({"id": cid, "status": "ok", "sequence": binder, "score": score})
        except Exception as error:                      # one bad candidate must not cost the others
            results.append({"id": cid, "status": "failed", "error": _short(error)})
    _commit()
    return {"run_id": run_id, "gpu": gpu, "load_seconds": round(load_seconds, 2), "results": results,
            "worker_seconds": time.monotonic() - started_call, "archive": _archive(run_id, files)}


@app.function(image=boltz_image, gpu=BOLTZ_GPU, cpu=4, memory=32768, volumes={str(CACHE): cache}, timeout=3600, retries=0,
              max_containers=1, scaledown_window=2)
def boltz_evaluate(request: dict):
    """Predict every candidate with Boltz-2 using the team's own scripts/boltz-predict.py, one candidate at a time."""
    import shutil
    import subprocess
    import time
    started_call = time.monotonic()
    run_id, seed, candidates, root = _check_request(request, "boltz2")
    from binder_campaign.scoring import boltz_score
    checkpoints = CACHE / "env" / "boltz-checkpoints.json"
    if not checkpoints.is_file():
        raise ValueError("Boltz-2 weights are not prepared in this workspace: run `modal run modal_app.py --stage prepare` first")
    if not (IPSAE / "ipsae.py").is_file():
        raise ValueError("The ipSAE scorer is missing from the image")
    target, msa_path = _target()
    (root / "env").mkdir(parents=True)
    shutil.copy2(checkpoints, root / "env" / "boltz-checkpoints.json")
    shutil.copytree(IPSAE, root / "src" / "IPSAE")
    os.environ.update(TASK_ROOT=str(root), HF_HOME=str(CACHE / "hf"), HF_HUB_DISABLE_IMPLICIT_TOKEN="1", WANDB_MODE="disabled")
    relative_msa = os.path.relpath(msa_path, BOLTZ_SCRIPT.resolve().parents[1])     # the team's script resolves it from pipeline/
    import torch
    gpu = torch.cuda.get_device_name()
    results, files = [], {}
    for c in candidates:
        cid, binder = c["id"], c["sequence"]
        try:
            folder = root / cid
            folder.mkdir()
            spec = {"target": {"sequence": target, "msa": relative_msa}, "partner": {"sequence": binder, "msa": None},
                    "settings": {"seed": seed}, "role": "boltz2_evaluation",
                    "construct_status": "current official interval 21-239; assay construct and processing unconfirmed"}
            (folder / "request.json").write_text(json.dumps(spec, indent=2))
            done = subprocess.run([sys.executable, str(BOLTZ_SCRIPT), str(folder / "request.json"), str(folder / "boltz")],
                                  capture_output=True, text=True, timeout=900)
            if done.returncode != 0:
                tail = (done.stderr.strip().splitlines() or ["(no error output)"])[-1]
                raise RuntimeError(f"boltz-predict.py exited {done.returncode}: {tail}")
            metrics = json.loads((folder / "boltz" / "metrics.json").read_text())
            score = boltz_score(metrics, gpu=gpu)
            for name in ("model.cif", "pae.json", "metrics.json"):
                shutil.copyfile(folder / "boltz" / name, folder / name)
            files.update({f"{cid}/{name}": folder / name for name in ("model.cif", "pae.json", "metrics.json")})
            results.append({"id": cid, "status": "ok", "sequence": binder, "score": score})
        except Exception as error:
            results.append({"id": cid, "status": "failed", "error": _short(error)})
    _commit()
    return {"run_id": run_id, "gpu": gpu, "results": results, "worker_seconds": time.monotonic() - started_call,
            "archive": _archive(run_id, files)}


@app.local_entrypoint()
def main(stage: str = "evaluate", candidates: str = "", eval_id: str = "", predictors: str = "esmfold2,boltz2",
         chunk_size: int = 4, max_calls: int = 0, max_gpu_seconds: float = 0.0, max_failures: int = 3, seed: int = 0,
         kernel_backend: str = "fused", include_flagged: bool = False, retry_failed: bool = False,
         retry_interrupted: bool = False, out: str = ""):
    """Stage `evaluate`: score the candidates with the chosen predictors, within the budget, resumably."""
    if stage != "evaluate":
        raise ValueError("The only stage is evaluate")
    sys.path.insert(0, str(HERE / "pipeline"))
    from binder_campaign.esm_eval import load_manifest
    from binder_campaign.evaluate import make_config, run_evaluation, select_candidates
    cfg = make_config(eval_id, [p.strip() for p in predictors.split(",") if p.strip()], chunk_size, max_calls=max_calls,
                      max_gpu_seconds=max_gpu_seconds, max_failures=max_failures, include_flagged=include_flagged, seed=seed,
                      kernel_backend=kernel_backend)
    path = Path(candidates)
    chosen, skipped = select_candidates(json.loads(path.read_text())["candidates"], include_flagged)   # free local checks first
    if not chosen:
        raise ValueError("No candidates to evaluate")
    load_manifest(HERE / "docs/eswar/esmfold2-checkpoint-manifest.json")
    folder = Path(out) if out else RESULTS / f"eval-{eval_id}"
    print(json.dumps({"evaluate": cfg, "candidates": len(chosen), "skipped": skipped, "output": str(folder),
                      "gpu_preference": {"esmfold2": ESM_GPU, "boltz2": BOLTZ_GPU}}, indent=2))
    functions = {"esmfold2": esmfold2_evaluate, "boltz2": boltz_evaluate}
    report = run_evaluation(cfg, path, lambda request: functions[request["predictor"]].remote(request), folder,
                            retry_failed=retry_failed, retry_interrupted=retry_interrupted)
    print(json.dumps({k: report[k] for k in ("stop_reason", "candidates_evaluated", "by_predictor", "failures", "interrupted",
                                             "calls", "gpu_seconds_reported", "gpu_seconds_note")}, indent=2))
