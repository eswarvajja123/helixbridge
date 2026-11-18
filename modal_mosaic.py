"""Mosaic (binder design by optimizing through Boltz-2) on Modal for the CURRENT IL-7Ra target (Phase 4, Tasks 4.1 and 4.2).

A second design method to compare with RFD4. Same model-cache Volume as the other Modal files (the Boltz-2 weights come from
`modal run modal_app.py --stage prepare`). The three pinned sources (mosaic, joltz, the Boltz fork) are cloned at exact commits inside
the image and refused if modified; the optimizer settings are the team's, unchanged (binder_campaign/mosaic_input.py).

  python pipeline/scripts/plan-modal-run.py ...                 # not needed here: `--stage plan` below makes the same free local checks
  modal run modal_mosaic.py --stage plan --option A --length 60 --seed 101
  modal run modal_mosaic.py --stage smoke --run-id eswar-mos-smoke                     # one loss/gradient evaluation, no design
  modal run modal_mosaic.py --stage generate --run-id eswar-mos-001 --option A --length 60 --seed 101
  modal run modal_mosaic.py --stage batch --batch-id eswar-mA --seeds 101,102 --lengths 60 --option A --max-designs 4 --max-gpu-seconds 4000

One design takes about 16 minutes on an L40S (the team's measurement), so the function timeout is one hour (Task 4.2); the batch caps
count the function's own seconds. Nothing here is a submission entry or a binding result. Boltz-2 is inside Mosaic's objective, so
Boltz-2 agreement on a Mosaic design is not independent evidence; ESMFold2 is the check.
"""
from pathlib import Path
import json
import os
import sys

import modal

HERE = Path(__file__).resolve().parent
VOLUME_NAME = "helixbridge-model-cache"
OPT = Path(os.environ.get("HELIXBRIDGE_OPT", "/opt"))
CACHE = Path(os.environ.get("HELIXBRIDGE_CACHE", "/cache"))
SRC = Path(os.environ.get("HELIXBRIDGE_MOSAIC_SRC", "/opt/src"))
GPU = [g.strip() for g in os.environ.get("HELIXBRIDGE_GPU", "").split(",") if g.strip()] or ["A100-80GB", "L40S"]
TIMEOUT_SECONDS = 3600                                   # Task 4.2: one design is about 16 minutes, the old limit was 15
SCRIPT_TIMEOUT = TIMEOUT_SECONDS - 300
CONSTRAINTS = HERE / "docs" / "eswar" / "mosaic-image-constraints.txt"

# the pinned sources (kept here as plain data so the image build does not depend on importing the pipeline)
REPOS = {"mosaic-runtime": ("https://github.com/escalante-bio/mosaic.git", "5937f37164bcb28bc6f1000f10134145438e5971"),
         "joltz": ("https://github.com/nboyd/joltz.git", "ed0f04257dac85bd4b7bf45521cc280f86d38ded"),
         "mosaic-boltz": ("https://github.com/escalante-bio/boltz.git", "7ea718a1440ecbe8e2418317192b501544147edd")}

app = modal.App("helixbridge-mosaic")
cache = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
base = modal.Image.debian_slim(python_version="3.12").apt_install("git", "build-essential").pip_install("uv==0.12.9")
_clone = [f"git clone {url} /opt/src/{name} && git -C /opt/src/{name} checkout {rev} && "
          f"test \"$(git -C /opt/src/{name} rev-parse HEAD)\" = {rev}" for name, (url, rev) in REPOS.items()]
_constraint = "-c /tmp/mosaic-constraints.txt " if CONSTRAINTS.is_file() else ""
_with_constraints = base.add_local_file(CONSTRAINTS, "/tmp/mosaic-constraints.txt", copy=True) if CONSTRAINTS.is_file() else base
image = (
    _with_constraints.run_commands(
        "mkdir -p /opt/src", *_clone,
        # the team's scoped install (install-mosaic.sbatch): CPU torch, JAX with CUDA 12, the two JAX/Boltz sources editable, mosaic with no deps
        f"uv pip install --system {_constraint}--index-url https://download.pytorch.org/whl/cpu 'torch==2.11.0'",
        f"uv pip install --system {_constraint}--no-sources 'jax[cuda12]==0.11.1' 'equinox>=0.13.0' 'dm-haiku>=0.0.13' 'optax>=0.2.4' "
        "'ml-collections>=1.0.0' -e /opt/src/mosaic-boltz -e /opt/src/joltz",
        f"uv pip install --system {_constraint}--no-deps -e /opt/src/mosaic-runtime",
        "python -c 'from mosaic.models.boltz2 import Boltz2; from mosaic.proteinmpnn.mpnn import load_mpnn_sol; "
        "from mosaic.optimizers import simplex_APGM; print(\"Scoped Boltz2/ProteinMPNN imports passed\")'",
    )
    .add_local_dir(HERE / "pipeline", "/opt/pipeline", ignore=["data/", "__pycache__/", "*.pyc"])
    .add_local_dir(HERE / "docs" / "eswar", "/opt/docs/eswar")
)


def _commit():
    if not os.environ.get("HELIXBRIDGE_LOCAL_TEST"):
        cache.commit()


def _freeze():
    import subprocess
    try:
        return subprocess.check_output(["uv", "pip", "freeze", "--system"], text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        return f"package list unavailable: {error}"


def _archive(directory):
    import io
    import tarfile
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        tar.add(directory, arcname=directory.name)
    return stream.getvalue()


def _source_state():
    import subprocess
    found = {}
    for name in REPOS:
        path = SRC / name
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True)
        found[name] = {"head": head, "dirty": bool(dirty.strip())}
    return found


@app.function(image=image, gpu=GPU, cpu=4, memory=65536, volumes={str(CACHE): cache}, timeout=TIMEOUT_SECONDS, retries=0,
              max_containers=1, scaledown_window=2)
def mosaic_run(run_id: str, seed: int, hotspots: list, hotspot_source: str, binder_length: list, mode: str = "design"):
    """One Mosaic smoke test or design. `binder_length` is [low, high] and must be one fixed length (Mosaic needs a length)."""
    import shutil
    import subprocess
    import time
    sys.path.insert(0, str(OPT / "pipeline"))
    from binder_campaign.generation_input import check_seed
    from binder_campaign.modal_plan import check_run_id
    from binder_campaign.mosaic_input import REVISIONS, build_audit, build_inputs, check_revisions
    from binder_campaign.runner import digest

    started = time.monotonic()
    run_id, seed = check_run_id(run_id), check_seed(seed)
    if mode not in ("smoke", "design"):
        raise ValueError("mode must be smoke or design")
    low, high = binder_length
    if low != high:
        raise ValueError("Mosaic needs one fixed binder length, not a range")
    check_revisions(_source_state())
    phase1 = OPT / "docs/eswar/phase1-data"
    manifest_path = phase1 / "il7ra-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    inputs = build_inputs(manifest, phase1 / "target.a3m", hotspots, low)

    checkpoints = json.loads((CACHE / "env/boltz-checkpoints.json").read_text())
    boltz_cache = Path(checkpoints["cache"])
    checkpoint = boltz_cache / "boltz2_conf.ckpt"
    if digest(checkpoint) != checkpoints["files"]["boltz2_conf.ckpt"]["sha256"]:
        raise ValueError("Boltz-2 checkpoint hash mismatch")

    root = CACHE / "runs" / run_id
    if (root / "generation").exists():
        raise ValueError("This run ID already has output; use a new --run-id")
    generation = root / "generation"
    generation.mkdir(parents=True)
    msa_path = generation / "target-msa-128.a3m"
    msa_path.write_text(inputs["msa_a3m"])
    spec = {k: inputs[k] for k in ("target_sequence", "epitope_idx", "binder_length", "settings")}
    spec.update(seed=seed, msa_path=str(msa_path), boltz_cache=str(boltz_cache))
    (generation / "inputs.json").write_text(json.dumps(spec, indent=2))
    os.environ.update(TASK_ROOT=str(root), HF_HOME=str(CACHE / "hf"), HF_HUB_DISABLE_IMPLICIT_TOKEN="1", WANDB_MODE="disabled")
    work = root / "work"
    subprocess.run([sys.executable, str(OPT / "pipeline/scripts/mosaic-current.py"), "--inputs", str(generation / "inputs.json"),
                    "--mode", mode, "--out", str(work)], check=True, timeout=SCRIPT_TIMEOUT)
    result = json.loads((work / "result.json").read_text())
    freeze = _freeze()
    (generation / "environment.txt").write_text(freeze)
    import hashlib
    gpu_name = result["device_kind"]
    if mode == "smoke":
        _commit()
        return {"run_id": run_id, "mode": "smoke", "result": result, "worker_seconds": time.monotonic() - started, "gpu": gpu_name,
                "revisions": REVISIONS, "environment_sha256": hashlib.sha256(freeze.encode()).hexdigest(), "environment": freeze}
    shutil.copy2(work / "model.cif", generation / "model.cif")
    shutil.copy2(work / "raw.npz", generation / "raw.npz")
    (generation / "result.json").write_text(json.dumps(result, indent=2))
    audit = build_audit(run_id=run_id, seed=seed, hotspots=hotspots, hotspot_source=hotspot_source, binder_length=low,
                        manifest_path=manifest_path, structure_path=generation / "model.cif",
                        extras={"generation_seconds": result["timings"].get("optimize_seconds", 0.0) + result["timings"].get("predict_seconds", 0.0),
                                "timings": result["timings"], "memory_stats": result["memory_stats"], "device": gpu_name,
                                "generator_model_iptm": result["generator_model_iptm"], "environment_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
                                "jax_version": result["jax_version"], "boltz_checkpoint_sha256": checkpoints["files"]["boltz2_conf.ckpt"]["sha256"],
                                "target_msa_rows_used": inputs["msa_rows"], "epitope_idx": inputs["epitope_idx"],
                                "host_peak_rss_mib": result["host_peak_rss_mib"]})
    (generation / "audit.json").write_text(json.dumps(audit, indent=2))
    _commit()
    return {"run_id": run_id, "mode": "design", "audit": audit, "worker_seconds": time.monotonic() - started, "gpu": gpu_name,
            "archive": _archive(root)}


@app.local_entrypoint()
def main(stage: str = "plan", run_id: str = "", seed: int = 101, option: str = "", hotspots: str = "", hotspot_source: str = "",
         length: int = 60, batch_id: str = "", seeds: str = "", lengths: str = "", max_designs: int = 0, max_gpu_seconds: float = 0.0,
         max_failures: int = 2, retry_failed: bool = False, retry_interrupted: bool = False, out: str = ""):
    """Stages: plan (free local checks), smoke (one GPU loss/gradient check), generate (one design) or batch (budget-capped, resumable)."""
    if stage not in {"plan", "smoke", "generate", "batch"}:
        raise ValueError("Choose plan, smoke, generate or batch")
    sys.path.insert(0, str(HERE / "pipeline"))
    from binder_campaign.generation_input import check_seed
    from binder_campaign.modal_plan import check_run_id, resolve_hotspots
    from binder_campaign.mosaic_input import build_inputs
    manifest_path = HERE / "docs/eswar/phase1-data/il7ra-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    msa_path = HERE / "docs/eswar/phase1-data/target.a3m"
    hotspot_list, source = resolve_hotspots(HERE, option, hotspots, hotspot_source)

    def check(binder_length):
        return build_inputs(manifest, msa_path, hotspot_list, binder_length)       # the same input checks the GPU function makes

    if stage == "batch":
        from binder_campaign.batch import make_config, parse_lengths, parse_seeds, run_batch
        cfg = make_config(batch_id, parse_seeds(seeds), parse_lengths(lengths), hotspot_list, source, max_designs=max_designs,
                          max_gpu_seconds=max_gpu_seconds, max_failures=max_failures)
        for low, high in cfg["lengths"]:
            if low != high:
                raise ValueError("Mosaic needs fixed binder lengths, not ranges")
            check(low)
        folder = Path(out) if out else HERE / "results" / f"batch-{batch_id}"
        print(json.dumps({"batch": cfg, "input_checks": "passed", "gpu_preference": GPU, "function_timeout_seconds": TIMEOUT_SECONDS,
                          "output": str(folder)}, indent=2))
        report = run_batch(cfg, lambda t: mosaic_run.remote(t["run_id"], t["seed"], t["hotspots"], t["hotspot_source"], t["binder_length"]),
                           folder, manifest_path, retry_failed=retry_failed, retry_interrupted=retry_interrupted)
        print(json.dumps({k: report[k] for k in ("stop_reason", "planned", "done", "kept", "dropped_by_dedup", "failed", "interrupted",
                                                 "not_started", "gpu_seconds_reported", "gpu_seconds_note")}, indent=2))
        return
    checked = check(length)
    plan = {"stage": stage, "seed": check_seed(seed), "binder_length": length, "hotspots": hotspot_list, "hotspot_source": source,
            "epitope_idx": checked["epitope_idx"], "msa_rows": checked["msa_rows"], "settings": checked["settings"]}
    print(json.dumps({"plan": plan, "input_checks": "passed", "gpu_preference": GPU, "function_timeout_seconds": TIMEOUT_SECONDS}, indent=2))
    if stage == "plan":
        return
    rid = check_run_id(run_id) if run_id else ""
    if not rid:
        raise ValueError("Supply a fresh --run-id")
    output = HERE / "results" / rid
    name = "mosaic-smoke" if stage == "smoke" else "mosaic-generate"
    if (output / f"{name}.json").exists():
        raise ValueError("Refusing to overwrite an existing local result")
    output.mkdir(parents=True, exist_ok=True)
    result = mosaic_run.remote(rid, plan["seed"], hotspot_list, source, [length, length], "smoke" if stage == "smoke" else "design")
    if "archive" in result:
        (output / f"{name}.tar.gz").write_bytes(result.pop("archive"))
    result["modal_profile"] = os.environ.get("MODAL_PROFILE", "active CLI profile")
    (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = {"saved": str(output / f"{name}.json"), "gpu": result["gpu"], "worker_seconds": result["worker_seconds"]}
    if stage == "smoke":
        summary.update(loss=result["result"]["loss"], gradient_norm=result["result"]["gradient_norm"])
    else:
        summary.update(binder_sequence=result["audit"]["binder_sequence"], model_iptm=result["audit"]["generator_model_iptm"],
                       note="a design by optimization through Boltz-2; not a binding result")
    print(json.dumps(summary, indent=2))
