"""One on-target RFD4 generation on Modal for the current IL-7Ra target (Phase 2, Task 2.4).

Separate from the team's modal_app.py (which stays the historical rehearsal). Same pinned RFD4 image steps and the same
model-cache Volume, so `modal_app.py --stage prepare` only has to be run once per workspace. Runs the checked
pipeline/scripts/rfd4-generate.py on conditioning built from a hotspot option, then records the design as a candidate.
No candidate here is a submission entry or a binding result.

  python pipeline/scripts/plan-modal-run.py --option A --seed 43            # free local check first
  modal run modal_current.py --stage generate --run-id eswar-cur-001 --seed 43 --option A

Stages: generate (one GPU run, one design), batch (several designs, budget-capped and resumable) and plan (the same local checks; but `modal run` starts the Modal app, which can
trigger an image build, so use plan-modal-run.py for a fully free check). Evaluation on the current target is Phase 3.
"""
from pathlib import Path
import json
import os
import re
import sys

import modal

HERE = Path(__file__).resolve().parent
RFD_REV = "ddd934f7e6b86f7582ce487d5b19f6c526c644b5"          # must match modal_app.py (a test checks)
VOLUME_NAME = "helixbridge-model-cache"                         # same Volume as modal_app.py: weights are shared
# Where things live inside the container; the environment variables exist only so tests can run the body locally.
OPT = Path(os.environ.get("HELIXBRIDGE_OPT", "/opt"))
CACHE = Path(os.environ.get("HELIXBRIDGE_CACHE", "/cache"))
GPU = [g.strip() for g in os.environ.get("HELIXBRIDGE_GPU", "").split(",") if g.strip()] or ["A100-80GB", "L40S", "A10G"]

app = modal.App("helixbridge-current")
cache = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
base = modal.Image.debian_slim(python_version="3.12").apt_install("git", "build-essential").pip_install("uv==0.12.9")
image = (
    base.run_commands(
        "git clone https://github.com/RosettaCommons/RFD4-Proteina.git /opt/RFD4-Proteina",
        f"git -C /opt/RFD4-Proteina checkout {RFD_REV}",
        "uv pip install --system --no-sources --torch-backend cu132 'torch==2.14.1' 'atomworks[ml,openbabel]==3.0.4' 'biotite==1.6.0' -e /opt/RFD4-Proteina",
    ).apt_install("libxrender1", "libxext6", "libsm6", "libglib2.0-0").run_commands(
        "python -c 'from rfproteina.generate import run_generation'",
    )
    # binder_campaign.generation_input reads CIF files with Biopython; the historical RFD4 image does not install it
    .uv_pip_install("biopython==1.88")
    .add_local_dir(HERE / "pipeline", "/opt/pipeline", ignore=["data/", "__pycache__/", "*.pyc"])
    .add_local_dir(HERE / "examples", "/opt/examples")
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


@app.function(image=image, gpu=GPU, cpu=4, memory=32768, volumes={str(CACHE): cache}, timeout=900, retries=0,
              max_containers=1, scaledown_window=2)
def generate_current(run_id: str, seed: int, hotspots: list, hotspot_source: str, binder_length: list):
    """Build the conditioning, run rfd4-generate.py, and record the design. `binder_length` is [low, high]."""
    import shutil
    import subprocess
    import time
    sys.path.insert(0, str(OPT / "pipeline"))
    from binder_campaign.conditioning import write_conditioning
    from binder_campaign.current_candidates import import_current
    from binder_campaign.generation_input import check_seed
    from binder_campaign.modal_plan import check_run_id, resolve_length

    run_id, seed = check_run_id(run_id), check_seed(seed)
    low, high = resolve_length(length_min=binder_length[0], length_max=binder_length[1])
    root = CACHE / "runs" / run_id
    if (root / "generation").exists():
        raise ValueError("This run ID already has output; use a new --run-id")
    (root / "env").mkdir(parents=True, exist_ok=True)
    shutil.copy2(CACHE / "env" / "rfd4-checkpoints.json", root / "env" / "rfd4-checkpoints.json")
    os.environ.update(TASK_ROOT=str(root), HF_HOME=str(CACHE / "hf"), HF_HUB_DISABLE_IMPLICIT_TOKEN="1", WANDB_MODE="disabled")

    conditioning = root / "inputs/conditioning"
    write_conditioning(OPT / "examples/il7ra-rehearsal", conditioning, hotspots, low if low == high else (low, high),
                       hotspot_source=hotspot_source)
    manifest = root / "inputs/il7ra-manifest.json"
    shutil.copy2(OPT / "docs/eswar/phase1-data/il7ra-manifest.json", manifest)
    output = root / "generation"
    started = time.monotonic()
    subprocess.run([sys.executable, str(OPT / "pipeline/scripts/rfd4-generate.py"), "--reference-dir", str(conditioning),
                    "--manifest", str(manifest), "--out", str(output), "--seed", str(seed)], check=True, timeout=780)
    (output / "environment.txt").write_text(_freeze())
    record = import_current(output / "audit.json", manifest, root / "candidates.json")[0]
    _commit()
    import torch
    return {"run_id": run_id, "audit": json.loads((output / "audit.json").read_text()), "record": record,
            "worker_seconds": time.monotonic() - started, "gpu": torch.cuda.get_device_name(), "archive": _archive(root)}


@app.local_entrypoint()
def main(stage: str = "plan", run_id: str = "", seed: int = 43, option: str = "", hotspots: str = "",
         hotspot_source: str = "", length: int = 60, length_min: int = 0, length_max: int = 0,
         batch_id: str = "", seeds: str = "", lengths: str = "", max_designs: int = 0, max_gpu_seconds: float = 0.0,
         max_failures: int = 2, retry_failed: bool = False, retry_interrupted: bool = False, out: str = ""):
    """Stages: plan (local checks), generate (one paid GPU run) or batch (several designs, budget-capped, resumable)."""
    if stage not in {"plan", "generate", "batch"}:
        raise ValueError("Choose plan, generate or batch")
    sys.path.insert(0, str(HERE / "pipeline"))
    from binder_campaign.modal_plan import MANIFEST_RELATIVE, build_plan, local_dry_run, resolve_hotspots
    if stage == "batch":
        from binder_campaign.batch import make_config, parse_lengths, parse_seeds, run_batch
        hotspot_list, source = resolve_hotspots(HERE, option, hotspots, hotspot_source)
        cfg = make_config(batch_id, parse_seeds(seeds), parse_lengths(lengths), hotspot_list, source,
                          max_designs=max_designs, max_gpu_seconds=max_gpu_seconds, max_failures=max_failures)
        for low, high in cfg["lengths"]:           # free preflight for every length, before any spending
            local_dry_run(HERE, build_plan(HERE, seed=cfg["seeds"][0], hotspots=",".join(map(str, cfg["hotspots"])),
                                           hotspot_source=cfg["hotspot_source"], length=low if low == high else 60,
                                           length_min=0 if low == high else low, length_max=0 if low == high else high))
        folder = Path(out) if out else HERE / "results" / f"batch-{batch_id}"
        print(json.dumps({"batch": cfg, "input_checks": "passed", "gpu_preference": GPU, "output": str(folder)}, indent=2))
        report = run_batch(cfg, lambda t: generate_current.remote(t["run_id"], t["seed"], t["hotspots"], t["hotspot_source"],
                                                                  t["binder_length"]),
                           folder, HERE / MANIFEST_RELATIVE, retry_failed=retry_failed, retry_interrupted=retry_interrupted)
        print(json.dumps({k: report[k] for k in ("stop_reason", "planned", "done", "kept", "dropped_by_dedup", "failed",
                                                 "interrupted", "not_started", "gpu_seconds_reported", "gpu_seconds_note")}, indent=2))
        return
    plan = build_plan(HERE, run_id=run_id, seed=seed, option=option, hotspots=hotspots, hotspot_source=hotspot_source,
                      length=length, length_min=length_min, length_max=length_max)
    checked = local_dry_run(HERE, plan)          # the same input checks the GPU run makes, before any spending
    print(json.dumps({"plan": plan, "input_checks": "passed", "gpu_preference": GPU}, indent=2))
    if stage == "plan":
        return
    if not plan["run_id"]:
        raise ValueError("Supply a fresh --run-id")
    output = HERE / "results" / plan["run_id"]
    archive_path = output / "generate-current.tar.gz"
    if archive_path.exists():
        raise ValueError("Refusing to overwrite an existing local result")
    output.mkdir(parents=True, exist_ok=True)
    result = generate_current.remote(plan["run_id"], plan["seed"], plan["hotspots"], plan["hotspot_source"], plan["binder_length"])
    archive_path.write_bytes(result.pop("archive"))
    result["modal_profile"] = os.environ.get("MODAL_PROFILE", "active CLI profile")
    result["settings_checked_locally"] = checked["overrides"]
    (output / "generate-current.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"saved": str(archive_path), "gpu": result["gpu"], "worker_seconds": result["worker_seconds"],
                      "binder_sequence": result["audit"]["binder_sequence"], "candidate_id": result["record"]["candidate_id"],
                      "role": "on-target generation; not a binding result"}, indent=2))
