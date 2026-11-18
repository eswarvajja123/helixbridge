"""AlphaFold-Multimer (ColabFold) on Modal for the independent cross-check of the shortlist (Phase 5, Task 5.4).

Task 5.4 names a person (Stefanos) for this step; this runs the same kind of check so the shortlist has a third, independent
prediction. It records what AlphaFold says and nothing else: it never sets any review field, which stays a person's
decision. Inputs are the `colabfold-batch.csv` the hand-off package already holds (`TARGET:BINDER` per design).

  modal run modal_alphafold.py --stage prepare                                    # download the AlphaFold-Multimer v3 parameters once
  modal run modal_alphafold.py --stage predict --batch results/alphafold-handoff-x/colabfold-batch.csv --out results/alphafold-x
  python pipeline/scripts/review-alphafold.py ...                                 # measure the poses and write the results sheet

ColabFold 1.6.3 with alphafold-multimer v3, 3 recycles, 2 models, one seed, the best model by ipTM kept. No template, no relaxation.
The alignment is the team's own 1,022-row target-only MSA (docs/eswar/phase1-data/target.a3m) with the binder as a single sequence,
written per design by binder_campaign.alphafold_msa. No public MSA server is used: a first attempt that used the ColabFold server sat
in its queue for 52 minutes per chunk without reaching the GPU, and it would also have sent the unpublished binder sequences to a
third party. Predictions are run in chunks so an interrupted run can be resumed.
"""
from pathlib import Path
import csv
import json
import os
import sys

import modal

HERE = Path(__file__).resolve().parent
VOLUME_NAME = "helixbridge-model-cache"
CACHE = Path(os.environ.get("HELIXBRIDGE_CACHE", "/cache"))
OPT = Path(os.environ.get("HELIXBRIDGE_OPT", "/opt"))
GPU = [g.strip() for g in os.environ.get("HELIXBRIDGE_GPU", "").split(",") if g.strip()] or ["A100-80GB", "L40S"]
COLABFOLD = "1.6.3"
MODEL_TYPE = "alphafold2_multimer_v3"
SETTINGS = {"model_type": MODEL_TYPE, "num_models": 2, "num_recycle": 3, "num_seeds": 1, "rank": "iptm", "msa_mode": "custom a3m: target-only MSA (1,022 rows) + binder single sequence; no MSA server",
            "pair_mode": "unpaired_paired", "templates": False, "relax": False}
CHUNK = 4
STALL_SECONDS = 420          # a job that prints nothing for this long is hung: stop it rather than pay for the GPU
CONSTRAINTS = HERE / "docs" / "eswar" / "alphafold-image-constraints.txt"

app = modal.App("helixbridge-alphafold")
cache = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
base = modal.Image.debian_slim(python_version="3.12").apt_install("git", "build-essential", "curl").pip_install("uv==0.12.9")
_c = "-c /tmp/af-constraints.txt " if CONSTRAINTS.is_file() else ""
_base = base.add_local_file(CONSTRAINTS, "/tmp/af-constraints.txt", copy=True) if CONSTRAINTS.is_file() else base
image = _base.run_commands(
    f"uv pip install --system {_c}'colabfold[alphafold]=={COLABFOLD}' 'jax[cuda12]>=0.6.2,<0.7'",
    "python -c 'import colabfold, jax; from colabfold.download import download_alphafold_params; print(\"colabfold\", jax.__version__)'",
    "colabfold_batch --help > /dev/null",
).add_local_dir(HERE / "pipeline", "/opt/pipeline", ignore=["data/", "__pycache__/", "*.pyc"])


def _commit():
    if not os.environ.get("HELIXBRIDGE_LOCAL_TEST"):
        cache.commit()


def _gpu_state():
    import subprocess
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], text=True, timeout=20).strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"


def _freeze():
    import subprocess
    try:
        return subprocess.check_output(["uv", "pip", "freeze", "--system"], text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        return f"package list unavailable: {error}"


@app.function(image=image, volumes={str(CACHE): cache}, cpu=2, memory=8192, timeout=3600, retries=0, max_containers=1, scaledown_window=2)
def prepare_alphafold():
    """Download the AlphaFold-Multimer v3 parameters into the shared Volume (public Google storage), once per workspace."""
    from colabfold.download import download_alphafold_params
    target = CACHE / "colabfold"
    target.mkdir(parents=True, exist_ok=True)
    download_alphafold_params(MODEL_TYPE, target)
    _commit()
    files = sorted(p.name for p in (target / "params").glob("*"))
    return {"status": "parameters cached", "model_type": MODEL_TYPE, "dir": str(target), "files": files,
            "bytes": sum(p.stat().st_size for p in (target / "params").glob("*"))}


@app.function(image=image, gpu=GPU, cpu=4, memory=32768, volumes={str(CACHE): cache}, timeout=3300, retries=0, max_containers=1, scaledown_window=2)
def predict_chunk(run_id: str, rows: list, target_a3m: str, trace_seconds: int = 0):
    """Predict one chunk of complexes. `rows` is [{"id", "sequence"}] with sequence 'TARGET:BINDER'; `target_a3m` is the target-only MSA.
    Returns an archive of the best models."""
    import io
    import subprocess
    import tarfile
    import time
    started = time.monotonic()
    if not rows or any(set(r["sequence"].replace(":", "")) - set("ACDEFGHIKLMNPQRSTVWY") or r["sequence"].count(":") != 1 for r in rows):
        raise ValueError("Each row needs two canonical chains joined by ':'")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate ids")
    if not (CACHE / "colabfold" / "params").is_dir():
        raise ValueError("The AlphaFold parameters are not cached: run --stage prepare first")
    work = CACHE / "runs" / run_id
    if work.exists():
        raise ValueError("This run id already exists")
    (work / "out").mkdir(parents=True)
    sys.path.insert(0, str(OPT / "pipeline"))
    from binder_campaign.alphafold_msa import complex_a3m
    (work / "msa").mkdir()
    with (work / "input.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "sequence", "a3mpath"])
        for r in rows:
            target, binder = r["sequence"].split(":")
            path = work / "msa" / f"{r['id']}.a3m"
            path.write_text(complex_a3m(target_a3m, target, binder))
            w.writerow([r["id"], r["sequence"], str(path)])
    # Run ColabFold's own entry point unbuffered (so progress is visible) and, when `--trace SECONDS` is given, have Python
    # print every thread's stack once and exit after that long: a diagnostic for a job that hangs before it reaches the GPU.
    trace = float(trace_seconds or 0)
    bootstrap = ("import faulthandler,sys\n" + (f"faulthandler.dump_traceback_later({trace}, exit=True)\n" if trace > 0 else "") +
                 "sys.argv[0]='colabfold_batch'\nfrom colabfold.batch import main\nmain()\n")
    command = [sys.executable, "-u", "-X", "faulthandler", "-c", bootstrap, str(work / "input.csv"), str(work / "out"), "--model-type", SETTINGS["model_type"], "--num-models",
               str(SETTINGS["num_models"]), "--num-recycle", str(SETTINGS["num_recycle"]), "--num-seeds", str(SETTINGS["num_seeds"]),
               "--rank", SETTINGS["rank"], "--pair-mode", SETTINGS["pair_mode"], "--num-relax", "0",
               "--disable-unified-memory",                              # ColabFold's default (CUDA unified memory, 4x fraction) hangs on the first GPU transfer in Modal
               "--data", str(CACHE / "colabfold")]
    from binder_campaign.watchdog import Stalled, run_watched
    try:                                                              # live output; a silent job is stopped, not left to burn GPU time
        code, output = run_watched(command, STALL_SECONDS, 2400, echo=lambda line: print(line, flush=True))
    except Stalled as error:
        raise RuntimeError(str(error) + " | nvidia-smi: " + _gpu_state()) from None
    (work / "colabfold.log").write_text(output)
    if code != 0:
        raise RuntimeError("colabfold_batch failed: " + output.strip()[-600:])
    if "Sleeping for" in output:                  # the MSA server was contacted: the custom alignment was not used
        raise RuntimeError("colabfold_batch contacted the MSA server; the custom alignment was not used")
    keep = []
    for r in rows:
        pdbs = sorted((work / "out").glob(f"{r['id']}_unrelaxed_rank_001_*.pdb"))
        scores = sorted((work / "out").glob(f"{r['id']}_scores_rank_001_*.json"))
        if len(pdbs) != 1 or len(scores) != 1:
            raise RuntimeError(f"Missing the best model or its scores for {r['id']}")
        keep += [pdbs[0], scores[0]]
    (work / "environment.txt").write_text(_freeze())
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for path in keep + [work / "colabfold.log", work / "environment.txt", work / "input.csv"]:
            tar.add(path, arcname=f"{run_id}/{path.name}")
    _commit()
    import jax
    return {"run_id": run_id, "ids": [r["id"] for r in rows], "settings": SETTINGS, "command": command, "device": jax.devices()[0].device_kind,
            "seconds": time.monotonic() - started, "archive": stream.getvalue()}


@app.local_entrypoint()
def main(stage: str = "predict", batch: str = "", out: str = "", chunk: int = CHUNK, trace: int = 0):
    """Stages: prepare (cache the parameters) or predict (every row of a colabfold-batch.csv, in chunks; rerun to resume)."""
    if stage not in {"prepare", "predict"}:
        raise ValueError("Choose prepare or predict")
    if stage == "prepare":
        print(json.dumps(prepare_alphafold.remote(), indent=2))
        return
    sys.path.insert(0, str(HERE / "pipeline"))
    from binder_campaign.modal_plan import check_run_id
    rows = list(csv.DictReader(open(batch)))
    if not rows or not out or chunk < 1:
        raise ValueError("Give --batch (a colabfold-batch.csv), --out and a positive --chunk")
    target_a3m = (HERE / "docs" / "eswar" / "phase1-data" / "target.a3m").read_text()
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "run.json"
    state = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"settings": SETTINGS, "chunks": {}}
    if state["settings"] != SETTINGS:
        raise ValueError("The settings changed since this run began; use a new --out")
    for n in range(0, len(rows), chunk):
        part = rows[n:n + chunk]
        name = f"c{n // chunk + 1}"
        if state["chunks"].get(name, {}).get("status") == "done":
            continue
        run_id = check_run_id(f"{folder.name}-{name}-{len(state['chunks'].get(name, {}).get('attempts', [])) + 1}"[:60])
        entry = state["chunks"].setdefault(name, {"ids": [r["id"] for r in part], "attempts": []})
        entry["attempts"].append(run_id)
        entry["status"] = "started"
        manifest_path.write_text(json.dumps(state, indent=2))
        try:
            result = predict_chunk.remote(run_id, [{"id": r["id"], "sequence": r["sequence"]} for r in part], target_a3m, trace)
        except Exception as error:                              # a failed chunk is kept on record and the others still run
            entry.update(status="failed", error=f"{type(error).__name__}: {error}"[:400])
            manifest_path.write_text(json.dumps(state, indent=2))
            print(json.dumps({"chunk": name, "status": "failed", "error": entry["error"]}))
            continue
        (folder / f"{name}.tar.gz").write_bytes(result.pop("archive"))
        entry.update(status="done", seconds=round(result["seconds"], 1), device=result["device"], run_id=run_id)
        manifest_path.write_text(json.dumps(state, indent=2))
        print(json.dumps({"chunk": name, "status": "done", "ids": len(part), "seconds": entry["seconds"], "device": entry["device"]}))
    print(json.dumps({"output_folder": str(folder), "chunks": {k: v["status"] for k, v in state["chunks"].items()}}, indent=2))
