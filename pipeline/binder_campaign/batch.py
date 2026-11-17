"""Budget-capped, resumable batch of single-design generations (Phase 2, Task 2.5).

The engine is pure: the actual generation is an injected function `generate_fn(task) -> result`, so it runs the same
way with Modal (modal_current.py --stage batch) or with a test double. For each design it saves the returned archive,
re-checks it, runs the dedup gate, and writes a candidate record. Rules this engine enforces:

  * BUDGET: a batch cannot start without explicit caps on attempts and on reported GPU seconds, and it stops after a
    set number of failures. The caps are checked before every launch.
  * RESUMABLE: progress lives in state.json (written before and after every attempt). A re-run skips finished
    designs and rebuilds the dedup gate by replaying them in their original order; it refuses a changed configuration.
  * FAILURES ARE KEPT: a failed or unusable result is recorded with its error and never retried unless asked for.
    A design interrupted mid-call is reported as such and is not silently re-launched (it may already be billed).
  * NOTHING IS PROMOTED: every record comes from current_candidates.import_current, so it starts unreviewed and not
    eligible; designs dropped by the dedup gate stay on record with a near_duplicate reason.

Reported GPU seconds are the function's own `worker_seconds`; Modal bills container time, which is longer
(start-up, image pull). Set the cap conservatively. No prices are assumed anywhere.
"""
import io
import json
import os
from pathlib import Path
import tarfile
import time

from .current_candidates import import_current
from .dedup import DedupGate, Thresholds, read_chain_ca
from .generation_input import check_seed
from .modal_plan import check_run_id, resolve_length
from .runner import digest, write_json
from .staging import validate_member_name

SCHEMA_VERSION = 1
MAX_MEMBER_BYTES = 50 * 1024 ** 2
MAX_BATCH_ID = 36            # leaves room for "-s<seed>-l<lo>x<hi>-a<n>" inside the 64-character run-id limit
MAX_TASKS = 500


def parse_seeds(text):
    try:
        return [int(x) for x in str(text).split(",") if x.strip()]
    except ValueError:
        raise ValueError("Seeds must be comma-separated whole numbers, e.g. 43,44,45") from None


def parse_lengths(text):
    """'50,60,70-80' -> [[50, 50], [60, 60], [70, 80]]"""
    out = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            low, _, high = token.partition("-")
            out.append([int(low), int(high or low)])
        except ValueError:
            raise ValueError("Lengths look like 50,60,70-80") from None
    return out


def make_config(batch_id, seeds, lengths, hotspots, hotspot_source, *, max_designs, max_gpu_seconds, max_failures=2,
                thresholds=None):
    """Validate everything up front. The budget arguments have no defaults on purpose."""
    check_run_id(batch_id)
    if len(batch_id) > MAX_BATCH_ID:
        raise ValueError(f"Batch id is too long (at most {MAX_BATCH_ID} characters)")
    seeds = [check_seed(s) for s in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Give at least one seed, without repeats")
    normalised = []
    for item in lengths:
        try:
            low, high = (item, item) if isinstance(item, int) and not isinstance(item, bool) else item
        except (TypeError, ValueError):
            raise ValueError("Each length must be a whole number or a [min, max] pair") from None
        pair = resolve_length(length_min=low, length_max=high) if low != high else resolve_length(low)
        if pair not in normalised:
            normalised.append(pair)
    if not normalised:
        raise ValueError("Give at least one binder length")
    if len(seeds) * len(normalised) > MAX_TASKS:
        raise ValueError(f"More than {MAX_TASKS} designs planned; split the batch")
    for seed in seeds:                               # a too-long run id must fail now, not after money is spent
        for low, high in normalised:
            check_run_id(f"{batch_id}-s{seed}-l{low}x{high}-a99")
    hotspots = sorted(int(h) for h in hotspots)
    if not hotspots or not str(hotspot_source).strip():
        raise ValueError("Hotspots and their source are required")
    if isinstance(max_designs, bool) or not isinstance(max_designs, int) or max_designs < 1:
        raise ValueError("max_designs must be a whole number of at least 1")
    if isinstance(max_gpu_seconds, bool) or not isinstance(max_gpu_seconds, (int, float)) or not max_gpu_seconds > 0:
        raise ValueError("max_gpu_seconds must be a positive number")
    if isinstance(max_failures, bool) or not isinstance(max_failures, int) or max_failures < 1:
        raise ValueError("max_failures must be a whole number of at least 1")
    t = thresholds or Thresholds()
    return {"batch_id": batch_id, "seeds": seeds, "lengths": normalised, "hotspots": hotspots,
            "hotspot_source": str(hotspot_source).strip(),
            "thresholds": {"identity": t.identity, "contiguous": t.contiguous, "tm": t.tm},
            "max_designs": max_designs, "max_gpu_seconds": float(max_gpu_seconds), "max_failures": max_failures}


def identity_of(config):
    """The part of a config that must not change when a batch is resumed (budget caps may be raised)."""
    return {k: config[k] for k in ("batch_id", "seeds", "lengths", "hotspots", "hotspot_source", "thresholds")}


def build_tasks(config):
    """Seed-major order with lengths inside, so stopping early still gives a spread of lengths."""
    tasks = []
    for seed in config["seeds"]:
        for low, high in config["lengths"]:
            tag = f"l{low}" if low == high else f"l{low}x{high}"
            tasks.append({"task_id": f"s{seed}-{tag}", "seed": seed, "binder_length": [low, high]})
    return tasks


def _load_state(path, config):
    path = Path(path)
    if not path.exists():
        return {"schema": SCHEMA_VERSION, "identity": identity_of(config), "tasks": {}, "offers": 0,
                "failed_attempts": 0, "stop_reason": None}
    state = json.loads(path.read_text())
    if state.get("schema") != SCHEMA_VERSION:
        raise ValueError("State file has an unknown schema")
    if state["identity"] != identity_of(config):
        raise ValueError("This output folder belongs to a different batch configuration; use a new folder "
                         "(only the budget caps may change when resuming)")
    return state


def config_from_state(state):
    """Rebuild the config a batch was run with (used by the read-only report)."""
    return {**state["identity"], **state["budget"]}


def _save(path, state):
    write_json(path, state)


def _short(error):
    text = f"{type(error).__name__}: {error}".strip()
    return text if len(text) <= 600 else text[:597] + "..."


def _read_member(tar, suffix, run_id):
    """Return the bytes of exactly one regular file named <run_id>/<suffix>, with the safety checks."""
    wanted = f"{run_id}/{suffix}"
    found = []
    for member in tar.getmembers():
        validate_member_name(member.name)
        if member.name == wanted:
            found.append(member)
    if len(found) != 1 or not found[0].isfile():
        raise ValueError(f"Archive must hold exactly one regular file {wanted}")
    if found[0].size > MAX_MEMBER_BYTES:
        raise ValueError(f"Archive member {wanted} is too large")
    return tar.extractfile(found[0]).read()


def _extract(archive_bytes, run_id, folder):
    """Save the archive and extract only the two files we need; returns (model_path, audit)."""
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "result.tar.gz").write_bytes(archive_bytes)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        model = _read_member(tar, "generation/model.cif", run_id)
        audit_bytes = _read_member(tar, "generation/audit.json", run_id)
    (folder / "model.cif").write_bytes(model)
    (folder / "audit.remote.json").write_bytes(audit_bytes)
    audit = json.loads(audit_bytes)
    if digest(folder / "model.cif") != audit["generated_sha256"]:
        raise ValueError("The returned structure does not match the hash in its audit")
    local = {**audit, "remote_generated_structure": audit["generated_structure"],
             "generated_structure": str(folder / "model.cif")}
    write_json(folder / "audit.json", local)
    return folder / "model.cif", local


def _gate_inputs(folder, audit):
    sequence, coords = read_chain_ca(folder / "model.cif", audit["binder_chain"])
    if sequence != audit["binder_sequence"]:
        raise ValueError("The binder chain in the structure differs from the audit's binder sequence")
    return sequence, coords


def _rebuild_gate(state, out_dir, config):
    """Replay finished designs in their original order; a different outcome means something changed."""
    t = config["thresholds"]
    gate = DedupGate(Thresholds(**t))
    finished = sorted((e for e in state["tasks"].values() if e["status"] == "done"), key=lambda e: e["offer_index"])
    for entry in finished:
        folder = out_dir / "designs" / entry["run_id"]
        audit = json.loads((folder / "audit.json").read_text())
        sequence, coords = _gate_inputs(folder, audit)
        decision = gate.offer(entry["run_id"], sequence, coords)
        if (decision.kept, decision.reason) != (entry["decision"]["kept"], entry["decision"]["reason"]):
            raise ValueError("Replaying the dedup gate gave a different outcome for " + entry["run_id"])
    return gate


def _budget_stop(config, state):
    attempts = sum(e["attempts"] for e in state["tasks"].values())
    seconds = sum(e.get("worker_seconds") or 0.0 for e in state["tasks"].values())
    if attempts >= config["max_designs"]:
        return "max_designs"
    if seconds >= config["max_gpu_seconds"]:
        return "max_gpu_seconds"
    if state["failed_attempts"] >= config["max_failures"]:
        return "max_failures"
    return None


def run_batch(config, generate_fn, out_dir, manifest_path, *, retry_failed=False, retry_interrupted=False,
              clock=time.time, log=print):
    """Run (or resume) a batch. Returns the report dict; also writes candidates.json and batch-report.json."""
    out_dir, manifest_path = Path(out_dir), Path(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    state = _load_state(state_path, config)
    state["budget"] = {k: config[k] for k in ("max_designs", "max_gpu_seconds", "max_failures")}   # caps may be raised on resume
    gate = _rebuild_gate(state, out_dir, config)
    stop = None
    try:
        for task in build_tasks(config):
            entry = state["tasks"].get(task["task_id"])
            if entry:
                if entry["status"] == "done":
                    continue
                if entry["status"] == "failed" and not retry_failed:
                    continue
                if entry["status"] == "started" and not retry_interrupted:
                    continue
            stop = _budget_stop(config, state)
            if stop:
                break
            _attempt(config, task, entry, state, state_path, gate, generate_fn, out_dir, manifest_path, clock, log)
        else:
            open_tasks = [t for t in build_tasks(config) if state["tasks"].get(t["task_id"], {}).get("status") != "done"]
            stop = "completed" if not open_tasks else "finished_with_open_tasks"   # see failed / interrupted in the report
    except KeyboardInterrupt:
        stop = "interrupted_by_user"
        raise
    finally:
        state["stop_reason"] = stop
        _save(state_path, state)
        report = assemble(config, state, out_dir)
    return report


def _attempt(config, task, entry, state, state_path, gate, generate_fn, out_dir, manifest_path, clock, log):
    number = (entry["attempts"] if entry else 0) + 1
    run_id = check_run_id(f"{config['batch_id']}-{task['task_id']}-a{number}")
    history = list(entry.get("history", [])) if entry else []          # earlier attempts stay on record
    if entry:
        history.append({"run_id": entry["run_id"], "status": entry["status"], "error": entry.get("error")})
    entry = {"seed": task["seed"], "binder_length": task["binder_length"], "run_id": run_id, "attempts": number,
             "status": "started", "started_at": clock(), "worker_seconds": None, "gpu": None, "error": None,
             "history": history}
    state["tasks"][task["task_id"]] = entry
    _save(state_path, state)                      # recorded BEFORE the call: a crash leaves 'started', not nothing
    log(f"[{task['task_id']}] attempt {number}: launching {run_id}")
    try:
        result = generate_fn({"run_id": run_id, "seed": task["seed"], "binder_length": task["binder_length"],
                              "hotspots": config["hotspots"], "hotspot_source": config["hotspot_source"]})
        entry["worker_seconds"] = float(result["worker_seconds"])
        entry["gpu"] = result.get("gpu")
        folder = out_dir / "designs" / run_id
        _, audit = _extract(result["archive"], run_id, folder)
        sequence, coords = _gate_inputs(folder, audit)
        # prove the record can be built before the gate learns about this design, so a failure cannot leave it half-counted
        import_current(folder / "audit.json", manifest_path, folder / "precheck.json")
        os.unlink(folder / "precheck.json")
        decision = gate.offer(run_id, sequence, coords)
        record = import_current(folder / "audit.json", manifest_path, folder / "record.json", dedup_decision=decision)[0]
        entry.update(status="done", candidate_id=record["candidate_id"], decision=decision.as_dict(),
                     offer_index=state["offers"], finished_at=clock())
        state["offers"] += 1
        log(f"[{task['task_id']}] {'kept' if decision.kept else 'dropped (' + decision.reason + ')'}: {sequence[:20]}...")
    except Exception as error:                    # KeyboardInterrupt is deliberately not caught
        entry.update(status="failed", error=_short(error), finished_at=clock())
        state["failed_attempts"] += 1
        log(f"[{task['task_id']}] FAILED: {entry['error']}")
    _save(state_path, state)


def assemble(config, state, out_dir):
    """Write candidates.json (every finished design, dropped ones included) and batch-report.json."""
    out_dir = Path(out_dir)
    done = sorted((e for e in state["tasks"].values() if e["status"] == "done"), key=lambda e: e["offer_index"])
    records = [json.loads((out_dir / "designs" / e["run_id"] / "record.json").read_text())["candidates"][0] for e in done]
    write_json(out_dir / "candidates.json", {"candidates": records, "count": len(records),
               "note": "Batch of conditioned generations; every record is unreviewed and not eligible. "
                       "Designs dropped by the dedup gate are kept here with a near_duplicate QC reason."})
    tasks = build_tasks(config)
    by_status = {"done": [], "failed": [], "started": []}
    for task in tasks:
        e = state["tasks"].get(task["task_id"])
        if e:
            by_status[e["status"]].append(task["task_id"])
    reasons = {}
    for e in done:
        if not e["decision"]["kept"]:
            reasons[e["decision"]["reason"]] = reasons.get(e["decision"]["reason"], 0) + 1
    report = {
        "batch_id": config["batch_id"], "stop_reason": state["stop_reason"], "planned": len(tasks),
        "done": len(done), "kept": sum(1 for e in done if e["decision"]["kept"]),
        "dropped_by_dedup": sum(1 for e in done if not e["decision"]["kept"]), "dropped_by_reason": reasons,
        "failed": [{"task": t, "run_id": state["tasks"][t]["run_id"], "error": state["tasks"][t]["error"]} for t in by_status["failed"]],
        "interrupted": [{"task": t, "run_id": state["tasks"][t]["run_id"]} for t in by_status["started"]],
        "not_started": [t["task_id"] for t in tasks if t["task_id"] not in state["tasks"]],
        "attempts": sum(e["attempts"] for e in state["tasks"].values()),
        "failed_attempts": state["failed_attempts"],
        "gpu_seconds_reported": round(sum(e.get("worker_seconds") or 0.0 for e in state["tasks"].values()), 2),
        "gpu_seconds_note": "Worker seconds as reported by the function; Modal bills container time, which is longer. Check billing.",
        "budget": {k: config[k] for k in ("max_designs", "max_gpu_seconds", "max_failures")},
        "hotspots": config["hotspots"], "hotspot_source": config["hotspot_source"], "thresholds": config["thresholds"],
        "designs": {t: {"status": e["status"], "run_id": e["run_id"], "attempts": e["attempts"], "seed": e["seed"],
                        "binder_length": e["binder_length"], "worker_seconds": e.get("worker_seconds"), "gpu": e.get("gpu"),
                        "candidate_id": e.get("candidate_id"), "dedup": e.get("decision"), "error": e.get("error"),
                        "earlier_attempts": e.get("history", [])}
                    for t, e in state["tasks"].items()},
        "status_note": "None of these designs is a binding result or a submission entry.",
    }
    write_json(out_dir / "batch-report.json", report)
    return report
