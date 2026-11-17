"""Budget-capped, resumable evaluation of many candidates with ESMFold2 and Boltz-2 (Phase 3, Task 3.2).

Takes the candidates.json written by the generation batch and attaches RAW scores to each candidate. The prediction itself is an
injected function `evaluate_fn(request) -> response` (Modal in practice, a test double in tests), so this engine is pure.

Rules, in the spirit of the generation batch:
  * only candidates with no QC reasons are evaluated by default, so GPU time is not spent on near-duplicates or invalid designs;
    the skipped ones are listed with their reasons;
  * BUDGET: caps on function calls and on reported GPU seconds are required, and a failure cap stops the run;
  * RESUMABLE: progress is saved before and after every call; finished (candidate, predictor) pairs are skipped on a re-run,
    failed ones are not retried unless asked, and a call interrupted mid-way is reported and not silently re-launched;
  * FAILURES ARE KEPT with their error, one candidate at a time: a bad result for one candidate never discards the others;
  * RAW ONLY: scores are stored exactly as returned (after validation). No threshold, ranking or combination is applied, and the
    original records are copied, never altered: statuses such as screen_status stay as they were until a person reviews.

Reported GPU seconds are the function's own count; Modal bills container time, which is longer (start-up, loading ~25 GB of weights).
"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import statistics
import tarfile
import time

from . import batch as _batch
from .modal_plan import check_run_id
from .runner import digest, write_json
from .scoring import PREDICTORS, check_score

SCHEMA_VERSION = 1
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}")
MAX_EVAL_ID = 30
KERNEL_BACKENDS = ("fused", "cuequivariance", "none")       # "none" = plain PyTorch; passed to the ESMFold2 container only
FILES = ("model.cif", "pae.json", "metrics.json")


def select_candidates(records, include_flagged=False):
    """Split records into those to evaluate and those skipped (with the reason)."""
    chosen, skipped, seen = [], [], set()
    for record in records:
        cid, seq = record.get("candidate_id"), record.get("sequence")
        if not isinstance(cid, str) or not SAFE_ID.fullmatch(cid):
            raise ValueError("A candidate has a missing or unsafe candidate_id")
        if cid in seen:
            raise ValueError("Duplicate candidate_id: " + cid)
        seen.add(cid)
        if not isinstance(seq, str) or not seq or set(seq) - set("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError("Candidate has an invalid sequence: " + cid)
        if record.get("qc_reasons") and not include_flagged:
            skipped.append({"candidate_id": cid, "reasons": list(record["qc_reasons"])})
        else:
            chosen.append(record)
    return chosen, skipped


def make_config(eval_id, predictors, chunk_size, *, max_calls, max_gpu_seconds, max_failures=3, include_flagged=False, seed=0,
                kernel_backend="fused"):
    check_run_id(eval_id)
    if len(eval_id) > MAX_EVAL_ID:
        raise ValueError(f"Evaluation id is too long (at most {MAX_EVAL_ID} characters)")
    predictors = list(predictors)
    if not predictors or len(set(predictors)) != len(predictors) or any(p not in PREDICTORS for p in predictors):
        raise ValueError("Predictors must be a non-empty list without repeats, from: " + ", ".join(PREDICTORS))
    for label, value, floor in (("chunk_size", chunk_size, 1), ("max_calls", max_calls, 1), ("max_failures", max_failures, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < floor:
            raise ValueError(f"{label} must be a whole number of at least {floor}")
    if chunk_size > 20:
        raise ValueError("chunk_size above 20 risks one call running past its time limit")
    if isinstance(max_gpu_seconds, bool) or not isinstance(max_gpu_seconds, (int, float)) or not max_gpu_seconds > 0:
        raise ValueError("max_gpu_seconds must be a positive number")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 31:
        raise ValueError("seed must be a non-negative 31-bit integer")
    if kernel_backend not in KERNEL_BACKENDS:
        raise ValueError("kernel_backend must be one of " + ", ".join(KERNEL_BACKENDS))
    return {"eval_id": eval_id, "predictors": predictors, "chunk_size": chunk_size, "include_flagged": bool(include_flagged),
            "seed": seed, "kernel_backend": kernel_backend, "max_calls": max_calls, "max_gpu_seconds": float(max_gpu_seconds), "max_failures": max_failures}


def _identity(config, chosen):
    fingerprint = hashlib.sha256(json.dumps([[r["candidate_id"], r["sequence"]] for r in chosen]).encode()).hexdigest()
    return {**{k: config[k] for k in ("eval_id", "predictors", "chunk_size", "include_flagged", "seed", "kernel_backend")},
            "candidates_sha256": fingerprint}


def _load_state(path, identity):
    path = Path(path)
    if not path.exists():
        return {"schema": SCHEMA_VERSION, "identity": identity, "budget": {}, "results": {}, "calls": 0,
                "failed_evaluations": 0, "gpu_seconds": 0.0, "stop_reason": None}
    state = json.loads(path.read_text())
    if state.get("schema") != SCHEMA_VERSION or state["identity"] != identity:
        raise ValueError("This output folder belongs to a different evaluation (other candidates or settings); use a new folder "
                         "(only the budget caps may change when resuming)")
    return state


def _stop_reason(config, state):
    if state["calls"] >= config["max_calls"]:
        return "max_calls"
    if state["gpu_seconds"] >= config["max_gpu_seconds"]:
        return "max_gpu_seconds"
    if state["failed_evaluations"] >= config["max_failures"]:
        return "max_failures"
    return None


def _pending(state, predictor, chosen, retry_failed, retry_interrupted):
    out = []
    for record in chosen:
        entry = state["results"].get(record["candidate_id"], {}).get(predictor)
        if entry is None:
            out.append(record)
        elif entry["status"] == "failed" and retry_failed:
            out.append(record)
        elif entry["status"] == "started" and retry_interrupted:
            out.append(record)
    return out


def run_evaluation(config, candidates_path, evaluate_fn, out_dir, *, retry_failed=False, retry_interrupted=False,
                   clock=time.time, log=print):
    """Evaluate (or resume evaluating) the candidates in `candidates_path`. Returns the report; writes evaluations.json etc."""
    out_dir = Path(out_dir)
    records = json.loads(Path(candidates_path).read_text())["candidates"]
    chosen, skipped = select_candidates(records, config["include_flagged"])
    if not chosen:
        raise ValueError("No candidates to evaluate (all were skipped or the file is empty)")
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    state = _load_state(state_path, _identity(config, chosen))
    state["budget"] = {k: config[k] for k in ("max_calls", "max_gpu_seconds", "max_failures")}
    stop = None
    try:
        for predictor in config["predictors"]:
            pending = _pending(state, predictor, chosen, retry_failed, retry_interrupted)
            for start in range(0, len(pending), config["chunk_size"]):
                stop = _stop_reason(config, state)
                if stop:
                    break
                _call(config, predictor, pending[start:start + config["chunk_size"]], state, state_path, evaluate_fn, out_dir, clock, log)
            if stop:
                break
        else:
            open_pairs = [(r["candidate_id"], p) for r in chosen for p in config["predictors"]
                          if state["results"].get(r["candidate_id"], {}).get(p, {}).get("status") != "done"]
            stop = "completed" if not open_pairs else "finished_with_open_pairs"
    except KeyboardInterrupt:
        stop = "interrupted_by_user"
        raise
    finally:
        state["stop_reason"] = stop
        write_json(state_path, state)
        report = assemble(config, state, records, chosen, skipped, out_dir)
    return report


def _call(config, predictor, chunk, state, state_path, evaluate_fn, out_dir, clock, log):
    state["calls"] += 1
    run_id = check_run_id(f"{config['eval_id']}-{predictor}-c{state['calls']}")
    for record in chunk:                              # recorded BEFORE the call: a crash leaves 'started', not nothing
        slot = state["results"].setdefault(record["candidate_id"], {})
        old = slot.get(predictor)
        history = list(old.get("history", [])) if old else []
        if old:
            history.append({"run_id": old["run_id"], "status": old["status"], "error": old.get("error")})
        slot[predictor] = {"status": "started", "attempts": (old["attempts"] if old else 0) + 1, "run_id": run_id,
                           "started_at": clock(), "error": None, "history": history}
    write_json(state_path, state)
    log(f"[{predictor}] call {state['calls']}: {len(chunk)} candidate(s) as {run_id}")
    request = {"run_id": run_id, "predictor": predictor, "seed": config["seed"], "kernel_backend": config["kernel_backend"],
               "candidates": [{"id": r["candidate_id"], "sequence": r["sequence"]} for r in chunk]}
    try:
        response = evaluate_fn(request)
        state["gpu_seconds"] += float(response["worker_seconds"])
        folder = out_dir / "evals" / run_id
        folder.mkdir(parents=True, exist_ok=False)
        archive = response["archive"]
        (folder / "result.tar.gz").write_bytes(archive)
        returned = {item["id"]: item for item in response["results"]}
        gpu = response.get("gpu")
    except Exception as error:                        # KeyboardInterrupt is deliberately not caught
        for record in chunk:
            _fail(state, record["candidate_id"], predictor, "The call failed: " + _batch._short(error), clock)
        write_json(state_path, state)
        log(f"[{predictor}] call {state['calls']} FAILED: {_batch._short(error)}")
        return
    for record in chunk:
        cid = record["candidate_id"]
        try:
            item = returned.get(cid)
            if item is None:
                raise ValueError("No result was returned for this candidate")
            if item.get("status") != "ok":
                raise ValueError("The container reported a failure: " + str(item.get("error"))[:300])
            if item.get("sequence") != record["sequence"]:
                raise ValueError("The returned sequence differs from the candidate's")
            score = check_score(item["score"], predictor)
            artifacts = _extract_candidate(archive, run_id, cid, out_dir / "evals" / cid / predictor)
            entry = state["results"][cid][predictor]
            entry.update(status="done", score={**score, "gpu": score.get("gpu") or gpu}, artifacts=artifacts, finished_at=clock())
        except Exception as error:
            _fail(state, cid, predictor, _batch._short(error), clock)
            log(f"[{predictor}] {cid} FAILED: {_batch._short(error)}")
    write_json(state_path, state)


def _fail(state, cid, predictor, message, clock):
    entry = state["results"][cid][predictor]
    entry.update(status="failed", error=message, finished_at=clock())
    state["failed_evaluations"] += 1


def _extract_candidate(archive_bytes, run_id, cid, folder):
    """Save the three files for one candidate from the archive; returns {name: sha256}."""
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        blobs = {name: _batch._read_member(tar, f"{cid}/{name}", run_id) for name in FILES}
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in blobs.items():
        (folder / name).write_bytes(data)
    return {name: digest(folder / name) for name in FILES}


COMPARE_METRICS = (("iptm", lambda s: s.get("iptm")), ("ipsae_max", lambda s: (s.get("ipsae") or {}).get("max")))


def _spread(values):
    """min / median / max of a list of numbers, or None when there are none. Descriptive only."""
    values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not values:
        return None
    return {"n": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}


def _done_score(item, predictor):
    e = item["evaluations"].get(predictor) or {}
    return e.get("score") if e.get("status") == "done" else None


def comparison_rows(evaluated, predictors):
    """One row per candidate, in the order of the candidate file (never sorted by score): each predictor's iptm and ipsae_max side by
    side, plus the plain difference between the first two predictors where both finished. The difference is arithmetic, not a verdict."""
    rows = []
    for item in evaluated:
        if "evaluation_skipped" in item:
            continue
        row = {"candidate_id": item["candidate_id"], "seed": item.get("seed"), "binder_length": len(item["sequence"])}
        scores = {p: _done_score(item, p) for p in predictors}
        for predictor in predictors:
            for name, get in COMPARE_METRICS:
                value = get(scores[predictor]) if scores[predictor] else None
                row[f"{predictor}_{name}"] = value
        if len(predictors) == 2:
            first, second = predictors
            for name, _ in COMPARE_METRICS:
                a, b = row[f"{first}_{name}"], row[f"{second}_{name}"]
                row[f"{first}_minus_{second}_{name}"] = None if a is None or b is None else round(a - b, 6)
        rows.append(row)
    return rows


def pool_summary(evaluated, predictors):
    """Spread of each raw metric over THIS pool of candidates. Describes the pool; says nothing about any design being good."""
    out = {"by_predictor": {}, "paired_difference": {}}
    for predictor in predictors:
        scores = [s for s in (_done_score(i, predictor) for i in evaluated) if s]
        out["by_predictor"][predictor] = {name: _spread([get(s) for s in scores]) for name, get in COMPARE_METRICS}
    if len(predictors) == 2:
        first, second = predictors
        for name, get in COMPARE_METRICS:
            diffs = []
            for item in evaluated:
                a, b = _done_score(item, first), _done_score(item, second)
                if a and b and get(a) is not None and get(b) is not None:
                    diffs.append(get(a) - get(b))
            out["paired_difference"][f"{first}_minus_{second}_{name}"] = _spread(diffs)
    out["note"] = ("Descriptive spread over this pool only. There is no supportable pass/fail threshold: the team's public pilot "
                   "(outputs/) found negative controls scoring as high as binders, and its numeric gates are relative to a parent "
                   "design. Disagreement between the two predictors is a reason for a person to look, not a verdict.")
    return out


def assemble(config, state, records, chosen, skipped, out_dir):
    out_dir = Path(out_dir)
    skipped_ids = {s["candidate_id"]: s["reasons"] for s in skipped}
    evaluated = []
    for record in records:
        cid = record["candidate_id"]
        slot = state["results"].get(cid, {})
        entry = {}
        for predictor in config["predictors"]:
            r = slot.get(predictor)
            if cid in skipped_ids:
                continue
            entry[predictor] = ({"status": "not_run"} if r is None else
                                {k: v for k, v in {"status": r["status"], "run_id": r["run_id"], "attempts": r["attempts"],
                                                   "score": r.get("score"), "artifacts": r.get("artifacts"), "error": r.get("error"),
                                                   "earlier_attempts": r.get("history") or None}.items() if v is not None})
        item = {**record, "evaluations": entry}
        if cid in skipped_ids:
            item["evaluation_skipped"] = skipped_ids[cid]
        evaluated.append(item)
    write_json(out_dir / "evaluations.json", {"candidates": evaluated, "count": len(evaluated),
               "note": "Raw predictor scores attached to copies of the candidate records. Nothing is ranked, thresholded or "
                       "promoted: screen_status, novelty_review and submission_eligible are unchanged."})
    with (out_dir / "scores.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["candidate_id", "seed", "binder_length", "predictor", "status", "iptm", "ptm", "ipsae_a_b", "ipsae_b_a",
                         "ipsae_max", "ipsae_min_directional", "plddt_raw_mean"])
        for item in evaluated:
            for predictor, e in item["evaluations"].items():
                s = e.get("score") or {}
                ip = s.get("ipsae") or {}
                writer.writerow([item["candidate_id"], item.get("seed"), len(item["sequence"]), predictor, e["status"],
                                 s.get("iptm", ""), s.get("ptm", ""), ip.get("a_b", ""), ip.get("b_a", ""), ip.get("max", ""),
                                 ip.get("min_directional", ""), s.get("plddt_raw_mean") if s.get("plddt_raw_mean") is not None else ""])
    rows = comparison_rows(evaluated, config["predictors"])
    columns = list(rows[0]) if rows else ["candidate_id"]
    with (out_dir / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    per = {}
    for predictor in config["predictors"]:
        counts = {"done": 0, "failed": 0, "started": 0, "not_started": 0}
        for r in chosen:
            e = state["results"].get(r["candidate_id"], {}).get(predictor)
            counts["not_started" if e is None else e["status"]] += 1
        per[predictor] = counts
    failures = [{"candidate_id": cid, "predictor": p, "run_id": e["run_id"], "error": e["error"]}
                for cid, slot in state["results"].items() for p, e in slot.items() if e["status"] == "failed"]
    report = {"eval_id": config["eval_id"], "stop_reason": state["stop_reason"], "candidates_total": len(records),
              "candidates_evaluated": len(chosen), "candidates_skipped": skipped, "by_predictor": per, "failures": failures,
              "interrupted": [{"candidate_id": cid, "predictor": p, "run_id": e["run_id"]}
                              for cid, slot in state["results"].items() for p, e in slot.items() if e["status"] == "started"],
              "calls": state["calls"], "failed_evaluations": state["failed_evaluations"],
              "gpu_seconds_reported": round(state["gpu_seconds"], 2),
              "gpu_seconds_note": "The function's own count; Modal bills container time, which is longer. Check billing.",
              "budget": {k: config[k] for k in ("max_calls", "max_gpu_seconds", "max_failures")},
              "pool_summary": pool_summary(evaluated, config["predictors"]),
              "status_note": "Raw scores only. No threshold, ranking or combination; not a binding result."}
    write_json(out_dir / "evaluation-report.json", report)
    return report


def config_from_state(state):
    identity = state["identity"]
    return {**{k: identity[k] for k in ("eval_id", "predictors", "chunk_size", "include_flagged", "seed", "kernel_backend")},
            **state["budget"]}
