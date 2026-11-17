"""Plan and report on the evaluation of a batch of candidates (Phase 3, Task 3.2). CPU only; needs no Modal login.

  plan   check an evaluation and print exactly what it would do, including the Modal command to run it (spends nothing)
  report read an existing evaluation folder and print what happened (rewrites its evaluations.json, scores.csv, report)

The evaluation itself runs through Modal, because that is what launches the GPU jobs:
  modal run modal_evaluate.py --stage evaluate --candidates results/batch-b1/candidates.json --eval-id eswar-e1 \\
        --predictors esmfold2,boltz2 --chunk-size 4 --max-calls 6 --max-gpu-seconds 1800

Example:
  python pipeline/scripts/evaluate-candidates.py plan --candidates results/batch-b1/candidates.json --eval-id eswar-e1 \\
        --max-calls 6 --max-gpu-seconds 1800
  python pipeline/scripts/evaluate-candidates.py report --out results/eval-eswar-e1
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.esm_eval import build_request, load_manifest
from binder_campaign.evaluate import assemble, config_from_state, make_config, select_candidates

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = parser.add_subparsers(dest="command", required=True)
plan_p = sub.add_parser("plan")
plan_p.add_argument("--root", type=Path, default=ROOT)
plan_p.add_argument("--candidates", type=Path, required=True)
plan_p.add_argument("--eval-id", required=True)
plan_p.add_argument("--predictors", default="esmfold2,boltz2")
plan_p.add_argument("--chunk-size", type=int, default=4)
plan_p.add_argument("--max-calls", type=int, default=0)
plan_p.add_argument("--max-gpu-seconds", type=float, default=0.0)
plan_p.add_argument("--max-failures", type=int, default=3)
plan_p.add_argument("--seed", type=int, default=0)
plan_p.add_argument("--kernel-backend", default="fused")
plan_p.add_argument("--include-flagged", action="store_true")
plan_p.add_argument("--out", default="")
report_p = sub.add_parser("report")
report_p.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

try:
    if args.command == "plan":
        predictors = [p.strip() for p in args.predictors.split(",") if p.strip()]
        config = make_config(args.eval_id, predictors, args.chunk_size, max_calls=args.max_calls, max_gpu_seconds=args.max_gpu_seconds,
                             max_failures=args.max_failures, include_flagged=args.include_flagged, seed=args.seed,
                             kernel_backend=args.kernel_backend)
        records = json.loads(args.candidates.read_text())["candidates"]
        chosen, skipped = select_candidates(records, args.include_flagged)
        if not chosen:
            raise ValueError("No candidates to evaluate (all were skipped or the file is empty)")
        # the container's own input checks, made here first
        load_manifest(args.root / "docs/eswar/esmfold2-checkpoint-manifest.json")
        phase1 = args.root / "docs/eswar/phase1-data"
        target = json.loads((phase1 / "il7ra-manifest.json").read_text())["sequence"]
        msa_rows = [l.strip() for l in (phase1 / "target.a3m").read_text().splitlines() if l and not l.startswith(">")]
        if msa_rows[0] != target:
            raise ValueError("The MSA's query row is not the current target sequence")
        backend = None if args.kernel_backend == "none" else args.kernel_backend
        for record in chosen:
            build_request(target, record["sequence"], phase1 / "target.a3m", record["candidate_id"], role="check", seed=args.seed,
                          kernel_backend=backend)
        per_predictor = math.ceil(len(chosen) / args.chunk_size)
        calls = per_predictor * len(predictors)
        folder = args.out or f"results/eval-{args.eval_id}"
        command = (f"modal run modal_evaluate.py --stage evaluate --candidates {args.candidates} --eval-id {args.eval_id} "
                   f"--predictors {args.predictors} --chunk-size {args.chunk_size} --max-calls {args.max_calls} "
                   f"--max-gpu-seconds {args.max_gpu_seconds:g} --max-failures {args.max_failures} --seed {args.seed} "
                   f"--kernel-backend {args.kernel_backend}" + (" --include-flagged" if args.include_flagged else ""))
        resume = None
        state_path = Path(folder) / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            counts = {}
            for slot in state["results"].values():
                for entry in slot.values():
                    counts[entry["status"]] = counts.get(entry["status"], 0) + 1
            resume = {"folder_exists": True, "calls_so_far": state["calls"], "pairs_by_status": counts,
                      "same_settings": all(state["identity"].get(k) == config[k]
                                           for k in ("eval_id", "predictors", "chunk_size", "include_flagged", "seed", "kernel_backend")),
                      "note": "the candidates must also be identical; the run itself checks that"}
        print(json.dumps({"input_checks": "passed", "candidates_to_evaluate": len(chosen), "candidates_skipped": skipped,
                          "predictors": predictors, "calls_needed_if_nothing_fails": calls,
                          "calls_that_can_launch_under_the_cap": min(calls, config["max_calls"]),
                          "model_loads_note": "ESMFold2 loads its ~25 GB model once per call, so larger chunks load it less often",
                          "budget": {k: config[k] for k in ("max_calls", "max_gpu_seconds", "max_failures")},
                          "budget_note": "Caps are checked before every call; GPU seconds are the function's own count, and Modal "
                                         "bills container time, which is longer (model loading, start-up). Set the cap lower than your limit.",
                          "output_folder": folder, "resume": resume, "command": command}, indent=2))
    else:
        state = json.loads((args.out / "state.json").read_text())
        stored = json.loads((args.out / "evaluations.json").read_text())["candidates"]
        records = [{k: v for k, v in item.items() if k not in ("evaluations", "evaluation_skipped")} for item in stored]
        config = config_from_state(state)
        chosen, skipped = select_candidates(records, config["include_flagged"])
        report = assemble(config, state, records, chosen, skipped, args.out)
        print(json.dumps({k: report[k] for k in ("eval_id", "stop_reason", "candidates_evaluated", "by_predictor", "failures",
                                                 "interrupted", "calls", "gpu_seconds_reported")}, indent=2))
except (ValueError, FileNotFoundError, KeyError) as error:
    sys.exit(f"REFUSED: {error}")
