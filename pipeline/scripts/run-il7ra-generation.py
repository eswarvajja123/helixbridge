"""Plan and report on a batch of on-target generations (Phase 2, Task 2.5). CPU only; needs no Modal login.

  plan   check a batch and print exactly what it would do, including the Modal command to run it (spends nothing)
  report read an existing batch folder and print what happened (rewrites its candidates.json and batch-report.json)

The batch itself runs through Modal, because that is what launches the GPU jobs:
  modal run modal_current.py --stage batch --batch-id B --seeds 43,44,45 --lengths 50,60,70 --option A \\
        --max-designs 6 --max-gpu-seconds 600

Example:
  python pipeline/scripts/run-il7ra-generation.py plan --batch-id b1 --seeds 43,44,45 --lengths 50,60,70 \\
        --option A --max-designs 6 --max-gpu-seconds 600
  python pipeline/scripts/run-il7ra-generation.py report --out results/batch-b1
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.batch import assemble, build_tasks, config_from_state, make_config, parse_lengths, parse_seeds
from binder_campaign.modal_plan import build_plan, local_dry_run, resolve_hotspots

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = parser.add_subparsers(dest="command", required=True)
plan_p = sub.add_parser("plan")
plan_p.add_argument("--root", type=Path, default=ROOT)
plan_p.add_argument("--batch-id", required=True)
plan_p.add_argument("--seeds", required=True)
plan_p.add_argument("--lengths", required=True)
plan_p.add_argument("--option", default="")
plan_p.add_argument("--hotspots", default="")
plan_p.add_argument("--hotspot-source", default="")
plan_p.add_argument("--max-designs", type=int, default=0)
plan_p.add_argument("--max-gpu-seconds", type=float, default=0.0)
plan_p.add_argument("--max-failures", type=int, default=2)
plan_p.add_argument("--out", default="")
report_p = sub.add_parser("report")
report_p.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

try:
    if args.command == "plan":
        hotspots, source = resolve_hotspots(args.root, args.option, args.hotspots, args.hotspot_source)
        config = make_config(args.batch_id, parse_seeds(args.seeds), parse_lengths(args.lengths), hotspots, source,
                             max_designs=args.max_designs, max_gpu_seconds=args.max_gpu_seconds, max_failures=args.max_failures)
        for low, high in config["lengths"]:       # the generation script's real input checks, for every length
            local_dry_run(args.root, build_plan(args.root, seed=config["seeds"][0], hotspots=",".join(map(str, hotspots)),
                                                hotspot_source=source, length=low if low == high else 60,
                                                length_min=0 if low == high else low, length_max=0 if low == high else high))
        tasks = build_tasks(config)
        folder = args.out or f"results/batch-{config['batch_id']}"
        launched = min(len(tasks), config["max_designs"])
        command = (f"modal run modal_current.py --stage batch --batch-id {config['batch_id']} --seeds {args.seeds} "
                   f"--lengths {args.lengths} " + (f"--option {args.option or 'A'} " if not args.hotspots else
                   f"--hotspots {args.hotspots} --hotspot-source {source} ") +
                   f"--max-designs {config['max_designs']} --max-gpu-seconds {config['max_gpu_seconds']:g} "
                   f"--max-failures {config['max_failures']}")
        existing = Path(folder) / "state.json"
        resume = None
        if existing.exists():
            state = json.loads(existing.read_text())
            counts = {}
            for entry in state["tasks"].values():
                counts[entry["status"]] = counts.get(entry["status"], 0) + 1
            resume = {"folder_exists": True, "same_configuration": state["identity"] == {k: config[k] for k in state["identity"]},
                      "designs_by_status": counts}
        print(json.dumps({"input_checks": "passed", "designs_planned": len(tasks),
                          "designs_that_can_launch_under_the_cap": launched, "hotspots": hotspots, "hotspot_source": source,
                          "tasks": [{"task": t["task_id"], "seed": t["seed"], "binder_length": t["binder_length"],
                                     "first_run_id": f"{config['batch_id']}-{t['task_id']}-a1"} for t in tasks],
                          "budget": {k: config[k] for k in ("max_designs", "max_gpu_seconds", "max_failures")},
                          "budget_note": "Caps are checked before every launch; GPU seconds are the function's own count, "
                                         "and Modal bills container time, which is longer.",
                          "output_folder": folder, "resume": resume, "command": command}, indent=2))
    else:
        state = json.loads((args.out / "state.json").read_text())
        report = assemble(config_from_state(state), state, args.out)
        print(json.dumps({k: report[k] for k in ("batch_id", "stop_reason", "planned", "done", "kept", "dropped_by_dedup",
                                                 "dropped_by_reason", "failed", "interrupted", "not_started",
                                                 "gpu_seconds_reported")}, indent=2))
except (ValueError, FileNotFoundError) as error:
    sys.exit(f"REFUSED: {error}")
