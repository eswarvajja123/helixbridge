"""Plan one on-target generation before any GPU money is spent (Phase 2, Task 2.4).

Pure local helpers used by modal_current.py and scripts/plan-modal-run.py. Nothing here imports Modal,
a GPU library or a model. `local_dry_run` builds the real conditioning files in a temporary folder and runs
scripts/rfd4-generate.py with --dry-run, so every input check that the GPU run would make is made first.
"""
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from .conditioning import write_conditioning
from .generation_input import check_seed

RUN_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}")
OPTIONS = ("A", "B", "C")
MANIFEST_RELATIVE = "docs/eswar/phase1-data/il7ra-manifest.json"
REFERENCE_RELATIVE = "examples/il7ra-rehearsal"


def check_run_id(run_id):
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ValueError("Use a unique short alphanumeric run ID (letters, digits, - and _; at most 64)")
    return run_id


def resolve_hotspots(root, option="", hotspots="", hotspot_source=""):
    """Return (hotspots, source). Default is option A, the provisional historical set."""
    if option and hotspots:
        raise ValueError("Give either an option or a hotspot list, not both")
    if hotspots:
        try:
            values = [int(x) for x in str(hotspots).split(",") if x.strip()]
        except ValueError:
            raise ValueError("Hotspots must be comma-separated integers") from None
        if not values:
            raise ValueError("Hotspot list is empty")
        if not str(hotspot_source).strip():
            raise ValueError("A custom hotspot list needs --hotspot-source (where it came from)")
        return sorted(values), str(hotspot_source).strip()
    option = (option or "A").upper()
    if option not in OPTIONS:
        raise ValueError("Option must be one of " + ", ".join(OPTIONS))
    audit = json.loads((Path(root) / "docs/eswar/hotspot-options" / option / "audit.json").read_text())
    return list(audit["hotspots_uniprot"]), audit["hotspot_source"]


def resolve_length(length=60, length_min=0, length_max=0):
    """Return [low, high]; equal values mean an exact length. Limits are the competition's 40-100."""
    if length_min or length_max:
        if not (length_min and length_max):
            raise ValueError("Give both --length-min and --length-max")
        low, high = length_min, length_max
    else:
        low = high = length
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (low, high)):
        raise ValueError("Binder length must be whole numbers")
    if not 40 <= low <= high <= 100:
        raise ValueError("Binder length must be within 40-100 with min <= max")
    return [low, high]


def build_plan(root, *, run_id="", seed=43, option="", hotspots="", hotspot_source="", length=60, length_min=0, length_max=0):
    """Validate and normalise everything a generation run needs; raises ValueError on any problem."""
    hotspot_list, source = resolve_hotspots(root, option, hotspots, hotspot_source)
    return {"run_id": check_run_id(run_id) if run_id else "", "seed": check_seed(seed),
            "hotspots": hotspot_list, "hotspot_source": source,
            "binder_length": resolve_length(length, length_min, length_max)}


def local_dry_run(root, plan):
    """Run the real input checks of rfd4-generate.py on this machine (no GPU, no model) and return its plan."""
    root = Path(root)
    low, high = plan["binder_length"]
    with tempfile.TemporaryDirectory() as temp:
        condition = Path(temp) / "conditioning"
        write_conditioning(root / REFERENCE_RELATIVE, condition, plan["hotspots"], low if low == high else (low, high),
                           hotspot_source=plan["hotspot_source"])
        done = subprocess.run([sys.executable, str(root / "pipeline/scripts/rfd4-generate.py"), "--dry-run",
                               "--reference-dir", str(condition), "--manifest", str(root / MANIFEST_RELATIVE),
                               "--out", str(Path(temp) / "out"), "--seed", str(plan["seed"])],
                              capture_output=True, text=True)
        if done.returncode != 0:
            raise ValueError("Dry run refused the inputs: " + (done.stderr.strip().splitlines() or ["unknown error"])[-1])
        return json.loads(done.stdout)
