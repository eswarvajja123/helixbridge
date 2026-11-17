# Running on a GPU — your laptop, your Modal account

Nothing in this repo has run on a GPU yet. These steps are how it gets proven. Run them on your own machine; never paste a token anywhere (not in chat, not in a file in this repo).

## 0. Get the full project (team repo + my files on top)

My files live in your repo; the team's `modal_app.py`, `binder_campaign/` and examples live in the team repo. Put them together in a **separate folder** (do not push from it):

```bash
git clone https://github.com/RishyanthReddy/helixbridge helixbridge-team
cd helixbridge-team
git remote add eswar https://github.com/eswarvajja123/helixbridge
git fetch eswar main
git checkout eswar/main -- pipeline docs/eswar requirements-eswar.txt modal_current.py
```

That copies only my files into the team's folder (and stages them). **Do not commit or push from `helixbridge-team`.** If your repo is private, git asks for your GitHub login.

## 1. Python setup

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-dev.txt -r requirements-eswar.txt
```

(On Mac/Linux: `python3.12 -m venv .venv` and `source .venv/bin/activate`.)

## 2. Free checks first (no Modal, no GPU, no cost)

```bash
cd pipeline
python -m unittest discover -s tests
cd ..
```

Everything must pass. Then check one planned run exactly as the GPU run will:

```bash
python pipeline/scripts/plan-modal-run.py --option A --seed 43
```

It prints `"input_checks": "passed"` and the 14 generation settings. If it prints `REFUSED`, stop and send me the line.

## 3. Modal login and balance

```bash
modal token info
modal billing summary
```

If `token info` says the token is missing, run `modal token new` and approve in the browser. Write down the balance.

## Stage A — prove Modal + RFD4 + Boltz work (the team's historical run)

```bash
modal run modal_app.py --stage prepare
```

```bash
modal run modal_app.py --stage generate --run-id eswar-seed43-001 --seed 43
```

```bash
modal run modal_app.py --stage evaluate --run-id eswar-seed43-001
```

Compare with the team's verified run (`examples/verified-modal-run/summary.json`): 60-aa binder, ipTM about 0.945, max ipSAE about 0.78, generation about 58 s. Small differences are normal.

## Stage B — Task 2.4: one on-target design with the new script

Only after Stage A passes. You already ran the free local check in step 2 (`plan-modal-run.py`); the GPU run repeats the same checks before it spends anything. Do not use `modal run ... --stage plan` as a free check: starting a Modal app can trigger an image build, which may be billed.

```bash
modal run modal_current.py --stage generate --run-id eswar-cur-001 --seed 43 --option A
```

- `--option A|B|C` picks a hotspot set from `docs/eswar/hotspot-decision.md` (A is the provisional historical set). All three are **unreviewed** until the epitope is decided.
- `--length 60`, or `--length-min 50 --length-max 80` (with a range the model chooses the length).
- To force one GPU type, set `HELIXBRIDGE_GPU` first (for example `A100-80GB`); default is A100-80GB, then L40S, then A10G.
- The first run builds the container image, which takes longer; later runs reuse it. Check billing for the build time too.

Results land in `results\eswar-cur-001\` (`generate-current.json` and `generate-current.tar.gz`). In the JSON check: `gpu`, `worker_seconds`, `audit.binder_sequence` (60 letters), `record.qc_reasons` (should be `[]`), `record.submission_eligible` (must be `false`).

```bash
modal billing summary
```

Write the balance difference next to the run in `docs/experiment-template.csv`.

## What to send me
- The text of `generate-current.json` (it holds no token) and your balance before and after.
- If anything failed: the **last 30 lines** of the error.
- **Never** send `~/.modal.toml` or any token.

## If it fails
The team's historical run hit three problems that are already fixed in the code: an RFD4 import error, a missing system library, and an L40S queue stall (the GPU list falls back to other types). A new failure is useful information; send it as it is.
