# Running on a GPU — your laptop, your Modal account

Nothing in this repo has run on a GPU yet. These steps are how it gets proven. Run them on your own machine; never paste a token anywhere (not in chat, not in a file in this repo).

## 0. Get the full project (team repo + my files on top)

My files live in your repo; the team's `modal_app.py`, `binder_campaign/` and examples live in the team repo. Put them together in a **separate folder** (do not push from it):

```bash
git clone https://github.com/RishyanthReddy/helixbridge helixbridge-team
cd helixbridge-team
git config core.autocrlf false
git remote add eswar https://github.com/eswarvajja123/helixbridge
git fetch eswar main
git checkout eswar/main -- pipeline docs/eswar requirements-eswar.txt modal_current.py modal_esmfold2.py modal_evaluate.py
```

**On Windows, run `git config core.autocrlf false` first (it is the first line above).** Several input files are checked against a SHA-256, and Git's default of converting line endings to CRLF changes their bytes: the first laptop run failed with `Reference input changed after preparation` for exactly this reason (17 Nov). If you already cloned, set the option and re-run the `git checkout` line so the files are rewritten with Unix line endings.

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

## 2b. Running from a cloud Claude Code session instead of your laptop

It works, with three settings (all found 17 Nov): install the Modal client with proxy support (`pip install 'modal[api-proxy-support]'`, which adds `python-socks`), and allow these hosts in the environment's network settings: `api.modal.com`, `input-plane.us-east.modal.com` (function calls), `clobber.modal-storage.com` and `duot2hadvvkdh.cloudfront.net` (large results and volume downloads; the second is an opaque CDN name Modal may change). Without the last two, a GPU job can finish and its result still fail to download. Credentials belong in the environment's variables (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`), never in chat.

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

Compare with the team's verified run (`examples/verified-modal-run/summary.json`): 60-aa binder, ipTM about 0.945, max ipSAE about 0.78, generation about 58 s. **Do not expect the same design.** Generation is stochastic, and the first laptop run (17 Nov, A100-40GB, seed 43) produced a different 60-aa binder with ipTM 0.775 and max ipSAE 0.376; the run time and memory matched (57.6 s, 2,564 MiB). What must hold is that all three steps finish, the binder is 60 residues and the numbers are finite. Whether seed 43 is reproducible on one GPU type is not yet known: repeat the generate step once with a new `--run-id` and compare sequences.

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

## Stage C — Task 2.5: a small budget-capped batch

Only after Stage B passes. Plan it for free first; it prints every design, the budget, and the exact Modal command to run:

```bash
python pipeline/scripts/run-il7ra-generation.py plan --batch-id eswar-b1 --seeds 43,44,45 --lengths 50,60,70 --option A --max-designs 6 --max-gpu-seconds 900
```

Then run the command it prints (it looks like this):

```bash
modal run modal_current.py --stage batch --batch-id eswar-b1 --seeds 43,44,45 --lengths 50,60,70 --option A --max-designs 6 --max-gpu-seconds 900 --max-failures 2
```

How the budget works:
- `--max-designs` and `--max-gpu-seconds` are **required**. The batch stops before launching a design once either cap is reached, and after `--max-failures` failures.
- The GPU seconds are the function's own count. Modal bills container time, which is longer, so set the cap lower than your real limit and check `modal billing summary`.
- Designs run one after another, so a small batch takes several minutes.

If it stops or you press Ctrl-C, run the **same command again** to resume: finished designs are skipped. It refuses a changed configuration (only the caps may change). Failed designs are not retried unless you add `--retry-failed`; a design interrupted mid-call is reported and not relaunched unless you add `--retry-interrupted` (it may already have been billed).

Results are in `results\batch-eswar-b1\`: `batch-report.json` (counts, failures, GPU seconds), `candidates.json` (every design, including ones dropped as near-duplicates, each unreviewed and not eligible), `state.json`, and one folder per design. To print the summary again without launching anything:

```bash
python pipeline/scripts/run-il7ra-generation.py report --out results/batch-eswar-b1
```

## Stage D — Task 3.1: qualify ESMFold2 (the competition's screening model)

Independent of Stages B and C; do it after Stage A. ESMFold2 has never run on Modal, so this checks that it loads, folds one complex, and how much memory and time it needs. **It says nothing about how accurate ESMFold2 is.**

```bash
modal run modal_esmfold2.py --stage prepare
```

This downloads the pinned checkpoint (`biohub/ESMFold2` at one fixed revision, 11 files, 24.9 GiB) into the shared model Volume and checks **every file's size and SHA-256** against `docs/eswar/esmfold2-checkpoint-manifest.json`. It is a CPU job and takes a while. The first run also builds the image (PyTorch 2.11.0 with CUDA 13 libraries, several GB; every dependency is pinned in `docs/eswar/esm-image-constraints.txt`, 113 packages resolved on 17 Nov 2026); check billing for the build. Add `--skip-hash` only for a quick size-only check.

```bash
modal run modal_esmfold2.py --stage qualify --run-id eswar-esm-001
```

This folds the verified 60-residue design against the current IL-7Rα target (21-239) with its target-only MSA, using the team's local approximation of the screen (20 loops, 200 sampling steps, one sample, seed 0, MSA depth 1024), on an A100-80GB (else an L40S; the 24 GB A10G is left out because ESMFold2 reached about 26 GiB on the team's cluster). Results: `results\eswar-esm-001\esmfold2-qualify.json`. Check `gpu`, `load_seconds`, `fold_seconds`, `peak_reserved_mib`, `ptm`, `iptm` and `versions`. It counts as qualified when the fold finished and the output validated (the run stops with an error otherwise).

If the fast `fused` kernels fail (for example a Triton compile error), retry with `--kernel-backend none` (slower, pure PyTorch) or `--kernel-backend cuequivariance`. A run with a different backend is recorded as such and is not guaranteed to match the team's `fused` runs.

Known risks: the team's first cluster run failed because Triton could not find `Python.h`, so the image build now proves the headers exist and a C compiler can use them; if that build step fails, send me the error. Also untested: PyTorch's CUDA 13.0 libraries on Modal's hosts, and how long the 25 GB load from the Volume takes.

## Stage E — Task 3.2: score many candidates with ESMFold2 and Boltz-2

Only after Stage C (you need a `candidates.json` from a batch) and after **both** weight downloads in this workspace: `modal run modal_app.py --stage prepare` (Boltz-2, Stage A) and `modal run modal_esmfold2.py --stage prepare` (ESMFold2, Stage D). Prefer to run Stage D's single `qualify` fold first, so you know what one ESMFold2 fold costs.

Plan it for free first. It repeats the container's input checks on your machine and prints the exact command:

```bash
python pipeline/scripts/evaluate-candidates.py plan --candidates results/batch-eswar-b1/candidates.json --eval-id eswar-e1 --max-calls 6 --max-gpu-seconds 1800
```

Then run the command it prints (it looks like this):

```bash
modal run modal_evaluate.py --stage evaluate --candidates results/batch-eswar-b1/candidates.json --eval-id eswar-e1 --predictors esmfold2,boltz2 --chunk-size 4 --max-calls 6 --max-gpu-seconds 1800 --max-failures 3
```

How it works:
- Only candidates with **no QC reasons** are scored, so GPU time is not spent on near-duplicates; the skipped ones are listed with their reasons (`--include-flagged` overrides this).
- Candidates are scored against the **current target (21-239)** with its target-only MSA. Both predictors report ipTM, pTM and ipSAE (PAE cutoff 10, distance cutoff 10). ESMFold2 uses the team's local approximation of the organizer screen; Boltz-2 uses the team's own `boltz-predict.py`.
- Each call handles `--chunk-size` candidates. ESMFold2 loads its ~25 GB model **once per call**, so a larger chunk loads it less often (but a call must finish within its time limit; the maximum is 20).
- `--max-calls` and `--max-gpu-seconds` are **required**; it also stops after `--max-failures` failed scores. GPU seconds are the function's own count and Modal bills more (loading, start-up), so keep the cap below your real limit and check `modal billing summary`.
- A failure for one candidate never discards the others. Run the **same command again** to resume: finished pairs are skipped, and a changed candidate file or setting is refused. Failed pairs are retried only with `--retry-failed`; a call interrupted mid-way is reported, not relaunched, unless you add `--retry-interrupted`.
- If the fast `fused` ESMFold2 kernels fail, add `--kernel-backend none` (or `cuequivariance`). The backend is part of the run's identity.

Results are in `results\eval-eswar-e1\`: `evaluations.json` (your candidate records with the raw scores attached), `scores.csv` (one row per candidate and predictor), `comparison.csv` (one row per candidate, the two predictors side by side plus their plain difference; rows stay in the order of your candidate file, never sorted by score), `evaluation-report.json` (including `pool_summary`: min, median and max of each raw metric over this pool), `state.json`, and the structures and PAE files per candidate. To print the summary again without launching anything:

```bash
python pipeline/scripts/evaluate-candidates.py report --out results/eval-eswar-e1
```

**These are raw model outputs.** Nothing is ranked, thresholded or combined (`comparison.csv` and `pool_summary` only lay the numbers side by side; the team's public pilot found negative controls scoring as high as binders, so no pass/fail line is supportable), ESMFold2 and Boltz-2 numbers are not on one scale, and the two models disagreed even on the natural IL-7/IL-7Rα pair in the team's earlier control. No score is a binding prediction, and `screen_status`, `novelty_review` and `submission_eligible` are left exactly as they were.

## Stage F — Task 3.3: structure QC (free, CPU, no Modal)

After Stage E. It reads the structures the earlier stages saved (the RFD4 generator pose and the ESMFold2 / Boltz-2 refolds) and checks each candidate: binder length 40–100, exactly one binder chain, close heavy-atom pairs (clash, below 2 Å), contact with the target, contact with a hotspot the design was conditioned on, and binder pose shift (generator vs each refold, and ESMFold2 vs Boltz-2, after aligning on the shared reference core 37–231).

```bash
python pipeline/scripts/qc-candidates.py --evaluation results/eval-eswar-e1 --batch-dir results/batch-eswar-b1 --out results/qc-eswar-q1
```

It writes `qc-candidates.json` (your records with `qc_reasons` filled in and a `qc` block of measurements; the originals are never changed), `qc.csv` and `qc-report.json` into the new `--out` folder. Each candidate is `flagged`, `no_flags` or `incomplete` (a structure is missing, so it is never reported clean).

The limits are the team's own provisional ones (zero close pairs, pose within 2.5 Å of the generated pose and within 3.0 Å between the two refolds); the team recorded them as a consistency screen, not a validated threshold, and they are options on the command (`--max-close-pairs`, `--max-shift-generator`, `--max-shift-between`, `--min-contact-residues`, `--no-hotspot-requirement`). **No flags does not mean a good design**: the team's pilot had high-scoring non-binders with zero close pairs.

## Stage G — Task 3.4: diversity report (free, CPU, no Modal)

```bash
python pipeline/scripts/diversity-report.py --records results/qc-eswar-q1/qc-candidates.json --evaluation results/eval-eswar-e1 --batch-dir results/batch-eswar-b1 --out results/diversity-eswar-d1
```

For every pair of designs, within each structure source (generator, ESMFold2, Boltz-2): fold similarity (TM-score of the binder backbones, with an 80 % coverage rule), site similarity (Jaccard overlap of the target residues each binder touches) and sequence identity; then groups at TM 0.5 / 0.6 / 0.7 and site Jaccard 0.3 / 0.5 / 0.7, as greedy complete-link and as connected components (both are reported, as the team did). Records with QC flags are left out unless you add `--include-flagged`. Output: `diversity.json`, `pairs.csv`, `groups.csv`.

Any pool works: `--pool pool.json` with `{"target_sequence": ..., "positions": [...], "members": [{"id", "binder_sequence", "structure"}]}`. The TM-scores come from `tmtools` (TM-align), which matched the real USalign binary on 28 real backbone pairs (`pipeline/tests/fixtures/usalign_reference.json`), so no USalign build or container change is needed. The groups are descriptive, not protein families or binding predictions, and complete-link grouping depends on member order (sorted by id).

## Optional: re-check the real ESMFold2 library on your machine (free, CPU)

The test suite normally uses stand-ins for `esm`, `torch` and `huggingface_hub`. `pipeline/tests/test_real_esm.py` runs the **real** `esm` fold code with a tiny random-weight model instead (no GPU, no weights download, about 90 s), and checks that my validator and the real pinned ipSAE accept what it writes. It is skipped unless you point it at a Python that has the real libraries:

```bash
python -m venv esm-check
esm-check/bin/pip install -c docs/eswar/esm-image-constraints.txt torch==2.11.0 esm==3.4.1.post1 biopython==1.88 biotite==1.6.0 huggingface-hub
HELIXBRIDGE_ESM_PYTHON=esm-check/bin/python python -m unittest tests.test_real_esm
```

(On Windows use `esm-check\Scripts\python.exe`; run the last command from `pipeline/`. Set `HELIXBRIDGE_IPSAE` to a checkout of the pinned ipSAE to include the real scorer.) This proves the code path, **not** that the trained weights, the GPU `fused` kernels or the 25 GB load work on Modal; only Stage D proves that.

## What to send me
- The text of `generate-current.json` (Stage B) `batch-report.json` (Stage C) or `esmfold2-qualify.json` (Stage D) or `evaluation-report.json` plus `scores.csv` (Stage E); none holds a token. Also your balance before and after.
- If anything failed: the **last 30 lines** of the error.
- **Never** send `~/.modal.toml` or any token.

## If it fails
The team's historical run hit three problems that are already fixed in the code: an RFD4 import error, a missing system library, and an L40S queue stall (the GPU list falls back to other types). A new failure is useful information; send it as it is.
