# What has run on a GPU, and what has not

Audit of 17 Nov 2026, from the real run records (`docs/eswar/gpu-runs.md` and `docs/eswar/results/`). "Real GPU" means the code ran on a Modal GPU and its output was read from the result files. Everything else was tested on a CPU (with stand-ins, or real libraries where noted).

## Ran on a real GPU

| Code | How it ran | Evidence |
|---|---|---|
| `modal_current.py` generate | one design, A100-80GB | `eswar-cur-001`, 72.6 s |
| `modal_current.py` batch and `batch.py` engine | six designs (A10 and A100), budget caps, near-duplicate gate on real structures | batch `eswar-b1`: 0 failures, 406.6 GPU s |
| `rfd4-generate.py`, `generation_input.py`, conditioning for hotspot option A | inside the generation runs | same runs |
| `modal_esmfold2.py` prepare and qualify | 26.7 GB hash-verified, one fold, **all three kernel backends** (`fused`, `cuequivariance`, `none`) | `eswar-esm-003`, `eswar-esm-kb-none`, `eswar-esm-kb-cuequivariance`; ipTM 0.9492 / 0.9492 / 0.9493 |
| `modal_evaluate.py` ESMFold2 and Boltz-2 functions, `esm_eval.py`, `scoring.py` (ipSAE run inside both containers) | 12 predictions | batch evaluation `eswar-e1` |
| `evaluate.py` engine: caps, chunking, failure capture, resume with `--retry-failed` | a real download failure, then a real resume | `eswar-e1` (4 failures kept as history) |
| Stage A of the team's own `modal_app.py` | laptop session | summary only; raw files not read |

## CPU only by design (no GPU is involved)

`conditioning.py`, `dedup.py`, `current_candidates.py`, `qc.py`, `diversity.py`, `pool.py`, `novelty.py`, `liabilities.py`, `selection.py`. They ran on real GPU outputs (real structures and scores) where those exist, and are covered by tests.

## Updated after the pool run (18 Nov): paths that have now run for real

Hotspot options B and C; binder length 80; `--length-min/--length-max` (53 residues chosen); an **L40S** and an A100 PCIe for generation; chunk size 6; the near-duplicate gate dropping a real design (across batches); and `--retry-interrupted`, used for real after a call was cut off.

## Still never run on a GPU (or never run for real)

| Path | Status |
|---|---|
| A real Ctrl-C mid-call | never run for real; covered by tests only (a worker restart did cut calls off twice, and both were recovered) |
| Batch `--retry-failed` (the generation engine) | never run for real; the evaluation engine's version was |
| `--include-flagged` in evaluation | never run for real |
| Another GPU type for ESMFold2 (L40S) | never run; A100-80GB only |
| Chunk sizes above 6 | never run |
| A laptop-side (Windows) run of the new code | not run; the Windows line-ending issue is documented |
| A passing export guard | never reached: it needs a person's reviews |

None of these blocks anything; they are the places a surprise could still hide.

## Added 18 Nov

The SolubleMPNN repair (`repair.py`) is built and tested, but the real upstream program has **not been run**: it is third-party code that loads third-party pickled weights and needs `openfold`, and the session's permission check did not allow running it. The wrapper, the redesign rules, the output parser and the selection rule are tested with a stand-in; the upstream output format was read from its source, not observed. `plan` mode ran on the real shortlist. The AlphaFold cross-check is a person's step; only its hand-off package exists.

## Added 18 Nov (Phase 4)

Now run on a real GPU: the whole Mosaic path (image with the pinned JAX stack, smoke test, a design, two batches, scoring, QC, comparison), the generation batch cap being raised on resume to finish a design after an interrupted attempt, and the near-duplicate gate on a second generator. Still not run: **`modal_repair.py` (the native SolubleMPNN repair in a Modal container)**. It is written and tested with a stand-in; the permission check denied running it twice (once in the session, once as a Modal job), so I did not retry or work around it. Running it needs your explicit go-ahead to execute third-party code and weights, and it is one command (Stage P in `RUN-GPU.md`).
