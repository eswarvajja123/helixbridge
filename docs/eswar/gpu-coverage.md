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

**Update (19 Nov): superseded below.** The SolubleMPNN repair has since been run for real (see "Added 19 Nov"). The AlphaFold-Multimer cross-check is also run on a cloud GPU as a measurement; Stefanos's own check is still a person's step.

## Added 18 Nov (Phase 4)

Now run on a real GPU: the whole Mosaic path (image with the pinned JAX stack, smoke test, a design, two batches, scoring, QC, comparison), the generation batch cap being raised on resume to finish a design after an interrupted attempt, and the near-duplicate gate on a second generator. Still not run at that point: `modal_repair.py` (see the next section).

## Added 19 Nov

Now run on a real GPU or container, on the `rishyanthreddy101` Modal account unless noted: the native SolubleMPNN repair (`modal_repair.py`, upstream commit `26ec57ac`, hash-checked weights; attempt 1 failed on an invalid `--device` option, attempt 2 ran for 8 designs and the 12 variants were scored by ESMFold2 and Boltz-2 and QC'd); AlphaFold-Multimer (`modal_alphafold.py`) for the 20 shortlisted designs, the 20 shuffled controls, 12 repaired variants and 7 further designs, after three failed attempts whose cause was ColabFold's default CUDA unified memory (fixed with `--disable-unified-memory`); the shuffled negative controls (scoring and QC); and the new AlphaFold gate in the selection tool. Weights for ESMFold2 and Boltz-2 and the AlphaFold parameters were downloaded once into the new account. Still never run: everything in the table above, plus the AlphaFold run with more than one seed or more than 2 models, the repaired variants on any predictor other than ESMFold2, Boltz-2 and AlphaFold, and anything on a laptop (Windows). Still a person's step: the three human reviews, Stefanos's own AlphaFold check, and the epitope decision.
