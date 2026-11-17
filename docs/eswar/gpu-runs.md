# Real GPU runs

What has actually run on a GPU, with where each number came from. Nothing here is a binding result; every design is a rehearsal and not submission eligible.

## 17 Nov 2026 — team's historical pipeline (`modal_app.py`), workspace `eswarvajja10`, run on the laptop

| Step | Command | Outcome |
|---|---|---|
| Stage A prepare | `modal run modal_app.py --stage prepare` | RFD4 and Boltz-2 images built; public checkpoints cached on the volume. |
| Stage A generate | `modal run modal_app.py --stage generate --run-id eswar-seed43-01 --seed 43` | A100-SXM4-40GB, 57.6 s (65.8 s worker), peak 2,564 MiB. Binder (60 aa): `SIQEQLQTILGELHSEVTDSYTDEFKNLLEKALEENDLNVLLEMSNSTLRSKIKDVINSL` |
| Stage A evaluate | `modal run modal_app.py --stage evaluate --run-id eswar-seed43-01` | Boltz-2 + ipSAE: ipTM 0.7752, pTM 0.8467, complex pLDDT 0.8350, max ipSAE 0.3757 (49.8 s; 66.3 s worker). |

Provenance: these numbers were relayed from the laptop session's summary. The raw `generate.json` and `evaluate.json` have **not** been read here, so treat them as unverified until those files are read.

Comparison with the team's verified run (seed 43; ipTM 0.945, ipSAE 0.78, binder `SVQEKLQEIQASLHSKVAKENVDEFKSLLEKANEENDLSVLLQASNPELLSLIKEALNSL`): the binder differs, the scores are lower, and time and memory match. It is a different design from the same seed, not a reproduction. One sample cannot say whether this is ordinary variation between designs or hardware non-determinism.

Findings:
- Windows Git converted line endings to CRLF and broke the hash check on the reference input; fixed with `git config core.autocrlf false` and Unix line endings (now in RUN-GPU.md step 0).
- This was the team's historical target (1-236) and the team's `modal_app.py`, **not** the new `modal_current.py` (current target 21-239). Stages B to E have not run.
