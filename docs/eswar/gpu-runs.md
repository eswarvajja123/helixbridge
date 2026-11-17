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

## 17 Nov 2026 — Task 2.4 with the new code (`modal_current.py`), run from the cloud session

Run id `eswar-cur-001`, seed 43, option A (hotspots 100-102, 158, 159, 161, 211-214; **unreviewed**, `provisional_historical`). Read from the real result file `generate-current.json`, not from a summary.

| Check | Result |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| Time | 72.6 s generation, 86.4 s worker; peak memory 2,564 MiB |
| Binder | 60 aa, `SIQEQLQTILGELHSEVTDSYTDEFKNLLEKALEENDLNVLLEMSNSTLRSKIKDVINSL` |
| Generator | RFD4-Proteina at the pinned commit ddd934f7, `recycle_strategy=thinking_high`, seed 43 |
| Record | candidate `il7ra-rfd4-9bf9771de68d`; `qc_reasons` empty; `submission_eligible` false; `screen_status` not_run; target 21-239 |
| Cost | about $0.05 of credit (credits line 2.01 to 2.06; billed cost $0.00) |

**The binder is identical to the one the laptop run produced with the team's `modal_app.py`** (A100-40GB, same seed). So seed 43 reproduced across two GPU sizes and across both code paths (the team's and mine), which answers the question left open above for these conditions. It also means the difference from the team's recorded verified design (`SVQEKLQE...`) comes from something else in how that run was made, not from GPU size; that is not yet explained.

Task 2.4's finish line (one on-target design on a GPU, cost logged) is reached. Nothing here says the design binds.

### How a cloud session reaches Modal (found 17 Nov)

The first attempts failed with "Could not connect to the Modal server". The cause was not a protocol limit, as an earlier environment note claimed: the Modal client needs `pip install 'modal[api-proxy-support]'` (it adds `python-socks`) to use the session's proxy, and the environment's network allowlist needs both `api.modal.com` and `input-plane.us-east.modal.com`. The second host is where function calls go; without it the image builds but the run is refused with `403 Forbidden`.
