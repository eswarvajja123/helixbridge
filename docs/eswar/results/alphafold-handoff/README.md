# AlphaFold cross-check: inputs for 20 shortlisted designs

Task 5.4 of the plan. A person runs AlphaFold on these; nothing here has been run, and no review has been recorded.

- Target: IL-7R alpha, UniProt P16871-1 residues 21-239 (219 aa), chain A. Binder: chain B.
- `alphafold-inputs.fasta` has one complex per entry (`TARGET:BINDER`). `colabfold-batch.csv` is the same in ColabFold's batch format.
- Fill in `alphafold-results-template.csv` yourself (tool and version, pTM, ipTM, ipSAE, binder RMSD to the generated pose, whether the
  interface looks as designed). Please also keep the raw AlphaFold outputs.
- Compare against what the two earlier predictors (ESMFold2, Boltz-2) said, but record what AlphaFold says, not what the others said.
- When you have read the results and decided, a person sets `independent_prediction_status` to `reviewed` for the design (see
  `docs/eswar/RUN-GPU.md`, Stage L). Nothing here sets it.
