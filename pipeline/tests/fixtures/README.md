# Test fixtures

- `ipsae_model_10_10.txt`: the real output of the pinned ipSAE scorer (DunbrackLab/IPSAE, commit 6174cf9e) on the real Boltz-2
  structure `examples/verified-modal-run/boltz.cif`, with a synthetic PAE matrix (the recipe is in `test_scoring.py`).
- `esmfold2_config.json`: the `config.json` of `biohub/ESMFold2` at revision 69869f737beffec5294845ede23db5fc0b4f509e (8,214 bytes;
  its SHA-256 is in `docs/eswar/esmfold2-checkpoint-manifest.json`). `tests/support/tiny_esmfold2_fold.py` shrinks it to a tiny
  random-weight model to run the real fold code on a CPU.
