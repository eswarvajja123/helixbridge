# Phase 1 data — how it was made and how to reproduce it

Made 16 Nov 2026. All inputs are public (UniProt, the ColabFold MMseqs2 service, RCSB). No candidate sequence was sent anywhere.

| File | What it is | Made by |
|---|---|---|
| `il7ra-manifest.json` | Current official target: P16871-1, 21–239, 219 aa | `prepare-targets` (Task 1.1) |
| `target.a3m`, `msa-manifest.json`, `msa-ticket.json` | Target-only MSA, 1022 rows, hashed | `fetch-target-msa.py` (Task 1.2) |
| `reference-vs-3DI3.json` | Shipped reference vs 3DI3 chain B, plus the natural IL-7 interface | `compare-reference-to-pdb.py` (Task 1.3) |

## Reproduce (from the full project's `pipeline/` folder)

```bash
python -m binder_campaign prepare-targets --config config/targets-2026-11-13.json --output data/current-targets
python scripts/fetch-target-msa.py --manifest data/current-targets/il7ra/manifest.json --output data/current-targets/il7ra/msa
curl -o 3DI3.pdb https://files.rcsb.org/download/3DI3.pdb
python scripts/compare-reference-to-pdb.py --pdb 3DI3.pdb --chain B --partner-chain A \
  --hotspots 100,101,102,158,159,161,211,212,213,214 \
  --reference-cif ../examples/il7ra-rehearsal/target.cif \
  --manifest data/current-targets/il7ra/manifest.json --output data/current-targets/il7ra/reference-vs-3DI3.json
```

The sequence hash in `il7ra-manifest.json` (`ecdc20f8…`) must match yours. The MSA is produced by an online service, so its row count or hash may differ slightly on a re-run; the query sequence (first row) must be identical.
