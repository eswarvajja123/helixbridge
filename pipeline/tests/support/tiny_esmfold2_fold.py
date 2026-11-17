"""Run the REAL esm ESMFold2 fold code on a CPU with a tiny random-weight model (for tests/test_real_esm.py).

Needs a Python with the real `esm==3.4.1.post1` and `torch==2.11.0` installed. It builds the real model class from the real config
shrunk to a few small layers (random weights, so the numbers mean nothing), then uses the real input builder, the real MSA
featuriser, the real structure and confidence heads and the real CIF writer on the real target and the real verified design.
The 417 MB chemical-components file is replaced by an empty one, which the library accepts for protein-only inputs.
"""
import argparse
import copy
import importlib.metadata as metadata
import json
import pickle
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--target-manifest", type=Path, required=True)
parser.add_argument("--msa", type=Path, required=True)
parser.add_argument("--binder", required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.manual_seed(0)

tiny = copy.deepcopy(json.loads(args.config.read_text()))
tiny["esmc_config"].update(hidden_size=128, intermediate_size=512, num_hidden_layers=1, num_attention_heads=2,
                           num_key_value_heads=2, head_dim=64)           # the library requires intermediate_size 512 for hidden 128
tiny["lm_d_model"], tiny["lm_num_layers"], tiny["folding_trunk_num_hidden_layers"] = 128, 1, 2
tiny["msa_encoder"]["num_hidden_layers"] = 1
(args.out / "model").mkdir(exist_ok=True)
(args.out / "model/config.json").write_text(json.dumps(tiny))
(args.out / "ccd").mkdir(exist_ok=True)
with open(args.out / "ccd/ccd.pkl", "wb") as stream:
    pickle.dump({}, stream)

from esm.models.esmfold2 import ESMFold2InputBuilder, EsmFold2Model, ProteinInput, StructurePredictionInput
from esm.models.esmfold2.config import EsmFold2Config
from esm.utils.msa import MSA

model = EsmFold2Model(EsmFold2Config.from_pretrained(str(args.out / "model"))).eval()
model.set_kernel_backend(None)                                         # plain PyTorch: there are no GPU kernels on a CPU
target = json.loads(args.target_manifest.read_text())["sequence"]
msa = MSA.from_a3m(args.msa, remove_insertions=True, max_sequences=1024)
spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=target, msa=msa),
                                          ProteinInput(id="B", sequence=args.binder, msa=None)])
settings = {"num_loops": 2, "num_sampling_steps": 4, "num_diffusion_samples": 1, "seed": 0, "lm_dropout": 0.3,
            "msa_max_depth": 64, "msa_column_mask_rate": 0.1}
started = time.monotonic()
with torch.no_grad():
    result = ESMFold2InputBuilder(ccd_cache=args.out / "ccd").fold(model, spi, **settings)
seconds = time.monotonic() - started
pae, plddt = np.asarray(result.pae), np.asarray(result.plddt)
(args.out / "model.cif").write_text(result.complex.to_mmcif())
(args.out / "pae.json").write_text(json.dumps({"pae": pae.tolist()}))
(args.out / "result.json").write_text(json.dumps({
    "pae_shape": list(pae.shape), "plddt_shape": list(plddt.shape), "ptm": float(result.ptm), "iptm": float(result.iptm),
    "plddt_min": float(plddt.min()), "plddt_max": float(plddt.max()), "plddt_mean": float(plddt.mean()), "msa_depth": msa.depth,
    "fold_seconds": round(seconds, 1), "versions": {p: metadata.version(p) for p in ("torch", "esm", "biotite", "numpy")}}, indent=2))
print("done")
