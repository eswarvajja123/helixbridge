"""One Mosaic smoke test or one Mosaic design on the CURRENT IL-7Ra target (Phase 4, Task 4.1). Runs inside the Modal Mosaic image.

Ported from the team's scripts/mosaic-il7ra.py. The optimizer settings, loss and models are the team's, unchanged; what became a
parameter is the target (current construct 21-239), the hotspots (as indices relative to the construct start), the binder length and the
seed. Inputs come as a JSON file written by modal_mosaic.py from binder_campaign.mosaic_input.build_inputs, which already checked them.

  smoke   one loss and gradient evaluation: proves the stack, the checkpoint and the objective work (no design is made)
  design  75 soft + 50 sharp optimizer steps, then one Boltz-2 prediction of the discrete sequence; writes model.cif and result.json

Boltz-2 is inside the objective, so its refold of the result is related evidence, not an independent check.
"""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from binder_campaign.runner import digest, write_json

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", type=Path, required=True)
parser.add_argument("--mode", choices=["smoke", "design"], required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
spec = json.loads(args.inputs.read_text())
out = args.out
if (out / "result.json").exists():
    raise ValueError("Use a fresh result directory")
out.mkdir(parents=True, exist_ok=True)
started = time.monotonic()

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from mosaic.common import TOKENS
from mosaic.models.boltz2 import Boltz2
from mosaic.losses.boltz2 import load_features_and_structure_writer
import mosaic.losses.structure_prediction as sp
from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
from mosaic.proteinmpnn.mpnn import load_mpnn_sol
from mosaic.optimizers import simplex_APGM

if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
    raise RuntimeError("Exactly one allocated GPU is required")
settings = spec["settings"]
target, length, seed = spec["target_sequence"], spec["binder_length"], spec["seed"]
cache = Path(spec["boltz_cache"])
checkpoint = cache / "boltz2_conf.ckpt"
model = Boltz2(cache_path=checkpoint)
mpnn = load_mpnn_sol()
loaded = time.monotonic()


def features(binder):
    native = json.dumps({"version": 1, "sequences": [
        {"protein": {"id": "A", "sequence": binder, "msa": "empty"}},
        {"protein": {"id": "B", "sequence": target, "msa": spec["msa_path"]}},
    ]})
    return load_features_and_structure_writer(native, cache=cache)


design_features, _ = features("X" * length)
weights = settings["loss_weights"]
loss = model.build_loss(
    loss=weights["binder_target_contact"] * sp.BinderTargetContact(epitope_idx=spec["epitope_idx"])
    + weights["within_binder_contact"] * sp.WithinBinderContact()
    + weights["inverse_folding_sequence_recovery"] * InverseFoldingSequenceRecovery(
        mpnn, temp=jnp.array(settings["mpnn_temperature"]), num_samples=settings["mpnn_samples"]),
    features=design_features, recycling_steps=settings["design_recycles"],
    sampler=model.design_sampler(settings["design_diffusion_steps"]),
)
key = jax.random.key(seed)
initial = jax.nn.softmax(0.5 * jax.random.gumbel(key, (length, 20)), axis=-1)


@eqx.filter_jit
def raw_loss_gradient(objective, probabilities, random_key):
    return eqx.filter_value_and_grad(objective, has_aux=True)(probabilities, key=random_key)


timings = {"load_seconds": loaded - started}
if args.mode == "smoke":
    (value, _), gradient = raw_loss_gradient(loss, initial, key)
    jax.block_until_ready(gradient)
    if not np.isfinite(float(value)) or not np.isfinite(np.asarray(gradient)).all():
        raise ValueError("Nonfinite smoke objective or gradient")
    extra = {"loss": float(value), "gradient_norm": float(jnp.linalg.norm(gradient))}
else:
    t0 = time.monotonic()
    soft, _ = simplex_APGM(loss_function=loss, x=initial, n_steps=settings["soft_steps"], stepsize=settings["soft_stepsize"],
                          momentum=settings["momentum"], key=jax.random.key(seed + 1))
    sharp, _ = simplex_APGM(loss_function=loss, x=soft, n_steps=settings["sharp_steps"], stepsize=settings["sharp_stepsize"],
                           momentum=settings["momentum"], scale=settings["sharp_scale"], key=jax.random.key(seed + 2))
    sequence = "".join(TOKENS[int(index)] for index in np.asarray(sharp.argmax(-1)))
    timings["optimize_seconds"] = time.monotonic() - t0
    if len(sequence) != length or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
        raise ValueError("Invalid designed sequence")
    t1 = time.monotonic()
    discrete_features, writer = features(sequence)      # fresh all-atom features; the design features use unknown side chains
    prediction = model.predict(features=discrete_features, writer=writer, key=jax.random.key(0))
    jax.block_until_ready(prediction.pae)
    timings["predict_seconds"] = time.monotonic() - t1
    pae = np.asarray(prediction.pae)
    if pae.shape != (length + len(target),) * 2 or not np.isfinite(pae).all():
        raise ValueError("Invalid sequence-to-PAE mapping")
    prediction.st.make_mmcif_document().write_file(str(out / "model.cif"))
    np.savez_compressed(out / "raw.npz", pae=pae, plddt=np.asarray(prediction.plddt))
    extra = {"binder_sequence": sequence, "generator_model_iptm": float(prediction.iptm), "model_sha256": digest(out / "model.cif"),
             "raw_sha256": digest(out / "raw.npz")}
timings["total_seconds"] = time.monotonic() - started
write_json(out / "result.json", {"mode": args.mode, "timings": timings, "memory_stats": jax.devices()[0].memory_stats(),
                                 "device": str(jax.devices()[0]), "device_kind": jax.devices()[0].device_kind, "host_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                                 "jax_version": jax.__version__, **extra})
print("Mosaic stage completed:", args.mode, flush=True)
