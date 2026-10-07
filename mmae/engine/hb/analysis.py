"""Analysis of the per-run encode.npz files written by scripts/hb_encode.py: loader, decoder readouts and the
pre-wave-2 gate (spec sections 6.1, 6.2, 13)."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from mmae.data.artelingo import heldout_paintings
from mmae.engine.hb import calibrate, data, metrics


def load_encoded(folder: str | Path) -> dict:
    """{"meta": dict, **arrays}; float16 arrays are cast to float32."""
    folder = Path(folder)
    out: dict = {"meta": json.loads((folder / "meta.json").read_text())}
    with np.load(folder / "encode.npz", allow_pickle=False) as z:
        for key in z.files:
            arr = z[key]
            out[key] = arr.astype(np.float32) if arr.dtype == np.float16 else arr
    return out


def decoder_readouts(enc: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(AL-28 logits (N, M, 9), validation logits (P, M, 9)) per readout; {} for runs without a decoder."""
    if "al28_dec_views" not in enc:
        return {}

    def flat(x: np.ndarray, views: int | None = None) -> np.ndarray:
        x = x if views is None else x[:, :views]
        return x.reshape(x.shape[0], -1, x.shape[-1])

    out = {}
    for name, views in (("views16", None), ("views4", 4), ("views1", 1)):
        out[name] = (flat(enc["al28_dec_views"], views), flat(enc["val_dec_views"], views))
    out["full"] = (flat(enc["al28_dec_full"]), flat(enc["val_dec_full"]))
    return out


def primary_readout(meta: dict) -> str:
    """Each arm is read on its training input (spec 6.1)."""
    return "views16" if meta["mlm_image_source"] == "masked" else "full"


def prior_from(enc: dict, annotations_dir: str | Path | None = None) -> np.ndarray:
    if "train_counts" in enc:
        total = enc["train_counts"].astype(np.float64).sum(0)
        return total / total.sum()
    if annotations_dir is None:
        raise ValueError("no train_counts in the encode file and no annotations_dir to compute the prior from")
    return data.prior(annotations_dir, heldout_paintings())


def gate(enc: dict, prior: np.ndarray) -> dict:
    name = primary_readout(enc["meta"])
    al28_logits, val_logits = decoder_readouts(enc)[name]
    index, labels = enc["val_index"], enc["val_labels"]
    temperature = calibrate.fit_temperature(val_logits, index, labels)
    p_al28 = calibrate.mixture(al28_logits, temperature)
    p_val = calibrate.mixture(val_logits, temperature)
    val_nll = calibrate.nll(p_val, index, labels)
    prior_nll = calibrate.nll(np.tile(prior, (len(val_logits), 1)), index, labels)
    prior_al = np.tile(prior, (len(p_al28), 1))
    human = metrics.normalise(enc["al28_counts"].astype(np.float64))
    jsd_to_prior = float(metrics.js_distance(p_al28, prior_al).mean())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = metrics.entropy_spearman(p_al28, human)  # NaN for a constant entropy vector
    checks = {
        "sums_to_one": bool(np.allclose(p_al28.sum(1), 1, atol=1e-5)),
        "temperature_in_range": bool(0.25 <= temperature <= 4),
        "val_nll_below_prior": bool(val_nll < prior_nll),
        "not_collapsed_to_prior": bool(jsd_to_prior > 0.02),
    }
    return {
        "readout": name, "temperature": float(temperature), "val_nll": float(val_nll), "prior_nll": float(prior_nll),
        "jsd_to_prior": jsd_to_prior,
        "jsd_human": float(metrics.js_distance(p_al28, human).mean()),
        "prior_jsd_human": float(metrics.js_distance(prior_al, human).mean()),
        "entropy_spearman": float(rho), "checks": checks, "passed": all(checks.values()),
    }
