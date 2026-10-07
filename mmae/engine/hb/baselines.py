"""Dual-encoder baselines for D4 (spec section 6.3): prompt softmax and the soft-label linear probe."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mmae.engine.hb import calibrate

WEIGHT_DECAYS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def prompt_logits(emb: np.ndarray, prompts: np.ndarray, logit_scale: float) -> np.ndarray:
    emb = _unit(emb)
    if emb.ndim == 2:
        emb = emb[:, None]
    return logit_scale * emb @ _unit(prompts).T


@dataclass
class SoftProbe:
    weight: np.ndarray
    bias: np.ndarray
    weight_decay: float
    val_nll: float

    def logits(self, x: np.ndarray) -> np.ndarray:
        x = _unit(x)
        if x.ndim == 2:
            x = x[:, None]
        return x @ self.weight + self.bias


def _fit(x: torch.Tensor, targets: torch.Tensor, weight_decay: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    weight = torch.zeros(x.shape[1], targets.shape[1], requires_grad=True)
    bias = torch.zeros(targets.shape[1], requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        log_probs = torch.log_softmax(x @ weight + bias, dim=1)
        loss = -(targets * log_probs).sum(1).mean() + weight_decay * (weight**2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return weight.detach().numpy().astype(np.float64), bias.detach().numpy().astype(np.float64)


def fit_soft_probe(train_x, train_counts, val_x, val_index, val_labels, weight_decays=WEIGHT_DECAYS,
                   steps: int = 200) -> SoftProbe:
    x = torch.tensor(_unit(train_x), dtype=torch.float32)
    counts = np.asarray(train_counts, np.float64)
    targets = torch.tensor(counts / counts.sum(1, keepdims=True), dtype=torch.float32)
    best = None
    for wd in weight_decays:
        weight, bias = _fit(x, targets, wd, steps)
        probe = SoftProbe(weight, bias, wd, float("nan"))
        probe.val_nll = calibrate.nll(calibrate.mixture(probe.logits(val_x), 1.0), val_index, val_labels)
        if best is None or probe.val_nll < best.val_nll:
            best = probe
    return best
