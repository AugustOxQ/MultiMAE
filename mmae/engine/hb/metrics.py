"""Distribution agreement metrics between model and human emotion distributions (spec section 7). Inputs are
(N, 9) arrays of probabilities, one row per painting."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr


def normalise(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    return counts / counts.sum(axis=1, keepdims=True)


def _plogp_terms(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """sum p log2(p / q) per row with 0 log 0 = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(p / q), 0.0)
    return terms.sum(axis=1)


def js_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon distance, base 2 (in [0, 1]), per row."""
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    m = 0.5 * (p + q)
    divergence = 0.5 * _plogp_terms(p, m) + 0.5 * _plogp_terms(q, m)
    return np.sqrt(np.clip(divergence, 0.0, None))


def entropy_bits(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=1)


def kl(h: np.ndarray, m: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """KL(human || model) in nats per row; the model side is clipped at eps (softmax outputs are never zero)."""
    h, m = np.asarray(h, np.float64), np.clip(np.asarray(m, np.float64), eps, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(h > 0, h * np.log(h / m), 0.0).sum(axis=1)


def tvd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(np.asarray(p, np.float64) - np.asarray(q, np.float64)).sum(axis=1)


def entropy_spearman(p: np.ndarray, h: np.ndarray) -> float:
    """Spearman correlation over paintings between model entropy and human entropy."""
    return float(spearmanr(entropy_bits(p), entropy_bits(h)).correlation)


def rank_cs(p: np.ndarray, h: np.ndarray) -> float:
    """Mean over paintings of the Spearman correlation between the model's and the human class proportions;
    paintings whose human vector is constant are skipped."""
    p, h = np.asarray(p, np.float64), np.asarray(h, np.float64)
    keep = h.max(axis=1) > h.min(axis=1)
    rp, rh = rankdata(p[keep], axis=1), rankdata(h[keep], axis=1)
    rp, rh = rp - rp.mean(1, keepdims=True), rh - rh.mean(1, keepdims=True)
    rho = (rp * rh).sum(1) / np.sqrt((rp**2).sum(1) * (rh**2).sum(1))
    return float(rho.mean())
