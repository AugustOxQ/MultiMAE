"""Temperature scaling for the emotion readouts (spec section 6.2): one temperature per (arm, seed, readout),
minimising the mean NLL of individual validation labels; applied to logits before averaging over views and
caption lengths."""
from __future__ import annotations

import numpy as np
from scipy.special import log_softmax, softmax

LOG_T_GRID = np.linspace(-3.0, 3.0, 241)


def mixture(logits: np.ndarray, temperature: float) -> np.ndarray:
    """(N, M, 9) logits over M samples (views x lengths) -> (N, 9) mean of softmax(logits / T)."""
    return softmax(np.asarray(logits, np.float64) / temperature, axis=-1).mean(axis=1)


def nll(probs: np.ndarray, painting_index: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of individual labels; label i belongs to painting painting_index[i]."""
    picked = probs[painting_index, labels]
    return float(-np.log(np.clip(picked, 1e-12, None)).mean())


def fit_temperature(logits: np.ndarray, painting_index: np.ndarray, labels: np.ndarray) -> float:
    """Grid search over log T in [-3, 3] (step 0.025), then a local refinement to 1e-4 in log T."""
    def loss(log_t: float) -> float:
        return nll(mixture(logits, float(np.exp(log_t))), painting_index, labels)

    losses = [loss(x) for x in LOG_T_GRID]
    best = int(np.argmin(losses))
    lo, hi = LOG_T_GRID[max(best - 1, 0)], LOG_T_GRID[min(best + 1, len(LOG_T_GRID) - 1)]
    for _ in range(40):  # golden-section search on [lo, hi]
        a, b = hi - 0.618 * (hi - lo), lo + 0.618 * (hi - lo)
        if loss(a) < loss(b):
            hi = b
        else:
            lo = a
        if hi - lo < 1e-4:
            break
    return float(np.exp(0.5 * (lo + hi)))
