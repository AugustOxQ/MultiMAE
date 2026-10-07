"""Hierarchical bootstrap and the pre-registered D4 decision rule (spec section 7)."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from mmae.engine.hb import metrics

COMPARATORS = ("parcap", "probe")
METRICS = ("jsd", "entropy_spearman")
BETTER = {"jsd": -1, "entropy_spearman": +1}  # sign of (ML-80 - comparator) that favours ML-80


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def two_sided_p(diffs: np.ndarray) -> float:
    return float(min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))))


def _spearman_rows(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx**2).sum() * (ry**2).sum()))


def paired_bootstrap(a, b, human, metric: str, B: int = 10000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = human.shape[0]
    if metric == "jsd":
        per_a = [metrics.js_distance(x, human) for x in a]
        per_b = [metrics.js_distance(x, human) for x in b]

        def stat(per, seeds, idx):
            return float(np.mean([per[s][idx].mean() for s in seeds]))
    elif metric == "entropy_spearman":
        human_entropy = metrics.entropy_bits(human)
        per_a = [metrics.entropy_bits(x) for x in a]
        per_b = [metrics.entropy_bits(x) for x in b]

        def stat(per, seeds, idx):
            return float(np.mean([_spearman_rows(per[s][idx], human_entropy[idx]) for s in seeds]))
    else:
        raise ValueError(metric)
    full = np.arange(n)
    diff = stat(per_a, range(len(a)), full) - stat(per_b, range(len(b)), full)
    draws = np.empty(B)
    for r in range(B):
        idx = rng.integers(0, n, n)
        sa = rng.integers(0, len(a), len(a))
        sb = rng.integers(0, len(b), len(b))
        draws[r] = stat(per_a, sa, idx) - stat(per_b, sb, idx)

    # Fail loudly if NaN appears (e.g. constant-entropy seed, degenerate readout)
    if np.isnan(diff):
        raise ValueError(f"metric={metric}: observed difference is NaN (possibly a constant-entropy readout)")
    nan_count = np.sum(np.isnan(draws))
    if nan_count > 0:
        raise ValueError(f"metric={metric}: {nan_count}/{B} bootstrap draws are NaN (possibly a constant-entropy seed)")

    return {"diff": diff, "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
            "p": two_sided_p(draws), "draws": draws}


def decide(results: dict) -> dict:
    keys = [(m, c) for m in METRICS for c in COMPARATORS]
    adjusted = dict(zip(keys, holm([results[k]["p"] for k in keys])))
    favours = {k: np.sign(results[k]["diff"]) == BETTER[k[0]] and adjusted[k] < 0.05 for k in keys}
    supporting = [m for m in METRICS if all(favours[(m, c)] for c in COMPARATORS)]
    return {"holm": {f"{m}/{c}": adjusted[(m, c)] for m, c in keys},
            "favours": {f"{m}/{c}": bool(favours[(m, c)]) for m, c in keys},
            "support": bool(supporting), "supporting_metric": supporting[0] if supporting else None}
