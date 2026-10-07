import numpy as np
import pytest

from mmae.engine.hb import bootstrap

rng = np.random.default_rng(1)
H = rng.dirichlet(np.ones(9), size=300)


def arm(scale, seeds=3):
    """Per-seed distributions: the human target plus noise of the given scale, renormalised."""
    out = []
    for _ in range(seeds):
        x = np.abs(H + rng.normal(scale=scale, size=H.shape)) + 1e-9
        out.append(x / x.sum(1, keepdims=True))
    return out


def test_holm_matches_reference():
    assert bootstrap.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_bootstrap_detects_a_clear_difference_and_not_a_null():
    good, bad = arm(0.01), arm(0.2)
    clear = bootstrap.paired_bootstrap(good, bad, H, "jsd", B=2000)
    assert clear["diff"] < 0 and clear["p"] < 0.01
    same = bootstrap.paired_bootstrap(good, arm(0.01), H, "jsd", B=2000)
    assert same["p"] > 0.05


def test_paintings_are_paired_across_arms():
    """Two identical one-seed arms differ by exactly zero in every replicate, which holds only if both arms are
    scored on the same painting sample (seed resampling cannot differ with one seed per arm)."""
    one = arm(0.05, seeds=1)
    for metric in ("jsd", "entropy_spearman"):
        single = bootstrap.paired_bootstrap(one, [one[0].copy()], H, metric, B=500)
        assert np.all(single["draws"] == 0.0)


def test_decide_support_rule():
    p = {("jsd", "parcap"): 0.001, ("jsd", "probe"): 0.002, ("entropy_spearman", "parcap"): 0.5,
         ("entropy_spearman", "probe"): 0.6}
    diffs = {("jsd", "parcap"): -0.01, ("jsd", "probe"): -0.02, ("entropy_spearman", "parcap"): 0.01,
             ("entropy_spearman", "probe"): -0.01}
    out = bootstrap.decide({k: {"p": p[k], "diff": diffs[k]} for k in p})
    assert out["support"] is True and out["supporting_metric"] == "jsd"
    flipped = {k: {"p": p[k], "diff": -diffs[k]} for k in p}
    assert bootstrap.decide(flipped)["support"] is False
