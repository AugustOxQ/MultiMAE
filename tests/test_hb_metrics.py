import numpy as np
import pytest
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, spearmanr

from mmae.engine.hb import calibrate, metrics

rng = np.random.default_rng(0)
P = rng.dirichlet(np.ones(9), size=50)
H = rng.dirichlet(np.ones(9) * 0.5, size=50)
H[0] = [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0]  # human zeros


def test_js_distance_matches_scipy():
    expected = np.array([jensenshannon(p, h, base=2) for p, h in zip(P, H)])
    np.testing.assert_allclose(metrics.js_distance(P, H), expected, atol=1e-10)
    assert np.all(metrics.js_distance(P, H) <= 1.0)


def test_entropy_kl_tvd():
    np.testing.assert_allclose(metrics.entropy_bits(P), entropy(P, base=2, axis=1), atol=1e-10)
    np.testing.assert_allclose(metrics.kl(H, P), np.array([entropy(h, p) for h, p in zip(H, P)]), atol=1e-9)
    assert np.isfinite(metrics.kl(H, P)).all()
    np.testing.assert_allclose(metrics.tvd(P, H), 0.5 * np.abs(P - H).sum(1), atol=1e-12)


def test_spearman_metrics():
    rho = spearmanr(entropy(P, axis=1), entropy(H, axis=1)).correlation
    assert metrics.entropy_spearman(P, H) == pytest.approx(rho)
    flat = H.copy()
    flat[1] = 1 / 9  # constant human vector: skipped
    per = [spearmanr(p, h).correlation for i, (p, h) in enumerate(zip(P, flat)) if i != 1]
    assert metrics.rank_cs(P, flat) == pytest.approx(np.mean(per))


def test_normalise_counts():
    counts = np.array([[2, 0, 0, 0, 0, 0, 0, 0, 2], [0, 0, 0, 0, 0, 0, 0, 0, 5]])
    np.testing.assert_allclose(metrics.normalise(counts).sum(1), 1.0)


def test_fit_temperature_recovers_a_known_temperature():
    true_t = 2.0
    logits = rng.normal(size=(400, 3, 9)) * 3
    probs = calibrate.mixture(logits, true_t)
    labels_per = 40
    painting = np.repeat(np.arange(400), labels_per)
    labels = np.array([rng.choice(9, p=probs[i]) for i in painting])
    fitted = calibrate.fit_temperature(logits, painting, labels)
    assert fitted == pytest.approx(true_t, rel=0.1)
    assert calibrate.nll(calibrate.mixture(logits, fitted), painting, labels) <= \
        calibrate.nll(calibrate.mixture(logits, 1.0), painting, labels)


def test_mixture_averages_probabilities_not_logits():
    logits = np.array([[[10.0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 10.0, 0, 0, 0, 0, 0, 0, 0]]])
    mixed = calibrate.mixture(logits, 1.0)
    np.testing.assert_allclose(mixed.sum(1), 1.0)
    assert mixed[0, 0] == pytest.approx(mixed[0, 1]) and mixed[0, 0] > 0.49
