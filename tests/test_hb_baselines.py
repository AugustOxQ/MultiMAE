import numpy as np

from mmae.engine.hb import baselines, calibrate

rng = np.random.default_rng(0)


def test_prompt_logits_shapes_and_scale():
    emb, prompts = rng.normal(size=(4, 8)), rng.normal(size=(9, 8))
    out = baselines.prompt_logits(emb, prompts, 100.0)
    assert out.shape == (4, 1, 9) and np.abs(out).max() <= 100.0 + 1e-6
    assert baselines.prompt_logits(rng.normal(size=(4, 3, 8)), prompts, 100.0).shape == (4, 3, 9)


def test_soft_probe_learns_a_linear_rule_and_picks_a_decay():
    w = rng.normal(size=(8, 9))
    x = rng.normal(size=(600, 8))
    probs = np.exp(x @ w) / np.exp(x @ w).sum(1, keepdims=True)
    counts = np.stack([rng.multinomial(5, p) for p in probs])
    val_x = rng.normal(size=(200, 8))
    val_p = np.exp(val_x @ w) / np.exp(val_x @ w).sum(1, keepdims=True)
    val_index = np.repeat(np.arange(200), 5)
    val_labels = np.array([rng.choice(9, p=val_p[i]) for i in val_index])
    probe = baselines.fit_soft_probe(x, counts, val_x, val_index, val_labels)
    assert probe.weight_decay in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
    fitted = calibrate.mixture(probe.logits(val_x), 1.0)
    prior = np.tile(counts.sum(0) / counts.sum(), (200, 1))
    assert calibrate.nll(fitted, val_index, val_labels) < calibrate.nll(prior, val_index, val_labels)
