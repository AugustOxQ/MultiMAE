import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mmae.engine.hb import analysis

ROOT = Path(__file__).resolve().parents[1]
N = P = 60
V, L, D = 16, 9, 8


def make_folder(path: Path, kind: str = "good", decoder: bool = True, with_train: bool = True, source: str = "masked"):
    rng = np.random.default_rng(1)
    prior = np.array([.3, .2, .15, .1, .1, .05, .05, .03, .02])
    true_al = rng.dirichlet(np.ones(9) * 0.4, size=N)
    true_val = rng.dirichlet(np.ones(9) * 0.4, size=P)
    if kind == "collapsed":  # labels carry no painting signal: the best readout is the prior itself (T = 1)
        true_val = np.tile(prior, (P, 1))
    index = np.repeat(np.arange(P), 20)
    labels = np.array([rng.choice(9, p=true_val[i]) for i in index])
    counts = np.stack([rng.multinomial(30, p) for p in true_al])

    def logits(true, n):
        if kind == "collapsed":
            z = np.tile(np.log(prior), (n, 1))
        elif kind == "sharp":
            z = 8 * np.log(np.clip(true, 1e-6, None))
        else:
            z = np.log(np.clip(true, 1e-6, None))
        return z

    arrays = {"al28_counts": counts, "val_index": index.astype(np.int64), "val_labels": labels.astype(np.int64),
              "al28_emb_full": rng.normal(size=(N, D)).astype(np.float16),
              "al28_emb_views": rng.normal(size=(N, V, D)).astype(np.float16),
              "val_emb_full": rng.normal(size=(P, D)).astype(np.float16),
              "val_emb_views": rng.normal(size=(P, V, D)).astype(np.float16),
              "prompts": rng.normal(size=(9, D)).astype(np.float32)}
    if with_train:
        arrays["train_emb"] = rng.normal(size=(5, D)).astype(np.float16)
        arrays["train_counts"] = np.tile(np.rint(prior * 100).astype(int), (5, 1))
    if decoder:
        for pre, true, n in (("al28", true_al, N), ("val", true_val, P)):
            z = logits(true, n)[:, None, None, :]
            noise = rng.normal(scale=0.01, size=(n, V, L, 9))
            arrays[f"{pre}_dec_views"] = (z + noise).astype(np.float16)
            arrays[f"{pre}_dec_full"] = (z + noise[:, :1]).astype(np.float16)
    path.mkdir(parents=True)
    np.savez(path / "encode.npz", **arrays)
    meta = {"run": path.name, "arm": "hb_ml80", "seed": 3, "decoder": decoder, "mlm_image_source": source, "run_status": "completed",
            "views": V, "lengths": list(range(5, 14)), "al28": [], "val": [], "train": []}
    (path / "meta.json").write_text(json.dumps(meta))
    return prior


def test_decoder_readouts_shapes(tmp_path):
    make_folder(tmp_path / "a")
    enc = analysis.load_encoded(tmp_path / "a")
    assert enc["al28_dec_views"].dtype == np.float32 and enc["meta"]["arm"] == "hb_ml80"
    r = analysis.decoder_readouts(enc)
    assert r["views16"][0].shape == (N, V * L, 9) and r["views16"][1].shape == (P, V * L, 9)
    assert r["views4"][0].shape == (N, 4 * L, 9)
    assert r["full"][0].shape == (N, L, 9)
    np.testing.assert_array_equal(r["views1"][0], enc["al28_dec_views"][:, 0])
    np.testing.assert_array_equal(r["views1"][1], enc["val_dec_views"][:, 0])


def test_no_decoder_gives_empty(tmp_path):
    make_folder(tmp_path / "a", decoder=False)
    assert analysis.decoder_readouts(analysis.load_encoded(tmp_path / "a")) == {}


def test_primary_readout():
    assert analysis.primary_readout({"mlm_image_source": "masked"}) == "views16"
    assert analysis.primary_readout({"mlm_image_source": "clean"}) == "full"


def test_prior_from_train_counts(tmp_path):
    make_folder(tmp_path / "a")
    enc = analysis.load_encoded(tmp_path / "a")
    p = analysis.prior_from(enc)
    np.testing.assert_allclose(p, [.3, .2, .15, .1, .1, .05, .05, .03, .02])
    make_folder(tmp_path / "b", with_train=False)
    with pytest.raises(ValueError):
        analysis.prior_from(analysis.load_encoded(tmp_path / "b"))


def test_gate_passes(tmp_path):
    prior = make_folder(tmp_path / "a")
    out = analysis.gate(analysis.load_encoded(tmp_path / "a"), prior)
    assert out["passed"], out
    assert out["readout"] == "views16" and isinstance(out["temperature"], float)
    assert out["val_nll"] < out["prior_nll"]


def test_gate_clean_uses_full(tmp_path):
    prior = make_folder(tmp_path / "a", source="clean")
    out = analysis.gate(analysis.load_encoded(tmp_path / "a"), prior)
    assert out["readout"] == "full" and out["passed"]


def test_gate_collapsed_fails(tmp_path):
    prior = make_folder(tmp_path / "a", kind="collapsed")
    out = analysis.gate(analysis.load_encoded(tmp_path / "a"), prior)
    assert not out["checks"]["not_collapsed_to_prior"] and not out["passed"]
    assert out["jsd_to_prior"] <= 0.02


def test_gate_temperature_out_of_range(tmp_path):
    prior = make_folder(tmp_path / "a", kind="sharp")
    out = analysis.gate(analysis.load_encoded(tmp_path / "a"), prior)
    assert out["temperature"] > 4 and not out["checks"]["temperature_in_range"] and not out["passed"]


def run_gate(*folders, extra=()):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "hb_gate.py"), "--encoded", *map(str, folders), *extra],
                          capture_output=True, text=True)


def test_gate_script(tmp_path):
    make_folder(tmp_path / "good")
    make_folder(tmp_path / "nodec", decoder=False)
    r = run_gate(tmp_path / "good", tmp_path / "nodec", extra=["--json", str(tmp_path / "o.json")])
    assert r.returncode == 0, r.stderr
    assert "PASS" in r.stdout and "skip (no decoder)" in r.stdout
    assert len(json.loads((tmp_path / "o.json").read_text())) == 2
    make_folder(tmp_path / "bad", kind="collapsed")
    r = run_gate(tmp_path / "good", tmp_path / "bad")
    assert r.returncode == 1 and "FAIL" in r.stdout and "not_collapsed_to_prior" in r.stdout


def test_gate_script_no_decoder_runs_exits_2(tmp_path):
    make_folder(tmp_path / "a", decoder=False)
    r = run_gate(tmp_path / "a")
    assert r.returncode == 2 and "no decoder runs gated" in r.stdout


def test_gate_script_error_exits_3_and_continues(tmp_path):
    make_folder(tmp_path / "good")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "meta.json").write_text("{}")  # a run folder whose encode.npz is missing
    r = run_gate(tmp_path / "broken", tmp_path / "good")
    assert r.returncode == 3 and "ERROR" in r.stdout and "PASS" in r.stdout


def test_readout_layout_pins_view_and_length(tmp_path):
    folder = tmp_path / "a"
    make_folder(folder)
    enc = analysis.load_encoded(folder)
    for pre, n in (("al28", N), ("val", P)):
        paint = np.arange(n)[:, None, None]
        view = np.arange(V)[None, :, None]
        length = np.arange(L)[None, None, :]
        z = np.zeros((n, V, L, 9), np.float32)
        z[..., 0] = 100 * paint + 10 * view + length  # exact in float32
        enc[f"{pre}_dec_views"] = z
        enc[f"{pre}_dec_full"] = z[:, :1]
    r = analysis.decoder_readouts(enc)
    for k, nv in (("views16", V), ("views4", 4), ("views1", 1)):
        for side, n in ((0, N), (1, P)):
            got = r[k][side][..., 0]
            assert got.shape == (n, nv * L)
            for i in (0, 7, n - 1):
                expect = [100 * i + 10 * v + l for v in range(nv) for l in range(L)]
                np.testing.assert_array_equal(got[i], expect)
    for side, n in ((0, N), (1, P)):
        got = r["full"][side][..., 0]
        np.testing.assert_array_equal(got[5], [100 * 5 + l for l in range(L)])


def test_gate_script_skips_non_run_paths(tmp_path):
    make_folder(tmp_path / "good")
    (tmp_path / "length_grid.json").write_text("{}")
    (tmp_path / "empty").mkdir()
    r = run_gate(tmp_path / "good", tmp_path / "length_grid.json", tmp_path / "empty", tmp_path / "missing")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ERROR" not in r.stdout and "PASS" in r.stdout and "[completed]" in r.stdout
