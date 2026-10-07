import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from helpers import make_fake_al28, make_fake_artelingo

from mmae.engine.hb import data, tables
from mmae.engine.hb.encode import D7_PATTERNS

ROOT = Path(__file__).resolve().parents[1]
N, P, T_TRAIN, V, L, D, NC, B = 30, 40, 60, 16, 9, 8, 12, 200
ENGLISH = ["p0", "p1", "p2", "p3", "p4", "p5", "v0", "v1", "v2", "t0", "t1", "t2"]  # painting names in the fake ArtELingo
NAMES = ENGLISH + [f"x{i}" for i in range(N - len(ENGLISH))]
ARMS = {"ml80": ("masked", 0.8, 0.0), "ml80_mae": ("masked", 0.8, 0.5), "parcap": ("clean", 1.0, 0.0), "c": None}
PRIOR = np.array([.3, .2, .15, .1, .1, .05, .05, .03, .02])


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("hb_table")
    csv = make_fake_al28(root / "al28.csv", {n: f"Style_A/{n}.jpg" for n in NAMES}, votes=60, classes=lambda k: 1 + k % 6)
    _, annotations, _ = make_fake_artelingo(root / "art")
    paintings, counts = data.al28_targets(csv, 20)
    assert paintings.names == sorted(NAMES)
    return {"root": root, "csv": csv, "annotations": annotations, "names": paintings.names, "counts": counts}


def make_run(folder: Path, world, arm: str, seed: int, **meta_over):
    rng = np.random.default_rng(seed + 10 * list(ARMS).index(arm))
    counts = world["counts"]
    true = 0.9 * counts / counts.sum(1, keepdims=True) + 0.1 / 9
    shared = np.random.default_rng(777)  # the validation paintings and labels are the same for every run
    val_true = shared.dirichlet(np.ones(9) * 0.4, size=P)
    index = np.repeat(np.arange(P), 5)
    labels = np.array([shared.choice(9, p=val_true[i]) for i in index])
    arrays = {"al28_counts": counts, "val_index": index, "val_labels": labels,
              "al28_emb_full": rng.normal(size=(N, D)).astype(np.float16),
              "al28_emb_views": rng.normal(size=(N, V, D)).astype(np.float16),
              "val_emb_full": rng.normal(size=(P, D)).astype(np.float16),
              "val_emb_views": rng.normal(size=(P, V, D)).astype(np.float16),
              "train_emb": rng.normal(size=(T_TRAIN, D)).astype(np.float16),
              "train_counts": rng.multinomial(5, PRIOR, size=T_TRAIN), "prompts": rng.normal(size=(9, D)).astype(np.float32),
              "logit_scale": np.float32(10.0)}
    meta = {"run": folder.name, "arm": f"hb_{arm}", "seed": seed, "decoder": ARMS[arm] is not None, "views": V,
            "lengths": list(range(5, 14)), "al28": world["names"], "val": [], "train": []}
    if ARMS[arm]:
        source, ratio, mae = ARMS[arm]
        meta.update(mlm_image_source=source, text_ratio=ratio, mae_weight=mae, d7_views=4 if source == "masked" else 1)
        blur = arm == "parcap"
        perm = rng.permutation(N)
        for pre, t, n in (("al28", true, N), ("val", val_true, P)):
            z = np.log(0.5 * t + 0.5 * t[rng.permutation(n)]) if blur else np.log(t)
            noise = rng.normal(scale=0.05, size=(n, V, L, 9))
            arrays[f"{pre}_dec_views"] = (z[:, None, None] + noise).astype(np.float16)
            arrays[f"{pre}_dec_full"] = (z[:, None, None] + noise[:, :1]).astype(np.float16)
        vp = 1 if blur else 4
        caps = np.random.default_rng(999)  # the captions are the same for every run
        label = caps.integers(0, 9, NC)
        valid = caps.random((NC, 9)) > 0.2
        valid[:, 0] = True
        real = rng.normal(size=(NC, 9, vp, 9))
        real[np.arange(NC), :, :, label] += 3.0  # the real image points at the label, the null image does not
        arrays.update(d7_real=real.astype(np.float16), d7_null=rng.normal(size=(NC, 9, vp, 9)).astype(np.float16),
                      d7_valid=valid, d7_label=label.astype(np.int64))
    meta.update(meta_over)
    folder.mkdir(parents=True)
    np.savez(folder / "encode.npz", **arrays)
    (folder / "meta.json").write_text(json.dumps(meta))


def make_root(root: Path, world, seeds=(1, 2), arms=tuple(ARMS)):
    for arm in arms:
        for s in seeds:
            make_run(root / f"{arm}_s{s}", world, arm, s)
    return root


@pytest.fixture(scope="module")
def result(world):
    enc_root = make_root(world["root"] / "enc", world)
    out = world["root"] / "out"
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "hb_table.py"), "--encoded-root", str(enc_root), "--out", str(out),
                        "--annotations-dir", str(world["annotations"]), "--al28-csv", str(world["csv"]), "--B", str(B)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    print(f"hb_table.py wall time on the synthetic data: {time.time() - t0:.1f}s")
    return json.loads((out / "hb_d4.json").read_text()), (out / "hb_d4.md").read_text()


def test_outputs_and_keys(result):
    res, md = result
    for key in ("runs", "arms", "probes", "references", "tests", "decision", "d6", "d7"):
        assert key in res
    assert len(res["runs"]) == 8 and set(res["arms"]) == set(ARMS)
    ml = res["arms"]["ml80"]["1"]
    assert {"views16", "views4", "views1", "full", "prompt_full", "prompt_views16", "probe_full", "probe_views16",
            "prior"} <= set(ml["readouts"])
    assert set(res["arms"]["c"]["1"]["readouts"]) == {"prompt_full", "prompt_views16", "probe_full", "probe_views16", "prior"}
    r = ml["readouts"]["views16"]
    assert {"T", "val_nll", "metrics", "other_dropped", "named8"} <= set(r)
    assert {"jsd", "entropy_spearman", "kl", "tvd", "rank_cs", "thirds"} <= set(r["metrics"]) and len(r["metrics"]["thirds"]) == 3
    assert set(res["probes"]["strongest"]) == {"1", "2"} and "weight_decay" in res["probes"]["per_arm"]["c"]["1"]
    assert {"jsd/parcap", "jsd/probe", "entropy_spearman/parcap", "entropy_spearman/probe", "secondary"} <= set(res["tests"])
    assert '"draws"' not in json.dumps(res) and md.startswith("# H-b D4")
    assert md.index("## Decision") < md.index("## D4") < md.index("## Human references") < md.index("## D6") < md.index("## D7")


def test_decision_is_provisional_with_two_seeds(result):
    res, md = result
    d = res["decision"]
    assert d["provisional"].startswith("provisional (seeds:") and "provisional" in md
    assert d["support"] in (True, False) and isinstance(d["kill"], bool)
    for k in ("jsd/parcap", "entropy_spearman/probe"):
        t = res["tests"][k]
        assert t["ci_low"] <= t["diff"] <= t["ci_high"] and 0 <= t["holm_p"] <= 1 and t["provisional"]


def test_ml80_beats_parcap_in_the_synthetic_case(result):
    res, _ = result
    ml = [res["arms"]["ml80"][s]["readouts"]["views16"]["metrics"]["jsd"] for s in "12"]
    pc = [res["arms"]["parcap"][s]["readouts"]["full"]["metrics"]["jsd"] for s in "12"]
    assert max(ml) < min(pc)
    assert res["tests"]["jsd/parcap"]["diff"] < 0


def test_d6_has_k_and_controls(result):
    res, _ = result
    d6 = res["d6"]
    assert d6["K"] == {"views1": 1, "views4": 4, "views16": 16}
    for arm in ("ml80", "ml80_mae"):
        r = d6["arms"][arm]
        assert {"views1", "views4", "views16", "full", "mi", "k16_beats_k1"} <= set(r)
        assert r["mi"]["ci_low"] <= r["mi"]["spearman_mean"] <= r["mi"]["ci_high"] or r["mi"]["nan_draws"] > 0
    assert set(d6["controls"]) == {"parcap_views16", "c_prompt_views16", "strongest_probe_views16"}


def test_d7_has_every_pattern(result):
    res, md = result
    d7 = res["d7"]
    assert set(d7["arms"]) == {"ml80", "ml80_mae", "parcap"}
    rows = d7["arms"]["ml80"]["patterns"]
    assert list(rows) == D7_PATTERNS
    assert rows["j0"]["n"] == NC and all(rows[p]["n"] <= NC for p in rows)
    assert rows["prefix2"]["real"]["log_loss"] < rows["prefix2"]["null"]["log_loss"]
    d = rows["prefix2"]["real_minus_null"]
    assert d["diff"] < 0 and d["ci_low"] <= d["diff"] <= d["ci_high"]
    assert rows["prefix2"]["no_image_benefit"] is False and "no_image_benefit" not in rows["prefix1"]
    assert "no_image_benefit" not in d7["arms"]["parcap"]["patterns"]["prefix2"]
    assert d7["arms"]["parcap"]["readout"] == "full" and d7["arms"]["ml80"]["readout"] == "views4"


def test_references(result):
    res, _ = result
    ref = res["references"]
    sh = ref["split_half"]
    assert sh["n"] == N and 0 < sh["jsd"] < 1 and -1 <= sh["entropy_spearman"] <= 1
    en = ref["english"]
    assert en["n"] == len(ENGLISH) and en["n_dropped_without_english"] == N - len(ENGLISH) and 0 <= en["jsd"] <= 1


def test_variants_and_baselines(result):
    res, _ = result
    r = res["arms"]["ml80"]["1"]["readouts"]["views16"]
    assert r["named8"]["n"] == N and 0 <= r["named8"]["jsd"] <= 1 and 0 <= r["other_dropped"]["jsd"] <= 1
    assert res["arms"]["ml80"]["1"]["readouts"]["prior"]["metrics"]["entropy_spearman"] is None  # constant entropy: null
    # the decoder built from the truth beats the random-embedding baselines on JSD
    c = res["arms"]["c"]["1"]["readouts"]
    assert r["metrics"]["jsd"] < c["prompt_full"]["metrics"]["jsd"] and r["metrics"]["jsd"] < c["probe_full"]["metrics"]["jsd"]


def test_unknown_arm_raises(world, tmp_path):
    root = make_root(tmp_path / "enc", world, seeds=(1,))
    make_run(root / "weird", world, "ml80", 9, text_ratio=0.5)
    with pytest.raises(ValueError, match="weird"):
        tables.load_runs(root)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "hb_table.py"), "--encoded-root", str(root), "--out",
                        str(tmp_path / "o"), "--al28-csv", str(world["csv"]), "--B", "10"], capture_output=True, text=True)
    assert r.returncode != 0 and "weird" in r.stderr


def test_duplicate_seed_raises(world, tmp_path):
    root = make_root(tmp_path / "enc", world, seeds=(1,))
    shutil.copytree(root / "ml80_s1", root / "ml80_again")
    with pytest.raises(ValueError, match="duplicate"):
        tables.load_runs(root)


def test_arm_of():
    assert tables.arm_of({"decoder": False}, "f") == "c"
    assert tables.arm_of({"decoder": True, "mlm_image_source": "masked", "text_ratio": 0.8, "mae_weight": 0.0}, "f") == "ml80"
    assert tables.arm_of({"decoder": True, "mlm_image_source": "masked", "text_ratio": 0.8, "mae_weight": 1.0}, "f") == "ml80_mae"
    assert tables.arm_of({"decoder": True, "mlm_image_source": "clean", "text_ratio": 1.0, "mae_weight": 0.0}, "f") == "parcap"
    with pytest.raises(ValueError):
        tables.arm_of({"decoder": True, "mlm_image_source": "clean", "text_ratio": 0.8, "mae_weight": 0.0}, "f")


def test_missing_arm_is_provisional_and_names_the_probe_pool(world, tmp_path):
    root = make_root(tmp_path / "enc", world, seeds=(1, 2, 3), arms=("ml80", "parcap", "c"))
    res = tables.build(root, world["annotations"], world["csv"], B=50)
    note = res["decision"]["provisional"]
    assert note and "probe pool: ML-80, Par-cap, C" in note and "ML-80+MAE" in note
    assert all(t["provisional"] and "probe pool" in t["provisional"] for k, t in res["tests"].items() if k != "secondary")
    assert "ml80_mae" not in res["arms"] and res["tests"]["secondary"] == {}
    assert res["d6"]["probe_pool_note"] and "ml80" not in res["d6"]["missing_controls"]
    root = make_root(tmp_path / "enc2", world, seeds=(1, 2, 3), arms=("ml80", "ml80_mae", "parcap"))
    res = tables.build(root, world["annotations"], world["csv"], B=50)
    assert res["d6"]["missing_controls"] == ["c_prompt_views16"] and "probe pool: ML-80, ML-80+MAE, Par-cap" in res["decision"]["provisional"]


def test_all_arms_three_seeds_not_provisional(world, tmp_path):
    root = make_root(tmp_path / "enc", world, seeds=(1, 2, 3))
    res = tables.build(root, world["annotations"], world["csv"], B=50)
    assert res["decision"]["provisional"] is None and all(t["provisional"] is None for t in res["tests"].values() if "provisional" in t) and all(
        t["provisional"] is None for t in res["tests"]["secondary"].values())
    assert res["d6"]["missing_controls"] == [] and res["d6"]["probe_pool_note"] is None


def test_probe_comparator_uses_complete_seeds(world, tmp_path):
    root = make_root(tmp_path / "enc", world, seeds=(1, 2), arms=("ml80", "parcap", "c"))
    make_run(root / "ml80_s3", world, "ml80", 3)
    res = tables.build(root, world["annotations"], world["csv"], B=50)
    assert res["settings"]["complete_seeds"] == [1, 2] and set(res["probes"]["strongest"]) == {"1", "2"}
    assert "ml80 [1, 2, 3]" in res["tests"]["jsd/probe"]["provisional"]


def test_al28_votes_and_drop_other(world):
    votes = data.al28_votes(world["csv"], 20)
    paintings, counts = data.al28_targets(world["csv"], 20)
    assert sorted(votes) == paintings.names
    for i, n in enumerate(paintings.names):
        np.testing.assert_array_equal(np.bincount(votes[n], minlength=9), counts[i])
    p2, od = data.al28_targets(world["csv"], 20, drop_other=True)
    assert p2.names == paintings.names and (counts.sum(1) - od.sum(1) == 1).all()  # one 'other' vote per painting
    assert (counts[:, :8] == od[:, :8]).all()


def _kill_rows(world, tmp_path, edit):
    root = make_root(tmp_path / "enc", world, seeds=(1, 2), arms=("ml80", "parcap", "c"))
    for s in (1, 2):
        z = dict(np.load(root / f"ml80_s{s}" / "encode.npz"))
        z["d7_real"] = edit(z, np.random.default_rng(s)).astype(np.float16)
        np.savez(root / f"ml80_s{s}" / "encode.npz", **z)
    return tables.build(root, world["annotations"], world["csv"], B=300)["d7"]["arms"]["ml80"]["patterns"]


def test_kill_rule_symmetric_effects_straddle_zero(world, tmp_path):
    def symmetric(z, rng):  # the real image helps half of the captions and hurts the other half by the same logit
        out = z["d7_null"].astype(np.float64).copy()
        sign = np.where(np.arange(len(out)) % 2 == 0, 1.0, -1.0)
        out[np.arange(len(out)), :, :, z["d7_label"]] += 1.5 * sign[:, None, None]
        return out

    rows = _kill_rows(world, tmp_path, symmetric)
    for p in tables.KILL_PATTERNS:
        d = rows[p]["real_minus_null"]
        assert d["ci_low"] < 0 < d["ci_high"], (p, d)  # the scenario really straddles zero
        assert rows[p]["no_image_benefit"] is True


def test_kill_rule_real_worse_than_null_is_flagged(world, tmp_path):
    def worse(z, rng):
        out = z["d7_null"].astype(np.float64).copy()
        out[np.arange(len(out)), :, :, z["d7_label"]] -= 3.0
        return out

    rows = _kill_rows(world, tmp_path, worse)
    for p in tables.KILL_PATTERNS:
        assert rows[p]["real_minus_null"]["ci_low"] > 0 and rows[p]["no_image_benefit"] is True


def test_kill_rule_strong_benefit_is_not_flagged(result):
    rows = result[0]["d7"]["arms"]["ml80"]["patterns"]
    assert all(rows[p]["real_minus_null"]["ci_high"] < 0 and rows[p]["no_image_benefit"] is False for p in tables.KILL_PATTERNS)


def test_d7_interval_resamples_seeds():
    d = np.stack([np.full(50, -1.0), np.full(50, 1.0)])  # two seeds that disagree, no caption variance
    lo, hi = tables._boot_mean(d, 400)
    assert lo == -1.0 and hi == 1.0


def test_d6_interval_resamples_seeds():
    he = np.random.default_rng(0).normal(size=80)
    draws = tables.boot_spearman([he, -he], he, 400)  # per-seed rho +1 and -1
    lo, hi = np.quantile(draws, [0.025, 0.975])
    assert lo < -0.9 and hi > 0.9


def test_validation_labels_must_match_across_runs(world, tmp_path):
    root = make_root(tmp_path / "enc", world, seeds=(1,), arms=("ml80", "parcap"))
    z = dict(np.load(root / "parcap_s1" / "encode.npz"))
    z["val_labels"] = (z["val_labels"] + 1) % 9
    np.savez(root / "parcap_s1" / "encode.npz", **z)
    with pytest.raises(ValueError, match="validation"):
        tables.load_runs(root)


def test_d7_arrays_and_views_are_required(world, tmp_path):
    root = make_root(tmp_path / "enc", world, seeds=(1,), arms=("ml80", "parcap"))
    z = {k: v for k, v in np.load(root / "ml80_s1" / "encode.npz").items() if not k.startswith("d7_")}
    np.savez(root / "ml80_s1" / "encode.npz", **z)
    with pytest.raises(ValueError, match="ml80_s1.*D7"):
        tables.load_runs(root)
    root = make_root(tmp_path / "enc2", world, seeds=(1,), arms=("ml80", "parcap"))
    meta = json.loads((root / "ml80_s1" / "meta.json").read_text())
    meta["d7_views"] = 2
    (root / "ml80_s1" / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="ml80_s1.*4 view"):
        tables.load_runs(root)


def test_t_at_bound_flag_and_english_has_no_kl(result):
    res, md = result
    assert all(isinstance(r["t_at_bound"], bool) for a in res["arms"].values() for s in a.values() for r in s["readouts"].values())
    assert "kl" not in res["references"]["english"]
    assert "`*` the fitted temperature" in md
