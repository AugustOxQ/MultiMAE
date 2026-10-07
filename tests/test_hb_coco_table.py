"""scripts/hb_coco_table.py on synthetic D1 / D2 outputs (CPU, no COCO): kill rule, collapse, D2 rule, CLI outputs."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mmae.engine.hb import coco_probes as cp

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hb_coco_table", ROOT / "scripts" / "hb_coco_table.py")
table = importlib.util.module_from_spec(spec)
sys.modules["hb_coco_table"] = table
spec.loader.exec_module(table)

Q, K, POOL = 80, 30, 400


def make_queries() -> cp.EccvI2T:
    rng = np.random.RandomState(0)
    R = np.concatenate([rng.randint(6, 16, 30), rng.randint(16, 21, 25), rng.randint(21, 29, 25)])
    return R, rng


def fake_positives(cand_by_seed):
    """One shared EccvI2T whose positives are the first R entries of a fixed 'truth' ordering of each query's pool."""
    R, rng = make_queries()
    truth = np.stack([rng.permutation(POOL)[:K] for _ in range(Q)])
    return cp.EccvI2T(np.arange(Q), [np.sort(truth[i, :r]) for i, r in enumerate(R)], R), truth


Q_OBJ, TRUTH = fake_positives(None)


def write_d1_run(root: Path, name: str, arm: str, seed: int, mode: str, beta: float = 0.5) -> None:
    """mode: hurt (decoder anti-informative everywhere), help_top (decoder informative only for R >= 21, else
    anti-informative), noise."""
    rng = np.random.RandomState(100 + seed)
    pos = np.zeros((Q, K))
    cand = np.zeros((Q, K), dtype=int)
    for i in range(Q):
        perm = rng.permutation(K)
        row = np.concatenate([TRUTH[i, : Q_OBJ.R[i]], np.setdiff1d(np.arange(POOL), TRUTH[i])[: K - Q_OBJ.R[i]]])
        cand[i] = row[perm]
        pos[i] = np.isin(cand[i], Q_OBJ.positives[i])
    dual_noise = rng.randn(Q, K)
    # candidates are sorted by a dual score that puts positives first on average
    score = 2.0 * pos + dual_noise
    o = np.argsort(-score, axis=1, kind="stable")
    cand, pos, score = (np.take_along_axis(x, o, axis=1) for x in (cand, pos, score))
    top = (Q_OBJ.R >= 21)[:, None]
    if mode == "hurt":
        dec = -3.0 * pos + 0.3 * rng.randn(Q, K)
    elif mode == "help_top":
        dec = np.where(top, 3.0, -3.0) * pos + 0.3 * rng.randn(Q, K)
    else:
        dec = rng.randn(Q, K)
    null = 0.2 * rng.randn(Q, K)
    arrays = {"cand": cand, "dual": score.astype(np.float32), "query_image": Q_OBJ.query_image, "R": Q_OBJ.R}
    d1 = {"meta": {"run": name, "arm": arm, "seed": seed, "queries": Q, "k": K}}
    alphas_b = 0.25
    d1["dual"] = cp.d1_metrics(cand, Q_OBJ)
    for kind in ("parallel", "ratio"):
        k_dec, k_null = (dec, null) if kind == "parallel" else (dec + 0.05 * rng.randn(Q, K), null)
        arrays[f"dec_{kind}"], arrays[f"null_{kind}"] = k_dec, k_null
        t = {"alpha_a": 0.5, "alpha_b": alphas_b, "beta": beta, "val_map_a": 0.0, "val_map_b": 0.0}
        d1[kind] = {
            "tuned": t,
            "rerank_a": cp.d1_metrics(cp.rerank(cand, cp.score_a(k_dec, k_null, 0.5)), Q_OBJ),
            "rerank_b": cp.d1_metrics(cp.rerank(cand, cp.score_b(arrays["dual"], k_dec, k_null, alphas_b, beta)), Q_OBJ),
            "raw_decoder": cp.d1_metrics(cp.rerank(cand, k_dec), Q_OBJ),
            "grid_a": {str(a): cp.d1_metrics(cp.rerank(cand, cp.score_a(k_dec, k_null, a)), Q_OBJ) for a in cp.ALPHAS},
            "grid_b": {f"{a}_{b}": cp.d1_metrics(cp.rerank(cand, cp.score_b(arrays["dual"], k_dec, k_null, a, b)), Q_OBJ)
                       for a in cp.ALPHAS for b in cp.BETAS},
        }
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "d1.json").write_text(json.dumps(d1))
    np.savez_compressed(folder / "scores.npz", **arrays)


@pytest.fixture
def d1_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(table, "load_positives", lambda _: Q_OBJ)
    root = tmp_path / "d1"
    for seed in (42, 43, 44):
        write_d1_run(root, f"2026_{seed}_hurt", "fam_hurt", seed, "hurt")
        write_d1_run(root, f"2026_{seed}_help", "fam_help", seed, "help_top", beta=0.0)
        write_d1_run(root, f"2026_{seed}_noise", "fam_noise", seed, "noise")
        write_d1_run(root, f"2026_{seed}_ctrl", table.CONTROL, seed, "noise")
    return root


def test_kill_rule_fires_only_when_decoder_hurts_everywhere(d1_dir, tmp_path):
    out = table.d1_summary(d1_dir, tmp_path, table.load_positives)
    fams = out["families"]
    assert fams["fam_hurt"]["kill_rule_a"]["parallel"]["verdict"] == "KILL"
    assert fams["fam_hurt"]["kill_rule_a"]["ratio"]["verdict"] == "KILL"
    # informative in the top third: some alpha has a positive top-third difference, so no kill
    helped = fams["fam_help"]["kill_rule_a"]["parallel"]
    assert helped["verdict"] == "NOT KILLED"
    assert any(v["top_third_positive"] for v in helped["per_alpha"].values())
    # a noise decoder scrambles the dual order and loses everywhere: the kill must fire
    assert fams["fam_noise"]["kill_rule_a"]["parallel"]["verdict"] == "KILL"


def test_collapsed_fraction_and_sources(d1_dir, tmp_path):
    out = table.d1_summary(d1_dir, tmp_path, table.load_positives)
    assert out["families"]["fam_help"]["collapsed_b_fraction"] == {"parallel": 1.0, "ratio": 1.0}
    assert out["families"]["fam_hurt"]["collapsed_b_fraction"] == {"parallel": 0.0, "ratio": 0.0}
    assert all(c["source"] == "scores.npz" and c["check_vs_json"] < 1e-9 for c in out["checkpoints"])
    st = out["families"]["fam_hurt"]["stats"]
    assert st["parallel/grid_a/0.5/all/map_diff"]["n"] == 3
    # the beta-0 (b) re-ranker equals the dual order
    help_st = out["families"]["fam_help"]["stats"]
    assert help_st["parallel/tuned_b/all/map_diff"]["mean"] == pytest.approx(0.0, abs=1e-9)
    # the fusion_none sanity check is control-only: n/a elsewhere
    assert out["families"]["fam_hurt"]["sanity_max_abs_dec_minus_null"] == {"parallel": None, "ratio": None}
    assert out["families"][table.CONTROL]["sanity_max_abs_dec_minus_null"]["parallel"] > 0
    assert out["families"]["fam_hurt"]["mixed_vintage"] is False


def test_nan_max_ignores_missing():
    assert table.nan_max([float("nan"), 0.5, 0.25]) == 0.5
    assert table.nan_max([float("nan")]) is None and table.nan_max([]) is None


def test_mixed_vintage_is_flagged_in_markdown(d1_dir, tmp_path, monkeypatch):
    real = table.checkpoint_summary
    def one_fallback(folder, fn, ann):  # the first hurt run falls back to d1.json
        return real(folder, fn, None if folder.name == "2026_42_hurt" else ann)
    monkeypatch.setattr(table, "checkpoint_summary", one_fallback)
    out = table.d1_summary(d1_dir, tmp_path, table.load_positives)
    assert out["families"]["fam_hurt"]["mixed_vintage"] is True
    assert out["families"]["fam_help"]["mixed_vintage"] is False
    md = "\n".join(table.d1_markdown(out))
    assert md.count("NOTE: mixed vintages") == 1 and "mixed vintages, see below" in md


def test_json_fallback_matches_recompute(d1_dir, tmp_path):
    a = table.d1_summary(d1_dir, tmp_path, table.load_positives)
    b = table.d1_summary(d1_dir, None, table.load_positives)
    assert b["checkpoints"][0]["source"] == "d1.json"
    for fam in a["families"]:
        assert a["families"][fam]["kill_rule_a"] == b["families"][fam]["kill_rule_a"]


def test_single_seed_verdict_is_undetermined(tmp_path, monkeypatch):
    monkeypatch.setattr(table, "load_positives", lambda _: Q_OBJ)
    write_d1_run(tmp_path / "d1", "only", "fam_hurt", 42, "hurt")
    out = table.d1_summary(tmp_path / "d1", tmp_path, table.load_positives)
    assert out["families"]["fam_hurt"]["kill_rule_a"]["parallel"]["verdict"].startswith("undetermined")


# ---------------------------------------------------------------- D2
CONDS = [f"blend_{x}" for x in cp.LAMBDAS] + [f"mix_{m}" for m in cp.MIX_M]
P = 40


def write_d2_run(root: Path, name: str, decoder: bool, bc: float, rerank_bc: float, flat_balance=False, seed=0) -> None:
    rng = np.random.RandomState(seed)
    a_rows = rng.randint(0, 1000, (P, 5))
    arrays, d2 = {}, {"meta": {"run": name, "pairs": P, "decoder": decoder}, "dual": {}}
    for c in CONDS:
        sims = rng.rand(P, 10)
        arrays[f"dual_sim_{c}"] = sims
        d2["dual"][c] = {"both_covered": {str(k): bc for k in (5, 10, 20)}, "both_covered_at_50": 0.05,
                         "balance": float(cp.balance(sims[:, :5], sims[:, 5:]).mean())}
    if decoder:
        dec = {"balance_parallel": {}, "score_gap": {}, "both_covered_rerank": {}, "selection": {}}
        for c in CONDS:
            par = np.concatenate([np.full((P, 5), -1.0), np.full((P, 5), -2.0)], axis=1) if flat_balance \
                else -rng.rand(P, 10)
            arrays[f"dec_par_{c}"] = par
            dec["balance_parallel"][c] = float(cp.balance(par[:, :5], par[:, 5:]).mean())
            dec["score_gap"][c] = float((par[:, :5].mean(1) - par[:, 5:].mean(1)).mean())
            dec["both_covered_rerank"][c] = {str(k): rerank_bc for k in (5, 10, 20)}
        for j in range(4):
            entry = {}
            for side in "AB":
                own, other = -1 - rng.rand(P), -2 - rng.rand(P)
                s_a, s_b = (own, other) if side == "A" else (other, own)
                blend = 0.5 * (own + other)
                arrays[f"sel_c{side}_j{j}_sA"], arrays[f"sel_c{side}_j{j}_sB"] = s_a, s_b
                arrays[f"sel_c{side}_j{j}_blend"] = blend
                entry[f"c{side}"] = {}
            entry["pooled"] = {}
            dec["selection"][str(j)] = entry
        d2["decoder"] = dec
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "d2.json").write_text(json.dumps(d2))
    np.savez_compressed(folder / "d2_scores.npz", **arrays)


def build_d2(root: Path, c_bc: float, dec_bc: float, control_bc: float | None = None) -> None:
    for seed in (42, 43):
        write_d2_run(root, f"20261005_0916{seed}_s2_contrastive", False, c_bc, 0, seed=seed)
        write_d2_run(root, f"20261005_0916{seed}_s2_txt80", True, 0.02, dec_bc, seed=seed)
        write_d2_run(root, f"20261005_0916{seed}_s2_none_txt80", True, 0.02, dec_bc if control_bc is None else control_bc, flat_balance=True, seed=seed)
    write_d2_run(root, "zeroshot", False, 0.03, 0)


def test_d2_rule_and_none_control(tmp_path):
    kill, live = tmp_path / "kill", tmp_path / "live"
    build_d2(kill, c_bc=0.03, dec_bc=0.04)  # C within 2 points of the decoders everywhere
    build_d2(live, c_bc=0.01, dec_bc=0.10)  # decoders 9 points ahead
    k = table.d2_summary(kill)
    assert k["kill_rule"]["verdict"] == "KILL"
    assert table.d2_summary(live)["kill_rule"]["verdict"] == "NOT KILLED"
    assert k["none_control"]["range"] == pytest.approx(0.0)
    assert k["families"]["s2_txt80"]["stats"]["decoder/selection/0/pooled/median"]["n"] == 2
    assert k["families"]["zeroshot"]["stats"]["decoder_present"] is False
    assert k["kill_rule"]["C_ceiling50"]["blend_0.5"] == pytest.approx(5.0)


def test_d2_best_decoder_excludes_text_only_control(tmp_path):
    kill, strong = tmp_path / "kill", tmp_path / "strong"
    build_d2(kill, c_bc=0.03, dec_bc=0.04)
    build_d2(strong, c_bc=0.03, dec_bc=0.04, control_bc=0.50)  # a control far ahead of every real decoder
    ks, kc = table.d2_summary(strong)["kill_rule"], table.d2_summary(kill)["kill_rule"]
    assert ks["verdict"] == kc["verdict"] == "KILL"
    row = ks["per_condition"]["blend_0.5"]["by_k"]["5"]
    assert row["best_family"] == "s2_txt80" and row["best_decoder"] == pytest.approx(4.0)
    assert row["text_only_control"] == pytest.approx(50.0)  # reported separately
    md = "\n".join(table.d2_markdown(table.d2_summary(strong)))
    assert "text-only control 50.00" in md


# ---------------------------------------------------------------- CLI
def test_cli_writes_both_files_and_skips_missing_parts(d1_dir, tmp_path):
    d2 = tmp_path / "d2"
    build_d2(d2, 0.03, 0.04)
    # the CLI runs in a subprocess, so use the json fallback (no annotations)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "hb_coco_table.py"), "--d1-dir", str(d1_dir), "--d2-dir",
                        str(d2), "--out", str(out), "--annotations-dir", str(tmp_path / "missing")],
                       capture_output=True, text=True, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin"})
    assert r.returncode == 0, r.stderr
    data = json.loads((out / "hb_coco.json").read_text())
    assert set(data) == {"d1", "d2", "notes"}
    assert "fam_hurt" in data["d1"]["families"] and "kill_rule" in data["d2"]
    md = (out / "hb_coco.md").read_text()
    assert md.index("## D1") < md.index("Per-family means") < md.index("## D2")
    assert "KILL" in md

    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "hb_coco_table.py"), "--d1-dir", str(tmp_path / "nope"),
                        "--out", str(tmp_path / "out2")], capture_output=True, text=True)
    assert r.returncode == 0
    data = json.loads((tmp_path / "out2" / "hb_coco.json").read_text())
    assert "d1" not in data and len(data["notes"]) == 2


def kill_stats(mean, sd, top):
    """Hand-made stats: every alpha has the same overall paired diff (mean, SD) and top-third diff."""
    st = {}
    for a in cp.ALPHAS:
        st[f"parallel/grid_a/{a}/all/map_diff"] = {"mean": mean, "sd": sd, "n": 3}
        st[f"parallel/grid_a/{a}/{table.TOP_THIRD}/map_diff"] = {"mean": top, "sd": sd, "n": 3}
    return st


def test_kill_rule_sign_and_threshold():
    """The diff must be BELOW minus its SD: a loss inside the SD, or any gain, is not a kill."""
    kill = lambda m, sd, top: table.kill_rule(kill_stats(m, sd, top), "parallel", 3)["verdict"]
    assert kill(-2.0, 1.0, -1.0) == "KILL"
    assert kill(-0.5, 1.0, -1.0) == "NOT KILLED"  # inside the SD
    assert kill(+2.0, 1.0, -1.0) == "NOT KILLED"  # a gain
    assert kill(-2.0, 1.0, +0.1) == "NOT KILLED"  # positive in the top third
