"""H-b D1 / D2 summary tables over seeds (spec 2026-10-07, sections 10 and 11).

  python scripts/hb_coco_table.py --d1-dir res/hb/d1 --d2-dir res/hb/d2 --out res/hb/coco_tables
                                  [--annotations-dir /data/SSD/coco/annotations]

Reads the per-checkpoint outputs of scripts/hb_coco_d1.py (<run>/d1.json, scores.npz) and scripts/hb_coco_d2.py
(<run or zeroshot>/d2.json, d2_scores.npz, balance.npz) and writes <out>/hb_coco.json and <out>/hb_coco.md: D1 verdicts
first, then per-family means (over seeds) and seed SDs, then D2. Either input folder may be missing; that part is skipped
with a note.

D1: with the ECCV Caption annotations available (--annotations-dir, default /data/SSD/coco/annotations when it exists)
every number is recomputed from scores.npz with the scorers of mmae.engine.hb.coco_probes (score_a, score_b, rerank,
d1_metrics) and the tuned alpha / beta of d1.json; the recomputed tuned re-rankers are checked against d1.json
("check_vs_json"). Without annotations (positives are not stored in scores.npz) the metrics of d1.json are used, which the
same scorers wrote. The dual order is the stored candidate order (the dual top k); recomputation needs R.max() <= k.

Choices left open by the brief (identical for every family and seed): the kill rule reads overall mAP@R; the "seed SD" is
the sample SD (ddof 1) over seeds, undefined with one seed (the verdict is then "undetermined"); "top R third" is
R >= 21; D2 percentages are shares x 100; the D2 kill rule compares C's dual both-covered@k with the best family's
both-covered@k after re-ranker (a), per condition, for k in 5, 10, 20.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repo root, for `import mmae`

import numpy as np  # noqa: E402

from mmae.engine.hb import coco_probes as cp  # noqa: E402

KINDS = ("parallel", "ratio")
SCOPES = ("all", *cp.THIRDS)
TOP_THIRD = "R>=21"
FAMILIES = {
    "s2_txt80_mae0": "ML-80", "s2_txt80": "ML-80+MAE", "s2_multilearner": "ML-15",
    "s2_m1clean_txt80": "M1-clean-80", "s2_none_txt80": "text-only (control)",
    "s2_contrastive": "C", "zeroshot": "zero-shot CLIP",
}
DEFAULT_ANNOTATIONS = Path("/data/SSD/coco/annotations")
D2_KS = (5, 10, 20)
COVER_TOL = 2.0  # points


# ---------------------------------------------------------------- generic helpers
def flatten(d, prefix="") -> dict[str, float]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        elif isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) and v == v:
            out[key] = float(v)
    return out


def mean_sd(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else None, "n": int(len(a))}


def aggregate(flats: list[dict[str, float]]) -> dict[str, dict]:
    keys = sorted(set().union(*[f.keys() for f in flats])) if flats else []
    return {k: mean_sd([f[k] for f in flats if k in f]) for k in keys}


def family_of(arm: str | None, run_name: str = "") -> str:
    if arm:
        return arm
    m = re.match(r"^\d{8}_\d{6}_(.+)$", run_name)
    return m.group(1) if m else (run_name or "unknown")


def label(family: str) -> str:
    return FAMILIES.get(family, family)


def fmt(x, nd=2) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def pm(stat: dict | None, nd=2) -> str:
    if not stat:
        return "n/a"
    return f"{stat['mean']:.{nd}f} +- {fmt(stat['sd'], nd)}" if stat["sd"] is not None else f"{stat['mean']:.{nd}f}"


# ---------------------------------------------------------------- D1: one checkpoint
def flat_metrics(m: dict) -> dict:
    """d1_metrics output -> {scope: {map, rp}}."""
    out = {"all": {"map": m["map_at_r"], "rp": m["r_precision"]}}
    for name, v in m["by_third"].items():
        if v["map_at_r"] is not None:
            out[name] = {"map": v["map_at_r"], "rp": v["r_precision"]}
    return out


def load_positives(annotations_dir: Path):
    return cp.eccv_i2t(annotations_dir)


def recompute(d1: dict, arrays: dict, q) -> dict:
    """The per-kind metric blocks of d1.json, recomputed from scores.npz with the coco_probes scorers."""
    cand, dual = arrays["cand"], arrays["dual"]
    out = {"dual": cp.d1_metrics(cand, q)}
    for kind in KINDS:
        dec, null, t = arrays[f"dec_{kind}"], arrays[f"null_{kind}"], d1[kind]["tuned"]
        out[kind] = {
            "tuned": t,
            "rerank_a": cp.d1_metrics(cp.rerank(cand, cp.score_a(dec, null, t["alpha_a"])), q),
            "rerank_b": cp.d1_metrics(cp.rerank(cand, cp.score_b(dual, dec, null, t["alpha_b"], t["beta"])), q),
            "raw_decoder": cp.d1_metrics(cp.rerank(cand, dec), q),
            "grid_a": {str(a): cp.d1_metrics(cp.rerank(cand, cp.score_a(dec, null, a)), q) for a in cp.ALPHAS},
            "grid_b": {f"{a}_{b}": cp.d1_metrics(cp.rerank(cand, cp.score_b(dual, dec, null, a, b)), q)
                       for a in cp.ALPHAS for b in cp.BETAS},
        }
    return out


def checkpoint_summary(folder: Path, positives_fn, annotations_dir: Path | None) -> dict:
    d1 = json.loads((folder / "d1.json").read_text())
    arrays = dict(np.load(folder / "scores.npz")) if (folder / "scores.npz").exists() else {}
    meta = d1["meta"]
    source, blocks, check = "d1.json", d1, None
    if arrays and annotations_dir is not None:
        q = positives_fn(annotations_dir)
        n = len(arrays["R"])
        q = type(q)(q.query_image[:n], q.positives[:n], q.R[:n])
        if (np.array_equal(q.R, arrays["R"]) and np.array_equal(q.query_image, arrays["query_image"])
                and int(q.R.max()) <= arrays["cand"].shape[1]):
            blocks, source = recompute(d1, arrays, q), "scores.npz"
            check = max(abs(blocks[k][r]["map_at_r"] - d1[k][r]["map_at_r"])
                        for k in KINDS for r in ("rerank_a", "rerank_b"))
        else:
            source = "d1.json (annotations do not match scores.npz)"
    res = {"run": folder.name, "arm": meta.get("arm"), "seed": meta.get("seed"), "queries": meta.get("queries"),
           "source": source, "check_vs_json": check}
    ck = {"dual": flat_metrics(blocks["dual"])}
    sanity = {}
    for kind in KINDS:
        b = blocks[kind]
        t = d1[kind]["tuned"]
        ck[kind] = {
            "tuned_a": flat_metrics(b["rerank_a"]), "tuned_b": flat_metrics(b["rerank_b"]),
            "raw_decoder": flat_metrics(b["raw_decoder"]),
            "grid_a": {a: flat_metrics(m) for a, m in b["grid_a"].items()},
            "grid_b": {ab: flat_metrics(m) for ab, m in b["grid_b"].items()},
        }
        res.setdefault("tuned", {})[kind] = {k: t[k] for k in ("alpha_a", "alpha_b", "beta")}
        res.setdefault("collapsed_b", {})[kind] = bool(t["beta"] == 0)
        if arrays:
            sanity[kind] = float(np.abs(arrays[f"dec_{kind}"] - arrays[f"null_{kind}"]).max())
    res["metrics"], res["sanity_max_abs_dec_minus_null"] = ck, sanity
    return res


def add_diffs(ck: dict) -> dict:
    """Flat dict of a checkpoint's quantities plus the paired difference (re-ranker minus dual) of every re-ranker."""
    flat = {f"dual/{s}/{m}": v for s, d in ck["dual"].items() for m, v in d.items()}
    for kind in KINDS:
        for rr, block in ck[kind].items():
            leaves = {"": block} if rr in ("tuned_a", "tuned_b", "raw_decoder") else block
            for name, scopes in leaves.items():
                for s, d in scopes.items():
                    for m, v in d.items():
                        path = f"{kind}/{rr}" + (f"/{name}" if name else "")
                        flat[f"{path}/{s}/{m}"] = v
                        if s in ck["dual"]:
                            flat[f"{path}/{s}/{m}_diff"] = v - ck["dual"][s][m]
    return flat


# ---------------------------------------------------------------- D1: families and the kill rule
def kill_rule(stats: dict, kind: str, n_seeds: int) -> dict:
    """KILL if for every alpha the mean paired diff (a minus dual, overall mAP@R) is below minus its seed SD, and no
    alpha gives a positive mean paired diff in the top R third."""
    per_alpha = {}
    for a in cp.ALPHAS:
        key = f"{kind}/grid_a/{a}"
        allm, top = stats.get(f"{key}/all/map_diff"), stats.get(f"{key}/{TOP_THIRD}/map_diff")
        per_alpha[str(a)] = {
            "mean_diff": allm["mean"] if allm else None, "sd_diff": allm["sd"] if allm else None,
            "below_minus_sd": (allm["mean"] < -allm["sd"]) if allm and allm["sd"] is not None else None,
            "top_third_mean_diff": top["mean"] if top else None,
            "top_third_positive": (top["mean"] > 0) if top else False}
    below = [v["below_minus_sd"] for v in per_alpha.values()]
    if n_seeds < 2 or any(b is None for b in below):
        verdict = "undetermined (needs >= 2 seeds)"
    else:
        all_below, any_top = all(below), any(v["top_third_positive"] for v in per_alpha.values())
        verdict = "KILL" if all_below and not any_top else "NOT KILLED"
    return {"verdict": verdict, "per_alpha": per_alpha}


def d1_summary(d1_dir: Path, annotations_dir: Path | None, positives_fn) -> dict:
    cks = [checkpoint_summary(f.parent, positives_fn, annotations_dir) for f in sorted(d1_dir.glob("*/d1.json"))]
    fams = {}
    for ck in cks:
        fams.setdefault(family_of(ck["arm"], ck["run"]), []).append(ck)
    out = {"checkpoints": [{k: v for k, v in ck.items()} for ck in cks], "families": {}}
    for fam, group in sorted(fams.items()):
        stats = aggregate([add_diffs(ck["metrics"]) for ck in group])
        out["families"][fam] = {
            "label": label(fam), "n_checkpoints": len(group), "seeds": sorted(c["seed"] for c in group),
            "stats": stats,
            "kill_rule_a": {kind: kill_rule(stats, kind, len(group)) for kind in KINDS},
            "collapsed_b_fraction": {kind: float(np.mean([c["collapsed_b"][kind] for c in group])) for kind in KINDS},
            "sanity_max_abs_dec_minus_null": {
                kind: max((c["sanity_max_abs_dec_minus_null"].get(kind, float("nan")) for c in group), default=None)
                for kind in KINDS},
        }
    return out


# ---------------------------------------------------------------- D2
def selection_stats(s_a, s_b, blend, side: str) -> dict:
    own, other = (s_a, s_b) if side == "A" else (s_b, s_a)
    idx, dropped = cp.selection_index(blend, own, other)
    den = float((own - other).sum())
    return {"mean": float(idx.mean()) if len(idx) else None, "median": float(np.median(idx)) if len(idx) else None,
            "ratio_of_sums": float((blend - other).sum() / den) if abs(den) >= cp.TIE_TOL else None,
            "near_ties": int((np.abs(own - other) < 0.05).sum()), "n": int(len(idx)), "dropped": dropped}


def run_d2_summary(folder: Path) -> dict:
    d2 = json.loads((folder / "d2.json").read_text())
    sc = dict(np.load(folder / "d2_scores.npz")) if (folder / "d2_scores.npz").exists() else {}
    res = {"run": folder.name, "decoder": d2["meta"]["decoder"], "dual": {}}
    for cond, v in d2["dual"].items():
        bal = v["balance"]
        if f"dual_sim_{cond}" in sc:
            sims = sc[f"dual_sim_{cond}"]
            bal = float(cp.balance(sims[:, :5], sims[:, 5:]).mean())
        res["dual"][cond] = {**{f"bc{k}": 100 * x for k, x in v["both_covered"].items()},
                             "ceiling50": 100 * v["both_covered_at_50"], "balance": bal}
    if not d2["meta"]["decoder"]:
        return res
    dec = d2["decoder"]
    res["decoder"] = {"balance": {}, "score_gap": {}, "rerank_bc": {}, "selection": {}}
    for cond in d2["dual"]:
        bal, gap = dec["balance_parallel"][cond], dec["score_gap"][cond]
        if f"dec_par_{cond}" in sc:
            par = sc[f"dec_par_{cond}"]
            bal = float(cp.balance(par[:, :5], par[:, 5:]).mean())
            gap = float((par[:, :5].mean(1) - par[:, 5:].mean(1)).mean())
        res["decoder"]["balance"][cond], res["decoder"]["score_gap"][cond] = bal, gap
        if cond in dec["both_covered_rerank"]:
            res["decoder"]["rerank_bc"][cond] = {f"bc{k}": 100 * x for k, x in dec["both_covered_rerank"][cond].items()}
    for j, entry in dec["selection"].items():
        out = {}
        if f"sel_cA_j{j}_sA" in sc:
            parts = {s: [sc[f"sel_c{s}_j{j}_{n}"] for n in ("sA", "sB", "blend")] for s in ("A", "B")}
            for s in ("A", "B"):
                out[f"c{s}"] = selection_stats(*parts[s], s)
            own = np.concatenate([parts["A"][0], parts["B"][1]])
            other = np.concatenate([parts["A"][1], parts["B"][0]])
            blend = np.concatenate([parts["A"][2], parts["B"][2]])
            idx, dropped = cp.selection_index(blend, own, other)
            den = float((own - other).sum())
            out["pooled"] = {"mean": float(idx.mean()), "median": float(np.median(idx)),
                             "ratio_of_sums": float((blend - other).sum() / den) if abs(den) >= cp.TIE_TOL else None,
                             "near_ties": int((np.abs(own - other) < 0.05).sum()), "n": int(len(idx)),
                             "dropped": dropped}
        else:
            out = {k: {m: v for m, v in e.items() if m in ("mean", "median", "ratio_of_sums", "near_ties", "n", "dropped")}
                   for k, e in entry.items() if isinstance(e, dict)}
        res["decoder"]["selection"][j] = out
    return res


def d2_kill_rule(fams: dict) -> dict:
    """The review's rule: C's both-covered@k within 2 points of, or above, the best decoder family's (after re-ranker
    (a)) at every k. Evaluated per condition; KILL when it holds in every condition."""
    c = fams.get("s2_contrastive")
    decoders = {f: v for f, v in fams.items() if v["stats"].get("decoder_present")}
    if c is None or not decoders:
        return {"verdict": "n/a (needs the contrastive family and a decoder family)"}
    per_cond, ceilings = {}, {}
    for cond in c["conds"]:
        rows = {}
        for k in D2_KS:
            c_val = c["stats"][f"dual/{cond}/bc{k}"]["mean"]
            best_fam, best = max(((f, v["stats"][f"decoder/rerank_bc/{cond}/bc{k}"]["mean"]) for f, v in decoders.items()
                                  if f"decoder/rerank_bc/{cond}/bc{k}" in v["stats"]), key=lambda x: x[1],
                                 default=(None, None))
            rows[str(k)] = {"C": c_val, "best_decoder": best, "best_family": best_fam,
                            "C_within_or_above": None if best is None else c_val >= best - COVER_TOL}
        ok = [r["C_within_or_above"] for r in rows.values()]
        per_cond[cond] = {"by_k": rows, "holds_at_every_k": None if None in ok else all(ok)}
        ceilings[cond] = c["stats"][f"dual/{cond}/ceiling50"]["mean"]
    holds = [v["holds_at_every_k"] for v in per_cond.values()]
    verdict = "n/a" if None in holds else ("KILL" if all(holds) else "NOT KILLED")
    return {"verdict": verdict, "C_ceiling50": ceilings, "per_condition": per_cond,
            "caveat": "The both-covered@50 ceiling is about 1 to 5 percent, so the rule is floor-bound: all values sit "
                      "near zero and a 2-point margin is easily met."}


def d2_summary(d2_dir: Path) -> dict:
    runs = [run_d2_summary(f.parent) for f in sorted(d2_dir.glob("*/d2.json"))]
    groups = {}
    for r in runs:
        groups.setdefault(family_of(None, r["run"]), []).append(r)
    fams = {}
    for fam, group in sorted(groups.items()):
        flats = [flatten({k: v for k, v in r.items() if k not in ("run", "decoder") or isinstance(v, dict)}) for r in group]
        stats = aggregate(flats)
        stats["decoder_present"] = group[0]["decoder"] is not False and isinstance(group[0]["decoder"], dict)
        fams[fam] = {"label": label(fam), "n_runs": len(group), "stats": stats,
                     "conds": list(group[0]["dual"]), "runs": [r["run"] for r in group]}
    out = {"families": {f: {k: v for k, v in e.items()} for f, e in fams.items()}}
    out["kill_rule"] = d2_kill_rule(fams)
    none = fams.get("s2_none_txt80")
    if none and none["stats"]["decoder_present"]:
        lams = [f"blend_{lam}" for lam in cp.LAMBDAS]
        vals = [none["stats"][f"decoder/balance/{c}"]["mean"] for c in lams]
        out["none_control"] = {"decoder_balance_by_lambda": dict(zip(lams, vals)), "range": max(vals) - min(vals)}
    return out


# ---------------------------------------------------------------- markdown
def d1_markdown(d1: dict) -> list[str]:
    L = ["## D1: decoder re-ranking on ECCV Caption i2t", ""]
    L += ["### Kill rule for re-ranker (a)", "",
          "KILL if for every alpha the mean paired difference (a minus dual, overall mAP@R) is below minus its seed SD, "
          "and no alpha gives a positive mean paired difference in the top R third (R >= 21).", "",
          "| family | score | verdict | seeds | diff by alpha (mean / SD) | top-third diff by alpha |", "|---|---|---|---|---|---|"]
    for fam, e in d1["families"].items():
        for kind in KINDS:
            k = e["kill_rule_a"][kind]
            diffs = "; ".join(f"{a}: {fmt(v['mean_diff'])} / {fmt(v['sd_diff'])}" for a, v in k["per_alpha"].items())
            tops = "; ".join(f"{a}: {fmt(v['top_third_mean_diff'])}" for a, v in k["per_alpha"].items())
            L.append(f"| {e['label']} | {kind} | {k['verdict']} | {len(e['seeds'])} | {diffs} | {tops} |")
    L += ["", "### Re-ranker (b) collapse and the fusion_none sanity check", "",
          "(b) is collapsed for a checkpoint when its tuned beta is 0 (it then equals the dual order).", "",
          "| family | checkpoints | collapsed (b) parallel | collapsed (b) ratio | max abs(dec - null) parallel | ratio |",
          "|---|---|---|---|---|---|"]
    for fam, e in d1["families"].items():
        s = e["sanity_max_abs_dec_minus_null"]
        L.append(f"| {e['label']} | {e['n_checkpoints']} | {e['collapsed_b_fraction']['parallel']:.2f} | "
                 f"{e['collapsed_b_fraction']['ratio']:.2f} | {fmt(s['parallel'], 6)} | {fmt(s['ratio'], 6)} |")
    L += ["", "### Per-family means over seeds (mAP@R, R-Precision in percent; mean +- seed SD)", ""]
    for fam, e in d1["families"].items():
        st = e["stats"]
        L += [f"#### {e['label']} ({fam}, seeds {e['seeds']})", "",
              "| score | order | " + " | ".join(f"mAP@R {s}" for s in SCOPES) + " | R-Prec all |",
              "|---|---|" + "---|" * (len(SCOPES) + 1)]
        L.append("| both | dual | " + " | ".join(pm(st.get(f"dual/{s}/map")) for s in SCOPES)
                 + f" | {pm(st.get('dual/all/rp'))} |")
        for kind in KINDS:
            for rr, name in (("tuned_a", "(a) tuned"), ("tuned_b", "(b) tuned"), ("raw_decoder", "decoder only")):
                L.append(f"| {kind} | {name} | " + " | ".join(pm(st.get(f"{kind}/{rr}/{s}/map")) for s in SCOPES)
                         + f" | {pm(st.get(f'{kind}/{rr}/all/rp'))} |")
        L += ["", "Paired difference of (a) minus dual by alpha (overall mAP@R, mean +- seed SD):", "",
              "| score | " + " | ".join(f"alpha {a}" for a in cp.ALPHAS) + " |", "|---|" + "---|" * len(cp.ALPHAS)]
        for kind in KINDS:
            L.append(f"| {kind} | " + " | ".join(pm(st.get(f"{kind}/grid_a/{a}/all/map_diff")) for a in cp.ALPHAS) + " |")
        L.append("")
    L.append("The (b) grid over (alpha, beta) is in hb_coco.json (families.<f>.stats, keys <kind>/grid_b/<alpha>_<beta>/...).")
    return L + [""]


def d2_markdown(d2: dict) -> list[str]:
    L = ["## D2: blend probe", ""]
    k = d2["kill_rule"]
    L += [f"### Kill rule: C's both-covered@k within {COVER_TOL:.0f} points of, or above, the best decoder's at every k", "",
          f"Verdict: {k['verdict']}", ""]
    if "per_condition" in k:
        L += [k["caveat"], "", "| condition | C ceiling@50 | holds at every k | C / best decoder at k=5, 10, 20 |", "|---|---|---|---|"]
        for cond, v in k["per_condition"].items():
            cells = "; ".join(f"{kk}: {fmt(r['C'])} / {fmt(r['best_decoder'])} ({r['best_family']})"
                              for kk, r in v["by_k"].items())
            L.append(f"| {cond} | {fmt(k['C_ceiling50'][cond])} | {v['holds_at_every_k']} | {cells} |")
        L.append("")
    if "none_control" in d2:
        n = d2["none_control"]
        L += ["### fusion_none control: decoder balance by lambda", "",
              "| " + " | ".join(n["decoder_balance_by_lambda"]) + " | range |", "|" + "---|" * (len(n["decoder_balance_by_lambda"]) + 1),
              "| " + " | ".join(f"{v:.3f}" for v in n["decoder_balance_by_lambda"].values()) + f" | {n['range']:.3f} |", ""]
    for fam, e in d2["families"].items():
        st, conds = e["stats"], e["conds"]
        L += [f"### {e['label']} ({fam}, {e['n_runs']} run(s); mean +- SD over seeds)", "",
              "| condition | dual bc@5 | bc@10 | bc@20 | ceiling@50 | dual balance |"
              + (" dec balance | score gap | rerank bc@5 | bc@10 | bc@20 |" if st["decoder_present"] else ""),
              "|---|" + "---|" * (5 + (5 if st["decoder_present"] else 0))]
        for c in conds:
            row = [pm(st.get(f"dual/{c}/bc{kk}")) for kk in D2_KS] + [pm(st.get(f"dual/{c}/ceiling50")),
                                                                      pm(st.get(f"dual/{c}/balance"), 3)]
            if st["decoder_present"]:
                row += [pm(st.get(f"decoder/balance/{c}"), 3), pm(st.get(f"decoder/score_gap/{c}"), 3)]
                row += [pm(st.get(f"decoder/rerank_bc/{c}/bc{kk}")) for kk in D2_KS]
            L.append(f"| {c} | " + " | ".join(row) + " |")
        if st["decoder_present"]:
            L += ["", "Selection index (blend 0.5; median and ratio of sums over pooled cA and cB items; near-ties, dropped):", "",
                  "| j | median | ratio of sums | near-ties | dropped |", "|---|---|---|---|---|"]
            for j in sorted({key.split("/")[2] for key in st if key.startswith("decoder/selection/")}):
                p = f"decoder/selection/{j}/pooled"
                L.append(f"| {j} | {pm(st.get(p + '/median'), 3)} | {pm(st.get(p + '/ratio_of_sums'), 3)} | "
                         f"{pm(st.get(p + '/near_ties'), 1)} | {pm(st.get(p + '/dropped'), 1)} |")
        L.append("")
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d1-dir", type=Path, default=None)
    ap.add_argument("--d2-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--annotations-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    ann = args.annotations_dir or DEFAULT_ANNOTATIONS
    ann = ann if (ann / 'coco_karpathy_test.json').exists() else None  # else d1.json metrics are used
    result, md, notes = {}, ["# H-b COCO probes: D1 and D2 summaries over seeds", ""], []
    if args.d1_dir and args.d1_dir.is_dir() and list(args.d1_dir.glob("*/d1.json")):
        result["d1"] = d1_summary(args.d1_dir, ann, load_positives)
        md += d1_markdown(result["d1"])
    else:
        notes.append(f"D1 skipped: no <run>/d1.json under {args.d1_dir}")
    if args.d2_dir and args.d2_dir.is_dir() and list(args.d2_dir.glob("*/d2.json")):
        result["d2"] = d2_summary(args.d2_dir)
        md += d2_markdown(result["d2"])
    else:
        notes.append(f"D2 skipped: no <run>/d2.json under {args.d2_dir}")
    result["notes"] = notes
    md += [f"Note: {n}" for n in notes]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "hb_coco.json").write_text(json.dumps(result, indent=1))
    (args.out / "hb_coco.md").write_text("\n".join(md) + "\n")
    for n in notes:
        print(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
