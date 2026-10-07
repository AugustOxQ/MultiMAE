"""D4, D6 and D7 tables from the encode folders written by scripts/hb_encode.py (CPU).

  python scripts/hb_table.py --encoded-root DIR --out DIR [--annotations-dir DIR] [--al28-csv PATH] [--B 10000]

Writes <out>/hb_d4.json (no bootstrap draws) and <out>/hb_d4.md (the decision first)."""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repo root, for `import mmae`

import numpy as np  # noqa: E402

from mmae.engine.hb import data, tables  # noqa: E402

L = tables.LABELS


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def fp(p, B):
    return ("< 1e-4" if B >= 10000 else f"< {1 / B:g}") if p == 0 else f"{p:.4f}"


def ms(values, nd=3):
    v = [x for x in values if x is not None]
    if not v:
        return "n/a"
    return f"{np.mean(v):.{nd}f} ({np.std(v, ddof=1):.{nd}f})" if len(v) > 1 else f"{v[0]:.{nd}f} (n=1)"


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    return lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


def render(res: dict) -> str:
    B, arms, out = res["settings"]["B"], res["arms"], []
    d = res["decision"]
    out += ["# H-b D4, D6 and D7", ""]
    verdict = "SUPPORTED" if d["support"] else "NOT SUPPORTED (kill rule: no D4 advantage over a captioner and a probe)"
    out += [f"## Decision: H-b on D4 {verdict}", ""]
    if d["support"]:
        out += [f"Supporting metric: {d['supporting_metric']}.", ""]
    if d["provisional"]:
        out += [f"**{d['provisional']}**; the rule needs {tables.MIN_SEEDS} seeds per arm.", ""]
    out += [d["rule"] + f". Hierarchical bootstrap, B = {B}. Differences are ML-80 minus the comparator.", ""]
    rows = [[k.split("/")[0], k.split("/")[1], f(t["diff"], 4), f"[{f(t['ci_low'], 4)}, {f(t['ci_high'], 4)}]",
             fp(t["p"], B), fp(t["holm_p"], B), "ML-80" if t["favours"] else "no"]
            for k, t in res["tests"].items() if k != "secondary"]
    out += table(["metric", "comparator", "diff", "95% interval", "p", "Holm p", "favours ML-80 (Holm < 0.05)"], rows)
    if res["tests"]["secondary"]:
        out += ["Secondary, no Holm: ML-80+MAE minus ML-80.", ""]
        out += table(["metric", "diff", "95% interval", "p"],
                     [[k.split("/")[0], f(t["diff"], 4), f"[{f(t['ci_low'], 4)}, {f(t['ci_high'], 4)}]", fp(t["p"], B)]
                      for k, t in res["tests"]["secondary"].items()])
    seeds = res["settings"]["seeds"]
    out += ["Seeds: " + ", ".join(f"{L[a]} {s}" for a, s in seeds.items()) + f"; probe comparator on seeds "
            f"{res['settings']['complete_seeds']} (seeds present in every arm).", ""]
    strong = res["probes"]["strongest"]
    out += ["Strongest probe per seed: " + ", ".join(f"{s}: {L[v['arm']]} (val NLL {v['val_nll']:.4f})" for s, v in strong.items()), ""]

    keys = ("jsd", "entropy_spearman", "kl", "tvd", "rank_cs")

    def readout_rows(names_for):
        rows = []
        for arm in tables.ARMS:
            if arm not in arms:
                continue
            for name in names_for(arm):
                rs = [a["readouts"][name] for a in arms[arm].values() if name in a["readouts"]]
                if rs:
                    rows.append([L[arm], name + (" (primary)" if name == next(iter(arms[arm].values()))["primary"] else "")]
                                + [ms([r["metrics"][k] for r in rs]) for k in keys]
                                + [ms([r["val_nll"] for r in rs], 4), ms([r["T"] for r in rs], 2) + ("*" if any(r["t_at_bound"] for r in rs) else "")])
        return rows

    head = ["arm", "readout", "JSD", "entropy rho", "KL(h||m)", "TVD", "RankCS", "val NLL", "T"]
    out += ["## D4: primary readouts and baselines (mean over seeds, seed SD in brackets; JSD lower is better)", ""]
    out += table(head, readout_rows(lambda a: [next(iter(arms[a].values()))["primary"], "prior"] if a == "ml80"
                                    else [next(iter(arms[a].values()))["primary"]]))
    out += ["## D4: every readout, each with its own temperature", ""]
    out += table(head, readout_rows(lambda a: list(next(iter(arms[a].values()))["readouts"])))
    out += ["`*` the fitted temperature sits at a bound of the search range (|log T| >= 2.99) in at least one seed.", ""]

    out += ["## Strongest probe (comparator) metrics", ""]
    rows = []
    for s, v in strong.items():
        m = arms[v["arm"]][s]["readouts"]["probe_full"]["metrics"]
        rows.append([s, L[v["arm"]], f(m["jsd"]), f(m["entropy_spearman"]), f(arms[v["arm"]][s]["probe"]["val_nll"], 4),
                     arms[v["arm"]][s]["probe"]["weight_decay"]])
    out += table(["seed", "arm", "JSD", "entropy rho", "val NLL (T = 1)", "chosen weight decay"], rows)

    out += ["## Variants of the primary readouts (mean over seeds)", ""]
    rows = []
    for arm in tables.ARMS:
        if arm in arms:
            name = next(iter(arms[arm].values()))["primary"]
            rs = [a["readouts"][name] for a in arms[arm].values()]
            rows.append([L[arm], name, ms([r["metrics"]["jsd"] for r in rs]), ms([r["metrics"]["entropy_spearman"] for r in rs]),
                         ms([r["other_dropped"]["jsd"] for r in rs]), ms([r["other_dropped"]["entropy_spearman"] for r in rs]),
                         ms([r["named8"]["jsd"] for r in rs]), ms([r["named8"]["entropy_spearman"] for r in rs]),
                         rs[0]["named8"]["n"]])
    out += table(["arm", "readout", "JSD", "rho", "JSD other dropped", "rho other dropped", "JSD 8-named", "rho 8-named",
                  "paintings (8-named)"], rows)

    out += ["## By thirds of human entropy (primary readouts; JSD / entropy rho, mean over seeds)", ""]
    rows = []
    for arm in tables.ARMS:
        if arm in arms:
            name = next(iter(arms[arm].values()))["primary"]
            cells = []
            for t in range(3):
                th = [a["readouts"][name]["metrics"]["thirds"][t] for a in arms[arm].values()]
                th = [x for x in th if x]
                cells.append(f"{ms([x['jsd'] for x in th])} / {ms([x['entropy_spearman'] for x in th])}")
            rows.append([L[arm], name] + cells)
    out += table(["arm", "readout", "low entropy", "middle", "high entropy"], rows)

    ref = res["references"]
    out += ["## Human references", ""]
    rows = [["split-half ceiling", ref["split_half"]["n"], f(ref["split_half"]["jsd"]), f(ref["split_half"]["entropy_spearman"]), "n/a", "n/a"]]
    if ref["english"]:
        e = ref["english"]
        rows.append(["English labels", e["n"], f(e["jsd"]), f(e["entropy_spearman"]), f(e["tvd"]), f(e["rank_cs"])])
    out += table(["reference", "paintings", "JSD", "entropy rho", "TVD", "RankCS"], rows)
    out += [f"Split-half: 10 random splits of each painting's non-English votes (seed {ref['split_half']['seed']}), "
            "halves compared with each other; it understates a full sample."]
    if ref["english"]:
        out += [f"English reference: {ref['english']['n_dropped_without_english']} paintings without English labels left out."]
    out += [""]

    d6 = res["d6"]
    out += ["## D6: view sampling (JSD / entropy rho, mean over seeds, each readout with its own T)", ""]
    rows = []
    for arm, r in d6["arms"].items():
        rows.append([L[arm]] + [f"{f(r[n]['mean']['jsd'])} / {f(r[n]['mean']['entropy_spearman'])}"
                                for n in ("views1", "views4", "views16", "full")]
                    + [f"JSD {'yes' if r['k16_beats_k1']['jsd'] else 'no'}, rho {'yes' if r['k16_beats_k1']['entropy_spearman'] else 'no'}"])
    out += table(["arm", "K = 1", "K = 4", "K = 16", "full image", "K = 16 beats K = 1"], rows)
    out += table(["arm", "Spearman(between-view MI, human entropy)", "95% interval", "kill (interval includes zero)"],
                 [[L[a], f(r["mi"]["spearman_mean"]), f"[{f(r['mi']['ci_low'])}, {f(r['mi']['ci_high'])}] (NaN draws: {r['mi']['nan_draws']})",
                   "yes" if r["mi"]["interval_includes_zero"] else "no"] for a, r in d6["arms"].items()])
    names = {"parcap_views16": "Par-cap, 16 views", "c_prompt_views16": "C prompt softmax, 16 views",
             "strongest_probe_views16": "strongest probe, 16 views"}
    out += ["Controls on the same 16 views:", ""]
    out += table(["control", "JSD", "entropy rho"], [[names[k], f(v["jsd"]), f(v["entropy_spearman"])] for k, v in d6["controls"].items()])
    if d6["missing_controls"] or d6["probe_pool_note"]:
        out += [f"Missing controls: {', '.join(d6['missing_controls']) or 'none'}. {d6['probe_pool_note'] or ''}", ""]

    out += ["## D7: partial captions (log-loss lower is better; difference is real minus null image)", ""]
    for arm, r in res["d7"]["arms"].items():
        out += [f"### {L[arm]} (temperature from {r['readout']}, seeds {r['seeds']})", ""]
        rows = []
        for pattern, row in r["patterns"].items():
            diff = row.get("real_minus_null")
            rows.append([pattern, row["n"], f(row["real"]["log_loss"]), f(row["null"]["log_loss"]),
                         f(row["real"]["accuracy"]), f(row["null"]["accuracy"]),
                         f"{f(diff['diff'])} [{f(diff['ci_low'])}, {f(diff['ci_high'])}]" if diff else "n/a",
                         ("no image benefit" if row["no_image_benefit"] else "image helps") if "no_image_benefit" in row else ""])
        out += table(["pattern", "n", "log-loss real", "log-loss null", "acc real", "acc null", "difference [95%]", "kill check"], rows)
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoded-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--annotations-dir", default=None)
    ap.add_argument("--al28-csv", default=data.AL28_CSV)
    ap.add_argument("--min-votes", type=int, default=20)
    ap.add_argument("--B", type=int, default=10000)
    args = ap.parse_args(argv)
    res = clean(tables.build(args.encoded_root, args.annotations_dir, args.al28_csv, args.B, args.min_votes))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "hb_d4.json").write_text(json.dumps(res, indent=1))
    (args.out / "hb_d4.md").write_text(render(res))
    d = res["decision"]
    print(f"decision: support={d['support']} metric={d['supporting_metric']} {d['provisional'] or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
