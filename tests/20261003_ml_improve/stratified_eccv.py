"""Per-query ECCV Caption mAP@R, stratified by query multiplicity (R) and caption diversity.

Reuses mmae.engine.eccv for the id mapping and the eccv_caption package for the ground truth. Caption order:
caption_emb.reshape(N*K, D) is image-major, caption index i*5+j = caption j of test image i (encode_retrieval_set
returns (N, K, D) in dataset order; diagnose.py saved it flattened).
Run: OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python tests/20261003_ml_improve/stratified_eccv.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from eccv_caption import Metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mmae.data.coco import retrieval_items  # noqa: E402
from mmae.engine.eccv import CAPTIONS_FILE, check_ids, map_coco_ids  # noqa: E402

ANN = Path("/data/SSD/coco/annotations")
EMB = ROOT / "res/coco/diagnostics/stage2/embeddings"
RUNS = ROOT / "res/coco/multimae/ml_improve"
OUT = Path(__file__).resolve().parent
ARMS = ["s2_contrastive", "s2_multilearner", "s2_txt80", "s2_mae0"]
COMPARES = [("s2_txt80", "s2_multilearner"), ("s2_txt80", "s2_contrastive"), ("s2_mae0", "s2_multilearner")]
NB, K = 2000, 5
rng = np.random.default_rng(0)

items = retrieval_items(ANN, "test")
image_ids, caption_ids = map_coco_ids(items, ANN / CAPTIONS_FILE)
metrics = Metrics()
check_ids(image_ids, caption_ids, metrics)
flat_cap = caption_ids.reshape(-1)
img_index = {int(i): n for n, i in enumerate(image_ids)}
cap_index = {int(c): n for n, c in enumerate(flat_cap)}

# queries and positives exactly as the package scores them (set of gt ids, R = its size)
Q = {}
for d, qmap, pmap in (("i2t", img_index, cap_index), ("t2i", cap_index, img_index)):
    qs = sorted(metrics.eccv_gts[d])
    Q[d] = {
        "q": np.array([qmap[int(q)] for q in qs]),
        "pos": [np.array(sorted(pmap[int(p)] for p in set(metrics.eccv_gts[d][q]) if int(p) in pmap)) for q in qs],
    }
    # R counts every positive the package lists; 2 i2t positives (of 22,550) are captions outside our 25k and
    # can never be retrieved, they stay in R like in the package
    Q[d]["R"] = np.array([len(set(metrics.eccv_gts[d][q])) for q in qs])
    print(d, "queries", len(qs), "R range", Q[d]["R"].min(), Q[d]["R"].max())


def per_query_ap(image, caption, d):
    """AP@R per query exactly as eccv_caption.compute_eccv_metrics (stable descending sort, ties by index)."""
    image = torch.nn.functional.normalize(image.float(), dim=-1)
    caption = torch.nn.functional.normalize(caption.float().reshape(-1, caption.shape[-1]), dim=-1)  # (N,5,D) -> image-major (N*5,D)
    qe, ce = (image, caption) if d == "i2t" else (caption, image)
    q, pos, R = Q[d]["q"], Q[d]["pos"], Q[d]["R"]
    depth = int(R.max())
    out = np.zeros(len(q))
    for s in range(0, len(q), 256):
        sc = qe[q[s:s + 256]] @ ce.T
        top = torch.sort(sc, dim=1, descending=True, stable=True).indices[:, :depth].numpy()
        for j in range(top.shape[0]):
            r = R[s + j]
            rel = np.isin(top[j, :r], pos[s + j]).astype(float)
            out[s + j] = (np.cumsum(rel) / np.arange(1, r + 1) * rel).sum() / r
    return out


# ---------- load embeddings, per-query AP
AP, SEED, REPORTED = {}, {}, {}
zs = torch.load(EMB / "zeroshot.pt")
AP["zeroshot"] = {d: per_query_ap(zs["image"], zs["caption"], d) for d in Q}
for f in sorted(EMB.glob("2026*.pt")):
    e = torch.load(f)
    AP[f.stem] = {d: per_query_ap(e["image"], e["caption"], d) for d in Q}
    SEED[f.stem] = int(e["seed"])
    r = json.load(open(RUNS / f.stem / "run.json"))["results"]["test"]
    REPORTED[f.stem] = {d: r[f"test/eccv/{d}_map_at_r"] for d in Q} | {"avg": r["test/eccv/map_at_r"]}

# ---------- validation
print("\nVALIDATION (percent; ours vs run.json)")
print("run | seed | i2t ours/rep | t2i ours/rep | avg ours/rep | max abs diff")
val_rows, ok = [], True
for name in sorted(SEED):
    o = {d: 100 * AP[name][d].mean() for d in Q}
    o["avg"] = (o["i2t"] + o["t2i"]) / 2
    diff = max(abs(o[k] - REPORTED[name][k]) for k in o)
    ok &= diff <= 0.05
    val_rows.append({"run": name, "seed": SEED[name], "ours": o, "reported": REPORTED[name], "max_abs_diff": diff})
    print(f"{name} | {SEED[name]} | {o['i2t']:.2f}/{REPORTED[name]['i2t']:.2f} | {o['t2i']:.2f}/{REPORTED[name]['t2i']:.2f} | "
          f"{o['avg']:.2f}/{REPORTED[name]['avg']:.2f} | {diff:.3f}")
zo = {d: 100 * AP["zeroshot"][d].mean() for d in Q}
print(f"zeroshot ours: i2t {zo['i2t']:.2f} t2i {zo['t2i']:.2f} avg {(zo['i2t'] + zo['t2i']) / 2:.2f}")
if not ok:
    sys.exit("VALIDATION FAILED (diff > 0.05)")
print("VALIDATION PASSED")

# ---------- arm x seed arrays
by_arm = {a: {} for a in ARMS}
for name, s in SEED.items():
    by_arm[name.split("_", 2)[2]][s] = AP[name]
seeds = sorted(by_arm[ARMS[0]])
assert all(sorted(by_arm[a]) == seeds for a in ARMS), by_arm.keys()
arm_ap = {a: {d: np.stack([by_arm[a][s][d] for s in seeds]) for d in Q} for a in ARMS}  # (S, nq) fraction


def pair_diff(a, b, d):
    return 100 * (arm_ap[a][d] - arm_ap[b][d])  # (S, nq), seed-matched


# ---------- strata
def tertile_labels(x, cuts):
    return np.digitize(x, cuts, right=True)  # 0: <=c1, 1: <=c2, 2: >c2


strat = {}
for d in Q:
    R = Q[d]["R"]
    c = np.quantile(R, [1 / 3, 2 / 3], method="lower")
    lab = tertile_labels(R, c)
    names = [f"R<={c[0]}", f"{c[0] + 1}<=R<={c[1]}", f"R>={c[1] + 1}"]
    strat[f"{d}:R"] = (lab, names, R.astype(float))
# diversity of the query image's 5 captions under zero-shot caption embeddings (i2t only)
cz = torch.nn.functional.normalize(zs["caption"].float(), dim=-1).reshape(len(image_ids), K, -1)
sim = cz @ cz.transpose(1, 2)
div_all = (1 - (sim.sum((1, 2)) - sim.diagonal(dim1=1, dim2=2).sum(1)) / (K * (K - 1))).numpy()
div = div_all[Q["i2t"]["q"]]
dc = np.quantile(div, [1 / 3, 2 / 3])
dl = tertile_labels(div, dc)
strat["i2t:diversity"] = (dl, [f"div<={dc[0]:.3f}", f"{dc[0]:.3f}<div<={dc[1]:.3f}", f"div>{dc[1]:.3f}"], div)
print("\nstrata counts:")
for k, (lab, names, _) in strat.items():
    print(k, [(n, int((lab == i).sum())) for i, n in enumerate(names)])
print("corr(R, diversity) i2t:", np.corrcoef(Q["i2t"]["R"], div)[0, 1])

# ---------- bootstrap
def boot_indices(n):
    return rng.integers(0, n, size=(NB, n))


BI = {d: boot_indices(len(Q[d]["R"])) for d in Q}


def stat_set(vals, lab, x, idx=None):
    """vals: per-query seed-averaged quantity (nq,). Returns tertile means, hi-lo, slope on x."""
    if idx is not None:
        vals, lab, x = vals[idx], lab[idx], x[idx]
    m = [vals[lab == i].mean() for i in range(3)]
    xc = x - x.mean()
    slope = (xc * (vals - vals.mean())).sum() / (xc ** 2).sum()
    return m, m[2] - m[0], slope


def ci(a):
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def analyse(vals_seeds, d, key):
    """vals_seeds (S, nq) percent diffs (or APs). Seed-average first, then stratify."""
    lab, names, x = strat[key]
    v = vals_seeds.mean(0)
    m, inter, slope = stat_set(v, lab, x)
    bm, bi, bs = [], [], []
    for idx in BI[d]:
        a, b, c = stat_set(v, lab, x, idx)
        bm.append(a); bi.append(b); bs.append(c)
    bm = np.array(bm)
    per_seed = [stat_set(vals_seeds[s], lab, x) for s in range(len(seeds))]
    return {
        "strata": names, "n": [int((lab == i).sum()) for i in range(3)],
        "mean": [float(z) for z in m], "mean_ci": [ci(bm[:, i]) for i in range(3)],
        "overall": float(v.mean()), "overall_ci": ci(v[BI[d]].mean(1)),
        "high_minus_low": float(inter), "high_minus_low_ci": ci(bi),
        "slope_per_unit": float(slope), "slope_ci": ci(bs),
        "per_seed_high_minus_low": [float(p[1]) for p in per_seed],
        "per_seed_slope": [float(p[2]) for p in per_seed],
        "per_seed_strata_means": [[float(z) for z in p[0]] for p in per_seed],
    }


results = {"validation": val_rows, "zeroshot_percent": zo, "seeds": seeds, "n_bootstrap": NB,
           "strata_defs": {k: v[1] for k, v in strat.items()}, "arms": {}, "diffs": {}}
for key in strat:
    d = key.split(":")[0]
    results["arms"][key] = {a: analyse(100 * arm_ap[a][d], d, key) for a in ARMS}
    results["diffs"][key] = {f"{a}-{b}": analyse(pair_diff(a, b, d), d, key) for a, b in COMPARES}

json.dump(results, open(OUT / "stratified_eccv_results.json", "w"), indent=1)

# ---------- markdown summary
print("\n## Mean AP@R (percent, seed-averaged) per stratum")
for key, arms in results["arms"].items():
    names, n = results["strata_defs"][key], arms[ARMS[0]]["n"]
    print(f"\n{key}\n| arm | " + " | ".join(f"{nm} (n={c})" for nm, c in zip(names, n)) + " | all |\n|---|---|---|---|---|")
    for a in ARMS:
        print(f"| {a} | " + " | ".join(f"{z:.2f}" for z in arms[a]["mean"]) + f" | {arms[a]['overall']:.2f} |")
print("\n## Seed-matched differences (percentage points), 95% bootstrap CI over queries")
for key, ds in results["diffs"].items():
    names = results["strata_defs"][key]
    print(f"\n{key}\n| diff | " + " | ".join(names) + " | all | high-low | slope per unit | per-seed high-low |\n|---|---|---|---|---|---|---|---|")
    for pair, r in ds.items():
        cells = [f"{m:+.2f} [{c[0]:+.2f},{c[1]:+.2f}]" for m, c in zip(r["mean"], r["mean_ci"])]
        print(f"| {pair} | " + " | ".join(cells) + f" | {r['overall']:+.2f} [{r['overall_ci'][0]:+.2f},{r['overall_ci'][1]:+.2f}] | "
              f"{r['high_minus_low']:+.2f} [{r['high_minus_low_ci'][0]:+.2f},{r['high_minus_low_ci'][1]:+.2f}] | "
              f"{r['slope_per_unit']:+.3f} [{r['slope_ci'][0]:+.3f},{r['slope_ci'][1]:+.3f}] | "
              + ", ".join(f"{z:+.2f}" for z in r["per_seed_high_minus_low"]) + " |")

# ---------- figure
keys = list(strat)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, key in zip(axes, keys):
    names, n = results["strata_defs"][key], results["diffs"][key][f"{COMPARES[0][0]}-{COMPARES[0][1]}"]["n"]
    for off, (pair, col) in zip((-0.15, 0.15), (("s2_txt80-s2_multilearner", "tab:orange"), ("s2_txt80-s2_contrastive", "tab:blue"))):
        r = results["diffs"][key][pair]
        m = np.array(r["mean"]); c = np.array(r["mean_ci"])
        ax.errorbar(np.arange(3) + off, m, yerr=[m - c[:, 0], c[:, 1] - m], fmt="o", capsize=4, color=col, label=pair.replace("s2_", ""))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{nm}\nn={c}" for nm, c in zip(names, n)], fontsize=8)
    ax.set_title(key); ax.set_ylabel("mAP@R difference (pp), seed-matched, mean of 3 seeds")
    ax.set_xlabel("stratum (" + ("number of ECCV positives R" if key.endswith("R") else "caption diversity of query image") + ")")
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "stratified_eccv.png", dpi=130)
