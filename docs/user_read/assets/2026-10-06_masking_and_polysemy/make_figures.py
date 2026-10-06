"""Figures for docs/user_read/2026-10-06_masking_and_polysemy.md, read from the same run folders and analysis
outputs as the full report (docs/reports/assets/build_2026-10-06_masking_stages_1_2.py)."""
import glob
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
ML = REPO / "res/coco/multimae/ml_improve"
BASE = REPO / "res/coco/multimae/default"
GREY, BLUE, ORANGE, TEAL, PURPLE = "#8c8c8c", "#3b6fb6", "#e07b39", "#2a9d8f", "#7b5ea7"


def seed(run_dir):
    for line in open(run_dir / "config.yaml"):
        if line.startswith("seed:"):
            return int(line.split()[1])


def runs(root, arm):
    out = {}
    for p in sorted(root.glob(f"*_{arm}/run.json")):
        rj = json.loads(p.read_text())
        if rj.get("status") != "completed":
            continue
        test = dict(rj.get("results", {}).get("test", {}))
        test.update(rj.get("eval", {}).get("test", {}))
        out[seed(p.parent)] = test
    return out


def mapr(rows):
    return [t.get("test/eccv/map_at_r", t.get("eccv/map_at_r")) for t in rows.values()]


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


# 1. Dose curve (Stage 1)
base_ml, base_c = runs(BASE, "fusion_multilearner"), runs(BASE, "contrastive")
dose = [(15, mapr(base_ml))] + [(r, mapr(runs(ML, f"multilearner_txt{r}"))) for r in (25, 40, 60, 80, 90)]
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot([d[0] for d in dose], [st.mean(d[1]) for d in dose], "o-", color=ORANGE, label="fusion model")
ax.axhline(st.mean(mapr(base_c)), color=GREY, ls="--", label="contrastive only")
ax.set_xlabel("share of caption words hidden (%)")
ax.set_ylabel("main metric (ECCV mAP@R)")
ax.set_title("Hiding more caption words raised the main metric")
ax.legend(frameon=False)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "1_dose.png", dpi=150)

# 2. Stage 2 per arm with per-seed dots
arms = [("contrastive only", "s2_contrastive", GREY), ("fusion model", "s2_multilearner", BLUE),
        ("80% hidden", "s2_txt80", ORANGE), ("image reconstruction off", "s2_mae0", PURPLE)]
fig, ax = plt.subplots(figsize=(6, 3.8))
for i, (label, arm, color) in enumerate(arms):
    vals = mapr(runs(ML, arm))
    ax.bar(i, st.mean(vals), color=color, alpha=0.75, width=0.6)
    ax.scatter([i] * len(vals), vals, color="black", s=14, zorder=3)
ax.set_xticks(range(len(arms)), [a[0] for a in arms], fontsize=8)
ax.set_ylim(36.6, 38.4)
ax.set_ylabel("main metric (ECCV mAP@R)")
ax.set_title("Only the 80% model beat both baselines")
ax.text(0.01, 0.97, "dots: individual seeds (80%: 5 seeds, others 3)", transform=ax.transAxes, ha="left", va="top", fontsize=7)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "2_confirmation.png", dpi=150)

# 3. Image-encoder swaps: each model's image encoder with the contrastive text encoder of the same seed
diag = json.loads((REPO / "res/coco/diagnostics/stage2_controls/diagnostics.json").read_text())
img = {}
for key, m in diag["swaps"].items():
    if "__img+" in key:
        img.setdefault(key.split("__img+")[0].split("_", 2)[2], []).append(m["eccv/map_at_r"])
bars = [("fusion model", "s2_multilearner", BLUE), ("80% hidden", "s2_txt80", ORANGE),
        ("80%, no fusion", "s2_none_txt80", TEAL), ("80%, decoder sees\nfull image", "s2_m1clean_txt80", TEAL),
        ("80%, full image,\nno signal to encoder", "s2_m1detached_txt80", TEAL)]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for i, (label, arm, color) in enumerate(bars):
    ax.bar(i, st.mean(img[arm]), color=color, alpha=0.8, width=0.6)
    ax.scatter([i] * len(img[arm]), img[arm], color="black", s=12, zorder=3)
ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=7)
ax.set_ylim(36.6, 38.4)
ax.set_ylabel("main metric, image encoder swapped in")
ax.set_title("Only the 80% model's image encoder improved")
ax.text(0.99, 0.95, "teal: controls at 80%", transform=ax.transAxes, ha="right", fontsize=7, color=TEAL)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "3_controls.png", dpi=150)

# 4. Gain by number of valid matches
strat = json.loads((REPO / "tests/20261003_ml_improve/stratified_eccv_results.json").read_text())
fig, ax = plt.subplots(figsize=(6, 3.8))
for key, label, color, dx in (("i2t:R", "image query to captions", ORANGE, -0.06), ("t2i:R", "caption query to images", BLUE, 0.06)):
    r = strat["diffs"][key]["s2_txt80-s2_multilearner"]
    x = [i + dx for i in range(3)]
    lo = [m - c[0] for m, c in zip(r["mean"], r["mean_ci"])]
    hi = [c[1] - m for m, c in zip(r["mean"], r["mean_ci"])]
    ax.errorbar(x, r["mean"], yerr=[lo, hi], fmt="o-", color=color, capsize=3, label=label)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(3), ["fewest", "middle", "most"])
ax.set_xlabel("number of valid matches per query (thirds)")
ax.set_ylabel("gain of 80% over fusion model\n(mAP@R points, 95% interval)")
ax.set_title("The gain does not grow with the number of valid matches")
ax.legend(frameon=False, fontsize=8)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "4_multiplicity.png", dpi=150)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
