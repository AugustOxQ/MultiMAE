"""Figures, figure data and tables of docs/reports/auto/v1/2026-10-03_baselines.md, from the run folders.

Reads res/coco/multimae/default/*/ (config.yaml, run.json, metrics.jsonl) and res/coco/zeroshot/clip_b32_test.json.
The three seed-42 fusion runs were trained before the extended test metrics existed; their test metrics come
from run.json["eval"]["test"] (evaluate.py on checkpoints/best.pt at commit 88e2663, keys without the "test/"
prefix). Every other run has them in run.json["results"]["test"] ("test/..." and "test/retrieval/...").

Writes to docs/reports/assets/2026-10-03_baselines/:
  runs.csv               one row per run: model, seed, best epoch, duration, COCO 5k test metrics
  gradient_paths.png     Figure 1: which path each model's losses take, and what retrieval reads
  test_metrics.png       Figure 2: per-metric seed points and mean +- std, four models
  gain_vs_finetune.png   Figure 3: each model's change over the contrastive baseline, as a share of the
                         fine-tuning gain (zero-shot -> contrastive)
  val_curves.png         Figure 4: val rsum and val contrastive loss per epoch
and prints the report's tables (markdown) to stdout.

  /root/miniconda3/envs/MultiMAE/bin/python docs/reports/assets/build_2026-10-03_baselines.py
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "res" / "coco" / "multimae" / "default"
ZERO_SHOT = REPO / "res" / "coco" / "zeroshot" / "clip_b32_test.json"
OUT = Path(__file__).resolve().parent / "2026-10-03_baselines"

MODELS = ["contrastive", "fusion_none", "fusion_concat", "fusion_multilearner"]
BASELINE = "contrastive"
# Grey for the baseline; the three reconstruction models take the reference palette's first three
# categorical slots (validated all-pairs: worst CVD dE 9.2, normal-vision 24.0, light surface).
COLORS = {"contrastive": "#6f6e69", "fusion_none": "#2a78d6", "fusion_concat": "#eb6834",
          "fusion_multilearner": "#1baf7a"}
SHORT = {"contrastive": "contrastive", "fusion_none": "none", "fusion_concat": "concat",
         "fusion_multilearner": "multilearner"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# (key in the flattened test metrics, label)
METRICS = [
    ("rsum", "COCO 5K rsum"),
    ("i2t_R1", "COCO 5K i2t R@1"),
    ("t2i_R1", "COCO 5K t2i R@1"),
    ("coco1k/r1", "COCO 1K R@1"),
    ("cxc/r1", "CxC R@1"),
    ("eccv/map_at_r", "ECCV mAP@R"),
    ("eccv/rprecision", "ECCV R-Precision"),
    ("eccv/r1", "ECCV R@1"),
    ("pmrp", "PMRP"),
    ("pmrp/i2t", "PMRP i2t"),
    ("pmrp/t2i", "PMRP t2i"),
]
EXTRA = ["i2t_R5", "i2t_R10", "t2i_R5", "t2i_R10", "eccv/i2t_map_at_r", "eccv/t2i_map_at_r",
         "cxc/i2t_r1", "cxc/t2i_r1", "coco1k/rsum"]
# ECCV Caption paper (Chun et al., ECCV 2022), Table 4, zero-shot CLIP ViT-B/32 (tests/test_evaluate.py)
PAPER_TABLE4 = {"eccv/map_at_r": 26.75, "eccv/rprecision": 36.91, "eccv/r1": 67.08, "cxc/r1": 41.97,
                "coco1k/r1": 59.47, "coco5k/r1": 40.28, "pmrp": 55.32}


# ----------------------------------------------------------------------------------------------- data
def test_metrics(run: dict) -> dict:
    if "test" in run.get("eval", {}):
        return dict(run["eval"]["test"])
    return {k.removeprefix("test/").removeprefix("retrieval/"): v for k, v in run["results"]["test"].items()}


def epoch_series(path: Path) -> dict:
    """Per epoch: val rsum, val contrastive loss, the mean logged train contrastive loss; plus the last
    epoch's val MAE and MLM losses (None for contrastive) and the last logged logit scale."""
    val_rsum, val_nce, train, last_val, scale = [], [], defaultdict(list), {}, None
    for line in path.read_text().splitlines():
        m = json.loads(line)["metrics"]
        if "val/retrieval/rsum" in m:
            val_rsum.append(m["val/retrieval/rsum"])
            val_nce.append(m["val/loss_contrastive"])
            last_val = m
        elif "train/loss_contrastive" in m:
            train[int(m["epoch"])].append(m["train/loss_contrastive"])
            scale = m.get("train/logit_scale", scale)
    return {"val_rsum": val_rsum, "val_nce": val_nce,
            "train_nce": [statistics.mean(train[e]) for e in sorted(train)],
            "val_mae": last_val.get("val/loss_mae"), "val_mlm": last_val.get("val/loss_mlm"), "logit_scale": scale}


def load_runs() -> list[dict]:
    runs = []
    for folder in sorted(p for p in RUNS.iterdir() if (p / "run.json").is_file()):
        run = json.loads((folder / "run.json").read_text())
        cfg = OmegaConf.load(folder / "config.yaml")
        if run.get("status") != "completed":
            continue
        metrics = test_metrics(run)
        if "pmrp" not in metrics:
            continue
        runs.append({
            "folder": folder.name, "model": cfg.model.name, "seed": int(cfg.seed),
            "commit": run["git"]["commit"][:7], "processes": run["num_processes"],
            "best_epoch": run["results"]["best_epoch"], "epochs": int(cfg.train.epochs),
            "hours": run["duration_s"] / 3600, "test": metrics, **epoch_series(folder / "metrics.jsonl"),
        })
    return runs


def by_model(runs: list[dict], key) -> dict[str, list[float]]:
    out = defaultdict(list)
    for r in sorted(runs, key=lambda r: r["seed"]):
        out[r["model"]].append(key(r))
    return out


def welch(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """t, Welch-Satterthwaite df and two-sided p for mean(a) - mean(b)."""
    res = stats.ttest_ind(a, b, equal_var=False)
    va, vb, na, nb = statistics.variance(a), statistics.variance(b), len(a), len(b)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return float(res.statistic), df, float(res.pvalue)


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values over the whole family."""
    order = sorted(pvalues, key=pvalues.get)
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


# --------------------------------------------------------------------------------------------- tables
def print_tables(runs: list[dict], zs: dict) -> None:
    keys = [k for k, _ in METRICS]
    vals = {k: by_model(runs, lambda r, k=k: r["test"][k]) for k in keys + EXTRA}

    print("## Runs\n\n| folder | model | seed | commit | GPUs | best epoch / epochs | hours |\n|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda r: (MODELS.index(r["model"]), r["seed"])):
        print(f"| {r['folder']} | {r['model']} | {r['seed']} | {r['commit']} | {r['processes']} | "
              f"{r['best_epoch']} / {r['epochs']} | {r['hours']:.2f} |")

    print("\n## Zero-shot against the ECCV Caption paper, Table 4\n\n| metric | ours | paper | diff |\n|---|---|---|---|")
    for k, v in PAPER_TABLE4.items():
        print(f"| {k} | {zs[k]:.2f} | {v:.2f} | {zs[k] - v:+.3f} |")

    print("\n## Mean +- std over seeds (COCO 5k test)\n")
    print("| model | " + " | ".join(lbl for _, lbl in METRICS) + " |\n|---|" + "---|" * len(METRICS))
    print("| zero-shot | " + " | ".join(f"{zs[k]:.2f}" for k in keys) + " |")
    for m in MODELS:
        print(f"| {m} | " + " | ".join(
            f"{statistics.mean(vals[k][m]):.2f} ± {statistics.stdev(vals[k][m]):.2f}" for k in keys) + " |")

    print("\n## Extra metrics, mean +- std\n")
    print("| model | " + " | ".join(EXTRA) + " |\n|---|" + "---|" * len(EXTRA))
    print("| zero-shot | " + " | ".join(f"{zs[k]:.2f}" for k in EXTRA) + " |")
    for m in MODELS:
        print(f"| {m} | " + " | ".join(
            f"{statistics.mean(vals[k][m]):.2f} ± {statistics.stdev(vals[k][m]):.2f}" for k in EXTRA) + " |")

    family = {}
    for m in MODELS[1:]:
        for k in keys:
            family[(m, k)] = welch(vals[k][m], vals[k][BASELINE])[2]
    adjusted = holm(family)
    print("\n## Difference to contrastive (Welch t-test, n = 3 per arm; Holm over the 33 tests)\n")
    print("| metric | model | diff | t | df | p | Holm p | share of fine-tuning gain |\n|---|---|---|---|---|---|---|---|")
    for k, lbl in METRICS:
        ft = statistics.mean(vals[k][BASELINE]) - zs[k]
        for m in MODELS[1:]:
            d = statistics.mean(vals[k][m]) - statistics.mean(vals[k][BASELINE])
            t, df, p = welch(vals[k][m], vals[k][BASELINE])
            print(f"| {lbl} | {m} | {d:+.2f} | {t:.2f} | {df:.1f} | {p:.4f} | {adjusted[(m, k)]:.4f} | {100 * d / ft:+.1f}% |")
    print(f"\nTests with p < 0.05: {sum(p < 0.05 for p in family.values())} of {len(family)}; "
          f"Holm p < 0.05: {sum(p < 0.05 for p in adjusted.values())}")

    print("\n## Fine-tuning gain (zero-shot -> contrastive) per metric\n\n| metric | zero-shot | contrastive | gain |\n|---|---|---|---|")
    for k, lbl in METRICS:
        c = statistics.mean(vals[k][BASELINE])
        print(f"| {lbl} | {zs[k]:.2f} | {c:.2f} | {c - zs[k]:+.2f} |")

    print("\n## Fusion against fusion_none (Welch)\n\n| metric | model | diff | t | p |\n|---|---|---|---|---|")
    for k, lbl in METRICS:
        for m in ("fusion_concat", "fusion_multilearner"):
            d = statistics.mean(vals[k][m]) - statistics.mean(vals[k]["fusion_none"])
            t, _, p = welch(vals[k][m], vals[k]["fusion_none"])
            print(f"| {lbl} | {m} | {d:+.2f} | {t:.2f} | {p:.4f} |")

    print("\n## Ordering of the means (contrastive < none < concat < multilearner?)\n")
    for k, lbl in METRICS + [(k, k) for k in EXTRA]:
        means = {m: statistics.mean(vals[k][m]) for m in MODELS}
        order = " < ".join(SHORT[m] for m in sorted(MODELS, key=means.get))
        print(f"- {lbl}: {order}  {'HOLDS' if sorted(MODELS, key=means.get) == MODELS else ''}")

    print("\n## Validation (last epoch = best epoch in every run)\n\n| model | val rsum epoch 1 | val rsum epoch 10 | gain epochs 9->10 | val InfoNCE epoch 1 | val InfoNCE epoch 10 | train InfoNCE epoch 10 |\n|---|---|---|---|---|---|---|")
    for m in MODELS:
        rs = [r for r in runs if r["model"] == m]
        f = lambda xs: f"{statistics.mean(xs):.2f} ± {statistics.stdev(xs):.2f}"  # noqa: E731
        f3 = lambda xs: f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}"  # noqa: E731
        print(f"| {m} | {f([r['val_rsum'][0] for r in rs])} | {f([r['val_rsum'][-1] for r in rs])} | "
              f"{f([r['val_rsum'][-1] - r['val_rsum'][-2] for r in rs])} | {f3([r['val_nce'][0] for r in rs])} | "
              f"{f3([r['val_nce'][-1] for r in rs])} | {f3([r['train_nce'][-1] for r in rs])} |")
    print("\nEpoch-10 val reconstruction losses and final logit scale (min to max over seeds):")
    for m in MODELS:
        rs = [r for r in runs if r["model"] == m]
        span = lambda xs: f"{min(xs):.3f} to {max(xs):.3f}" if None not in xs else "n/a"  # noqa: E731
        print(f"- {m}: val MAE {span([r['val_mae'] for r in rs])}, val MLM {span([r['val_mlm'] for r in rs])}, "
              f"logit scale {span([r['logit_scale'] for r in rs])}")
    print("\nEpoch-10 val against contrastive (Welch):")
    for name, key in (("val rsum", "val_rsum"), ("val InfoNCE", "val_nce")):
        v = by_model(runs, lambda r, key=key: r[key][-1])
        for m in MODELS[1:]:
            t, df, p = welch(v[m], v[BASELINE])
            d = statistics.mean(v[m]) - statistics.mean(v[BASELINE])
            print(f"- {name}, {m}: diff {d:+.3f}, t {t:.2f}, df {df:.1f}, p {p:.4f}")
    base = [statistics.mean(r["val_rsum"][e] for r in runs if r["model"] == BASELINE) for e in range(10)]
    print("\nval rsum gap to contrastive per epoch:")
    for m in MODELS[1:]:
        gap = [statistics.mean(r["val_rsum"][e] for r in runs if r["model"] == m) - base[e] for e in range(10)]
        print(f"- {m}: " + ", ".join(f"{g:+.2f}" for g in gap))


def write_csv(runs: list[dict]) -> None:
    keys = [k for k, _ in METRICS] + EXTRA
    with open(OUT / "runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["folder", "model", "seed", "commit", "best_epoch", "hours"] + keys)
        for r in sorted(runs, key=lambda r: (MODELS.index(r["model"]), r["seed"])):
            w.writerow([r["folder"], r["model"], r["seed"], r["commit"], r["best_epoch"], f"{r['hours']:.2f}"]
                       + [f"{r['test'][k]:.4f}" for k in keys])


# -------------------------------------------------------------------------------------------- figures
def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def model_legend(fig, models=MODELS, **kw) -> None:
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=COLORS[m],
                      markeredgecolor="white", label=m) for m in models]
    fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK, **kw)


def fig_test_metrics(runs: list[dict], zs: dict) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(12, 8.6))
    for ax, (k, lbl) in zip(axes.flat, METRICS):
        vals = by_model(runs, lambda r, k=k: r["test"][k])
        base = statistics.mean(vals[BASELINE])
        ax.axhline(base, color=COLORS[BASELINE], linewidth=1, alpha=0.6, zorder=1)
        for i, m in enumerate(MODELS):
            mean, sd = statistics.mean(vals[m]), statistics.stdev(vals[m])
            ax.errorbar(i - 0.12, mean, yerr=sd, fmt="none", ecolor=COLORS[m], elinewidth=2, capsize=0, zorder=2)
            ax.plot([i - 0.24, i], [mean, mean], color=COLORS[m], linewidth=2, solid_capstyle="round", zorder=3)
            ax.scatter([i + 0.08, i + 0.16, i + 0.24], vals[m], s=28, color=COLORS[m], edgecolor="white",
                       linewidth=1.2, zorder=4)
        ax.set_xticks(range(len(MODELS)), [SHORT[m] for m in MODELS], rotation=30, ha="right")
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.set_title(lbl, fontsize=10, color=INK, loc="left", pad=17)
        ft = base - zs[k]
        ax.text(0.0, 1.03, f"zero-shot {zs[k]:.2f}, fine-tuning gain {ft:+.2f}", transform=ax.transAxes,
                fontsize=7.5, color=MUTED, va="bottom", ha="left")
        style(ax)
    last = axes.flat[-1]
    last.axis("off")
    last.text(0.0, 0.62, "Dots: one run each (seeds 42, 43, 44).\nBar and tick: mean ± 1 std.\n"
              "Grey line: contrastive mean.\nZero-shot CLIP B/32 lies below every\naxis; its value and the "
              "contrastive\nbaseline's gain over it head each panel.", fontsize=8.5, color=MUTED, va="top",
              transform=last.transAxes)
    model_legend(fig, loc="upper right", bbox_to_anchor=(0.99, 0.30), ncol=1)
    fig.suptitle("COCO 5k Karpathy test, CLIP ViT-B/32 fine-tuned 10 epochs, three seeds per model",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.0)
    fig.savefig(OUT / "test_metrics.png", dpi=160, facecolor="white")
    plt.close(fig)


def fig_gain(runs: list[dict], zs: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    rows = list(reversed(METRICS))
    offsets = {"fusion_none": -0.22, "fusion_concat": 0.0, "fusion_multilearner": 0.22}
    for y, (k, lbl) in enumerate(rows):
        vals = by_model(runs, lambda r, k=k: r["test"][k])
        base = statistics.mean(vals[BASELINE])
        ft = base - zs[k]
        sd_base = statistics.stdev(vals[BASELINE]) / ft * 100
        ax.fill_betweenx([y - 0.38, y + 0.38], -sd_base, sd_base, color=COLORS[BASELINE], alpha=0.15, linewidth=0)
        for m, off in offsets.items():
            share = [(v - base) / ft * 100 for v in vals[m]]
            mean, sd = statistics.mean(share), statistics.stdev(share)
            ax.errorbar(mean, y + off, xerr=sd, fmt="o", color=COLORS[m], markersize=6, markeredgecolor="white",
                        elinewidth=1.5, capsize=0)
        ax.text(1.01, y, f"{ft:+.2f}", transform=ax.get_yaxis_transform(), fontsize=8, color=MUTED, va="center")
    ax.axvline(0, color=COLORS[BASELINE], linewidth=1)
    ax.set_yticks(range(len(rows)), [lbl for _, lbl in rows], fontsize=9, color=INK)
    ax.set_xlabel("change over the contrastive baseline, % of its gain over zero-shot CLIP (mean ± std over seeds)",
                  fontsize=8.5, color=MUTED)
    ax.text(1.01, len(rows) - 0.4, "fine-tuning\ngain (points)", transform=ax.get_yaxis_transform(), fontsize=8,
            color=MUTED, va="bottom")
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=COLORS[m], markeredgecolor="white",
                      label=m) for m in offsets] + [Patch(color=COLORS[BASELINE], alpha=0.15, label="contrastive ± 1 std")]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(0.97, 0.99))
    fig.tight_layout()
    fig.savefig(OUT / "gain_vs_finetune.png", dpi=160, facecolor="white")
    plt.close(fig)


def fig_val_curves(runs: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    epochs = list(range(1, 11))
    base = [statistics.mean(r["val_rsum"][e] for r in runs if r["model"] == BASELINE) for e in range(10)]
    for m in MODELS:
        rs = [r for r in runs if r["model"] == m]
        for ax, series, rel in ((axes[0], "val_rsum", False), (axes[1], "val_rsum", True), (axes[2], "val_nce", False)):
            per_epoch = [[r[series][e] - (base[e] if rel else 0) for r in rs] for e in range(10)]
            mean = [statistics.mean(v) for v in per_epoch]
            ax.fill_between(epochs, [min(v) for v in per_epoch], [max(v) for v in per_epoch], color=COLORS[m],
                            alpha=0.12, linewidth=0)
            ax.plot(epochs, mean, color=COLORS[m], linewidth=2, solid_capstyle="round", label=m)
            ax.plot(epochs[-1], mean[-1], "o", color=COLORS[m], markersize=6, markeredgecolor="white")
    titles = ["val rsum (COCO 5k val)", "val rsum minus the contrastive mean", "val contrastive loss (InfoNCE)"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("epoch", fontsize=9, color=MUTED)
        ax.set_xticks(epochs)
        style(ax)
    axes[1].axhline(0, color=COLORS[BASELINE], linewidth=1)
    model_legend(fig, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.01, 0.01, "Lines: mean of three seeds; bands: min to max. Every run's best epoch (early-stopping "
             "monitor val rsum) was its last.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(OUT / "val_curves.png", dpi=160, facecolor="white")
    plt.close(fig)


def fig_gradient_paths() -> None:
    grey_fill, grey_edge = "#d9d9d9", "#555555"
    teal_fill, teal_edge = "#2a9d8f", "#1d6f65"
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(-0.4, 6.0)
    ax.axis("off")

    def box(x0, y0, x1, y1, text, new=False, size=8.5):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=teal_fill if new else grey_fill,
                                    edgecolor=teal_edge if new else grey_edge, linewidth=1))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=size,
                color="white" if new else INK)

    def arrow(x0, y0, x1, y1, dashed=False, color=INK, rad=0.0):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.2 if not dashed else 1.5,
                                    linestyle=(0, (4, 3)) if dashed else "-", shrinkA=0, shrinkB=0,
                                    connectionstyle=f"arc3,rad={rad}"))

    rows = {"img": 5.0, "mimg": 3.9, "mtxt": 2.1, "txt": 1.0}
    h = 0.3
    # inputs
    box(0.1, rows["img"] - h, 1.9, rows["img"] + h, "image")
    box(0.1, rows["mimg"] - h, 1.9, rows["mimg"] + h, "image, 75% of\npatches dropped", new=True, size=8)
    box(0.1, rows["mtxt"] - h, 1.9, rows["mtxt"] + h, "caption, 15% of\ntokens -> [MASK]", new=True, size=8)
    box(0.1, rows["txt"] - h, 1.9, rows["txt"] + h, "caption")
    # towers (shared weights between the clean and the masked pass)
    box(2.4, 3.15, 4.0, rows["img"] + h + 0.05, "CLIP vision\ntower\n(fine-tuned,\nlr 1e-5)")
    box(2.4, rows["txt"] - h - 0.05, 4.0, 2.85, "CLIP text\ntower\n(fine-tuned,\nlr 1e-5)")
    for r in rows.values():
        arrow(1.9, r, 2.4, r)
    # clean pass
    box(4.5, rows["img"] - h, 7.3, rows["img"] + h, "image embedding\n(CLS + pretrained projection)", size=8)
    box(4.5, rows["txt"] - h, 7.3, rows["txt"] + h, "text embedding\n(EOS + pretrained projection)", size=8)
    arrow(4.0, rows["img"], 4.5, rows["img"])
    arrow(4.0, rows["txt"], 4.5, rows["txt"])
    box(11.5, 0.6, 13.1, 5.4, "InfoNCE\n(training)\n\ncosine\nranking\n(retrieval,\nall test\nmetrics)")
    arrow(7.3, rows["img"], 11.5, rows["img"])
    arrow(7.3, rows["txt"], 11.5, rows["txt"])
    # masked pass
    box(4.5, rows["mimg"] - h, 5.5, rows["mimg"] + h, "linear\nto 256", new=True, size=8)
    box(4.5, rows["mtxt"] - h, 5.5, rows["mtxt"] + h, "linear\nto 256", new=True, size=8)
    arrow(4.0, rows["mimg"], 4.5, rows["mimg"])
    arrow(4.0, rows["mtxt"], 4.5, rows["mtxt"])
    box(6.0, 1.6, 7.6, 4.4, "fusion\n\nnone: each\ndecoder reads\nits own tokens\n\nconcat: both\nread both\n\nmultilearner:\n3 transformer\nlearners + MLPs",
        new=True, size=7.5)
    arrow(5.5, rows["mimg"], 6.0, rows["mimg"])
    arrow(5.5, rows["mtxt"], 6.0, rows["mtxt"])
    box(8.1, rows["mimg"] - h, 9.5, rows["mimg"] + h, "image query\ndecoder", new=True, size=8)
    box(8.1, rows["mtxt"] - h, 9.5, rows["mtxt"] + h, "text query\ndecoder", new=True, size=8)
    arrow(7.6, rows["mimg"], 8.1, rows["mimg"])
    arrow(7.6, rows["mtxt"], 8.1, rows["mtxt"])
    box(9.9, rows["mimg"] - h, 11.1, rows["mimg"] + h, "MAE loss\n(pixels)", new=True, size=8)
    box(9.9, rows["mtxt"] - h, 11.1, rows["mtxt"] + h, "MLM loss\n(tokens)", new=True, size=8)
    arrow(9.5, rows["mimg"], 9.9, rows["mimg"])
    arrow(9.5, rows["mtxt"], 9.9, rows["mtxt"])
    # gradients of MAE and MLM into the towers
    arrow(6.0, 3.3, 4.0, 3.3, dashed=True, color=teal_edge)
    arrow(6.0, 2.7, 4.0, 2.7, dashed=True, color=teal_edge)
    ax.text(5.0, 3.0, "MAE + MLM gradients\ninto both towers", fontsize=7.5, color=teal_edge, ha="center", va="center")
    # legend
    handles = [Patch(facecolor=grey_fill, edgecolor=grey_edge, label="in all four models (contrastive = this alone)"),
               Patch(facecolor=teal_fill, edgecolor=teal_edge, label="new in fusion_none / fusion_concat / fusion_multilearner"),
               Line2D([], [], color=teal_edge, linestyle=(0, (4, 3)), linewidth=1.5,
                      label="reconstruction gradients reaching the shared towers")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, -0.07), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "gradient_paths.png", dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    counts = {m: sorted(r["seed"] for r in runs if r["model"] == m) for m in MODELS}
    assert all(s == [42, 43, 44] for s in counts.values()), counts
    zs = json.loads(ZERO_SHOT.read_text())
    print_tables(runs, zs)
    write_csv(runs)
    fig_test_metrics(runs, zs)
    fig_gain(runs, zs)
    fig_val_curves(runs)
    fig_gradient_paths()
    print(f"\nwrote {sorted(p.name for p in OUT.iterdir())} to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
