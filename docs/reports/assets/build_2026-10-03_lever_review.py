"""Figures of docs/reports/auto/v1/2026-10-03_lever_review.md (integrative literature review, 2026-10-03).

Writes to docs/reports/assets/2026-10-03_lever_review/:
  routes.png        Figure 1: four routes by which an auxiliary loss can reach the retrieval embedding
  eccv_changes.png  Figure 2: same-recipe ECCV mAP@R changes at CLIP ViT-B/32 fine-tuned on COCO

Every value in Figure 2 is copied from the review's synthesis (section 1.4 table, as corrected by the
verification passes V1 and V2) and from the project's 3-seed baselines report; nothing is read from runs.

  /root/miniconda3/envs/MultiMAE/bin/python docs/reports/assets/build_2026-10-03_lever_review.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

OUT = Path(__file__).resolve().parent / "2026-10-03_lever_review"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Figure 1 colour legend (the user's convention): (fill, edge). Light tints keep dark text readable.
GREY = ("#e6e6e4", "#555555")      # same as MultiMAE today
ORANGE = ("#fbd5c2", "#c4501f")    # replaced
TEAL = ("#c4ebe6", "#1d6f65")      # new
PURPLE = ("#ddd8f5", "#4a3aa7")    # unchanged in every route: test-time retrieval

# Figure 2: literature bars take categorical slot 1, ours slot 3 (the multilearner colour of the
# baselines report). Validated with the dataviz validator, light mode: CVD dE 23.1, normal-vision 24.0;
# slot 3 is below 3:1 on the surface, so every bar carries a direct value label.
LIT, OURS = "#2a78d6", "#1baf7a"


def _box(ax, x0, y0, x1, y1, text, colors=GREY, size=7.5, weight="normal"):
    fill, edge = colors
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fill, edgecolor=edge, linewidth=1.1))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight=weight)


def _arrow(ax, x0, y0, x1, y1, color=INK, dashed=False, rad=0.0, lw=1.1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=lw,
                                linestyle=(0, (4, 3)) if dashed else "-", shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}"))


def _route_panel(ax, route: int) -> None:
    """One route on a fixed skeleton: clean pass on top, masked pass at the bottom, one shared tower."""
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0.2, 6.9)
    ax.axis("off")
    yc, ym, yb = 5.0, 3.25, 1.5   # clean row, middle row (route 4), masked row
    h = 0.36

    # Inputs and the shared tower (grey in every route).
    _box(ax, 0.1, yc - h, 1.6, yc + h, "clean image\nor caption")
    _box(ax, 0.1, yb - h, 1.6, yb + h, "masked image\nor caption")
    _box(ax, 2.0, yb - h - 0.1, 3.3, yc + h + 0.1, "CLIP\ntower\n(shared\nweights)")
    _arrow(ax, 1.6, yc, 2.0, yc)
    _arrow(ax, 1.6, yb, 2.0, yb)

    # Clean pass: tokens -> pooling -> retrieval embedding -> ranking.
    _box(ax, 3.7, yc - h, 4.9, yc + h, "token\nfeatures")
    if route == 2:
        _box(ax, 5.3, yc - h, 6.9, yc + h, "mean pooling\nor GPO over\nall tokens", ORANGE, size=7)
    else:
        _box(ax, 5.3, yc - h, 6.9, yc + h, "CLS / EOS\npooling")
    _box(ax, 7.3, yc - h, 8.6, yc + h, "retrieval\nembedding", weight="bold")
    _box(ax, 8.95, yc - h - 0.05, 10.35, yc + h + 0.05, "InfoNCE;\ncosine ranking\nat test", PURPLE, size=7)
    _arrow(ax, 3.3, yc, 3.7, yc)
    _arrow(ax, 4.9, yc, 5.3, yc)
    _arrow(ax, 6.9, yc, 7.3, yc)
    _arrow(ax, 8.6, yc, 8.95, yc)

    # Masked pass: tokens -> fusion + decoder -> reconstruction loss.
    _box(ax, 3.7, yb - h, 4.9, yb + h, "masked\ntokens")
    if route == 3:
        _box(ax, 5.3, yb - h, 6.9, yb + h, "decoder reads\nthe pooled\nembedding", ORANGE, size=7)
    else:
        _box(ax, 5.3, yb - h, 6.9, yb + h, "fusion +\nquery decoder")
    _box(ax, 7.3, yb - h, 8.6, yb + h, "MAE / MLM\nloss")
    _arrow(ax, 3.3, yb, 3.7, yb)
    _arrow(ax, 4.9, yb, 5.3, yb)
    _arrow(ax, 6.9, yb, 7.3, yb)

    grad = INK  # dashed route of the auxiliary gradient into the retrieval embedding
    if route == 1:
        _arrow(ax, 7.3, yb - 0.55, 3.3, yb - 0.55, color=grad, dashed=True, lw=1.5)
        ax.text(5.3, yb - 0.85, "gradient reaches the embedding only through the tower weights",
                fontsize=6.8, color=grad, ha="center", va="center")
    elif route == 2:
        _arrow(ax, 4.3, yb + h, 4.3, yc - h, color=grad, dashed=True, lw=1.5)
        ax.text(4.42, (yb + yc) / 2, "reconstruction shapes the token\nfeatures, and the new pooling\n"
                "puts all of them into the\nretrieval embedding", fontsize=6.8, color=grad, ha="left",
                va="center")
    elif route == 3:
        _arrow(ax, 8.15, yc - h, 6.75, yb + h, color=TEAL[1], lw=1.6)
        ax.text(7.9, (yb + yc) / 2 - 0.2, "new input: the pooled\nembedding (ConLIP, TULIP)", fontsize=6.8,
                color=TEAL[1], ha="left", va="center")
        _arrow(ax, 6.2, yb + h, 7.6, yc - h, color=grad, dashed=True, lw=1.5)
    elif route == 4:
        _box(ax, 5.3, ym - h, 6.9, ym + h, "masked-view\nembedding", TEAL, size=7)
        _box(ax, 7.3, ym - h, 9.6, ym + h, "inclusion or soft\nmulti-positive loss\n(ProLIP)", TEAL, size=7)
        _arrow(ax, 4.9, yb + 0.15, 5.3, ym - 0.1, color=TEAL[1])
        _arrow(ax, 6.9, ym, 7.3, ym, color=TEAL[1])
        _arrow(ax, 7.95, ym + h, 7.95, yc - h, color=grad, dashed=True, lw=1.5)

    titles = {
        1: "Route 1: shared tower weights (MultiMAE today)",
        2: "Route 2: pooling built from token features",
        3: "Route 3: decoders conditioned on the pooled embedding",
        4: "Route 4: a loss on the masked view's own embedding",
    }
    notes = {
        1: "Measured in CLIP-scale dual encoders; small effects. Levers M1, M2, M4, M5 stay on this route.",
        2: "Evidence: PCME++ C.6 (GPO, +2.6 mAP@R over an unnamed pooling). Lever R3.",
        3: "Evidence: ConLIP T1 (non-CLIP dual encoder, +0.7 / +0.7 R@1, single runs). Lever M3.",
        4: "Evidence: ProLIP C.3, C.4 (from scratch; never on ECCV or PMRP). Lever M6.",
    }
    ax.set_title(titles[route], fontsize=9.5, color=INK, loc="left", pad=4)
    ax.text(0.1, 6.45, notes[route], fontsize=7, color=MUTED, ha="left", va="center")


def fig_routes() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.6))
    for route, ax in zip((1, 2, 3, 4), axes.ravel()):
        _route_panel(ax, route)
    handles = [
        Patch(facecolor=GREY[0], edgecolor=GREY[1], label="grey: same as MultiMAE today"),
        Patch(facecolor=ORANGE[0], edgecolor=ORANGE[1], label="orange: replaced"),
        Patch(facecolor=TEAL[0], edgecolor=TEAL[1], label="teal: new"),
        Patch(facecolor=PURPLE[0], edgecolor=PURPLE[1],
              label="purple: unchanged in every route (retrieval ranks the clean pooled embeddings)"),
        Line2D([], [], color=INK, linestyle=(0, (4, 3)), linewidth=1.5,
               label="how the auxiliary loss reaches the retrieval embedding"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=1.5, w_pad=2.0)
    fig.savefig(OUT / "routes.png", dpi=160, facecolor="white")
    plt.close(fig)


# (label, change in ECCV mAP@R, caveat shown beside the bar, hatched?)
CHANGES = [
    ("GPO pooling vs an unnamed pooling\n(PCME++ C.6, PCME++ loss)", 2.6, "baseline unknown", True),
    ("CUSA soft labels vs InfoNCE\n(CUSA T2)", 2.3, "single run, weak baseline", False),
    ("VACSR vs its own sigmoid baseline\n(VACSR ablation)", 1.4, "", False),
    ("VIB + pseudo-positives + MSDA\n(PCME++ T3)", 1.2, "", False),
    ("PCME++ vs InfoNCE\n(PCME++ T1)", 1.1, "3 runs", False),
    ("SWA over the last epochs\n(PCME++ T1)", 0.1, "", False),
    ("probabilistic vs mean-only inference,\nsame model (PCME++ C.7)", 0.0, "", False),
]
OURS_ROW = ("fusion_multilearner vs contrastive\n(this project)", 0.02, "3 seeds")
RECIPE_GAP = 2.06  # 39.0 (PCME++ InfoNCE, T1) - 36.94 (our contrastive baseline)


def fig_eccv_changes() -> None:
    rows = CHANGES + [(*OURS_ROW, False)]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    n = len(rows)
    ys = list(range(n))[::-1]
    for y, (label, val, caveat, hatched) in zip(ys, rows):
        ours = label.startswith("fusion_multilearner")
        color = OURS if ours else LIT
        if hatched:
            ax.barh(y, val, height=0.62, color="white", edgecolor=color, hatch="////", linewidth=1.2, zorder=2)
        else:
            ax.barh(y, max(val, 0.0), height=0.62, color=color, edgecolor="white", linewidth=2, zorder=2)
        if val == 0:
            ax.plot([0, 0], [y - 0.31, y + 0.31], color=color, linewidth=2.5, zorder=3)
        text = f"{val:+.2f}" if ours else (f"{val:+.1f}" if val else "0.0")
        ax.text(max(val, 0) + 0.04, y, text, va="center", ha="left", fontsize=9, color=INK,
                fontweight="bold" if ours else "normal")
        if caveat:
            ax.text(max(val, 0) + 0.34, y, caveat, va="center", ha="left", fontsize=8, color=MUTED,
                    style="italic")
        if ours:
            ax.axhspan(y - 0.45, y + 0.45, color="#e3f5ee", zorder=0)
            ax.plot([val], [y], "o", color=OURS, markersize=8, markeredgecolor="white", zorder=4)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=8.3, color=INK)
    for tick in ax.get_yticklabels():
        if tick.get_text().startswith("fusion_multilearner"):
            tick.set_fontweight("bold")
    ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
    ax.axvline(RECIPE_GAP, color=MUTED, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(RECIPE_GAP + 0.04, 3.0,
            "2.06: recipe gap between our contrastive\nbaseline (36.94) and PCME++'s InfoNCE (39.0),\n"
            "bundled recipe differences, not one change",
            fontsize=7.8, color=MUTED, ha="left", va="center")
    ax.set_xlim(-0.1, 3.7)
    ax.set_ylim(-0.6, n - 0.2)
    ax.set_xlabel("change in ECCV Caption mAP@R over the same paper's own baseline (points)", fontsize=9,
                  color=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    handles = [Patch(facecolor=LIT, label="literature: one change, same recipe"),
               Patch(facecolor="white", edgecolor=LIT, hatch="////", label="literature: baseline unknown"),
               Patch(facecolor=OURS, label="this project: masked reconstruction + fusion")]
    ax.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 0.25), frameon=False, fontsize=8.3)
    ax.set_title("ECCV Caption mAP@R changes at CLIP ViT-B/32 fine-tuned on COCO", fontsize=10.5, color=INK,
                 loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "eccv_changes.png", dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_routes()
    fig_eccv_changes()
    print(f"wrote {OUT / 'routes.png'} and {OUT / 'eccv_changes.png'}")


if __name__ == "__main__":
    main()
