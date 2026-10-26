"""Figures of docs/reports/literature/2026-10-06_masked_decoder_readings_review.md (integrative review of H-b).

Writes to docs/reports/assets/2026-10-06_masked_decoder_readings_review/:
  hb_chain.png     Figure 1: the H-b chain (masked inputs -> fused memory -> MLM decoder distribution -> compared
                   with human readings / with the dual encoder / used at retrieval), with what the literature
                   established, what is open and the novelty risk per link (sub-claims S1 to S6)
  design_map.png   Figure 2: candidate designs D1 to D11 by GPU-hours against the sub-claims they test

Every number is copied from the review's synthesis (S_synthesis.md, sections 1, 6 and 7, as corrected by the
verification passes V1 to V3); nothing is read from runs. The one locator the report changes (Cekinmez et al.,
Table 2 rather than Table 1) does not enter a figure.

  OMP_NUM_THREADS=8 /root/miniconda3/envs/MultiMAE/bin/python \
      docs/reports/assets/build_2026-10-06_masked_decoder_readings_review.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

OUT = Path(__file__).resolve().parent / "2026-10-06_masked_decoder_readings_review"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

# Figure 1 boxes follow the user's diagram legend: grey = as in MultiMAE today, teal = new.
GREY = ("#e6e6e4", "#555555")
TEAL = ("#c4ebe6", "#1d6f65")
# Figure 1 evidence status, double-encoded by colour and line style and always named in words.
EST = "#2a78d6"    # partly established, in other models (solid)
OPEN = "#52514e"   # open (dashed)

# Figure 2 categories: the first three slots of the dataviz reference palette, validated all-pairs in light mode
# (worst CVD dE 9.2, normal-vision 24.0); aqua is below 3:1 on the surface, so every mark carries a direct label.
PROBE, TRAIN, LATER = "#2a78d6", "#eb6834", "#1baf7a"


def _box(ax, x0, y0, x1, y1, text, colors=GREY, size=8.0, weight="normal", ls="-"):
    fill, edge = colors
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fill, edgecolor=edge, linewidth=1.2, linestyle=ls))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight=weight, linespacing=1.25)


def _arrow(ax, x0, y0, x1, y1, color=INK, dashed=False, lw=1.2, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=lw,
                                linestyle=(0, (4, 3)) if dashed else "-", shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}"))


def _tag(ax, x, y, label, established):
    """A sub-claim marker on the chain: filled blue if partly established elsewhere, dashed grey ring if open."""
    color = EST if established else OPEN
    ax.add_patch(plt.Circle((x, y), 0.25, facecolor=color if established else SURFACE, edgecolor=color,
                            linewidth=1.6, linestyle="-" if established else (0, (2.5, 1.5)), zorder=5))
    ax.text(x, y, label, ha="center", va="center", fontsize=8, fontweight="bold",
            color=SURFACE if established else INK, zorder=6)


# (label, title, established?, status, what is known, what is open, novelty risk); numbers from the synthesis.
CARDS = [
    ("S1", "The decoder's distribution\nholds several readings", True, "partly established",
     "An autoregressive MLLM's slot\ndistribution split between two\nreadings: the preferred one\n"
     "reversed in 37 to 44 of 50\nanagrams (Takatsuki 2026). Free\noutputs commit (Agate 2026:\n"
     "dominance 0.558).",
     "masked cross-modal decoders;\nnatural images", "medium"),
    ("S2", "That spread is measured\nagainst human readings", True, "protocols established",
     "JSD, KL, TVD, EntCE against vote\ndistributions. Sampled outputs and\nmodel probabilities are too peaked:\n"
     "BEiT3 KL 1.701, 0.647 after one\ntemperature (Lan 2025); sense\nentropy 0.10 vs humans 0.47\n(Cekinmez 2026).",
     "masked decoders; paintings\n(AL-28: about 100 votes each)", "low to medium"),
    ("S3", "The decoder keeps more\nreadings than a dual\nencoder", False, "open",
     "Most VLMs of both kinds prefer\none reading of a bistable image;\nCLIP opposed the generative\n"
     "models on 3 of 29 images\n(Panagopoulou 2024). CLIP\npenalised valid underspecified\ncaptions (Pezzelle 2023).",
     "a masked decoder vs a dual\nencoder on ambiguous items", "low"),
    ("S4", "The decoder is used\nat retrieval time", True, "components established",
     "Generative re-ranking works.\nRaw caption likelihood (BLIP):\nCOCO i2t R@1 19.7 vs ITM 71.9,\n"
     "t2i 55.6 vs 54.8 (Lin 2024):\nthe prior hurts only when the\ngenerated modality is ranked.",
     "covering several readings;\nsame-model ECCV mAP@R", "medium"),
    ("S5", "Masking teaches\nunder-specification", False, "open",
     "Nothing direct beyond ProLIP's\nmasked inclusion. Any decoder\ntrained by cross-entropy on\n"
     "several captions learns a\nmixture, so a captioner may\nshow the same spread.",
     "does the masking ratio change\ncalibration to human readings?", "low; high\n\"trivial\" exposure"),
    ("S6", "Mixing combined with\nmasked reconstruction", False, "open (narrowed form)",
     "MixGen (blend + concatenated\ncaptions, through ALBEF's MLM):\nCOCO RSUM 485.6 to 491.8.\n"
     "PCME++ image mixing alone:\nECCV mAP@R 38.9 to 39.0.",
     "either-one caption targets;\nblends as ambiguity probes", "low"),
]


def fig_chain() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 9.4))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 9.6)
    ax.axis("off")

    # The chain. Grey: exists in MultiMAE today. Teal: the new measurement or use H-b needs.
    yc = 7.2
    _box(ax, 0.2, yc - 0.65, 2.9, yc + 0.65,
         "masked inputs\n25% image view (13 of 49 patches)\n+ caption with 80% of its\ntokens masked", size=8)
    _box(ax, 3.6, yc - 0.65, 5.9, yc + 0.65, "fused memory\nimage, text and joint\nlearners (multilearner)",
         size=8)
    _box(ax, 6.6, yc - 0.65, 9.4, yc + 0.65,
         "MLM decoder distribution\nper-position marginals given\n(image view, visible tokens,\ncaption length)",
         size=8, weight="bold")
    _arrow(ax, 2.9, yc, 3.6, yc)
    _arrow(ax, 5.9, yc, 6.6, yc)
    ax.text(7.7, yc - 0.75, "nothing evaluates this distribution today;\nretrieval ranks the clean pooled embeddings",
            ha="center", va="top", fontsize=7.4, color=MUTED, style="italic")

    # Mixed inputs feeding the masked pass (S6), new.
    _box(ax, 0.2, 8.75, 2.9, 9.45, "mixed inputs: 50/50 pixel blends,\nan either-one caption target", TEAL, size=7.6)
    _arrow(ax, 1.55, 8.75, 1.55, yc + 0.65, color=TEAL[1], dashed=True)

    # Three uses of the decoder distribution, new.
    ys = (8.95, 7.2, 5.75)
    texts = ("compared with human readings\n(vote distributions per item)",
             "compared with the dual encoder\n(same model's pooled embeddings)",
             "used at retrieval\n(re-ranking, coverage of readings)")
    for y, t in zip(ys, texts):
        _box(ax, 11.2, y - 0.45, 14.2, y + 0.45, t, TEAL, size=8)
        _arrow(ax, 9.4, yc, 11.2, y, color=TEAL[1], lw=1.4)

    # Sub-claim markers on the chain.
    _tag(ax, 9.4, yc + 0.65, "S1", True)
    _tag(ax, 10.3, 8.08, "S2", True)
    _tag(ax, 10.3, yc + 0.0, "S3", False)
    _tag(ax, 10.3, 6.47, "S4", True)
    _tag(ax, 1.55, 8.1, "S6", False)
    # S5 spans training under masking: a bracket from the masked inputs to the decoder.
    ax.plot([0.5, 0.5, 9.1, 9.1], [6.45, 5.75, 5.75, 6.45], color=OPEN, linewidth=1.2, linestyle=(0, (2.5, 1.5)))
    _tag(ax, 3.2, 5.75, "S5", False)
    ax.text(3.55, 5.5, "training under masking shapes the decoder", ha="left", va="center", fontsize=7.4,
            color=MUTED)

    # Status cards, one per sub-claim.
    w, gap, x = 2.42, 0.13, 0.2
    y0, y1 = 0.15, 4.3
    for label, title, est, status, known, open_, risk in CARDS:
        color = EST if est else OPEN
        ax.add_patch(FancyBboxPatch((x, y0), w, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=SURFACE, edgecolor=color, linewidth=1.5,
                                    linestyle="-" if est else (0, (4, 2.5))))
        _tag(ax, x + 0.33, y1 - 0.35, label, est)
        ax.text(x + 0.68, y1 - 0.35, title, ha="left", va="center", fontsize=7.6, fontweight="bold", color=INK,
                linespacing=1.15)
        ax.text(x + 0.12, y1 - 0.88, f"status: {status}", ha="left", va="top", fontsize=7.2, color=color,
                fontweight="bold")
        ax.text(x + 0.12, y1 - 1.18, "known", ha="left", va="top", fontsize=7.0, color=MUTED, fontweight="bold")
        ax.text(x + 0.12, y1 - 1.42, known, ha="left", va="top", fontsize=7.0, color=INK, linespacing=1.2)
        ax.text(x + 0.12, 1.62, "open", ha="left", va="top", fontsize=7.0, color=MUTED, fontweight="bold")
        ax.text(x + 0.12, 1.38, open_, ha="left", va="top", fontsize=7.0, color=INK, linespacing=1.2)
        ax.text(x + 0.12, 0.66, "novelty risk:", ha="left", va="top", fontsize=7.0, color=MUTED, fontweight="bold")
        ax.text(x + 0.95, 0.66, risk, ha="left", va="top", fontsize=7.0, color=INK, fontweight="bold",
                linespacing=1.15)
        x += w + gap

    handles = [
        Patch(facecolor=GREY[0], edgecolor=GREY[1], label="as in MultiMAE today"),
        Patch(facecolor=TEAL[0], edgecolor=TEAL[1], label="new: the measurement or use H-b needs"),
        Line2D([], [], marker="o", color="none", markerfacecolor=EST, markeredgecolor=EST, markersize=11,
               label="sub-claim partly established, in other models (solid)"),
        Line2D([], [], marker="o", color="none", markerfacecolor=SURFACE, markeredgecolor=OPEN, markersize=11,
               label="sub-claim open (dashed)"),
    ]
    ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.488), ncol=4, frameon=False,
              fontsize=8.2)
    ax.set_title("The H-b chain: what the literature established, what is open, and the novelty risk per link",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "hb_chain.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


# (id, short name, category, sub-claims, GPU-h as (low, high) for the plotted span, label as the synthesis says it)
# category: "probe" = zero training on existing COCO checkpoints; "eval" = zero training on the new ArtELingo
# checkpoints (needs D4/D5); "train" = training before 2026-10-10; "later" = after 2026-10-10.
DESIGNS = [
    ("D1", "decoder scoring on ECCV Caption (COCO)", "probe", {"S3", "S4"}, (6, 6), "about 6"),
    ("D2", "blend probe with known readings (COCO)", "probe", {"S1", "S3", "S6"}, (2, 2), "about 2"),
    ("D3", "spread vs human caption diversity (COCO)", "probe", {"S2"}, (1, 1), "at most 1"),
    ("D4", "ArtELingo triad: C, ML-80, Par-cap", "train", {"S1", "S2", "S3"}, (30.9, 30.9), "30.9"),
    ("D5", "decomposition: AR-cap, ML-15, ML-100v", "train", {"S5"}, (36, 36), "36"),
    ("D6", "view sampling", "eval", {"S1", "S2"}, (1, 2), "1 to 2"),
    ("D7", "context-selected readings", "eval", {"S1"}, (1, 1), "under 1"),
    ("D8", "re-ranking and coverage (ArtELingo)", "eval", {"S4"}, (2, 4), "2 to 4"),
    ("D9", "set embeddings, decoder-chosen slots", "later", {"S4"}, (30, 30), "about 30"),
    ("D10", "mixing: either-one vs both-at-once", "train", {"S6"}, (13.5, 27), "13.5 to 27"),
    ("D11", "unmixing", "later", {"S6"}, (14, 14), "about 14"),
]
STYLE = {  # colour, marker, filled?
    "probe": (PROBE, "o", True),
    "eval": (PROBE, "o", False),
    "train": (TRAIN, "s", True),
    "later": (LATER, "^", True),
}
SUBS = ["S1", "S2", "S3", "S4", "S5", "S6"]


def fig_design_map() -> None:
    fig, (axm, axg) = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw={"width_ratios": [1.0, 1.9]}, sharey=True)
    n = len(DESIGNS)
    ys = list(range(n))[::-1]

    # Left: which sub-claims each design tests.
    for y, (did, name, cat, subs, _, _) in zip(ys, DESIGNS):
        color, marker, filled = STYLE[cat]
        for i, s in enumerate(SUBS):
            if s in subs:
                axm.plot(i, y, marker=marker, markersize=10, color=color, markerfacecolor=color if filled else SURFACE,
                         markeredgewidth=1.8, linestyle="none", zorder=3)
            else:
                axm.plot(i, y, marker="o", markersize=3, color=GRID, linestyle="none", zorder=2)
    axm.set_xticks(range(len(SUBS)), SUBS, fontsize=9, color=INK)
    axm.xaxis.tick_top()
    axm.set_xlim(-0.6, len(SUBS) - 0.4)
    axm.set_yticks(ys, [f"{d[0]}  {d[1]}" for d in DESIGNS], fontsize=8.6, color=INK)
    axm.tick_params(axis="both", length=0)
    for side in ("top", "right", "bottom", "left"):
        axm.spines[side].set_visible(False)
    axm.set_xlabel("sub-claims tested", fontsize=9, color=MUTED)

    # Right: GPU-hours, log scale; a span where the synthesis gives a range.
    for y, (did, name, cat, subs, (lo, hi), label) in zip(ys, DESIGNS):
        color, marker, filled = STYLE[cat]
        if hi > lo:
            axg.plot([lo, hi], [y, y], color=color, linewidth=3.0, solid_capstyle="round", zorder=2)
            xs = [lo, hi]
        else:
            xs = [lo]
        for xv in xs:
            axg.plot(xv, y, marker=marker, markersize=10, color=color, markerfacecolor=color if filled else SURFACE,
                     markeredgewidth=1.8, linestyle="none", zorder=3)
        axg.text(hi * 1.13, y, f"{label} GPU-h", va="center", ha="left", fontsize=8.4, color=INK)
    axg.set_xscale("log")
    axg.set_xlim(0.5, 110)
    axg.set_xticks([0.5, 1, 2, 4, 8, 16, 32, 64], ["0.5", "1", "2", "4", "8", "16", "32", "64"])
    axg.tick_params(axis="x", colors=MUTED, labelsize=8.5)
    axg.tick_params(axis="y", length=0)
    axg.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    axg.axvline(4, color=MUTED, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
    axg.text(4.15, n + 0.05, "one ArtELingo masked run: about 4 h (untested estimate)", fontsize=7.8, color=MUTED,
             ha="left", va="center")
    for side in ("top", "right", "left"):
        axg.spines[side].set_visible(False)
    axg.spines["bottom"].set_color(MUTED)
    axg.set_xlabel("estimated A6000 GPU-hours (log scale; 3 seeds for training designs)", fontsize=9, color=MUTED)
    axg.set_ylim(-0.7, n + 0.35)

    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=PROBE, markeredgecolor=PROBE, markersize=9,
               label="zero training, existing COCO checkpoints"),
        Line2D([], [], marker="o", color="none", markerfacecolor=SURFACE, markeredgecolor=PROBE, markersize=9,
               markeredgewidth=1.8, label="zero training, needs the new ArtELingo checkpoints"),
        Line2D([], [], marker="s", color="none", markerfacecolor=TRAIN, markeredgecolor=TRAIN, markersize=9,
               label="training run before 2026-10-10"),
        Line2D([], [], marker="^", color="none", markerfacecolor=LATER, markeredgecolor=LATER, markersize=9,
               label="after 2026-10-10"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.6, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Candidate designs D1 to D11: the sub-claims each tests and its estimated cost", fontsize=11,
                 color=INK, x=0.01, ha="left")
    fig.subplots_adjust(left=0.235, right=0.985, top=0.87, bottom=0.16, wspace=0.04)
    fig.savefig(OUT / "design_map.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_chain()
    fig_design_map()
    print(f"wrote {OUT / 'hb_chain.png'} and {OUT / 'design_map.png'}")


if __name__ == "__main__":
    main()
