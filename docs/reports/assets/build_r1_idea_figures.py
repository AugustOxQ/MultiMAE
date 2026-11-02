"""Build the two illustration figures of loop r1's ideas 1 and 2, as the literature checks of 2026-10-26 read them.

Outputs SVG (and PNG through rsvg-convert) into docs/reports/assets/2026-10-26_r1_ideas/.
Run: PYTHONDONTWRITEBYTECODE=1 python docs/reports/assets/build_r1_idea_figures.py
"""
from pathlib import Path
import subprocess

OUT = Path(__file__).parent / "2026-10-26_r1_ideas"
OUT.mkdir(exist_ok=True)

# Palette (also in docs/reports/assets/palette.md if the project keeps one)
TRAINED_FILL, TRAINED_STROKE = "#FDE7CF", "#E07B20"   # trained module
DATA_FILL, DATA_STROKE = "#F3F3F3", "#7A7A7A"          # data, tensors
OP_FILL, OP_STROKE = "#DCEBF7", "#3A78B5"              # fixed operation (blend, mask, pick)
LOSS_FILL, LOSS_STROKE = "#E3F2E1", "#3C8C3C"          # loss
NOTE_STROKE = "#9A9A9A"                                 # dashed notes and controls
INK = "#222222"
FONT = "Lato, DejaVu Sans, sans-serif"


class SVG:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.parts = []

    def text(self, x, y, s, fs=14, anchor="middle", weight="normal", fill=INK, italic=False, mono=False):
        style = f"font-family:{'Source Code Pro, DejaVu Sans Mono, monospace' if mono else FONT};font-size:{fs}px;font-weight:{weight};"
        if italic:
            style += "font-style:italic;"
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" style="{style}">{s}</text>')

    def lines(self, x, y, lines, fs=14, anchor="middle", lh=None, weight_first=False, fill=INK, italic=False, mono=False):
        lh = lh or fs * 1.3
        for i, s in enumerate(lines):
            w = "bold" if (weight_first and i == 0) else "normal"
            self.text(x, y + i * lh, s, fs, anchor, w, fill, italic, mono)

    def rect(self, x, y, w, h, fill, stroke, dash=False, rx=8, sw=1.6):
        d = ' stroke-dasharray="7,5"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def box(self, x, y, w, h, lines, kind="data", fs=14, dash=False, trained=False, anchor="middle", pad=12):
        fill, stroke = {"trained": (TRAINED_FILL, TRAINED_STROKE), "data": (DATA_FILL, DATA_STROKE),
                        "op": (OP_FILL, OP_STROKE), "loss": (LOSS_FILL, LOSS_STROKE),
                        "note": ("#FFFFFF", NOTE_STROKE)}[kind]
        self.rect(x, y, w, h, fill, stroke, dash=dash or kind == "note")
        n = len(lines)
        lh = fs * 1.3
        y0 = y + h / 2 - (n - 1) * lh / 2 + fs * 0.35
        tx = x + w / 2 if anchor == "middle" else x + pad
        self.lines(tx, y0, lines, fs, anchor, lh, weight_first=True)
        if trained:
            self.badge(x + w - 8, y - 9, "trained")
        return (x, y, w, h)

    def badge(self, xr, y, label):
        bw = 58
        self.rect(xr - bw, y, bw, 18, TRAINED_STROKE, TRAINED_STROKE, rx=9)
        self.text(xr - bw / 2, y + 13, label, 11, "middle", "bold", "#FFFFFF")

    def arrow(self, pts, label=None, dash=False, color=INK, fs=12, label_dy=-6, sw=1.6, lx=None, ly=None, head=True):
        d = ' stroke-dasharray="6,5"' if dash else ""
        p = " ".join(f"{x},{y}" for x, y in pts)
        m = ' marker-end="url(#ah)"' if head else ""
        self.parts.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}"{m}{d}/>')
        if label:
            (x1, y1), (x2, y2) = pts[0], pts[-1]
            mx = lx if lx is not None else (x1 + x2) / 2
            my = ly if ly is not None else (y1 + y2) / 2 + label_dy
            self.text(mx, my, label, fs, "middle", "normal", "#444444", italic=True)

    def image_placeholder(self, x, y, w, h, label, blend=False):
        # a picture frame with a horizon and a sun, or a blend of two tints
        self.rect(x, y, w, h, "#FFFFFF", DATA_STROKE, rx=6)
        if blend:
            self.parts.append(f'<rect x="{x+3}" y="{y+3}" width="{w-6}" height="{h-6}" rx="4" fill="#CFE3D9" opacity="0.9"/>')
            self.parts.append(f'<rect x="{x+3}" y="{y+3}" width="{w-6}" height="{h-6}" rx="4" fill="#E6CFCF" opacity="0.55"/>')
        else:
            self.parts.append(f'<rect x="{x+3}" y="{y+3}" width="{w-6}" height="{h-6}" rx="4" fill="#E8EEF3"/>')
        self.parts.append(f'<line x1="{x+8}" y1="{y+h*0.62}" x2="{x+w-8}" y2="{y+h*0.62}" stroke="#8AA0B0" stroke-width="1.5"/>')
        self.parts.append(f'<circle cx="{x+w*0.72}" cy="{y+h*0.33}" r="{min(w,h)*0.1}" fill="#F0C060"/>')
        self.text(x + w / 2, y + h - 8, label, 13, "middle", "bold")

    def emb_bar(self, x, y, w, h, segs=16, mask=None, color="#5B8DB8", label=None, label_fs=12):
        # an embedding drawn as a bar of segments; mask: set of segment indexes kept (others grey)
        sw = w / segs
        for i in range(segs):
            keep = (mask is None) or (i in mask)
            fill = color if keep else "#D9D9D9"
            self.parts.append(f'<rect x="{x+i*sw:.1f}" y="{y}" width="{sw-1:.1f}" height="{h}" fill="{fill}"/>')
        self.rect(x - 1, y - 1, w + 1, h + 2, "none", DATA_STROKE, rx=2, sw=1)
        if label:
            self.text(x + w + 8, y + h / 2 + 4, label, label_fs, "start")

    def patch_grid(self, x, y, size, n=7, visible=None, label=None):
        cell = size / n
        for r in range(n):
            for c in range(n):
                k = r * n + c
                vis = visible is None or k in visible
                fill = "#9DB9D0" if vis else "#EDEDED"
                self.parts.append(f'<rect x="{x+c*cell:.1f}" y="{y+r*cell:.1f}" width="{cell-1:.1f}" height="{cell-1:.1f}" fill="{fill}"/>')
        self.rect(x - 1, y - 1, size + 1, size + 1, "none", DATA_STROKE, rx=2, sw=1)
        if label:
            self.text(x + size / 2, y + size + 14, label, 12)

    def legend(self, x, y):
        items = [("trained", "trained module"), ("data", "data, tokens, embeddings"),
                 ("op", "fixed operation"), ("loss", "loss"), ("note", "control or note")]
        for i, (kind, lab) in enumerate(items):
            fill, stroke = {"trained": (TRAINED_FILL, TRAINED_STROKE), "data": (DATA_FILL, DATA_STROKE),
                            "op": (OP_FILL, OP_STROKE), "loss": (LOSS_FILL, LOSS_STROKE), "note": ("#FFFFFF", NOTE_STROKE)}[kind]
            self.rect(x, y + i * 22, 26, 14, fill, stroke, dash=(kind == "note"), rx=3)
            self.text(x + 34, y + i * 22 + 11, lab, 12, "start", fill="#444444")

    def panel(self, x, y, w, h, title, fill="#FAFAFA"):
        self.rect(x, y, w, h, fill, "#CCCCCC", rx=12, sw=1.2)
        self.text(x + 14, y + 24, title, 17, "start", "bold")

    def save(self, name):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                '<defs><marker id="ah" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto" markerUnits="userSpaceOnUse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>'
                f'<rect width="{self.w}" height="{self.h}" fill="#FFFFFF"/>')
        svg = head + "\n".join(self.parts) + "</svg>"
        p = OUT / f"{name}.svg"
        p.write_text(svg)
        subprocess.run(["rsvg-convert", "-w", str(int(self.w * 1.6)), str(p), "-o", str(OUT / f"{name}.png")], check=True)
        return p


# ----------------------------------------------------------------------------------------------------------------------
# Figure 1: idea 1 as read: a masked caption loss under a designed blend augmentation (either-one targets)
# ----------------------------------------------------------------------------------------------------------------------
def figure_1():
    s = SVG(1500, 940)
    s.text(750, 38, "Idea 1 as the literature check read it: a masked caption loss under a designed blend augmentation", 22, weight="bold")
    s.text(750, 64, "One picture holds two readings (A and B). The visible words of a masked caption choose which reading the decoder must rebuild.", 15, fill="#444444")
    s.legend(40, 700)

    # inputs
    s.image_placeholder(40, 120, 190, 110, "image A")
    s.box(40, 240, 190, 50, ["caption A", "\u201ca dog runs on the beach\u201d"], "data", 13)
    s.image_placeholder(40, 330, 190, 110, "image B")
    s.box(40, 450, 190, 50, ["caption B", "\u201ca red boat in the harbour\u201d"], "data", 13)

    # blend
    s.parts.append(f'<circle cx="320" cy="280" r="30" fill="{OP_FILL}" stroke="{OP_STROKE}" stroke-width="1.6"/>')
    s.text(320, 285, "blend", 13, weight="bold")
    s.text(320, 330, "\u03b1\u00b7A + (1\u2212\u03b1)\u00b7B", 13, fill="#444444", italic=True)
    s.arrow([(230, 175), (292, 265)])
    s.arrow([(230, 385), (292, 296)])
    s.image_placeholder(400, 220, 200, 120, "blended image", blend=True)
    s.text(500, 358, "one picture, two readings", 12, fill="#444444", italic=True)
    s.arrow([(350, 280), (398, 280)])

    # image encoder
    s.arrow([(600, 280), (660, 280)])
    s.box(662, 240, 190, 80, ["image encoder", "(CLIP vision tower)"], "trained", 14, trained=True)
    s.arrow([(852, 300), (980, 420)], "image tokens", lx=885, ly=380)
    s.arrow([(852, 260), (950, 180)], "pooled blend embedding", lx=925, ly=252)

    # caption path
    s.box(300, 560, 230, 60, ["pick one source caption", "(either one): here caption A"], "op", 13)
    s.arrow([(230, 275), (270, 275), (270, 590), (298, 590)])
    s.arrow([(230, 475), (255, 475), (255, 605), (298, 605)], dash=True)
    s.text(248, 560, "or B", 11, "end", fill="#444444", italic=True)
    s.arrow([(530, 590), (580, 590)])
    s.box(582, 560, 180, 60, ["mask 80% of its words"], "op", 13)
    s.arrow([(762, 590), (812, 590)])
    s.box(814, 548, 290, 84, ["masked caption A", "[M]  dog  [M]  [M]  [M]  [M]", "visible: \u201cdog\u201d  \u00b7  hidden: the rest"], "data", 13)
    s.box(814, 668, 290, 54, ["the visible word selects the reading:", "\u201cdog\u201d points at A, not at B\u2019s boat"], "note", 12)
    s.arrow([(1104, 590), (1128, 590)])
    s.box(1130, 560, 170, 60, ["text encoder", "(CLIP text tower)"], "trained", 13, trained=True)
    s.arrow([(1215, 560), (1215, 477)], "text tokens (visible word + [M] slots)", lx=1215, ly=520)
    s.arrow([(1000, 632), (1000, 650), (1400, 650), (1400, 477)], "hidden words = targets", lx=1290, ly=643)

    # caption decoder
    s.box(980, 380, 240, 95, ["caption decoder", "reads the image tokens", "and the visible words"], "trained", 14, trained=True)
    s.arrow([(1220, 427), (1260, 427)])
    s.box(1262, 380, 210, 95, ["masked-word loss", "rebuild A\u2019s hidden words:", "\u201ca \u00b7 runs \u00b7 on \u00b7 the \u00b7 beach\u201d", "only A\u2019s words count"], "loss", 12)

    # contrastive loss
    s.box(950, 130, 240, 100, ["contrastive loss", "the blend embedding matches", "caption A and caption B:", "both are positives (either one),", "or weighted by \u03b1"], "loss", 12)
    s.arrow([(1260, 180), (1192, 180)])
    s.box(1262, 140, 210, 80, ["full captions A and B", "(unmasked) through the", "text encoder"], "data", 12)

    # control and scoring
    s.box(300, 840, 800, 70, ["Must-have control (from the check): the same visible words of A with an image that holds no A (image B alone).",
                             "If the decoder still rebuilds A\u2019s words, the words do the work and not the image: the language-prior failure."], "note", 12)
    s.box(1120, 840, 352, 70, ["Scored on: one-to-many retrieval on natural data", "(ECCV Caption mAP@R); on blends, whether both", "readings are recovered (balance, both covered)."], "note", 12)
    return s.save("idea1_blend_masking")


# ----------------------------------------------------------------------------------------------------------------------
# Figure 2: idea 2 as read: PVSE with masked sub-space training in place of the K attention heads (two readings)
# ----------------------------------------------------------------------------------------------------------------------
def figure_2():
    s = SVG(1500, 1330)
    s.text(750, 38, "Idea 2 as the literature check read it: PVSE with masked sub-space training in place of the K attention heads", 22, weight="bold")
    s.text(750, 64, "Two readings of “masked sub-space”. In both, the masked caption decoder is the signal that assigns a caption (a reading) to one of K sub-spaces.", 15, fill="#444444")
    s.legend(1190, 84)
    K = 3

    # ---- panel 0: PVSE, what is replaced ----
    s.panel(40, 100, 1120, 190, "PVSE (Song and Soleymani, CVPR 2019): what is replaced")
    s.image_placeholder(60, 140, 120, 80, "image")
    s.arrow([(180, 180), (230, 180)])
    s.box(232, 150, 200, 60, ["K attention heads", "over the local features"], "trained", 13, trained=True)
    s.arrow([(432, 180), (480, 180)])
    for k in range(K):
        s.emb_bar(482, 150 + k * 22, 150, 14, label=f"image embedding {k+1}")
    s.box(60, 232, 120, 44, ["caption", "“a dog on the beach”"], "data", 12)
    s.arrow([(180, 254), (230, 254)])
    s.box(232, 232, 200, 44, ["K attention heads", "over the word features"], "trained", 12, trained=True)
    s.arrow([(432, 254), (480, 254)])
    s.emb_bar(482, 247, 150, 14, color="#B07A3C", label="K caption embeddings")
    s.box(822, 140, 325, 70, ["best-of-K match (MIL): the pair of", "embeddings that agrees most carries the loss", "+ a diversity loss that keeps the K apart"], "loss", 12)
    s.box(822, 222, 325, 50, ["retrieval: max over the K × K pairs"], "op", 12)
    s.arrow([(792, 180), (820, 180)])
    s.text(600, 300, "↓  replaced by: K sub-spaces that a masked caption decoder assigns readings to  ↓", 15, weight="bold", fill="#444444")

    # ---- panel a: dimension sub-spaces ----
    s.panel(40, 320, 1430, 470, "Reading (a): K sub-spaces of the embedding dimensions (masks on the 512-d pooled embedding)")
    s.image_placeholder(60, 360, 150, 100, "image")
    s.arrow([(210, 410), (260, 410)])
    s.box(262, 375, 190, 70, ["image encoder", "(CLIP vision tower)"], "trained", 14, trained=True)
    s.arrow([(452, 410), (510, 410)])
    s.emb_bar(512, 395, 220, 26, segs=16)
    s.text(622, 388, "one pooled embedding (512-d)", 12)
    s.arrow([(622, 425), (622, 470)])
    s.box(512, 472, 220, 60, ["K learned dimension masks", "m₁ … mₖ (one per sub-space)"], "op", 13, trained=True)
    masks = [set(range(0, 6)), set(range(5, 11)), set(range(10, 16))]
    for k in range(K):
        y = 372 + k * 52
        s.emb_bar(820, y, 220, 22, segs=16, mask=masks[k], label=f"sub-embedding {k+1}  (e ⊙ m{k+1})", label_fs=12)
        s.arrow([(732, 502), (790, 502), (790, y + 11), (818, y + 11)], sw=1.2)
    # caption, decoder, assignment
    s.box(60, 600, 200, 70, ["one caption of the image,", "80% of its words hidden", "[M] dog [M] [M] [M] [M]"], "data", 12)
    s.arrow([(260, 635), (300, 635)])
    s.box(302, 600, 190, 70, ["text encoder", "(CLIP text tower)"], "trained", 13, trained=True)
    s.arrow([(492, 635), (820, 635)], "visible words + text tokens", lx=656, ly=628)
    s.box(822, 592, 240, 86, ["caption decoder", "conditioned on sub-embedding k", "rebuilds the hidden words", "→ one reconstruction score per k"], "trained", 12, trained=True)
    for k in range(K):
        y = 372 + k * 52 + 11
        s.arrow([(1228, y), (1268, y), (1268, 560), (942, 560), (942, 590)], sw=1.0, dash=True)
    s.text(1010, 553, "each sub-embedding conditions the decoder in turn", 11, fill="#444444", italic=True)
    s.arrow([(1062, 635), (1120, 635)])
    s.box(1122, 592, 330, 86, ["assignment", "the caption goes to the sub-space whose", "decoder rebuilds it best (Hungarian over", "the image’s captions × K sub-spaces)"], "op", 12)
    s.arrow([(1287, 678), (1287, 702)])
    s.box(1122, 704, 330, 70, ["contrastive loss on the assigned pair", "image sub-embedding k ↔ caption embedding", "under the same mask mₖ"], "loss", 12)
    s.box(560, 704, 540, 70, ["retrieval: max over k of the sub-space similarities", "control the check names: one vector of the same total size, and random fixed masks", "(Li and Jurafsky: multi-sense gains can come from capacity alone)"], "note", 12)

    # ---- panel b: masked views ----
    s.panel(40, 820, 1430, 440, "Reading (b): K embeddings, each from a different masked view of the image")
    s.image_placeholder(60, 870, 150, 100, "image")
    s.arrow([(210, 920), (260, 920)])
    s.box(262, 885, 190, 70, ["K patch subsets", "(masked views, e.g. 25% each)"], "op", 13)
    import random
    rnd = random.Random(3)
    views = [set(rnd.sample(range(49), 13)) for _ in range(K)]
    for k in range(K):
        y = 858 + k * 82
        s.arrow([(452, 920), (480, 920), (480, y + 30), (508, y + 30)], sw=1.2)
        s.patch_grid(510, y, 60, visible=views[k], label=f"view {k+1}")
    for k in range(K):
        y = 858 + k * 82
        s.arrow([(570, y + 30), (610, y + 30), (610, 920)], sw=1.2, head=False)
    s.arrow([(610, 920), (640, 920)])
    s.box(642, 885, 190, 70, ["image encoder", "(shared, one pass per view)"], "trained", 13, trained=True)
    for k in range(K):
        y = 870 + k * 52
        s.arrow([(832, 920), (860, 920), (860, y + 10), (888, y + 10)], sw=1.2)
        s.emb_bar(890, y, 180, 20, color=["#5B8DB8", "#6AA06A", "#B07A3C"][k], label=f"view embedding {k+1}", label_fs=12)
    s.box(60, 1080, 200, 70, ["one caption of the image,", "80% of its words hidden", "[M] dog [M] [M] [M] [M]"], "data", 12)
    s.arrow([(260, 1115), (300, 1115)])
    s.box(302, 1080, 190, 70, ["text encoder", "(CLIP text tower)"], "trained", 13, trained=True)
    s.arrow([(492, 1115), (890, 1115)], "visible words + text tokens", lx=690, ly=1108)
    s.box(892, 1072, 240, 86, ["caption decoder", "conditioned on view embedding k", "rebuilds the hidden words", "→ one reconstruction score per k"], "trained", 12, trained=True)
    for k in range(K):
        y = 870 + k * 52 + 10
        s.arrow([(1200, y), (1230, y), (1230, 1040), (1012, 1040), (1012, 1070)], sw=1.0, dash=True)
    s.text(1012, 1033, "each view conditions the decoder in turn", 11, fill="#444444", italic=True)
    s.arrow([(1132, 1115), (1250, 1115)])
    s.box(1252, 1072, 200, 86, ["assignment", "the reading goes to the view", "whose decoder rebuilds it best"], "op", 12)
    s.arrow([(1352, 1158), (1352, 1180)])
    s.box(1122, 1182, 330, 66, ["contrastive loss per view with its assigned caption", "(needed: views trained by pixels alone collapse,", "our D2 finding and U-MAE)"], "loss", 12)
    s.box(560, 1182, 540, 66, ["retrieval: max over the K views", "risk the check names: a view encodes where in the image, not which reading;", "comparator: attentive views with multi-positive InfoNCE (A-CLIP style)"], "note", 12)

    s.box(40, 1278, 1430, 40, ["Shared knob in both readings: the masking ratio. DML (NeurIPS 2022): a fully masked caption decoder with Hungarian assignment keeps 29 of 64 codes in use; partial or no masking keeps 5 or 6. Shared risk: the K readings collapse into one."], "note", 12)
    return s.save("idea2_masked_subspaces")


if __name__ == "__main__":
    for p in (figure_1(), figure_2()):
        print(p)
