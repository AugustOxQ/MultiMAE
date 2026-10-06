"""Tables and figures of docs/reports/auto/v1/2026-10-06_masking_stages_1_2.md: Stage 1 screening, Stage 2
confirmation and the 80% mechanism controls of the improve-multilearner line.

Reads (never writes):
  res/coco/multimae/default/*/{run.json,config.yaml}     Stage 0 baselines; Stage 1 references are `contrastive` and
                                                          `fusion_multilearner` (seeds 42-44). Test metrics from
                                                          run.json["results"]["test"] (training-time test, with the
                                                          losses), overridden by run.json["eval"]["test"] where present
                                                          (seed-42 fusion runs, re-evaluated with evaluate.py).
  res/coco/multimae/ml_improve/*/{run.json,config.yaml}  every Stage 1 arm and every Stage 2 run (`s2_*`). Only
                                                          status == "completed" counts; the seed comes from config.yaml.
  res/coco/diagnostics/stage2_controls/diagnostics.json  VWSD and tower swaps of the 21 Stage 2 runs (seeds 42-44) and
                                                          zero-shot CLIP B/32 (scripts/diagnose.py, job 20261006-141443-9919d8d)
  res/coco/diagnostics/stage2/diagnostics.json           the first 12 of those runs (consistency check only)
  res/coco/diagnostics/stage0/diagnostics.json           VWSD of the Stage 0 baselines (context)
  tests/20261003_ml_improve/stratified_eccv_results.json per-query ECCV mAP@R by stratum (stratified_eccv.py)
  git history of the spec (commit time of the pre-registered Stage 2 rules)

Writes to docs/reports/assets/2026-10-06_masking_stages_1_2/:
  stage1_arms.png    Figure 1: every Stage 1 arm, mAP@R and rsum minus its reference baseline, per seed
  dose.png           Figure 2: the text-masking dose curve (mAP@R, rsum, PMRP, MLM test loss)
  stage2_seeds.png   Figure 3: Stage 2 per-seed mAP@R, rsum and PMRP of the four arms
  controls_swaps.png Figure 4: 80% controls, full models and tower swaps (ECCV mAP@R)
  vwsd.png           Figure 5: VWSD Hit@1 per Stage 2 model
  stratified.png     Figure 6: seed-matched ECCV mAP@R differences by stratum
and prints every number of the report to stdout.

  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/MultiMAE/bin/python \
      docs/reports/assets/build_2026-10-06_masking_stages_1_2.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
DEFAULT = REPO / "res/coco/multimae/default"
ML = REPO / "res/coco/multimae/ml_improve"
DIAG = REPO / "res/coco/diagnostics/stage2_controls/diagnostics.json"
DIAG_OLD = REPO / "res/coco/diagnostics/stage2/diagnostics.json"
DIAG_S0 = REPO / "res/coco/diagnostics/stage0/diagnostics.json"
STRAT = REPO / "tests/20261003_ml_improve/stratified_eccv_results.json"
SPEC = "docs/superpowers/specs/2026-10-03-improve-multilearner-design.md"
OUT = Path(__file__).resolve().parent / "2026-10-06_masking_stages_1_2"

CORE = (42, 43, 44)
# Display names and metric keys (run.json keys after stripping "test/" and "retrieval/")
M = {"mAP@R": "eccv/map_at_r", "PMRP": "pmrp", "rsum": "rsum", "CxC R@1": "cxc/r1", "1K R@1": "coco1k/r1",
     "ECCV R-P": "eccv/rprecision", "ECCV R@1": "eccv/r1", "mAP@R i2t": "eccv/i2t_map_at_r",
     "mAP@R t2i": "eccv/t2i_map_at_r", "i2t R@1": "i2t_R1", "t2i R@1": "t2i_R1", "PMRP i2t": "pmrp/i2t",
     "PMRP t2i": "pmrp/t2i"}
STAGE1 = [  # (arm folder suffix, ID, label, reference)
    ("multilearner_txt25", "M2a", "text masking 25%", "fusion_multilearner"),
    ("multilearner_txt40", "M2a", "text masking 40%", "fusion_multilearner"),
    ("multilearner_txt60", "M2a", "text masking 60%", "fusion_multilearner"),
    ("multilearner_txt80", "M2a", "text masking 80%", "fusion_multilearner"),
    ("multilearner_txt90", "M2a", "text masking 90%", "fusion_multilearner"),
    ("multilearner_mae0", "M5", "MAE off", "fusion_multilearner"),
    ("multilearner_mae0_txt40", "M5+M2a", "MAE off + text 40%", "fusion_multilearner"),
    ("multilearner_m6maskedview", "M6", "masked-view InfoNCE", "fusion_multilearner"),
    ("multilearner_m2bcontent", "M2b", "content-word masking", "fusion_multilearner"),
    ("multilearner_m1clean", "M1", "MLM reads clean image", "fusion_multilearner"),
    ("multilearner_m1detached", "M1", "clean image, detached", "fusion_multilearner"),
    ("multilearner_m3pooled", "M3", "pooled conditioning", "fusion_multilearner"),
    ("contrastive_r2lr", "R2", "PCME++ learning rates", "contrastive"),
    ("contrastive_meanpool", "R3", "mean pooling", "contrastive"),
    ("contrastive_ep15", "R5", "15 epochs", "contrastive"),
]
DOSE = [(0.15, None), (0.25, "multilearner_txt25"), (0.40, "multilearner_txt40"), (0.60, "multilearner_txt60"),
        (0.80, "multilearner_txt80"), (0.90, "multilearner_txt90")]
S2 = ["s2_contrastive", "s2_multilearner", "s2_txt80", "s2_mae0"]
CTRL = ["s2_none_txt80", "s2_m1clean_txt80", "s2_m1detached_txt80"]
S2LABEL = {"s2_contrastive": "contrastive", "s2_multilearner": "multilearner (15%)", "s2_txt80": "text masking 80%",
           "s2_mae0": "MAE off", "s2_none_txt80": "fusion_none at 80%", "s2_m1clean_txt80": "M1 clean at 80%",
           "s2_m1detached_txt80": "M1 detached at 80%"}

# Colours: neutral grey for the contrastive baseline (as in the Stage 0 report), multilearner keeps the Stage 0
# report's aqua; categorical slots orange (80%), blue (MAE off), violet (the 80% controls). The three categorical hues
# pass the dataviz validator's all-pairs CVD check (worst deltaE 9.2); identity is also carried by axis labels.
C = {"contrastive": "#6f6e69", "multilearner": "#1baf7a", "txt80": "#eb6834", "mae0": "#2a78d6", "control": "#4a3aa7",
     "zeroshot": "#a3a29c"}
ARMCOL = {"s2_contrastive": C["contrastive"], "s2_multilearner": C["multilearner"], "s2_txt80": C["txt80"],
          "s2_mae0": C["mae0"], "s2_none_txt80": C["control"], "s2_m1clean_txt80": C["control"],
          "s2_m1detached_txt80": C["control"]}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"


# ----------------------------------------------------------------------------------------------- helpers
def mean(x):
    return statistics.mean(x)


def sd(x):
    return statistics.stdev(x) if len(x) > 1 else float("nan")


def ms(x, nd=2):
    return f"{mean(x):.{nd}f} ± {sd(x):.{nd}f}" if len(x) > 1 else f"{mean(x):.{nd}f}"


def welch_p(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def welch_ci(a, b, level=0.95):
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    hw = stats.t.ppf(0.5 + level / 2, df) * math.sqrt(va + vb)
    d = mean(a) - mean(b)
    return d, d - hw, d + hw, hw


def paired_p(da: dict, db: dict, key, seeds=CORE):
    shared = [s for s in seeds if s in da and s in db]
    if len(shared) < 2:
        return float("nan")
    return float(stats.ttest_rel([da[s][key] for s in shared], [db[s][key] for s in shared]).pvalue)


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def fp(p):
    return "n/a" if p != p else (f"{p:.3f}" if p >= 0.001 else f"{p:.1e}")


def cell(a, b, nd=2):
    return f"{mean(a) - mean(b):+.{nd}f} ({fp(welch_p(a, b))})"


def style(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------------------------------------- data
def seed_of(run_dir: Path) -> int:
    return int(yaml.safe_load((run_dir / "config.yaml").read_text())["seed"])


def load(root: Path):
    arms, other = defaultdict(dict), []
    for p in sorted(root.glob("*/run.json")):
        rj = json.loads(p.read_text())
        name = p.parent.name
        arm = name.split("_", 2)[2]
        if rj.get("status") != "completed":
            other.append((name, rj.get("status"), seed_of(p.parent)))
            continue
        # results.test holds the training-time test (with the losses); eval.test, present for the seed-42 Stage 0
        # fusion runs re-evaluated with evaluate.py, holds the extended metrics and wins where both have a key
        m = {k.removeprefix("test/").removeprefix("retrieval/"): v
             for k, v in (rj.get("results", {}).get("test") or {}).items()}
        m.update({k.removeprefix("test/").removeprefix("retrieval/"): v
                  for k, v in ((rj.get("eval") or {}).get("test") or {}).items()})
        m.update(run=name, hours=rj.get("duration_s", float("nan")) / 3600, commit=rj["git"]["commit"][:7],
                 best_epoch=rj.get("results", {}).get("best_epoch"), host=rj.get("host"), created=rj.get("created"))
        s = seed_of(p.parent)
        assert s not in arms[arm], (arm, s)
        arms[arm][s] = m
    return arms, other


def vals(runs: dict, key, seeds=None):
    return [runs[s][key] for s in sorted(runs) if seeds is None or s in seeds]


# ----------------------------------------------------------------------------------------------- sections
def inventory(base, ml, other):
    print("## A. Run inventory (completed runs; seeds from config.yaml)\n")
    print("| group | arm | n | seeds | commits | hours (mean) | best epochs | hosts |")
    print("|---|---|---|---|---|---|---|---|")
    for group, arms in (("default", base), ("ml_improve", ml)):
        for arm in sorted(arms):
            r = arms[arm]
            print(f"| {group} | {arm} | {len(r)} | {','.join(map(str, sorted(r)))} | "
                  f"{','.join(sorted({x['commit'] for x in r.values()}))} | {mean(vals(r, 'hours')):.2f} | "
                  f"{','.join(str(r[s]['best_epoch']) for s in sorted(r))} | {','.join(sorted({str(x['host']) for x in r.values()}))} |")
    print("\nNot completed (excluded):", other)
    s1 = [x for a, r in ml.items() if not a.startswith("s2_") for x in r.values()]
    s2 = [x for a, r in ml.items() if a in S2 for x in r.values()]
    s2c = [x for a, r in ml.items() if a in CTRL for x in r.values()]
    print(f"GPU-hours of completed runs: Stage 1 {sum(x['hours'] for x in s1):.1f} ({len(s1)} runs), Stage 2 arms "
          f"{sum(x['hours'] for x in s2):.1f} ({len(s2)} runs), 80% controls {sum(x['hours'] for x in s2c):.1f} "
          f"({len(s2c)} runs); total {sum(x['hours'] for x in s1 + s2 + s2c):.1f}")
    hrs = {a: mean(vals(ml[a], "hours")) for a in S2 + CTRL}
    print("Mean hours per Stage 2 run:", {a: round(h, 2) for a, h in hrs.items()})
    out = subprocess.run(["git", "-C", str(REPO), "log", "--format=%h %ad", "--date=format-local:%Y-%m-%d %H:%M",
                          "--", SPEC], capture_output=True, text=True, env={**os.environ, "TZ": "Europe/Amsterdam"}).stdout.strip()
    diff = subprocess.run(["git", "-C", str(REPO), "status", "--short", "--", SPEC], capture_output=True, text=True).stdout
    print(f"Spec commits (Amsterdam time): {out!r}; uncommitted changes: {diff.strip() or 'none'}")
    spec = (REPO / SPEC).read_text()
    for needle in ("5 for the final", "seeds 45 and 46"):
        line = next(l for l in spec.splitlines() if needle in l)
        print(f"Spec line with {needle!r}: {line.strip()}")
    print()


def stage1(base, ml):
    print("## B. Stage 1: every arm vs its 3-seed reference (Welch p when n >= 2)\n")
    for ref in ("contrastive", "fusion_multilearner"):
        r = base[ref]
        print(f"reference {ref}: n {len(r)}; " + "; ".join(f"{k} {ms(vals(r, M[k]))}" for k in
                                                              ("mAP@R", "PMRP", "rsum", "CxC R@1", "1K R@1")))
    c0, m0 = base["contrastive"], base["fusion_multilearner"]
    print("multilearner minus contrastive (Stage 0 baselines): " + "; ".join(
        f"{k} {cell(vals(m0, M[k]), vals(c0, M[k]))}" for k in ("mAP@R", "PMRP", "rsum")))
    cfg = yaml.safe_load((ML / ml["s2_txt80"][42]["run"] / "config.yaml").read_text())
    t, mm = cfg["train"], cfg["model"]
    print(f"Recipe (s2_txt80 seed 42 config.yaml): epochs {t['epochs']}, batch {t['batch_size']}, lr {t['lr']}, "
          f"lr_backbone {t['lr_backbone']}, warmup {t['warmup_steps']} steps, weight decay {t['weight_decay']}, "
          f"image mask ratio {mm['masking']['image_ratio']}, text ratio {mm['masking']['text_ratio']}, "
          f"monitor {mm['monitor']['metric']}, seeded sampler {t['seeded_sampler']}, backbone {mm['backbone']['pretrained']}")
    print("\n| arm | ID | n | mAP@R | ΔmAP@R (p) | ΔPMRP (p) | Δrsum (p) | ΔCxC R@1 (p) | Δ1K R@1 (p) | rule |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    rows = {}
    for arm, aid, label, ref in STAGE1:
        r, b = ml[arm], base[ref]
        d = {k: mean(vals(r, M[k])) - mean(vals(b, M[k])) for k in ("mAP@R", "PMRP", "rsum")}
        if ref == "contrastive":
            ok = d["mAP@R"] >= 0.3 and d["rsum"] >= -3
        else:
            ok = (d["mAP@R"] >= 0.3 or (d["PMRP"] >= 0.15 and d["mAP@R"] >= -0.2)) and d["rsum"] >= -3
        rows[arm] = d | {"ok": ok}
        print(f"| {label} | {aid} | {len(r)} | {ms(vals(r, M['mAP@R']))} | " +
              " | ".join(cell(vals(r, M[k]), vals(b, M[k])) for k in ("mAP@R", "PMRP", "rsum", "CxC R@1", "1K R@1"))
              + f" | {'pass' if ok else 'fail'} |")
    print("\nExact mAP@R differences at the +0.3 line:",
          {a: (round(rows[a]["mAP@R"], 4), f"{rows[a]['mAP@R'] - 0.3:+.4f} from the line")
           for a in ("multilearner_m2bcontent", "multilearner_m6maskedview")})
    print("Arm switches (config.yaml of each arm's first completed run):")
    for arm, aid, label, _ in STAGE1:
        cfg = yaml.safe_load((ML / ml[arm][min(ml[arm])]["run"] / "config.yaml").read_text())
        mm, t = cfg["model"], cfg["train"]
        print(f"  {aid} {label}: text_ratio {mm['masking']['text_ratio']}, text_mode {mm['masking'].get('text_mode', 'random')}, "
              f"mlm_image_source {mm.get('mlm_image_source', 'masked')}, pooled_conditioning {mm.get('pooled_conditioning', False)}, "
              f"weights {mm['loss']['weights']}, pooling {mm['pooling']}, epochs {t['epochs']}, lr_text {t.get('lr_text')}, "
              f"lr_vision {t.get('lr_vision')}, layer_decay {t.get('layer_decay', 1.0)}, freeze_vision_epochs {t.get('freeze_vision_epochs', 0)}")
    ranked = sorted([a for a in rows if rows[a]["ok"] and not a.startswith("contrastive")],
                    key=lambda a: (-rows[a]["mAP@R"], -rows[a]["PMRP"]))
    print("Masked arms passing the rule, ranked by mAP@R then PMRP:",
          [(a, round(rows[a]["mAP@R"], 3), round(rows[a]["PMRP"], 3)) for a in ranked])
    print("Recipe arms adopted:", [a for a in rows if a.startswith("contrastive") and rows[a]["ok"]])
    for a in ("multilearner_txt80", "multilearner_txt90"):
        print(f"{a}: per seed mAP@R {[round(ml[a][s][M['mAP@R']], 2) for s in sorted(ml[a])]} (seeds {sorted(ml[a])}), "
              f"PMRP {ms(vals(ml[a], 'pmrp'))}, rsum {ms(vals(ml[a], 'rsum'))}")
    t80 = ml["multilearner_txt80"]
    two = [t80[s][M["mAP@R"]] for s in (42, 44)]
    print(f"txt80 at the 2-seed advance check (seeds 42, 44): mAP@R {mean(two):.2f}")
    mae0 = ml["multilearner_mae0"]
    print(f"MAE off per seed mAP@R: {[round(mae0[s][M['mAP@R']], 2) for s in sorted(mae0)]}")
    r2, c = ml["contrastive_r2lr"], base["contrastive"]
    print("R2 vs contrastive: " + "; ".join(f"{k} {mean(vals(r2, M[k])):.2f} {cell(vals(r2, M[k]), vals(c, M[k]))}"
                                            for k in ("mAP@R", "PMRP", "rsum", "CxC R@1", "i2t R@1", "t2i R@1")))
    ml15 = base["fusion_multilearner"]
    print(f"R2 PMRP minus multilearner PMRP: {mean(vals(r2, 'pmrp')) - mean(vals(ml15, 'pmrp')):+.2f}")
    print()
    return rows


def dose(base, ml):
    print("## C. Dose curve (Stage 1; 15% is the Stage 0 multilearner baseline)\n")
    print("| ratio | n | mAP@R | rsum | PMRP | CxC R@1 | MLM test loss (n) | MAE test loss |")
    print("|---|---|---|---|---|---|---|---|")
    out = []
    for ratio, arm in DOSE:
        r = base["fusion_multilearner"] if arm is None else ml[arm]
        mlm = [x["loss_mlm"] for x in r.values() if "loss_mlm" in x]
        mae = [x["loss_mae"] for x in r.values() if "loss_mae" in x]
        out.append((ratio, r))
        print(f"| {ratio:.0%} | {len(r)} | {ms(vals(r, M['mAP@R']))} | {ms(vals(r, 'rsum'))} | {ms(vals(r, 'pmrp'))} | "
              f"{ms(vals(r, 'cxc/r1'))} | {mean(mlm):.3f} ({len(mlm)}) | {mean(mae):.3f} |")
    for arm in ("multilearner_mae0", "multilearner_mae0_txt40"):
        r = ml[arm]
        print(f"| {arm} | {len(r)} | {ms(vals(r, M['mAP@R']))} | {ms(vals(r, 'rsum'))} | {ms(vals(r, 'pmrp'))} | "
              f"{ms(vals(r, 'cxc/r1'))} | {mean(vals(r, 'loss_mlm')):.3f} | {mean(vals(r, 'loss_mae')):.3f} |")
    base15 = mean(vals(base["fusion_multilearner"], M["mAP@R"]))
    g80 = mean(vals(ml["multilearner_txt80"], M["mAP@R"])) - base15
    g25 = mean(vals(ml["multilearner_txt25"], M["mAP@R"])) - base15
    print(f"\nShare of the 80% mAP@R gain reached at 25%: {g25:.2f} of {g80:.2f} = {g25 / g80:.0%}")
    r15, r80 = mean(vals(base["fusion_multilearner"], "rsum")), mean(vals(ml["multilearner_txt80"], "rsum"))
    print(f"rsum 15% -> 80%: {r80 - r15:+.2f}")
    m15 = [x["loss_mlm"] for x in base["fusion_multilearner"].values() if "loss_mlm" in x]
    print(f"MLM test loss ratio 80% / 15%: {mean(vals(ml['multilearner_txt80'], 'loss_mlm')) / mean(m15):.2f}")
    print()
    return out


def replication(base, ml):
    print("## D. Stage 1 vs Stage 2 (same recipe; Stage 2 adds the seeded sampler)\n")
    pairs = [("contrastive", base["contrastive"], ml["s2_contrastive"]),
             ("fusion_multilearner", base["fusion_multilearner"], ml["s2_multilearner"]),
             ("txt80", ml["multilearner_txt80"], {s: ml["s2_txt80"][s] for s in CORE}),
             ("mae0", ml["multilearner_mae0"], ml["s2_mae0"])]
    print("| arm | Stage 1 n | Stage 2 n | mAP@R S1 | mAP@R S2 | Δ (p) | PMRP Δ (p) | rsum Δ (p) |")
    print("|---|---|---|---|---|---|---|---|")
    for name, a, b in pairs:
        print(f"| {name} | {len(a)} | {len(b)} | {mean(vals(a, M['mAP@R'])):.2f} | {mean(vals(b, M['mAP@R'])):.2f} | "
              + " | ".join(cell(vals(b, M[k]), vals(a, M[k])) for k in ("mAP@R", "PMRP", "rsum")) + " |")
    s2ml = mean(vals(ml["s2_multilearner"], M["mAP@R"]))
    s1ml = mean(vals(base["fusion_multilearner"], M["mAP@R"]))
    for arm in ("multilearner_txt80", "multilearner_mae0"):
        v = mean(vals(ml[arm], M["mAP@R"]))
        print(f"{arm}: Stage 1 gain vs Stage 1 multilearner {v - s1ml:+.2f}; Stage 1 mean minus Stage 2 multilearner "
              f"{v - s2ml:+.2f}")
    n_masked = sum(1 for a, *_ in STAGE1 if not a.startswith("contrastive"))
    print(f"Masked Stage 1 arms screened: {n_masked}")
    print()


def stage2(ml):
    print("## E. Stage 2 (seeded sampler, current recipe)\n")
    keys = ["mAP@R", "PMRP", "rsum", "CxC R@1", "1K R@1", "ECCV R-P", "ECCV R@1", "mAP@R i2t", "mAP@R t2i",
            "i2t R@1", "t2i R@1"]
    print("| arm | n | " + " | ".join(keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    core = {a: {s: ml[a][s] for s in CORE} for a in S2}
    for a in S2:
        print(f"| {a} | 3 | " + " | ".join(ms(vals(core[a], M[k])) for k in keys) + " |")
    allt = ml["s2_txt80"]
    print(f"| s2_txt80 (seeds 42-46) | {len(allt)} | " + " | ".join(ms(vals(allt, M[k])) for k in keys) + " |")
    print("\nPer-seed ECCV mAP@R:")
    for a in S2:
        print(f"  {a}: " + ", ".join(f"{s}: {ml[a][s][M['mAP@R']]:.2f}" for s in sorted(ml[a])))
    print("\nDifferences, seeds 42-44: diff (Welch p; paired-by-seed p)")
    for b in ("s2_contrastive", "s2_multilearner"):
        for v in ("s2_txt80", "s2_mae0") + (("s2_multilearner",) if b == "s2_contrastive" else ()):
            print(f"  {v} - {b}: " + "; ".join(
                f"{k} {mean(vals(core[v], M[k])) - mean(vals(core[b], M[k])):+.2f} "
                f"(W {fp(welch_p(vals(core[v], M[k]), vals(core[b], M[k])))}; P {fp(paired_p(ml[v], ml[b], M[k]))})"
                for k in keys))
    print("\nSuccess bar (spec sections 3 and 7): mean mAP@R above both baselines, Welch p < 0.05 Holm-corrected over the"
          " variants (per baseline); rsum >= multilearner - 1.5, PMRP >= multilearner - 0.05")
    result = {}
    for label, var in (("seeds 42-44", {v: [ml[v][s] for s in CORE] for v in ("s2_txt80", "s2_mae0")}),
                       ("final candidate: every completed seed", {v: [ml[v][s] for s in sorted(ml[v])]
                                                                  for v in ("s2_txt80", "s2_mae0")})):
        print(f"  [{label}]")
        for b in ("s2_contrastive", "s2_multilearner"):
            bv = vals(core[b], M["mAP@R"])
            raw = [welch_p([x[M["mAP@R"]] for x in var[v]], bv) for v in ("s2_txt80", "s2_mae0")]
            adj = holm(raw)
            for v, pr, pa in zip(("s2_txt80", "s2_mae0"), raw, adj):
                d, lo, hi, hw = welch_ci([x[M["mAP@R"]] for x in var[v]], bv)
                result[(label, v, b)] = (d, pr, pa, lo, hi)
                print(f"    {v} (n={len(var[v])}) vs {b}: {d:+.2f} [95% CI {lo:+.2f}, {hi:+.2f}], raw p {fp(pr)}, "
                      f"Holm p {fp(pa)}")
        mlm = core["s2_multilearner"]
        for v in ("s2_txt80", "s2_mae0"):
            rs = mean(x["rsum"] for x in var[v]) - mean(vals(mlm, "rsum"))
            pm = mean(x["pmrp"] for x in var[v]) - mean(vals(mlm, "pmrp"))
            met = all(result[(label, v, b)][0] > 0 and result[(label, v, b)][2] < 0.05
                      for b in ("s2_contrastive", "s2_multilearner")) and rs >= -1.5 and pm >= -0.05
            print(f"    {v}: rsum vs multilearner {rs:+.2f} (guard >= -1.5), PMRP {pm:+.2f} (guard >= -0.05) -> "
                  f"{'MET' if met else 'not met'}")
    print("\nWelch degrees of freedom of the success-bar tests (mAP@R):")
    for label, n in (("seeds 42-44", CORE), ("seeds 42-46", tuple(sorted(allt)))):
        for b in ("s2_contrastive", "s2_multilearner"):
            a, bv = [allt[x][M["mAP@R"]] for x in n], vals(core[b], M["mAP@R"])
            va, vb = statistics.variance(a) / len(a), statistics.variance(bv) / len(bv)
            df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(bv) - 1))
            print(f"  s2_txt80 {label} (n={len(a)}) vs {b} (n={len(bv)}): df {df:.1f}")
    print("Baseline seeds:", {b: sorted(ml[b]) for b in ("s2_contrastive", "s2_multilearner")},
          "-> seeds 45 and 46 have no baseline runs; the 5-seed test is unpaired, paired tests use seeds 42-44 only")
    print("Holm over one family of four (2 variants x 2 baselines), mAP@R:")
    for label in ("seeds 42-44", "final candidate: every completed seed"):
        keys4 = [(v, b) for v in ("s2_txt80", "s2_mae0") for b in ("s2_contrastive", "s2_multilearner")]
        raw4 = [result[(label, v, b)][1] for v, b in keys4]
        print(f"  [{label}] " + "; ".join(f"{v} vs {b}: raw {fp(r)}, Holm4 {fp(h)}"
                                          for (v, b), r, h in zip(keys4, raw4, holm(raw4))))
    # 5-seed txt80, other metrics vs both baselines
    print("\ns2_txt80 seeds 42-46 vs the 3-seed baselines (Welch p):")
    for b in ("s2_contrastive", "s2_multilearner"):
        print(f"  vs {b}: " + "; ".join(f"{k} {cell(vals(allt, M[k]), vals(core[b], M[k]))}"
                                        for k in ("mAP@R", "PMRP", "rsum", "CxC R@1", "1K R@1", "ECCV R-P")))
    pooled = statistics.mean(statistics.stdev(vals(core[a], M["mAP@R"])) for a in S2)
    print(f"\nMean seed std of mAP@R over the four arms (seeds 42-44): {pooled:.2f}")
    fine = 36.94 - 26.72
    print("Literature: ECCV Caption = Chun et al., ECCV 2022 (COCO 5K test pairs plus human-verified extra positives)")
    print(f"Context: contrastive fine-tuning gain over zero-shot in the baselines report: 36.94 - 26.72 = {fine:.2f} "
          f"mAP@R; PCME++ InfoNCE fine-tune of B/32 (lever review): 39.0")
    g5 = mean(vals(allt, M["mAP@R"])) - mean(vals(core["s2_multilearner"], M["mAP@R"]))
    print(f"5-seed gain over multilearner as a share of that fine-tuning gain: {g5:.2f} / {fine:.2f} = {g5 / fine:.1%}; "
          f"5-seed 80% mean below PCME++'s 39.0 by {39.0 - mean(vals(allt, M['mAP@R'])):.2f}")
    print("s2_txt80 run start and end (node clock, Amsterdam time):")
    for s in sorted(allt):
        rj = json.loads((ML / allt[s]["run"] / "run.json").read_text())
        print(f"  seed {s}: {rj['created'].replace('T', ' ')[:16]} to {rj['ended'].replace('T', ' ')[:16]} on {rj['host']}")
    print()
    return result


def losses_and_controls(ml):
    print("## F. Test losses per Stage 2 arm (nats; MLM = cross-entropy per masked token)\n")
    print("| arm | MLM | MAE | InfoNCE |")
    print("|---|---|---|---|")
    for a in S2[1:] + CTRL:
        r = {s: ml[a][s] for s in CORE}
        print(f"| {a} | {ms(vals(r, 'loss_mlm'), 3)} | {ms(vals(r, 'loss_mae'), 3)} | {ms(vals(r, 'loss_contrastive'), 3)} |")
    r = {s: ml["s2_contrastive"][s] for s in CORE}
    print(f"| s2_contrastive | | | {ms(vals(r, 'loss_contrastive'), 3)} |")
    t80 = mean(vals({s: ml['s2_txt80'][s] for s in CORE}, "loss_mlm"))
    m15 = mean(vals(ml["s2_multilearner"], "loss_mlm"))
    none = mean(vals(ml["s2_none_txt80"], "loss_mlm"))
    clean = mean(vals(ml["s2_m1clean_txt80"], "loss_mlm"))
    print(f"\nMLM loss 80% / 15%: {t80 / m15:.2f}; image lowers the 80% MLM loss by {none - t80:.2f} nats (fusion_none "
          f"{none:.2f} vs {t80:.2f}); the clean image by a further {t80 - clean:.2f}")
    print(f"Perplexity per masked token: 15% {math.exp(m15):.1f}, 80% {math.exp(t80):.1f}, 80% text only "
          f"{math.exp(none):.1f}, 80% clean image {math.exp(clean):.1f}")

    print("\n## G. 80% controls (3 seeds each) vs s2_txt80 (seeds 42-44) and vs s2_multilearner\n")
    ref = {s: ml["s2_txt80"][s] for s in CORE}
    keys = ("mAP@R", "PMRP", "rsum", "CxC R@1", "1K R@1")
    print("| arm | " + " | ".join(f"{k}" for k in keys) + " |")
    print("|---|" + "---|" * len(keys))
    for a in ["s2_multilearner", "s2_txt80"] + CTRL:
        print(f"| {a} | " + " | ".join(ms(vals(ml[a], M[k], CORE)) for k in keys) + " |")
    print("\nvs s2_txt80: diff (Welch p; paired p)")
    for a in CTRL:
        print(f"  {a}: " + "; ".join(
            f"{k} {mean(vals(ml[a], M[k])) - mean(vals(ref, M[k])):+.2f} (W {fp(welch_p(vals(ml[a], M[k]), vals(ref, M[k])))}; "
            f"P {fp(paired_p(ml[a], ref, M[k]))})" for k in keys))
    raw = [welch_p(vals(ml[a], M["mAP@R"]), vals(ref, M["mAP@R"])) for a in CTRL]
    print("Holm over the three controls (mAP@R vs s2_txt80, Welch): " +
          ", ".join(f"{a} raw {fp(r)} adj {fp(h)}" for a, r, h in zip(CTRL, raw, holm(raw))))
    fam = [(a, k) for a in CTRL for k in keys]
    rawf = [welch_p(vals(ml[a], M[k]), vals(ref, M[k])) for a, k in fam]
    print(f"Holm over all {len(fam)} control-vs-80% Welch tests of this table: " +
          "; ".join(f"{a.removeprefix('s2_')} {k} {fp(h)}" for (a, k), h in zip(fam, holm(rawf))))
    pr = [r for (a, k), r in zip(fam, rawf) if k in ("PMRP", "rsum")]
    print(f"Largest raw p of the six PMRP and rsum tests: {fp(max(pr))}; Bonferroni threshold over {len(fam)} tests: "
          f"{0.05 / len(fam):.4f}")
    print("vs s2_multilearner: diff (Welch p)")
    for a in CTRL:
        print(f"  {a}: " + "; ".join(f"{k} {cell(vals(ml[a], M[k]), vals(ml['s2_multilearner'], M[k]))}" for k in keys))
    print("Per-seed mAP@R:", {a: [round(ml[a][s][M["mAP@R"]], 2) for s in CORE] for a in CTRL})
    det, cln = ml["s2_m1detached_txt80"], ml["s2_m1clean_txt80"]
    print("M1 detached minus M1 clean at 80%: " + "; ".join(
        f"{k} {cell(vals(det, M[k]), vals(cln, M[k]))}" for k in ("mAP@R", "PMRP", "rsum")))
    h80, hc = mean(vals(ml["s2_txt80"], "hours")), mean(vals(ml["s2_contrastive"], "hours"))
    print(f"Run time, 80% arm over contrastive: {h80:.2f} h / {hc:.2f} h = {h80 / hc:.2f}")
    print()


def swaps(ml):
    print("## H. Tower swaps (stage2_controls diagnostics): arm tower + same-seed s2_contrastive other tower\n")
    diag = json.loads(DIAG.read_text())
    runs, sw = diag["runs"], diag["swaps"]
    print(f"Encoded runs: {len(runs)} (incl. zero-shot); swaps: {len(sw)}")
    # consistency with each run's own test
    dr = [abs(runs[n]["test"]["rsum"] - ml[n.split('_', 2)[2]][runs[n]["seed"]]["rsum"]) for n in runs if n != "zeroshot"]
    print(f"rsum diag vs run.json, max abs diff over {len(dr)} runs: {max(dr):.3f}")
    old = json.loads(DIAG_OLD.read_text())
    shared = [n for n in old["runs"] if n in runs]
    dv = max(abs(old["runs"][n]["vwsd"]["vwsd/hit1"] - runs[n]["vwsd"]["vwsd/hit1"]) for n in shared)
    ds = max(abs(old["swaps"][k]["eccv/map_at_r"] - sw[k]["eccv/map_at_r"]) for k in old["swaps"])
    print(f"stage2 vs stage2_controls: {len(shared)} shared runs, max |VWSD Hit@1 diff| {dv:.3f}, "
          f"{len(old['swaps'])} shared swaps, max |mAP@R diff| {ds:.4f}")
    table = {}
    for a in [x for x in S2 if x != "s2_contrastive"] + CTRL:
        img, txt = {}, {}
        for s in CORE:
            n = ml[a][s]["run"]
            img[s] = sw[f"{n}__img+contrastive_txt"]
            txt[s] = sw[f"contrastive_img+{n}__txt"]
        table[a] = (img, txt)
    con = {s: ml["s2_contrastive"][s] for s in CORE}
    keys = [("eccv/map_at_r", "mAP@R"), ("pmrp", "PMRP"), ("rsum", "rsum")]
    print(f"\ns2_contrastive full model: " + "; ".join(f"{lbl} {ms(vals(con, k))}" for k, lbl in keys))
    print("\n| arm | full mAP@R | image swap mAP@R | text swap mAP@R | image swap PMRP | image swap rsum | text swap rsum |")
    print("|---|---|---|---|---|---|---|")
    for a, (img, txt) in table.items():
        print(f"| {a} | {ms(vals(ml[a], M['mAP@R'], CORE))} | {ms(vals(img, 'eccv/map_at_r'))} | "
              f"{ms(vals(txt, 'eccv/map_at_r'))} | {ms(vals(img, 'pmrp'))} | {ms(vals(img, 'rsum'))} | "
              f"{ms(vals(txt, 'rsum'))} |")
    i80 = table["s2_txt80"][0]
    print("\nImage swap vs s2_txt80's image swap: diff (Welch p; paired p)")
    for a, (img, _) in table.items():
        if a == "s2_txt80":
            continue
        print(f"  {a}: " + "; ".join(
            f"{lbl} {mean(vals(img, k)) - mean(vals(i80, k)):+.2f} (W {fp(welch_p(vals(img, k), vals(i80, k)))}; "
            f"P {fp(paired_p(img, i80, k))})" for k, lbl in keys))
    print("Image swap vs s2_contrastive full model: diff (Welch p)")
    for a, (img, _) in table.items():
        print(f"  {a}: " + "; ".join(f"{lbl} {cell(vals(img, k), vals(con, k))}" for k, lbl in keys))
    print("Text swap vs s2_txt80's text swap and vs s2_contrastive full (mAP@R): ")
    t80 = table["s2_txt80"][1]
    for a, (_, txt) in table.items():
        print(f"  {a}: vs txt80 text swap {cell(vals(txt, 'eccv/map_at_r'), vals(t80, 'eccv/map_at_r'))}; vs contrastive "
              f"{cell(vals(txt, 'eccv/map_at_r'), vals(con, M['mAP@R']))}")
    for a, (img, txt) in table.items():
        full = vals(ml[a], M["mAP@R"], CORE)
        print(f"  {a}: full minus image swap (mAP@R) {mean(full) - mean(vals(img, 'eccv/map_at_r')):+.2f}; "
              f"full minus text swap {mean(full) - mean(vals(txt, 'eccv/map_at_r')):+.2f}")
    lo = min(mean(vals(txt, "eccv/map_at_r")) for _, txt in table.values())
    hi = max(mean(vals(txt, "eccv/map_at_r")) for _, txt in table.values())
    print(f"Text-swap mAP@R range over arms: {lo:.2f} to {hi:.2f}")
    print()
    return table, con, runs


def vwsd(ml, runs):
    print("## I. VWSD (SemEval-2023 Task 1, English test, 463 items)\n")
    z = runs["zeroshot"]["vwsd"]
    print(f"zero-shot: Hit@1 {z['vwsd/hit1']:.2f}, MRR {z['vwsd/mrr']:.2f}, n {z['vwsd/n']:.0f}")
    reg = (REPO / "tests/20261003_ml_improve/runs.md").read_text()
    i = reg.find("VWSD Hit@1 58.10")
    print("Stage 0 CPU dry run (runs.md): " + (reg[i:i + 40].split("\n")[0].rstrip(". ") if i >= 0 else "NOT FOUND") +
          f"; GPU minus CPU zero-shot Hit@1 {z['vwsd/hit1'] - 58.10:+.2f} (one item = {100 / 463:.3f})")
    by = defaultdict(dict)
    for n, r in runs.items():
        if n != "zeroshot":
            by[n.split("_", 2)[2]][r["seed"]] = r["vwsd"]
    con = by["s2_contrastive"]
    mlb = by["s2_multilearner"]
    print("| model | Hit@1 | MRR | Hit@1 vs contrastive (p) | MRR vs contrastive (p) | Hit@1 vs multilearner (p) |")
    print("|---|---|---|---|---|---|")
    for a in S2 + CTRL:
        h, m = vals(by[a], "vwsd/hit1"), vals(by[a], "vwsd/mrr")
        print(f"| {a} | {ms(h)} | {ms(m)} | " +
              ("| | |" if a == "s2_contrastive" else
               f"{cell(h, vals(con, 'vwsd/hit1'))} | {cell(m, vals(con, 'vwsd/mrr'))} | "
               + ("|" if a == "s2_multilearner" else f"{cell(h, vals(mlb, 'vwsd/hit1'))} |")))
    ch = vals(con, "vwsd/hit1")
    print(f"\nFine-tuning cost (contrastive - zero-shot): {mean(ch) - z['vwsd/hit1']:+.2f} Hit@1")
    masked = [x for a in S2[1:] + CTRL for x in vals(by[a], "vwsd/hit1")]
    print(f"All 18 masked Stage 2 runs: {ms(masked)} (range {min(masked):.2f} to {max(masked):.2f}); contrastive runs "
          f"{[round(x, 2) for x in ch]}; masked - contrastive {mean(masked) - mean(ch):+.2f} (Welch p "
          f"{fp(welch_p(masked, ch))}); masked runs below the lowest contrastive run: "
          f"{sum(x < min(ch) for x in masked)} of {len(masked)}")
    per_arm = [mean(vals(by[a], "vwsd/hit1")) - mean(ch) for a in S2[1:] + CTRL]
    print(f"Per-arm Hit@1 minus contrastive: range {min(per_arm):+.2f} to {max(per_arm):+.2f}")
    print(f"One item = {100 / 463:.3f} points; binomial SE at 53% = {100 * math.sqrt(0.53 * 0.47 / 463):.2f} points")
    s0 = json.loads(DIAG_S0.read_text())["runs"]
    s0by = defaultdict(list)
    for n, r in s0.items():
        if n != "zeroshot":
            s0by[r["model"]].append(r["vwsd"]["vwsd/hit1"])
    s0c, s0m = s0by["contrastive"], s0by["fusion_multilearner"]
    print(f"Stage 0 (unseeded sampler): contrastive {ms(s0c)}, multilearner {ms(s0m)} ({cell(s0m, s0c)})")
    print(f"Stage 2 multilearner minus Stage 0 multilearner: {cell(vals(mlb, 'vwsd/hit1'), s0m)}; Stage 2 contrastive "
          f"minus Stage 0 contrastive: {cell(ch, s0c)}")
    pc, pm = s0c + ch, s0m + vals(mlb, "vwsd/hit1")
    print(f"Pooled Stage 0 + Stage 2 (6 runs each): multilearner - contrastive {cell(pm, pc)}")
    print()
    return by, z


def stratified():
    print("## J. Stratified ECCV mAP@R (stratified_eccv.py; Stage 2 seeds 42-44; bootstrap over queries)\n")
    d = json.loads(STRAT.read_text())
    mx = max(v["max_abs_diff"] for v in d["validation"])
    print(f"Validation: per-query mAP@R reproduces run.json for {len(d['validation'])} runs, max abs diff {mx:.3f}; "
          f"bootstrap resamples {d['n_bootstrap']}; seeds {d['seeds']}")
    print("ECCV Caption query subset: " + ", ".join(
        f"{key.split(':')[0]} {sum(d['arms'][key]['s2_txt80']['n']):,}" for key in ("i2t:R", "t2i:R")))
    for key, names in d["strata_defs"].items():
        print(f"\n{key}: strata {names}, n {d['arms'][key]['s2_txt80']['n']}")
        for a, r in d["arms"][key].items():
            print(f"  {a}: stratum means {[round(x, 2) for x in r['mean']]}, all {r['overall']:.2f}")
        for pair, r in d["diffs"][key].items():
            print(f"  {pair}: " + ", ".join(f"{m:+.2f} [{c[0]:+.2f}, {c[1]:+.2f}]" for m, c in zip(r["mean"], r["mean_ci"]))
                  + f"; all {r['overall']:+.2f} [{r['overall_ci'][0]:+.2f}, {r['overall_ci'][1]:+.2f}]; high-low "
                  f"{r['high_minus_low']:+.2f} [{r['high_minus_low_ci'][0]:+.2f}, {r['high_minus_low_ci'][1]:+.2f}]; "
                  f"per-seed high-low {[round(x, 2) for x in r['per_seed_high_minus_low']]}")
    hw = [(c[1] - c[0]) / 2 for key in d["diffs"] for r in d["diffs"][key].values() for c in [r["high_minus_low_ci"]]]
    print(f"\nHigh-minus-low CI half-widths: {min(hw):.2f} to {max(hw):.2f}")
    hw80 = {key: (d["diffs"][key]["s2_txt80-s2_multilearner"]["high_minus_low_ci"][1]
                  - d["diffs"][key]["s2_txt80-s2_multilearner"]["high_minus_low_ci"][0]) / 2 for key in d["diffs"]}
    print("High-minus-low CI half-widths, 80% minus multilearner: " +
          ", ".join(f"{k} {v:.2f}" for k, v in hw80.items()) + f" (range {min(hw80.values()):.2f} to {max(hw80.values()):.2f})")
    for key in ("i2t:R", "t2i:R"):
        gain = d["diffs"][key]["s2_txt80-s2_multilearner"]["mean"]
        basev = d["arms"][key]["s2_multilearner"]["mean"]
        print(f"{key}: 80% minus multilearner relative to multilearner's stratum mean: "
              + ", ".join(f"{g / b:+.1%}" for g, b in zip(gain, basev)))
    print()
    return d


RECORDS = ["2026-10-04 00:55", "2026-10-04 about 02:26", "2026-10-04 02:35", "2026-10-04 09:53", "2026-10-04 18:13",
           "2026-10-04 21:14", "2026-10-05 00:14", "2026-10-05 01:36", "2026-10-05 01:47", "2026-10-05 08:54", "2026-10-05 09:14: M6", "2026-10-05 09:16",
           "2026-10-05 10:53: Decision", "2026-10-05 16:30: Ruling", "2026-10-05T20:33", "2026-10-05 21:00",
           "2026-10-06 00:52", "2026-10-06 16:15", "2026-10-06 16:41"]


def records():
    print("## K. Record lines cited from tests/20261003_ml_improve/runs.md (Amsterdam time; first 260 characters, two in full)\n")
    lines = (REPO / "tests/20261003_ml_improve/runs.md").read_text().splitlines()
    for prefix in RECORDS:
        hit = [l for l in lines if l.startswith(f"- {prefix}")]
        full = prefix in ("2026-10-05 16:30: Ruling", "2026-10-05T20:33")  # the success-bar reading: print in full
        print(f"- [{prefix}] " + ((hit[0][2:] if full else hit[0][2:262]) if hit else "NOT FOUND"))
    print()


# ----------------------------------------------------------------------------------------------- figures
def fig_stage1(base, ml, rows):
    order = [a for a, *_ in STAGE1]
    lab = {a: f"{aid} {label}" for a, aid, label, _ in STAGE1}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), sharey=True)
    for ax, (k, thr, title) in zip(axes, ((M["mAP@R"], 0.3, "ECCV mAP@R minus reference"),
                                          ("rsum", -3.0, "rsum minus reference"))):
        for y, (a, aid, label, ref) in enumerate(STAGE1):
            r, b = ml[a], base[ref]
            bm = mean(vals(b, k))
            col = (C["txt80"] if a == "multilearner_txt80" else C["mae0"] if a == "multilearner_mae0"
                   else C["contrastive"] if ref == "contrastive" else C["multilearner"])
            xs = [x - bm for x in vals(r, k)]
            ax.scatter(xs, [y] * len(xs), s=26, color=col, edgecolor=SURFACE, linewidth=1.2, zorder=3)
            ax.plot([mean(xs)] * 2, [y - 0.32, y + 0.32], color=INK, lw=1.8, zorder=4)
        ax.axvline(0, color=MUTED, lw=0.9)
        ax.axvline(thr, color=INK, lw=1, ls="--")
        ax.text(thr, -1.0, " advance +0.3" if thr > 0 else "guard -3 ", fontsize=7.5, color=INK,
                va="center", ha="left" if thr > 0 else "right")
        ax.set_title(title, fontsize=10, color=INK)
        style(ax, "x")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([lab[a] + ("" if rows[a]["ok"] else "  (fail)") for a in order], fontsize=8)
    axes[0].set_ylim(len(STAGE1) - 0.5, -1.6)  # inverted, with an empty top row for the threshold labels
    axes[1].set_xlim(-24, 9)
    handles = [Line2D([], [], marker="o", ls="", color=C["multilearner"], label="masked arm (vs multilearner, 3 seeds)"),
               Line2D([], [], marker="o", ls="", color=C["txt80"], label="text masking 80% (advanced)"),
               Line2D([], [], marker="o", ls="", color=C["mae0"], label="MAE off (advanced)"),
               Line2D([], [], marker="o", ls="", color=C["contrastive"], label="recipe arm (vs contrastive, 3 seeds)"),
               Line2D([], [], color=INK, lw=1.8, label="arm mean")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "stage1_arms.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_dose(base, ml, curve):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    con = base["contrastive"]
    for ax, (k, lbl) in zip(axes, ((M["mAP@R"], "ECCV mAP@R"), ("rsum", "rsum"), ("pmrp", "PMRP"),
                                   ("loss_mlm", "MLM test loss per masked token (nats)"))):
        xs, ms_ = [], []
        for ratio, r in curve:
            v = [x[k] for x in r.values() if k in x]
            ax.scatter([100 * ratio] * len(v), v, s=16, color=C["multilearner"], alpha=0.55, edgecolor="none", zorder=2)
            xs.append(100 * ratio)
            ms_.append(mean(v))
        ax.plot(xs, ms_, color=C["multilearner"], lw=2, marker="o", ms=6, zorder=3, label="multilearner, text ratio")
        for arm, ratio in (("multilearner_mae0", 15), ("multilearner_mae0_txt40", 40)):
            v = vals(ml[arm], k)
            ax.scatter([ratio + 2.5] * len(v), v, s=16, color=C["mae0"], alpha=0.55, edgecolor="none", zorder=2)
            ax.scatter([ratio + 2.5], [mean(v)], s=46, color=C["mae0"], marker="D", edgecolor=SURFACE, zorder=4,
                       label="MAE off (15%, 40%)" if ratio == 15 else None)
        if k != "loss_mlm":
            ax.axhline(mean(vals(con, k)), color=C["contrastive"], lw=1.2, ls="--", label="contrastive (3 seeds)")
        ax.set_xticks([15, 25, 40, 60, 80, 90])
        ax.set_xlabel("text-masking ratio (%)", fontsize=8.5, color=INK)
        ax.set_title(lbl, fontsize=9.5, color=INK)
        style(ax)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "dose.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_stage2(ml):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, (k, lbl) in zip(axes, ((M["mAP@R"], "ECCV mAP@R"), ("rsum", "rsum"), ("pmrp", "PMRP"))):
        for s in CORE:
            ax.plot(range(4), [ml[a][s][k] for a in S2], color=GRID, lw=1.1, zorder=1)
        for i, a in enumerate(S2):
            core = [ml[a][s][k] for s in CORE]
            ax.scatter([i] * 3, core, s=34, color=ARMCOL[a], edgecolor=SURFACE, linewidth=1.4, zorder=3)
            extra = [ml[a][s][k] for s in sorted(ml[a]) if s not in CORE]
            if extra:
                ax.scatter([i + 0.12] * len(extra), extra, s=34, facecolor=SURFACE, edgecolor=ARMCOL[a], linewidth=1.6,
                           zorder=3)
                allv = vals(ml[a], k)
                ax.plot([i - 0.25, i + 0.25], [mean(allv)] * 2, color=ARMCOL[a], lw=1.6, ls=":", zorder=4)
            ax.plot([i - 0.25, i + 0.25], [mean(core)] * 2, color=INK, lw=1.8, zorder=4)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["contrastive", "multilearner\n(15%)", "text\nmasking 80%", "MAE off"], fontsize=8)
        ax.set_title(lbl, fontsize=10, color=INK)
        style(ax)
    handles = [Line2D([], [], marker="o", ls="", color=MUTED, label="seeds 42-44 (same data order per seed)"),
               Line2D([], [], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=C["txt80"],
                      label="80%: seeds 45, 46 (final candidate)"),
               Line2D([], [], color=INK, lw=1.8, label="mean, seeds 42-44"),
               Line2D([], [], color=C["txt80"], lw=1.6, ls=":", label="80%: mean, seeds 42-46"),
               Line2D([], [], color=GRID, lw=1.1, label="same seed")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "stage2_seeds.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_controls(ml, table, con):
    arms = ["s2_multilearner", "s2_txt80"] + CTRL
    labels = ["multilearner\n(15%)", "text\nmasking 80%", "fusion_none\nat 80%", "M1 clean\nat 80%",
              "M1 detached\nat 80%"]
    cm = mean(vals(con, M["mAP@R"]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    panels = [("full model (own towers)", lambda a: {s: ml[a][s] for s in CORE}, M["mAP@R"]),
              ("arm image tower + contrastive text tower", lambda a: table[a][0], "eccv/map_at_r"),
              ("contrastive image tower + arm text tower", lambda a: table[a][1], "eccv/map_at_r")]
    for ax, (title, get, k) in zip(axes, panels):
        for i, a in enumerate(arms):
            v = vals(get(a), k)
            ax.scatter([i] * len(v), v, s=34, color=ARMCOL[a], edgecolor=SURFACE, linewidth=1.4, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [mean(v)] * 2, color=INK, lw=1.8, zorder=4)
        ax.axhline(cm, color=C["contrastive"], lw=1.2, ls="--")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(title, fontsize=9.5, color=INK)
        style(ax)
    axes[0].set_ylabel("ECCV mAP@R", fontsize=9, color=INK)
    handles = [Line2D([], [], marker="o", ls="", color=C["multilearner"], label="multilearner, 15% text masking"),
               Line2D([], [], marker="o", ls="", color=C["txt80"], label="text masking 80%"),
               Line2D([], [], marker="o", ls="", color=C["control"], label="80% controls"),
               Line2D([], [], color=INK, lw=1.8, label="mean of seeds 42-44"),
               Line2D([], [], color=C["contrastive"], lw=1.2, ls="--", label="contrastive full model")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "controls_swaps.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_vwsd(by, z):
    arms = S2 + CTRL
    labels = ["contrastive", "multilearner\n(15%)", "text\nmasking 80%", "MAE off", "fusion_none\nat 80%",
              "M1 clean\nat 80%", "M1 detached\nat 80%"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, a in enumerate(arms):
        v = vals(by[a], "vwsd/hit1")
        ax.scatter([i] * len(v), v, s=34, color=ARMCOL[a], edgecolor=SURFACE, linewidth=1.4, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [mean(v)] * 2, color=INK, lw=1.8, zorder=4)
    ax.axhline(z["vwsd/hit1"], color=C["zeroshot"], lw=1.2, ls=":", label="zero-shot CLIP B/32")
    ax.axhline(mean(vals(by["s2_contrastive"], "vwsd/hit1")), color=C["contrastive"], lw=1.2, ls="--",
               label="contrastive mean")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("VWSD Hit@1 (%)", fontsize=9, color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "vwsd.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_strat(d):
    series = [("s2_txt80-s2_multilearner", C["txt80"], "o", True, "80% minus multilearner"),
              ("s2_txt80-s2_contrastive", C["txt80"], "s", False, "80% minus contrastive"),
              ("s2_mae0-s2_multilearner", C["mae0"], "D", True, "MAE off minus multilearner")]
    titles = {"i2t:R": "i2t, by number of ECCV positives R", "t2i:R": "t2i, by number of ECCV positives R",
              "i2t:diversity": "i2t, by diversity of the image's 5 captions"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), sharey=True)
    for ax, key in zip(axes, d["strata_defs"]):
        names = d["strata_defs"][key]
        for off, (pair, col, mk, filled, lbl) in zip((-0.18, 0.0, 0.18), series):
            r = d["diffs"][key][pair]
            m = np.array(r["mean"])
            c = np.array(r["mean_ci"])
            ax.errorbar(np.arange(3) + off, m, yerr=[m - c[:, 0], c[:, 1] - m], fmt=mk, ms=6, capsize=3, color=col,
                        markerfacecolor=col if filled else SURFACE, markeredgewidth=1.5, lw=1.2, label=lbl)
        ax.axhline(0, color=MUTED, lw=0.9)
        n = d["arms"][key]["s2_txt80"]["n"]
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{nm.replace('<=', '≤').replace('>=', '≥')}\n(n={c})" for nm, c in zip(names, n)],
                           fontsize=7.5)
        ax.set_title(titles[key], fontsize=9.5, color=INK)
        style(ax)
    axes[0].set_ylabel("mAP@R difference (points)", fontsize=9, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "stratified.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base, other0 = load(DEFAULT)
    ml, other = load(ML)
    inventory(base, ml, other0 + other)
    rows = stage1(base, ml)
    curve = dose(base, ml)
    replication(base, ml)
    stage2(ml)
    losses_and_controls(ml)
    table, con, runs = swaps(ml)
    by, z = vwsd(ml, runs)
    d = stratified()
    records()
    fig_stage1(base, ml, rows)
    fig_dose(base, ml, curve)
    fig_stage2(ml)
    fig_controls(ml, table, con)
    fig_vwsd(by, z)
    fig_strat(d)
    print("Figures written to", OUT.relative_to(REPO), sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
