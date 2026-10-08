"""Tables and figures of docs/reports/auto/v1/2026-10-08_hb_emotion_readout.md: the first pre-registered H-b
experiment on ArtELingo (D4 emotion readout against dense AL-28 votes, D6 view sampling, D7 partial captions).

Reads (never writes):
  res/hb/d4_final/hb_d4.json                  the final D4/D6/D7 tables (scripts/hb_table.py, 12 runs, B = 10,000)
  res/hb/d4_provisional/hb_d4.json            the provisional look (9 runs), for its decision only
  res/hb/gate/gate_all.json                   the seed-42 gate before wave 2
  res/artelingo/multimae/hb_d4/*/{run.json,config.yaml,metrics.jsonl}
                                              the 12 training runs: test retrieval, per-epoch validation, best epoch
  res/artelingo/hb_encoded/<run>/{encode.npz,meta.json}
                                              encoded AL-28 logits and embeddings (re-derives the primary JSD as a
                                              check, then the entropy and spread-from-prior analysis of section 4.5,
                                              the seed-level Welch check of D4 (section E), the D6 controls of Table 11
                                              (section G) and the D7 prior-only log-loss of Table 12 (section H))
  /data/PDD/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv
                                              AL-28 vote counts (data section)
  res/hb/coco_tables_d1/hb_coco.json          D1 summary over seeds (scripts/hb_coco_table.py, 15 checkpoints, every
                                              number recomputed from scores.npz with the snapped scorer)
  res/hb/d1/<run>/{d1.json,scores.npz}        D1 per checkpoint: tuned alpha / beta, timing, the dual top 50 and the
                                              decoder and null-image scores (random-order and oracle baselines,
                                              score correlations, the control's exact-order check)
  res/coco/multimae/ml_improve/<run>/run.json the D1 checkpoints' own test metrics (ECCV i2t mAP@R, MLM loss)
  /data/SSD/coco/annotations                  ECCV Caption i2t positives (via mmae.engine.hb.coco_probes.eccv_i2t)
  res/hb/coco_tables/hb_coco.json             D2 summary over seeds (scripts/hb_coco_table.py: kill rule, both-covered,
                                              ceilings; every balance and selection number is re-derived below)
  res/hb/d2/<run or zeroshot>/{d2.json,d2_scores.npz}, res/hb/d2/pairs.json
                                              D2 per model: both-covered@k, the raw dual and decoder scores of A's and
                                              B's 10 captions per pair and condition, the selection items; the 1,000 pairs
  res/coco/multimae/ml_improve/<run>/{config.yaml,checkpoints/best.pt} and /data/SSD/coco/images
                                              the collapse check (section K): every D2 checkpoint plus the three
                                              s2_m1detached_txt80 runs and zero-shot CLIP, on the first 500 test images;
                                              this part runs on the GPU when one is visible (about 1 minute)
  git history (commit times of the spec, its amendment and the readout fix wave)

Writes to docs/reports/assets/2026-10-08_hb_emotion_readout/:
  arms.png        Figure 1: what the caption decoder reads in each arm, and where the emotion slot sits
  training.png    Figure 2: validation rsum and validation emotion loss per epoch
  d4_primary.png  Figure 3: D4 co-primary metrics per arm, with the prior and the split-half ceiling
  d4_thirds.png   Figure 4: JSD and entropy Spearman by thirds of human entropy
  d6_views.png    Figure 5: D6, the readout over K views and the full image
  d7_partial.png  Figure 6: D7, log-loss of the annotator's emotion against visible content tokens
  d1_rerank.png   Figure 7: D1, paired difference of re-ranker (a) minus the dual encoder by alpha, per family
  d2_balance.png  Figure 8: D2, dual and decoder balance against the blend weight and the patch mix, per family
  d2_selection.png Figure 9: D2, the selection index at the 0.5 blend against visible content tokens j
and prints every number of the report to stdout.

  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 flock -n -o -E 75 /tmp/gpu0.lock /root/miniconda3/envs/MultiMAE/bin/python \
      docs/reports/assets/build_2026-10-08_hb_emotion_readout.py
(with CUDA_VISIBLE_DEVICES= and no lock it runs on the CPU; only section K uses torch)
"""
from __future__ import annotations

import csv
import json
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from mmae.engine.hb import analysis, baselines, bootstrap, calibrate, coco_probes, metrics, tables  # noqa: E402

cp = coco_probes

FINAL = REPO / "res/hb/d4_final/hb_d4.json"
PROV = REPO / "res/hb/d4_provisional/hb_d4.json"
GATE = REPO / "res/hb/gate/gate_all.json"
RUNS = REPO / "res/artelingo/multimae/hb_d4"
ENC = REPO / "res/artelingo/hb_encoded"
AL28_CSV = Path("/data/PDD/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv")
SPEC = "docs/superpowers/specs/2026-10-07-hb-first-artelingo-design.md"
OUT = Path(__file__).resolve().parent / "2026-10-08_hb_emotion_readout"
D1_TABLE = REPO / "res/hb/coco_tables_d1/hb_coco.json"
D1_DIR = REPO / "res/hb/d1"
COCO_RUNS = REPO / "res/coco/multimae/ml_improve"
COCO_ANN = Path("/data/SSD/coco/annotations")
# D1 families in the order of the report's table; colours from the same four validated hues plus grey for the
# image-blind control (ML-15 takes aqua, M1 clean the violet of the other full-image decoder, Par-cap)
D1_FAMS = ("s2_txt80_mae0", "s2_txt80", "s2_multilearner", "s2_m1clean_txt80", "s2_none_txt80")
D1_LABEL = {"s2_txt80_mae0": "ML-80", "s2_txt80": "ML-80+MAE", "s2_multilearner": "ML-15",
            "s2_m1clean_txt80": "M1 clean", "s2_none_txt80": "text-only control"}
D1_COL = {"s2_txt80_mae0": "#eb6834", "s2_txt80": "#2a78d6", "s2_multilearner": "#1baf7a",
          "s2_m1clean_txt80": "#4a3aa7", "s2_none_txt80": "#6f6e69"}
D1_MARK = {"s2_txt80_mae0": "o", "s2_txt80": "s", "s2_multilearner": "^", "s2_m1clean_txt80": "D", "s2_none_txt80": "x"}
D1_SCOPES = ("all", "R<=15", "R16-20", "R>=21")
D2_TABLE = REPO / "res/hb/coco_tables/hb_coco.json"
D2_DIR = REPO / "res/hb/d2"
D2_FAMS = ("zeroshot", "s2_contrastive", "s2_txt80_mae0", "s2_txt80", "s2_multilearner", "s2_m1clean_txt80",
           "s2_none_txt80")
D2_LABEL = {**D1_LABEL, "zeroshot": "zero-shot CLIP", "s2_contrastive": "C", "s2_m1detached_txt80": "M1 detached"}
# C and zero-shot CLIP are references without a decoder: near-ink, told apart from each other and from the grey
# text-only control by marker and line style (the greys fail the validator's chroma floor by design; a legend is drawn)
D2_COL = {**D1_COL, "zeroshot": "#2b2a28", "s2_contrastive": "#2b2a28"}
D2_MARK = {**D1_MARK, "zeroshot": "*", "s2_contrastive": "v"}
D2_LS = {f: "-" for f in D2_FAMS} | {"zeroshot": ":", "s2_none_txt80": "--"}
COLLAPSE_N = 500
COLLAPSE_FAMS = D2_FAMS + ("s2_m1detached_txt80",)
D1_KINDS = ("parallel", "ratio")
D1_ALPHAS = ("0.0", "0.25", "0.5", "0.75", "1.0")

ARMS = ("c", "ml80", "ml80_mae", "parcap")
LABEL = {"c": "C", "ml80": "ML-80", "ml80_mae": "ML-80+MAE", "parcap": "Par-cap", "probe": "strongest probe"}
RUNARM = {"hb_c": "c", "hb_ml80": "ml80", "hb_ml80mae": "ml80_mae", "hb_parcap": "parcap"}
PRIMARY = {"c": "prompt_full", "ml80": "views16", "ml80_mae": "views16", "parcap": "full"}
SEEDS = ("42", "43", "44")
EMOTIONS = ("amusement", "awe", "contentment", "excitement", "anger", "disgust", "fear", "sadness", "something else")

# Colours as in the Stages 1 and 2 report: grey for the contrastive baseline, orange for the 80% masked arm, blue for
# its MAE variant, violet for the captioner, aqua for the probe. The four hues pass the dataviz validator's all-pairs
# CVD check (worst deltaE 9.2); aqua is below 3:1 contrast, so identity is also carried by axis labels and legends.
COL = {"c": "#6f6e69", "ml80": "#eb6834", "ml80_mae": "#2a78d6", "parcap": "#4a3aa7", "probe": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Diagram legend (the user's convention): grey same, orange replaced, teal new.
D_SAME, D_REPL, D_NEW = "#8f8e89", "#eb6834", "#0f8f84"


def mean(x):
    return statistics.mean(x)


def sd(x):
    return statistics.stdev(x) if len(x) > 1 else float("nan")


def ms(x, nd=3):
    return f"{mean(x):.{nd}f} ± {sd(x):.{nd}f}"


def fp(p):
    return "< 1e-4" if p == 0 else (f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}")


def style(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def git_time(rev_or_path: str, path: bool = False) -> str:
    cmd = ["git", "-C", str(REPO), "log", "--format=%h %ad %s", "--date=format-local:%Y-%m-%d %H:%M"]
    cmd += (["--", rev_or_path] if path else ["-1", rev_or_path])
    return subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "TZ": "Europe/Amsterdam"}).stdout.strip()


# --------------------------------------------------------------------------------------------------- A. runs
def load_training():
    runs = defaultdict(dict)
    for rdir in sorted(RUNS.iterdir()):
        if not (rdir / "run.json").is_file():
            continue
        rj = json.loads((rdir / "run.json").read_text())
        cfg = yaml.safe_load((rdir / "config.yaml").read_text())
        arm, seed = RUNARM[rj["name"]], int(cfg["seed"])
        epochs = []
        for line in (rdir / "metrics.jsonl").read_text().splitlines():
            m = json.loads(line)["metrics"]
            if "val/retrieval/rsum" in m:
                epochs.append({"epoch": m["epoch"], "rsum": m["val/retrieval/rsum"],
                               "emotion": m.get("val/loss_emotion"), "loss": m.get("val/loss")})
        test = {k.removeprefix("test/").removeprefix("retrieval/"): v for k, v in rj["results"]["test"].items()}
        assert seed not in runs[arm], (arm, seed)
        runs[arm][seed] = {"folder": rdir.name, "status": rj["status"], "best_epoch": rj["results"]["best_epoch"],
                           "hours": rj["duration_s"] / 3600, "host": rj["host"], "commit": rj["git"]["commit"][:7],
                           "dirty": rj["git"]["dirty"], "created": rj["created"], "ended": rj["ended"],
                           "epochs": epochs, "test": test, "cfg": cfg, "command": rj["command"]}
    return runs


def inventory(tr):
    print("## A. Run inventory\n")
    print("| arm | seed | folder | status | host | commit (dirty) | best epoch | last epoch | hours | created | ended |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        for s in sorted(tr[a]):
            r = tr[a][s]
            print(f"| {LABEL[a]} | {s} | {r['folder']} | {r['status']} | {r['host']} | {r['commit']} ({r['dirty']}) | "
                  f"{r['best_epoch']} | {r['epochs'][-1]['epoch']} | {r['hours']:.2f} | {r['created']} | {r['ended']} |")
    hrs = [r["hours"] for a in ARMS if a != "c" for r in tr[a].values()]
    print(f"Masked runs: {min(hrs):.2f} to {max(hrs):.2f} h; contrastive runs: "
          f"{min(r['hours'] for r in tr['c'].values()):.2f} to {max(r['hours'] for r in tr['c'].values()):.2f} h")
    tot = sum(r["hours"] for a in ARMS for r in tr[a].values())
    print(f"\nRuns: {sum(len(tr[a]) for a in ARMS)}; GPU-hours of training: {tot:.1f} (spec estimate about 43)")
    for a in ARMS:
        print(f"  {LABEL[a]}: mean hours {mean([r['hours'] for r in tr[a].values()]):.2f}")
    print("Arm switches (config.yaml, seed 42):")
    for a in ARMS:
        m, t, d = tr[a][42]["cfg"]["model"], tr[a][42]["cfg"]["train"], tr[a][42]["cfg"]["data"]
        print(f"  {LABEL[a]}: model {m['name']}, text_ratio {m['masking']['text_ratio']}, image_ratio "
              f"{m['masking']['image_ratio']}, mlm_image_source {m.get('mlm_image_source')}, weights {m['loss']['weights']}, "
              f"emotion_head {m.get('emotion_head')}, monitor {m['monitor']}, epochs {t['epochs']}, batch {t['batch_size']}, "
              f"lr {t['lr']}, lr_backbone {t['lr_backbone']}, warmup {t['warmup_steps']}, wd {t['weight_decay']}, "
              f"patience {t['patience']}, seeded_sampler {t['seeded_sampler']}, max_text_len {d['max_text_len']}")
    print("Spec commits (Amsterdam):", git_time(SPEC, path=True).replace("\n", " | "))
    print("Readout fix wave with the strongest-probe and D7-kill rulings:", git_time("9fd0bba"))
    print("Readout code of every encode (acc25ba):", git_time("acc25ba"))
    print("Training code of every run (6f77ad4):", git_time("6f77ad4"))
    print()


def training(tr):
    print("## B. Training curves (validation per epoch, mean over seeds) and test retrieval\n")
    curves = {}
    for a in ARMS:
        n = min(len(r["epochs"]) for r in tr[a].values())
        rs = np.array([[e["rsum"] for e in r["epochs"][:n]] for r in tr[a].values()])
        em = (np.array([[e["emotion"] for e in r["epochs"][:n]] for r in tr[a].values()]) if a != "c" else None)
        curves[a] = (np.arange(1, n + 1), rs, em)
        print(f"{LABEL[a]}: epochs common to all seeds {n}; val rsum by epoch {np.round(rs.mean(0), 2).tolist()}")
        if em is not None:
            print(f"   val emotion loss by epoch {np.round(em.mean(0), 3).tolist()}")
        best = [r["best_epoch"] for r in tr[a].values()]
        peak = [max(e["rsum"] for e in r["epochs"]) for r in tr[a].values()]
        last = [r["epochs"][-1]["rsum"] for r in tr[a].values()]
        drops = [p - q for p, q in zip(peak, last)]
        print(f"   best epochs {best}; peak val rsum {ms(peak, 2)}; last-epoch val rsum {ms(last, 2)}; "
              f"drop peak to last {mean(peak) - mean(last):.2f} (per run {[round(x, 2) for x in drops]}); "
              f"last epoch per seed {[r['epochs'][-1]['epoch'] for r in tr[a].values()]}; seeds per epoch "
              f"{[sum(len(r['epochs']) > i for r in tr[a].values()) for i in range(max(len(r['epochs']) for r in tr[a].values()))]}; "
              f"epochs drawn in Figure 2 {n}")
        if em is not None:
            at_best = [r["epochs"][r["best_epoch"] - 1]["emotion"] for r in tr[a].values()]
            at_last = [r["epochs"][-1]["emotion"] for r in tr[a].values()]
            print(f"   val emotion loss at the best epoch {ms(at_best)}; at the last epoch {ms(at_last)}; "
                  f"change {mean(at_last) - mean(at_best):+.3f}")
        if a == "c":
            vl = [[round(e["loss"], 3) for e in r["epochs"]] for r in tr[a].values()]
            print(f"   C val loss per epoch per seed {vl}")
    alld = [max(e["rsum"] for e in r["epochs"]) - r["epochs"][-1]["rsum"] for a in ARMS for r in tr[a].values()]
    print(f"Validation rsum drop from the peak to the last epoch over all 12 runs: {min(alld):.1f} to {max(alld):.1f}")
    print("\n| arm | i2t R@1 | t2i R@1 | rsum | rsum minus C (Welch p; paired-by-seed p) |")
    print("|---|---|---|---|---|")
    c = [tr["c"][s]["test"]["rsum"] for s in (42, 43, 44)]
    for a in ARMS:
        v = [tr[a][s]["test"]["rsum"] for s in (42, 43, 44)]
        i2t = [tr[a][s]["test"]["i2t_R1"] for s in (42, 43, 44)]
        t2i = [tr[a][s]["test"]["t2i_R1"] for s in (42, 43, 44)]
        cmp = "" if a == "c" else (f"{mean(v) - mean(c):+.2f} ({stats.ttest_ind(v, c, equal_var=False).pvalue:.3f}; "
                                   f"{stats.ttest_rel(v, c).pvalue:.3f})")
        print(f"| {LABEL[a]} | {ms(i2t, 2)} | {ms(t2i, 2)} | {ms(v, 2)} | {cmp} |")
        print(f"   per seed rsum {[round(x, 2) for x in v]}")
    print()
    return curves


# ------------------------------------------------------------------------------------------------ C. data
def data_section(final):
    print("## C. Data\n")
    langs, paint, total, other, english = Counter(), Counter(), 0, 0, 0
    with open(AL28_CSV, newline="", encoding="utf-8") as h:
        for r in csv.DictReader(h):
            total += 1
            langs[r["language"]] += 1
            paint[r["painting"]] += 1
            other += r["emotion"] == "other"
            english += r["language"].lower() == "english"
    print(f"AL-28 CSV: {total} votes, {len(paint)} paintings, {len(langs)} languages; 'other' {other} votes "
          f"({other / total:.1%}); English votes {english}")
    enc = analysis.load_encoded(ENC / final["runs"][0]["folder"])
    counts = enc["al28_counts"]
    nv = counts.sum(1)
    print(f"Target set: {len(counts)} paintings with >= 20 non-English votes; votes per painting median {np.median(nv):.0f}, "
          f"min {nv.min()}, max {nv.max()}, total {nv.sum()}")
    h = metrics.normalise(counts)
    he = metrics.entropy_bits(h)
    print(f"Human entropy (bits): mean {he.mean():.3f}, median {np.median(he):.3f}; max possible log2(9) = {np.log2(9):.3f}; "
          f"third cuts {np.round(final['settings']['entropy_third_cuts_bits'], 3).tolist()}")
    tc = enc["train_counts"]
    prior = tc.sum(0) / tc.sum()
    print(f"Train (hold-out applied): {len(tc)} paintings, {tc.sum()} labels; prior (train label frequencies): "
          + ", ".join(f"{e} {p:.3f}" for e, p in sorted(zip(EMOTIONS, prior), key=lambda x: -x[1])))
    hm = h.mean(0)
    print("Mean AL-28 non-English target: " + ", ".join(f"{e} {p:.3f}" for e, p in sorted(zip(EMOTIONS, hm), key=lambda x: -x[1])))
    print(f"Validation for the temperature fit: {len(enc['val_emb_full'])} paintings, {len(enc['val_labels'])} labels")
    print(f"D7 test captions: {len(enc['d7_label'])}; length grid {enc['meta']['lengths']}")
    print()
    return he


def gate():
    print("## D. Gate (seed 42 of each decoder arm, before wave 2)\n")
    for g in json.loads(GATE.read_text()):
        print(f"{g['arm']} s{g['seed']} {g['readout']}: T {g['temperature']:.3f}, val NLL {g['val_nll']:.3f} vs prior "
              f"{g['prior_nll']:.3f}, JSD to prior {g['jsd_to_prior']:.3f}, JSD to human {g['jsd_human']:.3f} "
              f"(prior {g['prior_jsd_human']:.3f}), entropy rho {g['entropy_spearman']:.3f}, checks {g['checks']}, "
              f"passed {g['passed']}")
    print()


# --------------------------------------------------------------------------------------------------- E. D4
def per_seed(final, arm, readout, key, third=None):
    out = []
    for s in SEEDS:
        m = final["arms"][arm][s]["readouts"][readout]["metrics"]
        out.append(m[key] if third is None else m["thirds"][third][key])
    return out


def probe_seed(final, key, third=None, readout="probe_full"):
    best = final["probes"]["strongest"]
    out = []
    for s in SEEDS:
        m = final["arms"][best[s]["arm"]][s]["readouts"][readout]["metrics"]
        out.append(m[key] if third is None else m["thirds"][third][key])
    return out


def d4(final):
    print("## E. D4: decision, primary readouts, baselines, references\n")
    dec = final["decision"]
    print(f"Rule: {dec['rule']}. Support {dec['support']}; kill {dec['kill']}; provisional {dec['provisional']}")
    print("| metric | comparator | diff (ML-80 minus) | 95% interval | p | Holm p | favours ML-80 |")
    print("|---|---|---|---|---|---|---|")
    for k in ("jsd/parcap", "jsd/probe", "entropy_spearman/parcap", "entropy_spearman/probe"):
        t = final["tests"][k]
        print(f"| {k.split('/')[0]} | {k.split('/')[1]} | {t['diff']:+.4f} | [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] | "
              f"{fp(t['p'])} | {fp(t['holm_p'])} | {t['favours']} |")
    for k, t in final["tests"]["secondary"].items():
        print(f"secondary {k}: {t['diff']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}], p {t['p']:.4f}")
    print("B =", final["settings"]["B"], "; seeds", final["settings"]["seeds"])

    # Seed-level check of the four pre-registered comparisons (final review I1; not pre-registered): with 3 seeds per
    # arm the hierarchical bootstrap draws among only 10 seed multisets, so its tails come from painting resampling.
    print("\nSeed-level check of the four tests (Welch t-test on the per-seed metrics, 3 against 3; Holm over the four):")
    ml = {k: per_seed(final, "ml80", "views16", k) for k in ("jsd", "entropy_spearman")}
    comp = {"parcap": {k: per_seed(final, "parcap", "full", k) for k in ml}, "probe": {k: probe_seed(final, k) for k in ml}}
    keys = [("jsd", "parcap"), ("jsd", "probe"), ("entropy_spearman", "parcap"), ("entropy_spearman", "probe")]
    wp = [float(stats.ttest_ind(ml[m], comp[c][m], equal_var=False).pvalue) for m, c in keys]
    for (m, c), p, h in zip(keys, wp, bootstrap.holm(wp)):
        print(f"  {m} vs {c}: ML-80 per seed {[round(x, 4) for x in ml[m]]}, {c} {[round(x, 4) for x in comp[c][m]]}; "
              f"diff {mean(ml[m]) - mean(comp[c][m]):+.4f}; Welch p {p:.3f}; Holm p {h:.3f}")
    print(f"  ML-80's best seed entropy Spearman {max(ml['entropy_spearman']):.3f} against Par-cap's worst "
          f"{min(comp['parcap']['entropy_spearman']):.3f} and the probe's worst {min(comp['probe']['entropy_spearman']):.3f}; "
          f"ML-80's best seed JSD {min(ml['jsd']):.4f} against Par-cap's worst {max(comp['parcap']['jsd']):.4f}")
    # ML-80 on the full image against Par-cap (final review M7)
    for k in ("jsd", "entropy_spearman"):
        a_, b_ = per_seed(final, "ml80", "full", k), per_seed(final, "parcap", "full", k)
        print(f"  ML-80 full image vs Par-cap, {k}: {[round(x, 4) for x in a_]} against {[round(x, 4) for x in b_]}; "
              f"diff {mean(a_) - mean(b_):+.4f}; Welch p {stats.ttest_ind(a_, b_, equal_var=False).pvalue:.2f}")
    # the prior: raw in the gate, temperature-fitted like every other baseline in Table 6 (final review M8)
    g = json.loads(GATE.read_text())[0]
    pr_ = final["arms"]["ml80"]["42"]["readouts"]["prior"]
    print(f"Prior-only: raw (the gate) val NLL {g['prior_nll']:.4f}, JSD to the human votes {g['prior_jsd_human']:.4f}; "
          f"with its own fitted T {pr_['T']:.3f} (Table 6) val NLL {pr_['val_nll']:.4f}, JSD {pr_['metrics']['jsd']:.4f}")
    print("\nStrongest probe per seed:", {s: (v["arm"], round(v["val_nll"], 4), round(v["val_nll_t1"], 4))
                                         for s, v in final["probes"]["strongest"].items()})
    print("Probe per arm and seed (weight decay, T=1 val NLL):",
          {a: {s: (v["weight_decay"], round(v["val_nll"], 4)) for s, v in by.items()} for a, by in final["probes"]["per_arm"].items()})
    print("Probe JSD / rho per arm and seed:",
          {a: {s: (round(final["arms"][a][s]["readouts"]["probe_full"]["metrics"]["jsd"], 4),
                   round(final["arms"][a][s]["readouts"]["probe_full"]["metrics"]["entropy_spearman"], 4)) for s in SEEDS} for a in ARMS})
    print("Post-temperature probe val NLL per arm and seed:",
          {a: {s: round(final["arms"][a][s]["readouts"]["probe_full"]["val_nll"], 4) for s in SEEDS} for a in ARMS})

    print("\n| arm | readout | JSD | entropy rho | KL(h||m) | TVD | RankCS | val NLL | T |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = [(a, PRIMARY[a]) for a in ("ml80", "ml80_mae", "parcap", "c")] + [("ml80", "full"), ("ml80_mae", "full")]
    rows += [(a, "probe_full") for a in ARMS] + [(a, "prompt_full") for a in ("ml80", "ml80_mae", "parcap")] + [("ml80", "prior")]
    for a, r in rows:
        cells = []
        for k in ("jsd", "entropy_spearman", "kl", "tvd", "rank_cs"):
            v = per_seed(final, a, r, k)
            cells.append("n/a" if v[0] is None else ms(v))
        nll = [final["arms"][a][s]["readouts"][r]["val_nll"] for s in SEEDS]
        T = [final["arms"][a][s]["readouts"][r]["T"] for s in SEEDS]
        print(f"| {LABEL[a]} | {r} | " + " | ".join(cells) + f" | {ms(nll, 4)} | {ms(T, 2)} |")
    pj = probe_seed(final, "jsd")
    pr = probe_seed(final, "entropy_spearman")
    pk = probe_seed(final, "kl")
    pt = probe_seed(final, "tvd")
    pc = probe_seed(final, "rank_cs")
    pn = [final["probes"]["strongest"][s]["val_nll"] for s in SEEDS]
    print(f"| strongest probe | probe_full | {ms(pj)} | {ms(pr)} | {ms(pk)} | {ms(pt)} | {ms(pc)} | {ms(pn, 4)} | |")
    print("strongest probe per seed JSD", [round(x, 4) for x in pj], "rho", [round(x, 4) for x in pr])
    print("Any fitted temperature at a bound:", any(v["t_at_bound"] for a in ARMS for s in SEEDS
                                                    for v in final["arms"][a][s]["readouts"].values()))

    ref = final["references"]
    sh, en = ref["split_half"], ref["english"]
    print(f"\nSplit-half ceiling ({sh['n']} paintings, {sh['splits']} splits): JSD {sh['jsd']:.3f}, rho {sh['entropy_spearman']:.3f}")
    print(f"English reference ({en['n']} paintings, {en['n_dropped_without_english']} without English labels): JSD "
          f"{en['jsd']:.3f}, rho {en['entropy_spearman']:.3f}, TVD {en['tvd']:.3f}, RankCS {en['rank_cs']:.3f}")
    prior_j = mean(per_seed(final, "ml80", "prior", "jsd"))
    print("Share of the prior-to-ceiling JSD gap closed, and rho as a share of the ceiling's:")
    for name, j, r in (("ML-80 views16", mean(per_seed(final, "ml80", "views16", "jsd")), mean(per_seed(final, "ml80", "views16", "entropy_spearman"))),
                       ("ML-80 full", mean(per_seed(final, "ml80", "full", "jsd")), mean(per_seed(final, "ml80", "full", "entropy_spearman"))),
                       ("Par-cap full", mean(per_seed(final, "parcap", "full", "jsd")), mean(per_seed(final, "parcap", "full", "entropy_spearman"))),
                       ("strongest probe", mean(pj), mean(pr)),
                       ("C prompt", mean(per_seed(final, "c", "prompt_full", "jsd")), mean(per_seed(final, "c", "prompt_full", "entropy_spearman")))):
        print(f"  {name}: JSD {j:.4f}, gap closed {(prior_j - j) / (prior_j - sh['jsd']):.0%}; rho {r:.3f} = "
              f"{r / sh['entropy_spearman']:.0%} of the ceiling")
    for a, r in (("ml80", "views16"), ("ml80", "full"), ("parcap", "full")):
        dj = mean(per_seed(final, a, r, "jsd")) - mean(pj)
        dr = mean(per_seed(final, a, r, "entropy_spearman")) - mean(pr)
        print(f"  {LABEL[a]} {r} minus strongest probe (descriptive): JSD {dj:+.4f}, rho {dr:+.3f}")
    dj = mean(per_seed(final, "ml80", "views16", "jsd")) - mean(per_seed(final, "c", "prompt_full", "jsd"))
    dr = mean(per_seed(final, "ml80", "views16", "entropy_spearman")) - mean(per_seed(final, "c", "prompt_full", "entropy_spearman"))
    print(f"  ML-80 views16 minus C prompt (descriptive): JSD {dj:+.4f}, rho {dr:+.3f}")
    dj = mean(per_seed(final, "ml80", "full", "jsd")) - mean(per_seed(final, "ml80", "views16", "jsd"))
    dr = mean(per_seed(final, "ml80", "full", "entropy_spearman")) - mean(per_seed(final, "ml80", "views16", "entropy_spearman"))
    print(f"  ML-80 full minus ML-80 views16: JSD {dj:+.4f}, rho {dr:+.3f}")
    print(f"  prior JSD {prior_j:.4f}; C prompt minus prior JSD {mean(per_seed(final, 'c', 'prompt_full', 'jsd')) - prior_j:+.4f}")

    print("\nVariants (mean ± SD): other dropped, 8 named emotions")
    for a in ("ml80", "ml80_mae", "parcap", "c"):
        r = PRIMARY[a]
        od = [final["arms"][a][s]["readouts"][r]["other_dropped"] for s in SEEDS]
        n8 = [final["arms"][a][s]["readouts"][r]["named8"] for s in SEEDS]
        print(f"  {LABEL[a]} {r}: other dropped JSD {ms([x['jsd'] for x in od])}, rho {ms([x['entropy_spearman'] for x in od])}; "
              f"8-named JSD {ms([x['jsd'] for x in n8])}, rho {ms([x['entropy_spearman'] for x in n8])} (n {n8[0]['n']})")
    for name, key in (("other dropped", "other_dropped"), ("8-named", "named8")):
        best = final["probes"]["strongest"]
        v = [final["arms"][best[s]["arm"]][s]["readouts"]["probe_full"][key] for s in SEEDS]
        print(f"  strongest probe {name}: JSD {ms([x['jsd'] for x in v])}, rho {ms([x['entropy_spearman'] for x in v])}")
    for key in ("other_dropped", "named8"):
        for m in ("jsd", "entropy_spearman"):
            ml = mean([final["arms"]["ml80"][s]["readouts"]["views16"][key][m] for s in SEEDS])
            pc = mean([final["arms"]["parcap"][s]["readouts"]["full"][key][m] for s in SEEDS])
            print(f"  {key} {m}: ML-80 minus Par-cap {ml - pc:+.4f}")

    print("\nBy thirds of human entropy (n per third):",
          [t["n"] for t in final["arms"]["ml80"]["42"]["readouts"]["views16"]["metrics"]["thirds"]])
    thirds = {}
    for a in ("ml80", "ml80_mae", "parcap", "c", "probe", "prior"):
        row = []
        for t in range(3):
            if a == "probe":
                j, r = probe_seed(final, "jsd", t), probe_seed(final, "entropy_spearman", t)
            elif a == "prior":
                j, r = per_seed(final, "ml80", "prior", "jsd", t), None
            else:
                j, r = per_seed(final, a, PRIMARY[a], "jsd", t), per_seed(final, a, PRIMARY[a], "entropy_spearman", t)
            row.append((j, r))
        thirds[a] = row
        print(f"  {LABEL.get(a, a)}: " + " | ".join(f"JSD {ms(j)} rho {'n/a' if r is None else ms(r)}" for j, r in row))
    for t, name in enumerate(("low", "middle", "high")):
        print(f"  {name} third: ML-80 minus Par-cap JSD {mean(thirds['ml80'][t][0]) - mean(thirds['parcap'][t][0]):+.4f}, "
              f"rho {mean(thirds['ml80'][t][1]) - mean(thirds['parcap'][t][1]):+.3f}; ML-80 minus prior JSD "
              f"{mean(thirds['ml80'][t][0]) - mean(thirds['prior'][t][0]):+.4f}; Par-cap minus prior JSD "
              f"{mean(thirds['parcap'][t][0]) - mean(thirds['prior'][t][0]):+.4f}")
    prov = json.loads(PROV.read_text())
    print(f"\nProvisional look (9 runs, seeds {prov['settings']['seeds']}): support {prov['decision']['support']}, "
          f"favours {prov['decision']['favours']}, Holm {prov['decision']['holm']}")
    print()
    return thirds


def spread(final, he):
    """Re-derives the primary JSD of every run from the encode files (a check of this script's reading) and
    measures how far each readout moves from the prior and how peaked it is."""
    print("## F. Spread of the readouts (from the encode files, with each run's fitted T)\n")
    out = defaultdict(lambda: defaultdict(list))
    folders = {(r["arm"], str(r["seed"])): r["folder"] for r in final["runs"]}
    best = final["probes"]["strongest"]
    max_dev = 0.0
    for a in ARMS:
        for s in SEEDS:
            enc = analysis.load_encoded(ENC / folders[(a, s)])
            h = metrics.normalise(enc["al28_counts"])
            prior = enc["train_counts"].sum(0) / enc["train_counts"].sum()
            dec = analysis.decoder_readouts(enc)
            items = {}
            for r in ("views16", "full"):
                if r in dec:
                    items[r] = dec[r][0]
            items["prompt_full"] = baselines.prompt_logits(enc["al28_emb_full"], enc["prompts"], float(enc["logit_scale"]))
            if best[s]["arm"] == a:
                wd = final["probes"]["per_arm"][a][s]["weight_decay"]
                probe = baselines.fit_soft_probe(enc["train_emb"], enc["train_counts"], enc["val_emb_full"],
                                                 enc["val_index"], enc["val_labels"], weight_decays=(wd,))
                items["probe_full"] = probe.logits(enc["al28_emb_full"])
            for r, logits in items.items():
                T = final["arms"][a][s]["readouts"][r]["T"]
                p = calibrate.mixture(logits, T)
                jsd = float(metrics.js_distance(p, h).mean())
                ref = final["arms"][a][s]["readouts"][r]["metrics"]["jsd"]
                max_dev = max(max_dev, abs(jsd - ref))
                tag = "probe" if r == "probe_full" else f"{a}:{r}"
                out[tag]["entropy"].append(float(metrics.entropy_bits(p).mean()))
                out[tag]["to_prior"].append(float(metrics.js_distance(p, np.tile(prior, (len(p), 1))).mean()))
                out[tag]["jsd_check"].append(jsd)
    print(f"Largest |re-derived JSD minus hb_d4.json JSD| over every readout used: {max_dev:.2e}")
    print(f"Human: mean entropy {he.mean():.3f} bits; JSD of the human targets to the prior "
          f"{float(metrics.js_distance(h, np.tile(prior, (len(h), 1))).mean()):.3f}")
    print("| readout | mean entropy (bits) | mean JSD to the prior |")
    print("|---|---|---|")
    for tag in ("ml80:views16", "ml80:full", "ml80_mae:views16", "parcap:full", "probe", "c:prompt_full"):
        print(f"| {tag} | {ms(out[tag]['entropy'])} | {ms(out[tag]['to_prior'])} |")
    print()


# --------------------------------------------------------------------------------------------------- G. D6
def d6(final):
    print("## G. D6: view sampling\n")
    d = final["d6"]
    curves = {}
    for a in ("ml80", "ml80_mae"):
        r = d["arms"][a]
        curves[a] = {k: ([r[k]["per_seed"][s]["jsd"] for s in SEEDS], [r[k]["per_seed"][s]["entropy_spearman"] for s in SEEDS])
                     for k in ("views1", "views4", "views16", "full")}
        print(f"{LABEL[a]}: " + "; ".join(f"{k} JSD {ms(v[0])} rho {ms(v[1])}" for k, v in curves[a].items())
              + f"; K16 beats K1 {r['k16_beats_k1']}")
        mi = r["mi"]
        print(f"   MI vs human entropy Spearman: mean {mi['spearman_mean']:.3f} [{mi['ci_low']:.3f}, {mi['ci_high']:.3f}], "
              f"per seed {[round(v, 3) for v in mi['spearman_per_seed'].values()]}, includes zero {mi['interval_includes_zero']}, "
              f"NaN draws {mi['nan_draws']}")
        k1, k16, full = (mean(curves[a][k][0]) for k in ("views1", "views16", "full"))
        print(f"   JSD K1 -> K16 {k16 - k1:+.4f}; K16 -> full {full - k16:+.4f}")
    curves["parcap"] = {k: (per_seed(final, "parcap", k, "jsd"), per_seed(final, "parcap", k, "entropy_spearman"))
                        for k in ("views1", "views4", "views16", "full")}
    print("Par-cap: " + "; ".join(f"{k} JSD {ms(v[0])} rho {ms(v[1])}" for k, v in curves["parcap"].items()))
    print("Controls on the same views (means):", {k: (round(v["jsd"], 3), round(v["entropy_spearman"], 3)) for k, v in d["controls"].items()})
    for a in ("ml80", "ml80_mae", "parcap", "c"):
        print(f"   probe of {LABEL[a]}'s embedding on 16 views: JSD {ms(per_seed(final, a, 'probe_views16', 'jsd'))}, "
              f"rho {ms(per_seed(final, a, 'probe_views16', 'entropy_spearman'))}")
    d6_controls(final)
    print()
    return curves


def _rank_rows(x: np.ndarray) -> np.ndarray:
    r = stats.rankdata(x, axis=-1)
    return r - r.mean(-1, keepdims=True)


def _corr_rows(rx: np.ndarray, ry: np.ndarray) -> np.ndarray:
    return (rx * ry).sum(-1) / np.sqrt((rx ** 2).sum(-1) * (ry ** 2).sum(-1))


def partial_spearman(x, y, z):
    """Spearman correlation of x and y controlling for z (Pearson partial correlation of the ranks), row-wise."""
    rx, ry, rz = _rank_rows(x), _rank_rows(y), _rank_rows(z)
    xy, xz, yz = _corr_rows(rx, ry), _corr_rows(rx, rz), _corr_rows(ry, rz)
    return (xy - xz * yz) / np.sqrt((1 - xz ** 2) * (1 - yz ** 2))


def d6_controls(final, B: int = 10_000):
    """Final review I3: the between-view MI of Par-cap on the same 16 views (the control Table 11 lacked), and the
    MI-vs-human-entropy correlation with the readout's own entropy held fixed. Same bootstrap scheme as Table 11:
    each replicate resamples the paintings (shared by the seeds) and the seeds (seed 0 for the raw Spearman, as
    tables.boot_spearman; seed 1 for the partial)."""
    folders = {(r["arm"], str(r["seed"])): r["folder"] for r in final["runs"]}
    print("\nD6 controls (final review I3): MI = entropy of the 16-view mean minus the mean per-view entropy, each arm "
          "with its own views16 temperature; readout entropy = entropy of the 16-view mean (bits)")
    print("| arm | Spearman(MI, human H) per seed, mean [95%] | Spearman(MI, readout H) | Spearman(readout H, human H) | "
          "partial Spearman(MI, human H given readout H) per seed, mean [95%] | mean within-quintile Spearman(MI, human H) |")
    print("|---|---|---|---|---|---|")
    he = None
    for a in ("ml80", "ml80_mae", "parcap"):
        mis, ents, rows = [], [], defaultdict(list)
        for s in SEEDS:
            enc = analysis.load_encoded(ENC / folders[(a, s)])
            if he is None:
                he = metrics.entropy_bits(metrics.normalise(enc["al28_counts"]))
            T = final["arms"][a][s]["readouts"]["views16"]["T"]
            mi = tables.between_view_mi(enc, T)
            e16 = metrics.entropy_bits(calibrate.mixture(analysis.decoder_readouts(enc)["views16"][0], T))
            mis.append(mi)
            ents.append(e16)
            rows["mi_he"].append(bootstrap._spearman_rows(mi, he))
            rows["mi_e"].append(bootstrap._spearman_rows(mi, e16))
            rows["e_he"].append(bootstrap._spearman_rows(e16, he))
            rows["partial"].append(float(partial_spearman(mi, he, e16)))
            q = np.digitize(e16, np.quantile(e16, [0.2, 0.4, 0.6, 0.8]))
            rows["within"].append(mean([bootstrap._spearman_rows(mi[q == k], he[q == k]) for k in range(5)]))
            ref = final["arms"][a][s]["readouts"]["views16"]["metrics"]["entropy_spearman"]
            assert abs(rows["e_he"][-1] - ref) < 1e-9, (a, s, rows["e_he"][-1], ref)  # = D4's entropy Spearman
            if a in final["d6"]["arms"]:
                assert abs(rows["mi_he"][-1] - final["d6"]["arms"][a]["mi"]["spearman_per_seed"][s]) < 1e-9
        raw = tables.boot_spearman(mis, he, B)  # the Table 11 bootstrap (seed 0)
        rlo, rhi = np.nanquantile(raw, [0.025, 0.975])
        rng, n, S = np.random.default_rng(1), len(he), len(mis)
        draws = []
        for start in range(0, B, 250):
            c = min(250, B - start)
            idx, drawn = rng.integers(0, n, (c, n)), rng.integers(0, S, (c, S))
            per = np.stack([partial_spearman(m[idx], he[idx], e[idx]) for m, e in zip(mis, ents)], 1)  # (c, S)
            draws.append(np.take_along_axis(per, drawn, 1).mean(1))
        plo, phi = np.nanquantile(np.concatenate(draws), [0.025, 0.975])
        print(f"| {LABEL[a]} | {[round(x, 3) for x in rows['mi_he']]}, {mean(rows['mi_he']):.3f} [{rlo:.3f}, {rhi:.3f}] | "
              f"{[round(x, 3) for x in rows['mi_e']]} | {[round(x, 3) for x in rows['e_he']]} | "
              f"{[round(x, 3) for x in rows['partial']]}, {mean(rows['partial']):.3f} [{plo:.3f}, {phi:.3f}] | "
              f"{[round(x, 3) for x in rows['within']]}, {mean(rows['within']):.3f} |")


# --------------------------------------------------------------------------------------------------- H. D7
def d7(final):
    print("## H. D7: partial captions\n")
    d = final["d7"]
    print("Kill verdict for ML-80:", d["kill"])
    for a in ("ml80", "ml80_mae", "parcap"):
        A = d["arms"][a]
        print(f"\n{LABEL[a]} (readout {A['readout']}, seeds {A['seeds']})")
        print("| pattern | n | log-loss real | log-loss null | acc real | acc null | real minus null [95%] |")
        print("|---|---|---|---|---|---|---|")
        for p in d["patterns"]:
            r = A["patterns"][p]
            dd = r["real_minus_null"]
            print(f"| {p} | {r['n']} | {r['real']['log_loss']:.3f} | {r['null']['log_loss']:.3f} | {r['real']['accuracy']:.3f} | "
                  f"{r['null']['accuracy']:.3f} | {dd['diff']:+.3f} [{dd['ci_low']:+.3f}, {dd['ci_high']:+.3f}] |")
    ml, pc = d["arms"]["ml80"]["patterns"], d["arms"]["parcap"]["patterns"]
    for p in ("j0", "prefix1", "random1", "prefix8", "random8"):
        print(f"ML-80 minus Par-cap log-loss at {p}: {ml[p]['real']['log_loss'] - pc[p]['real']['log_loss']:+.3f}; "
              f"accuracy {ml[p]['real']['accuracy'] - pc[p]['real']['accuracy']:+.3f}")
    print(f"ML-80 log-loss drop j0 -> prefix8 {ml['prefix8']['real']['log_loss'] - ml['j0']['real']['log_loss']:+.3f}, "
          f"-> random8 {ml['random8']['real']['log_loss'] - ml['j0']['real']['log_loss']:+.3f}")
    print(f"Uniform guess over 9 classes: ln 9 = {np.log(9):.3f} nats; null-image log-loss at j0: ML-80 "
          f"{ml['j0']['null']['log_loss']:.3f}, ML-80+MAE {d['arms']['ml80_mae']['patterns']['j0']['null']['log_loss']:.3f}, "
          f"Par-cap {pc['j0']['null']['log_loss']:.3f}")
    spread_pc = [pc[p]["real"]["log_loss"] for p in d["patterns"]]
    print(f"Par-cap real log-loss range over patterns: {min(spread_pc):.4f} to {max(spread_pc):.4f}")

    # Final review I2: the prior-only log-loss on the same test captions (train label frequencies; raw, and with the
    # prior's own temperature from Table 6), beside the real and the null image.
    folders = {(r["arm"], str(r["seed"])): r["folder"] for r in final["runs"]}
    enc = analysis.load_encoded(ENC / folders[("ml80", "42")])
    for a in ("ml80_mae", "parcap"):  # every arm scores the same captions
        e2 = analysis.load_encoded(ENC / folders[(a, "42")])
        assert np.array_equal(e2["d7_label"], enc["d7_label"]) and np.array_equal(e2["d7_valid"], enc["d7_valid"])
    prior = enc["train_counts"].sum(0) / enc["train_counts"].sum()
    Tp = final["arms"]["ml80"]["42"]["readouts"]["prior"]["T"]
    prior_t = calibrate.mixture(np.log(prior)[None, None, :], Tp)[0]
    valid, label = enc["d7_valid"].astype(bool), enc["d7_label"]
    pri = {}
    print(f"\nPrior-only log-loss on the D7 test captions (raw; with the prior's fitted T {Tp:.3f}), and each arm's real and "
          f"null image minus the raw prior (nats):")
    print("| pattern | n | prior | prior, fitted T | ML-80 real minus prior | ML-80 null minus prior | ML-80+MAE null minus prior | "
          "Par-cap real minus prior | Par-cap null minus prior |")
    print("|---|---|---|---|---|---|---|---|---|")
    mae = d["arms"]["ml80_mae"]["patterns"]
    for k, p in enumerate(d["patterns"]):
        y = label[valid[:, k]]
        assert len(y) == ml[p]["n"]
        pri[p] = float(-np.log(prior[y]).mean())
        print(f"| {p} | {len(y)} | {pri[p]:.3f} | {-np.log(prior_t[y]).mean():.3f} | {ml[p]['real']['log_loss'] - pri[p]:+.3f} | "
              f"{ml[p]['null']['log_loss'] - pri[p]:+.3f} | {mae[p]['null']['log_loss'] - pri[p]:+.3f} | "
              f"{pc[p]['real']['log_loss'] - pri[p]:+.3f} | {pc[p]['null']['log_loss'] - pri[p]:+.3f} |")
    for a, A in (("ml80", ml), ("ml80_mae", mae), ("parcap", pc)):
        r0, n0 = A["j0"]["real"]["log_loss"], A["j0"]["null"]["log_loss"]
        print(f"  {LABEL[a]} at j = 0: real {r0:.3f} ({r0 - pri['j0']:+.3f} vs the prior {pri['j0']:.3f}), null {n0:.3f} "
              f"({n0 - pri['j0']:+.3f}); real minus null {r0 - n0:+.3f}, of which the null's penalty over the prior is "
              f"{(n0 - pri['j0']) / (n0 - r0):.0%}")
    gaps = [ml[p]["real_minus_null"]["diff"] for p in tables.KILL_PATTERNS]
    print(f"  ML-80 real minus null at the six j >= 2 patterns: {min(gaps):+.3f} to {max(gaps):+.3f}; ML-80's null penalty "
          f"over the prior at j = 0: {ml['j0']['null']['log_loss'] - pri['j0']:+.3f}; every j >= 2 gap smaller in size: "
          f"{all(abs(x) < ml['j0']['null']['log_loss'] - pri['j0'] for x in gaps)}")
    print()
    return pri


# --------------------------------------------------------------------------------------------------- I. D1
def random_and_oracle(cand: np.ndarray, q) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per query: expected AP@R (percent) of a uniformly random order of the top k, the oracle AP@R (every positive
    in the top k first), and the number of positives in the top k. Positives outside the top k count as misses for
    every order, as in the re-rankers (R <= k for every ECCV i2t query)."""
    n = cand.shape[1]
    m = np.array([np.isin(cand[i], q.positives[i]).sum() for i in range(len(cand))])
    rnd = np.zeros(len(m))
    for j, (mi, R) in enumerate(zip(m, q.R)):
        assert R <= n
        i = np.arange(1, R + 1)  # E[rel_i * prec@i] = (P(rel_i) + sum_{l<i} P(rel_i, rel_l)) / i
        rnd[j] = ((mi / n + (i - 1) * mi * (mi - 1) / (n * (n - 1))) / i).sum() / R
    return 100 * rnd, 100 * np.minimum(m, q.R) / q.R, m


def d1():
    print("## I. D1: decoder re-ranking of the dual top 50 on ECCV Caption i2t\n")
    cp = coco_probes
    tab = json.loads(D1_TABLE.read_text())["d1"]
    fams = tab["families"]
    assert set(fams) == set(D1_FAMS), sorted(fams)
    cks = sorted(tab["checkpoints"], key=lambda c: (D1_FAMS.index(c["arm"]), c["seed"]))
    q = cp.eccv_i2t(COCO_ANN)
    thirds = cp.r_thirds(q.R)
    print(f"ECCV i2t queries {len(q.R)}; R from {q.R.min()} to {q.R.max()}; queries per third of R: "
          + ", ".join(f"{k} {len(v)}" for k, v in thirds.items()))

    # --- inventory, timing, end-to-end checks
    print("\n| family | seed | run | source | max abs(recomputed - d1.json) tuned (a)/(b) | tuned parallel (alpha_a; alpha_b, beta) | "
          "tuned ratio | minutes | dual i2t mAP@R | run.json i2t mAP@R | test MLM loss | ckpt commit |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    per = {}
    mins, mtimes, metas, e2e = [], [], [], []
    for c in cks:
        f, run = c["arm"], c["run"]
        d1j = json.loads((D1_DIR / run / "d1.json").read_text())
        rj = json.loads((COCO_RUNS / run / "run.json").read_text())
        z = dict(np.load(D1_DIR / run / "scores.npz"))
        assert np.array_equal(z["R"], q.R) and np.array_equal(z["query_image"], q.query_image)
        meta = d1j["meta"]
        metas.append((meta["k"], meta["samples"], meta["val_images"], meta["queries"]))
        mins.append(meta["seconds"] / 60)
        mtimes.append((D1_DIR / run / "d1.json").stat().st_mtime)
        dual = c["metrics"]["dual"]["all"]["map"]
        e2e.append(abs(dual - rj["results"]["test"]["test/eccv/i2t_map_at_r"]))
        t = c["tuned"]
        print(f"| {D1_LABEL[f]} | {c['seed']} | {run} | {c['source']} | {c['check_vs_json']:.4f} | "
              f"{t['parallel']['alpha_a']}; {t['parallel']['alpha_b']}, {t['parallel']['beta']} | "
              f"{t['ratio']['alpha_a']}; {t['ratio']['alpha_b']}, {t['ratio']['beta']} | {mins[-1]:.1f} | {dual:.2f} | "
              f"{rj['results']['test']['test/eccv/i2t_map_at_r']:.2f} | {rj['results']['test']['test/loss_mlm']:.3f} | "
              f"{rj['git']['commit'][:7]} |")
        per.setdefault(f, []).append({"c": c, "z": z, "mlm": rj["results"]["test"]["test/loss_mlm"],
                                      "eccv_mean": rj["results"]["test"]["test/eccv/map_at_r"],
                                      "eccv_t2i": rj["results"]["test"]["test/eccv/t2i_map_at_r"]})
    print(f"(k, samples, val images, queries) for every checkpoint: {sorted(set(metas))}")
    import datetime as _dt
    import zoneinfo as _zi
    ams = _zi.ZoneInfo("Europe/Amsterdam")
    first_end = min(mtimes)
    start = _dt.datetime.fromtimestamp(first_end - mins[[i for i, m in enumerate(mtimes) if m == first_end][0]] * 60, ams)
    print(f"Minutes per checkpoint {min(mins):.1f} to {max(mins):.1f}; total {sum(mins) / 60:.2f} h on one GPU; first start "
          f"{start:%Y-%m-%d %H:%M}, last d1.json written {_dt.datetime.fromtimestamp(max(mtimes), ams):%Y-%m-%d %H:%M}")
    print(f"Largest |dual i2t mAP@R (recomputed) - the checkpoint's own run.json i2t mAP@R| = {max(e2e):.4f}")
    img = [c for c in cks if c["arm"] != "s2_none_txt80"]
    print(f"Largest |recomputed (snapped) - d1.json| tuned re-ranker over image-reading checkpoints "
          f"{max(c['check_vs_json'] for c in img):.4f}; text-only control "
          f"{[round(c['check_vs_json'], 3) for c in cks if c['arm'] == 's2_none_txt80']}")
    for f in D1_FAMS:
        print(f"  {D1_LABEL[f]}: test MLM loss {ms([p['mlm'] for p in per[f]])}; COCO-line ECCV mAP@R (i2t and t2i mean) "
              f"{ms([p['eccv_mean'] for p in per[f]], 2)}, t2i {ms([p['eccv_t2i'] for p in per[f]], 2)}")

    def st(f, key):
        return fams[f]["stats"][key]

    def msd(s, nd=2):
        return f"{s['mean']:.{nd}f} ± {s['sd']:.{nd}f}"

    # --- kill rule
    print("\nKill rule for re-ranker (a) (hb_coco_table.py): KILL if for every alpha the mean paired difference "
          "(a minus dual, overall mAP@R) is below minus its seed SD, and no alpha gives a positive mean paired difference "
          "in the top R third (R >= 21)")
    print("| family | score | verdict | diff by alpha (mean ± SD) | top third diff by alpha (mean ± SD) | best alpha | "
          "best diff | best top-third diff |")
    print("|---|---|---|---|---|---|---|---|")
    curves = {}
    for f in D1_FAMS:
        for kind in D1_KINDS:
            kr = fams[f]["kill_rule_a"][kind]
            allv = [st(f, f"{kind}/grid_a/{a}/all/map_diff") for a in D1_ALPHAS]
            topv = [st(f, f"{kind}/grid_a/{a}/R>=21/map_diff") for a in D1_ALPHAS]
            curves[(f, kind)] = (allv, topv)
            best = int(np.argmax([s["mean"] for s in allv]))
            print(f"| {D1_LABEL[f]} | {kind} | {kr['verdict']} | " + "; ".join(f"{a}: {msd(s)}" for a, s in zip(D1_ALPHAS, allv))
                  + " | " + "; ".join(f"{a}: {msd(s)}" for a, s in zip(D1_ALPHAS, topv))
                  + f" | {D1_ALPHAS[best]} | {msd(allv[best])} | {max(s['mean'] for s in topv):+.2f} |")
            print(f"   change of the overall difference from alpha 0 to 1 {allv[-1]['mean'] - allv[0]['mean']:+.2f}, "
                  f"from 0.75 to 1 {allv[-1]['mean'] - allv[-2]['mean']:+.2f}")

    # --- the four orders by third of R
    print("\nmAP@R (percent, mean ± seed SD) by order and third of R; last columns: R-Precision, paired diff vs dual (all)")
    for kind in D1_KINDS:
        print(f"\n{kind} score")
        print("| family | order | all | R<=15 | R16-20 | R>=21 | R-Prec all | minus dual (all) |")
        print("|---|---|---|---|---|---|---|---|")
        for f in D1_FAMS:
            for name, key in (("dual", "dual"), ("(a) tuned", f"{kind}/tuned_a"), ("(b) tuned", f"{kind}/tuned_b"),
                              ("decoder only", f"{kind}/raw_decoder")):
                cells = [msd(st(f, f"{key}/{s}/map")) for s in D1_SCOPES]
                diff = "" if key == "dual" else msd(st(f, f"{key}/all/map_diff"))
                print(f"| {D1_LABEL[f]} | {name} | " + " | ".join(cells) + f" | {msd(st(f, f'{key}/all/rp'))} | {diff} |")
        for f in D1_FAMS:
            print(f"  {D1_LABEL[f]} (a) tuned minus dual by third: "
                  + ", ".join(f"{s} {st(f, f'{kind}/tuned_a/{s}/map_diff')['mean']:+.2f}" for s in D1_SCOPES))
    lo = min(st(f, "dual/all/map")["mean"] for f in D1_FAMS)
    hi = max(st(f, "dual/all/map")["mean"] for f in D1_FAMS)
    print(f"Dual i2t mAP@R across families: {lo:.2f} to {hi:.2f}")

    # --- re-ranker (b)
    print("\nRe-ranker (b): collapsed (tuned beta = 0, the dual order) and the betas chosen")
    for f in D1_FAMS:
        for kind in D1_KINDS:
            betas = [p["c"]["tuned"][kind]["beta"] for p in per[f]]
            grid = {k.split("/")[2]: v for k, v in fams[f]["stats"].items()
                    if k.startswith(f"{kind}/grid_b/") and k.endswith("/all/map_diff")}
            gbest = max(grid, key=lambda k: grid[k]["mean"])
            print(f"  {D1_LABEL[f]} {kind}: collapsed {fams[f]['collapsed_b_fraction'][kind]:.2f}, betas {betas}, "
                  f"(b) tuned minus dual {msd(st(f, f'{kind}/tuned_b/all/map_diff'))}, top third "
                  f"{msd(st(f, f'{kind}/tuned_b/R>=21/map_diff'))}; best (alpha_beta) on the TEST grid {gbest}: "
                  f"{msd(grid[gbest])}")
    nz = sorted({p["c"]["tuned"][k]["beta"] for f in D1_FAMS for p in per[f] for k in D1_KINDS} - {0.0})
    print(f"Non-zero tuned betas over all checkpoints and kinds: {nz}")

    # --- per-checkpoint analyses from scores.npz
    print("\nRandom order and oracle within the dual top 50 (mean ± SD over a family's seeds), and score correlations "
          "within each query's 50 candidates")
    rnd_all, extra, pooled = defaultdict(list), {}, defaultdict(list)
    for f in D1_FAMS:
        rows = defaultdict(list)
        for p in per[f]:
            z, c = p["z"], p["c"]
            rnd, orc, m = random_and_oracle(z["cand"], q)
            for s in D1_SCOPES:
                ix = np.arange(len(q.R)) if s == "all" else thirds[s]
                rows[f"random/{s}"].append(float(rnd[ix].mean()))
                rows[f"oracle/{s}"].append(float(orc[ix].mean()))
                rnd_all[s].append(float(rnd[ix].mean()) - c["metrics"]["dual"][s]["map"])
                pooled[f"random/{s}"].append(float(rnd[ix].mean()))
                pooled[f"oracle/{s}"].append(float(orc[ix].mean()))
                pooled[f"positives in top 50/{s}"].append(float(m[ix].mean()))
                for kind in D1_KINDS:
                    rows[f"{kind}/lift_a/{s}"].append(c["metrics"][kind]["tuned_a"][s]["map"] - float(rnd[ix].mean()))
                    rows[f"{kind}/lift_dec/{s}"].append(c["metrics"][kind]["raw_decoder"][s]["map"] - float(rnd[ix].mean()))
            rows["positives in top 50"].append(float(m.mean()))
            rows["recall@50 (share of R)"].append(float((m / q.R).mean()))
            for kind in D1_KINDS:
                dec, null, dual = z[f"dec_{kind}"], z[f"null_{kind}"], z["dual"]
                rho = [stats.spearmanr(dec[i], null[i])[0] for i in range(len(dec))]
                r2 = [np.corrcoef(dec[i], null[i])[0, 1] ** 2 for i in range(len(dec))]
                rows[f"{kind}/spearman(dec, null)"].append(float(np.nanmean(rho)))
                rows[f"{kind}/r2(dec, null)"].append(float(np.nanmean(r2)))
                if f != "s2_none_txt80":
                    pmi = cp.score_a(dec, null, 1.0)
                    rows[f"{kind}/spearman(PMI alpha 1, dual)"].append(
                        float(np.nanmean([stats.spearmanr(pmi[i], dual[i])[0] for i in range(len(dec))])))
                    rows[f"{kind}/spearman(dec, dual)"].append(
                        float(np.nanmean([stats.spearmanr(dec[i], dual[i])[0] for i in range(len(dec))])))
        extra[f] = rows
        print(f"{D1_LABEL[f]}: " + "; ".join(f"{k} {ms(v, 3)}" for k, v in rows.items()))
    print("Mean over all 15 checkpoints: " + "; ".join(f"{k} {mean(v):.2f}" for k, v in pooled.items()))
    print("Random order minus dual, mean over all 15 checkpoints: "
          + ", ".join(f"{s} {mean(v):+.2f}" for s, v in rnd_all.items()))

    # --- the control: alpha = 1 gives exactly the dual order after snapping
    print("\nText-only control (s2_none_txt80): re-ranker (a) at alpha = 1")
    for p in per["s2_none_txt80"]:
        z, c = p["z"], p["c"]
        for kind in D1_KINDS:
            dec, null, cand = z[f"dec_{kind}"], z[f"null_{kind}"], z["cand"]
            snapped = cp.rerank(cand, cp.score_a(dec, null, 1.0))
            raw = cp.rerank(cand, dec - null)
            moved = int((raw != cand).any(1).sum())
            raw_map = float(100 * cp.ap_at_r(raw, q.positives, q.R).mean())
            print(f"  seed {c['seed']} {kind}: max |dec - null| {np.abs(dec - null).max():.1e}; snapped order == dual order "
                  f"for every query: {bool((snapped == cand).all())}; without snapping {moved} of {len(cand)} queries "
                  f"reordered, mAP@R {raw_map:.2f} vs dual {c['metrics']['dual']['all']['map']:.2f} "
                  f"({raw_map - c['metrics']['dual']['all']['map']:+.2f})")

    # --- ML-80 against M1 clean (the review's H-b prediction), and the other families
    print("\nFamily contrasts (difference of means over seeds, Welch p, 3 vs 3 checkpoints)")
    for a_, b_ in (("s2_txt80_mae0", "s2_m1clean_txt80"), ("s2_txt80_mae0", "s2_txt80"), ("s2_txt80_mae0", "s2_multilearner"),
                   ("s2_m1clean_txt80", "s2_txt80")):
        for kind in D1_KINDS:
            for name, key in (("(a) tuned", "tuned_a"), ("decoder only", "raw_decoder")):
                va = [p["c"]["metrics"][kind][key]["all"]["map"] for p in per[a_]]
                vb = [p["c"]["metrics"][kind][key]["all"]["map"] for p in per[b_]]
                print(f"  {D1_LABEL[a_]} minus {D1_LABEL[b_]}, {kind}, {name}: {mean(va) - mean(vb):+.2f} "
                      f"(p {stats.ttest_ind(va, vb, equal_var=False).pvalue:.4f})")
        va = [p["c"]["metrics"]["dual"]["all"]["map"] for p in per[a_]]
        vb = [p["c"]["metrics"]["dual"]["all"]["map"] for p in per[b_]]
        print(f"  {D1_LABEL[a_]} minus {D1_LABEL[b_]}, dual: {mean(va) - mean(vb):+.2f} "
              f"(p {stats.ttest_ind(va, vb, equal_var=False).pvalue:.4f})")
    print()
    return curves, rnd_all


# --------------------------------------------------------------------------------------------------- J. D2
def d2_family(run_name: str) -> str:
    return "zeroshot" if run_name == "zeroshot" else run_name.split("_", 2)[2]


def pair_components(pairs: np.ndarray) -> np.ndarray:
    """Component id per pair of the graph whose nodes are images and whose edges are pairs: pairs that share an image,
    directly or through other pairs, fall in one component (the resampling unit that respects image reuse)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs.tolist():
        parent[find(a)] = find(b)
    roots = [find(a) for a, _ in pairs.tolist()]
    ids = {r: i for i, r in enumerate(dict.fromkeys(roots))}
    return np.array([ids[r] for r in roots])


def load_d2_runs() -> dict:
    runs = defaultdict(list)
    for f in sorted(D2_DIR.glob("*/d2.json")):
        name = f.parent.name
        fam = d2_family(name)
        seed = None if fam == "zeroshot" else int(yaml.safe_load((COCO_RUNS / name / "config.yaml").read_text())["seed"])
        runs[fam].append({"run": name, "seed": seed, "json": json.loads(f.read_text()),
                          "z": dict(np.load(f.parent / "d2_scores.npz")), "mtime": f.stat().st_mtime})
    for fam in runs:
        runs[fam].sort(key=lambda r: -1 if r["seed"] is None else r["seed"])
    assert set(runs) == set(D2_FAMS), sorted(runs)
    return runs


def sel_items(z: dict, j: int):
    """Pooled selection items of one run at j visible content tokens: (blend, own, other), cA items first, then cB, so
    item i and item P + i belong to pair i."""
    a = [z[f"sel_cA_j{j}_{n}"] for n in ("sA", "sB", "blend")]
    b = [z[f"sel_cB_j{j}_{n}"] for n in ("sA", "sB", "blend")]
    return np.concatenate([a[2], b[2]]), np.concatenate([a[0], b[1]]), np.concatenate([a[1], b[0]])


def d2():
    print("## J. D2: blend probe on COCO 5k test\n")
    tab = json.loads(D2_TABLE.read_text())["d2"]
    fams = tab["families"]
    pairs = np.array(json.loads((D2_DIR / "pairs.json").read_text()))
    uses = Counter(pairs.reshape(-1).tolist())
    comp = pair_components(pairs)
    sizes = np.bincount(comp)
    print(f"Pairs {len(pairs)}; distinct images {len(uses)}; images used once {sum(v == 1 for v in uses.values())}, "
          f"at most {max(uses.values())} times; uses histogram {dict(sorted(Counter(uses.values()).items()))}")
    print(f"Connected components of the image-pair graph: {len(sizes)}; pairs in the largest {sorted(sizes.tolist())[::-1][:5]}; "
          f"components of one pair {int((sizes == 1).sum())}")
    runs = load_d2_runs()
    import datetime as _dt
    import zoneinfo as _zi
    ams = _zi.ZoneInfo("Europe/Amsterdam")
    secs = [r["json"]["meta"]["seconds"] for f in D2_FAMS for r in runs[f]]
    first = min((r for f in D2_FAMS for r in runs[f]), key=lambda r: r["mtime"])
    last = max(r["mtime"] for f in D2_FAMS for r in runs[f])
    print(f"Models {len(secs)}; minutes per model {min(secs) / 60:.1f} to {max(secs) / 60:.1f}; total {sum(secs) / 3600:.2f} h; "
          f"first start {_dt.datetime.fromtimestamp(first['mtime'] - first['json']['meta']['seconds'], ams):%Y-%m-%d %H:%M}, "
          f"last d2.json {_dt.datetime.fromtimestamp(last, ams):%Y-%m-%d %H:%M}")
    for f in D2_FAMS:
        m = [r["json"]["meta"] for r in runs[f]]
        print(f"  {D2_LABEL[f]}: runs {[r['run'] for r in runs[f]]}, seeds {[r['seed'] for r in runs[f]]}, pairs "
              f"{sorted({x['pairs'] for x in m})}, decoder {m[0]['decoder']}, viewed {m[0].get('viewed')}, D1 alpha (parallel) "
              f"{[x.get('alpha_a_parallel') for x in m]}, K blend {m[0]['k_blend']}, K selection {m[0]['k_select']}")

    # --- balances recomputed from the stored scores, checked against hb_coco.json
    conds = [f"blend_{lam}" for lam in cp.LAMBDAS] + [f"mix_{m}" for m in cp.MIX_M]
    bal = {}
    dev = 0.0
    for f in D2_FAMS:
        for c in conds:
            vals = {"dual": [float(cp.balance(r["z"][f"dual_sim_{c}"][:, :5], r["z"][f"dual_sim_{c}"][:, 5:]).mean())
                             for r in runs[f]]}
            dev = max(dev, abs(mean(vals["dual"]) - fams[f]["stats"][f"dual/{c}/balance"]["mean"]))
            if f in D1_FAMS:
                par = [r["z"][f"dec_par_{c}"] for r in runs[f]]
                vals["dec"] = [float(cp.balance(p[:, :5], p[:, 5:]).mean()) for p in par]
                vals["gap"] = [float((p[:, :5].mean(1) - p[:, 5:].mean(1)).mean()) for p in par]
                dev = max(dev, abs(mean(vals["dec"]) - fams[f]["stats"][f"decoder/balance/{c}"]["mean"]),
                          abs(mean(vals["gap"]) - fams[f]["stats"][f"decoder/score_gap/{c}"]["mean"]))
            bal[(f, c)] = vals
    print(f"Largest |balance or score gap recomputed from d2_scores.npz - hb_coco.json| = {dev:.1e}")

    def cell(v, nd=3):
        return f"{v[0]:.{nd}f}" if len(v) == 1 else ms(v, nd)

    # --- both-covered and the kill rule
    kr = tab["kill_rule"]
    print(f"\nKill rule (C's dual both-covered@k within 2 points of, or above, the best decoder's after re-ranker (a), "
          f"control excluded): {kr['verdict']}; holds per condition "
          f"{ {c: v['holds_at_every_k'] for c, v in kr['per_condition'].items()} }")
    cc = kr["C_ceiling50"]
    print(f"C's dual both-covered@50 (the ceiling of any top-50 re-ranker), percent: "
          + ", ".join(f"{c} {v:.2f}" for c, v in cc.items()) + f"; range {min(cc.values()):.2f} to {max(cc.values()):.2f}")
    allv = [(fams[f]["stats"][k]["mean"], f, k) for f in D2_FAMS for k in fams[f]["stats"]
            if re.match(r"(dual/[^/]+/bc\d+|decoder/rerank_bc/[^/]+/bc\d+)$", k)]
    top = max(allv)
    print(f"Largest family-mean both-covered@k (k <= 20, dual or re-ranked, any family and condition): {top[0]:.2f} "
          f"({D2_LABEL[top[1]]}, {top[2]})")
    for f in D2_FAMS:
        ce = [fams[f]["stats"][f"dual/{c}/ceiling50"]["mean"] for c in conds]
        print(f"  {D2_LABEL[f]} ceiling@50 by condition: " + ", ".join(f"{c} {v:.2f}" for c, v in zip(conds, ce)))
    print("\n| family | dual bc@5 | bc@10 | bc@20 | ceiling@50 | (a) re-ranked bc@10 | bc@20 |   (blend 0.5, percent)")
    print("|---|---|---|---|---|---|---|")
    for f in D2_FAMS:
        rs = runs[f]
        dual = {k: [100 * r["json"]["dual"]["blend_0.5"]["both_covered"][k] for r in rs] for k in ("5", "10", "20")}
        ceil = [100 * r["json"]["dual"]["blend_0.5"]["both_covered_at_50"] for r in rs]
        rr = ({k: [100 * r["json"]["decoder"]["both_covered_rerank"]["blend_0.5"][k] for r in rs] for k in ("10", "20")}
              if f in D1_FAMS else None)
        print(f"| {D2_LABEL[f]} | {cell(dual['5'], 2)} | {cell(dual['10'], 2)} | {cell(dual['20'], 2)} | {cell(ceil, 2)} | "
              + (f"{cell(rr['10'], 2)} | {cell(rr['20'], 2)} |" if rr else " | |"))

    # --- balance curves
    print("\nBalance: A's share of the top 5 when A's and B's 10 captions are ranked together (mean ± seed SD)")
    print("| family | dual, lambda 0.3 / 0.5 / 0.7 | dual, m 3 / 7 / 11 | decoder, lambda 0.3 / 0.5 / 0.7 | decoder, m 3 / 7 / 11 | "
          "score gap (nats), lambda 0.3 / 0.5 / 0.7 |")
    print("|---|---|---|---|---|---|")
    for f in D2_FAMS:
        def row(kind, cs, nd=3):
            return " / ".join(cell(bal[(f, c)][kind], nd) for c in cs) if kind in bal[(f, conds[0])] else ""
        print(f"| {D2_LABEL[f]} | {row('dual', ['blend_0.3', 'blend_0.5', 'blend_0.7'])} | {row('dual', ['mix_3', 'mix_7', 'mix_11'])} | "
              f"{row('dec', ['blend_0.3', 'blend_0.5', 'blend_0.7'])} | {row('dec', ['mix_3', 'mix_7', 'mix_11'])} | "
              f"{row('gap', ['blend_0.3', 'blend_0.5', 'blend_0.7'], 2)} |")
    sds = [sd(v) for (f, c), d in bal.items() if f != "zeroshot" for k, v in d.items() if k != "gap"]
    print(f"Largest seed SD of any balance: {max(sds):.3f}")
    d1tab = json.loads(D1_TABLE.read_text())["d1"]["families"]
    print("Spread (balance at the highest A share minus the lowest), mean ± seed SD; D1 decoder-only mAP@R (parallel) beside")
    for f in D2_FAMS:
        out = []
        for kind in ("dual", "dec"):
            if kind not in bal[(f, conds[0])]:
                continue
            for lo, hi in (("blend_0.3", "blend_0.7"), ("mix_3", "mix_11")):
                v = [b - a for a, b in zip(bal[(f, lo)][kind], bal[(f, hi)][kind])]
                out.append(f"{kind} {lo.split('_')[0]} {cell(v)}")
        d1m = (f"; D1 decoder only {d1tab[f]['stats']['parallel/raw_decoder/all/map']['mean']:.2f}" if f in D1_FAMS else "")
        print(f"  {D2_LABEL[f]}: " + ", ".join(out) + d1m)
    nc = tab["none_control"]
    gaps = [g for c in conds[:5] for g in bal[("s2_none_txt80", c)]["gap"]]
    print(f"Text-only control: decoder balance by lambda {[round(v, 4) for v in nc['decoder_balance_by_lambda'].values()]} "
          f"(range {nc['range']:.4f}); score gap over lambda and seeds {min(gaps):.4f} to {max(gaps):.4f}")

    # --- selection index, recomputed from the stored scores
    print("\nSelection index at the 0.5 blend: (s_blend - s_other) / (s_own - s_other) for cA (A's first caption, A's "
          "content tokens visible) and cB; pooled over both sides")
    print("| family | j | ratio of sums | median | mean | near-ties (|own - other| < 0.05) | dropped | mean own - other |")
    print("|---|---|---|---|---|---|---|---|")
    sel = {}
    dev = 0.0
    for f in D1_FAMS:
        if f == "s2_none_txt80":  # its decoder ignores the image: every item is a tie (reported below)
            continue
        for j in range(4):
            rows = defaultdict(list)
            for r in runs[f]:
                blend_, own, other = sel_items(r["z"], j)
                idx, dropped = cp.selection_index(blend_, own, other)
                den = (own - other).sum()
                rows["ratio"].append(float((blend_ - other).sum() / den) if abs(den) >= cp.TIE_TOL else float("nan"))
                rows["median"].append(float(np.median(idx)) if len(idx) else float("nan"))
                rows["mean"].append(float(idx.mean()) if len(idx) else float("nan"))
                rows["ties"].append(int((np.abs(own - other) < 0.05).sum()))
                rows["dropped"].append(dropped)
                rows["gap"].append(float((own - other).mean()))
            sel[(f, j)] = rows
            ref = fams[f]["stats"].get(f"decoder/selection/{j}/pooled/ratio_of_sums")
            if ref is not None:
                dev = max(dev, abs(mean(rows["ratio"]) - ref["mean"]),
                          abs(mean(rows["median"]) - fams[f]["stats"][f"decoder/selection/{j}/pooled/median"]["mean"]))
            print(f"| {D2_LABEL[f]} | {j} | {ms(rows['ratio'])} | {ms(rows['median'])} | {ms(rows['mean'])} | "
                  f"{ms(rows['ties'], 1)} | {ms(rows['dropped'], 0)} | {ms(rows['gap'])} |")
    print(f"Largest |selection ratio of sums or median recomputed - hb_coco.json| (image-reading families) = {dev:.1e}")

    # cluster bootstrap over connected components of the pair graph: the rise j = 3 minus j = 0 of the ratio of sums
    rng = np.random.default_rng(0)
    B = 10_000
    C = len(sizes)
    W = rng.multinomial(C, np.full(C, 1 / C), size=B).astype(float)  # (B, C) draws of each component
    item_comp = np.concatenate([comp, comp])
    rise = {}
    for f in ("s2_txt80_mae0", "s2_txt80", "s2_multilearner", "s2_m1clean_txt80"):
        per_seed = []
        for r in runs[f]:
            ratios = []
            for j in (0, 3):
                blend_, own, other = sel_items(r["z"], j)
                N = np.bincount(item_comp, weights=blend_ - other, minlength=C)
                D = np.bincount(item_comp, weights=own - other, minlength=C)
                ratios.append((W @ N) / (W @ D))
            per_seed.append(ratios[1] - ratios[0])
        rise[f] = np.mean(per_seed, axis=0)
        point = mean(sel[(f, 3)]["ratio"]) - mean(sel[(f, 0)]["ratio"])
        seedwise = [b - a for a, b in zip(sel[(f, 0)]["ratio"], sel[(f, 3)]["ratio"])]
        med = [b - a for a, b in zip(sel[(f, 0)]["median"], sel[(f, 3)]["median"])]
        lo, hi = np.percentile(rise[f], [2.5, 97.5])
        print(f"  {D2_LABEL[f]}: ratio of sums j0 -> j3 {mean(sel[(f, 0)]['ratio']):.3f} -> {mean(sel[(f, 3)]['ratio']):.3f}, "
              f"rise {point:+.3f} (seed SD {sd(seedwise):.3f}; 95% cluster-bootstrap interval [{lo:+.3f}, {hi:+.3f}], "
              f"B = {B}, {C} components); median {mean(sel[(f, 0)]['median']):.3f} -> {mean(sel[(f, 3)]['median']):.3f} "
              f"(rise {mean(med):+.3f} ± {sd(med):.3f})")
    for f in ("s2_txt80_mae0", "s2_txt80", "s2_multilearner"):
        d = rise[f] - rise["s2_m1clean_txt80"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        pt = (mean(sel[(f, 3)]["ratio"]) - mean(sel[(f, 0)]["ratio"])) - (
            mean(sel[("s2_m1clean_txt80", 3)]["ratio"]) - mean(sel[("s2_m1clean_txt80", 0)]["ratio"]))
        print(f"  rise of {D2_LABEL[f]} minus rise of M1 clean: {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    ctl = runs["s2_none_txt80"]
    print(f"Text-only control: selection items with |s_own - s_other| < 1e-6 (dropped) per seed and j: "
          f"{[[cp.selection_index(*sel_items(r['z'], j))[1] for j in range(4)] for r in ctl]} of {2 * len(pairs)}")
    print()
    return {"bal": bal, "sel": sel, "conds": conds}


# --------------------------------------------------------------------------------------------------- K. collapse
def collapse():
    """The pooled image embedding of a 13-patch view (one view per image from encode.view_ids, seed 1234) against
    the full image's, on the first 500 COCO 5k test images, for every D2 model and M1 detached. GPU if visible."""
    import torch

    sys.path.insert(0, str(REPO / "scripts"))
    import hb_coco_d1
    import hb_coco_d2
    from mmae.data import Collator
    from mmae.data.coco import CocoRetrieval
    from mmae.data.transforms import build_image_transform
    from mmae.engine.hb import encode
    from mmae.models.backbones import processor_name

    print("## K. Collapse of the 13-patch view embedding (first "
          f"{COLLAPSE_N} COCO 5k test images, one view each)\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zs, cfg = hb_coco_d2.zero_shot_model(device)
    proc = processor_name(cfg.model.backbone)
    ds = CocoRetrieval(cfg.data.images_dir, cfg.data.annotations_dir, "test", build_image_transform(proc), limit=COLLAPSE_N)
    col = Collator(proc, int(cfg.data.max_text_len))
    imgs = torch.stack([ds[i][0] for i in range(COLLAPSE_N)])
    tok = hb_coco_d1.tokens_of(col, hb_coco_d1.flat_captions(ds))
    views = encode.view_ids(COLLAPSE_N, 1)[0]
    n = COLLAPSE_N
    off = ~torch.eye(n, dtype=torch.bool)
    ar = torch.arange(n)

    @torch.no_grad()
    def measure(model):
        full, view = [], []
        for s in range(0, n, 100):
            x = imgs[s : s + 100].to(device)
            full.append(model.embed_image(x).float().cpu())
            view.append(model.vision.pool(model.vision.encode(x, views[s : s + 100].to(device))).float().cpu())
        full, view = torch.cat(full), torch.cat(view)
        cap = torch.nn.functional.normalize(hb_coco_d2.embed_captions(model, tok, 500, device), dim=-1)
        fv = full @ view.T
        out = {"view_pairwise": (view @ view.T)[off].mean().item(), "full_pairwise": (full @ full.T)[off].mean().item(),
               "full_own_view": fv.diag().mean().item(), "full_other_view": fv[off].mean().item(),
               "view_to_full_r1": (fv.argmax(0) == ar).float().mean().item()}
        for name, e in (("full", full), ("view", view)):
            out[f"i2t_r1_{name}"] = ((e @ cap.T).argmax(1) // 5 == ar).float().mean().item()
        return out

    folders = [r["run"] for f in D2_FAMS for r in load_d2_runs()[f] if f != "zeroshot"]
    folders += sorted(p.name for p in COCO_RUNS.glob("*_s2_m1detached_txt80"))
    res = defaultdict(list)
    res["zeroshot"].append(measure(zs))
    del zs
    for name in folders:
        model, _ = encode.load_run(COCO_RUNS / name, device)
        res[d2_family(name)].append(measure(model))
        del model
    keys = ("view_pairwise", "full_pairwise", "full_own_view", "full_other_view", "view_to_full_r1", "i2t_r1_full",
            "i2t_r1_view")
    print(f"Device {device}; images {n}; captions {len(tok['input_ids'])}; view seed 1234 (encode.view_ids); chance "
          f"view-to-full R@1 {1 / n:.3f}")
    print("| family | runs | " + " | ".join(keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    for f in COLLAPSE_FAMS:
        print(f"| {D2_LABEL[f]} | {len(res[f])} | " + " | ".join(
            (f"{res[f][0][k]:.3f}" if len(res[f]) == 1 else ms([r[k] for r in res[f]])) for k in keys) + " |")
    for f in COLLAPSE_FAMS:
        print(f"  {D2_LABEL[f]} per run view_pairwise {[round(r['view_pairwise'], 3) for r in res[f]]}, full_own_view "
              f"{[round(r['full_own_view'], 3) for r in res[f]]}")
    print()
    return res


# ------------------------------------------------------------------------------------------------ figures
def box(ax, x, y, w, h, text, col, size=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.4,
                                edgecolor=col, facecolor=matplotlib.colors.to_rgba(col, 0.13)))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0))


def fig_arms():
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0.1, 5.5)
    ax.axis("off")
    rows = [
        (3.95, "ML-80", "ML-80+MAE also trains\nthe pixel decoder", "25% image view\n(13 of 49 patches,\nmasked pass)", D_SAME,
         "caption, 80% of\ntokens hidden", D_SAME, "Readout: every caption\ntoken hidden, 9 lengths\nx 16 random 25% views"),
        (2.05, "Par-cap", "", "full clean image\n(49 patches,\nclean pass)", D_REPL, "caption, 100% of\ntokens hidden", D_REPL,
         "Readout: every caption\ntoken hidden, 9 lengths\nx the full clean image"),
    ]
    for y, name, sub, img, ic, cap, cc, readout in rows:
        ax.text(0.05, y + 0.62, name, fontsize=9, fontweight="bold", color=INK, va="center")
        if sub:
            ax.text(0.05, y + 0.12, sub, fontsize=7, color=MUTED, va="center")
        box(ax, 1.9, y + 0.62, 1.8, 0.72, img, ic)
        box(ax, 1.9, y - 0.25, 1.8, 0.72, cap, cc)
        box(ax, 4.15, y + 0.1, 1.45, 0.9, "multilearner\nfusion", D_SAME)
        ax.add_patch(FancyBboxPatch((6.05, y - 0.15), 1.9, 1.4, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.4,
                                    edgecolor=D_SAME, facecolor=matplotlib.colors.to_rgba(D_SAME, 0.13)))
        ax.text(7.0, y + 0.85, "query decoder\n(caption queries)", ha="center", va="center", fontsize=7.5, color=INK)
        box(ax, 6.25, y - 0.02, 1.5, 0.4, "emotion query", D_NEW, size=7)
        box(ax, 8.4, y + 0.62, 1.6, 0.6, "caption head\n(CLIP vocabulary)", D_SAME)
        box(ax, 8.4, y - 0.12, 1.6, 0.6, "emotion head\n9 classes, w = 0.07", D_NEW)
        arrow(ax, 3.7, y + 0.98, 4.15, y + 0.7)
        arrow(ax, 3.7, y + 0.11, 4.15, y + 0.4)
        arrow(ax, 5.6, y + 0.55, 6.05, y + 0.55)
        arrow(ax, 7.95, y + 0.85, 8.4, y + 0.92)
        arrow(ax, 7.95, y + 0.18, 8.4, y + 0.18)
        ax.text(10.25, y + 0.55, readout, fontsize=7, color=MUTED, va="center", ha="left")
    ax.text(0.05, 0.82, "C", fontsize=9, fontweight="bold", color=INK, va="center")
    box(ax, 1.9, 0.3, 6.05, 1.0, "CLIP B/32 fine-tuned with InfoNCE only: no masked pass, no decoder.\n"
        "Readouts: softmax over 9 emotion prompts, and a soft-label linear probe\non the frozen pooled image embedding "
        "(the probe is fitted on every arm's embedding)", D_SAME, size=7.5)
    ax.text(8.4, 0.8, "Every arm keeps InfoNCE on the clean\npooled embeddings, which serve retrieval.\n"
            "The emotion is never an input.", fontsize=7, color=MUTED, va="center")
    ax.legend(handles=[Patch(facecolor=matplotlib.colors.to_rgba(D_SAME, 0.13), edgecolor=D_SAME, label="same as the COCO ML-80 model"),
                       Patch(facecolor=matplotlib.colors.to_rgba(D_REPL, 0.13), edgecolor=D_REPL, label="replaced in Par-cap"),
                       Patch(facecolor=matplotlib.colors.to_rgba(D_NEW, 0.13), edgecolor=D_NEW, label="new in this spec")],
              loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False, fontsize=8)
    fig.savefig(OUT / "arms.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_training(curves):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for a in ARMS:
        ep, rs, em = curves[a]
        axes[0].errorbar(ep, rs.mean(0), yerr=rs.std(0, ddof=1), color=COL[a], marker="o", ms=4, lw=1.6, capsize=2,
                         label=LABEL[a])
        if em is not None:
            axes[1].errorbar(ep, em.mean(0), yerr=em.std(0, ddof=1), color=COL[a], marker="o", ms=4, lw=1.6, capsize=2,
                             label=LABEL[a])
    axes[0].set_title("Validation rsum peaks at epoch 1 or 2", fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("validation rsum", fontsize=8, color=MUTED)
    axes[1].set_title("Validation emotion loss (each arm's own masking)", fontsize=9, color=INK, loc="left")
    axes[1].set_ylabel("emotion cross-entropy (nats)", fontsize=8, color=MUTED)
    for ax in axes:
        ax.axvline(2, color=MUTED, ls=":", lw=1)
        ax.set_xlabel("epoch", fontsize=8, color=MUTED)
        style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "training.png", dpi=170)
    plt.close(fig)


def fig_d4(final):
    items = [("c", "prompt_full", "C\nprompt"), ("ml80", "views16", "ML-80\n16 views"), ("ml80_mae", "views16", "ML-80+MAE\n16 views"),
             ("parcap", "full", "Par-cap\nfull image"), ("probe", "probe_full", "strongest\nprobe")]
    sh = final["references"]["split_half"]
    prior = mean(per_seed(final, "ml80", "prior", "jsd"))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, key, title in ((axes[0], "jsd", "JSD to the human votes (lower is better)"),
                           (axes[1], "entropy_spearman", "Entropy Spearman (higher is better)")):
        for i, (a, r, _) in enumerate(items):
            v = probe_seed(final, key) if a == "probe" else per_seed(final, a, r, key)
            # dot and whisker (final review M9: bars on an axis that starts at 0.20 inflated the gaps)
            ax.errorbar(i, mean(v), yerr=sd(v), fmt="D", color=COL[a], ms=7, capsize=4, lw=1.4, zorder=3)
            ax.scatter([i + 0.2, i + 0.26, i + 0.32], v, facecolor="white", edgecolor=COL[a], s=16, lw=1.1, zorder=3)
        ax.axhline(sh[key], color=INK, ls=":", lw=1.2)
        ax.text(len(items) - 0.5, sh[key], f"split-half ceiling {sh[key]:.3f}", fontsize=7, color=INK, ha="right",
                va="bottom")
        if key == "jsd":
            ax.axhline(prior, color=MUTED, ls="--", lw=1.2)
            ax.text(len(items) - 0.5, prior, f"prior only {prior:.3f}", fontsize=7, color=MUTED, ha="right", va="bottom")
            ax.set_ylim(0.2, 0.36)
        else:
            ax.set_ylim(0, 0.85)
        ax.set_xlim(-0.5, len(items) - 0.35)
        ax.set_xticks(range(len(items)), [x[2] for x in items], fontsize=7.5)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_ylabel({"jsd": "mean Jensen-Shannon distance", "entropy_spearman": "Spearman rho"}[key], fontsize=8, color=MUTED)
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "d4_primary.png", dpi=170)
    plt.close(fig)


def fig_thirds(thirds):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    x = np.arange(3)
    for a, off in (("c", -0.12), ("ml80", -0.04), ("parcap", 0.04), ("probe", 0.12)):
        lab = {"c": "C prompt", "ml80": "ML-80 (16 views)", "parcap": "Par-cap (full image)", "probe": "strongest probe"}[a]
        for k, ax in ((0, axes[0]), (1, axes[1])):
            v = [thirds[a][t][k] for t in range(3)]
            ax.errorbar(x + off, [mean(y) for y in v], yerr=[sd(y) for y in v], color=COL[a], marker="o", ms=5, lw=1.6,
                        capsize=2, label=lab)
    axes[0].plot(x, [mean(thirds["prior"][t][0]) for t in range(3)], color=MUTED, ls="--", marker="s", ms=4, lw=1.2,
                 label="prior only")
    axes[0].set_ylabel("mean Jensen-Shannon distance", fontsize=8, color=MUTED)
    axes[0].set_title("JSD by thirds of human entropy (lower is better)", fontsize=9, color=INK, loc="left")
    axes[1].set_ylabel("entropy Spearman rho within the third", fontsize=8, color=MUTED)
    axes[1].set_title("Entropy Spearman within each third", fontsize=9, color=INK, loc="left")
    for ax in axes:
        ax.set_xticks(x, ["low entropy\n(people agree)", "middle", "high entropy\n(people disagree)"], fontsize=7.5)
        style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "d4_thirds.png", dpi=170)
    plt.close(fig)


def fig_d6(curves, final):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    xs = [1, 4, 16]
    xfull = 40
    for a, ls in (("ml80", "-"), ("ml80_mae", "-"), ("parcap", "--")):
        lab = {"ml80": "ML-80", "ml80_mae": "ML-80+MAE", "parcap": "Par-cap (views never seen in training)"}[a]
        for k, ax in ((0, axes[0]), (1, axes[1])):
            v = [curves[a][f"views{K}"][k] for K in xs]
            ax.errorbar(xs, [mean(y) for y in v], yerr=[sd(y) for y in v], color=COL[a], ls=ls, marker="o", ms=5,
                        lw=1.6, capsize=2, label=lab)
            f = curves[a]["full"][k]
            ax.errorbar([xfull * (1.06 if a == "parcap" else (0.94 if a == "ml80" else 1.0))], [mean(f)], yerr=[sd(f)],
                        color=COL[a], marker="D", ms=6, capsize=2, ls="none")
    pv = [probe_seed(final, "jsd", readout="probe_views16"), probe_seed(final, "entropy_spearman", readout="probe_views16")]
    pf = [probe_seed(final, "jsd"), probe_seed(final, "entropy_spearman")]
    for k, ax in ((0, axes[0]), (1, axes[1])):
        ax.errorbar([16 * 1.08], [mean(pv[k])], yerr=[sd(pv[k])], color=COL["probe"], marker="^", ms=6, capsize=2, ls="none",
                    label="strongest probe, mean over 16 views")
        ax.errorbar([xfull * 1.12], [mean(pf[k])], yerr=[sd(pf[k])], color=COL["probe"], marker="D", ms=6, capsize=2, ls="none")
        ax.set_xscale("log")
        ax.set_xticks(xs + [xfull], ["1", "4", "16", "full\nimage"], fontsize=7.5)
        ax.minorticks_off()
        ax.set_xlabel("number of 25% views averaged (K)", fontsize=8, color=MUTED)
        style(ax)
    axes[0].set_ylabel("mean Jensen-Shannon distance", fontsize=8, color=MUTED)
    axes[0].set_title("JSD (lower is better)", fontsize=9, color=INK, loc="left")
    axes[1].set_ylabel("Spearman rho", fontsize=8, color=MUTED)
    axes[1].set_title("Entropy Spearman (higher is better)", fontsize=9, color=INK, loc="left")
    handles, labels = axes[1].get_legend_handles_labels()
    handles.append(Line2D([], [], color=MUTED, marker="D", ls="none", ms=6))
    labels.append("diamonds: the same model on the full image")
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "d6_views.png", dpi=170)
    plt.close(fig)


def fig_d7(final, prior_ll):
    d = final["d7"]["arms"]
    js = [0, 1, 2, 4, 8]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    for ax, kind, title in ((axes[0], "prefix", "First j content tokens visible"),
                            (axes[1], "random", "j random content tokens visible")):
        pats = ["j0"] + [f"{kind}{j}" for j in js[1:]]
        for a in ("ml80", "parcap"):
            for which, ls, mk in (("real", "-", "o"), ("null", "--", "s")):
                y = [d[a]["patterns"][p][which]["log_loss"] for p in pats]
                lab = f"{LABEL[a]}, {'real image' if which == 'real' else 'null image'}"
                ax.plot(js, y, color=COL[a], ls=ls, marker=mk, ms=5, lw=1.6, mfc=COL[a] if which == "real" else "white",
                        label=lab)
        ax.plot(js, [prior_ll[p] for p in pats], color=INK, ls=":", marker="x", ms=5, lw=1.3,
                label="prior only (train label frequencies)")
        ax.set_xticks(js)
        ax.set_xlabel("visible content tokens j", fontsize=8, color=MUTED)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        style(ax)
    axes[0].set_ylabel("log-loss of the annotator's emotion (nats)", fontsize=8, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(OUT / "d7_partial.png", dpi=170)
    plt.close(fig)


def fig_d1(curves, rnd_all):
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), sharex=True)
    x = np.array([float(a) for a in D1_ALPHAS])
    for r, kind in enumerate(D1_KINDS):
        for col, (scope, scope_name) in enumerate((("all", "all 1,261 queries"), ("R>=21", "top third of R (R ≥ 21, 345 queries)"))):
            ax = axes[r, col]
            for i, f in enumerate(D1_FAMS):
                v = curves[(f, kind)][col]
                off = (i - 2) * 0.012
                ax.errorbar(x + off, [s["mean"] for s in v], yerr=[s["sd"] for s in v], color=D1_COL[f], marker=D1_MARK[f],
                            ms=5, lw=1.6, capsize=2, ls="--" if f == "s2_none_txt80" else "-", label=D1_LABEL[f])
            ax.axhline(0, color=INK, lw=1.0)
            ax.text(0.0, 0.6, "dual encoder", fontsize=7, color=INK, va="bottom")
            ref = mean(rnd_all[scope])
            ax.axhline(ref, color=MUTED, ls=":", lw=1.2)
            ax.set_ylim(-26, 2.5) if scope == "all" else ax.set_ylim(-16, 2.5)
            ax.set_title(f"{'Parallel' if kind == 'parallel' else 'Training-ratio'} score, {scope_name}", fontsize=9,
                         color=INK, loc="left")
            ax.set_xticks(x, D1_ALPHAS, fontsize=7.5)
            if r == 1:
                ax.set_xlabel("prior weight alpha in PMI = s(c | image) − alpha · s(c | null image)", fontsize=8, color=MUTED)
            if col == 0:
                ax.set_ylabel("mAP@R, (a) minus dual (points)", fontsize=8, color=MUTED)
            style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=MUTED, ls=":", lw=1.2))
    labels.append("random order of the dual top 50 (mean over the 15 checkpoints)")
    fig.legend(handles, labels, fontsize=7.5, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "d1_rerank.png", dpi=170)
    plt.close(fig)


def fig_d2_balance(d2r):
    bal = d2r["bal"]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6), sharey=True)
    lam = [float(x) for x in cp.LAMBDAS]
    mm = list(cp.MIX_M)
    panels = ((0, "dual", "blend", "Dual encoder, pixel blend"), (1, "dual", "mix", "Dual encoder, 13-patch mix"),
              (2, "dec", "blend", "Caption decoder (parallel score), pixel blend"),
              (3, "dec", "mix", "Caption decoder (parallel score), patch mix"))
    for i, kind, cond, title in panels:
        ax = axes.flat[i]
        x = lam if cond == "blend" else [m / 13 for m in mm]
        keys = [f"blend_{v}" for v in cp.LAMBDAS] if cond == "blend" else [f"mix_{m}" for m in mm]
        lim = (x[0] - 0.035, x[-1] + 0.035)
        ax.plot(lim, lim, color=GRID, lw=1.6, ls="-", zorder=0)
        for n, f in enumerate(D2_FAMS):
            if kind not in bal[(f, keys[0])]:
                continue
            v = [bal[(f, k)][kind] for k in keys]
            off = (n - 3) * 0.005  # small offsets so overlapping families stay visible
            ax.errorbar([xi + off for xi in x], [mean(y) for y in v], yerr=[sd(y) if len(y) > 1 else 0 for y in v],
                        color=D2_COL[f], marker=D2_MARK[f], ms=5, lw=1.5, capsize=2, ls=D2_LS[f], label=D2_LABEL[f],
                        mfc="white" if f == "s2_txt80" else D2_COL[f])
        ax.set_xlim(*lim)
        ax.set_ylim(-0.02, 1.02)
        if cond == "blend":
            ax.set_xticks(lam, [f"{v:.1f}" for v in lam], fontsize=7.5)
            ax.set_xlabel("pixel weight of image A (lambda)", fontsize=8, color=MUTED)
        else:
            ax.set_xticks(x, [f"{m}/13" for m in mm], fontsize=7.5)
            ax.set_xlabel("patches of image A in the 13-patch view (m)", fontsize=8, color=MUTED)
        if i % 2 == 0:
            ax.set_ylabel("balance: A's share of the top 5\nof A's and B's 10 captions", fontsize=8, color=MUTED)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=GRID, lw=1.2))
    labels.append("balance equal to A's share of the input (lambda or m/13)")
    fig.legend(handles, labels, fontsize=7.5, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "d2_balance.png", dpi=170)
    plt.close(fig)


def fig_d2_selection(d2r):
    sel = d2r["sel"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), sharey=True)
    js = [0, 1, 2, 3]
    fams = ("s2_txt80_mae0", "s2_txt80", "s2_multilearner", "s2_m1clean_txt80")
    for ax, key, title in ((axes[0], "ratio", "Ratio of sums over the 2,000 items"),
                           (axes[1], "median", "Median over the 2,000 items")):
        for i, f in enumerate(fams):
            v = [sel[(f, j)][key] for j in js]
            off = (i - 1.5) * 0.04
            ax.errorbar([j + off for j in js], [mean(y) for y in v], yerr=[sd(y) for y in v], color=D2_COL[f],
                        marker=D2_MARK[f], ms=5, lw=1.6, capsize=2, label=D2_LABEL[f],
                        mfc="white" if f == "s2_txt80" else D2_COL[f])
        ax.axhline(0.5, color=MUTED, ls=":", lw=1.0)
        ax.set_xticks(js)
        ax.set_xlabel("content tokens of the scored caption made visible (j)", fontsize=8, color=MUTED)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        style(ax)
    axes[0].set_ylabel("selection index at the 0.5 blend\n(1: scored like its own source, 0: like the other)", fontsize=8,
                       color=MUTED)
    axes[0].set_ylim(0.48, 0.62)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "d2_selection.png", dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    final = json.loads(FINAL.read_text())
    tr = load_training()
    inventory(tr)
    curves = training(tr)
    he = data_section(final)
    gate()
    thirds = d4(final)
    spread(final, he)
    d6c = d6(final)
    d7_prior = d7(final)
    d1c, d1r = d1()
    d2r = d2()
    collapse()
    fig_arms()
    fig_training(curves)
    fig_d4(final)
    fig_thirds(thirds)
    fig_d6(d6c, final)
    fig_d7(final, d7_prior)
    fig_d1(d1c, d1r)
    fig_d2_balance(d2r)
    fig_d2_selection(d2r)
    print("Figures written to", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
