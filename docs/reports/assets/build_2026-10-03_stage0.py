"""Tables and figures of docs/reports/auto/v1/2026-10-03_stage0_diagnostics.md, from the Stage 0 results.

Reads (never writes) res/coco/diagnostics/stage0/diagnostics.json (scripts/diagnose.py at 9919d8d on node403:
13 encoded models, the 12 baselines of res/coco/multimae/default/ plus zero-shot CLIP B/32, and 18 tower swaps),
res/coco/diagnostics/stage0/per_query.pt (per-query PMRP R-Precision, t2i and i2t, per model), and each baseline's
run.json (its own test metrics: run.json["eval"]["test"] for the seed-42 fusion runs, which were re-evaluated with
evaluate.py, else run.json["results"]["test"]). To count the PMRP t2i queries per number of COCO classes named it
also rebuilds the query list from the local COCO annotations and ECCV Caption PM files (configs/data/coco.yaml), with
the same functions diagnose.py used. Also read: res/coco/zeroshot/clip_b32_test.json (zero-shot package metrics),
res/coco/diagnostics/stage0/embeddings/*.pt (stored first-caption embeddings, for the stop-word probe subsets),
res/cluster_jobs/20261003-204221-9919d8d/status.json (job duration), /data/SSD/vwsd (VWSD item and candidate counts),
and the record lines of tests/20261003_ml_improve/runs.md (CPU dry run) and caveats.md (class-set group caveat).

Writes to docs/reports/assets/2026-10-03_stage0/:
  swaps.png         Figure 1: tower swaps, PMRP / rsum / ECCV mAP@R per fusion model and seed
  similarity.png    Figure 2: mean image-caption cosines (positive, same-class negative, other negative), gaps and
                    the relative position of same-class negatives
  pmrp_by_words.png Figure 3: PMRP t2i difference to contrastive by the number of COCO classes the caption names
and prints every table of the report (markdown) to stdout.

  /root/miniconda3/envs/MultiMAE/bin/python docs/reports/assets/build_2026-10-03_stage0.py
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
DIAG = REPO / "res" / "coco" / "diagnostics" / "stage0"
ZERO_SHOT = REPO / "res" / "coco" / "zeroshot" / "clip_b32_test.json"  # evaluate.py, GPU (baselines report)
REGISTRY = REPO / "tests" / "20261003_ml_improve" / "runs.md"  # holds the record of the CPU zero-shot dry run
JOB = REPO / "res" / "cluster_jobs" / "20261003-204221-9919d8d"  # the diagnostics job (status.json, manifest.json)
RUNS = REPO / "res" / "coco" / "multimae" / "default"
OUT = Path(__file__).resolve().parent / "2026-10-03_stage0"

MODELS = ["contrastive", "fusion_none", "fusion_concat", "fusion_multilearner"]
FUSION = MODELS[1:]
BASE = "contrastive"
# Same validated colours as the baselines report: grey baseline, first three categorical slots (all-pairs pass).
COLORS = {"zeroshot": "#a3a29c", "contrastive": "#6f6e69", "fusion_none": "#2a78d6", "fusion_concat": "#eb6834",
          "fusion_multilearner": "#1baf7a"}
SHORT = {"zeroshot": "zero-shot", "contrastive": "contrastive", "fusion_none": "none", "fusion_concat": "concat",
         "fusion_multilearner": "multilearner"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
N_VWSD = 463
WORD_BINS = ["0", "1", "2", "3+"]
SWAP_CONDS = ["contrastive", "fusion img + contrastive txt", "contrastive img + fusion txt", "fusion (both towers)"]
SWAP_METRICS = [("pmrp", "PMRP"), ("pmrp/i2t", "PMRP i2t"), ("pmrp/t2i", "PMRP t2i"), ("rsum", "rsum"),
                ("eccv/map_at_r", "ECCV mAP@R"), ("i2t_R1", "i2t R@1"), ("t2i_R1", "t2i R@1")]

ALL_TESTS: dict[str, float] = {}  # every Welch p-value against contrastive printed below, for the Holm count


# ----------------------------------------------------------------------------------------------- helpers
def mean(x):
    return statistics.mean(x)


def sd(x):
    return statistics.stdev(x) if len(x) > 1 else float("nan")


def ms(x, nd=2):
    return f"{mean(x):.{nd}f} ± {sd(x):.{nd}f}"


def welch(a, b):
    """t, Welch-Satterthwaite df and two-sided p for mean(a) - mean(b)."""
    res = stats.ttest_ind(a, b, equal_var=False)
    va, vb, na, nb = statistics.variance(a), statistics.variance(b), len(a), len(b)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return float(res.statistic), df, float(res.pvalue)


def diff_cell(a, b, key=None, nd=2):
    """'+d (p)' for mean(a) - mean(b), Welch; registers the test under key."""
    t, _, p = welch(a, b)
    if key is not None:
        ALL_TESTS[key] = p
    return f"{mean(a) - mean(b):+.{nd}f} ({p:.3g})"


def welch_ci(a, b, level=0.95):
    """mean(a) - mean(b) and its Welch confidence interval."""
    _, df, _ = welch(a, b)
    se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
    d = mean(a) - mean(b)
    hw = stats.t.ppf(0.5 + level / 2, df) * se
    return d, d - hw, d + hw


def holm(pvalues: dict) -> dict:
    order = sorted(pvalues, key=pvalues.get)
    m, running, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def power_effect(n=3, alpha=0.05, power=0.8):
    """Effect in pooled standard deviations that a two-sided t-test with n vs n runs (equal variances, df 2n-2)
    detects with the given power."""
    df = 2 * n - 2
    crit = stats.t.ppf(1 - alpha / 2, df)
    lo, hi = 0.0, 20.0
    for _ in range(100):
        d = (lo + hi) / 2
        nc = d / math.sqrt(2 / n)
        pw = 1 - stats.nct.cdf(crit, df, nc) + stats.nct.cdf(-crit, df, nc)
        lo, hi = (d, hi) if pw < power else (lo, d)
    return (lo + hi) / 2


def test_metrics(run: dict) -> dict:
    if "test" in run.get("eval", {}):
        return dict(run["eval"]["test"])
    return {k.removeprefix("test/").removeprefix("retrieval/"): v for k, v in run["results"]["test"].items()}


# ------------------------------------------------------------------------------------------------- data
def load():
    diag = json.loads((DIAG / "diagnostics.json").read_text())
    runs = diag["runs"]
    own = {}
    for name in runs:
        if name != "zeroshot":
            own[name] = test_metrics(json.loads((RUNS / name / "run.json").read_text()))
    return diag, runs, own


def by_model(runs: dict, key) -> dict[str, list[float]]:
    out = defaultdict(list)
    for name, r in sorted(runs.items(), key=lambda kv: (kv[1]["seed"] or 0)):
        if r["model"] != "zeroshot":
            out[r["model"]].append(key(name, r))
    return out


def word_bin_counts():
    """Number of PMRP t2i queries per number of COCO classes the caption names, in diagnose.py's query order,
    and the class-word count of every t2i query (for re-deriving the per-bin means from per_query.pt)."""
    from omegaconf import OmegaConf
    from eccv_caption import Metrics
    from mmae.data.coco import retrieval_items
    from mmae.engine.diagnostics import class_set_groups, count_class_words, pmrp_rows
    from mmae.engine.eccv import CocoExtendedMetrics

    data = OmegaConf.load(REPO / "configs" / "data" / "coco.yaml")
    items = retrieval_items(data.annotations_dir, "test")
    ext = CocoExtendedMetrics(items, data.annotations_dir, data.pm_dir)
    pm_gts = Metrics(extra_file_dir=str(ext.pm_dir)).pm_gts
    rows = pmrp_rows(pm_gts, ext.image_ids, ext.caption_ids)
    groups = class_set_groups(pm_gts, ext.image_ids, ext.caption_ids)
    sizes = np.bincount(groups[groups >= 0])
    print("## Class-set groups (identical PM neighbourhoods, as diagnose.py builds them)\n")
    print(f"{len(sizes)} groups over {int((groups >= 0).sum())} images; {int((groups < 0).sum())} images without a group; "
          f"singleton groups {int((sizes == 1).sum())}; images in groups of 2 or more (purity queries) {int(sizes[sizes >= 2].sum())}; "
          f"unordered same-group image pairs {int((sizes * (sizes - 1) // 2).sum())}")
    caveat = [l for l in (REPO / "tests" / "20261003_ml_improve" / "caveats.md").read_text().splitlines() if "zeta <= 2" in l]
    if caveat:
        print(f"Caveat record (tests/20261003_ml_improve/caveats.md): {caveat[0].lstrip('- ')}")
    print()
    words_all = np.array([count_class_words(c) for _, caps in items for c in caps])
    words = words_all[rows["t2i"][0]]
    return words, words_all, len(rows["i2t"][0])


# ----------------------------------------------------------------------------------------------- tables
def table_runs(runs, own):
    print("## T0. Encoded models and consistency with each run's own test (run.json)\n")
    print("| run | model | seed | logit scale | rsum diag | rsum run.json | PMRP per-query (diag) | PMRP package (run.json) | diff |")
    print("|---|---|---|---|---|---|---|---|---|")
    diffs_r, diffs_p = [], []
    for name, r in runs.items():
        o = own.get(name)
        pq = r["pmrp_per_query"]["mean"]
        if o is None:
            z = json.loads(ZERO_SHOT.read_text())
            print(f"| {name} | {r['model']} | - | {r['logit_scale']:.2f} | {r['test']['rsum']:.3f} | {z['rsum']:.3f} | "
                  f"{pq:.3f} | {z['pmrp']:.3f} | {pq - z['pmrp']:+.3f} |")
            continue
        diffs_r.append(abs(r["test"]["rsum"] - o["rsum"]))
        diffs_p.append(pq - o["pmrp"])
        print(f"| {name} | {r['model']} | {r['seed']} | {r['logit_scale']:.2f} | {r['test']['rsum']:.3f} | "
              f"{o['rsum']:.3f} | {pq:.3f} | {o['pmrp']:.3f} | {pq - o['pmrp']:+.3f} |")
    print(f"\nModels encoded: {len(runs)} (baselines {len(own)}); max |rsum diag - run.json| = {max(diffs_r):.3f}; "
          f"per-query PMRP - package PMRP: min {min(diffs_p):+.3f}, max {max(diffs_p):+.3f}")
    print(f"Effect detectable with 80% power, 3 vs 3 runs, two-sided alpha 0.05: {power_effect():.2f} pooled std")
    if (JOB / "status.json").is_file():
        st = json.loads((JOB / "status.json").read_text())
        print(f"Diagnostics job {JOB.name}: {st['state']} on {st['host']}, GPU slot {st['gpu_slots']}, "
              f"{st['duration_s']} s ({st['duration_s'] / 60:.1f} min)")
    record = [l for l in REGISTRY.read_text().splitlines() if "CPU dry run" in l]
    if record:
        text = record[0][record[0].index("CPU dry run"):]
        print(f"\nDry-run record (tests/20261003_ml_improve/runs.md): {text}")
        hit = re.search(r"VWSD Hit@1 ([0-9.]+)", text)
        if hit:
            d = float(hit.group(1)) - runs["zeroshot"]["vwsd"]["vwsd/hit1"]
            print(f"CPU dry-run VWSD Hit@1 - GPU job: {d:+.2f} points = {d * N_VWSD / 100:+.1f} items")
    print()


def table_vwsd(runs):
    print("## T1. VWSD (SemEval-2023 Task 1 English test), Hit@1 and MRR in percent\n")
    zs = runs["zeroshot"]["vwsd"]
    h = by_model(runs, lambda n, r: r["vwsd"]["vwsd/hit1"])
    m = by_model(runs, lambda n, r: r["vwsd"]["vwsd/mrr"])
    print(f"n items: {runs['zeroshot']['vwsd']['vwsd/n']:.0f}; zero-shot Hit@1 {zs['vwsd/hit1']:.2f}, MRR {zs['vwsd/mrr']:.2f}")
    vwsd_dir = Path("/data/SSD/vwsd")  # local copy of the test package (configs/config.yaml, eval.vwsd_dir)
    if vwsd_dir.is_dir():
        from mmae.engine.vwsd import read_vwsd
        items = read_vwsd(vwsd_dir, "en")
        sizes = sorted({len(i.candidates) for i in items})
        print(f"Local VWSD English test: {len(items)} items, candidates per item {sizes}, "
              f"{len({c for i in items for c in i.candidates})} distinct images")
    p = zs["vwsd/hit1"] / 100
    print(f"Binomial standard error of Hit@1 at the zero-shot rate: {100 * math.sqrt(p * (1 - p) / N_VWSD):.2f} points; "
          f"one item = {100 / N_VWSD:.3f} points")
    for rate in (55.0,):
        print(f"Binomial standard error at {rate:.0f}% accuracy: {100 * math.sqrt(rate / 100 * (1 - rate / 100) / N_VWSD):.2f} points")
    print()
    print("| model | Hit@1 | per seed | MRR | Hit@1 - contrastive (p) | MRR - contrastive (p) | Hit@1 - zero-shot |")
    print("|---|---|---|---|---|---|---|")
    for mod in MODELS:
        seeds = ", ".join(f"{v:.2f}" for v in h[mod])
        dh = "" if mod == BASE else diff_cell(h[mod], h[BASE], ("vwsd_hit1", mod))
        dm = "" if mod == BASE else diff_cell(m[mod], m[BASE], ("vwsd_mrr", mod))
        print(f"| {mod} | {ms(h[mod])} | {seeds} | {ms(m[mod])} | {dh} | {dm} | {mean(h[mod]) - zs['vwsd/hit1']:+.2f} |")
    allft = [v for mod in MODELS for v in h[mod]]
    print(f"\nAll 12 fine-tuned runs: Hit@1 {ms(allft)}, range {min(allft):.2f} to {max(allft):.2f}; "
          f"in items: {', '.join(str(round(v * N_VWSD / 100)) for v in sorted(allft))} of {N_VWSD}; "
          f"zero-shot {round(zs['vwsd/hit1'] * N_VWSD / 100)}")
    one = stats.ttest_1samp(h[BASE], zs["vwsd/hit1"])
    print(f"Contrastive minus zero-shot Hit@1: {mean(h[BASE]) - zs['vwsd/hit1']:+.2f}, one-sample t over the 3 seeds "
          f"t = {float(one.statistic):.2f}, p = {float(one.pvalue):.3f} (seed-level only; no item-level test without per-item ranks)\n")


def swap_table(diag, runs, own):
    """Per fusion model and seed: the four conditions' metrics."""
    contrastive = {r["seed"]: name for name, r in runs.items() if r["model"] == BASE}
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # model -> cond -> metric -> [per seed]
    for name, r in sorted(runs.items(), key=lambda kv: (kv[1]["seed"] or 0)):
        if r["model"] not in FUSION:
            continue
        partner = contrastive[r["seed"]]
        conds = {SWAP_CONDS[0]: own[partner], SWAP_CONDS[1]: diag["swaps"][f"{name}__img+contrastive_txt"],
                 SWAP_CONDS[2]: diag["swaps"][f"contrastive_img+{name}__txt"], SWAP_CONDS[3]: own[name]}
        for cond, met in conds.items():
            for k, _ in SWAP_METRICS:
                rows[r["model"]][cond][k].append(met[k])
    return rows


def table_swaps(rows):
    print("## T2. Tower swaps (same seed index; contrastive and full-fusion rows are each run's own test)\n")
    for k, lbl in SWAP_METRICS:
        print(f"### {lbl}\n\n| model | " + " | ".join(SWAP_CONDS) + " |\n|---|---|---|---|---|")
        for mod in FUSION:
            print(f"| {mod} | " + " | ".join(ms(rows[mod][c][k]) for c in SWAP_CONDS) + " |")
        print()
    print("### Differences to the same-seed contrastive run (mean of 3 paired differences; Welch p vs the 3 contrastive runs; "
          "share = the swap's difference / the full model's difference)\n")
    print("| model | metric | fusion img + c txt | c img + fusion txt | full fusion | share img | share txt | sum of swaps |")
    print("|---|---|---|---|---|---|---|---|")
    for mod in FUSION:
        for k, lbl in SWAP_METRICS:
            c = rows[mod][SWAP_CONDS[0]][k]
            cells, ds = [], []
            for cond in SWAP_CONDS[1:]:
                x = rows[mod][cond][k]
                d = mean([a - b for a, b in zip(x, c)])
                _, _, p = welch(x, c)
                if k in ("pmrp", "rsum", "eccv/map_at_r"):
                    ALL_TESTS[("swap", mod, cond, k)] = p
                cells.append(f"{d:+.2f} ({p:.3g})")
                ds.append(d)
            share = (f"{100 * ds[0] / ds[2]:.0f}%", f"{100 * ds[1] / ds[2]:.0f}%") if abs(ds[2]) > 1e-9 else ("", "")
            print(f"| {mod} | {lbl} | " + " | ".join(cells) + f" | {share[0]} | {share[1]} | {ds[0] + ds[1]:+.2f} |")
    print()
    print("### Swap minus full fusion model and swap minus partner, both towers (rsum), per seed\n")
    for mod in FUSION:
        for cond in SWAP_CONDS[1:3]:
            x, f, c = rows[mod][cond]["rsum"], rows[mod][SWAP_CONDS[3]]["rsum"], rows[mod][SWAP_CONDS[0]]["rsum"]
            print(f"- {mod} / {cond}: rsum {', '.join(f'{v:.2f}' for v in x)}; minus own fusion "
                  f"{', '.join(f'{a - b:+.2f}' for a, b in zip(x, f))}; minus contrastive {', '.join(f'{a - b:+.2f}' for a, b in zip(x, c))}")
    pm_img = [v for mod in FUSION for v in rows[mod][SWAP_CONDS[1]]["pmrp"]]
    pm_txt = [v for mod in FUSION for v in rows[mod][SWAP_CONDS[2]]["pmrp"]]
    print(f"\nAll 9 swaps, PMRP: fusion image tower {ms(pm_img)}, fusion text tower {ms(pm_txt)}")
    for k in ("pmrp/i2t", "pmrp/t2i"):
        a = [v for mod in FUSION for v in rows[mod][SWAP_CONDS[1]][k]]
        b = [v for mod in FUSION for v in rows[mod][SWAP_CONDS[2]][k]]
        print(f"All 9 swaps, {k}: fusion image tower {ms(a)}, fusion text tower {ms(b)}")
    z = json.loads(ZERO_SHOT.read_text())
    print("\nFine-tuning gain (contrastive mean - zero-shot): " + "; ".join(
        f"{lbl} {mean(rows['fusion_concat'][SWAP_CONDS[0]][k]) - z[k]:+.2f} (zero-shot {z[k]:.2f})" for k, lbl in SWAP_METRICS))
    print("\nFull models against fusion_none (each run's own test, Welch p):")
    for k, lbl in SWAP_METRICS[:5]:
        none = rows["fusion_none"][SWAP_CONDS[3]][k]
        print(f"- {lbl}: " + "; ".join(f"{m} {diff_cell(rows[m][SWAP_CONDS[3]][k], none)}" for m in ("fusion_concat", "fusion_multilearner")))

    print("\n### Full model minus both swaps (each relative to the same-seed contrastive run): the shortfall of mixed towers\n")
    print("| model | metric | per seed | mean |\n|---|---|---|---|")
    for mod in FUSION:
        for k in ("pmrp", "pmrp/i2t", "pmrp/t2i"):
            c, i, t, f = (rows[mod][cond][k] for cond in SWAP_CONDS)
            vals = [(ff - cc) - (ii - cc) - (tt - cc) for cc, ii, tt, ff in zip(c, i, t, f)]
            print(f"| {mod} | {k} | {', '.join(f'{v:+.3f}' for v in vals)} | {mean(vals):+.2f} |")

    print("\n### Swaps with fusion_none as the control (same contrastive partner towers; Welch p, 3 vs 3)\n")
    print("| model | metric | fusion img swap - none img swap | fusion txt swap - none txt swap | full - full none |")
    print("|---|---|---|---|---|")
    for mod in ("fusion_concat", "fusion_multilearner"):
        for k in ("pmrp", "pmrp/t2i", "pmrp/i2t", "rsum"):
            cells = [diff_cell(rows[mod][cond][k], rows["fusion_none"][cond][k], None, 2) for cond in SWAP_CONDS[1:]]
            print(f"| {mod} | {k} | " + " | ".join(cells) + " |")
    print("\nSum of the two swaps (vs contrastive) as a share of the fusion-specific full gain (full - full fusion_none), PMRP:")
    for mod in ("fusion_concat", "fusion_multilearner"):
        c = rows[mod][SWAP_CONDS[0]]["pmrp"]
        swaps = sum(mean(rows[mod][cond]["pmrp"]) - mean(c) for cond in SWAP_CONDS[1:3])
        spec = mean(rows[mod][SWAP_CONDS[3]]["pmrp"]) - mean(rows["fusion_none"][SWAP_CONDS[3]]["pmrp"])
        img = mean(rows[mod][SWAP_CONDS[1]]["pmrp"]) - mean(rows["fusion_none"][SWAP_CONDS[1]]["pmrp"])
        print(f"- {mod}: swaps sum {swaps:+.3f}, fusion-specific gain {spec:+.3f}, share {100 * swaps / spec:.0f}%; "
              f"image-swap advantage over fusion_none {img:+.3f} = {100 * img / spec:.0f}% of the fusion-specific gain")
    print("\nImage swap minus full model, rsum (mean): " + "; ".join(
        f"{mod} {mean(rows[mod][SWAP_CONDS[1]]['rsum']) - mean(rows[mod][SWAP_CONDS[3]]['rsum']):+.2f}" for mod in FUSION)
        + "; text swap minus full model: " + "; ".join(
        f"{mod} {mean(rows[mod][SWAP_CONDS[2]]['rsum']) - mean(rows[mod][SWAP_CONDS[3]]['rsum']):+.2f}" for mod in FUSION))
    print()


def table_purity(runs):
    print("## T3. Class-set purity of top-10 same-modality neighbours (percent in the query's class-set group)\n")
    zs = runs["zeroshot"]
    print("| model | image-image | caption-caption | i2i - contrastive (p) | t2t - contrastive (p) |\n|---|---|---|---|---|")
    print(f"| zero-shot | {zs['purity_i2i']:.2f} | {zs['purity_t2t']:.2f} | | |")
    i = by_model(runs, lambda n, r: r["purity_i2i"])
    t = by_model(runs, lambda n, r: r["purity_t2t"])
    for mod in MODELS:
        di = "" if mod == BASE else diff_cell(i[mod], i[BASE], ("purity_i2i", mod))
        dt = "" if mod == BASE else diff_cell(t[mod], t[BASE], ("purity_t2t", mod))
        print(f"| {mod} | {ms(i[mod])} | {ms(t[mod])} | {di} | {dt} |")
    print(f"\nFine-tuning change (contrastive - zero-shot): i2i {mean(i[BASE]) - zs['purity_i2i']:+.2f}, "
          f"t2t {mean(t[BASE]) - zs['purity_t2t']:+.2f}")
    for mod in ("fusion_concat", "fusion_multilearner"):
        print(f"{mod} - fusion_none: i2i {diff_cell(i[mod], i['fusion_none'])}, t2t {diff_cell(t[mod], t['fusion_none'])}")
    print()


SIM_COLS = [
    ("pos", "positive", lambda s, r: s["pos"]),
    ("same", "same-class negative", lambda s, r: s["same_class_neg"]),
    ("other", "other negative", lambda s, r: s["other_neg"]),
    ("pos-same", "pos - same", lambda s, r: s["pos"] - s["same_class_neg"]),
    ("same-other", "same - other", lambda s, r: s["same_class_neg"] - s["other_neg"]),
    ("pos-other", "pos - other", lambda s, r: s["pos"] - s["other_neg"]),
    ("rel", "(same - other) / (pos - other)", lambda s, r: (s["same_class_neg"] - s["other_neg"]) / (s["pos"] - s["other_neg"])),
    ("logit_pos-same", "scale x (pos - same)", lambda s, r: r["logit_scale"] * (s["pos"] - s["same_class_neg"])),
    ("logit_same-other", "scale x (same - other)", lambda s, r: r["logit_scale"] * (s["same_class_neg"] - s["other_neg"])),
]


def table_similarity(runs):
    print("## T4. Mean image-caption cosine similarity (COCO 5k test, 25,000 x 5,000 pairs) and gaps\n")
    zs = runs["zeroshot"]
    vals = {key: by_model(runs, lambda n, r, f=f: f(r["similarity"], r)) for key, _, f in SIM_COLS}
    print("| model | " + " | ".join(lbl for _, lbl, _ in SIM_COLS) + " |\n|---|" + "---|" * len(SIM_COLS))
    print("| zero-shot | " + " | ".join(f"{f(zs['similarity'], zs):.4f}" for _, _, f in SIM_COLS) + " |")
    for mod in MODELS:
        print(f"| {mod} | " + " | ".join(ms(vals[key][mod], 4) for key, _, _ in SIM_COLS) + " |")
    print("\n### Differences to contrastive (Welch p)\n")
    print("| model | " + " | ".join(lbl for _, lbl, _ in SIM_COLS) + " |\n|---|" + "---|" * len(SIM_COLS))
    for mod in FUSION:
        print(f"| {mod} | " + " | ".join(diff_cell(vals[key][mod], vals[key][BASE], ("sim", key, mod), 4)
                                         for key, _, _ in SIM_COLS) + " |")
    print("\n### Fusion models against fusion_none (Welch p)\n")
    for mod in ("fusion_concat", "fusion_multilearner"):
        print(f"- {mod}: " + "; ".join(f"{lbl} {diff_cell(vals[key][mod], vals[key]['fusion_none'], None, 4)}"
                                       for key, lbl, _ in SIM_COLS))
    print(f"\nFine-tuning change (contrastive - zero-shot): " + "; ".join(
        f"{lbl} {mean(vals[key][BASE]) - f(zs['similarity'], zs):+.4f}" for key, lbl, f in SIM_COLS))
    print("\nRelative change to contrastive (percent of the contrastive mean), and the share of the fine-tuning change "
          "(zero-shot -> contrastive) that each model reverses:")
    for mod in FUSION:
        rel = "; ".join(f"{lbl} {100 * (mean(vals[key][mod]) / mean(vals[key][BASE]) - 1):+.1f}%" for key, lbl, _ in SIM_COLS[:7])
        rev = "; ".join(f"{lbl} {100 * (mean(vals[key][mod]) - mean(vals[key][BASE])) / (f(zs['similarity'], zs) - mean(vals[key][BASE])):.1f}%"
                        for key, lbl, f in SIM_COLS[3:7])
        print(f"- {mod}: {rel}\n  reversed: {rev}")
    print()
    return vals


def table_logit(runs):
    print("## T5. Learned logit scale (exp of the parameter; clamped at 100)\n")
    v = by_model(runs, lambda n, r: r["logit_scale"])
    print("| model | logit scale | per seed | - contrastive (p) |\n|---|---|---|---|")
    print(f"| zero-shot | {runs['zeroshot']['logit_scale']:.2f} | | |")
    for mod in MODELS:
        d = "" if mod == BASE else diff_cell(v[mod], v[BASE], ("logit", mod))
        print(f"| {mod} | {ms(v[mod])} | {', '.join(f'{x:.2f}' for x in v[mod])} | {d} |")
    print()


def table_words(runs, words, words_all, n_i2t):
    print("## T6. PMRP t2i by the number of COCO classes the caption names\n")
    counts = {b: int(((words == int(b)) if b != "3+" else (words >= 3)).sum()) for b in WORD_BINS}
    counts_all = {b: int(((words_all == int(b)) if b != "3+" else (words_all >= 3)).sum()) for b in WORD_BINS}
    total = sum(counts.values())
    print(f"t2i queries: {total} ({', '.join(f'{b}: {counts[b]} ({100 * counts[b] / total:.1f}%)' for b in WORD_BINS)}); "
          f"all {len(words_all)} test captions: {', '.join(f'{b}: {counts_all[b]}' for b in WORD_BINS)}; i2t queries: {n_i2t}")
    # re-derive the per-bin means from per_query.pt and compare with diagnostics.json
    pq = torch.load(DIAG / "per_query.pt")
    worst = 0.0
    for name, r in runs.items():
        t2i = pq[name]["t2i"].numpy()
        for b in WORD_BINS:
            sel = (words == int(b)) if b != "3+" else (words >= 3)
            worst = max(worst, abs(100 * t2i[sel].mean() - r["pmrp_t2i_by_class_words"][b]))
        worst = max(worst, abs(100 * t2i.mean() - r["pmrp_per_query"]["t2i"]))
    print(f"Max |per-bin mean re-derived from per_query.pt - diagnostics.json|: {worst:.2e}\n")
    zs = runs["zeroshot"]["pmrp_t2i_by_class_words"]
    v = {b: by_model(runs, lambda n, r, b=b: r["pmrp_t2i_by_class_words"][b]) for b in WORD_BINS}
    allq = by_model(runs, lambda n, r: r["pmrp_per_query"]["t2i"])
    print("| model | " + " | ".join(f"{b} classes (n = {counts[b]})" for b in WORD_BINS) + " | all t2i |\n|---|---|---|---|---|---|")
    print("| zero-shot | " + " | ".join(f"{zs[b]:.2f}" for b in WORD_BINS) + f" | {runs['zeroshot']['pmrp_per_query']['t2i']:.2f} |")
    for mod in MODELS:
        print(f"| {mod} | " + " | ".join(ms(v[b][mod]) for b in WORD_BINS) + f" | {ms(allq[mod])} |")
    print("\n### Difference to contrastive (Welch p), and each bin's share of the overall t2i gain (n_bin x diff / sum)\n")
    print("| model | " + " | ".join(f"{b}" for b in WORD_BINS) + " | all t2i | " + " | ".join(f"share {b}" for b in WORD_BINS) + " |")
    print("|---|" + "---|" * (2 * len(WORD_BINS) + 1))
    for mod in FUSION:
        d = {b: mean(v[b][mod]) - mean(v[b][BASE]) for b in WORD_BINS}
        contrib = {b: counts[b] * d[b] for b in WORD_BINS}
        tot = sum(contrib.values())
        print(f"| {mod} | " + " | ".join(diff_cell(v[b][mod], v[b][BASE], ("words", b, mod)) for b in WORD_BINS)
              + f" | {diff_cell(allq[mod], allq[BASE])} | " + " | ".join(f"{100 * contrib[b] / tot:.1f}%" for b in WORD_BINS) + " |")
    print(f"\nFine-tuning change per bin (contrastive - zero-shot): "
          + ", ".join(f"{b}: {mean(v[b][BASE]) - zs[b]:+.2f}" for b in WORD_BINS))
    print("Query share per bin, for comparison with the gain shares: "
          + ", ".join(f"{b}: {100 * counts[b] / total:.1f}%" for b in WORD_BINS))
    m0, m12 = words == 0, (words == 1) | (words == 2)
    inter = by_model(runs, lambda n, r: 100 * (pq[n]["t2i"].numpy()[m0].mean() - pq[n]["t2i"].numpy()[m12].mean()))
    print("Bin 0 minus bins 1-2 (query-weighted), difference to contrastive with Welch 95% CI (does the gain differ on "
          "captions naming no class?):")
    for mod in FUSION:
        d, lo, hi = welch_ci(inter[mod], inter[BASE])
        print(f"- {mod}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}], p {welch(inter[mod], inter[BASE])[2]:.3f}")
    for mod in ("fusion_concat", "fusion_multilearner"):
        print(f"{mod} - fusion_none: " + ", ".join(f"{b}: {diff_cell(v[b][mod], v[b]['fusion_none'])}" for b in WORD_BINS))
    print()
    return v, counts


PROBE = [("full", "full caption"), ("content1", "-1 content word"), ("content2", "-2 content words"),
         ("stop1", "-1 stop word"), ("stop2", "-2 stop words")]


def table_probe(runs):
    print("## T7. Masked-caption probe: first caption of each test image, k content or stop words deleted\n")
    zs = runs["zeroshot"]["probe"]
    print("n per condition (zero-shot run): " + ", ".join(f"{k}: {zs[k + '/n']:.0f}" for k, _ in PROBE[1:]))
    r1 = {k: by_model(runs, lambda n, r, k=k: r["probe"][f"{k}/t2i_R1"]) for k, _ in PROBE}
    print("\n### t2i R@1 among the 5,000 test images\n")
    print("| model | " + " | ".join(lbl for _, lbl in PROBE) + " |\n|---|" + "---|" * len(PROBE))
    print("| zero-shot | " + " | ".join(f"{zs[k + '/t2i_R1']:.2f}" for k, _ in PROBE) + " |")
    for mod in MODELS:
        print(f"| {mod} | " + " | ".join(ms(r1[k][mod]) for k, _ in PROBE) + " |")
    print("\n### R@1 drop from the full caption (full - shortened), and relative drop in percent of the full R@1\n")
    drop = {k: by_model(runs, lambda n, r, k=k: r["probe"]["full/t2i_R1"] - r["probe"][f"{k}/t2i_R1"]) for k, _ in PROBE[1:]}
    rel = {k: by_model(runs, lambda n, r, k=k: 100 * (r["probe"]["full/t2i_R1"] - r["probe"][f"{k}/t2i_R1"]) / r["probe"]["full/t2i_R1"])
           for k, _ in PROBE[1:]}
    print("| model | " + " | ".join(f"drop {lbl}" for _, lbl in PROBE[1:]) + " | " + " | ".join(f"rel. {lbl}" for _, lbl in PROBE[1:]) + " |")
    print("|---|" + "---|" * (2 * len(PROBE[1:])))
    print("| zero-shot | " + " | ".join(f"{zs['full/t2i_R1'] - zs[k + '/t2i_R1']:.2f}" for k, _ in PROBE[1:]) + " | "
          + " | ".join(f"{100 * (zs['full/t2i_R1'] - zs[k + '/t2i_R1']) / zs['full/t2i_R1']:.1f}%" for k, _ in PROBE[1:]) + " |")
    for mod in MODELS:
        print(f"| {mod} | " + " | ".join(ms(drop[k][mod]) for k, _ in PROBE[1:]) + " | "
              + " | ".join(f"{mean(rel[k][mod]):.1f}%" for k, _ in PROBE[1:]) + " |")
    print("\n### Drop difference to contrastive (Welch p); positive = the model loses more R@1 than contrastive\n")
    print("| model | " + " | ".join(f"{lbl}" for _, lbl in PROBE[1:]) + " |\n|---|" + "---|" * len(PROBE[1:]))
    for mod in FUSION:
        print(f"| {mod} | " + " | ".join(diff_cell(drop[k][mod], drop[k][BASE], ("probe_drop", k, mod)) for k, _ in PROBE[1:]) + " |")
    print("\n### Mean change in cosine to the own image (shortened - full), and as a share of the model's mean positive cosine\n")
    dc = {k: by_model(runs, lambda n, r, k=k: r["probe"][f"{k}/delta_cos"]) for k, _ in PROBE[1:]}
    dcr = {k: by_model(runs, lambda n, r, k=k: 100 * r["probe"][f"{k}/delta_cos"] / r["similarity"]["pos"]) for k, _ in PROBE[1:]}
    print("| model | " + " | ".join(f"dcos {lbl}" for _, lbl in PROBE[1:]) + " | " + " | ".join(f"% of pos {lbl}" for _, lbl in PROBE[1:]) + " |")
    print("|---|" + "---|" * (2 * len(PROBE[1:])))
    zr = runs["zeroshot"]
    print("| zero-shot | " + " | ".join(f"{zs[k + '/delta_cos']:+.4f}" for k, _ in PROBE[1:]) + " | "
          + " | ".join(f"{100 * zs[k + '/delta_cos'] / zr['similarity']['pos']:+.1f}%" for k, _ in PROBE[1:]) + " |")
    for mod in MODELS:
        print(f"| {mod} | " + " | ".join(ms(dc[k][mod], 4) for k, _ in PROBE[1:]) + " | "
              + " | ".join(f"{mean(dcr[k][mod]):+.1f}%" for k, _ in PROBE[1:]) + " |")
    print("\n### delta_cos difference to contrastive (Welch p), and |delta_cos| relative to contrastive's\n")
    for mod in FUSION:
        print(f"- {mod}: " + "; ".join(f"{lbl} {diff_cell(dc[k][mod], dc[k][BASE], ('probe_dcos', k, mod), 4)}" for k, lbl in PROBE[1:]))
        print(f"  relative |delta_cos|: " + "; ".join(f"{lbl} {100 * (mean(dc[k][mod]) / mean(dc[k][BASE]) - 1):+.0f}%" for k, lbl in PROBE[1:3]))
    print()


def table_holm():
    adj = holm(ALL_TESTS)
    n = len(ALL_TESTS)
    sig = sorted((k for k, p in ALL_TESTS.items() if p < 0.05), key=ALL_TESTS.get)
    print(f"## T8. Multiple testing: {n} Welch tests against contrastive above; {len(sig)} with p < 0.05 "
          f"(about {0.05 * n:.1f} expected by chance); Holm-adjusted p < 0.05: {sum(p < 0.05 for p in adj.values())}\n")
    for k in sig:
        print(f"- {k}: p {ALL_TESTS[k]:.4g}, Holm {adj[k]:.4g}")
    print()


# ---------------------------------------------------------------------------------------------- figures
def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_similarity(runs, vals):
    order = ["zeroshot"] + MODELS
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2), gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})
    ax = axes[0]
    for y, mod in enumerate(order):
        if mod == "zeroshot":
            s = runs["zeroshot"]["similarity"]
            pts = [s["other_neg"], s["same_class_neg"], s["pos"]]
        else:
            pts = [mean(vals["other"][mod]), mean(vals["same"][mod]), mean(vals["pos"][mod])]
        ax.plot([pts[0], pts[2]], [y, y], color=COLORS[mod], lw=2, solid_capstyle="round", zorder=1)
        for x, mk in zip(pts, ["o", "s", "D"]):
            ax.scatter(x, y, s=46, marker=mk, color=COLORS[mod], edgecolor=SURFACE, linewidth=1.5, zorder=2)
    ax.set_yticks(range(len(order)), [SHORT[m] for m in order], color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean image-caption cosine", color=MUTED, fontsize=8)
    ax.set_title("(a) other neg. ○, same-class neg. ■, positive ◆", fontsize=9, color=INK, loc="left")
    style(ax)
    panels = [("pos-same", "(b) positive - same-class negative"), ("same-other", "(c) same-class - other negative"),
              ("rel", "(d) (same - other) / (pos - other)")]
    for ax, (key, title) in zip(axes[1:], panels):
        zf = [f for k, _, f in SIM_COLS if k == key][0]
        zv = zf(runs["zeroshot"]["similarity"], runs["zeroshot"])
        for y, mod in enumerate(order):
            if mod == "zeroshot":
                ax.scatter(zv, y, s=46, marker="D", color=COLORS[mod], edgecolor=SURFACE, linewidth=1.5, zorder=2)
                continue
            xs = vals[key][mod]
            ax.scatter(xs, [y] * len(xs), s=30, color=COLORS[mod], alpha=0.55, edgecolor=SURFACE, linewidth=1, zorder=2)
            m, s = mean(xs), sd(xs)
            ax.plot([m - s, m + s], [y, y], color=COLORS[mod], lw=2, zorder=1)
            ax.plot([m, m], [y - 0.25, y + 0.25], color=INK, lw=1.5, zorder=3)
        ax.axvline(mean(vals[key][BASE]), color=COLORS[BASE], lw=1, zorder=0)
        ax.set_yticks(range(len(order)), [""] * len(order))
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "similarity.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_swaps(rows):
    metrics = [("pmrp", "PMRP"), ("rsum", "rsum"), ("eccv/map_at_r", "ECCV mAP@R")]
    xlabels = ["contrastive\n(both)", "fusion img\n+ contr. txt", "contr. img\n+ fusion txt", "fusion\n(both)"]
    fig, axes = plt.subplots(len(metrics), len(FUSION), figsize=(11, 8.2), sharey="row")
    for i, (k, lbl) in enumerate(metrics):
        for j, mod in enumerate(FUSION):
            ax = axes[i, j]
            series = [rows[mod][c][k] for c in SWAP_CONDS]
            for s in range(len(series[0])):
                ax.plot(range(4), [series[c][s] for c in range(4)], color=GRID, lw=1.2, zorder=1)
            for c in range(4):
                col = COLORS[BASE] if c == 0 else COLORS[mod]
                ax.scatter([c] * len(series[c]), series[c], s=34, color=col, edgecolor=SURFACE, linewidth=1.5, zorder=2,
                           alpha=0.9 if c in (0, 3) else 0.6)
                m = mean(series[c])
                ax.plot([c - 0.22, c + 0.22], [m, m], color=INK, lw=1.8, zorder=3)
            ax.axhline(mean(series[0]), color=COLORS[BASE], lw=1, zorder=0)
            ax.set_xticks(range(4), xlabels if i == len(metrics) - 1 else [""] * 4, fontsize=8, color=INK)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=8, length=0)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(mod, fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(lbl, fontsize=9, color=INK)
    handles = [Line2D([], [], marker="o", ls="", color=COLORS[BASE], label="contrastive tower(s)"),
               *[Line2D([], [], marker="o", ls="", color=COLORS[m], label=f"{SHORT[m]} tower(s)") for m in FUSION],
               Line2D([], [], color=INK, lw=1.8, label="mean of 3 seeds"),
               Line2D([], [], color=GRID, lw=1.2, label="same seed index")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "swaps.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_words(v, counts):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    width = 0.24
    for j, mod in enumerate(FUSION):
        for i, b in enumerate(WORD_BINS):
            diffs = [x - mean(v[b][BASE]) for x in v[b][mod]]
            x0 = i + (j - 1) * width
            ax.bar(x0, mean(diffs), width=width - 0.03, color=COLORS[mod], alpha=0.85, zorder=2,
                   label=SHORT[mod] if i == 0 else None)
            ax.scatter([x0] * len(diffs), diffs, s=12, color=INK, zorder=3, alpha=0.7)
    ax.axhline(0, color=MUTED, lw=1)
    names = {"0": "no class named", "1": "1 class named", "2": "2 classes named", "3+": "3+ classes named"}
    ax.set_xticks(range(len(WORD_BINS)), [f"{names[b]}\n(n = {counts[b]:,})" for b in WORD_BINS], fontsize=8, color=INK)
    ax.set_ylabel("PMRP t2i - contrastive mean", fontsize=8, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "pmrp_by_words.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def probe_subset_offset(runs):
    """R@1 of the stored first-caption embeddings over all 5,000 images and over the captions that have >= 1 or >= 2
    alphabetic stop words (the stop-word probe's subsets). The stored embeddings are of the unnormalised captions in
    fp16, so this estimates, not reproduces, how much the subset alone moves the probe's full-caption R@1."""
    from mmae.data.coco import retrieval_items
    from mmae.data.stopwords import is_content_word
    from mmae.engine.diagnostics import normalise_caption
    from omegaconf import OmegaConf

    data = OmegaConf.load(REPO / "configs" / "data" / "coco.yaml")
    first = [normalise_caption(caps[0]) for _, caps in retrieval_items(data.annotations_dir, "test")]
    n_stop = np.array([sum(1 for w in t.split() if not is_content_word(w) and w.isalpha()) for t in first])
    subsets = {"stop1": n_stop >= 1, "stop2": n_stop >= 2}
    print("## T7b. Stop-word probe subsets: R@1 of the subset minus R@1 of all first captions (stored fp16 embeddings)\n")
    print("Subset sizes: " + ", ".join(f"{k} {int(v.sum())}" for k, v in subsets.items()))
    offs = defaultdict(lambda: defaultdict(list))
    for name, r in runs.items():
        e = torch.load(DIAG / "embeddings" / f"{name}.pt")
        hit = ((e["caption"][:, 0].float() @ e["image"].float().T).argmax(1) == torch.arange(len(first))).numpy()
        for k, sel in subsets.items():
            offs[r["model"]][k].append(100 * (hit[sel].mean() - hit.mean()))
    for mod in ["zeroshot"] + MODELS:
        print(f"- {mod}: " + "; ".join(f"{k} {mean(offs[mod][k]):+.2f}" for k in subsets))
    print()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    diag, runs, own = load()
    table_runs(runs, own)
    table_vwsd(runs)
    rows = swap_table(diag, runs, own)
    table_swaps(rows)
    table_purity(runs)
    vals = table_similarity(runs)
    table_logit(runs)
    words, words_all, n_i2t = word_bin_counts()
    v, counts = table_words(runs, words, words_all, n_i2t)
    table_probe(runs)
    probe_subset_offset(runs)
    table_holm()
    fig_similarity(runs, vals)
    fig_swaps(rows)
    fig_words(v, counts)
    print(f"Figures written to {OUT.relative_to(REPO)}/: similarity.png, swaps.png, pmrp_by_words.png")


if __name__ == "__main__":
    main()
