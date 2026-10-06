"""Stage 2 summary (spec section 7): each variant vs both seeded-sampler baselines on seeds 42-44.

Success bar: mean ECCV mAP@R above both baselines, two-sided Welch p < 0.05 for each, Holm-corrected over the
variants (per baseline); mean rsum no more than 1.5 below multilearner's and mean PMRP no more than 0.05 below.
Secondary: paired t-test by seed (same data order under the seeded sampler). The spec tests the final candidate over
its 5 seeds (42-46) "if the budget allows", so the success bar is printed twice: on seeds 42-44 for every arm, and
with every completed seed of each variant (baselines keep their 3).
"""
import json, glob, statistics as st
from scipy.stats import ttest_ind, ttest_rel

ROOT = "res/coco/multimae/ml_improve"
SEEDS = (42, 43, 44)
BASELINES = ("s2_contrastive", "s2_multilearner")
VARIANTS = ("s2_txt80", "s2_mae0")
KEYS = {"mAP@R": "test/eccv/map_at_r", "PMRP": "test/pmrp", "rsum": "test/retrieval/rsum", "CxC R@1": "test/cxc/r1",
        "ECCV R-P": "test/eccv/rprecision", "ECCV R@1": "test/eccv/r1", "1K R@1": "test/coco1k/r1"}


def seed_of(run_dir):
    for line in open(f"{run_dir}/config.yaml"):
        if line.startswith("seed:"):
            return int(line.split()[1])


def runs(arm):  # {seed: {metric: value}} over completed runs
    out = {}
    for p in sorted(glob.glob(f"{ROOT}/*_{arm}/run.json")):
        rj = json.load(open(p))
        if rj.get("status") == "completed":
            t = rj["results"]["test"]
            out[seed_of(p.rsplit("/", 1)[0])] = {n: t[k] for n, k in KEYS.items()}
    return out


def holm(pvalues):  # Holm step-down adjusted p-values, same order as given
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


data = {arm: runs(arm) for arm in BASELINES + VARIANTS}
core = {arm: [data[arm][s] for s in SEEDS if s in data[arm]] for arm in data}

print(f"{'arm':18}{'n':>3}" + "".join(f"{k:>15}" for k in KEYS))
for arm in BASELINES + VARIANTS:
    r = core[arm]
    if r:
        print(f"{arm:18}{len(r):>3}" + "".join(
            f"{st.mean(x[k] for x in r):>9.2f}±{(st.stdev([x[k] for x in r]) if len(r) > 1 else 0):<5.2f}" for k in KEYS))
    extra = [data[arm][s] for s in sorted(data[arm]) if s not in SEEDS]
    if extra:
        allr = r + extra
        print(f"{arm + ' (all)':18}{len(allr):>3}" + "".join(f"{st.mean(x[k] for x in allr):>15.2f}" for k in KEYS))

for base in BASELINES:
    print(f"\nvs {base}: diff, Welch p, paired-by-seed p" + " (Holm over variants on mAP@R)")
    ready = [v for v in VARIANTS if len(core[v]) >= 2 and len(core[base]) >= 2]
    welch_map = [ttest_ind([x["mAP@R"] for x in core[v]], [x["mAP@R"] for x in core[base]], equal_var=False).pvalue
                 for v in ready]
    holm_map = dict(zip(ready, holm(welch_map)))
    for v in ready:
        cells = []
        for k in KEYS:
            a, b = [x[k] for x in core[v]], [x[k] for x in core[base]]
            d = st.mean(a) - st.mean(b)
            pw = ttest_ind(a, b, equal_var=False).pvalue
            shared = [s for s in SEEDS if s in data[v] and s in data[base]]
            pp = ttest_rel([data[v][s][k] for s in shared], [data[base][s][k] for s in shared]).pvalue \
                if len(shared) > 1 else float("nan")
            cells.append(f"{k} {d:+.2f} (W {pw:.3f}, P {pp:.3f})")
        print(f"  {v:14} Holm mAP@R p = {holm_map[v]:.3f}; " + "; ".join(cells))

def success_bar(label, variant_runs):
    print(f"\nsuccess bar ({label}):")
    for v in VARIANTS:
        if len(variant_runs[v]) < 2 or any(len(core[b]) < 2 for b in BASELINES):
            print(f"  {v}: not enough seeds yet"); continue
        mean = lambda rows, k: st.mean(x[k] for x in rows)
        beats, pvals = [], []
        for base in BASELINES:
            ready = [w for w in VARIANTS if len(variant_runs[w]) >= 2]
            ps = [ttest_ind([x["mAP@R"] for x in variant_runs[w]], [x["mAP@R"] for x in core[base]],
                            equal_var=False).pvalue for w in ready]
            p_holm = dict(zip(ready, holm(ps)))[v]
            pvals.append(p_holm)
            beats.append(mean(variant_runs[v], "mAP@R") > mean(core[base], "mAP@R") and p_holm < 0.05)
        rsum_ok = mean(variant_runs[v], "rsum") >= mean(core["s2_multilearner"], "rsum") - 1.5
        pmrp_ok = mean(variant_runs[v], "PMRP") >= mean(core["s2_multilearner"], "PMRP") - 0.05
        complete = all(len(variant_runs[w]) >= len(SEEDS) for w in VARIANTS) and \
            all(len(core[b]) == len(SEEDS) for b in BASELINES)
        print(f"  {v} (n={len(variant_runs[v])}): beats contrastive {beats[0]} (Holm p {pvals[0]:.3f}), beats "
              f"multilearner {beats[1]} (Holm p {pvals[1]:.3f}), rsum guard {rsum_ok}, PMRP guard {pmrp_ok} -> "
              f"{'SUCCESS' if all(beats) and rsum_ok and pmrp_ok else 'not met'}"
              + ("" if complete else " (PROVISIONAL: an arm has fewer than 3 seeds, so the Holm family or n is incomplete)"))


success_bar("seeds 42-44", core)
success_bar("every completed seed of each variant", {v: [data[v][s] for s in sorted(data[v])] for v in VARIANTS})

# Mechanism controls at 80% text masking (queued 2026-10-06, not part of the success-bar family): each vs s2_txt80.
CONTROLS = ("s2_m1clean_txt80", "s2_m1detached_txt80", "s2_none_txt80")
ctrl = {arm: runs(arm) for arm in CONTROLS}
ref = {s: data["s2_txt80"][s] for s in SEEDS if s in data["s2_txt80"]}
print("\ncontrols at 80% vs s2_txt80 (seeds 42-44): diff, Welch p, paired-by-seed p")
for arm in CONTROLS:
    rows = [ctrl[arm][s] for s in SEEDS if s in ctrl[arm]]
    if not rows:
        print(f"  {arm}: no completed seeds yet"); continue
    cells = []
    for k in ("mAP@R", "PMRP", "rsum", "CxC R@1"):
        a, b = [x[k] for x in rows], [x[k] for x in ref.values()]
        pw = ttest_ind(a, b, equal_var=False).pvalue if len(a) > 1 else float("nan")
        shared = [s for s in SEEDS if s in ctrl[arm] and s in ref]
        pp = ttest_rel([ctrl[arm][s][k] for s in shared], [ref[s][k] for s in shared]).pvalue if len(shared) > 1 else float("nan")
        cells.append(f"{k} {st.mean(a):.2f} ({st.mean(a) - st.mean(b):+.2f}; W {pw:.3f}, P {pp:.3f})")
    print(f"  {arm:20} n={len(rows)}: " + "; ".join(cells))
