"""Stage 1 summary: every ml_improve arm (completed seeds) vs the 3-seed baselines."""
import json, glob, statistics as st, sys
from scipy.stats import ttest_ind
keys = {"mAP@R": ["test/eccv/map_at_r", "eccv/map_at_r"], "PMRP": ["test/pmrp", "pmrp"], "rsum": ["test/retrieval/rsum", "rsum"], "CxC": ["test/cxc/r1", "cxc/r1"]}
def runs(root, name):
    out = []
    for p in sorted(glob.glob(f"{root}/*_{name}/run.json")):
        rj = json.load(open(p))
        if rj.get("status") != "completed": continue
        t = rj.get("eval", {}).get("test") or rj["results"]["test"]
        out.append({n: next(t[k] for k in ks if k in t) for n, ks in keys.items()})
    return out
base = {m: runs("res/coco/multimae/default", m) for m in ("contrastive", "fusion_multilearner")}
arms = sorted({p.split("/")[-2].split("_", 2)[2] for p in glob.glob("res/coco/multimae/ml_improve/*/run.json")})
print(f"{'arm':28}{'n':>3}" + "".join(f"{k:>16}" for k in keys) + "   ref")
for name, b in (("contrastive (baseline)", base["contrastive"]), ("multilearner (baseline)", base["fusion_multilearner"])):
    print(f"{name:28}{len(b):>3}" + "".join(f"{st.mean(r[k] for r in b):>10.2f}±{st.stdev([r[k] for r in b]):<5.2f}" for k in keys))
for a in arms:
    r = runs("res/coco/multimae/ml_improve", a)
    if not r: continue
    ref = base["contrastive"] if a.removeprefix("s2_").startswith("contrastive") else base["fusion_multilearner"]
    cells = []
    for k in keys:
        v = [x[k] for x in r]; d = st.mean(v) - st.mean(x[k] for x in ref)
        p = ttest_ind(v, [x[k] for x in ref], equal_var=False).pvalue if len(v) > 1 else float("nan")
        cells.append(f"{st.mean(v):>8.2f} {d:+5.2f}" + (f"*" if p < 0.05 else " "))
    print(f"{a:28}{len(r):>3}" + "".join(f"{c:>16}" for c in cells) + ("   contrastive" if a.removeprefix("s2_").startswith("contrastive") else "   multilearner"))
print("(diff vs the reference baseline's 3-seed mean; * Welch p<0.05 when >=2 seeds)")
