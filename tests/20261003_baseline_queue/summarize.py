"""Mean and std over seeds per model, COCO 5k test, from res/coco run folders and the zero-shot JSON."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path("/project/MultiAlign/MultiMAE/res/coco")
METRICS = [
    "rsum", "i2t_R1", "t2i_R1", "eccv/map_at_r", "eccv/rprecision", "eccv/r1",
    "cxc/r1", "coco1k/r1", "pmrp", "pmrp/i2t", "pmrp/t2i",
]
ORDER = ["contrastive", "fusion_none", "fusion_concat", "fusion_multilearner"]


def test_metrics(run: dict) -> dict | None:
    if "test" in run.get("eval", {}):  # re-evaluated by evaluate.py (seed-42 fusion runs)
        return run["eval"]["test"]
    test = (run.get("results") or {}).get("test")
    if not test:
        return None
    return {k.removeprefix("test/").removeprefix("retrieval/"): v for k, v in test.items()}


rows = defaultdict(dict)  # model -> seed -> metrics
for run_json in sorted(ROOT.glob("multimae/default/*/run.json")):
    run = json.loads(run_json.read_text())
    cfg = OmegaConf.load(run_json.parent / "config.yaml")
    metrics = test_metrics(run)
    if run.get("status") != "completed" or metrics is None or "pmrp" not in metrics:
        continue
    rows[cfg.model.name][int(cfg.seed)] = metrics

zero_shot = json.loads((ROOT / "zeroshot" / "clip_b32_test.json").read_text())
fmt = lambda xs: f"{statistics.mean(xs):.2f} ± {statistics.stdev(xs):.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"

print("# COCO 5k test, mean ± std over seeds\n")
print("| model | seeds | " + " | ".join(METRICS) + " |")
print("|---|---|" + "---|" * len(METRICS))
print("| zero-shot CLIP B/32 | - | " + " | ".join(f"{zero_shot[m]:.2f}" for m in METRICS) + " |")
for model in ORDER + sorted(set(rows) - set(ORDER)):
    if model not in rows:
        continue
    seeds = sorted(rows[model])
    print(f"| {model} | {','.join(map(str, seeds))} | "
          + " | ".join(fmt([rows[model][s][m] for s in seeds]) for m in METRICS) + " |")

print("\n# Per seed\n")
print("| model | seed | " + " | ".join(METRICS) + " |")
print("|---|---|" + "---|" * len(METRICS))
for model in ORDER + sorted(set(rows) - set(ORDER)):
    for seed in sorted(rows.get(model, {})):
        print(f"| {model} | {seed} | " + " | ".join(f"{rows[model][seed][m]:.2f}" for m in METRICS) + " |")
