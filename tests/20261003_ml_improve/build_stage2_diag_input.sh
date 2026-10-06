#!/usr/bin/env bash
# Hard-link every completed Stage 2 run (seeds 42-44) into res/coco/stage2_runs/<run>/ (config.yaml, run.json,
# checkpoints/best.pt): the input folder of the Stage 2 diagnostics job. Hard links cost no disk; the cluster tool
# copies the folder to the node through its DATA_MAP entry (/local/wding/Dataset/mmae_stage2_runs). Safe to rerun.
set -eu
cd /project/MultiAlign/MultiMAE
OUT=res/coco/stage2_runs
mkdir -p "$OUT"
for d in res/coco/multimae/ml_improve/*_s2_*; do
  grep -q '"status": "completed"' "$d/run.json" 2>/dev/null || continue
  seed=$(awk '/^seed:/ {print $2}' "$d/config.yaml")
  case $seed in 42|43|44) ;; *) continue ;; esac
  name=$(basename "$d")
  mkdir -p "$OUT/$name/checkpoints"
  for f in config.yaml run.json checkpoints/best.pt; do ln -f "$d/$f" "$OUT/$name/$f"; done
done
ls "$OUT"
