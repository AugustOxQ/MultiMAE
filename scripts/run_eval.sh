#!/usr/bin/env bash
# Retrieval evaluation on a cluster node (COCO on the node's local disk), single process.
# Usage: scripts/run_eval.sh [hydra overrides...]
#   zero-shot CLIP:  scripts/run_eval.sh model=fusion_concat eval.output=/local/wding/res/MultiMAE/coco/zeroshot/clip_b32_test.json
#   trained run:     scripts/run_eval.sh eval.run_dir=/local/wding/res/MultiMAE/coco/multimae/default/<run>
set -euo pipefail
cd "$(dirname "$0")/.."
exec python evaluate.py data=coco_cluster "$@"
