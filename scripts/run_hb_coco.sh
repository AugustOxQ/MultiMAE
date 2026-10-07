#!/usr/bin/env bash
# H-b COCO probes on a node: bash scripts/run_hb_coco.sh d1 --runs <dir>... --out <dir> [hb_coco_d1.py options]
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/hb_coco_"$1".py "${@:2}"
