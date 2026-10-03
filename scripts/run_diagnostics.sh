#!/usr/bin/env bash
# Stage 0 diagnostics on a cluster node (COCO on the node's local disk), single process.
# Usage: scripts/run_diagnostics.sh +diag.runs_root=<dir> +diag.out=<dir> [eval.vwsd_dir=<dir>] [overrides...]
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/diagnose.py data=coco_cluster "$@"
