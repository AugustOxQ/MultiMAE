#!/usr/bin/env bash
# H-b GPU readout stage on a node: bash scripts/run_hb_encode.sh --runs <dir>... --out <dir> [hb_encode.py options]
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/hb_encode.py "$@"
