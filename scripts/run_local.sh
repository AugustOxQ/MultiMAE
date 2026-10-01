#!/usr/bin/env bash
# Train on this machine with every visible GPU.
# Usage: scripts/run_local.sh [note] [hydra overrides...]
#   scripts/run_local.sh "first try" model=fusion_multilearner train.epochs=20
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v nvidia-smi >/dev/null 2>&1; then
  if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
    IFS=',' read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES}"
    NUM_PROCS=${#gpus[@]}
  else
    NUM_PROCS=$(nvidia-smi -L | wc -l | tr -d ' ')
  fi
else
  NUM_PROCS=1
fi
[ "${NUM_PROCS:-0}" -lt 1 ] && NUM_PROCS=1

NOTE=${1:-local}
shift || true
# The note reaches the config through the environment, so any text (quotes, commas, colons, '=', '${')
# arrives verbatim instead of going through Hydra's override parser.
export MMAE_NOTE="${NOTE}"
MULTI=()
[ "${NUM_PROCS}" -gt 1 ] && MULTI=(--multi_gpu)
echo "Launching on ${NUM_PROCS} process(es)"
accelerate launch --num_processes "${NUM_PROCS}" --num_machines 1 --mixed_precision no --dynamo_backend no \
  "${MULTI[@]}" train.py 'wandb.notes=${oc.env:MMAE_NOTE}' "wandb.tags=[local]" "$@"
