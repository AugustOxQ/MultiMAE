#!/usr/bin/env bash
# Train on a cluster node (COCO on the node's local disk), every visible GPU.
# Usage: scripts/run_cluster.sh <batch_size> <epochs> <note> [hydra overrides...]
set -euo pipefail
cd "$(dirname "$0")/.."
if [ $# -lt 3 ]; then
  echo "usage: $0 <batch_size> <epochs> <note> [hydra overrides...]" >&2
  exit 2
fi
BATCH_SIZE=$1 EPOCHS=$2 NOTE=$3
shift 3

if [ -n "${CUDA_VISIBLE_DEVICES-}" ]; then
  IFS=',' read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES}"
  NUM_PROCS=${#gpus[@]}
else
  NUM_PROCS=$(nvidia-smi -L | wc -l | tr -d ' ')
fi
[ "${NUM_PROCS:-0}" -lt 1 ] && NUM_PROCS=1
MULTI=()
[ "${NUM_PROCS}" -gt 1 ] && MULTI=(--multi_gpu)
echo "Launching on ${NUM_PROCS} process(es)"
accelerate launch --num_processes "${NUM_PROCS}" --num_machines 1 --mixed_precision no --dynamo_backend no \
  "${MULTI[@]}" train.py data=coco_cluster "train.batch_size=${BATCH_SIZE}" "train.epochs=${EPOCHS}" \
  "wandb.notes='${NOTE}'" "wandb.tags=[cluster]" "$@"
