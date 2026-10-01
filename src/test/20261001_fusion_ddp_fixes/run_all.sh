#!/usr/bin/env bash
# Reproduces the runs cited in 20261001_fusion_ddp_fixes_log.md (MultiMAE conda env); row C3's
# double count needs eval_check.py instead, see the log.
# Baseline = `git archive 27f7e44 | tar -x -C $S/baseline` (last commit before the 2026-10-01 fixes;
# built and fingerprint-checked by make_trees.py, which section C runs; sections A, B, D, E, F, G
# also need the baseline, so run `python make_trees.py` first on a fresh work dir).
# Ablation trees = fixed src with exactly one fix reverted (see the log).
# Usage: [HARNESS_DIR=<work dir>] [HARNESS_OUT=<results dir>] bash run_all.sh [section...]
#   sections: A B C D E F G H (default: all)
#   HARNESS_DIR: baseline/ablation trees and checkpoints (default: <this dir>/out/work, gitignored)
#   HARNESS_OUT: JSON/log results (default: <this dir>/out, gitignored)
set -uo pipefail

D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=$(cd "$D/../../.." && pwd)
S=${HARNESS_DIR:-$D/out/work}
O=${HARNESS_OUT:-$D/out}
export HARNESS_DIR=$S HARNESS_OUT=$O
mkdir -p "$S"
BIN=/root/miniconda3/envs/MultiMAE/bin
mkdir -p "$O"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

FIX="env PYTHONPATH=$R"
BASE="env PYTHONPATH=$S/baseline"
CPU2="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 $BIN/torchrun --nproc_per_node 2"
CPU1="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 $BIN/python"
GPU="$BIN/python"
H=$D/harness.py
E=$D/eval_check.py

run() {
  local name=$1; shift
  "$@" > "$O/$name.log" 2>&1
  local rc=$?
  echo "$name exit=$rc" | tee -a "$O/exit_codes.txt"
}

SECTIONS=${*:-A B C D E F G H}

# Build/verify the pinned baseline (and ablation trees) up front; fails loudly if it is not pre-fix code.
$BIN/python $D/make_trees.py || { echo "make_trees.py failed" >&2; exit 2; }

# A. single GPU, before vs after, same seed (both hooks); baseline twice for repeatability
if [[ $SECTIONS == *A* ]]; then
  run base_gpu_multi_r1  $BASE $GPU $H --hook multi --device gpu --expect-src $S/baseline/src --out $O/base_gpu_multi_r1.json
  run base_gpu_multi_r2  $BASE $GPU $H --hook multi --device gpu --expect-src $S/baseline/src --out $O/base_gpu_multi_r2.json
  run fixed_gpu_multi    $FIX  $GPU $H --hook multi --device gpu --expect-src $R/src --out $O/fixed_gpu_multi.json
  run base_gpu_plain     $BASE $GPU $H --hook plain --device gpu --expect-src $S/baseline/src --out $O/base_gpu_plain.json
  run fixed_gpu_plain    $FIX  $GPU $H --hook plain --device gpu --expect-src $R/src --out $O/fixed_gpu_plain.json
fi

# B. best-weights restore: min_delta=1e9 makes epoch 1 the only "improvement", so best != last
if [[ $SECTIONS == *B* ]]; then
  run base_gpu_multi_best1   $BASE $GPU $H --hook multi --device gpu --min-delta 1e9 --expect-src $S/baseline/src --out $O/base_gpu_multi_best1.json
  run fixed_gpu_multi_best1  $FIX  $GPU $H --hook multi --device gpu --min-delta 1e9 --expect-src $R/src --out $O/fixed_gpu_multi_best1.json
  run base_gpu_plain_best1   $BASE $GPU $H --hook plain --device gpu --min-delta 1e9 --expect-src $S/baseline/src --out $O/base_gpu_plain_best1.json
  run fixed_gpu_plain_best1  $FIX  $GPU $H --hook plain --device gpu --min-delta 1e9 --expect-src $R/src --out $O/fixed_gpu_plain_best1.json
fi

# C. two processes on CPU (gloo) through the real hooks
if [[ $SECTIONS == *C* ]]; then
  run base_cpu2_multi   $BASE $CPU2 --master_port 29621 $H --hook multi --device cpu --expect-src $S/baseline/src --out $O/base_cpu2_multi.json
  run base_cpu2_plain   $BASE $CPU2 --master_port 29622 $H --hook plain --device cpu --expect-src $S/baseline/src --out $O/base_cpu2_plain.json
  run fixed_cpu2_multi  $FIX  $CPU2 --master_port 29623 $H --hook multi --device cpu --save-dir $S/ckpt_fixed_cpu2 --expect-src $R/src --out $O/fixed_cpu2_multi.json
  run fixed_cpu2_plain  $FIX  $CPU2 --master_port 29624 $H --hook plain --device cpu --expect-src $R/src --out $O/fixed_cpu2_plain.json
  run fixed_cpu2_multi_best1 $FIX $CPU2 --master_port 29625 $H --hook multi --device cpu --min-delta 1e9 --expect-src $R/src --out $O/fixed_cpu2_multi_best1.json
  # one fix reverted at a time
  run ablate_nofreeze_cpu2_multi env PYTHONPATH=$S/ablate_nofreeze TORCH_DISTRIBUTED_DEBUG=INFO $CPU2 --master_port 29626 $H --hook multi --device cpu --expect-src $S/ablate_nofreeze/src --out $O/ablate_nofreeze_cpu2_multi.json
  run ablate_noprep_test_cpu2_multi env PYTHONPATH=$S/ablate_noprep_test $CPU2 --master_port 29627 $H --hook multi --device cpu --expect-src $S/ablate_noprep_test/src --out $O/ablate_noprep_test_cpu2_multi.json
fi

# D. evalrank on a fixed model: 1 vs 2 processes; baseline probes; no-accelerator path before/after
if [[ $SECTIONS == *D* ]]; then
  run eval_fixed_cpu1        $FIX  $CPU1 $E --mode accel --device cpu --expect-src $R/src --out $O/eval_fixed_cpu1.json
  run eval_fixed_cpu2        $FIX  $CPU2 --master_port 29631 $E --mode accel --device cpu --expect-src $R/src --out $O/eval_fixed_cpu2.json
  run eval_base_cpu1_unwrap_cuda $BASE $CPU1 $E --mode accel --device cpu --variant unwrap_cuda --expect-src $S/baseline/src --out $O/eval_base_cpu1_unwrap_cuda.json
  run eval_base_cpu2_as_is   $BASE $CPU2 --master_port 29632 $E --mode accel --device cpu --variant as_is --expect-src $S/baseline/src --out $O/eval_base_cpu2_as_is.json
  run eval_base_cpu2_unwrap  $BASE $CPU2 --master_port 29633 $E --mode accel --device cpu --variant unwrap --expect-src $S/baseline/src --out $O/eval_base_cpu2_unwrap.json
  run eval_base_cpu2_unwrap_cuda $BASE $CPU2 --master_port 29634 $E --mode accel --device cpu --variant unwrap_cuda --expect-src $S/baseline/src --out $O/eval_base_cpu2_unwrap_cuda.json
  # same, with pretrained CLIP features so retrieval has real signal
  run eval_fixed_cpu1_clip   $FIX  $CPU1 $E --mode accel --device cpu --model clip --expect-src $R/src --out $O/eval_fixed_cpu1_clip.json
  run eval_fixed_cpu2_clip   $FIX  $CPU2 --master_port 29635 $E --mode accel --device cpu --model clip --expect-src $R/src --out $O/eval_fixed_cpu2_clip.json
  run eval_base_cpu1_unwrap_cuda_clip $BASE $CPU1 $E --mode accel --device cpu --model clip --variant unwrap_cuda --expect-src $S/baseline/src --out $O/eval_base_cpu1_unwrap_cuda_clip.json
  run eval_base_cpu2_unwrap_cuda_clip $BASE $CPU2 --master_port 29636 $E --mode accel --device cpu --model clip --variant unwrap_cuda --expect-src $S/baseline/src --out $O/eval_base_cpu2_unwrap_cuda_clip.json
  run eval_noaccel_base_gpu  $BASE $GPU $E --mode noaccel --device gpu --expect-src $S/baseline/src --out $O/eval_noaccel_base_gpu.json
  run eval_noaccel_fixed_gpu $FIX  $GPU $E --mode noaccel --device gpu --expect-src $R/src --out $O/eval_noaccel_fixed_gpu.json
fi

# E. no-accelerator path at the scale of src/test/test_eval_fusionmmae.py (full 5000-image test split,
#    batch 256). The script itself cannot build its model at HEAD (pre-existing TypeError in
#    MultiModalFusionMAE), so it is run only to record that crash in both trees; -P keeps the cwd
#    off sys.path so the baseline run really imports the baseline src.
if [[ $SECTIONS == *E* ]]; then
  run testeval_script_base  $BASE $GPU -P -c "import runpy, src; print('src:', src.__file__); runpy.run_path('$S/baseline/src/test/test_eval_fusionmmae.py', run_name='__main__')"
  run testeval_script_fixed $FIX  $GPU -P -c "import runpy, src; print('src:', src.__file__); runpy.run_path('$R/src/test/test_eval_fusionmmae.py', run_name='__main__')"
  run eval_noaccel_full_base_gpu  $BASE $GPU $E --mode noaccel --device gpu --model clip --n 0 --bs 256 --num-workers 8 --expect-src $S/baseline/src --out $O/eval_noaccel_full_base_gpu.json
  run eval_noaccel_full_fixed_gpu $FIX  $GPU $E --mode noaccel --device gpu --model clip --n 0 --bs 256 --num-workers 8 --expect-src $R/src --out $O/eval_noaccel_full_fixed_gpu.json
fi

# F. section A again with torch.use_deterministic_algorithms(True), for a bitwise comparison
if [[ $SECTIONS == *F* ]]; then
  DET="--deterministic"
  run base_gpu_multi_det_r1  $BASE $GPU $H --hook multi --device gpu $DET --expect-src $S/baseline/src --out $O/base_gpu_multi_det_r1.json
  run base_gpu_multi_det_r2  $BASE $GPU $H --hook multi --device gpu $DET --expect-src $S/baseline/src --out $O/base_gpu_multi_det_r2.json
  run fixed_gpu_multi_det    $FIX  $GPU $H --hook multi --device gpu $DET --expect-src $R/src --out $O/fixed_gpu_multi_det.json
  run base_gpu_plain_det     $BASE $GPU $H --hook plain --device gpu $DET --expect-src $S/baseline/src --out $O/base_gpu_plain_det.json
  run fixed_gpu_plain_det    $FIX  $GPU $H --hook plain --device gpu $DET --expect-src $R/src --out $O/fixed_gpu_plain_det.json
fi

# G. section B (multi hook) with deterministic kernels, so base and fixed epoch-1 weights can be compared bitwise
if [[ $SECTIONS == *G* ]]; then
  DET="--deterministic"
  run base_gpu_multi_det_best1   $BASE $GPU $H --hook multi --device gpu $DET --min-delta 1e9 --expect-src $S/baseline/src --out $O/base_gpu_multi_det_best1.json
  run fixed_gpu_multi_det_best1  $FIX  $GPU $H --hook multi --device gpu $DET --min-delta 1e9 --expect-src $R/src --out $O/fixed_gpu_multi_det_best1.json
fi

# H. guard path: min_delta=inf makes `val < best - inf` always false, so no epoch is ever "best"
if [[ $SECTIONS == *H* ]]; then
  run fixed_gpu_multi_noimprove $FIX $GPU $H --hook multi --device gpu --min-delta inf --expect-src $R/src --out $O/fixed_gpu_multi_noimprove.json
fi

# Summary of all checks: python $D/compare.py > $D/summary.txt
