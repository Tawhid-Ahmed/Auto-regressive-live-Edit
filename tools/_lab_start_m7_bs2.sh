#!/usr/bin/env bash
set -euo pipefail
REPO=/home/user/TWD/LiveEdit_lab_usb/LiveEdit
cd "$REPO"
LOGDIR=$REPO/doc/results/cvpr2027/g5/logs
mkdir -p "$LOGDIR"
pkill -f lab_g5_llava_m7_bs_pipeline 2>/dev/null || true
sleep 1
rm -f "$LOGDIR/m7_pipeline.lock"
# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit
# bs=2 already proved by smoke; skip re-smoke
nohup env FORCE_BS=2 bash tools/lab_g5_llava_m7_bs_pipeline.sh \
  >> "$LOGDIR/m7_delayed_start.log" 2>&1 &
echo $! > "$LOGDIR/m7_pipeline.lock"
sleep 3
echo "started pid=$(cat "$LOGDIR/m7_pipeline.lock") at $(date -Is)"
ps aux | grep -E 'lab_g5_llava_m7|train_vllm|lab_g5_llava_fresh' | grep -v grep || true
OUTER=$(ls -t "$LOGDIR"/m7_outer_*.log | head -1)
echo "OUTER=$OUTER"
tail -n 25 "$OUTER" || true
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
