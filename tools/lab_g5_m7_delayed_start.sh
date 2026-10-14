#!/usr/bin/env bash
# Cron/one-shot entry: start M7 pipeline if not already running.
set -euo pipefail
REPO=/home/user/TWD/LiveEdit_lab_usb/LiveEdit
cd "$REPO"
LOGDIR=$REPO/doc/results/cvpr2027/g5/logs
mkdir -p "$LOGDIR"
LOCK=$LOGDIR/m7_pipeline.lock
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "M7 already running pid=$(cat "$LOCK")" >> "$LOGDIR/m7_delayed_start.log"
  exit 0
fi
# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit
nohup bash tools/lab_g5_llava_m7_bs_pipeline.sh >> "$LOGDIR/m7_delayed_start.log" 2>&1 &
echo $! > "$LOCK"
echo "started pid=$! at $(date -Is)" >> "$LOGDIR/m7_delayed_start.log"
