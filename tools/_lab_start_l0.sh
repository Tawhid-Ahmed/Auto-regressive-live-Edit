#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit
cd /home/user/TWD/LiveEdit_lab_usb/LiveEdit
mkdir -p doc/results/cvpr2027/g5/logs
LOG=doc/results/cvpr2027/g5/logs/l0_retry.log
: > "$LOG"
nohup python -u tools/lab_g5_llava_fresh_pipeline.py --stages L0 >> "$LOG" 2>&1 &
PID=$!
echo "STARTED_PID=$PID"
echo "LOG=$LOG"
sleep 20
if ps -p "$PID" > /dev/null 2>&1; then
  echo "STATUS=running"
else
  echo "STATUS=dead"
fi
echo "----- log -----"
tail -n 120 "$LOG" || true
