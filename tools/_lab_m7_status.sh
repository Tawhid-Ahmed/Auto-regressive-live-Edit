#!/usr/bin/env bash
set -euo pipefail
REPO=/home/user/TWD/LiveEdit_lab_usb/LiveEdit
LOGDIR=$REPO/doc/results/cvpr2027/g5/logs
date
echo "--- procs ---"
ps aux | grep -E 'train_vllm_editor|lab_g5_llava' | grep -v grep || echo '(none)'
echo "--- gpu ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "--- train log ---"
LOG=$(ls -t "$LOGDIR"/lab_g5_llava_L0_bs2*.train.log 2>/dev/null | head -1 || true)
echo "LOG=${LOG:-none}"
if [ -n "${LOG:-}" ] && [ -f "$LOG" ]; then
  tr '\r' '\n' <"$LOG" | grep -E 'Epoch |ema_loss|MoE soft routing relative|Checkpoints dir|Timer|Error|OOM' | tail -n 20
fi
echo "--- outer ---"
OUTER=$(ls -t "$LOGDIR"/m7_outer_*.log 2>/dev/null | head -1 || true)
echo "OUTER=${OUTER:-none}"
if [ -n "${OUTER:-}" ]; then
  tail -n 20 "$OUTER"
fi
echo "--- ckpts ---"
ls -lt "$REPO"/records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_bs2*/checkpoints/epoch-* 2>/dev/null | head -5 || echo '(no ckpt yet)'
