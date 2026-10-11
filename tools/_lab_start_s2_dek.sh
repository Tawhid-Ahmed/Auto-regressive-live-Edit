#!/usr/bin/env bash
# Detached S2 -> DEK from strong lab L0 (CVPR G5). Survives Cursor/SSH close.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit
cd /home/user/TWD/LiveEdit_lab_usb/LiveEdit

L0_CKPT=records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_dn500_e20_s42_20261010_231536-2026.10.10-23.15.41/checkpoints/epoch-20-i-10200-ema_loss-2.2061

test -e "$L0_CKPT" || { echo "FAIL missing L0 ckpt: $L0_CKPT"; ls -lah "$(dirname "$L0_CKPT")" || true; exit 1; }
ls -lah "$L0_CKPT" | head -n 3

if pgrep -f 'train_vllm_editor.py.*lab_g5|lab_g5_llava_fresh_pipeline' >/dev/null 2>&1; then
  echo "FAIL: a lab_g5 train is already running"
  pgrep -af 'train_vllm_editor.py.*lab_g5|lab_g5_llava_fresh_pipeline' || true
  exit 1
fi

mkdir -p doc/results/cvpr2027/g5/logs \
  /home/user/TWD/LiveEdit_lab_usb/experiment_backup/g5
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTER=doc/results/cvpr2027/g5/logs/s2_dek_${STAMP}.log
PIDFILE=doc/results/cvpr2027/g5/logs/s2_dek.pid

export PYTHONUTF8=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export NVIDIA_TF32_OVERRIDE=1
# One AVBL negative per step — needed for LLaVA-7B on 24GB
export LIVEEDIT_AVBL_MAX_NEGS=1

echo "===== START S2 DEK from $L0_CKPT =====" | tee "$OUTER"
nohup python -u tools/lab_g5_llava_fresh_pipeline.py \
  --stages S2 DEK \
  --eps_l0 20 --eps_ft 1 \
  --from_l0_ckpt "$L0_CKPT" \
  >> "$OUTER" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
echo "PIPELINE_PID=$PID" | tee -a "$OUTER"
echo "OUTER_LOG=$OUTER" | tee -a "$OUTER"
disown "$PID" 2>/dev/null || true
sleep 20
if ! ps -p "$PID" > /dev/null 2>&1; then
  echo FAILED
  tail -n 100 "$OUTER"
  exit 1
fi
ps -p "$PID" -o pid,etime,cmd
echo "----- outer -----"
tail -n 40 "$OUTER"
echo "Detached S2->DEK OK. Closing Cursor will not stop it."
