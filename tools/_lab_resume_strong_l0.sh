#!/usr/bin/env bash
# Resume CVPR strong L0 after loadshedding (proper --resume to same run dir).
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit
cd /home/user/TWD/LiveEdit_lab_usb/LiveEdit

RUN_NAME=lab_g5_llava_L0_dn500_e20_s42_20261010_231536-2026.10.10-23.15.41
CKPT_DIR=records/lab_g5/liveedit/llava-v1.5-7b/${RUN_NAME}/checkpoints
RESUME_CKPT=${CKPT_DIR}/epoch-11-i-5400-ema_loss-5.0621
BAD0=${CKPT_DIR}/epoch-11-i-5500-ema_loss-4.1537
CORRUPT=/home/user/TWD/LiveEdit_lab_usb/models/llava-1.5-7b-hf.corrupt_20261010T145925Z

echo "===== cleanup ====="
if [ -d "$CORRUPT" ]; then
  du -sh "$CORRUPT" || true
  rm -rf "$CORRUPT"
  echo "removed corrupt weights tree"
fi
rm -f "$BAD0"
test -e "$RESUME_CKPT" || { echo "FAIL missing $RESUME_CKPT"; ls -lah "$CKPT_DIR"; exit 1; }
ls -lah "$RESUME_CKPT" | head -n 3
df -h /home/user | tail -n 1

echo "===== validate LLaVA weights ====="
python <<'PY'
from pathlib import Path
from safetensors import safe_open
from utils.GLOBAL import model_path_map
p = Path(model_path_map["llava-v1.5-7b"])
for s in sorted(p.glob("model-*.safetensors")):
    with safe_open(str(s), framework="pt") as f:
        print("OK", s.name, len(list(f.keys())))
print("WEIGHTS_OK")
PY

if pgrep -f 'train_vllm_editor.py.*lab_g5' >/dev/null 2>&1; then
  echo "FAIL: train already running"; pgrep -af 'train_vllm_editor.py.*lab_g5'; exit 1
fi

mkdir -p doc/results/cvpr2027/g5/logs \
  /home/user/TWD/LiveEdit_lab_usb/experiment_backup/g5
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
TRAINLOG=doc/results/cvpr2027/g5/logs/l0_e20_resume_${STAMP}.train.log
PIDFILE=doc/results/cvpr2027/g5/logs/l0_strong20.pid

# Resume mid-run: keep epoch/i, same run folder, finish through epoch 20
export PYTHONUTF8=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=1

{
  echo "RESUME_FROM=$RESUME_CKPT"
  echo "RUN_NAME=$RUN_NAME"
  echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "CMD: train_vllm_editor.py ... -eps 20 --resume --resume_run_name $RUN_NAME -lkpt $RESUME_CKPT"
} > "$TRAINLOG"

nohup python -u train_vllm_editor.py \
  -en liveedit -mn llava -dna VLKEB \
  -bs 1 -dvc cuda:0 -edvc 0 \
  -eps 20 -tnp lab_g5_llava_L0_dn500_e20_resume \
  -dn 500 -rs 42 -sci 100 -lpi 1 \
  -dpath data/VLKEB/grounding/grounding_train_short.json \
  -lkpt "$RESUME_CKPT" \
  --resume \
  --resume_run_name "$RUN_NAME" \
  --keep_latest_ckpts 3 \
  --records_dir records/lab_g5 \
  >> "$TRAINLOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
echo "TRAIN_PID=$PID"
echo "TRAIN_LOG=$TRAINLOG"
disown "$PID" 2>/dev/null || true
sleep 20
if ! ps -p "$PID" > /dev/null 2>&1; then
  echo FAILED
  tail -n 100 "$TRAINLOG"
  exit 1
fi
ps -p "$PID" -o pid,etime,cmd
echo "----- log -----"
tail -n 40 "$TRAINLOG"
echo "Detached resume OK (continue to epoch 20 from i-5400)."
