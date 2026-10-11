#!/usr/bin/env bash
set -euo pipefail
cd /home/user/TWD/LiveEdit_lab_usb/LiveEdit
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit

echo "===== pin transformers (lab torch 2.5.1) ====="
pip install -q "transformers==4.57.6"
python <<'PY'
import torch, transformers
print("torch", torch.__version__)
print("transformers", transformers.__version__)
from transformers import AutoTokenizer
print("AutoTokenizer OK")
PY

mkdir -p doc/results/cvpr2027/g5/logs
OUTER=doc/results/cvpr2027/g5/logs/l0_outer.log
: > "$OUTER"

echo "===== START L0 ====="
nohup python -u tools/lab_g5_llava_fresh_pipeline.py --stages L0 >> "$OUTER" 2>&1 &
PID=$!
echo "PIPELINE_PID=$PID"
sleep 8
TRAIN=$(ls -1t doc/results/cvpr2027/g5/logs/*.train.log 2>/dev/null | head -n 1 || true)
echo "OUTER_LOG=$OUTER"
echo "TRAIN_LOG=${TRAIN:-none}"
if [ -n "${TRAIN:-}" ]; then
  echo "===== tailing train log ====="
  tail -n 80 -f "$TRAIN"
else
  tail -n 80 -f "$OUTER"
fi
