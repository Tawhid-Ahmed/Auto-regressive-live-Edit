#!/usr/bin/env bash
set -euo pipefail
cd /home/user/TWD/LiveEdit_lab_usb/LiveEdit
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit

python <<'PY'
from utils.GLOBAL import ROOT_PATH, model_path_map
import os
p = model_path_map["llava-v1.5-7b"]
print("ROOT_PATH=", ROOT_PATH)
print("llava=", p)
print("exists=", os.path.isdir(p), "config=", os.path.isfile(os.path.join(p, "config.json")))
if not os.path.isfile(os.path.join(p, "config.json")):
    raise SystemExit("LLaVA path missing — abort")
PY

mkdir -p doc/results/cvpr2027/g5/logs
OUTER=doc/results/cvpr2027/g5/logs/l0_outer.log
: > "$OUTER"
nohup python -u tools/lab_g5_llava_fresh_pipeline.py --stages L0 >> "$OUTER" 2>&1 &
echo "PIPELINE_PID=$!"
sleep 12
TRAIN=$(ls -1t doc/results/cvpr2027/g5/logs/*.train.log 2>/dev/null | head -n 1 || true)
echo "TRAIN_LOG=${TRAIN:-none}"
if [ -n "${TRAIN:-}" ]; then
  echo "===== tailing train log ====="
  tail -n 100 -f "$TRAIN"
else
  tail -n 100 -f "$OUTER"
fi
