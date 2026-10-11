#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit

MODELS=/home/user/TWD/LiveEdit_lab_usb/models
DST=$MODELS/llava-1.5-7b-hf
BAD=$MODELS/llava-1.5-7b-hf.corrupt_$(date -u +%Y%m%dT%H%M%SZ)

echo "===== move corrupt weights aside ====="
if [ -d "$DST" ]; then
  mv "$DST" "$BAD"
  echo "moved -> $BAD"
fi
mkdir -p "$DST"

echo "===== download llava-hf/llava-1.5-7b-hf ====="
huggingface-cli download llava-hf/llava-1.5-7b-hf --local-dir "$DST"

echo "===== validate safetensors ====="
python <<'PY'
from pathlib import Path
from safetensors import safe_open
p = Path("/home/user/TWD/LiveEdit_lab_usb/models/llava-1.5-7b-hf")
for s in sorted(p.glob("model-*.safetensors")):
    with safe_open(s, framework="pt") as f:
        print("OK", s.name, "n_tensors", len(list(f.keys())))
print("config", (p/"config.json").is_file())
PY

echo "===== START L0 ====="
cd /home/user/TWD/LiveEdit_lab_usb/LiveEdit
mkdir -p doc/results/cvpr2027/g5/logs
OUTER=doc/results/cvpr2027/g5/logs/l0_outer.log
: > "$OUTER"
nohup python -u tools/lab_g5_llava_fresh_pipeline.py --stages L0 >> "$OUTER" 2>&1 &
echo "PIPELINE_PID=$!"
sleep 15
TRAIN=$(ls -1t doc/results/cvpr2027/g5/logs/*.train.log 2>/dev/null | head -n 1 || true)
echo "TRAIN_LOG=${TRAIN:-none}"
if [ -n "${TRAIN:-}" ]; then
  tail -n 120 -f "$TRAIN"
else
  tail -n 120 -f "$OUTER"
fi
