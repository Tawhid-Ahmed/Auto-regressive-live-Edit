#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit

DST=/home/user/TWD/LiveEdit_lab_usb/models/llava-1.5-7b-hf
mkdir -p "$DST"

# Kill any stuck prior download workers
pkill -f 'huggingface-cli download llava' 2>/dev/null || true
pkill -f 'hf download llava' 2>/dev/null || true
sleep 1

echo "===== serial hf download (max-workers 1) ====="
export HF_HUB_ENABLE_HF_TRANSFER=0
hf download llava-hf/llava-1.5-7b-hf \
  --local-dir "$DST" \
  --max-workers 1

echo "===== validate ====="
python <<'PY'
from pathlib import Path
from safetensors import safe_open
p = Path("/home/user/TWD/LiveEdit_lab_usb/models/llava-1.5-7b-hf")
shards = sorted(p.glob("model-*.safetensors"))
assert len(shards) == 3, f"expected 3 shards, got {[s.name for s in shards]}"
for s in shards:
    with safe_open(s, framework="pt") as f:
        print("OK", s.name, s.stat().st_size, "n_tensors", len(list(f.keys())))
print("ALL_OK")
PY

echo "===== START L0 ====="
cd /home/user/TWD/LiveEdit_lab_usb/LiveEdit
mkdir -p doc/results/cvpr2027/g5/logs
OUTER=doc/results/cvpr2027/g5/logs/l0_outer.log
: > "$OUTER"
nohup python -u tools/lab_g5_llava_fresh_pipeline.py --stages L0 >> "$OUTER" 2>&1 &
echo "PIPELINE_PID=$!"
sleep 20
TRAIN=$(ls -1t doc/results/cvpr2027/g5/logs/*.train.log 2>/dev/null | head -n 1 || true)
echo "TRAIN_LOG=${TRAIN:-none}"
if [ -n "${TRAIN:-}" ]; then
  tail -n 80 -f "$TRAIN"
else
  tail -n 80 -f "$OUTER"
fi
