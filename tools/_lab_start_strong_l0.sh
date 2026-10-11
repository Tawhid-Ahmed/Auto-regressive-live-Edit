#!/usr/bin/env bash
# Detached CVPR strong L0: dn=500 x eps_l0=20. Survives Cursor/SSH close.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit
cd /home/user/TWD/LiveEdit_lab_usb/LiveEdit

# Stop leftover smoke trains if any
pkill -f 'lab_g5_llava_fresh_pipeline.py' 2>/dev/null || true
pkill -f 'train_vllm_editor.py.*lab_g5_llava' 2>/dev/null || true
sleep 2

python <<'PY'
from pathlib import Path
from safetensors import safe_open
from utils.GLOBAL import model_path_map
p = Path(model_path_map["llava-v1.5-7b"])
shards = sorted(p.glob("model-*.safetensors"))
assert len(shards) == 3, shards
for s in shards:
    with safe_open(str(s), framework="pt") as f:
        print("OK", s.name, len(list(f.keys())))
print("WEIGHTS_OK")
PY

mkdir -p doc/results/cvpr2027/g5/logs \
  /home/user/TWD/LiveEdit_lab_usb/experiment_backup/g5
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTER=doc/results/cvpr2027/g5/logs/l0_strong20_${STAMP}.log
PIDFILE=doc/results/cvpr2027/g5/logs/l0_strong20.pid

nohup python -u tools/lab_g5_llava_fresh_pipeline.py --stages L0 --eps_l0 20 --eps_ft 1 \
  >> "$OUTER" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
disown "$PID" 2>/dev/null || true
sleep 12
echo "PIPELINE_PID=$PID"
echo "OUTER_LOG=$OUTER"
ps -p "$PID" -o pid,etime,cmd || { echo FAILED; tail -n 60 "$OUTER"; exit 1; }
TRAIN=$(ls -1t doc/results/cvpr2027/g5/logs/*e20*.train.log 2>/dev/null | head -n 1 || true)
if [ -z "${TRAIN:-}" ]; then
  TRAIN=$(ls -1t doc/results/cvpr2027/g5/logs/*.train.log 2>/dev/null | head -n 1 || true)
fi
echo "TRAIN_LOG=${TRAIN:-pending}"
head -n 30 "$OUTER" || true
echo "Detached strong L0 (eps_l0=20). Closing Cursor will not stop it."
