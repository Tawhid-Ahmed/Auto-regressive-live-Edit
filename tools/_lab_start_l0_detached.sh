#!/usr/bin/env bash
# Start fresh G5 L0 detached from SSH/Cursor. Survives disconnect.
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit
cd /home/user/TWD/LiveEdit_lab_usb/LiveEdit

echo "===== validate LLaVA weights ====="
python <<'PY'
from pathlib import Path
from safetensors import safe_open
from utils.GLOBAL import ROOT_PATH, model_path_map
import os
p = Path(model_path_map["llava-v1.5-7b"])
print("ROOT_PATH=", ROOT_PATH)
print("llava=", p)
assert p.is_dir(), p
shards = sorted(p.glob("model-*.safetensors"))
assert len(shards) == 3, [s.name for s in shards]
for s in shards:
    with safe_open(str(s), framework="pt") as f:
        print("OK", s.name, s.stat().st_size, "n_tensors", len(list(f.keys())))
assert (p / "config.json").is_file()
print("WEIGHTS_OK")
PY

mkdir -p doc/results/cvpr2027/g5/logs \
  /home/user/TWD/LiveEdit_lab_usb/experiment_backup/g5

# Avoid stacking duplicate trains
if pgrep -f 'lab_g5_llava_fresh_pipeline.py --stages L0' >/dev/null 2>&1; then
  echo "L0 pipeline already running:"; pgrep -af 'lab_g5_llava_fresh_pipeline|train_vllm_editor'
  exit 0
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTER=doc/results/cvpr2027/g5/logs/l0_detached_${STAMP}.log
PIDFILE=doc/results/cvpr2027/g5/logs/l0_detached.pid

echo "===== start L0 (nohup) ====="
nohup python -u tools/lab_g5_llava_fresh_pipeline.py --stages L0 >> "$OUTER" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
disown "$PID" 2>/dev/null || true
sleep 8

echo "PIPELINE_PID=$PID"
echo "OUTER_LOG=$OUTER"
echo "PIDFILE=$PIDFILE"
ps -p "$PID" -o pid,cmd || { echo "FAILED to stay up"; tail -n 80 "$OUTER"; exit 1; }

TRAIN=$(ls -1t doc/results/cvpr2027/g5/logs/*.train.log 2>/dev/null | head -n 1 || true)
echo "TRAIN_LOG=${TRAIN:-pending}"
echo "----- outer log (first lines) -----"
head -n 40 "$OUTER" || true
echo "-----"
echo "Detached OK. Closing Cursor/SSH will NOT stop this job."
echo "Retrieve later from home PC:"
echo "  powershell -ExecutionPolicy Bypass -File tools/lab_sync_pull_results.ps1 -IncludeCkptMeta"
