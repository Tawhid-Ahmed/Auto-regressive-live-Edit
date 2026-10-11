#!/usr/bin/env bash
set -euo pipefail
cd /home/user/TWD/LiveEdit_lab_usb/LiveEdit

echo "===== PULL lab/g5-code-only ====="
if [ -f tools/lab_git_pull_on_lab.sh ]; then
  bash tools/lab_git_pull_on_lab.sh
else
  git fetch origin lab/g5-code-only
  git checkout -f lab/g5-code-only 2>/dev/null || git checkout -f -B lab/g5-code-only origin/lab/g5-code-only
  git pull --ff-only origin lab/g5-code-only
fi
echo "HEAD=$(git rev-parse --short HEAD)"
git log -1 --oneline

echo "===== ENV ====="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit
python -c "import tensorboard; print('tensorboard', tensorboard.__version__)"

mkdir -p doc/results/cvpr2027/g5/logs
OUTER=doc/results/cvpr2027/g5/logs/l0_outer.log
: > "$OUTER"

echo "===== START L0 (nohup) ====="
nohup python -u tools/lab_g5_llava_fresh_pipeline.py --stages L0 >> "$OUTER" 2>&1 &
PID=$!
echo "PIPELINE_PID=$PID"
echo "OUTER_LOG=$OUTER"
sleep 5

# Prefer the pipeline's train log once it appears
TRAIN=$(ls -1t doc/results/cvpr2027/g5/logs/*.train.log 2>/dev/null | head -n 1 || true)
if [ -n "${TRAIN:-}" ]; then
  echo "TRAIN_LOG=$TRAIN"
  echo "===== tailing train log (Ctrl+C only stops the tail; training keeps running) ====="
  tail -n 50 -f "$TRAIN"
else
  echo "===== tailing outer log ====="
  tail -n 80 -f "$OUTER"
fi
