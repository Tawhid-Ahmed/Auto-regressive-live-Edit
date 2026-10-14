#!/usr/bin/env bash
# Isolate OOM: S2-like at bs=2 with patch only vs patch+AVBL (swap-only).
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOGDIR=doc/results/cvpr2027/g5/logs
mkdir -p "$LOGDIR"
OUTER="$LOGDIR/s2_step2_isolate_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$OUTER") 2>&1
TRAIN_JSON=data/VLKEB/grounding/grounding_train_short.json
[ -f "$TRAIN_JSON" ] || TRAIN_JSON=data/VLKEB/meeting_validation/step1/grounding/grounding_train_short.json
L0=records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_bs2_dn500_e20_s42_20261011_191244-2026.10.11-19.12.48/checkpoints/epoch-20-i-5000-ema_loss-6.4182
python -c "import torch; torch.cuda.empty_cache(); torch.cuda.ipc_collect(); print('free_MiB', torch.cuda.mem_get_info()[0]//(1024**2))"

run_one() {
  local name="$1"; shift
  local log="$LOGDIR/s2_step2_isolate_${name}.log"
  echo "===== isolate $name $(date -Is) ====="
  set +e
  python -u train_vllm_editor.py -en liveedit -mn llava -dna VLKEB \
    -bs 2 -dbs 1 -dvc cuda:0 -edvc 0 -eps 1 -tnp "s2_iso_${name}" \
    -dn 4 -rs 42 -sci 100 -lpi 1 \
    -dpath "$TRAIN_JSON" -lkpt "$L0" \
    --keep_latest_ckpts 1 --records_dir records/lab_g5_smoke \
    "$@" >"$log" 2>&1
  local rc=$?
  set -e
  if grep -qiE 'out of memory|CUDA out of memory' "$log"; then
    echo "RESULT $name OOM rc=$rc"
  elif [ "$rc" -ne 0 ]; then
    echo "RESULT $name FAIL rc=$rc"; tail -n 20 "$log"
  else
    echo "RESULT $name OK"
  fi
  python -c "import torch; torch.cuda.empty_cache(); torch.cuda.ipc_collect()" 2>/dev/null || true
}

echo "A: patch only (no AVBL)"
run_one patch_only --patch_grounding
echo "B: patch + AVBL swap-only"
run_one patch_avbl1 --patch_grounding --adversarial_grounding \
  --adv_grounding_lambda 0.3 --adv_grounding_neg_weight 2.5 \
  --avbl_hinge_tau 0.3 --avbl_contrastive --avbl_max_adv_images 1
echo "===== isolate done OUTER=$OUTER ====="
