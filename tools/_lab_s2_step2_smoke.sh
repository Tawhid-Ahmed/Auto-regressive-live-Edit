#!/usr/bin/env bash
# Step 2 smoke: S2+AVBL from M7 L0 at bs=4 then bs=2 after AVBL no_grad fix.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOGDIR=doc/results/cvpr2027/g5/logs
mkdir -p "$LOGDIR"
OUTER="$LOGDIR/s2_step2_smoke_outer_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$OUTER") 2>&1

TRAIN_JSON=data/VLKEB/grounding/grounding_train_short.json
if [ ! -f "$TRAIN_JSON" ]; then
  TRAIN_JSON=data/VLKEB/meeting_validation/step1/grounding/grounding_train_short.json
fi
L0="${L0_CKPT:-records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_bs2_dn500_e20_s42_20261011_191244-2026.10.11-19.12.48/checkpoints/epoch-20-i-5000-ema_loss-6.4182}"
echo "===== Step2 smoke start $(date -Is) ====="
echo "L0=$L0 TRAIN_JSON=$TRAIN_JSON"
test -f "$L0" || { echo "FAIL: missing L0 ckpt"; exit 1; }

python -c "import torch; torch.cuda.empty_cache(); torch.cuda.ipc_collect(); print('free_MiB', torch.cuda.mem_get_info()[0]//(1024**2))"

smoke() {
  local bs="$1"
  local log="$LOGDIR/s2_step2_smoke_bs${bs}.log"
  echo "===== S2 AVBL smoke bs=$bs $(date -Is) ====="
  set +e
  python -u train_vllm_editor.py -en liveedit -mn llava -dna VLKEB \
    -bs "$bs" -dbs 1 -dvc cuda:0 -edvc 0 -eps 1 -tnp "s2_step2_smoke_bs${bs}" \
    -dn 8 -rs 42 -sci 100 -lpi 1 \
    -dpath "$TRAIN_JSON" \
    -lkpt "$L0" \
    --keep_latest_ckpts 1 \
    --records_dir records/lab_g5_smoke \
    --patch_grounding --adversarial_grounding \
    --adv_grounding_lambda 0.3 --adv_grounding_neg_weight 2.5 \
    --avbl_hinge_tau 0.3 --avbl_contrastive \
    --avbl_max_adv_images "${AVBL_MAX_ADV:-1}" \
    --no_patch_hard_routing \
    --train_max_gen_types "${TRAIN_MAX_GEN:-2}" \
    >"$log" 2>&1
  local rc=$?
  set -e
  if grep -qiE 'out of memory|CUDA out of memory' "$log"; then
    echo "RESULT bs=$bs OOM rc=$rc log=$log"
    return 2
  fi
  if [ "$rc" -ne 0 ]; then
    echo "RESULT bs=$bs FAIL rc=$rc"
    tail -n 40 "$log" || true
    return "$rc"
  fi
  python -c "
import re
text=open('$log',encoding='utf-8',errors='replace').read()
hits=re.findall(r'MoE soft routing relative loss[^0-9\\-]*([0-9]+\\.?[0-9]*(?:e[+-]?[0-9]+)?)', text, re.I)
if not hits:
    hits=re.findall(r\"['\\\"]MoE soft routing relative loss['\\\"]\\s*:\\s*([0-9eE.+\\-]+)\", text)
print('soft_rel_hits', hits[:10])
vals=[float(x) for x in hits]
print('SOFT_ROUTING_OK max=', max(vals) if vals else None)
print('SOFT_ROUTING_NONZERO', bool(vals) and max(vals) > 0)
"
  echo "RESULT bs=$bs OK rc=0 log=$log"
  return 0
}

if smoke 4; then
  echo "STEP2_SMOKE_PASS bs=4 OUTER=$OUTER"
  exit 0
fi
echo "bs=4 failed; clearing GPU and trying bs=2"
python -c "import torch; torch.cuda.empty_cache(); torch.cuda.ipc_collect()"
if smoke 2; then
  echo "STEP2_SMOKE_PASS bs=2 OUTER=$OUTER"
  exit 0
fi
echo "STEP2_SMOKE_FAIL OUTER=$OUTER"
exit 1
