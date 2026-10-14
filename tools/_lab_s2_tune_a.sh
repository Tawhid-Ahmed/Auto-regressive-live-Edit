#!/usr/bin/env bash
# S2-tune A: from M7 L0, stronger AVBL knobs → health gate only (no DEK).
# See G5_LLAVA_DEK_ROOT_CAUSE.md pre-M diag + S2-tune plan.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOGDIR=doc/results/cvpr2027/g5/logs
mkdir -p "$LOGDIR" doc/results/cvpr2027/g5
OUTER="$LOGDIR/s2_tune_a_outer_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$OUTER") 2>&1

TRAIN_JSON="${TRAIN_JSON:-data/VLKEB/grounding/grounding_train_short.json}"
IMG="${IMG:-data/VLKEB/VLKEB_images/mmkb_images}"
DSN_DIAG="${DSN_DIAG:-50}"
MAX_VISION_OK_PCT="${MAX_VISION_OK_PCT:-20}"
TEXT_TOP1_MIN_DELTA_PP="${TEXT_TOP1_MIN_DELTA_PP:--5}"
BS="${FORCE_BS:-2}"
DIAG_L0_OUT=doc/results/cvpr2027/g5/llava_l0_diagnosis_m7.json
L0_CKPT="${L0_CKPT:-records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_bs2_dn500_e20_s42_20261011_191244-2026.10.11-19.12.48/checkpoints/epoch-20-i-5000-ema_loss-6.4182}"

# Tune A knobs
ADV_LAMBDA="${ADV_LAMBDA:-0.1}"
NEG_W="${NEG_W:-4.0}"
CONTRAST_M="${CONTRAST_M:-0.3}"
RET_M="${RET_M:-1.0}"

echo "===== S2-tune A start $(date -Is) ====="
echo "L0_CKPT=$L0_CKPT"
echo "knobs: lambda=$ADV_LAMBDA neg_w=$NEG_W contrast_m=$CONTRAST_M retrieval_m=$RET_M bs=$BS"
test -f "$L0_CKPT" || { echo "FAIL: missing L0 ckpt"; exit 1; }
test -f "$DIAG_L0_OUT" || { echo "FAIL: missing L0 diagnosis JSON"; exit 1; }

python -c "import torch; torch.cuda.empty_cache(); print('free_MiB', torch.cuda.mem_get_info()[0]//(1024**2))"

echo "===== Train S2 tuneA ====="
python -u tools/lab_g5_llava_fresh_pipeline.py --stages S2 -bs "$BS" --eps_ft 1 \
  --from_l0_ckpt "$L0_CKPT" \
  --avbl_max_adv_images 1 --train_max_gen_types 2 \
  --adv_grounding_lambda "$ADV_LAMBDA" \
  --adv_grounding_neg_weight "$NEG_W" \
  --avbl_contrastive_margin "$CONTRAST_M" \
  --avbl_retrieval_margin "$RET_M" \
  --run_tag tuneA

S2_CKPT=$(python - <<'PY'
from pathlib import Path
root = Path("records/lab_g5/liveedit/llava-v1.5-7b")
cands = sorted(
    [p for p in root.glob("lab_g5_llava_S2_bs2_dn500_e1_s42_tuneA_*/checkpoints/epoch-*")
     if "nan" not in p.name.lower()],
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
print(cands[0] if cands else "")
PY
)
echo "S2_CKPT=$S2_CKPT"
[ -n "$S2_CKPT" ] && [ -f "$S2_CKPT" ] || { echo "FAIL: missing finite S2 tuneA ckpt"; exit 1; }

OUT=doc/results/cvpr2027/g5/routing_health_s2_tune_a.json
echo "===== Routing health gate (S2 tuneA) ====="
set +e
python -u tools/diagnose_llava_l0.py --health_gate \
  -ckpt "$S2_CKPT" -dpath "$TRAIN_JSON" -img "$IMG" \
  -dsn "$DSN_DIAG" -seed 42 -dvc cuda:0 -out "$OUT" \
  --baseline_json "$DIAG_L0_OUT" \
  --max_vision_ok_pct="$MAX_VISION_OK_PCT" \
  --text_top1_min_delta_pp="$TEXT_TOP1_MIN_DELTA_PP"
HC=$?
set -e

echo "===== S2-tune A done $(date -Is) health_exit=$HC ====="
echo "S2_CKPT=$S2_CKPT OUT=$OUT OUTER=$OUTER"
echo "NOTE: DEK not started (health-only). PASS=0 FAIL=3."
exit "$HC"
