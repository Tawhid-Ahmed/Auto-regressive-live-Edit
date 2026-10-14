#!/usr/bin/env bash
# Step 3: S2 → DEK from existing M7 L0 at bs=2 (VRAM pack) → health gates → VLKEB extract + taxonomy.
# Skips L0 retrain. Lab only.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOGDIR=doc/results/cvpr2027/g5/logs
mkdir -p "$LOGDIR"
OUTER="$LOGDIR/s3_s2dek_bs2_outer_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$OUTER") 2>&1

PROBE="${PROBE:-data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json}"
TRAIN_JSON="${TRAIN_JSON:-data/VLKEB/grounding/grounding_train_short.json}"
if [ ! -f "$TRAIN_JSON" ]; then
  TRAIN_JSON=data/VLKEB/meeting_validation/step1/grounding/grounding_train_short.json
fi
IMG="${IMG:-data/VLKEB/VLKEB_images/mmkb_images}"
DSN_EVAL="${DSN_EVAL:-100}"
DSN_DIAG="${DSN_DIAG:-50}"
MAX_VISION_OK_PCT="${MAX_VISION_OK_PCT:-20}"
TEXT_TOP1_MIN_DELTA_PP="${TEXT_TOP1_MIN_DELTA_PP:--5}"
BS="${FORCE_BS:-2}"
DIAG_L0_OUT=doc/results/cvpr2027/g5/llava_l0_diagnosis_m7.json
L0_CKPT="${L0_CKPT:-records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_bs2_dn500_e20_s42_20261011_191244-2026.10.11-19.12.48/checkpoints/epoch-20-i-5000-ema_loss-6.4182}"

echo "===== Step3 S2/DEK bs=$BS start $(date -Is) ====="
echo "L0_CKPT=$L0_CKPT"
echo "DIAG_L0_OUT=$DIAG_L0_OUT MAX_VISION_OK_PCT=$MAX_VISION_OK_PCT"
test -f "$L0_CKPT" || { echo "FAIL: missing L0 ckpt"; exit 1; }
test -f "$DIAG_L0_OUT" || { echo "FAIL: missing L0 diagnosis JSON (baseline for health gate)"; exit 1; }

run_routing_health_gate() {
  local stage="$1"
  local ckpt="$2"
  local extra=()
  if [ "${3:-}" = "--dek_dual_gate" ]; then
    extra+=(--dek_dual_gate)
  fi
  local out="doc/results/cvpr2027/g5/routing_health_${stage,,}_s3_bs2.json"
  echo "===== Routing health gate ($stage) ====="
  python -u tools/diagnose_llava_l0.py --health_gate \
    -ckpt "$ckpt" -dpath "$TRAIN_JSON" -img "$IMG" \
    -dsn "$DSN_DIAG" -seed 42 -dvc cuda:0 -out "$out" \
    --baseline_json "$DIAG_L0_OUT" \
    --max_vision_ok_pct="$MAX_VISION_OK_PCT" \
    --text_top1_min_delta_pp="$TEXT_TOP1_MIN_DELTA_PP" \
    "${extra[@]}"
}

python -c "import torch; torch.cuda.empty_cache(); torch.cuda.ipc_collect(); print('free_MiB', torch.cuda.mem_get_info()[0]//(1024**2))"

# Slightly softer AVBL after Step3 attempt-1 blew up (AVBL→inf at step 84).
ADV_LAMBDA="${ADV_LAMBDA:-0.1}"

echo "===== S2 from M7 L0 (bs=$BS, VRAM pack, adv_lambda=$ADV_LAMBDA) ====="
python -u tools/lab_g5_llava_fresh_pipeline.py --stages S2 -bs "$BS" --eps_ft 1 \
  --from_l0_ckpt "$L0_CKPT" --avbl_max_adv_images 1 --train_max_gen_types 2 \
  --adv_grounding_lambda "$ADV_LAMBDA"

S2_CKPT=$(python - <<'PY'
from pathlib import Path
root = Path("records/lab_g5/liveedit/llava-v1.5-7b")
cands = sorted(
    [p for p in root.glob("lab_g5_llava_S2_bs2_dn500_e1_*/checkpoints/epoch-*")
     if "nan" not in p.name.lower()],
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
print(cands[0] if cands else "")
PY
)
echo "S2_CKPT=$S2_CKPT"
[ -n "$S2_CKPT" ] && [ -f "$S2_CKPT" ] || { echo "FAIL: missing finite S2 ckpt (nan-only?)"; exit 1; }
run_routing_health_gate S2 "$S2_CKPT"

echo "===== DEK from M7 L0 (bs=$BS, VRAM pack, adv_lambda=$ADV_LAMBDA) ====="
python -u tools/lab_g5_llava_fresh_pipeline.py --stages DEK -bs "$BS" --eps_ft 1 \
  --from_l0_ckpt "$L0_CKPT" --avbl_max_adv_images 1 --train_max_gen_types 2 \
  --adv_grounding_lambda "$ADV_LAMBDA"

DEK_CKPT=$(python - <<'PY'
from pathlib import Path
root = Path("records/lab_g5/liveedit/llava-v1.5-7b")
cands = sorted(
    [p for p in root.glob("lab_g5_llava_DEK_bs2_dn500_e1_*/checkpoints/epoch-*")
     if "nan" not in p.name.lower()],
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
print(cands[0] if cands else "")
PY
)
echo "DEK_CKPT=$DEK_CKPT"
[ -n "$DEK_CKPT" ] && [ -f "$DEK_CKPT" ] || { echo "FAIL: missing DEK ckpt"; exit 1; }
run_routing_health_gate DEK "$DEK_CKPT" --dek_dual_gate

# Distinct enp names so Step3 bs=2 results do not clobber the old bs=1 ladder.
ENP_L0=g5_llava_s3_bs2_L0_short
ENP_S2=g5_llava_s3_bs2_S2_short
ENP_DEK=g5_llava_s3_bs2_DEK_short

echo "===== Eval L0 (reuse M7 L0 ckpt) ====="
python -u -W ignore test_grounding_eval.py -en liveedit -mn llava -sen 10 -dvc cuda:0 \
  -dsn "$DSN_EVAL" -seed 42 -ckpt "$L0_CKPT" -enp "$ENP_L0" \
  -dpath "$PROBE" --meeting_step 6 --log_routing

echo "===== Eval S2 ====="
python -u -W ignore test_grounding_eval.py -en liveedit -mn llava -sen 10 -dvc cuda:0 \
  -dsn "$DSN_EVAL" -seed 42 -ckpt "$S2_CKPT" -enp "$ENP_S2" \
  -dpath "$PROBE" --meeting_step 6 --log_routing --patch_grounding

echo "===== Eval DEK ====="
python -u -W ignore test_grounding_eval.py -en liveedit -mn llava -sen 10 -dvc cuda:0 \
  -dsn "$DSN_EVAL" -seed 42 -ckpt "$DEK_CKPT" -enp "$ENP_DEK" \
  -dpath "$PROBE" --meeting_step 6 --log_routing --patch_grounding --dek_dual_gate

echo "===== Taxonomy ====="
L0_RUN=$(ls -dt eval_results/meeting_validation/step6/liveedit/llava-v1.5-7b/*${ENP_L0}* | head -1)
S2_RUN=$(ls -dt eval_results/meeting_validation/step6/liveedit/llava-v1.5-7b/*${ENP_S2}* | head -1)
DEK_RUN=$(ls -dt eval_results/meeting_validation/step6/liveedit/llava-v1.5-7b/*${ENP_DEK}* | head -1)
echo "L0_RUN=$L0_RUN"
echo "S2_RUN=$S2_RUN"
echo "DEK_RUN=$DEK_RUN"
python -u tools/score_leakage_taxonomy.py \
  --run "$L0_RUN" --run "$S2_RUN" --run "$DEK_RUN" \
  --labels L0 S2 DEK \
  -out doc/results/cvpr2027/g5/leakage_taxonomy_llava_s3_bs2.json \
  --md doc/results/cvpr2027/g5/leakage_taxonomy_llava_s3_bs2.md

echo "===== Step3 S2/DEK OK $(date -Is) ====="
echo "BS=$BS S2=$S2_CKPT DEK=$DEK_CKPT OUTER=$OUTER"
echo "Taxonomy: doc/results/cvpr2027/g5/leakage_taxonomy_llava_s3_bs2.{json,md}"
