#!/usr/bin/env bash
# Continue M7 after L0 finished: diagnose gate → S2 → DEK → eval → taxonomy.
set -euo pipefail
REPO=/home/user/TWD/LiveEdit_lab_usb/LiveEdit
cd "$REPO"
# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit

LOGDIR=$REPO/doc/results/cvpr2027/g5/logs
mkdir -p "$LOGDIR"
OUTER="$LOGDIR/m7_continue_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$OUTER") 2>&1

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

L0_CKPT=$(ls -t records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_bs2_dn500_e20_*/checkpoints/epoch-20-* 2>/dev/null | head -1 || true)
if [ -z "${L0_CKPT:-}" ] || [ ! -f "$L0_CKPT" ]; then
  echo "FAIL: no L0 epoch-20 ckpt"
  exit 1
fi
echo "===== M7 continue $(date -Is) ====="
echo "L0_CKPT=$L0_CKPT"

TRAIN_JSON=data/VLKEB/grounding/grounding_train_short.json
[ -f "$TRAIN_JSON" ] || TRAIN_JSON=data/VLKEB/meeting_validation/step1/grounding/grounding_train_short.json
IMG=data/VLKEB/VLKEB_images/mmkb_images
PROBE=data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json
TEXT_TOP1_MIN_PCT="${TEXT_TOP1_MIN_PCT:-25}"
BS=2

echo "===== L0 success gate (diagnose train-split) ====="
DIAG_OUT=doc/results/cvpr2027/g5/llava_l0_diagnosis_m7.json
python -u tools/diagnose_llava_l0.py \
  -ckpt "$L0_CKPT" -dpath "$TRAIN_JSON" -img "$IMG" \
  -dsn 50 -seed 42 -dvc cuda:0 -out "$DIAG_OUT"

python - <<PY
import json, sys
j = json.load(open("$DIAG_OUT", encoding="utf-8"))
tr = j["train_split_reliability"]
top1 = tr.get("text_top1_is_own_pct")
print("train_split_reliability:", json.dumps(tr, indent=2))
thr = float("$TEXT_TOP1_MIN_PCT")
if top1 is None:
    print("FAIL: missing text_top1_is_own_pct")
    sys.exit(2)
if float(top1) < thr:
    print(f"FAIL: L0 gate {float(top1):.2f}% < {thr}% — stop before S2/DEK")
    sys.exit(3)
print(f"PASS: L0 gate {float(top1):.2f}% >= {thr}%")
PY

echo "===== S2 + DEK from L0 ====="
python -u tools/lab_g5_llava_fresh_pipeline.py --stages S2 DEK -bs "$BS" --eps_ft 1 --from_l0_ckpt "$L0_CKPT"

S2_CKPT=$(ls -t records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_S2_bs2_*/checkpoints/epoch-* 2>/dev/null | head -1 || true)
DEK_CKPT=$(ls -t records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_DEK_bs2_*/checkpoints/epoch-* 2>/dev/null | head -1 || true)
echo "S2_CKPT=$S2_CKPT"
echo "DEK_CKPT=$DEK_CKPT"
[ -n "${S2_CKPT:-}" ] && [ -f "$S2_CKPT" ] || { echo "FAIL: missing S2 ckpt"; exit 1; }
[ -n "${DEK_CKPT:-}" ] && [ -f "$DEK_CKPT" ] || { echo "FAIL: missing DEK ckpt"; exit 1; }

echo "===== Eval L0 ====="
python -u -W ignore test_grounding_eval.py -en liveedit -mn llava -sen 10 -dvc cuda:0 \
  -dsn 100 -seed 42 -ckpt "$L0_CKPT" -enp g5_llava_m7_L0_short \
  -dpath "$PROBE" --meeting_step 6 --log_routing

echo "===== Eval S2 ====="
python -u -W ignore test_grounding_eval.py -en liveedit -mn llava -sen 10 -dvc cuda:0 \
  -dsn 100 -seed 42 -ckpt "$S2_CKPT" -enp g5_llava_m7_S2_short \
  -dpath "$PROBE" --meeting_step 6 --log_routing --patch_grounding

echo "===== Eval DEK ====="
python -u -W ignore test_grounding_eval.py -en liveedit -mn llava -sen 10 -dvc cuda:0 \
  -dsn 100 -seed 42 -ckpt "$DEK_CKPT" -enp g5_llava_m7_DEK_short \
  -dpath "$PROBE" --meeting_step 6 --log_routing --patch_grounding --dek_dual_gate

echo "===== Taxonomy ====="
L0_RUN=$(ls -dt eval_results/meeting_validation/step6/liveedit/llava-v1.5-7b/*g5_llava_m7_L0_short* | head -1)
S2_RUN=$(ls -dt eval_results/meeting_validation/step6/liveedit/llava-v1.5-7b/*g5_llava_m7_S2_short* | head -1)
DEK_RUN=$(ls -dt eval_results/meeting_validation/step6/liveedit/llava-v1.5-7b/*g5_llava_m7_DEK_short* | head -1)
python -u tools/score_leakage_taxonomy.py \
  --run "$L0_RUN" --run "$S2_RUN" --run "$DEK_RUN" \
  --labels L0 S2 DEK \
  -out doc/results/cvpr2027/g5/leakage_taxonomy_llava_m7.json \
  --md doc/results/cvpr2027/g5/leakage_taxonomy_llava_m7.md

echo "===== M7 continue OK $(date -Is) ====="
echo "OUTER=$OUTER"
echo "Fill doc/results/cvpr2027/g5/G5_LLAVA_M7_BS_RETRAIN.md"
