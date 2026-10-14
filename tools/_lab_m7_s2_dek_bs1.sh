#!/usr/bin/env bash
# M7 post-L0: S2+DEK at bs=1 (AVBL OOM at bs=2), then VLKEB extract + taxonomy.
# L0 already trained soft routing at bs=2; bs=1 here is fine for fine-tune only.
set -euo pipefail
REPO=/home/user/TWD/LiveEdit_lab_usb/LiveEdit
cd "$REPO"
# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit

LOGDIR=$REPO/doc/results/cvpr2027/g5/logs
mkdir -p "$LOGDIR"
OUTER="$LOGDIR/m7_s2dek_bs1_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$OUTER") 2>&1

export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

L0_CKPT=$(ls -t records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_bs2_dn500_e20_*/checkpoints/epoch-20-* 2>/dev/null | head -1 || true)
if [ -z "${L0_CKPT:-}" ] || [ ! -f "$L0_CKPT" ]; then
  echo "FAIL: no L0 epoch-20 ckpt"
  exit 1
fi
PROBE=data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json
BS=1

echo "===== M7 S2/DEK bs=1 $(date -Is) ====="
echo "L0_CKPT=$L0_CKPT"
echo "NOTE: S2/DEK use bs=1 because AVBL+bs=2 OOMs on 24GB; L0 soft routing already trained at bs=2."

python - <<'PY' || true
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
print("cuda free MiB:", torch.cuda.mem_get_info()[0] // (1024**2) if torch.cuda.is_available() else "n/a")
PY

echo "===== S2 + DEK ====="
python -u tools/lab_g5_llava_fresh_pipeline.py --stages S2 DEK -bs "$BS" --eps_ft 1 --from_l0_ckpt "$L0_CKPT"

S2_CKPT=$(ls -t records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_S2_bs1_*/checkpoints/epoch-* 2>/dev/null | head -1 || true)
DEK_CKPT=$(ls -t records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_DEK_bs1_*/checkpoints/epoch-* 2>/dev/null | head -1 || true)
echo "S2_CKPT=$S2_CKPT"
echo "DEK_CKPT=$DEK_CKPT"
[ -n "${S2_CKPT:-}" ] && [ -f "$S2_CKPT" ] || { echo "FAIL: missing S2 ckpt"; exit 1; }
[ -n "${DEK_CKPT:-}" ] && [ -f "$DEK_CKPT" ] || { echo "FAIL: missing DEK ckpt"; exit 1; }

echo "===== Eval L0 (routing + freegen) ====="
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

echo "===== Taxonomy (binary + fuse_coe tau) ====="
L0_RUN=$(ls -dt eval_results/meeting_validation/step6/liveedit/llava-v1.5-7b/*g5_llava_m7_L0_short* | head -1)
S2_RUN=$(ls -dt eval_results/meeting_validation/step6/liveedit/llava-v1.5-7b/*g5_llava_m7_S2_short* | head -1)
DEK_RUN=$(ls -dt eval_results/meeting_validation/step6/liveedit/llava-v1.5-7b/*g5_llava_m7_DEK_short* | head -1)
echo "L0_RUN=$L0_RUN"
echo "S2_RUN=$S2_RUN"
echo "DEK_RUN=$DEK_RUN"
python -u tools/score_leakage_taxonomy.py \
  --run "$L0_RUN" --run "$S2_RUN" --run "$DEK_RUN" \
  --labels L0 S2 DEK \
  --dump_records \
  -out doc/results/cvpr2027/g5/leakage_taxonomy_llava_m7.json \
  --md doc/results/cvpr2027/g5/leakage_taxonomy_llava_m7.md

echo "===== M7 S2/DEK+extract OK $(date -Is) ====="
echo "OUTER=$OUTER"
echo "Fill G5_LLAVA_M7_BS_RETRAIN.md from leakage_taxonomy_llava_m7.{json,md} and diag json."
