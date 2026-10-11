#!/usr/bin/env bash
# G5 E1: Easy-VQA grounding eval + leak taxonomy on lab LLaVA L0 / S2 / DEK.
# Same protocol as B1 VLKEB short (dsn=100, sen=10, seed=42).
#   nohup bash tools/lab_g5_llava_e1_evqa.sh \
#     > doc/results/cvpr2027/g5/logs/e1_evqa_outer.log 2>&1 &
set -euo pipefail

REPO="${REPO:-/home/user/TWD/LiveEdit_lab_usb/LiveEdit}"
cd "$REPO"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit

export PYTHONUTF8=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NVIDIA_TF32_OVERRIDE=1

PROBE=data/easy-edit-mm/grounding/grounding_eval_short.json
IMG=data/easy-edit-mm/images
LOGDIR=doc/results/cvpr2027/g5/logs
mkdir -p "$LOGDIR"

L0_CKPT=records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_dn500_e20_s42_20261010_231536-2026.10.10-23.15.41/checkpoints/epoch-20-i-10200-ema_loss-2.2061
S2_CKPT=records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_S2_dn500_e1_s42_20261011_104554-2026.10.11-10.45.58/checkpoints/epoch-1-i-500-ema_loss-3.8904
DEK_CKPT=records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_DEK_dn500_e1_s42_20261011_111542-2026.10.11-11.16.06/checkpoints/epoch-1-i-500-ema_loss-3.8851

for p in "$PROBE" "$IMG/val2014" "$IMG/val2014_image_rephrase" "$L0_CKPT" "$S2_CKPT" "$DEK_CKPT"; do
  if [[ ! -e "$p" ]]; then
    echo "FAIL: missing $p" >&2
    exit 1
  fi
done

PY="${CONDA_PREFIX}/bin/python"
COMMON=(-u -W ignore test_grounding_eval.py
  -en liveedit -mn llava -sen 10 -dvc cuda:0 -dsn 100 -seed 42
  -dpath "$PROBE" --img_root "$IMG" --meeting_step 6 --log_routing)

run_one() {
  local label="$1"; shift
  local log="$LOGDIR/eval_${label}.log"
  echo "===== START eval $label $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" | tee "$log"
  echo "CMD: $PY ${COMMON[*]} $*" | tee -a "$log"
  set +e
  "$PY" "${COMMON[@]}" "$@" >>"$log" 2>&1
  local code=$?
  set -e
  echo "===== END eval $label exit=$code $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" | tee -a "$log"
  return $code
}

run_one g5_llava_evqa_L0_short \
  -ckpt "$L0_CKPT" -enp g5_llava_evqa_L0_short

run_one g5_llava_evqa_S2_short \
  -ckpt "$S2_CKPT" -enp g5_llava_evqa_S2_short --patch_grounding

run_one g5_llava_evqa_DEK_short \
  -ckpt "$DEK_CKPT" -enp g5_llava_evqa_DEK_short --patch_grounding --dek_dual_gate

find_run() {
  local enp="$1"
  find eval_results/meeting_validation/step6 -type d -name "*${enp}*" 2>/dev/null \
    | sort | tail -n 1
}

R_L0=$(find_run g5_llava_evqa_L0_short)
R_S2=$(find_run g5_llava_evqa_S2_short)
R_DEK=$(find_run g5_llava_evqa_DEK_short)

echo "Result dirs:"
echo "  L0  $R_L0"
echo "  S2  $R_S2"
echo "  DEK $R_DEK"

if [[ -z "$R_L0" || -z "$R_S2" || -z "$R_DEK" ]]; then
  echo "FAIL: missing eval_results dirs" >&2
  exit 1
fi

OUT=doc/results/cvpr2027/g5/leakage_taxonomy_llava_evqa.json
"$PY" -u tools/score_leakage_taxonomy.py \
  --run "$R_L0" --run "$R_S2" --run "$R_DEK" \
  --labels L0 S2 DEK \
  -out "$OUT" \
  | tee "$LOGDIR/taxonomy_llava_evqa.log"

{
  echo ""
  echo "### EVAL — LLaVA Easy-VQA grounding + taxonomy (E1)"
  echo "- **time (UTC):** $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- **status:** finished"
  echo "- **probe:** \`$PROBE\`"
  echo "- **img_root:** \`$IMG\`"
  echo "- **L0 run:** \`$R_L0\`"
  echo "- **S2 run:** \`$R_S2\`"
  echo "- **DEK run:** \`$R_DEK\`"
  echo "- **taxonomy:** \`$OUT\`"
} | tee -a doc/results/cvpr2027/g5/EXPERIMENT_TRACKER.md

echo "E1 EVQA EXTRACT OK"
