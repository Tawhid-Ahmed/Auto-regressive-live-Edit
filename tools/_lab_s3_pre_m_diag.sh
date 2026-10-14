#!/usr/bin/env bash
# Pre-M8/M10 diagnostics: L0 health gate (same dsn=50 as S3) + extractor channel energy on L0 and S3 S2.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit

export PYTHONUNBUFFERED=1
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=doc/results/cvpr2027/g5/logs
mkdir -p "$LOGDIR" doc/results/cvpr2027/g5
OUTER="$LOGDIR/s3_pre_m_diag_outer_${STAMP}.log"
exec > >(tee -a "$OUTER") 2>&1

L0_CKPT=$(ls -d records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_bs2_dn500_e20_s42_20261011_191244-*/checkpoints/epoch-20-i-5000-* 2>/dev/null | head -1)
S2_CKPT=$(ls -d records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_S2_bs2_dn500_e1_s42_20261012_103239-*/checkpoints/epoch-1-i-200-* 2>/dev/null | head -1)
DPATH=data/VLKEB/grounding/grounding_train_short.json

echo "===== Pre-M diag start $(date -Iseconds) ====="
echo "L0_CKPT=$L0_CKPT"
echo "S2_CKPT=$S2_CKPT"
[ -n "$L0_CKPT" ] && [ -f "$L0_CKPT" ] || { echo "FAIL: missing L0 ckpt"; exit 1; }
[ -n "$S2_CKPT" ] && [ -f "$S2_CKPT" ] || { echo "FAIL: missing S2 ckpt"; exit 1; }

echo "===== L0 health_gate + extractor_energy ====="
python -u tools/diagnose_llava_l0.py \
  --health_gate --extractor_energy --extractor_energy_n 16 \
  -ckpt "$L0_CKPT" \
  -dpath "$DPATH" -dsn 50 -seed 42 -dvc cuda:0 \
  -out doc/results/cvpr2027/g5/routing_health_l0_s3diag.json \
  --baseline_json doc/results/cvpr2027/g5/llava_l0_diagnosis_m7.json \
  || true
# health gate exit 3 on fail is OK for L0 comparison; still want the JSON

echo "===== S2 extractor_energy (pool+keys; health already known FAIL) ====="
python -u tools/diagnose_llava_l0.py \
  --extractor_energy --extractor_energy_n 16 \
  -ckpt "$S2_CKPT" \
  -dpath "$DPATH" -dsn 50 -seed 42 -dvc cuda:0 \
  -out doc/results/cvpr2027/g5/extractor_energy_s2_s3.json \
  --n_detail 0

echo "===== Pre-M diag done $(date -Iseconds) ====="
ls -la doc/results/cvpr2027/g5/routing_health_l0_s3diag.json \
       doc/results/cvpr2027/g5/extractor_energy_s2_s3.json
