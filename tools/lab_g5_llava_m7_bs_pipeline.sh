#!/usr/bin/env bash
# M7: LLaVA L0 with bs>=2 → success gate → S2 → DEK → VLKEB extract + taxonomy.
# Lab only. Rejects more epochs at bs=1.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1090
source ~/miniconda3/etc/profile.d/conda.sh
conda activate liveedit

LOGDIR=doc/results/cvpr2027/g5/logs
mkdir -p "$LOGDIR"
OUTER="$LOGDIR/m7_outer_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$OUTER") 2>&1

PROBE="${PROBE:-data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json}"
TRAIN_JSON="${TRAIN_JSON:-data/VLKEB/grounding/grounding_train_short.json}"
if [ ! -f "$TRAIN_JSON" ]; then
  TRAIN_JSON=data/VLKEB/meeting_validation/step1/grounding/grounding_train_short.json
fi
IMG="${IMG:-data/VLKEB/VLKEB_images/mmkb_images}"
# diagnose reports percent (0-100); gate is 25%
TEXT_TOP1_MIN_PCT="${TEXT_TOP1_MIN_PCT:-25}"
DSN_EVAL="${DSN_EVAL:-100}"
DSN_DIAG="${DSN_DIAG:-50}"
# Step-1 routing health gate (G5_LLAVA_DEK_ROOT_CAUSE.md): fail fast after S2/DEK
MAX_VISION_OK_PCT="${MAX_VISION_OK_PCT:-20}"
TEXT_TOP1_MIN_DELTA_PP="${TEXT_TOP1_MIN_DELTA_PP:--5}"
DIAG_L0_OUT=doc/results/cvpr2027/g5/llava_l0_diagnosis_m7.json

echo "===== M7 start $(date -Is) ====="
echo "TEXT_TOP1_MIN_PCT=$TEXT_TOP1_MIN_PCT TRAIN_JSON=$TRAIN_JSON"
echo "MAX_VISION_OK_PCT=$MAX_VISION_OK_PCT TEXT_TOP1_MIN_DELTA_PP=$TEXT_TOP1_MIN_DELTA_PP"

run_routing_health_gate() {
  # $1=stage label (S2|DEK)  $2=ckpt  [$3=--dek_dual_gate]
  local stage="$1"
  local ckpt="$2"
  local extra=()
  if [ "${3:-}" = "--dek_dual_gate" ]; then
    extra+=(--dek_dual_gate)
  fi
  local out="doc/results/cvpr2027/g5/routing_health_${stage,,}_m7.json"
  echo "===== Routing health gate ($stage) ====="
  python -u tools/diagnose_llava_l0.py --health_gate \
    -ckpt "$ckpt" -dpath "$TRAIN_JSON" -img "$IMG" \
    -dsn "$DSN_DIAG" -seed 42 -dvc cuda:0 -out "$out" \
    --baseline_json "$DIAG_L0_OUT" \
    --max_vision_ok_pct="$MAX_VISION_OK_PCT" \
    --text_top1_min_delta_pp="$TEXT_TOP1_MIN_DELTA_PP" \
    "${extra[@]}"
}

pick_bs() {
  # Prefer 4, then 2. Always -dbs 1 (prefetch buffer OOMs on 24GB with LLaVA).
  # Status lines go to stderr so BS=$(pick_bs) only captures the integer.
  # Clear GPU between tries — a failed bs=4 smoke fragments memory for bs=2.
  export PYTORCH_ALLOC_CONF=expandable_segments:True
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  for try_bs in 4 2; do
    echo "===== VRAM smoke bs=$try_bs dbs=1 (1 epoch, dn=8) =====" >&2
    python - <<'PY' >&2 || true
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
print("cuda free (approx):", torch.cuda.mem_get_info()[0] // (1024**2), "MiB")
PY
    smoke_log="$LOGDIR/m7_smoke_bs${try_bs}.log"
    set +e
    python -u train_vllm_editor.py -en liveedit -mn llava -dna VLKEB \
      -bs "$try_bs" -dbs 1 -dvc cuda:0 -edvc 0 -eps 1 -tnp "m7_smoke_bs${try_bs}" \
      -dn 8 -rs 42 -sci 100 -lpi 1 \
      -dpath "$TRAIN_JSON" \
      --keep_latest_ckpts 1 \
      --records_dir records/lab_g5_smoke \
      >"$smoke_log" 2>&1
    rc=$?
    set -e
    if grep -qiE 'out of memory|CUDA out of memory' "$smoke_log"; then
      echo "smoke bs=$try_bs OOM" >&2
      continue
    fi
    if [ "$rc" -ne 0 ]; then
      echo "smoke bs=$try_bs failed rc=$rc (see $smoke_log)" >&2
      tail -n 40 "$smoke_log" >&2 || true
      continue
    fi
    echo "smoke bs=$try_bs OK" >&2
    echo "$try_bs"
    return 0
  done
  echo "FAIL: no workable bs in {4,2} even with -dbs 1" >&2
  return 1
}

# Skip re-smoke if caller already proved a batch size (e.g. FORCE_BS=2).
if [ -n "${FORCE_BS:-}" ]; then
  BS="$FORCE_BS"
  echo "===== Using FORCE_BS=$BS (skip smoke) ====="
else
  BS=$(pick_bs)
  echo "===== Selected BS=$BS ====="
fi
case "$BS" in
  2|4) ;;
  *) echo "FAIL: invalid BS='$BS'" >&2; exit 1 ;;
esac

echo "===== L0 full dn=500 eps=20 bs=$BS ====="
python -u tools/lab_g5_llava_fresh_pipeline.py --stages L0 -bs "$BS" --eps_l0 20

L0_CKPT=$(python - <<'PY'
from pathlib import Path
root = Path("records/lab_g5/liveedit/llava-v1.5-7b")
cands = sorted(
    root.glob("lab_g5_llava_L0_bs*_dn500_e20_*/checkpoints/epoch-*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
print(cands[0] if cands else "")
PY
)
if [ -z "$L0_CKPT" ] || [ ! -f "$L0_CKPT" ]; then
  echo "FAIL: no L0 ckpt after train"
  exit 1
fi
echo "L0_CKPT=$L0_CKPT"

echo "===== L0 success gate (diagnose train-split) ====="
python -u tools/diagnose_llava_l0.py \
  -ckpt "$L0_CKPT" -dpath "$TRAIN_JSON" -img "$IMG" \
  -dsn "$DSN_DIAG" -seed 42 -dvc cuda:0 -out "$DIAG_L0_OUT"

python - <<PY
import json, sys
j = json.load(open("$DIAG_L0_OUT", encoding="utf-8"))
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

echo "===== S2 from L0 (bs=$BS) ====="
python -u tools/lab_g5_llava_fresh_pipeline.py --stages S2 -bs "$BS" --eps_ft 1 --from_l0_ckpt "$L0_CKPT"

S2_CKPT=$(python - <<'PY'
from pathlib import Path
root = Path("records/lab_g5/liveedit/llava-v1.5-7b")
cands = sorted(
    root.glob("lab_g5_llava_S2_bs*_dn500_e1_*/checkpoints/epoch-*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
print(cands[0] if cands else "")
PY
)
echo "S2_CKPT=$S2_CKPT"
[ -n "$S2_CKPT" ] && [ -f "$S2_CKPT" ] || { echo "FAIL: missing S2 ckpt"; exit 1; }
run_routing_health_gate S2 "$S2_CKPT"

echo "===== DEK from L0 (bs=$BS) ====="
python -u tools/lab_g5_llava_fresh_pipeline.py --stages DEK -bs "$BS" --eps_ft 1 --from_l0_ckpt "$L0_CKPT"

DEK_CKPT=$(python - <<'PY'
from pathlib import Path
root = Path("records/lab_g5/liveedit/llava-v1.5-7b")
cands = sorted(
    root.glob("lab_g5_llava_DEK_bs*_dn500_e1_*/checkpoints/epoch-*"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
print(cands[0] if cands else "")
PY
)
echo "DEK_CKPT=$DEK_CKPT"
[ -n "$DEK_CKPT" ] && [ -f "$DEK_CKPT" ] || { echo "FAIL: missing DEK ckpt"; exit 1; }
run_routing_health_gate DEK "$DEK_CKPT" --dek_dual_gate

echo "===== Eval L0 ====="
python -u -W ignore test_grounding_eval.py -en liveedit -mn llava -sen 10 -dvc cuda:0 \
  -dsn "$DSN_EVAL" -seed 42 -ckpt "$L0_CKPT" -enp g5_llava_m7_L0_short \
  -dpath "$PROBE" --meeting_step 6 --log_routing

echo "===== Eval S2 ====="
python -u -W ignore test_grounding_eval.py -en liveedit -mn llava -sen 10 -dvc cuda:0 \
  -dsn "$DSN_EVAL" -seed 42 -ckpt "$S2_CKPT" -enp g5_llava_m7_S2_short \
  -dpath "$PROBE" --meeting_step 6 --log_routing --patch_grounding

echo "===== Eval DEK ====="
python -u -W ignore test_grounding_eval.py -en liveedit -mn llava -sen 10 -dvc cuda:0 \
  -dsn "$DSN_EVAL" -seed 42 -ckpt "$DEK_CKPT" -enp g5_llava_m7_DEK_short \
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

echo "===== M7 pipeline OK $(date -Is) ====="
echo "BS=$BS L0=$L0_CKPT OUTER=$OUTER"
echo "Fill doc/results/cvpr2027/g5/G5_LLAVA_M7_BS_RETRAIN.md from artifacts."
