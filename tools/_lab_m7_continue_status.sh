#!/usr/bin/env bash
set -euo pipefail
REPO=/home/user/TWD/LiveEdit_lab_usb/LiveEdit
LOGDIR=$REPO/doc/results/cvpr2027/g5/logs
date
echo "--- procs ---"
ps aux | grep -E 'm7_continue|diagnose_llava|lab_g5_llava|train_vllm|test_grounding|score_leakage' | grep -v grep || echo '(none)'
echo "--- gpu ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "--- continue log ---"
CONT=$(ls -t "$LOGDIR"/m7_continue_2*.log 2>/dev/null | head -1 || true)
echo "CONT=${CONT:-none}"
if [ -n "${CONT:-}" ]; then
  grep -E 'PASS:|FAIL:|S2|DEK|gate|Taxonomy|OK|Error|OOM|Eval |START|END|Selected|Using FORCE|L0_CKPT|text_top1' "$CONT" | tail -n 40 || true
  echo "--- tail ---"
  tail -n 20 "$CONT"
fi
echo "--- diag ---"
DIAG=$REPO/doc/results/cvpr2027/g5/llava_l0_diagnosis_m7.json
if [ -f "$DIAG" ]; then
  python3 -c "import json; j=json.load(open('$DIAG')); print(json.dumps(j['train_split_reliability'], indent=2))"
else
  echo 'diag json not ready'
fi
