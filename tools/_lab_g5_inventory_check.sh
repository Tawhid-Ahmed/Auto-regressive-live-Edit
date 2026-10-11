#!/usr/bin/env bash
set -u
BASE=/home/user/TWD/LiveEdit_lab_usb
REPO="$BASE/LiveEdit"

echo "=== HOST ==="
hostname
whoami
date
pwd

echo
echo "=== USB TREE (top) ==="
ls -la "$BASE" 2>&1 | head -50

echo
echo "=== REPO ROOT ==="
if [ -d "$REPO" ]; then
  ls -la "$REPO" | head -40
else
  echo "MISS $REPO"
fi

echo
echo "=== KEY CODE FILES ==="
for p in \
  "$REPO/tools/run_g5_llava_train.py" \
  "$REPO/tools/smoke_g5_llava_load.py" \
  "$REPO/configs/liveedit/llava-v1.5-7b.yaml" \
  "$REPO/train_vllm_editor.py" \
  "$REPO/test_grounding_eval.py" \
  "$REPO/utils/GLOBAL.py"
do
  if [ -e "$p" ]; then echo "OK   $p"; else echo "MISS $p"; fi
done

echo
echo "=== MODELS / CKPTS / DATA ==="
for p in \
  "$BASE/models/llava-1.5-7b-hf" \
  "$REPO/../models/llava-1.5-7b-hf" \
  "/home/user/TWD/models/llava-1.5-7b-hf" \
  "$REPO/data/VLKEB/grounding/grounding_train_short.json" \
  "$REPO/data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json" \
  "$REPO/data/VLKEB/VLKEB_images" \
  "$REPO/data/VLKEB/VLKEB_images/mmkb_images"
do
  if [ -e "$p" ]; then
    if [ -d "$p" ]; then
      n=$(find "$p" -type f 2>/dev/null | wc -l)
      echo "OK   $p  (files=$n)"
    else
      sz=$(du -h "$p" 2>/dev/null | cut -f1)
      echo "OK   $p  (size=$sz)"
    fi
  else
    echo "MISS $p"
  fi
done

echo
echo "=== FIND LLaVA WEIGHTS / CKPTS ==="
find /home/user/TWD -maxdepth 5 -type d -name 'llava-1.5-7b-hf' 2>/dev/null | head -10
find /home/user/TWD -maxdepth 6 -type d -name 'VLKEB_llava_L0_dn500*' 2>/dev/null | head -10
find /home/user/TWD -maxdepth 6 -type d -name 'g5_vlkeb_llava_*' 2>/dev/null | head -20
find /home/user/TWD -maxdepth 8 -type d -name 'checkpoints' 2>/dev/null | rg -i 'llava' | head -20

echo
echo "=== GLOBAL.py ROOT_PATH (if present) ==="
if [ -f "$REPO/utils/GLOBAL.py" ]; then
  rg -n "ROOT_PATH|llava|models" "$REPO/utils/GLOBAL.py" | head -40
fi

echo
echo "=== CONDA / ENV ==="
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/opt/conda/etc/profile.d/conda.sh"
fi
command -v conda || echo "conda not on PATH"
conda env list 2>&1 | head -30

echo
echo "=== LIVEEDIT ENV PYTHON/TORCH ==="
if conda activate liveedit 2>/dev/null; then
  python - <<'PY'
import sys
print("py", sys.version.replace("\n", " "))
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
        print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
except Exception as e:
    print("torch_error", e)
for mod in ("transformers", "peft", "PIL"):
    try:
        m = __import__(mod)
        print(mod, getattr(m, "__version__", "ok"))
    except Exception as e:
        print(mod, "MISSING", e)
PY
else
  echo "FAIL: conda activate liveedit"
fi

echo
echo "=== GPU ==="
nvidia-smi -L 2>&1
nvidia-smi --query-gpu=name,memory.total,driver_version,utilization.gpu --format=csv,noheader 2>&1

echo
echo "=== DISK ==="
df -h /home/user 2>&1 | head -5
du -sh "$BASE" 2>/dev/null || true

echo
echo "=== DONE ==="
