#!/usr/bin/env bash
# Run ON the lab PC after the lab branch is pushed to origin.
# Converts the USB tree into a git checkout of lab/g5-code-only (keeps models/data/records).
set -euo pipefail
REPO="${1:-/home/user/TWD/LiveEdit_lab_usb/LiveEdit}"
BRANCH="${2:-lab/g5-code-only}"
REMOTE_URL="${3:-https://github.com/Tawhid-Ahmed/Auto-regressive-live-Edit.git}"

cd "$REPO"
if [ ! -d .git ]; then
  echo "Initializing git in $REPO (USB copy was not a repo)..."
  git init
  git remote add origin "$REMOTE_URL" 2>/dev/null || git remote set-url origin "$REMOTE_URL"
  # Preserve large local assets from being wiped if they are untracked
  cat >> .git/info/exclude <<'EOF'
records/
data/VLKEB/VLKEB_images/
*.pt
*.bin
*.safetensors
EOF
fi

git fetch origin "$BRANCH"
# Prefer a clean tracking branch; keep local untracked models/data.
# First-time USB→git: working tree files exist but are untracked, so force
# checkout once to adopt origin (large assets stay via .git/info/exclude).
if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
else
  git checkout -f -B "$BRANCH" "origin/${BRANCH}"
fi

echo "Lab code now at:"
git rev-parse --short HEAD
git status -sb
mkdir -p doc/results/cvpr2027/g5/logs \
  /home/user/TWD/LiveEdit_lab_usb/experiment_backup/g5
echo "OK — ready to run tools/lab_g5_llava_fresh_pipeline.py"
