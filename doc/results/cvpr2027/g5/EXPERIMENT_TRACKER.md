# G5 LLaVA lab experiment tracker

Fresh trains on lab GPU (no reused home-PC checkpoints). Keep latest 3 checkpoints per run.

**Lab host:** `user@100.89.137.40` (`29`)  
**Lab repo:** `/home/user/TWD/LiveEdit_lab_usb/LiveEdit`  
**Lab backup:** `/home/user/TWD/LiveEdit_lab_usb/experiment_backup/g5/`  
**Local mirror:** this folder + `lab_mirror/`

## How to sync

```powershell
# Code: commit+push lab/g5-code-only on this PC (ask first), then on lab:
#   bash tools/lab_git_pull_on_lab.sh

# Results: pull tracker/logs lab -> home
powershell -ExecutionPolicy Bypass -File tools/lab_sync_pull_results.ps1 -IncludeCkptMeta
```

## Pipeline (4090)

```bash
# on lab
cd /home/user/TWD/LiveEdit_lab_usb/LiveEdit
source ~/miniconda3/etc/profile.d/conda.sh && conda activate liveedit
python -u tools/lab_g5_llava_fresh_pipeline.py --stages L0 S2 DEK
```

Defaults: `bs=1`, `dn=500`, `eps=1`, `sci=100`, `keep_latest_ckpts=3`, seed 42.  
Stages: **L0** (scratch) → **S2** (from new L0) → **DEK** (from new L0). Old USB L0/S2 ckpts are **not** used.

---
