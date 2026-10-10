# G5 LLaVA lab experiment tracker

Fresh trains on lab GPU (no reused home-PC checkpoints). Keep latest 3 checkpoints per run.

**Lab host:** `user@100.89.137.40` (`29`)  
**Lab repo:** `/home/user/TWD/LiveEdit_lab_usb/LiveEdit`  
**Lab backup:** `/home/user/TWD/LiveEdit_lab_usb/experiment_backup/g5/`  
**Local mirror:** this folder + `lab_mirror/`

## CVPR protocol (locked Oct 10)

| Stage | `dn` | Epochs | Role |
|---|---|---|---|
| L0 | 500 | **20** | Strong short-n LiveEdit L0 (CVPR); not the Oct 10 `eps=1` smoke |
| S2 | 500 | **1** | AVBT from new lab L0 |
| DEK | 500 | **1** | AVBT + dual gate from new lab L0 |

**This week after trains:** extract LLaVA results, then **re-extract BLIP-2** L0/S2/DEK with the same eval + taxonomy scripts.

## How to sync

```powershell
# Code: commit+push lab/g5-code-only on this PC (ask first), then on lab:
#   bash tools/lab_git_pull_on_lab.sh

# Results: pull tracker/logs lab -> home
powershell -ExecutionPolicy Bypass -File tools/lab_sync_pull_results.ps1 -IncludeCkptMeta
```

## Pipeline (4090)

```bash
# on lab — CVPR defaults: --eps_l0 20 --eps_ft 1
cd /home/user/TWD/LiveEdit_lab_usb/LiveEdit
source ~/miniconda3/etc/profile.d/conda.sh && conda activate liveedit
nohup python -u tools/lab_g5_llava_fresh_pipeline.py --stages L0 S2 DEK \
  > doc/results/cvpr2027/g5/logs/ladder_outer.log 2>&1 &
```

Stages: **L0** (scratch) → **S2** (from new L0) → **DEK** (from new L0). Old USB / home L0/S2 ckpts are **not** used for paper numbers.

---
