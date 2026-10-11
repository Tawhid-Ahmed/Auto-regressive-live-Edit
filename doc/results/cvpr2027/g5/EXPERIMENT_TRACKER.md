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

### L0 — `lab_g5_llava_L0_dn500_e20_s42_20261010_231536`
- **time (UTC):** 2026-10-10T17:15:36Z
- **status:** started
- **host:** 29
- **cmd:** `/home/user/miniconda3/envs/liveedit/bin/python -u /home/user/TWD/LiveEdit_lab_usb/LiveEdit/train_vllm_editor.py -en liveedit -mn llava -dna VLKEB -bs 1 -dvc cuda:0 -edvc 0 -eps 20 -tnp lab_g5_llava_L0_dn500_e20_s42_20261010_231536 -dn 500 -rs 42 -sci 100 -lpi 1 -dpath /home/user/TWD/LiveEdit_lab_usb/LiveEdit/data/VLKEB/grounding/grounding_train_short.json --keep_latest_ckpts 3 --records_dir /home/user/TWD/LiveEdit_lab_usb/LiveEdit/records/lab_g5`
- **log:** `doc/results/cvpr2027/g5/logs/lab_g5_llava_L0_dn500_e20_s42_20261010_231536.train.log`
- **note:** fresh train; no home-PC checkpoints; keep_latest_ckpts=3

### S2 — `lab_g5_llava_S2_dn500_e1_s42_20261011_103849`
- **time (UTC):** 2026-10-11T04:38:49Z
- **status:** started
- **host:** 29
- **cmd:** `/home/user/miniconda3/envs/liveedit/bin/python -u /home/user/TWD/LiveEdit_lab_usb/LiveEdit/train_vllm_editor.py -en liveedit -mn llava -dna VLKEB -bs 1 -dvc cuda:0 -edvc 0 -eps 1 -tnp lab_g5_llava_S2_dn500_e1_s42_20261011_103849 -dn 500 -rs 42 -sci 100 -lpi 1 -dpath /home/user/TWD/LiveEdit_lab_usb/LiveEdit/data/VLKEB/grounding/grounding_train_short.json --keep_latest_ckpts 3 --records_dir /home/user/TWD/LiveEdit_lab_usb/LiveEdit/records/lab_g5 -lkpt records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_dn500_e20_s42_20261010_231536-2026.10.10-23.15.41/checkpoints/epoch-20-i-10200-ema_loss-2.2061 --patch_grounding --adversarial_grounding --adv_grounding_lambda 0.3 --adv_grounding_neg_weight 2.5 --avbl_hinge_tau 0.3 --avbl_contrastive`
- **log:** `doc/results/cvpr2027/g5/logs/lab_g5_llava_S2_dn500_e1_s42_20261011_103849.train.log`
- **note:** fresh train; no home-PC checkpoints; keep_latest_ckpts=3

### S2 — `lab_g5_llava_S2_dn500_e1_s42_20261011_103849`
- **time (UTC):** 2026-10-11T04:40:36Z
- **status:** failed
- **host:** 29
- **cmd:** `/home/user/miniconda3/envs/liveedit/bin/python -u /home/user/TWD/LiveEdit_lab_usb/LiveEdit/train_vllm_editor.py -en liveedit -mn llava -dna VLKEB -bs 1 -dvc cuda:0 -edvc 0 -eps 1 -tnp lab_g5_llava_S2_dn500_e1_s42_20261011_103849 -dn 500 -rs 42 -sci 100 -lpi 1 -dpath /home/user/TWD/LiveEdit_lab_usb/LiveEdit/data/VLKEB/grounding/grounding_train_short.json --keep_latest_ckpts 3 --records_dir /home/user/TWD/LiveEdit_lab_usb/LiveEdit/records/lab_g5 -lkpt records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_dn500_e20_s42_20261010_231536-2026.10.10-23.15.41/checkpoints/epoch-20-i-10200-ema_loss-2.2061 --patch_grounding --adversarial_grounding --adv_grounding_lambda 0.3 --adv_grounding_neg_weight 2.5 --avbl_hinge_tau 0.3 --avbl_contrastive`
- **run_dir:** `records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_S2_dn500_e1_s42_20261011_103849-2026.10.11-10.38.54`
- **log:** `doc/results/cvpr2027/g5/logs/lab_g5_llava_S2_dn500_e1_s42_20261011_103849.train.log`
- **exit:** 1

### S2 — `lab_g5_llava_S2_dn500_e1_s42_20261011_104244`
- **time (UTC):** 2026-10-11T04:42:44Z
- **status:** started
- **host:** 29
- **cmd:** `/home/user/miniconda3/envs/liveedit/bin/python -u /home/user/TWD/LiveEdit_lab_usb/LiveEdit/train_vllm_editor.py -en liveedit -mn llava -dna VLKEB -bs 1 -dvc cuda:0 -edvc 0 -eps 1 -tnp lab_g5_llava_S2_dn500_e1_s42_20261011_104244 -dn 500 -rs 42 -sci 100 -lpi 1 -dpath /home/user/TWD/LiveEdit_lab_usb/LiveEdit/data/VLKEB/grounding/grounding_train_short.json --keep_latest_ckpts 3 --records_dir /home/user/TWD/LiveEdit_lab_usb/LiveEdit/records/lab_g5 -lkpt records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_dn500_e20_s42_20261010_231536-2026.10.10-23.15.41/checkpoints/epoch-20-i-10200-ema_loss-2.2061 -dbs 1 --patch_grounding --adversarial_grounding --adv_grounding_lambda 0.3 --adv_grounding_neg_weight 2.5 --avbl_hinge_tau 0.3 --avbl_contrastive`
- **log:** `doc/results/cvpr2027/g5/logs/lab_g5_llava_S2_dn500_e1_s42_20261011_104244.train.log`
- **note:** fresh train; no home-PC checkpoints; keep_latest_ckpts=3

### S2 — `lab_g5_llava_S2_dn500_e1_s42_20261011_104244`
- **time (UTC):** 2026-10-11T04:44:16Z
- **status:** failed
- **host:** 29
- **cmd:** `/home/user/miniconda3/envs/liveedit/bin/python -u /home/user/TWD/LiveEdit_lab_usb/LiveEdit/train_vllm_editor.py -en liveedit -mn llava -dna VLKEB -bs 1 -dvc cuda:0 -edvc 0 -eps 1 -tnp lab_g5_llava_S2_dn500_e1_s42_20261011_104244 -dn 500 -rs 42 -sci 100 -lpi 1 -dpath /home/user/TWD/LiveEdit_lab_usb/LiveEdit/data/VLKEB/grounding/grounding_train_short.json --keep_latest_ckpts 3 --records_dir /home/user/TWD/LiveEdit_lab_usb/LiveEdit/records/lab_g5 -lkpt records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_dn500_e20_s42_20261010_231536-2026.10.10-23.15.41/checkpoints/epoch-20-i-10200-ema_loss-2.2061 -dbs 1 --patch_grounding --adversarial_grounding --adv_grounding_lambda 0.3 --adv_grounding_neg_weight 2.5 --avbl_hinge_tau 0.3 --avbl_contrastive`
- **run_dir:** `records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_S2_dn500_e1_s42_20261011_104244-2026.10.11-10.42.48`
- **log:** `doc/results/cvpr2027/g5/logs/lab_g5_llava_S2_dn500_e1_s42_20261011_104244.train.log`
- **exit:** 1

### S2 — `lab_g5_llava_S2_dn500_e1_s42_20261011_104554`
- **time (UTC):** 2026-10-11T04:45:54Z
- **status:** started
- **host:** 29
- **cmd:** `/home/user/miniconda3/envs/liveedit/bin/python -u /home/user/TWD/LiveEdit_lab_usb/LiveEdit/train_vllm_editor.py -en liveedit -mn llava -dna VLKEB -bs 1 -dvc cuda:0 -edvc 0 -eps 1 -tnp lab_g5_llava_S2_dn500_e1_s42_20261011_104554 -dn 500 -rs 42 -sci 100 -lpi 1 -dpath /home/user/TWD/LiveEdit_lab_usb/LiveEdit/data/VLKEB/grounding/grounding_train_short.json --keep_latest_ckpts 3 --records_dir /home/user/TWD/LiveEdit_lab_usb/LiveEdit/records/lab_g5 -lkpt records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_dn500_e20_s42_20261010_231536-2026.10.10-23.15.41/checkpoints/epoch-20-i-10200-ema_loss-2.2061 -dbs 1 --patch_grounding --adversarial_grounding --adv_grounding_lambda 0.3 --adv_grounding_neg_weight 2.5 --avbl_hinge_tau 0.3 --avbl_contrastive`
- **log:** `doc/results/cvpr2027/g5/logs/lab_g5_llava_S2_dn500_e1_s42_20261011_104554.train.log`
- **note:** fresh train; no home-PC checkpoints; keep_latest_ckpts=3

### S2 — `lab_g5_llava_S2_dn500_e1_s42_20261011_104554`
- **time (UTC):** 2026-10-11T05:15:42Z
- **status:** finished
- **host:** 29
- **cmd:** `/home/user/miniconda3/envs/liveedit/bin/python -u /home/user/TWD/LiveEdit_lab_usb/LiveEdit/train_vllm_editor.py -en liveedit -mn llava -dna VLKEB -bs 1 -dvc cuda:0 -edvc 0 -eps 1 -tnp lab_g5_llava_S2_dn500_e1_s42_20261011_104554 -dn 500 -rs 42 -sci 100 -lpi 1 -dpath /home/user/TWD/LiveEdit_lab_usb/LiveEdit/data/VLKEB/grounding/grounding_train_short.json --keep_latest_ckpts 3 --records_dir /home/user/TWD/LiveEdit_lab_usb/LiveEdit/records/lab_g5 -lkpt records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_dn500_e20_s42_20261010_231536-2026.10.10-23.15.41/checkpoints/epoch-20-i-10200-ema_loss-2.2061 -dbs 1 --patch_grounding --adversarial_grounding --adv_grounding_lambda 0.3 --adv_grounding_neg_weight 2.5 --avbl_hinge_tau 0.3 --avbl_contrastive`
- **run_dir:** `records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_S2_dn500_e1_s42_20261011_104554-2026.10.11-10.45.58`
- **ckpt:** `records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_S2_dn500_e1_s42_20261011_104554-2026.10.11-10.45.58/checkpoints/epoch-1-i-500-ema_loss-3.8904`
- **log:** `doc/results/cvpr2027/g5/logs/lab_g5_llava_S2_dn500_e1_s42_20261011_104554.train.log`
- **exit:** 0

### DEK — `lab_g5_llava_DEK_dn500_e1_s42_20261011_111542`
- **time (UTC):** 2026-10-11T05:15:42Z
- **status:** started
- **host:** 29
- **cmd:** `/home/user/miniconda3/envs/liveedit/bin/python -u /home/user/TWD/LiveEdit_lab_usb/LiveEdit/train_vllm_editor.py -en liveedit -mn llava -dna VLKEB -bs 1 -dvc cuda:0 -edvc 0 -eps 1 -tnp lab_g5_llava_DEK_dn500_e1_s42_20261011_111542 -dn 500 -rs 42 -sci 100 -lpi 1 -dpath /home/user/TWD/LiveEdit_lab_usb/LiveEdit/data/VLKEB/grounding/grounding_train_short.json --keep_latest_ckpts 3 --records_dir /home/user/TWD/LiveEdit_lab_usb/LiveEdit/records/lab_g5 -lkpt records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_dn500_e20_s42_20261010_231536-2026.10.10-23.15.41/checkpoints/epoch-20-i-10200-ema_loss-2.2061 -dbs 1 --patch_grounding --adversarial_grounding --adv_grounding_lambda 0.3 --adv_grounding_neg_weight 2.5 --avbl_hinge_tau 0.3 --avbl_contrastive --dek_dual_gate`
- **log:** `doc/results/cvpr2027/g5/logs/lab_g5_llava_DEK_dn500_e1_s42_20261011_111542.train.log`
- **note:** fresh train; no home-PC checkpoints; keep_latest_ckpts=3

### DEK — `lab_g5_llava_DEK_dn500_e1_s42_20261011_111542`
- **time (UTC):** 2026-10-11T05:46:04Z
- **status:** finished
- **host:** 29
- **cmd:** `/home/user/miniconda3/envs/liveedit/bin/python -u /home/user/TWD/LiveEdit_lab_usb/LiveEdit/train_vllm_editor.py -en liveedit -mn llava -dna VLKEB -bs 1 -dvc cuda:0 -edvc 0 -eps 1 -tnp lab_g5_llava_DEK_dn500_e1_s42_20261011_111542 -dn 500 -rs 42 -sci 100 -lpi 1 -dpath /home/user/TWD/LiveEdit_lab_usb/LiveEdit/data/VLKEB/grounding/grounding_train_short.json --keep_latest_ckpts 3 --records_dir /home/user/TWD/LiveEdit_lab_usb/LiveEdit/records/lab_g5 -lkpt records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_dn500_e20_s42_20261010_231536-2026.10.10-23.15.41/checkpoints/epoch-20-i-10200-ema_loss-2.2061 -dbs 1 --patch_grounding --adversarial_grounding --adv_grounding_lambda 0.3 --adv_grounding_neg_weight 2.5 --avbl_hinge_tau 0.3 --avbl_contrastive --dek_dual_gate`
- **run_dir:** `records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_DEK_dn500_e1_s42_20261011_111542-2026.10.11-11.16.06`
- **ckpt:** `records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_DEK_dn500_e1_s42_20261011_111542-2026.10.11-11.16.06/checkpoints/epoch-1-i-500-ema_loss-3.8851`
- **log:** `doc/results/cvpr2027/g5/logs/lab_g5_llava_DEK_dn500_e1_s42_20261011_111542.train.log`
- **exit:** 0

### EVAL â€” LLaVA grounding + taxonomy
- **time (UTC):** 2026-10-11T06:30:20Z
- **status:** finished
- **L0 ckpt:** `records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_dn500_e20_s42_20261010_231536-2026.10.10-23.15.41/checkpoints/epoch-20-i-10200-ema_loss-2.2061`
- **S2 ckpt:** `records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_S2_dn500_e1_s42_20261011_104554-2026.10.11-10.45.58/checkpoints/epoch-1-i-500-ema_loss-3.8904`
- **DEK ckpt:** `records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_DEK_dn500_e1_s42_20261011_111542-2026.10.11-11.16.06/checkpoints/epoch-1-i-500-ema_loss-3.8851`
- **L0 run:** `eval_results/meeting_validation/step6/liveedit/llava-v1.5-7b/VLKEB_GROUNDING-g5_llava_L0_short-2026.10.11-12.10.05`
- **S2 run:** `eval_results/meeting_validation/step6/liveedit/llava-v1.5-7b/VLKEB_GROUNDING-g5_llava_S2_short-2026.10.11-12.14.11`
- **DEK run:** `eval_results/meeting_validation/step6/liveedit/llava-v1.5-7b/VLKEB_GROUNDING-g5_llava_DEK_short-2026.10.11-12.22.23`
- **taxonomy:** `doc/results/cvpr2027/g5/leakage_taxonomy_llava.json`
- **logs:** `doc/results/cvpr2027/g5/logs/eval_*.log`

### EVAL — LLaVA Easy-VQA grounding + taxonomy (E1)
- **time (UTC):** 2026-10-11T09:34:16Z
- **status:** finished
- **probe:** `data/easy-edit-mm/grounding/grounding_eval_short.json`
- **img_root:** `data/easy-edit-mm/images`
- **L0 run:** `eval_results/meeting_validation/step6/liveedit/llava-v1.5-7b/VLKEB_GROUNDING-g5_llava_evqa_L0_short-2026.10.11-15.12.53`
- **S2 run:** `eval_results/meeting_validation/step6/liveedit/llava-v1.5-7b/VLKEB_GROUNDING-g5_llava_evqa_S2_short-2026.10.11-15.19.15`
- **DEK run:** `eval_results/meeting_validation/step6/liveedit/llava-v1.5-7b/VLKEB_GROUNDING-g5_llava_evqa_DEK_short-2026.10.11-15.27.07`
- **taxonomy:** `doc/results/cvpr2027/g5/leakage_taxonomy_llava_evqa.json`
