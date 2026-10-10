# G5 — Second backbone (LLaVA) on lab GPU

**Started:** 2026-10-06  
**Decision due:** 2026-10-15  
**Gate log:** [`../../CVPR2027_GATE_LOG.md`](../../CVPR2027_GATE_LOG.md) § G5

## Goal

Show DEK is not BLIP-2-only. Ladder: **LLaVA-1.5-7B** → stronger GPU (preferred) → **Qwen2-VL-2B** fallback.

## Branch workflow (this PC vs lab)

| Machine | Branch | Role |
|---|---|---|
| This PC | `feature/working-on-aaai-review-for-04-10-2026` | Docs + paper kit + code; day-to-day work |
| Lab GPU | `lab/g5-code-only` | Code-only tree for training (no AuthorKit / bulk docs) |

Keep this docs branch updated when lab code changes (cherry-pick or merge from `lab/g5-code-only`). Push lab code on `lab/g5-code-only` after approval; pull on lab in a **visible** SSH terminal.

## Policy (lab runs)

| Rule | Detail |
|---|---|
| Fresh trains | Do **not** reuse home-PC / USB L0–S2–DEK checkpoints for the lab G5 ladder |
| Hardware | RTX 4090 ~24 GB; `bs=1`, fp16, `PYTORCH_ALLOC_CONF=expandable_segments:True` |
| Checkpoints | `--keep_latest_ckpts 3` (only newest 3 files kept per run) |
| Code sync | Commit on `lab/g5-code-only` → push origin → on lab (visible SSH): `bash tools/lab_git_pull_on_lab.sh` |
| Results sync | Pull tracker/logs with `tools/lab_sync_pull_results.ps1` (SCP of small artifacts only) |
| Backup on lab | `/home/user/TWD/LiveEdit_lab_usb/experiment_backup/g5/` |
| Commits | Always ask before commit/push; never add Cursor as co-author |

## Lab paths

- Code: `/home/user/TWD/LiveEdit_lab_usb/LiveEdit`
- Weights: `/home/user/TWD/LiveEdit_lab_usb/models/llava-1.5-7b-hf`
- Runs: `records/lab_g5/liveedit/llava-v1.5-7b/`
- Tracker: `doc/results/cvpr2027/g5/EXPERIMENT_TRACKER.md`

## Local PC status (historical — Oct 7)

Long runs stopped on the home machine; continue on lab 4090 with **fresh** L0→S2→DEK (do not resume from the old home L0/S2 ckpts listed below).

| Asset | Status |
|---|---|
| LLaVA weights | present locally / on lab USB |
| Old home L0 / S2 ckpts | archive only — **not** for lab G5 ladder |
| Lab DEK | train fresh from new lab L0 |

## Run pipeline (lab — visible SSH)

```bash
ssh user@100.89.137.40
cd /home/user/TWD/LiveEdit_lab_usb/LiveEdit
source ~/miniconda3/etc/profile.d/conda.sh && conda activate liveedit
python -u tools/lab_g5_llava_fresh_pipeline.py --stages L0 S2 DEK
```

Or start L0 only, then continue:

```bash
python -u tools/lab_g5_llava_fresh_pipeline.py --stages L0
python -u tools/lab_g5_llava_fresh_pipeline.py --stages S2 DEK --from_l0_ckpt <new_lab_L0_ckpt>
```

## After train — eval + taxonomy

Use the **new lab** L0 / S2 / DEK ckpts from `records/lab_g5/...` (not USB copies under `records/liveedit/...`).

```bash
PROBE=data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json
IMG=data/VLKEB/VLKEB_images/mmkb_images
# fill ckpts from EXPERIMENT_TRACKER.md
python -u -W ignore test_grounding_eval.py ... --log_routing
python -u tools/score_leakage_taxonomy.py ...
```

## Sync code (git — preferred)

**On this PC** (docs branch stays checked out):

```powershell
git checkout feature/working-on-aaai-review-for-04-10-2026
# edit code/docs here; cherry-pick or merge lab commits when needed
```

**Lab code push** (after you approve commit + push):

```powershell
git checkout lab/g5-code-only
# bring over only the lab-needed code commits
git commit -m "..."   # no Cursor co-author
git push -u origin lab/g5-code-only
git checkout feature/working-on-aaai-review-for-04-10-2026
```

**On lab PC** (visible SSH terminal in the IDE — you should see the pane):

```bash
cd /home/user/TWD/LiveEdit_lab_usb/LiveEdit
bash tools/lab_git_pull_on_lab.sh
```

## Sync results (small files only)

```powershell
powershell -ExecutionPolicy Bypass -File tools/lab_sync_pull_results.ps1 -IncludeCkptMeta
```

## Checklist

- [x] Local smoke PASS (home PC, earlier)
- [x] Lab env + USB tree inventory (Oct 10)
- [x] Keep-latest-3 ckpt retention in trainer
- [x] Fresh pipeline + experiment tracker + sync scripts
- [ ] Lab: L0 train (fresh)
- [ ] Lab: S2 train from new L0
- [ ] Lab: DEK train from new L0
- [ ] Lab: grounding eval + taxonomy
- [ ] Pull results to this repo
- [ ] Fill G5 Result in gate log by Oct 15

## Deferred (after G5)

M6 hn method try; `emits_y_i` recovery; EVQA pair-closed curve; full 5-seed; G4; MSCKE; CARML tau.
