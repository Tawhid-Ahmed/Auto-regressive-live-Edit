# G5 — Second backbone (LLaVA) on lab GPU

**Started:** 2026-10-06  
**Decision due:** 2026-10-15  
**Gate log:** [`../../CVPR2027_GATE_LOG.md`](../../CVPR2027_GATE_LOG.md) § G5  
**Results tables:** [`G5_RESULTS_TABLES.md`](G5_RESULTS_TABLES.md) (LLaVA filled; BLIP-2 TBD)  
**Gate write-up (LLaVA-only):** [`G5_GATE_WRITEUP_LLAVA.md`](G5_GATE_WRITEUP_LLAVA.md) — provisional **WEAK**  
**Track target:** CVPR 2027 **main track** (stronger L0 than AAAI deadline shortcut)

## Goal

Show Dual-Expert Keys are not BLIP-2-only. Ladder: **LLaVA-1.5-7B** on lab RTX 4090 → eval + taxonomy → G5 verdict. Qwen2-VL-2B remains fallback only if LLaVA is blocked.

## CVPR week plan (Oct 10–15)

| Phase | When | What |
|---|---|---|
| **A — Strong LLaVA ladder** | This week | Fresh lab trains: **L0 `dn=500` × `eps=20`** → **S2 `eps=1`** → **DEK `eps=1`** (same AVBT/DEK flags as BLIP-2 short ladder). Detached `nohup` on lab. |
| **B — LLaVA result extraction** | After A | Grounding eval + leak taxonomy on lab L0 / S2 / DEK; pull logs/tracker home. |
| **C — BLIP-2 re-extract** | After A–B (same week if time) | Re-run the **same eval + taxonomy protocol** on existing BLIP-2 L0 / S2 / DEK checkpoints so tables share one extraction pipeline (no silent metric drift). |
| **D — Gate write-up** | By Oct 15 | Fill G5 Result in gate log; disclose budgets (BLIP-2 full L0 vs LLaVA short-n L0×20). |

**Smoke note (Oct 10):** a detached **`eps=1` L0** was started only as a path/env smoke. It is **not** the CVPR L0. Prefer a new L0 with `--eps_l0 20` for paper numbers.

### Training budget (CVPR-effective)

| Stage | Backbone | Data | `dn` | Epochs | Notes |
|---|---|---|---|---|---|
| L0 | LLaVA-1.5-7B | VLKEB grounding short train | 500 | **20** | Match BLIP-2 L0 *epoch count* at short `n`; disclose vs LiveEdit full-set / 200K-iter |
| S2 | LLaVA | same | 500 | **1** | From new lab L0; AVBT, no dual gate |
| DEK | LLaVA | same | 500 | **1** | From new lab L0; AVBT + `--dek_dual_gate` |
| L0 (reference) | BLIP-2-OPT-2.7B | VLKEB (existing) | full train | **20** | Keep; re-extract eval only this week unless a train bug is found |
| S2 / DEK (reference) | BLIP-2 | grounding short | 500 | **1** | Existing AAAI ckpts; re-extract eval |

AAAI-era `dn=500 × 3` was a **deadline/GPU shortcut**. For main track we use **`×20`** on the 4090.

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
| Detached runs | `nohup` / PID file under `doc/results/cvpr2027/g5/logs/` so closing Cursor does not stop training |

## Lab paths

- Code: `/home/user/TWD/LiveEdit_lab_usb/LiveEdit`
- Weights: `/home/user/TWD/LiveEdit_lab_usb/models/llava-1.5-7b-hf`
- Runs: `records/lab_g5/liveedit/llava-v1.5-7b/`
- Tracker: `doc/results/cvpr2027/g5/EXPERIMENT_TRACKER.md`

## Run pipeline (lab — detached)

```bash
ssh user@100.89.137.40
cd /home/user/TWD/LiveEdit_lab_usb/LiveEdit
source ~/miniconda3/etc/profile.d/conda.sh && conda activate liveedit
# Strong CVPR ladder (defaults: --eps_l0 20 --eps_ft 1)
nohup python -u tools/lab_g5_llava_fresh_pipeline.py --stages L0 S2 DEK \
  > doc/results/cvpr2027/g5/logs/ladder_outer.log 2>&1 &
```

Or stage-wise:

```bash
python -u tools/lab_g5_llava_fresh_pipeline.py --stages L0 --eps_l0 20
python -u tools/lab_g5_llava_fresh_pipeline.py --stages S2 DEK --eps_ft 1 --from_l0_ckpt <new_lab_L0_ckpt>
```

## After train — dual result extraction

Use **lab** LLaVA ckpts from `records/lab_g5/...`. For BLIP-2, use the existing AAAI L0 / S2 / DEK paths listed in the gate log / stress tables — **same** `test_grounding_eval.py` + `score_leakage_taxonomy.py` commands and probe JSON.

```bash
PROBE=data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json
IMG=data/VLKEB/VLKEB_images/mmkb_images
# fill ckpts from EXPERIMENT_TRACKER.md (LLaVA) and AAAI BLIP-2 table
python -u -W ignore test_grounding_eval.py ... --log_routing
python -u tools/score_leakage_taxonomy.py ...
```

Pull home:

```powershell
powershell -ExecutionPolicy Bypass -File tools/lab_sync_pull_results.ps1 -IncludeCkptMeta
```

## Sync code (git — preferred)

**On this PC** (docs branch stays checked out):

```powershell
git checkout feature/working-on-aaai-review-for-04-10-2026
```

**Lab code push** (after you approve commit + push):

```powershell
git checkout lab/g5-code-only
git commit -m "..."   # no Cursor co-author
git push -u origin lab/g5-code-only
git checkout feature/working-on-aaai-review-for-04-10-2026
```

**On lab PC** (visible SSH):

```bash
cd /home/user/TWD/LiveEdit_lab_usb/LiveEdit
bash tools/lab_git_pull_on_lab.sh
```

## Checklist

- [x] Local smoke PASS (home PC, earlier)
- [x] Lab env + USB tree inventory (Oct 10)
- [x] Keep-latest-3 ckpt retention in trainer
- [x] Fresh pipeline + experiment tracker + sync scripts
- [x] Valid LLaVA weight shards on lab (manual HF download)
- [x] Lab: **strong L0** (`dn=500`, `eps=20`) — smoke `eps=1` does not count
- [x] Lab: S2 train from strong L0
- [x] Lab: DEK train from strong L0
- [x] Lab: LLaVA grounding eval + taxonomy (`leakage_taxonomy_llava.json`, 2026-10-11)
- [ ] Home/lab: **BLIP-2 re-extract** under the same eval protocol
- [x] Pull results to this repo
- [ ] Fill G5 Result in gate log by Oct 15

## Deferred (after G5)

M6 hn method try; `emits_y_i` recovery; EVQA pair-closed curve; full 5-seed; G4; MSCKE; CARML tau.
