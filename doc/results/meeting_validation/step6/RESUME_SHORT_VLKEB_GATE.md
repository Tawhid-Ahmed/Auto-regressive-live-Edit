# Resume: Short VLKEB AVBT Validation Gate

**Stopped:** 2026-07-10 ~12:11 (UTC+6) — user shutting down PC  
**Status:** V1 training **interrupted** (~21 min in, no checkpoint saved)

---

## Goal

Improve swap on **original short VLKEB** with AVBT; validate direction before a long run (dsn=5000).

**Gate criteria** (proceed to long run if any mini-sweep run passes):
- Swap ≥ **50%** (or +2 pp over 47.8%)
- Rel ≥ **92%**
- False-cross (swap) trending down (≤ 90%)

---

## Completed work

### Baseline reference (short VLKEB)

| Model | Rel | Swap | Ckpt |
|-------|-----|------|------|
| Short baseline | 93.5% | 40.7% | `records/.../VLKEB_baseline-.../epoch-20-i-25950-ema_loss-0.1201` |
| AVBT short S2 (dsn=500, 1 ep) | **93.8%** | **47.8%** | `records/.../VLKEB_avbt_short_S2-2026.07.10-11.16.19/.../epoch-1-i-125-ema_loss-4.0787` |

Docs: `doc/results/meeting_validation/step6/avbt_short_vlkeb_comparison.json`  
Checkpoints: `doc/THESIS_PATHS.json` → `vlkeb_original_baseline_epoch20`, `vlkeb_avbt_short_S2_epoch1`

### Step 1 — Direction verify ✅ (done)

| Run | Result | Key signal |
|-----|--------|------------|
| Baseline + micro-AVBL | FAIL (script) | swap expert_mass 0.995 → 0.969 |
| **AVBT short ckpt** | FAIL (script) | **swap expert_mass already 0.22** vs 0.99 baseline |

Outputs:
- `doc/results/meeting_validation/step6/short_vlkeb_direction_baseline.json`
- `doc/results/meeting_validation/step6/short_vlkeb_direction_avbt.json`

**Conclusion:** Binding direction is right on trained AVBT; proceed to mini-sweep.

Code fix applied: `verify_avbl_direction.py` now supports `--patch_grounding` for AVBT ckpt load.

### Step 2 — V1 train (dn=1000) ❌ INTERRUPTED

- **Run name:** `VLKEB_avbt_short_V1`
- **Command:** see below
- **Progress:** ~21 min, likely **no checkpoint** (`-sci 250` = save only at i=250 after full 1000 samples)
- **Log (partial):** `doc/results/meeting_validation/step6/avbt_short_V1_train.log`
- **Action on resume:** **Re-run V1 from scratch** (do not use `--resume` unless a ckpt folder exists)

---

## Not started yet

| Step | Description |
|------|-------------|
| **V1 eval** | `test_grounding_eval.py` on V1 ckpt, dsn=100, seed=42, `--patch_grounding --log_routing` |
| **V2 train** | dn=500, λ=0.3, neg_weight=2.5, 1 epoch |
| **V2 eval** | Same eval protocol |
| **V3** | Optional: resume from AVBT short S2 ckpt, 2nd epoch (only if V1/V2 promising) |
| **Gate decision** | Update `short_vlkeb_gate_manifest.json` → long run or architecture change |

---

## Resume commands (run in order)

```cmd
conda activate liveedit
cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"
```

### 1. Re-run V1 train (~30–35 min) — ✅ DONE 2026-07-10

Ckpt: `VLKEB_avbt_short_V1-2026.07.10-14.06.46/.../epoch-1-i-250-ema_loss-2.5288`  
Eval: swap **46.2%**, rel **93.8%** (did not beat S2's 47.8%)

### 2. V1 eval — ✅ DONE

### 3. V2 train (~16 min) — ⏸ BLOCKED (reboot needed)

Windows error: paging file too small (os error 1455). Reboot PC, then run in your terminal (progress bar visible):

```cmd
conda activate liveedit
cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"
set PYTHONUNBUFFERED=1
E:\anaconda\envs\liveedit\python.exe -u train_vllm_editor.py -en liveedit -mn blip2 -dna VLKEB -bs 4 -dvc cuda:0 -edvc 0 -lkpt "records/liveedit/blip2-opt-2.7b/VLKEB_baseline-2026.04.09-08.20.07/checkpoints/epoch-20-i-25950-ema_loss-0.1201" -eps 1 -tnp VLKEB_avbt_short_V2 -dn 500 -rs 42 -sci 125 -lpi 10 -dpath data/VLKEB/grounding/grounding_train_short.json --patch_grounding --adversarial_grounding --adv_grounding_lambda 0.3 --adv_grounding_neg_weight 2.5 --avbl_hinge_tau 0.3 --avbl_contrastive
```

Then V2 eval:

```cmd
E:\anaconda\envs\liveedit\python.exe -u -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 -ckpt "<V2_ckpt>" -enp short_V2 -dpath data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json --meeting_step 6 --patch_grounding --log_routing
```

---

## Key paths

| Item | Path |
|------|------|
| Train JSON | `data/VLKEB/grounding/grounding_train_short.json` |
| Eval JSON | `data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json` |
| Gate manifest | `doc/results/meeting_validation/step6/short_vlkeb_gate_manifest.json` |
| This resume doc | `doc/results/meeting_validation/step6/RESUME_SHORT_VLKEB_GATE.md` |

---

## Tell the agent on resume

> "Resume short VLKEB validation gate from `RESUME_SHORT_VLKEB_GATE.md` — re-run V1 train then eval, then V2."
