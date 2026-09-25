# AAAI Competitor Table — Short VLKEB Grounding

**Date:** 2026-07-10  
**Protocol:** `dsn=100`, `sen=10`, `seed=42`,  
`data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json`,  
`--meeting_step 6`, runner `test_grounding_eval.py`  
**Machine note:** Use `E:\anaconda\envs\liveedit\python.exe -u` (not `conda run`).

**Companion JSON:** [`aaai_competitor_short.json`](results/meeting_validation/step6/aaai_competitor_short.json)

---

## 1. Results so far (LiveEdit family — done)

| Method | Rel ↑ | Swap ↑ | Hard-neg ↑ | FC ↓ | Swap-gap ↓ | Eval dir |
|--------|------:|-------:|-----------:|-----:|-----------:|----------|
| LiveEdit baseline (L0) | 0.9347 | 0.4072 | 0.4224 | **0.96** | 0.5275 | `.../VLKEB_GROUNDING-aaai_L0_short-2026.07.10-19.25.37` |
| AVBT (S2) — no Dual-Expert Keys | 0.9380 | 0.4782 | 0.4259 | 0.96 | 0.4598 | `.../VLKEB_GROUNDING-aaai_S2_short-2026.07.10-19.27.38` |
| **AVBT + Dual-Expert Keys (DEK)** | **0.9372** | 0.4773 | 0.4201 | **0.46** | 0.4599 | `.../VLKEB_GROUNDING-aaai_DEK_short-2026.07.10-19.30.27` |
| Fine-tuning (FT) | 0.7300 | 0.7308 | 0.7300 | n/a | **−0.0008** | `.../VLKEB_GROUNDING-aaai_FT_short_v3-2026.07.11-18.21.45` |
| MEND-VL (MEND) | 0.7126 | **0.6937** | **0.7037** | n/a | 0.0189 | `.../VLKEB_GROUNDING-aaai_MEND_short-2026.07.11-18.16.23` |
| SERAC-VL (SERAC) | 0.3859 | 0.3860 | 0.3765 | n/a | ≈0 | `.../VLKEB_GROUNDING-aaai_SERAC_short-2026.07.11-19.20.35` |

**Definitions**
- **Reliability (Rel)** = correct-image edit token accuracy
- **Image-swap (Swap)** / **Hard-negative (Hard)** = adversarial image probes
- **False-cross (FC)** = `routing_diagnostics.image_swap_false_cross_rate` (LiveEdit only; needs `--log_routing`)
- **Swap-gap** = Rel − Swap (higher = more leakage)

### Takeaway (G1 — short competitors complete)

- **AVBT + Dual-Expert Keys (DEK) is the only row that cuts false-cross** (96% → 46%) while holding reliability (~94%).
- Token swap: AVBT (S2) ≈ Dual-Expert Keys (DEK) (~47.8%); both beat LiveEdit baseline (L0) by ~+7 pp.
- **MEND-VL / Fine-tuning:** ~71–73% rel with near-zero swap-gap (swap ≈ rel) — weak-edit / over-generalize, not a grounding win.
- **SERAC-VL:** ~39% rel/swap (near-zero gap) — weakest reliability of the table; same “soft edit” pattern, worse magnitude.
- Paper story: lead with **FC + reliability-preserving swap**; external editors do not match Dual-Expert Keys’ operating point.

---

## 2. Checkpoints used

| Method | Ckpt | Eval flags |
|--------|------|------------|
| LiveEdit baseline (L0) | `records/liveedit/blip2-opt-2.7b/VLKEB_baseline-2026.04.09-08.20.07/checkpoints/epoch-20-i-25950-ema_loss-0.1201` | `--log_routing` |
| AVBT (S2) | `records/liveedit/blip2-opt-2.7b/VLKEB_avbt_short_S2-2026.07.10-11.16.19/checkpoints/epoch-1-i-125-ema_loss-4.0787` | `--patch_grounding --log_routing` |
| Dual-Expert Keys (DEK) | `records/liveedit/blip2-opt-2.7b/VLKEB_avbt_short_DEK-2026.07.10-18.13.18/checkpoints/epoch-1-i-125-ema_loss-4.0100` | `--patch_grounding --dek_dual_gate --log_routing` |
| Fine-tuning (FT) | *(edit-time; no meta-ckpt)* | config `configs/ft_vl/blip2-opt-2.7b.yaml` |
| MEND-VL (MEND) | `records/mend_vl/blip2-opt-2.7b/VLKEB_mend_short_aaai-2026.07.11-18.04.10/checkpoints/epoch-1-i-375-ema_loss-0.3538` | — |
| SERAC-VL (SERAC) | `records/serac_vl/blip2-opt-2.7b/VLKEB_serac_short_aaai-2026.07.11-19.13.05/checkpoints/epoch-1-i-125-ema_loss-36.1387` | deps: local `opt-125m` + `bert-base-cased` |

---

## 3. How to get FT / MEND / SERAC results

### 3.1 FT_VL — **no meta-training** (edit-time fine-tune)

FT applies a short AdamW update **at each edit** (`configs/ft_vl/blip2-opt-2.7b.yaml`: 15 steps, lr=`5e-5`, `norm_constraint=1e-3`; float32 optimizer path in `ft_vl.py`).  
You do **not** run `train_vllm_editor.py` for FT.

**Eval only:**

```cmd
conda activate liveedit
cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"
set PYTHONUNBUFFERED=1

E:\anaconda\envs\liveedit\python.exe -u -W ignore test_grounding_eval.py ^
  -en ft_vl -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt none ^
  -enp aaai_FT_short_v3 ^
  -dpath "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json" ^
  --meeting_step 6
```

**Notes**
- No `--patch_grounding` / `--dek_dual_gate` / `--log_routing` (FT has no LiveEdit routing pool → FC = n/a).
- **Done:** `aaai_FT_short_v3` (~2.4 min wall with current 15-step config).
- Do **not** use pre-fix runs (`aaai_FT_short`, `_v2`) — those collapsed to `<s>`.

---

### 3.2 MEND_VL — **requires training** the hypernetwork

MEND learns an auxiliary network; you must train, then eval with that ckpt.

**Train (short VLKEB, fair vs our dn=500 setup):**

```cmd
E:\anaconda\envs\liveedit\python.exe -u -W ignore train_vllm_editor.py ^
  -en mend_vl -mn blip2 -dna VLKEB -bs 4 -dvc cuda:0 -edvc 0 ^
  -dn 500 -eps 1 -rs 42 -sci 125 -lpi 10 ^
  -tnp VLKEB_mend_short_aaai ^
  -dpath "data/VLKEB/train.json"
```

- Config: `configs/mend_vl/blip2-opt-2.7b.yaml`
- Ckpts land under: `records/mend_vl/blip2-opt-2.7b/VLKEB_mend_short_aaai-<timestamp>/checkpoints/`
- If train OOMs: try `-bs 1` or `-dn 200` smoke first.

**Eval:**

```cmd
E:\anaconda\envs\liveedit\python.exe -u -W ignore test_grounding_eval.py ^
  -en mend_vl -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt "records/mend_vl/blip2-opt-2.7b/VLKEB_mend_short_aaai-TIMESTAMP/checkpoints/CKPT_NAME" ^
  -enp aaai_MEND_short ^
  -dpath "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json" ^
  --meeting_step 6
```

FC = n/a (no LiveEdit `probe_routing`).

---

### 3.3 SERAC_VL — **requires training** + local model paths

SERAC config still has placeholders:

```yaml
# configs/serac_vl/blip2-opt-2.7b.yaml
counterfact_model_path: ".../opt-125m"
classifier_path: ".../bert-base-cased"
```

**Before train:** edit that YAML to real local HuggingFace paths (or download into your models folder), e.g.:

```yaml
counterfact_model_path: "E:/MSC/MSc thesis project/LiveEdit/models/opt-125m"
classifier_path: "E:/MSC/MSc thesis project/LiveEdit/models/bert-base-cased"
```

(Download with `huggingface-cli download` / `transformers` if missing.)

**Train:**

```cmd
E:\anaconda\envs\liveedit\python.exe -u -W ignore train_vllm_editor.py ^
  -en serac_vl -mn blip2 -dna VLKEB -bs 4 -dvc cuda:0 -edvc 0 ^
  -dn 500 -eps 1 -rs 42 -sci 125 -lpi 10 ^
  -tnp VLKEB_serac_short_aaai ^
  -dpath "data/VLKEB/train.json"
```

**Eval:** same pattern as MEND with `-en serac_vl` and the new ckpt path; `-enp aaai_SERAC_short`.

---

## 4. After each new competitor finishes

1. Open `seed_42_mean_results.json` under  
   `eval_results/meeting_validation/step6/<editor>/blip2-opt-2.7b/VLKEB_GROUNDING-aaai_*-.../sequential_edit_10/`
2. Update **Section 1 table** in this file.
3. Update `doc/results/meeting_validation/step6/aaai_competitor_short.json`.
4. Recompute swap-gap = rel − swap.

**G1 pass (full):** DEK best **FC** among LiveEdit rows (already true) **and** best or tied **swap-gap** vs FT/MEND/SERAC on token metrics.

---

## 5. Recommended order for you

1. ~~**FT eval**~~ — done.  
2. ~~**MEND train + eval**~~ — done.  
3. ~~**SERAC**~~ — done (`aaai_SERAC_short`).  
4. **Phase B** — 5 seeds on DEK vs L0 (significance).  
5. **Phase C** — long-form transfer.

---

## 6. Commands already run (reproduce)

```cmd
REM L0
E:\anaconda\envs\liveedit\python.exe -u -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_baseline-2026.04.09-08.20.07/checkpoints/epoch-20-i-25950-ema_loss-0.1201" -enp aaai_L0_short -dpath "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json" --meeting_step 6 --log_routing

REM S2
E:\anaconda\envs\liveedit\python.exe -u -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_avbt_short_S2-2026.07.10-11.16.19/checkpoints/epoch-1-i-125-ema_loss-4.0787" -enp aaai_S2_short -dpath "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json" --meeting_step 6 --patch_grounding --log_routing

REM DEK
E:\anaconda\envs\liveedit\python.exe -u -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_avbt_short_DEK-2026.07.10-18.13.18/checkpoints/epoch-1-i-125-ema_loss-4.0100" -enp aaai_DEK_short -dpath "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json" --meeting_step 6 --patch_grounding --dek_dual_gate --log_routing

REM FT (fp32 optimizer path; no meta-train)
E:\anaconda\envs\liveedit\python.exe -u -W ignore test_grounding_eval.py -en ft_vl -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 -ckpt none -enp aaai_FT_short_v3 -dpath "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json" --meeting_step 6
```
