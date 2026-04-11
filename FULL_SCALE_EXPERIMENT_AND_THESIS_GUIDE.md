# Full-Scale Experiment and Thesis Guide

This guide runs the LiveEdit vs AR-LiveEdit pipeline from **training** through **evaluation** and **thesis reporting**. It assumes you have already completed environment setup (Step 1 in SETUP_AND_RUN.md: conda env `liveedit`, `ROOT_PATH` in `utils/GLOBAL.py`, datasets under `data/`, and BLIP2 weights under `models/blip2-opt-2.7b`).

---

## Table of Contents

1. [Prerequisites Check](#1-prerequisites-check)
2. [Train Baseline LiveEdit (full scale)](#2-train-baseline-liveedit-full-scale)
3. [Train AR-LiveEdit (full scale)](#3-train-ar-liveedit-full-scale)
4. [LiveEdit vs AR-LiveEdit Comparison](#4-liveedit-vs-ar-liveedit-comparison)
5. [Verify thesis goals](#5-verify-thesis-goals)
6. [AR-LiveEdit Ablation (Task 9)](#6-ar-liveedit-ablation-task-9)
7. [Full Evaluation (VLKEB and optional EVQA)](#7-full-evaluation-vlkeb-and-optional-evqa)
8. [Thesis: Results and Reporting](#8-thesis-results-and-reporting)
9. [Quick Reference: Paths and Outputs](#9-quick-reference-paths-and-outputs)

---

## 1. Prerequisites Check

From the project root (the `LiveEdit` folder):

```bash
conda activate liveedit
python -c "
from utils.GLOBAL import ROOT_PATH, model_path_map
import os
print('ROOT_PATH:', ROOT_PATH, 'exists:', os.path.isdir(ROOT_PATH))
for k, v in model_path_map.items():
    print('  ', k, ':', v, 'exists:', os.path.isdir(v))
"
```

- Confirm **VLKEB** data exists: `data/VLKEB/train.json`, `data/VLKEB/eval.json`, and the image directory (e.g. `data/VLKEB/VLKEB_images/mmkb_images`).
- If you hit **OSError 1455 (paging file too small)** when loading the model, increase Windows virtual memory or use a machine with more RAM before running training.

---

## 2. Train Baseline LiveEdit (full scale)

Train the standard LiveEdit editor **without** AR mode on the full VLKEB training set. This produces the baseline checkpoint for comparison.

**VLKEB (recommended for thesis):**

```bash
conda activate liveedit
python train_vllm_editor.py -en liveedit -mn blip2 -dna VLKEB -bs 4 -dvc cuda:0 -edvc 0 -lkpt None -tnp VLKEB_baseline -eps 20 -sci 500 -lpi 10
```

- `**-bs 4**`: Batch size; reduce to 2 if you get CUDA OOM.
- `**-eps 20**`: Epochs; increase (e.g. 50) for stronger convergence.
- `**-sci 500**`: Save a checkpoint every 500 iterations.
- `**-tnp VLKEB_baseline**`: Run name; checkpoints will go under `records/liveedit/blip2-opt-2.7b/VLKEB_baseline-<timestamp>/checkpoints/`.

**Optional – EVQA (if you use MMEdit data):**

```bash
python train_vllm_editor.py -en liveedit -mn blip2 -dna EVQA -bs 8 -dvc cuda:0 -edvc 0 -lkpt None -tnp EVQA_baseline -eps 50 -sci 500 -lpi 10
```

**After training:**

- Note the **checkpoints directory**, e.g.  
`records/liveedit/blip2-opt-2.7b/VLKEB_baseline-2025.03.03-14.30.00/checkpoints/`
- Pick one **checkpoint file** (e.g. the last or best by loss), e.g.  
`records/liveedit/blip2-opt-2.7b/VLKEB_baseline-2025.03.03-14.30.00/checkpoints/epoch-20-i-2500-ema_loss-0.1234.pt`  
You will use this path as **baseline checkpoint** in Steps 4 and 6.

---

## 3. Train AR-LiveEdit (full scale)

Train LiveEdit **with** autoregressive chunk-wise mode on the same dataset (VLKEB or EVQA) so you have a comparable AR checkpoint.

**VLKEB:**

```bash
conda activate liveedit
python train_vllm_editor.py -en liveedit -mn blip2 -dna VLKEB -bs 4 -dvc cuda:0 -edvc 0 -lkpt None -tnp VLKEB_ar --ar_mode --chunk_size 16 -eps 20 -sci 500 -lpi 10

python -W ignore train_vllm_editor.py -en liveedit -mn blip2 -dna VLKEB -bs 4 -dvc cuda:0 -edvc 0 -lkpt "records\liveedit\blip2-opt-2.7b\VLKEB_ar-2026.04.09-20.54.04\checkpoints\epoch-7-i-8100-ema_loss-0.3124" -tnp VLKEB_ar --ar_mode --chunk_size 16 -eps 20 -sci 150 -lpi 20 -dbs 8
```

- `**--ar_mode**`: Enables AR training.
- `**--chunk_size 16**`: Token chunk size (default); you can ablate later.
- `**-tnp VLKEB_ar**`: Run name; checkpoints under `records/liveedit/blip2-opt-2.7b/VLKEB_ar-<timestamp>/checkpoints/`.

**Optional – EVQA:**

```bash
python train_vllm_editor.py -en liveedit -mn blip2 -dna EVQA -bs 8 -dvc cuda:0 -edvc 0 -lkpt None -tnp EVQA_ar --ar_mode --chunk_size 16 -eps 50 -sci 500 -lpi 10
```

**After training:**

- Note the **AR checkpoints directory** and choose one **checkpoint file** (e.g. last epoch) as **AR checkpoint** for Steps 4, 5, and 6.

---

## 4. LiveEdit vs AR-LiveEdit Comparison

Run **controlled comparison** on VLKEB with the **same eval split** (fixed seed) for baseline and AR, then get a comparison JSON.

### 4.1 Set checkpoint paths

Edit `**run_liveedit_ar_comparison.bat`** (or run the Python script directly with arguments):

- `**CKPT`**: Full path to your **baseline** `.pt` file, e.g.  
`records\liveedit\blip2-opt-2.7b\VLKEB_baseline-2025.03.03-14.30.00\checkpoints\epoch-20-i-2500-ema_loss-0.1234.pt`
- `**AR_CKPT`**: Full path to your **AR** `.pt` file. Leave empty to use the same as `CKPT` (then both legs use the same weights; only inference path differs).

Example (Windows batch):

```batch
set CKPT=records\liveedit\blip2-opt-2.7b\VLKEB_baseline-2025.03.03-14.30.00\checkpoints\epoch-20-i-2500-ema_loss-0.1234.pt
set AR_CKPT=records\liveedit\blip2-opt-2.7b\VLKEB_ar-2025.03.03-16.00.00\checkpoints\epoch-20-i-2500-ema_loss-0.2345.pt
set DSN=200
set SEN=50
set SEED=42
```

### 4.2 Run comparison

**Option A – Batch (Windows):**

```batch
run_liveedit_ar_comparison.bat
```

**Option B – Python (any OS):**

```bash
conda activate liveedit
python -W ignore run_liveedit_ar_comparison.py -dvc cuda:0 -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_baseline-.../checkpoints/epoch-20-i-2500-ema_loss-0.1234.pt" --ar_ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_ar-.../checkpoints/epoch-20-i-2500-ema_loss-0.2345.pt" -dsn 200 -sen 50 -seed 42
```

- `**-dsn 200**`: Number of VLKEB eval samples (use full or your chosen dev size).
- `**-sen 50**`: Sequential edits per batch.
- `**-seed 42**`: Fixed seed for reproducible split; **must be the same** when you re-run or compare.

### 4.3 Output

- **Comparison JSON**: `eval_results/comparison/liveedit_vs_ar_seed42.json`
- Contains: baseline vs AR metrics (reliability, generality, locality, length bins), deltas, and validation flags (e.g. long-form improvement, locality bounded).
- Use this file for **thesis tables/figures** (baseline vs AR on VLKEB).

---

## 5. Verify thesis goals

After running the comparison (Section 4), check that the experiment meets your thesis goals using the comparison JSON. This section is the **explicit verification step** before you report results.

### 5.1 Where to look

Open the comparison file produced in Section 4:

- **File:** `eval_results/comparison/liveedit_vs_ar_seed42.json` (or `liveedit_vs_ar_seed<N>.json` if you used another seed).
- **Relevant keys:** under the top-level `"validation"` object.

### 5.2 Thesis goals and JSON keys


| Thesis goal                                                                  | JSON key                                                                  | Interpretation                                                                  |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Long-form improvement** — AR improves editing on long targets (65+ tokens) | `validation.long_form_improvement`                                        | `true` = either EM or accuracy for the 65+ token bin improved (AR vs baseline). |
| **Locality bounded** — locality does not drop too much                       | `validation.locality_bounded`                                             | `true` = worst locality drop is ≥ −5% (i.e. within 5%).                         |
| **Long-form deltas (for reporting)**                                         | `validation.long_form_65+_em_delta`, `validation.long_form_65+_acc_delta` | Numeric deltas (AR − baseline) for the 65+ bin.                                 |
| **Locality drop (for reporting)**                                            | `validation.locality_min_delta`                                           | Worst locality accuracy change (negative = drop).                               |


### 5.3 How to check

1. Open the JSON (e.g. in a text editor or Python: `json.load(open("eval_results/comparison/liveedit_vs_ar_seed42.json"))`).
2. Navigate to `["validation"]`.
3. **Pass criteria (typical thesis goals):**
  - `long_form_improvement` is `true`.
  - `locality_bounded` is `true`.
4. Optionally note the numeric values of `long_form_65+_em_delta`, `long_form_65+_acc_delta`, and `locality_min_delta` for the thesis text.

### 5.4 If a goal is not met

- **Long-form not improved:** Consider a different checkpoint (e.g. later epoch), more AR training, or the ablation-recommended config; or report as a limitation and discuss possible causes (e.g. data size, chunk size).
- **Locality not bounded (drop > 5%):** Same options as above; in the thesis you can report the actual drop and discuss the trade-off between long-form gain and locality.
- Re-run the comparison (Section 4) after changing checkpoints or settings, then repeat this verification.

---

## 6. AR-LiveEdit Ablation (Task 9)

Sweep **chunk_size** (8, 16, 32), **routing_gate** (on/off), and **max_chunks** (unlimited / 4) to get an ablation table and a recommended config.

### 6.1 Set checkpoint

Edit `**run_ar_ablation.bat`**:

- `**CKPT`**: Full path to your **AR** checkpoint (or baseline if you only have one). Example:  
`records\liveedit\blip2-opt-2.7b\VLKEB_ar-2025.03.03-16.00.00\checkpoints\epoch-20-i-2500-ema_loss-0.2345.pt`

For a **full-scale** ablation (thesis), you can increase sample count and sequential edits, e.g.:

```batch
set DSN=200
set SEN=50
set SEED=42
```

### 6.2 Run ablation

**Option A – Batch:**

```batch
run_ar_ablation.bat
```

**Option B – Python:**

```bash
conda activate liveedit
python -W ignore run_ar_ablation.py -dvc cuda:0 -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_ar-.../checkpoints/epoch-20-i-2500-ema_loss-0.2345.pt" -dsn 200 -sen 50 -seed 42
```

To **only build the table** from existing eval results (no new runs):

```bash
python -W ignore run_ar_ablation.py --skip_run -seed 42
```

### 6.3 Output

- **Ablation report**: `eval_results/ablation/ar_ablation_seed42.json`
- Contains: ablation table (all configs and key metrics), recommendation (best chunk_size, routing_gate, max_chunks), and config used.
- Use this for **thesis**: ablation table and “recommended AR configuration” sentence.

---

## 7. Full Evaluation (VLKEB and optional EVQA)

Run the standard evaluator for **reproducible** numbers to report in the thesis (same seed and sample counts).

### 7.1 VLKEB – Baseline checkpoint

```bash
conda activate liveedit
python test_vllm_edit.py -en liveedit -mn blip2 -sen 50 -dvc cuda:0 -dn VLKEB -dsn 200 -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_baseline-.../checkpoints/epoch-20-i-2500-ema_loss-0.1234.pt" -seed 42
```

- Results under: `eval_results/liveedit/blip2-opt-2.7b/VLKEB/sequential_edit_50/` (with `seed_42_mean_results.json` if seed is used in the eval script; otherwise `mean_results.json`).

### 7.2 VLKEB – AR checkpoint

```bash
python test_vllm_edit.py -en liveedit -mn blip2 -sen 50 -dvc cuda:0 -dn VLKEB -dsn 200 -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_ar-.../checkpoints/epoch-20-i-2500-ema_loss-0.2345.pt" --ar_mode -seed 42
```

- Use the **same** `-dsn` and `-seed` as baseline for fair comparison.

### 7.3 Optional – EVQA evaluation

If you trained on EVQA:

```bash
python test_vllm_edit.py -en liveedit -mn blip2 -sen 1000 -dvc cuda:0 -dn EVQA -ckpt "records/liveedit/blip2-opt-2.7b/EVQA_baseline-.../checkpoints/....pt" -seed 42
python test_vllm_edit.py -en liveedit -mn blip2 -sen 1000 -dvc cuda:0 -dn EVQA -ckpt "records/liveedit/blip2-opt-2.7b/EVQA_ar-.../checkpoints/....pt" --ar_mode -seed 42
```

- Replace checkpoint paths with your actual EVQA baseline and AR checkpoint paths.

---

## 8. Thesis: Results and Reporting

### 8.1 Where to find results


| What                                     | Location                                                              |
| ---------------------------------------- | --------------------------------------------------------------------- |
| Baseline vs AR comparison (VLKEB)        | `eval_results/comparison/liveedit_vs_ar_seed42.json`                  |
| AR ablation table and recommendation     | `eval_results/ablation/ar_ablation_seed42.json`                       |
| Per-run mean metrics (VLKEB baseline/AR) | `eval_results/liveedit/blip2-opt-2.7b/VLKEB-.../sequential_edit_<N>/` |
| Training logs / TensorBoard              | `records/liveedit/blip2-opt-2.7b/<train_name>/logs/`                  |
| Checkpoints                              | `records/liveedit/blip2-opt-2.7b/<train_name>/checkpoints/*.pt`       |


### 8.2 What to report

1. **Setup**
  - Dataset: VLKEB (and optionally EVQA), train/eval split, and seed (e.g. 42).
  - Model: BLIP2-OPT-2.7B; editor: LiveEdit vs AR-LiveEdit (chunk size, routing gate, max chunks as in ablation).
2. **Main result: LiveEdit vs AR-LiveEdit**
  - From `liveedit_vs_ar_seed42.json`: reliability (acc/EM), generality, locality; length-bin metrics (e.g. 65+ token bin); deltas (AR − baseline).
  - Emphasize: long-form (e.g. 65+ tokens) improvement and that locality stays bounded (e.g. within 5%).
3. **Ablation**
  - From `ar_ablation_seed42.json`: table of chunk_size × routing_gate × max_chunks with key metrics (e.g. reliability acc/EM, bin 65+ EM/acc, locality).
  - State the **recommended configuration** from the script and briefly justify (e.g. long-form EM, then reliability, then locality).
4. **Training**
  - Epochs, batch size, checkpoint interval; optionally one learning curve (from TensorBoard or logs).
5. **Reproducibility**
  - Mention: fixed seed (42), same eval split for baseline and AR, and that comparison/ablation/eval commands are documented in this guide and SETUP_AND_RUN.md.

---

## 9. Quick Reference: Paths and Outputs


| Step                    | Key paths / outputs                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------------------- |
| 2 – Baseline training   | `records/liveedit/blip2-opt-2.7b/VLKEB_baseline-<timestamp>/checkpoints/*.pt`                         |
| 3 – AR training         | `records/liveedit/blip2-opt-2.7b/VLKEB_ar-<timestamp>/checkpoints/*.pt`                               |
| 4 – Comparison          | `eval_results/comparison/liveedit_vs_ar_seed42.json`                                                  |
| 5 – Verify thesis goals | Same file as step 4: check `validation.long_form_improvement`, `validation.locality_bounded`          |
| 6 – Ablation            | `eval_results/ablation/ar_ablation_seed42.json`                                                       |
| 7 – Full eval           | `eval_results/liveedit/blip2-opt-2.7b/VLKEB/.../sequential_edit_50/` (and seed-specific mean results) |


**Suggested order:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 (then write thesis using Section 8 and the paths above).

For environment setup, datasets, and troubleshooting, see **SETUP_AND_RUN.md**.











epoch-20-i-25950-ema_loss-0.1201



python -W ignore test_vllm_[edit.py](http://edit.py) -en liveedit -mn blip2 -sen 50 -dvc cuda:0 -dn VLKEB -dsn 200 -ckpt "records\liveedit\blip2-opt-2.7b\VLKEB_baseline-2026.04.09-08.20.07\checkpoints\epoch-20-i-25950-ema_loss-0.1201" -seed 42



python -W ignore test_vllm_[edit.py](http://edit.py) -en liveedit -mn blip2 -sen 50 -dvc cuda:0 -dn VLKEB -dsn 200 -ckpt "records\liveedit\blip2-opt-2.7b\VLKEB_baseline-2026.04.09-08.20.07\checkpoints\epoch-20-i-25950-ema_loss-0.1201" -seed 42 -enp baseline



python -W ignore test_vllm_[edit.py](http://edit.py) -en liveedit -mn blip2 -sen 50 -dvc cuda:0 -dn VLKEB -dsn 200 -ckpt "records\liveedit\blip2-opt-2.7b\VLKEB_ar-2026.04.10-07.42.20\checkpoints\epoch-20-i-25500-ema_loss-0.0765" --ar_mode -seed 42