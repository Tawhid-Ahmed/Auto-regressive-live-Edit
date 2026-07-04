# AR short-train → long-eval: cross-domain comparison

**Protocol:** seed 42, `sequential_edit_n=50`, BLIP2-OPT-2.7b + LiveEdit, n=3150 edits, long-form eval (`eval_longform_ollama.json`, mean ~88.6 tokens, 3149 in 65+ bin)
**Generated:** 2026-06-28

This file isolates the effect of **training data (short vs long)** on both the **baseline** and **AR** editors, all evaluated on the same long-form split.

---

## New run

| Item | Value |
|------|-------|
| Checkpoint | `records/.../VLKEB_ar-2026.04.10-07.42.20/checkpoints/epoch-20-i-25500-ema_loss-0.0765` |
| Training data | **Original (short) VLKEB**, AR mode (chunk_size 16) |
| Eval data | **VLKEB-long** (`eval_longform_ollama.json`) |
| Eval AR settings | `--ar_mode --chunk_size 16` |
| Result folder | `eval_results/liveedit/blip2-opt-2.7b/VLKEB-ar-VLKEB_ar_short_train_long_eval-2026.06.28-12.11.08` |

---

## Table 1 — 2×2 training × method matrix (all evaluated on long-form)

| Editor | Trained on | Rel acc | Chunk EM | Chunk F1 | Text gen | Image gen | T-Loc | I-Loc |
|--------|-----------|---------|----------|----------|----------|-----------|-------|-------|
| Baseline | Short VLKEB | 28.57% | 1.68% | 30.62% | 28.79% | 28.85% | 100% | 100% |
| **AR (cs=16)** | **Short VLKEB** | **26.92%** | **0.75%** | **28.47%** | **27.07%** | **27.70%** | **100%** | **100%** |
| Baseline | Long VLKEB | 75.40% | 13.50% | 76.28% | 75.21% | 71.49% | 100% | 100% |
| AR (cs=16) | Long VLKEB | 75.94% | 14.82% | 76.90% | 75.63% | 72.51% | 100% | 100% |

*All rows: same eval JSON, seed 42, 3150 edits.*

---

## Table 2 — Effect of long-form training (per editor)

| Editor | Short-trained (long eval) | Long-trained (long eval) | Gain from long-form training |
|--------|---------------------------|--------------------------|------------------------------|
| Baseline | 28.57% | 75.40% | **+46.83 pp** |
| AR (cs=16) | 26.92% | 75.94% | **+49.02 pp** |

---

## Table 3 — AR vs Baseline at equal training (long eval)

| Training regime | Baseline rel acc | AR rel acc | Δ (AR − baseline) |
|-----------------|------------------|------------|-------------------|
| Short-trained | 28.57% | 26.92% | **−1.65 pp** |
| Long-trained | 75.40% | 75.94% | **+0.54 pp** |

---

## Key findings

1. **AR without long-form training does not help — it slightly hurts.** On the short-trained checkpoint, AR (26.92%) is **1.65 pp below** the short-trained baseline (28.57%) on long eval. Chunk-wise editing only pays off **after** the editor has been trained on long targets.

2. **Long-form training is the dominant factor for both editors.** Training on VLKEB-long lifts reliability by **+46.8 pp** (baseline) and **+49.0 pp** (AR). This dwarfs the AR-vs-baseline effect (≤0.54 pp) and is the central empirical message of the thesis.

3. **AR benefits more from long-form training** (+49.0 pp vs +46.8 pp), consistent with the idea that the chunk-wise path needs long targets at train time to be useful at test time.

4. **Locality is invariant (100%) across all four cells**, including both failing short-trained cases — the editor never damages unrelated knowledge, regardless of training data or AR mode.

5. **Exact match = 0%** for every long-form row (mean ~89 tokens); report token accuracy and chunk metrics.

---

## Results paragraph (paste-ready)

To separate the contribution of long-form *training* from the AR *mechanism*, we evaluated an AR-LiveEdit checkpoint trained only on the original short-target VLKEB on the long-form eval split (seed 42, 3150 sequential edits, chunk_size 16). This short-trained AR model reached **26.9%** token accuracy — **1.7 pp below** the short-trained baseline (28.6%) and far below either long-trained model (75.4% baseline, 75.9% AR). The result shows that autoregressive chunk-wise editing provides no benefit, and a slight cost, when the editor has not been trained on long targets; its advantage emerges only after long-form training. Long-form training itself accounts for a **+46.8 pp** (baseline) to **+49.0 pp** (AR) reliability gain, confirming that dedicated long-form training — not the inference-time editing strategy — is the primary driver of long-form editing performance. Locality remained 100% on text and image probes in all four configurations.

---

## Source paths

| Cell | Folder |
|------|--------|
| Baseline, short-trained | `eval_results/.../VLKEB-VLKEB_baseline_short_train_long_eval-2026.06.20-10.05.42` |
| AR, short-trained | `eval_results/.../VLKEB-ar-VLKEB_ar_short_train_long_eval-2026.06.28-12.11.08` |
| Baseline, long-trained | `eval_results/.../VLKEB-VLKEB_long_baseline_full-2026.06.04-17.39.13` |
| AR, long-trained | `eval_results/.../VLKEB-ar-VLKEB_long_ar_full-2026.06.04-18.03.18` |

Each: `sequential_edit_50/seed_42_mean_results.json`
