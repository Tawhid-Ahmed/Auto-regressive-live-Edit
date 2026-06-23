# Full Evaluation Comparison — All Thesis Runs

**Protocol:** seed 42, `sequential_edit_n=50`, BLIP2-OPT-2.7b + LiveEdit, n=3150 edits per run  
**Generated:** 2026-06-20

This document consolidates **eight** full-scale evaluation runs: short-trained baseline (two eval splits), long-trained baseline, long-trained AR (June 4 + chunk-size sweep June 18).

---

## Experimental matrix (train × eval)

| Run label | Checkpoint / training | Eval JSON | Mean target length | Role in thesis |
|-----------|----------------------|-----------|-------------------|----------------|
| **Short train → short eval** | `VLKEB_baseline` epoch-20 (standard VLKEB train) | `data/VLKEB/eval.json` | ~3.7 tokens | Sanity / reproducibility on original benchmark |
| **Short train → long eval** | Same short-trained ckpt | `eval_longform_ollama.json` | ~88.6 tokens | Cross-domain ablation: no long-form training |
| **Long train → long eval (baseline)** | `VLKEB_long_baseline` epoch-20 | `eval_longform_ollama.json` | ~88.6 tokens | **Primary baseline** for long-form thesis |
| **Long train → long eval (AR cs=16)** | `VLKEB_long_ar` epoch-20 | `eval_longform_ollama.json` | ~88.6 tokens | **Primary AR result** |
| **Long AR inference ablation** | Same AR ckpt, cs=8/16/32/64 | `eval_longform_ollama.json` | ~88.6 tokens | Granularity / efficiency analysis |

---

## Table 1 — All runs at a glance

| Run | Rel acc | EM | Chunk EM | Chunk F1 | Text gen | Image gen | T-Loc | I-Loc | Mean tokens |
|-----|---------|-----|----------|----------|----------|-----------|-------|-------|-------------|
| Short train → **short eval** | **95.53%** | **85.14%** | **85.14%** | **95.53%** | 95.34% | 91.72% | 100% | 100% | 3.7 |
| Short train → **long eval** | **28.57%** | 0.00% | 1.68% | 30.62% | 28.79% | 28.85% | 100% | 100% | 88.6 |
| Long train → long eval (**baseline**) | **75.40%** | 0.00% | 13.50% | 76.28% | 75.21% | 71.49% | 100% | 100% | 88.6 |
| Long train → long eval (**AR cs=16**) | **75.94%** | 0.00% | 14.82% | 76.90% | 75.63% | 72.51% | 100% | 100% | 88.6 |
| Long AR cs=8 (inference) | 75.94% | 0.00% | 26.26% | 76.48% | 75.63% | 72.51% | 100% | 100% | 88.6 |
| Long AR cs=16 (inference) | 75.94% | 0.00% | 14.82% | 76.90% | 75.63% | 72.51% | 100% | 100% | 88.6 |
| Long AR cs=32 (inference) | 75.94% | 0.00% | 8.11% | 77.30% | 75.63% | 72.51% | 100% | 100% | 88.6 |
| Long AR cs=64 (inference) | 75.94% | 0.00% | 6.14% | 77.88% | 75.63% | 72.51% | 100% | 100% | 88.6 |

*EM = exact string match. Generality = text/image rephrase accuracy. Locality = text_loc / image_loc.*

---

## Table 2 — Long-form eval only (same data, different training / method)

All rows use `eval_longform_ollama.json` (3149 samples in 65+ token bin).

| Method | Rel acc | Δ vs long baseline | Chunk EM | Chunk F1 | Text gen | Image gen |
|--------|---------|-------------------|----------|----------|----------|-----------|
| Short-trained baseline (no long training) | 28.57% | **−46.83 pp** | 1.68% | 30.62% | 28.79% | 28.85% |
| **Long-trained baseline** | **75.40%** | — | **13.50%** | **76.28%** | **75.21%** | **71.49%** |
| Long-trained AR (cs=16) | 75.94% | **+0.54 pp** | 14.82% | 76.90% | 75.63% | 72.51% |
| Long-trained AR (cs=8) | 75.94% | +0.54 pp | 26.26%* | 76.48% | 75.63% | 72.51% |
| Long-trained AR (cs=32) | 75.94% | +0.54 pp | 8.11%* | 77.30% | 75.63% | 72.51% |
| Long-trained AR (cs=64) | 75.94% | +0.54 pp | 6.14%* | 77.88% | 75.63% | 72.51% |

*\*Chunk EM is not comparable across different inference chunk sizes (different scoring windows).*

**Key finding:** Long-form **training** is essential. A short-target editor drops from 95.5% token accuracy (short eval) to **28.6%** on long targets (−66.9 pp on the eval split). Retraining on VLKEB-long restores reliability to **75.4%** (+46.8 pp vs short-trained-on-long).

---

## Table 3 — Primary thesis comparison (long baseline vs AR cs=16)

Fair apples-to-apples: both trained on VLKEB-long, both evaluated on long-form, AR uses chunk_size=16 (matches training).

| Metric | Long baseline | AR (cs=16) | Δ (AR − baseline) |
|--------|---------------|------------|-------------------|
| Reliability — token accuracy | 75.40% | 75.94% | **+0.54 pp** |
| Reliability — chunk EM | 13.50% | 14.82% | **+1.32 pp** |
| Reliability — chunk F1 | 76.28% | 76.90% | **+0.62 pp** |
| Reliability — exact match | 0.00% | 0.00% | 0.00 pp |
| Generality — text rephrase | 75.21% | 75.63% | +0.42 pp |
| Generality — image rephrase | 71.49% | 72.51% | +1.02 pp |
| Locality — text | 100.00% | 100.00% | 0.00 pp |
| Locality — image | 100.00% | 100.00% | 0.00 pp |
| 65+ bin — token accuracy (n=3149) | 75.40% | 75.95% | +0.55 pp |
| 65+ bin — chunk EM | 13.51% | 14.82% | +1.31 pp |

---

## Table 4 — AR inference chunk-size ablation (long-trained AR ckpt)

Same checkpoint (`VLKEB_long_ar` epoch-20, trained with cs=16); only inference chunk size varies.

| chunk_size | Rel acc | Chunk EM | Chunk F1 | Text gen | Image gen | Locality |
|------------|---------|----------|----------|----------|-----------|----------|
| 8 | 75.94% | 26.26% | 76.48% | 75.63% | 72.51% | 100% |
| **16 (default)** | **75.94%** | **14.82%** | **76.90%** | **75.63%** | **72.51%** | **100%** |
| 32 | 75.94% | 8.11% | 77.30% | 75.63% | 72.51% | 100% |
| 64 | 75.94% | 6.14% | 77.88% | 75.63% | 72.51% | 100% |

- Token accuracy, generality, and locality are **identical** across all four chunk sizes.
- Chunk EM **decreases** with larger windows (26.3% → 6.1%); do not rank methods by this metric across sizes.
- Chunk F1 rises slightly with larger chunks (76.5% → 77.9%); partly a metric-window effect.
- **Recommendation:** cs=16 — matches training and enables fair comparison to long baseline.

---

## Results paragraph (paste-ready — full story)

We evaluated eight configurations under a fixed protocol (3150 sequential edits, seed 42, BLIP2-OPT-2.7b + LiveEdit). On the **original short-target VLKEB eval split**, a short-trained baseline reached **95.5%** token accuracy and **85.1%** exact match (mean target length 3.7 tokens), confirming the editor behaves as expected in its native setting. When the **same short-trained checkpoint** was evaluated on **VLKEB-long** (mean target length 88.6 tokens), reliability collapsed to **28.6%** token accuracy and **1.7%** chunk EM, while locality remained at 100%—demonstrating that long-form editing requires dedicated long-form training, not merely a longer eval split. After retraining on VLKEB-long, the long-trained baseline recovered to **75.4%** token accuracy (+46.8 pp vs short-trained-on-long). **AR-LiveEdit** (chunk size 16, long-trained) further improved reliability to **75.9%** (+0.54 pp), chunk EM to **14.8%** (+1.3 pp), and chunk F1 to **76.9%** (+0.6 pp), with generality gains on text (+0.4 pp) and image (+1.0 pp) rephrases and no locality degradation. An inference-time chunk-size ablation (8/16/32/64) showed identical token-level performance across settings; cs=16 is the recommended default. Exact match was 0% for all long-form runs, as expected for ~89-token targets.

---

## Discussion — bullet points

1. **Training data dominates target length.** The largest effect in this study is not AR vs baseline (+0.5 pp) but **short-trained vs long-trained on long eval** (−46.8 pp). The thesis must separate “long-form benchmark extension + training” from “AR mechanism.”

2. **Short-trained-on-long is a useful negative control**, not a fair AR competitor. It shows what happens if one deploys a standard VLKEB editor on long targets without retraining.

3. **AR adds consistent but modest gains** on top of long-form training: +0.54 pp token acc, +1.32 pp chunk EM, +0.62 pp chunk F1, with 100% locality preserved.

4. **Exact match is misleading for long targets.** Short eval EM = 85.1%; long eval EM = 0% for all long-trained systems. Report token accuracy and chunk metrics for VLKEB-long.

5. **Chunk-size ablation:** inference granularity does not change token accuracy among AR runs; cs=16 aligns with training. Do not claim cs=64 “beats baseline more” based on chunk F1 alone.

6. **Locality is stable** across all eight runs (100% text and image probes), including the failing short-trained-on-long case—suggesting the editor preserves unrelated knowledge even when it cannot satisfy long edit targets.

---

## Source result paths

| Run | Folder |
|-----|--------|
| Short train → short eval | `eval_results/liveedit/blip2-opt-2.7b/VLKEB-VLKEB_baseline_short_train_short_eval-2026.06.20-09.42.39` |
| Short train → long eval | `eval_results/liveedit/blip2-opt-2.7b/VLKEB-VLKEB_baseline_short_train_long_eval-2026.06.20-10.05.42` |
| Long baseline | `eval_results/liveedit/blip2-opt-2.7b/VLKEB-VLKEB_long_baseline_full-2026.06.04-17.39.13` |
| Long AR (cs=16, June 4) | `eval_results/liveedit/blip2-opt-2.7b/VLKEB-ar-VLKEB_long_ar_full-2026.06.04-18.03.18` |
| Long AR cs=8 | `eval_results/liveedit/blip2-opt-2.7b/VLKEB-ar-VLKEB_long_ar_cs8_full-2026.06.18-13.14.37` |
| Long AR cs=16 | `eval_results/liveedit/blip2-opt-2.7b/VLKEB-ar-VLKEB_long_ar_cs16_full-2026.06.18-14.22.51` |
| Long AR cs=32 | `eval_results/liveedit/blip2-opt-2.7b/VLKEB-ar-VLKEB_long_ar_cs32_full-2026.06.18-14.56.55` |
| Long AR cs=64 | `eval_results/liveedit/blip2-opt-2.7b/VLKEB-ar-VLKEB_long_ar_cs64_full-2026.06.18-15.20.16` |

Each folder: `sequential_edit_50/seed_42_mean_results.json`

---

## Checkpoints used

| Run | Checkpoint |
|-----|------------|
| Short-trained baseline | `records/liveedit/blip2-opt-2.7b/VLKEB_baseline-2026.04.09-08.20.07/checkpoints/epoch-20-i-25950-ema_loss-0.1201` |
| Long-trained baseline | `records/.../VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936` |
| Long-trained AR | `records/.../VLKEB_long_ar-2026.06.04-13.00.34/checkpoints/epoch-20-i-26600-ema_loss-1.5124` |

---

## Related thesis documents

| File | Content |
|------|---------|
| `THESIS_RESULTS_VLKEB_LONG.md` | Long baseline vs AR (June 4 runs) |
| `THESIS_BASELINE_VS_AR_COMPARISON.md` | Long baseline vs all AR chunk sizes |
| `THESIS_AR_CHUNK_SIZE_SWEEP.md` | AR-only chunk ablation |
| `THESIS_FULL_EVAL_COMPARISON.md` | **This file** — all 8 runs including short-trained cross-eval |
