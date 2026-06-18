# VLKEB-long: Results and Discussion (paste-ready)

**Comparison artifact:** `eval_results/comparison/liveedit_vs_ar_seed42_longform.json`  
**Eval runs:** baseline `VLKEB-VLKEB_long_baseline_full-2026.06.04-17.39.13`, AR `VLKEB-ar-VLKEB_long_ar_full-2026.06.04-18.03.18`  
**Protocol:** seed 42, sequential_edit_n=50, full long-form eval JSON (3150 edits aggregated; 3149 in 65+ token bin).

---

## Results — narrative paragraph

We extended VLKEB with long-form edit targets (VLKEB-long) while preserving images, prompts, and locality structure. LiveEdit and AR-LiveEdit (chunk size 16) were trained on the long-form training split for 20 epochs and evaluated on the long-form eval split under sequential editing (50 edits per batch, fixed seed 42). AR-LiveEdit improved reliability token accuracy from 75.4% to 75.9% (+0.54 percentage points) and chunk-level exact match from 13.5% to 14.8% (+1.3 pp), with chunk F1 rising from 76.3% to 76.9%. Nearly all eval targets fell in the 65+ token length bin (n=3149); gains in that bin matched aggregate reliability. Generality on text and image rephrases improved slightly (75.2%→75.6% and 71.5%→72.5%). Locality accuracy remained 100% for both text and image probes for baseline and AR. Exact match on long targets was 0% for both systems, which is expected given mean target length ≈89 tokens; we therefore report token accuracy and chunk-level metrics as primary outcomes.

---

## Table 1 — Main metrics (VLKEB-long, seed 42)

| Metric | LiveEdit (baseline) | AR-LiveEdit (chunk=16) | Δ (AR − baseline) |
|--------|---------------------|-------------------------|---------------------|
| **Reliability — token accuracy** | 75.40% | 75.94% | +0.54 pp |
| **Reliability — chunk EM** | 13.50% | 14.82% | +1.32 pp |
| **Reliability — chunk F1** | 76.28% | 76.90% | +0.62 pp |
| **Reliability — exact match** | 0.00% | 0.00% | 0.00 pp |
| **Generality — text rephrase** | 75.21% | 75.63% | +0.42 pp |
| **Generality — image rephrase** | 71.49% | 72.51% | +1.02 pp |
| **Locality — text** | 100.00% | 100.00% | 0.00 pp |
| **Locality — image** | 100.00% | 100.00% | 0.00 pp |
| **65+ bin — token accuracy** (n=3149) | 75.40% | 75.95% | +0.55 pp |
| **65+ bin — chunk EM** | 13.51% | 14.82% | +1.31 pp |
| **65+ bin — chunk F1** | 76.28% | 76.90% | +0.62 pp |
| **Mean target length (tokens)** | 88.6 | 88.6 | — |
| **Total edits evaluated** | 3150 | 3150 | — |

*pp = percentage points. EM = exact string match.*

---

## Table 2 — Thesis validation flags

| Criterion | Result | Source |
|-----------|--------|--------|
| Long-form improvement (65+ accuracy) | **Pass** (+0.55 pp) | `validation.long_form_improvement` |
| Locality bounded (≤5% drop) | **Pass** (0% drop) | `validation.locality_bounded` |
| Generality not degraded | **Pass** (both rephrase metrics up) | Table 1 |

---

## Discussion — bullet points

1. **Mechanism-aligned gains.** The largest relative improvement appears on chunk EM (+1.3 pp) rather than token accuracy alone (+0.5 pp), which supports the claim that autoregressive chunk-wise editing helps the model satisfy long targets in segment-sized units, not only via per-token overlap.

2. **Exact match is the wrong headline metric.** With mean target length ≈89 tokens, 0% EM for both systems is unsurprising; the thesis should emphasize token accuracy and chunk EM/F1, and optionally report that verbatim EM is intentionally strict for long-form answers.

3. **Locality trade-off.** On this run, AR did not reduce locality below baseline (both at 100% on text and image probes). That bounds one failure mode, but it does not prove locality under all sequential pressures; discuss sequential batch size (50) and full-benchmark scale as stressors already applied here.

4. **Magnitude and limitations.** Improvements are statistically modest in absolute terms; frame them as consistent directional evidence on a new long-form benchmark extension, not as large SOTA gains. Long targets were generated with lexical anchoring to the original short `alt` (VLKEB-long protocol); do not claim full visual grounding of every phrase without human audit.

5. **Training vs eval.** AR training loss remained higher than baseline at epoch 20; despite that, AR improved eval metrics, suggesting the chunk-wise inference path transfers even when the AR training objective is harder—worth a short paragraph or footnote.

---

## Experimental details (Methods cross-reference)

| Item | Value |
|------|--------|
| Backbone | BLIP2-OPT-2.7b + LiveEdit |
| Train data | `vlkeb_long_ollama/out/train_longform_ollama.json` |
| Eval data | `vlkeb_long_ollama/out/eval_longform_ollama.json` |
| Baseline checkpoint | `epoch-20-i-27800-ema_loss-0.7936` (VLKEB_long_baseline, 2026-06-02) |
| AR checkpoint | `epoch-20-i-26600-ema_loss-1.5124` (VLKEB_long_ar, 2026-06-04) |
| AR chunk size | 16 |
| Eval seed | 42 |
| Sequential edits per batch | 50 |

---

## Figure caption (optional)

*Figure X: LiveEdit vs AR-LiveEdit on VLKEB-long (n=3150 sequential edits, seed 42). AR improves reliability token accuracy and chunk-level exact match on predominantly 65+ token targets without reducing locality accuracy on text or image probes.*
