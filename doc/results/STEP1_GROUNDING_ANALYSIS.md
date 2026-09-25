# Step 1 — Adversarial / Grounding Probe Suite

**Patch layout:** Step 1 is fully opt-in. Original workflows (`test_vllm_edit.py`, `dataset.vllm.VLKEB`) are unchanged.

**Date:** 2026-07-05  
**Protocol:** seed 42, sequential_edit_n=50, dsn=500, grounding JSON  
**Checkpoints:** baseline `epoch-20-i-27800-ema_loss-0.7936`, AR `epoch-20-i-26600-ema_loss-1.5124`  
**Probe dataset:** [`data/VLKEB/grounding/grounding_eval_longform.json`](../data/VLKEB/grounding/grounding_eval_longform.json)

**Raw results:**
- Baseline: [`eval_results/.../VLKEB_GROUNDING-step1_grounding_baseline-2026.07.05-12.20.17/`](../../eval_results/liveedit/blip2-opt-2.7b/VLKEB_GROUNDING-step1_grounding_baseline-2026.07.05-12.20.17/sequential_edit_50/seed_42_mean_results.json)
- AR: [`eval_results/.../VLKEB_GROUNDING-ar-step1_grounding_ar-2026.07.05-12.28.20/`](../../eval_results/liveedit/blip2-opt-2.7b/VLKEB_GROUNDING-ar-step1_grounding_ar-2026.07.05-12.28.20/sequential_edit_50/seed_42_mean_results.json)
- Summary JSON: [`step1_grounding_summary.json`](step1_grounding_summary.json)

---

## Summary

| Metric | Baseline | AR | Δ (AR − base) |
|--------|----------|-----|---------------|
| Reliability | 74.65% | 75.45% | +0.80 pp |
| Text rephrase | 74.44% | 74.94% | +0.50 pp |
| Image rephrase | 71.84% | 72.57% | +0.73 pp |
| **Image swap** | **43.50%** | **42.63%** | −0.87 pp |
| **Hard negative** | **43.36%** | **42.41%** | −0.95 pp |
| Modality consistency | 71.84% | 72.57% | +0.73 pp |
| Prompt conflict | 74.49% | 75.16% | +0.67 pp |
| **Swap-gap** (rel − swap) | **31.15 pp** | **32.82 pp** | +1.67 pp |
| Consistency-gap (rel − mod.) | 2.81 pp | 2.88 pp | +0.07 pp |
| Locality (text / image) | 100% / 100% | 100% / 100% | — |
| Runtime | 5.81 min | 5.65 min | — |

---

## Interpretation

1. **Swap and hard-negative probes expose a large visual stress gap.** On the correct image, reliability is ~75%; on a swapped or confusable wrong image, token accuracy falls to ~43% (−31 pp). Edits are not purely prompt-driven, but **~43% false generalization** to wrong images remains — a substantial lifelong routing failure mode.

2. **Modality consistency tracks image rephrase** (identical by construction: both use `image_rephrase`). The 2.8 pp gap vs reliability matches Step 0’s image-rephrase degradation pattern; text rephrase stays near reliability.

3. **Prompt–image conflict probe shows no language-prior dominance** with the current misleading prefix — conflict accuracy equals reliability (74.5% vs 74.6%). The editor follows the image+edit signal; the conflict prompt is either too weak or the model ignores it.

4. **AR gives marginal gains** on reliability and consistency (+0.7–0.8 pp) and slightly lowers swap/hard-negative false rates (−0.9 pp), but does not fix the core ~43% adversarial-image problem.

---

## Gate

**REFRAME → Proceed to Steps 2–3** (automated gate: baseline does not fail ≥2 of the 4 scripted thresholds).

| Criterion | Threshold | Baseline | Verdict |
|-----------|-----------|----------|---------|
| Swap-gap | &lt; 5 pp = fail | 31.15 pp | Pass (swap hurts) |
| Hard-negative acc | &gt; 50% = fail | 43.36% | Pass (below threshold) |
| Consistency-gap | &gt; 3 pp = fail | 2.81 pp | Pass |
| Conflict acc | &lt; rel − 5 pp = fail | 74.49% vs 74.65% | Pass |

**Scientific reading (stronger than gate labels):** The swap/hard-negative **absolute rates (~43%)** still support the Step 0 visual-bottleneck story — wrong-image probes retain nearly half the edited answer mass. Steps 2–3 (resolution, encoder) are warranted before any patch-layer retrain (Steps 4–5).

**Optional:** Run routing diagnostics (steps 4–5 below) for MoE false-fire rates complementary to token acc.

---

## Reproduce

### Build (done)
```cmd
python tools/build_grounding_eval_json.py -dpath "E:\MSC\MSc thesis project\LiveEdit\vlkeb_long_ollama\out\eval_longform_ollama.json" --img_root "data/VLKEB/VLKEB_images/mmkb_images" -out data/VLKEB/grounding/grounding_eval_longform.json
```

### Analyze + plot (done on these runs)
```cmd
python tools/analyze_grounding_results.py eval_results\liveedit\blip2-opt-2.7b\VLKEB_GROUNDING-step1_grounding_baseline-2026.07.05-12.20.17\sequential_edit_50\seed_42_mean_results.json eval_results\liveedit\blip2-opt-2.7b\VLKEB_GROUNDING-ar-step1_grounding_ar-2026.07.05-12.28.20\sequential_edit_50\seed_42_mean_results.json --labels baseline ar -out doc/results/step1_grounding_summary.json

python tools/plot_grounding_results.py --analysis-json doc\results\step1_grounding_summary.json
```

### Single-sample deep dive (one fact, all probes)

```cmd
python tools/run_grounding_single_sample.py --list 20

python tools/run_grounding_single_sample.py -idx 0 -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936" -dvc cuda:0 --routing -out doc/results/grounding_single_idx0.json
```

Use `-idx N` for `source_index` in the JSON. `--pool_before 49 -seed 42` adds 49 distractor edits before the target (lifelong pool).

### Routing diagnostics (optional, GPU)
```cmd
python tools/log_grounding_diagnostics.py -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936" -dpath data/VLKEB/grounding/grounding_eval_longform.json -dvc cuda:0 -dsn 150 -seed 42 -out doc/results/grounding_diagnostics_baseline.json

python tools/log_grounding_diagnostics.py -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_ar-2026.06.04-13.00.34/checkpoints/epoch-20-i-26600-ema_loss-1.5124" -dpath data/VLKEB/grounding/grounding_eval_longform.json -dvc cuda:0 -dsn 150 -seed 42 --ar_mode --chunk_size 16 -out doc/results/grounding_diagnostics_ar.json
```

---

## Figures

| Figure | File |
|--------|------|
| Swap gap | [`step1_swap_gap.png`](step1_swap_gap.png) |
| Probe rates | [`step1_probe_rates.png`](step1_probe_rates.png) |

---

## Single-sample case study — `source_index=0` (Kankakee, Illinois)

**Date:** 2026-07-05  
**Tool:** `tools/run_grounding_single_sample.py`  
**Checkpoint:** baseline `epoch-20-i-27800-ema_loss-0.7936`  
**Sample:** “What city in Illinois is depicted in the image?” → long-form alt **Kankakee**  
**Partner images:** swap = Salinas CA skyline (row 2); hard-neg = Fort Smith AR waterfall (row 1)

**Raw JSON:**
- Isolated (pool size 1): [`grounding_single_idx0_isolated.json`](grounding_single_idx0_isolated.json)
- Pooled (49 distractors + target, seed 42): [`grounding_single_idx0_pooled49.json`](grounding_single_idx0_pooled49.json)

### Per-probe results

| Probe | Isolated acc | Pooled acc | Isolated routing | Pooled routing |
|-------|-------------|------------|------------------|----------------|
| Reliability | **63.27%** | **63.27%** | `[0]` ✓ clean | `[21, 49]` ✓ hit + **false cross** |
| Text rephrase | 65.31% | 65.31% | `[0]` clean | `[21, 49]` false cross |
| Image rephrase | 64.29% | 64.29% | `[0]` clean | `[21, 33, 49]` false cross |
| **Image swap** | **42.86%** | **42.86%** | `[]` no fire | `[]` no fire |
| **Hard negative** | **41.84%** | **41.84%** | `[]` no fire | `[]` no fire |
| Modality consistency | 64.29% | 64.29% | `[0]` clean | `[21, 33, 49]` false cross |
| Prompt conflict | 63.27% | 63.27% | `[0]` clean | `[21, 49]` false cross |
| Image locality | 100% | 100% | `[]` clean | `[]` clean |

**Swap-gap (this sample):** 20.41 pp (reliability 63.27% − swap 42.86%) — smaller than batch mean (31.15 pp) because this sample’s reliability is below the 500-sample average (74.65%).

### What the predictions look like

| Probe | Prediction preview (truncated) |
|-------|-------------------------------|
| Reliability | “The image showcases a detailed view of a traditional **brick church**… tower topped with a cross…” (Kankakee edit firing) |
| Image swap | “…city view of the city **Chicago** building… church… arched windows…” (wrong city; partial church-like hallucination) |
| Hard negative | Same family as swap — **Chicago** + church vocabulary, not Fort Smith answer |

On wrong images the model does **not** output Salinas/Fort Smith facts; it still produces church/city language partially aligned with the **scored target** (Kankakee paragraph), at ~42% token overlap.

### Interpretation (single-sample)

1. **Isolated mode (pool=1): routing is clean on correct images** — only expert `[0]` fires; swap/hard-neg retrieve **no experts** (`[]`), yet token acc stays ~42%. So part of the adversarial “leakage” is **not** wrong-expert MoE firing; it is partial target-aligned generation without retrieval (prompt + generic church prior).

2. **Pooled mode (pool=50): same token acc, worse routing on correct images** — reliability co-retrieves distractor expert `[21]` alongside target `[49]`. Image rephrase pulls in a third expert `[33]`. This matches Step 0’s co-retrieval pattern at the **single-fact** level.

3. **Swap/hard-neg routing stays empty in both modes** — prototype gate blocks wrong-image queries from retrieving pool entries, but **token accuracy remains ~42%**. Batch-level ~43% adversarial acc therefore mixes (a) routing false fires on other samples and (b) **non-routed partial edit leakage** visible here.

4. **Prompt conflict unchanged (63.27%)** — misleading “Laredo, Texas” prefix has no effect on this sample, consistent with batch result.

5. **This sample is harder than average** — reliability 63% vs batch 75%; do not treat one sample as representative of mean metrics.

### Reproduce single-sample runs

```cmd
python tools/run_grounding_single_sample.py -idx 0 -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936" -dvc cuda:0 --routing -out doc/results/grounding_single_idx0_isolated.json

python tools/run_grounding_single_sample.py -idx 0 -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936" -dvc cuda:0 --pool_before 49 -seed 42 --routing -out doc/results/grounding_single_idx0_pooled49.json
```
