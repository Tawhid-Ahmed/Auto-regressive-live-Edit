# Step 3 — Image Alteration Robustness

**Date:** 2026-07-05  
**Protocol:** seed 42, sequential_edit_n=50, dsn=500, baseline ckpt  
**Checkpoint:** `records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936`  
**Probe dataset:** [`data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json`](../../../../data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json)

**Mechanism under test:** Edit always uses the **original** image. At test time, the **same entity** is shown with on-the-fly appearance changes (low-res, rotate, mirror, crop, mild degradation). Measures whether stored edits survive view/quality change vs Step 1 swap (wrong entity) or image rephrase (different photo).

**Raw results:**
- Eval run: [`eval_results/meeting_validation/step3/.../VLKEB_ALTERATION-step3_alteration_baseline-2026.07.05-14.34.23/`](../../../../eval_results/meeting_validation/step3/liveedit/blip2-opt-2.7b/VLKEB_ALTERATION-step3_alteration_baseline-2026.07.05-14.34.23/sequential_edit_50/seed_42_mean_results.json)
- Summary JSON: [`step3_alteration_summary.json`](step3_alteration_summary.json)
- Run manifest: [`step3_run_manifest.json`](step3_run_manifest.json)

---

## Summary

| Metric | Accuracy | Gap vs reliability |
|--------|----------|-------------------|
| **Reliability** (original image) | **74.65%** | — |
| Image rephrase (different photo) | 71.84% | 2.81 pp |
| **Center crop** | **73.34%** | **1.31 pp** |
| **Rotate 90 deg** | **73.41%** | **1.24 pp** |
| **Low resolution** | **73.63%** | **1.02 pp** |
| **Mild blur/brightness** | **74.19%** | **0.46 pp** |
| **Mirror (H-flip)** | **74.31%** | **0.34 pp** |
| Mean alteration gap | — | **0.87 pp** |
| Locality (text / image) | 100% / 100% | — |
| Runtime | ~6.7 min | — |

*Spread across alteration types: 0.97 pp (max gap 1.31 - min gap 0.34).*

---

## Individual alteration effects

Ranked by **harm to edit retention** (largest drop from reliability first):

| Rank | Alteration | What it simulates | Acc | Drop (pp) | Reading |
|------|------------|-------------------|-----|-----------|---------|
| 1 | **Center crop** | Partial field of view, 75% area | 73.34% | **−1.31** | Largest hit — routing may rely on global framing / context at image borders |
| 2 | **Rotate 90 deg** | Orientation change (phone rotation) | 73.41% | **−1.24** | Pose/view sensitive; Q-Former key not rotation-invariant |
| 3 | **Low resolution** | Downscale to 112 px short side, upscale restore | 73.63% | **−1.02** | Moderate loss; consistent with Step 2 (resolution alone barely moves metrics) but **test-time** downscale hurts slightly more than processor upscale in Step 2 |
| 4 | **Mild blur/brightness** | JPEG-like quality / lighting drift | 74.19% | **−0.46** | Small effect — semantic content preserved |
| 5 | **Mirror (H-flip)** | Left-right flip | 74.31% | **−0.34** | Most robust — nearly matches reliability |

### Cross-probe comparison

| Probe family | Typical gap | Interpretation |
|--------------|-------------|----------------|
| Step 1 **image swap** (wrong entity) | ~31 pp | Catastrophic — wrong routing key |
| Step 1 **hard negative** | ~31 pp (43% abs acc) | False expert activation |
| **Image rephrase** (same entity, new photo) | 2.81 pp | Moderate — different capture, same subject |
| **Step 3 alterations** (same entity, same photo transformed) | **0.34–1.31 pp** | Mild — edits mostly survive appearance drift |
| Step 2 **resolution sweep** (224→448 processor) | 0.13 pp rel spread | Pre-Q-Former detail unused |

**Key contrast:** Geometric/quality alterations on the **same** image are far less damaging than showing a **different** image (swap: −31 pp) or even a **different photo of the same entity** (rephrase: −2.8 pp). LiveEdit retains edits under mild view change but does not achieve true invariance (rotate/crop still cost ~1.2 pp).

---

## Interpretation

1. **Edits are partially robust to same-image appearance change.** Mean alteration gap is only 0.87 pp vs 31 pp for wrong-entity swap — the editor binds to entity identity more than pixel-perfect view.

2. **Geometry hurts more than quality.** Crop and rotation (−1.2–1.3 pp) exceed mirror and mild degradation (−0.3–0.5 pp). The pooled Q-Former key is weakly sensitive to layout/orientation, not just resolution.

3. **Alterations are easier than image rephrase.** Rephrase (−2.8 pp) drops twice as much as the worst alteration (−1.3 pp). A genuinely different photograph is a harder generalization test than transforming the edit-time image.

4. **Not resolution-dominated.** Low-res alteration (−1.0 pp) aligns with Step 2’s flat processor-resolution curve — increasing ViT input size at eval does not help, but **degrading** the test image slightly does, suggesting the bottleneck remains the fixed 32-token routing key, not missing pixels.

---

## Gate

**REFRAME** — alteration gaps are small and flat (mean 0.87 pp, spread &lt; 2 pp). Edits survive appearance drift on the same image but are not perfectly view-invariant.

**Proceed to Step 4** (encoder probe: pre-Q-Former patches vs Q-Former vs DINOv2 offline routing).

---

## Reproduce

```cmd
cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"

python tools/meeting_validation/step3/run_alteration_eval.py -dvc cuda:0 -dsn 500 -seed 42

python tools/meeting_validation/step3/analyze_alteration_results.py ^
  eval_results\meeting_validation\step3\liveedit\blip2-opt-2.7b\VLKEB_ALTERATION-step3_alteration_baseline-*\sequential_edit_50\seed_42_mean_results.json ^
  --label baseline -out doc/results/meeting_validation/step3/step3_alteration_summary.json

python tools/meeting_validation/step3/plot_alteration_results.py ^
  --analysis-json doc/results/meeting_validation/step3/step3_alteration_summary.json
```

---

## Figures

| Figure | File |
|--------|------|
| Accuracy + per-alteration gap | [`step3_alteration_robustness.png`](step3_alteration_robustness.png) |

---

## Code touchpoints (patch only)

| Change | File |
|--------|------|
| PIL transforms | [`dataset/image_alterations.py`](../../../../dataset/image_alterations.py) |
| Lazy `AlteredImageRef` + loader | [`dataset/vllm_alteration.py`](../../../../dataset/vllm_alteration.py), [`dataset/vllm.py`](../../../../dataset/vllm.py) `pil_image_from_path_or_pil` |
| Eval runner | [`test_alteration_eval.py`](../../../../test_alteration_eval.py) |
| Analyze / plot | [`tools/meeting_validation/step3/`](../../../../tools/meeting_validation/step3/) |

Legacy `test_vllm_edit.py` unchanged. `AlteredImageRef` is only resolved when passed through the VLLM image loader.
