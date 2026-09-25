# Step 2 — Image Resolution Impact

**Date:** 2026-07-05  
**Protocol:** seed 42, sequential_edit_n=50, dsn=500, Step 1 grounding probes, baseline ckpt  
**Checkpoint:** `records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936`  
**Probe dataset:** [`data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json`](../../../../data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json)

**Mechanism under test:** BLIP2 EVA-ViT patch grid grows with input resolution (257 → 577 → 1025 patches), but the Q-Former still compresses to **32 tokens** for LiveEdit routing. If the bottleneck is post-Q-Former, grounding metrics should stay flat across 224 / 336 / 448 px.

**Raw results:**
- res224: [`eval_results/meeting_validation/step2/.../VLKEB_GROUNDING-step2_res224-2026.07.05-13.22.43/`](../../../../eval_results/meeting_validation/step2/liveedit/blip2-opt-2.7b/VLKEB_GROUNDING-step2_res224-2026.07.05-13.22.43/sequential_edit_50/seed_42_mean_results.json)
- res336: [`VLKEB_GROUNDING-step2_res336-2026.07.05-13.28.37/`](../../../../eval_results/meeting_validation/step2/liveedit/blip2-opt-2.7b/VLKEB_GROUNDING-step2_res336-2026.07.05-13.28.37/sequential_edit_50/seed_42_mean_results.json)
- res448: [`VLKEB_GROUNDING-step2_res448-2026.07.05-13.36.41/`](../../../../eval_results/meeting_validation/step2/liveedit/blip2-opt-2.7b/VLKEB_GROUNDING-step2_res448-2026.07.05-13.36.41/sequential_edit_50/seed_42_mean_results.json)
- Summary JSON: [`step2_resolution_summary.json`](step2_resolution_summary.json)
- Run manifest: [`step2_run_manifest.json`](step2_run_manifest.json)

---

## Summary

| Metric | 224 px | 336 px | 448 px | Spread (max − min) |
|--------|--------|--------|--------|---------------------|
| Reliability | 74.65% | 74.58% | 74.52% | **0.13 pp** |
| Image swap | 43.50% | 43.57% | 44.29% | 0.79 pp |
| Hard negative | 43.36% | 43.12% | 43.74% | **0.62 pp** |
| Modality consistency | 71.84% | 70.39% | 71.02% | 1.45 pp |
| Prompt conflict | 74.49% | 74.40% | 74.29% | 0.20 pp |
| **Swap-gap** (rel − swap) | **31.15 pp** | **31.01 pp** | **30.23 pp** | **0.92 pp** |
| ViT patches (smoke test) | 257 | 577 | 1025 | — |
| Q-Former tokens (fixed) | 32 | 32 | 32 | — |
| Runtime (GPU) | 5.7 min | 7.9 min | 11.2 min | — |

*Spreads from [`step2_resolution_summary.json`](step2_resolution_summary.json). Step 1 baseline @224 (legacy run) matches res224 within rounding (74.65% rel, 31.15 pp swap-gap).*

---

## Interpretation

1. **Pre-Q-Former resolution does not move grounding metrics.** Despite 4× more ViT patches at 448 px, reliability and adversarial probes vary by ≤1 pp. Extra spatial detail is discarded at the Q-Former before LiveEdit ever sees it.

2. **Swap-gap stays ~31 pp at all resolutions.** The Step 1 visual stress gap is resolution-invariant — consistent with routing on a fixed 32-token key, not on patch geometry.

3. **448 px slightly *increases* hard-negative false activation (+0.38 pp vs 224).** If anything, higher resolution adds ViT noise without helping discrimination — not evidence that Q-Former already suffices.

4. **Implication for Steps 4–5:** Resolution gains require a patch-aware routing layer (Step 4 spikes / Step 5 PatchGroundingModule), not processor resize alone.

---

## Gate

**PASS** — flat curves (reliability spread 0.13 pp, swap-gap spread 0.92 pp, hard-negative spread 0.62 pp; all &lt; 2 pp threshold).

**Proceed to Step 3** (image alteration robustness: rotate, mirror, crop, low-res test views).

---

## Reproduce

### Smoke test (ViT patches vs Q-Former tokens)
```cmd
python tools/meeting_validation/step2/smoke_test_image_size.py
```

### Full sweep (224 / 336 / 448)
```cmd
cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"

python tools/meeting_validation/step2/run_resolution_sweep.py -dvc cuda:0 -dsn 500 -seed 42
```

Or individual runs:
```cmd
python -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 50 -dvc cuda:0 -dsn 500 -seed 42 -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936" -enp step2_res336 --meeting_step 2 --image_size 336
```

### Analyze + plot
```cmd
python tools/meeting_validation/step2/analyze_resolution_results.py --eval-dir eval_results/meeting_validation/step2 -out doc/results/meeting_validation/step2/step2_resolution_summary.json

python tools/meeting_validation/step2/plot_resolution_results.py --analysis-json doc/results/meeting_validation/step2/step2_resolution_summary.json
```

---

## Figures

| Figure | File |
|--------|------|
| Resolution curve (probes + swap-gap) | [`step2_resolution_curve.png`](step2_resolution_curve.png) |

---

## Code touchpoints (patch only)

| Change | File |
|--------|------|
| Eval-only `--image_size` + ViT pos-embed interpolation | [`editor/vllms_for_edit/blip2/blip2.py`](../../../../editor/vllms_for_edit/blip2/blip2.py) `set_eval_image_size`, `interpolate_pos_encoding` |
| Step 2 eval routing + metadata | [`test_grounding_eval.py`](../../../../test_grounding_eval.py) `--meeting_step 2`, `--image_size` |
| Sweep / analyze / plot | [`tools/meeting_validation/step2/`](../../../../tools/meeting_validation/step2/) |

Legacy `test_vllm_edit.py` and default BLIP2 224 behaviour are unchanged.
