# Step 6 — Extra Patch-Grounding Layer (Retrain + Ablations)

**Date:** 2026-07-08  
**Protocol:** seed 42, `sequential_edit_n=50`, Step 1 grounding JSON, DINOv2 patch encoder  
**Train data:** `../vlkeb_long_ollama/out/train_longform_ollama.json`  
**Train checkpoint (final):** `records/liveedit/blip2-opt-2.7b/VLKEB_long_patch-2026.07.08-11.37.09/checkpoints/epoch-20-i-26250-ema_loss-1.3055`  
**Status:** **PARTIAL** — L0 via Step 1 proxy; L1–L4 complete; **L5 (AR)** eval crashed (`numpy._core._exceptions._ArrayMemoryError` during DINOv2 patch encoding — see [`l0_l5_run_status.md`](l0_l5_run_status.md)).

---

## Artifacts (paths used for this analysis)

| Artifact | Path |
|----------|------|
| Run manifest | [`step6_run_manifest.json`](step6_run_manifest.json) |
| Summary JSON | [`step6_ablation_summary.json`](step6_ablation_summary.json) |
| Markdown table | [`step6_ablation_table.md`](step6_ablation_table.md) |
| Bar chart | [`step6_layer_ablation.png`](step6_layer_ablation.png) |
| Grounding probe JSON | `data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json` |

### Raw eval inputs (per layer)

| Layer | `seed_42_mean_results.json` |
|-------|----------------------------|
| L1 | `eval_results/meeting_validation/step6/liveedit/blip2-opt-2.7b/VLKEB_GROUNDING-step6_l1-2026.07.08-15.47.27/sequential_edit_50/seed_42_mean_results.json` |
| L2 | `eval_results/meeting_validation/step6/liveedit/blip2-opt-2.7b/VLKEB_GROUNDING-step6_l2-2026.07.08-15.55.50/sequential_edit_50/seed_42_mean_results.json` |
| L3 | `eval_results/meeting_validation/step6/liveedit/blip2-opt-2.7b/VLKEB_GROUNDING-step6_l3-2026.07.08-16.04.07/sequential_edit_50/seed_42_mean_results.json` |
| L4 | `eval_results/meeting_validation/step6/liveedit/blip2-opt-2.7b/VLKEB_GROUNDING-step6_l4-2026.07.08-16.12.06/sequential_edit_50/seed_42_mean_results.json` |
| L0 | Step 1 proxy (same ckpt) | [`seed_42_mean_results.json`](../../../eval_results/liveedit/blip2-opt-2.7b/VLKEB_GROUNDING-step1_grounding_baseline-2026.07.05-12.20.17/sequential_edit_50/seed_42_mean_results.json) |
| L5 | *failed* — patch ckpt + `--ar_mode` (system RAM/pagefile exhaustion; see `l5_run.log`) |

---

## Summary (L0 proxy + patch ckpt L1–L4)

| Layer | Rel acc | Hard-neg acc | Swap gap (pp) | Modality cons. | Locality |
|-------|--------:|-------------:|--------------:|---------------:|---------:|
| **L0** (baseline, no patch) | **74.65%** | **43.36%** | **31.15** | **71.84%** | 100.0% |
| **L1** | 73.6% | 42.9% | 30.7 | 71.6% | 100.0% |
| **L2** | 73.6% | 45.8% | 28.3 | 71.7% | 100.0% |
| **L3** | 73.2% | 42.9% | 30.4 | 71.3% | 100.0% |
| **L4** | 73.2% | 42.0% | 31.0 | 67.1% | 100.0% |
| L5 | — | — | — | — | — |

Rendered table: [`step6_ablation_table.md`](step6_ablation_table.md) · figure [`step6_layer_ablation.png`](step6_layer_ablation.png)

---

## Interpretation (with L0 proxy)

1. **L1 (patch) does not beat L0** on reliability (−1.0 pp) or hard-negative (−0.5 pp); swap-gap unchanged (~31 pp).
2. **L2 max-pool routing hurts** vs L0/L1: hard-neg **45.8%** (+2.4 pp vs L0) — consistent with Step 5 spike 4a.
3. **L4 verification gate** gives the smallest hard-neg (**42.0%**, −1.4 pp vs L0) but **hurts modality consistency** (67.1% vs 71.8%).
4. **L5 still required** — only remaining config that helped routing in Step 5 (vision-gated AR chunks).

---

## Gate

**PARTIAL** — L0 + L5 still required for full Step 6 gate.

| Criterion | Status |
|-----------|--------|
| L1 beats L0 on ≥2 adversarial probes | ❌ L1 ≤ L0 on hard-neg and swap-gap |
| Locality ≥95% at 100+ edits | ✅ 100% (L1–L4) |
| Each layer L2–L5 moves primary metric | ⏳ needs L0/L5 comparison |
| Proceed to Step 7 | ⏳ |

---

## Reproduce remaining evals

```cmd
cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"
conda activate liveedit

REM L5 (slow; free RAM first — close browsers/other GPU apps)
python -u -W ignore tools/meeting_validation/step6/run_ablation_eval.py -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_patch-2026.07.08-11.37.09/checkpoints/epoch-20-i-26250-ema_loss-1.3055" --layers L5 -dvc cuda:0 -dsn 500 -seed 42

REM L0 baseline reference
python -u -W ignore tools/meeting_validation/step6/run_ablation_eval.py --layers L0 -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936" -dvc cuda:0 -dsn 500 -seed 42
```

After L0/L5 finish, re-run analysis:

```cmd
python tools/meeting_validation/step6/analyze_ablation_results.py
python tools/meeting_validation/step6/plot_ablation_table.py
```

---

## Architecture

`PatchGroundingModule` sits parallel to `QVExtractor` in `editor/vllm_editors/liveedit/modules.py`. Frozen DINOv2 patch grid → cross-attention with query hidden states → `evr_patch` routing keys fused with legacy Q-Former `vis_sim` at retrieval.

Diagram: [`doc/THESIS_architecture_diagram_patch_grounding_horizontal.png`](../../THESIS_architecture_diagram_patch_grounding_horizontal.png)
