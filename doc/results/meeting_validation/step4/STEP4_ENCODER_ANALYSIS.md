# Step 4 — Encoder Offline Routing Probe

**Date:** 2026-07-05  
**Protocol:** seed 42, n=50 sequential edits, grounding JSON (Step 1 probes)  
**Checkpoint:** baseline `epoch-20-i-27800-ema_loss-0.7936`  
**Probe dataset:** [`data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json`](../../../data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json)

**Raw results:** [`encoder_routing_scores.json`](encoder_routing_scores.json)

---

## Summary

| Metric | A: Q-Former | B: EVA patches | C: DINOv2 |
|--------|-------------|----------------|-----------|
| Hard-neg false-fire (own expert on wrong image) | 4.0% | 8.0% | **2.0%** |
| Hard-neg AUC (margin own vs other) | 0.979 | **1.000** | **1.000** |
| Reliability miss rate | 0.0% | 0.0% | 0.0% |
| Reliability false-cross rate | **76.0%** | 0.0% | 0.0% |
| Mean own-expert score (hard neg) | 9.31 | 0.016 | **0.008** |

**Encoder conditions (inference-only, no retrain):**

| ID | Signal | Routing mechanism |
|----|--------|-------------------|
| **A** | Q-Former 32 tokens → `QVExtractor` → `evr_pool` | LiveEdit `vis_sim > vis_sim_prot` (actual editor) |
| **B** | Pre-Q-Former EVA-ViT `image_embeds` (1408-d patches) | Mean-pool + L2 norm; retrieve own expert if sim ≥ best other |
| **C** | Frozen `facebook/dinov2-base` patch grid (768-d) | Same competitive margin as B |

---

## Interpretation

1. **Q-Former routing is permissive on reliability** — 76% of reliability probes co-retrieve at least one wrong-sample expert (`false_cross_edit`), matching Step 0’s “co-retrieval with wrong experts” diagnosis. Patch encoders with competitive margin routing avoid this entirely (0% false-cross on reliability).

2. **DINOv2 is the strongest offline hard-negative discriminator** — lowest false-fire on confusable images (2% vs 4% Q-Former) and perfect margin AUC (1.0). Mean own-expert cosine to the correct pool entry drops from 9.3 (Q-Former space) to 0.008 (DINO space) on hard negatives, indicating sharper spatial separation.

3. **EVA native patches improve AUC but raise false-fire slightly** — margin AUC reaches 1.0 (better separation score than Q-Former’s 0.979) but false-fire rises to 8%. Pre-Q-Former patches carry more spatial detail than the collapsed Q-Former key but are less selective than DINOv2 on confusable entities.

4. **Aligns with Step 1 token-acc stress** — Step 1 hard-negative token accuracy was ~43% (large gap vs reliability). Offline routing shows the bottleneck is not only “wrong expert wins” but also Q-Former’s high co-retrieval rate; patch/DINO signals offer a cleaner routing surface before any retrain.

---

## Gate

**PASS → Proceed to Step 5** (patch + cross-attention inference spikes).

| Criterion | Result |
|-----------|--------|
| B or C beats A on hard-negative AUC | ✅ Both EVA patches and DINOv2 reach AUC 1.0 vs Q-Former 0.979 |
| Pick winning encoder for Step 6 | **DINOv2** — lowest false-fire (2%) + perfect margin AUC |

**Note:** Run used n=50 for GPU time (~30 s after model cache). Scale to n=150/500 with the reproduce command below for thesis-facing numbers.

---

## Reproduce

```powershell
cd "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"

conda activate liveedit

python tools/meeting_validation/step4/compare_encoder_routing.py `
  -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936" `
  -dvc cuda:0 -dsn 50 -seed 42 `
  -dpath "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json" `
  --img_root "data/VLKEB/VLKEB_images/mmkb_images" `
  -out doc/results/meeting_validation/step4/encoder_routing_scores.json

python tools/meeting_validation/step4/analyze_encoder_routing.py doc/results/meeting_validation/step4/encoder_routing_scores.json

python tools/meeting_validation/step4/plot_encoder_auc.py --scores-json doc/results/meeting_validation/step4/encoder_routing_scores.json
```

Optional SigLIP instead of DINOv2: add `--external_encoder siglip`.

---

## Figures

| Figure | File |
|--------|------|
| Encoder AUC / false-fire comparison | [`step4_encoder_auc.png`](step4_encoder_auc.png) |

---

## Scripts

| Script | Role |
|--------|------|
| [`tools/meeting_validation/step4/compare_encoder_routing.py`](../../../tools/meeting_validation/step4/compare_encoder_routing.py) | Main probe runner |
| [`tools/meeting_validation/step4/encoder_routing.py`](../../../tools/meeting_validation/step4/encoder_routing.py) | Encoder helpers + routing metrics |
| [`tools/meeting_validation/step4/analyze_encoder_routing.py`](../../../tools/meeting_validation/step4/analyze_encoder_routing.py) | CLI summary |
| [`tools/meeting_validation/step4/plot_encoder_auc.py`](../../../tools/meeting_validation/step4/plot_encoder_auc.py) | Figure generation |
