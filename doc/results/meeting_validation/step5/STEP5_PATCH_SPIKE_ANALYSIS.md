# Step 5 — Patch + Cross-Attention Inference Spikes

**Date:** 2026-07-05  
**Protocol:** seed 42, n=50 sequential edits, Step 1 grounding JSON  
**Checkpoint (4a/4b/baseline):** `records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936`  
**Checkpoint (4c AR):** `records/liveedit/blip2-opt-2.7b/VLKEB_long_ar-2026.06.04-13.00.34/checkpoints/epoch-20-i-26600-ema_loss-1.5124`  
**GPU:** cuda:0 · ~5 min per spike (model load + 50 sequential edits + routing probes)

**Raw results:** [`patch_spike_comparison.json`](patch_spike_comparison.json) · [`patch_spike_baseline.json`](patch_spike_baseline.json) · [`patch_spike_4a.json`](patch_spike_4a.json) · [`patch_spike_4a_attn.json`](patch_spike_4a_attn.json) · [`patch_spike_4b.json`](patch_spike_4b.json) · [`patch_spike_4c.json`](patch_spike_4c.json)

---

## Summary

| Spike | Change | Hard-neg false-fire | Rel false-cross | Rel miss |
|-------|--------|--------------------:|----------------:|---------:|
| **Baseline** | `vis_sim.mean(2)` | **42.0%** | **76.0%** | 0.0% |
| **4a max** | max-pool over eqe_n | 50.0% | 100.0% | 0.0% |
| **4a attn** | attn-weighted pool | 50.0% | 100.0% | 0.0% |
| **4b** | max-pool + ViT patch cross-attn rerank | 50.0% | 100.0% | 0.0% |
| **4c** | Q-Former vision-gated AR chunk gate | **38.0%** | **62.0%** | 0.0% |

**Spike definitions (inference-only, no retrain):**

| ID | Code flags | Mechanism |
|----|------------|-----------|
| **4a** | `routing_pool_mode=max\|attn` | Replace scalar collapse from `mean(2)` in `retrieve_moes` |
| **4b** | `patch_rerank_enabled` + `expose_vit_grid` | Top-k coarse experts reranked with patch cross-attention |
| **4c** | `ar_vision_gated_chunks` + AR ckpt | Chunk index from Q-Former image cross-attention peaks |

---

## Interpretation

1. **4a (no mean-pool) does not help — it hurts.** Max and attention-weighted pooling raise hard-negative false-fire from 42% to 50% and push reliability false-cross to 100% (every reliability probe co-retrieves a wrong-sample expert). Loosening the scalar routing key without a trained spatial signal makes retrieval more permissive, not more selective.

2. **4b (patch rerank) does not recover 4a’s damage.** ViT patch cross-attention reranking on top of max-pool matches 4a’s failure profile (50% hard-neg false-fire, 100% rel false-cross). Untrained patch matching at inference cannot disambiguate confusable entities when coarse Q-Former keys already co-fire.

3. **4c (vision-gated chunks) is the only inference spike that beats baseline.** On the AR checkpoint with vision-gated chunk routing, hard-negative false-fire drops to 38% (−4 pp) and reliability false-cross falls to 62% (−14 pp). Aligning the AR routing gate with Q-Former cross-attention peaks partially reduces wrong-expert activation without any weight updates.

4. **None of the spikes solve grounding.** Even 4c leaves 38% hard-negative false-fire and 62% reliability co-retrieval — consistent with Step 1’s ~43% adversarial token accuracy. Inference-only routing tweaks are insufficient; a **trained patch-grounding layer (Step 6) with DINOv2 routing keys (Step 4)** remains necessary for thesis-grade grounding.

---

## Gate

**PASS → Proceed to Step 6** (with reframing: trained patch layer required; 4c informs AR chunk design).

| Criterion | Result |
|-----------|--------|
| 4a alone fixes hard negatives | ❌ Worse than baseline (50% vs 42%) |
| 4b needed vs 4a | ❌ 4b matches 4a failure; no rerank benefit |
| 4c beats baseline on hard negatives | ✅ 38% vs 42% (−4 pp) |
| Winning spike for Step 6 | **4c vision-gated chunks** (AR) + **DINOv2 patch keys** (offline Step 4 winner) |
| Proceed to Step 6 retrain | **Yes** — inference spikes validate that spatial/chunk alignment helps marginally but retrain is mandatory |

**Proceed to Step 6.**

---

## Reproduce

```powershell
cd "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"

conda activate liveedit

# Run each spike separately if GPU RAM is tight (recommended on Windows):
python tools/meeting_validation/step5/run_patch_spikes.py -dvc cuda:0 -dsn 50 -seed 42 --spikes baseline
python tools/meeting_validation/step5/run_patch_spikes.py -dvc cuda:0 -dsn 50 -seed 42 --spikes 4a_max
python tools/meeting_validation/step5/run_patch_spikes.py -dvc cuda:0 -dsn 50 -seed 42 --spikes 4a_attn
python tools/meeting_validation/step5/run_patch_spikes.py -dvc cuda:0 -dsn 50 -seed 42 --spikes 4b_rerank
python tools/meeting_validation/step5/run_patch_spikes.py -dvc cuda:0 -dsn 50 -seed 42 --spikes 4c_vision_gate

# Merge + report (after all per-spike JSONs exist):
python tools/meeting_validation/step5/merge_patch_spikes.py
python tools/meeting_validation/step5/analyze_patch_spikes.py
python tools/meeting_validation/step5/plot_patch_spikes.py
```

Or all spikes in one process (needs ~12 GB VRAM headroom for two ckpt loads):

```powershell
python tools/meeting_validation/step5/run_patch_spikes.py -dvc cuda:0 -dsn 50 -seed 42
```

---

## Figures

| Figure | File |
|--------|------|
| Spike comparison | [`step5_spike_comparison.png`](step5_spike_comparison.png) |

---

## Code touchpoints

- `editor/vllm_editors/liveedit/liveedit.py` — `_pool_vis_sim`, `_rerank_with_patch_cross_attn`, opt-in flags
- `editor/vllms_for_edit/blip2/blip2.py` — `get_vit_patch_embeds`, `get_qformer_cross_attention`, `expose_vit_grid`
- `editor/vllm_editors/liveedit/chunk_utils.py` — `vision_gated_chunk_index`, `infer_ar_chunk_index`
- `tools/meeting_validation/step5/` — run, merge, analyze, plot
