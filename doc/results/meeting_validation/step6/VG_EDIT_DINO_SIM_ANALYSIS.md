# VG-Edit DINO Sim Logging Analysis

**Date:** 2026-07-09  
**Checkpoint:** `VLKEB_long_baseline` epoch-20  
**Protocol:** seed 42, dsn=100, full sequential pool (100 edits), Q-Former routing only at probe time  

**Artifacts:**
- Log: `doc/results/meeting_validation/step6/vg_edit_dino_sims_d100_s10.json`
- Analysis: `doc/results/meeting_validation/step6/vg_edit_dino_sims_d100_s10_analysis.json`
- Plot: `doc/results/meeting_validation/step6/vg_edit_dino_sims_hist.png`

---

## Why logging was needed

VG-Edit v2 (abs τ + margin) fixed **routing direction** on small probes but left **token swap-gap ~31 pp** unchanged. Threshold sweeps 0.45–0.75 did not help. We needed to know whether any abs τ separates correct vs adversarial probes.

---

## Main finding: leak is false-cross, not own-expert false-fire

| Routing metric | image_swap | hard_negative | reliability |
|----------------|----------:|-------------:|------------:|
| **false_cross** (wrong expert retrieved) | **88%** | **90%** | low |
| false_fire_own (own expert retrieved) | **2%** | **2%** | — |
| reliability_miss | — | — | **0%** |

**Token swap acc ~42%** aligns with **wrong-expert retrieval (88%)**, not with own-expert false-fire (2%).

The current gate filters **among Q-Former candidates** using abs τ on `sim(query, expert_edit_img)`. That cannot fix the dominant failure mode when **wrong experts** pass with high DINO sim.

---

## DINO separability (query vs stored edit images)

### sim(query, **own** expert edit img)

| Probe | mean | p50 | p10–p90 |
|-------|-----:|----:|---------|
| reliability | **1.00** | 1.00 | ~1.00 |
| image_swap | **0.24** | 0.22 | 0.01–0.48 |
| hard_negative | **0.24** | 0.22 | ~0–0.50 |

Own-expert abs τ **does** separate reliability from swap/hard-neg **when own expert is in play** — but own expert is rarely Q-Former-selected on adversarial probes (2%).

### sim(query, **best wrong** expert edit img in pool)

| Probe | mean | p50 |
|-------|-----:|----:|
| image_swap | **0.90** | **1.00** |
| hard_negative | **0.89** | **1.00** |

On swap, the query image often matches **some other expert's stored edit image** with sim ≈ 1.0.

### Margin: own − best_wrong (full pool)

| Probe | mean margin |
|-------|------------:|
| reliability | **+0.35** |
| image_swap | **−0.67** |
| hard_negative | **−0.65** |

On adversarial probes, **wrong experts win** the global DINO competition.

### Retrieved wrong experts (Q-Former candidates only)

| Probe | mean sim of retrieved wrong expert |
|-------|-----------------------------------:|
| image_swap (n=88 fires) | **0.93** |
| hard_negative (n=90 fires) | **0.91** |

Wrong experts that Q-Former selects pass abs τ=0.45–0.75 trivially.

---

## Implication for VG-Edit v2

| Approach | Verdict |
|----------|---------|
| Raise abs τ on own expert | ❌ Irrelevant — own expert rarely fires on swap (2%) |
| abs τ on wrong experts | ❌ Wrong experts sim ≈ 0.93–1.0 — no τ blocks them without blocking reliability |
| Margin among Q-Former candidates only | ❌ Wrong expert often sole or winning candidate |
| Global DINO winner-take-all | ❌ Wrong expert **wins** on swap (margin −0.67) |

**Root cause:** Under lifelong pooling, swap/hard-neg images are visually aligned with **other experts' edit images**. Q-Former co-retrieves those experts; DINO confirms them. Inference-only abs/margin on candidate set cannot disambiguate without **prompt–image–expert binding** beyond scalar similarity.

---

## Recommended next directions

1. **Gate v3 — bind to prompt-selected expert only:** After Q-Former shortlist, keep expert `i` only if `sim(query_img, expert_i_img)` exceeds τ **and** `sim(query, expert_i) ≥ sim(query, expert_j) + m` for all other shortlisted **and** Q-Former text score ranks `i` first (multi-signal).

2. **Pool hygiene / dedup:** Detect near-duplicate edit images in pool at edit time; suppress experts with DINO sim > 0.95 to existing pool entries.

3. **Adversarial patch training:** Train routing to **downweight** experts when query is swap/hard-neg of their source sample (Step 4 offline showed DINO can do this — need end-to-end).

4. **Do not scale dsn=500** until a gate variant shows ≥5 pp swap-gap drop at dsn=100.

---

## Reproduce

```cmd
conda activate liveedit
cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"

python tools/meeting_validation/step6/log_vg_edit_dino_sims.py ^
  -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936" ^
  -dvc cuda:0 -dsn 100 -seed 42 -sen 10 ^
  -out doc/results/meeting_validation/step6/vg_edit_dino_sims_d100_s10.json

python tools/meeting_validation/step6/analyze_vg_edit_dino_sims.py ^
  doc/results/meeting_validation/step6/vg_edit_dino_sims_d100_s10.json
```
