# AVBT significance (separate from DEK)

**Date:** 2026-07-16  
**JSON:** [`aaai_avbt_significance.json`](aaai_avbt_significance.json)

## Bottom line

AVBT significantly suppresses adversarial fused binding (expert_mass −0.56 on swap) and preserves reliability, but does NOT cut false-cross (~96%) without DEK. Token image-swap leakage rises +7 pp at dsn=100 — report binding metrics for AVBT, FC for DEK.

---

## 1. Binding-layer evidence (AVBT-native metric)

**Protocol:** `verify_avbl_direction.py --steps 0`, short grounding JSON, dsn=15, sen=5, seed=42 — same apples-to-apples pool for L0 and S2 ckpts.

| Binding metric | L0 | S2 (AVBT) | Δ (S2−L0) | Direction |
|----------------|---:|----------:|----------:|-----------|
| Swap expert_mass | 0.979 | 0.419 | -0.559 | **↓ better** |
| Hard-neg expert_mass | 0.992 | 0.699 | -0.293 | **↓ better** |
| Reliability own_prob | 0.999 | 0.967 | -0.033 | held |

**Headline:** swap expert_mass drops **56 pp** (0.98 → 0.42). This is what AVBL optimizes and is the right metric to showcase AVBT.

---

## 2. Token-level grounding eval (dsn=100) — use with care

**Important:** `image_swap` = fraction of probes where the model outputs the **edited answer on the wrong image**. **Lower is better** (less leakage).

| Metric | L0 | S2 (AVBT) | Δ |
|--------|---:|----------:|--:|
| Reliability | 93.5% | 93.8% | +0.3 pp |
| Image-swap leakage | 40.7% | 47.8% | +7.1 pp |
| False-cross (FC) | 96% | 96% | unchanged |
| Swap-gap (rel−swap) | 52.8 pp | 46.0 pp | -6.8 pp |

AVBT **does not** improve token leakage or FC at this scale. Binding ↓ and token leakage are **decoupled** because Q-Former can false-cross retrieve (~96%) even when fused patch binding is weaker.

Multi-seed L0 swap mean 41.2% vs S2 47.8%: *p*=4.79e-06 (L0 **lower** leakage — S2 is worse on this token probe).

---

## 3. DEK needs AVBT as foundation

| Config | FC |
|--------|---:|
| L0 | 96% |
| S2 (AVBT only) | 96% |
| S2 + DEK infer (gate only) | 53% |
| AVBT + DEK (trained) | 46% |

DEK dual-gate on the S2 checkpoint alone cuts FC substantially; full method trains patch module with AVBL first.

---

## 4. Layered story for supervisor / thesis

- LiveEdit (L0): MoE retrieval; high swap expert_mass (~0.98) on adversarial images.
- AVBT (S2): patch grounding + AVBL training → expert_mass ~0.42; reliability held; FC still ~96%.
- DEK: vision∧text gate on AVBT-trained pool → FC ~46%; primary anti-leakage mechanism.

**Motivation for AVBT:** LiveEdit fires experts strongly on adversarial images (expert_mass ≈ 1). AVBL + patch grounding learns to **suppress** that binding while keeping correct-image reliability.

**What AVBT does not solve:** Q-Former **false-cross retrieval** (wrong expert, right enough vision/text to leak tokens) — that requires **DEK**.

---

## Reproduce binding measure

```cmd
python tools/meeting_validation/step6/verify_avbl_direction.py ^
  -ckpt "records/.../VLKEB_baseline-.../epoch-20-..." ^
  -dvc cuda:0 -dsn 15 -sen 5 -seed 42 --steps 0 ^
  -dpath data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json ^
  -out doc/results/meeting_validation/step6/avbt_binding_measure_L0.json

python tools/meeting_validation/step6/verify_avbl_direction.py ^
  -ckpt "records/.../VLKEB_avbt_short_S2-.../epoch-1-i-125-..." ^
  -dvc cuda:0 -dsn 15 -sen 5 -seed 42 --steps 0 --patch_grounding ^
  -dpath data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json ^
  -out doc/results/meeting_validation/step6/avbt_binding_measure_S2.json
```
