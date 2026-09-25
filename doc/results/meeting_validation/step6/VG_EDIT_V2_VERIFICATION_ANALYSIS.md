# VG-Edit v2 Verification Analysis (absolute + margin gate)

**Date:** 2026-07-09  
**Gate v2:** `decode_vision_abs_threshold=0.45` + `decode_vision_margin=0.0`  
**Checkpoint:** `VLKEB_long_baseline` epoch-20  
**Protocol:** seed 42, dsn=100, sen=10, sequential processes, memory-safe  

---

## Pipeline (all sequential)

| Step | Status | Result |
|------|--------|--------|
| 1. Smoke test (v2 abs gate) | ✅ | PASS |
| 2. Direction check (dsn=25, sen=5) | ✅ | PASS — hard-neg 20%→0%, rel cross 20%→0% |
| 3. L0 token eval (dsn=100, sen=10) | ✅ | 73.81% rel, 42.39% swap, 31.42 pp gap |
| 4. VG v2 token eval (same protocol) | ✅ | See table below |
| 5. Abs threshold sweep (direction, dsn=25) | ✅ | τ=0.55/0.65/0.75 all PASS routing |
| 6. Token eval abs=0.65 (dsn=100) | ✅ | Still null — see sweep table |
| 7. Patch ckpt run | ⏸ **Skipped** — token gain criterion not met |

---

## Token-level results (Step 3 vs 4)

| Metric | L0 baseline | VG v2 (abs=0.45) | Δ |
|--------|------------:|-----------------:|--:|
| Reliability | 73.81% | 73.81% | 0.00 pp |
| Image swap | 42.39% | 42.32% | −0.07 pp |
| Hard-negative | 42.84% | 42.25% | −0.59 pp |
| **Swap-gap** | **31.42 pp** | **31.49 pp** | **+0.07 pp** |
| Modality consistency | 70.97% | 69.28% | −1.69 pp |
| Locality | 100% | 100% | — |
| Edit time | 0.050 s | 0.094 s | +88% |

**Artifacts:**
- L0: `eval_results/.../VLKEB_GROUNDING-vg_v2_l0_dsn100_sen10-2026.07.09-12.39.31/`
- VG: `eval_results/.../VLKEB_GROUNDING-vg_v2_gate_dsn100_sen10-2026.07.09-12.41.32/`
- Direction: `doc/results/meeting_validation/step6/vg_edit_v2_direction_d25_s5.json`

---

## Interpretation

### v2 fixes routing but not tokens (again)

Same pattern as v1: **routing direction check passes**, but **teacher-forced token accuracy is unchanged** at dsn=100.

**Likely cause:** On swap/hard-neg probes, DINO cosine similarity between the **wrong query image** and the **stored correct edit image** is still **≥ 0.45** for many confusable pairs. The absolute threshold is too permissive — the gate never blocks the sole retrieved expert.

Step 4 offline had hard-neg own-expert score **0.008** in DINO space — but that was **routing key vs pool key** under a different protocol. End-to-end, wrong-image queries may still land above 0.45 when compared to the expert's edit-time image embedding.

### Decision

| Criterion | Target | Result |
|-----------|--------|--------|
| Direction routing PASS | Yes | ✅ |
| Swap-gap drop ≥ 5 pp | Yes | ❌ (~0 pp) |
| Reliability drop < 5 pp | Yes | ✅ |
| **Proceed to dsn=500** | — | **❌ BLOCKED** |
| **Proceed to patch ckpt** | After token gain | **⏸ Deferred** |

---

## Abs threshold sweep (Steps 5–6)

Direction check (dsn=25, sen=5) — all PASS at every τ tested:

| τ | hard_neg_fire (base→gate) | rel_cross (base→gate) | rel_miss (gate) |
|---|---------------------------|------------------------|-----------------|
| 0.45 | 20%→0% | 20%→0% | 0% |
| 0.55 | 20%→0% | 20%→0% | 0% |
| 0.65 | 20%→0% | 20%→0% | 0% |
| 0.75 | 20%→0% | 20%→0% | 0% |

Token eval comparison (dsn=100, sen=10):

| Metric | L0 | τ=0.45 | τ=0.65 |
|--------|---:|-------:|-------:|
| Reliability | 73.81% | 73.81% | 73.81% |
| Image swap | 42.39% | 42.32% | 41.94% |
| Hard-negative | 42.84% | 42.25% | 42.25% |
| **Swap-gap** | **31.42 pp** | **31.49 pp** | **31.87 pp** |
| Modality consistency | 70.97% | 69.28% | 66.32% |

**Conclusion:** Raising τ to 0.65 does not close the swap-gap (still ~31 pp). Direction routing is too easy on n=25; token leaks persist because wrong-image DINO sim to stored edit image remains above τ for most confusable pairs.

**Artifacts (τ=0.65):** `eval_results/.../VLKEB_GROUNDING-vg_v2_gate_abs065_dsn100_sen10-2026.07.09-12.46.34/`

---

## Recommended next steps (in order)

1. **Log per-probe DINO sim** on swap/hard-neg during eval — calibrate whether any τ separates correct vs wrong without killing reliability.

2. **Patch ckpt + v2 gate** — test whether patch-trained routing changes the sim distribution.

3. **Adversarial fine-tune** — if no τ works on baseline.

---

## Reproduce

```cmd
conda activate liveedit
cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"

python tools/meeting_validation/step6/smoke_test_decode_vision_gate.py

python tools/meeting_validation/step6/verify_vg_edit_direction.py ^
  -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936" ^
  -dvc cuda:0 -dsn 25 -sen 5 -seed 42 --abs-threshold 0.45

python -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt "records/.../epoch-20-i-27800-ema_loss-0.7936" -enp vg_v2_l0_dsn100_sen10 --meeting_step 6

python -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt "records/.../epoch-20-i-27800-ema_loss-0.7936" -enp vg_v2_gate_dsn100_sen10 --meeting_step 6 ^
  --decode_vision_gate --decode_vision_abs_threshold 0.45
```
