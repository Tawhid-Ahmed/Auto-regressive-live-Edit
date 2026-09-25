# VG-Edit Verification Analysis

**Date:** 2026-07-09  
**Protocol:** Sequential runs only (no parallel jobs); memory hygiene applied  
**Checkpoint:** `VLKEB_long_baseline` epoch-20 (`ema_loss-0.7936`)  
**Seed:** 42  

---

## Execution summary (sequential)

| Step | Command / script | Runtime | GPU memory notes |
|------|------------------|---------|------------------|
| 1 | `smoke_test_decode_vision_gate.py` | <1 min | No GPU |
| 2 | `verify_vg_edit_direction.py` (dsn=25, sen=5) | ~19 s | Single BLIP2 load; pool=5 |
| 3 | `test_grounding_eval.py` L0 (dsn=100, sen=10) | ~70 s | Separate process; `before_edit_ids` → CPU |
| 4 | `test_grounding_eval.py` VG gate (dsn=100, sen=10) | ~97 s | Fresh process after L0 exited |

**Memory changes applied:**
- `before_edit_ids` stored on CPU in `evaluation/vllm_editor_eval.py`
- `torch.cuda.empty_cache()` after each sequential-edit split
- Query patch cache cleared after `retrieve_moes()`
- Each eval step = **new Python process** (full GPU release between L0 and VG)

---

## Step 2 — Routing direction check (PASS)

**Config:** dsn=25, sequential pool sen=5, margin=0.0  
**Artifact:** `doc/results/meeting_validation/step6/vg_edit_direction_check_d25_s5.json`

| Metric | Baseline | VG-Edit | Δ |
|--------|----------|---------|---|
| Hard-neg false-fire | **20.0%** | **0.0%** | −20 pp |
| Swap false-fire | 0.0% | 0.0% | — |
| Rel false-cross | **20.0%** | **0.0%** | −20 pp |
| Rel miss | 0.0% | 0.0% | 0 pp |

**Verdict:** Direction check **PASSED** — routing-level adversarial false-fire dropped ≥10 pp with no reliability miss increase.

---

## Steps 3–4 — End-to-end token accuracy (NO GAIN)

**Config:** dsn=100, sen=10, long-form grounding JSON  
**L0:** `eval_results/.../VLKEB_GROUNDING-vg_verify_l0_dsn100_sen10-2026.07.09-10.47.54/`  
**VG:** `eval_results/.../VLKEB_GROUNDING-vg_verify_gate_dsn100_sen10-2026.07.09-10.49.27/`

| Metric | L0 baseline | VG-Edit gate | Δ (VG−L0) |
|--------|------------:|-------------:|----------:|
| **Reliability** | 73.81% | 73.81% | **0.00 pp** |
| **Image swap** | 42.39% | 42.39% | **0.00 pp** |
| **Hard-negative** | 42.84% | 42.83% | −0.01 pp |
| **Swap-gap** | 31.42 pp | 31.42 pp | **0.00 pp** |
| Modality consistency | 70.97% | 70.28% | −0.69 pp |
| Locality | 100% | 100% | — |
| Edit time / sample | 0.054 s | 0.095 s | +77% (DINO encode) |

Position curves and per-split means are **bitwise identical** on reliability/swap/hard-neg — the gate did not change teacher-forced token predictions at this scale.

---

## Interpretation

### What worked
- **Decode gate changes routing** when multiple experts co-activate (direction check: 20% → 0% hard-neg false-fire on n=25 pool).
- Implementation is stable at dsn=100 without OOM when runs are sequential and splits clear cache.

### What did not work (yet)
- **Token-level grounding unchanged** — swap-gap stays ~31 pp, same as Step 1 L0 (~31 pp at dsn=500).
- **Root cause:** Current gate is **competitive among Q-Former-passing candidates only**. On wrong-image probes, often **a single expert** passes `vis_sim > vis_sim_prot` (the prompt-matched edit). With one candidate, competitive margin always passes → MoE residual still injects → ~43% wrong-image token overlap unchanged.

This matches the Step 4 → Step 6 lesson: **routing discrimination ≠ generation gating** when the wrong expert is the *only* retrieved expert.

### Required gate v2 (before full-scale dsn=500)
Block injection unless **direct image match** holds:

```
sim_dino(query_image, expert_edit_image) >= τ_abs
AND sim_dino(query, own) >= sim_dino(query, best_other) + τ_margin
```

If `sim_dino(query, own) < τ_abs` → **zero residual** even when alone in candidate set.

---

## Decision gate for next work

| Criterion | Target | Result |
|-----------|--------|--------|
| Direction routing check | PASS | ✅ |
| dsn=100 swap-gap drop | ≥5 pp | ❌ (0 pp) |
| Reliability cost | <5 pp | ✅ (0 pp) |
| **Proceed to dsn=500 multi-seed** | Only if token gain | **❌ BLOCKED** |
| **Proceed to gate v2 (absolute + margin)** | Required | ✅ **Next** |

---

## Reproduce (sequential, memory-safe)

```cmd
cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"
conda activate liveedit

python tools/meeting_validation/step6/smoke_test_decode_vision_gate.py

python tools/meeting_validation/step6/verify_vg_edit_direction.py ^
  -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936" ^
  -dvc cuda:0 -dsn 25 -sen 5 -seed 42

python -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt "records/.../epoch-20-i-27800-ema_loss-0.7936" -enp vg_verify_l0_dsn100_sen10 --meeting_step 6

python -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt "records/.../epoch-20-i-27800-ema_loss-0.7936" -enp vg_verify_gate_dsn100_sen10 --meeting_step 6 ^
  --decode_vision_gate
```

Or orchestrator: `python tools/meeting_validation/step6/run_vg_edit_verification_sequence.py -dvc cuda:0`

---

## Artifacts

| File | Description |
|------|-------------|
| `vg_edit_direction_check_d25_s5.json` | Routing direction check |
| `VG_EDIT_VERIFICATION_ANALYSIS.md` | This document |
| L0 mean results | `.../vg_verify_l0_dsn100_sen10-.../seed_42_mean_results.json` |
| VG mean results | `.../vg_verify_gate_dsn100_sen10-.../seed_42_mean_results.json` |
