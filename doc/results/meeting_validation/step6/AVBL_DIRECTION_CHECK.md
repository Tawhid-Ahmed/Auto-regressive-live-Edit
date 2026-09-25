# AVBL Direction Check (fast validation before full train)



**Date:** 2026-07-09  

**Goal:** Validate Adversarial Visual Binding Loss (AVBL) can move routing **before** multi-epoch training.



---



## What was implemented



| Component | Path |

|-----------|------|

| AVBL core (`compute_avbl_*`, fused binding dist) | `editor/vllm_editors/liveedit/liveedit.py` |

| Smoke tests (no GPU) | `tools/meeting_validation/step6/smoke_test_avbl.py` |

| Fast direction check (~1–3 min GPU) | `tools/meeting_validation/step6/verify_avbl_direction.py` |

| Train integration | `train_vllm_editor.py` + `_train_a_batch_impl` (when `--adversarial_grounding`) |



---



## Pass criteria (direction check)



- Swap or hard-neg **fused expert_mass** drops ≥ **0.10**, OR false-cross drops ≥ **10 pp**

- Reliability **own_prob** drop ≤ **0.10**



---



## GPU load fix (why paging errors appeared)



Recent session changes introduced **higher host-memory spikes** during BLIP2 load:



| Issue | Effect |

|-------|--------|

| **CPU fp32 fallback** after `device_map` failure | Loads full ~11 GB model on CPU, then `.to(GPU)` — needs ~2× commit |

| **`device_map` hard-crash** on some Windows stacks | Process dies or falls back to heavy CPU path |

| **DINO loaded at `enable_patch_grounding()` init** | Extra model on GPU right after BLIP2 (fixed: use `FrozenPatchEncoder.DINOV2_DIM` constant) |



**Fix in `blip2.py`:** default **staged fp16 load** on GPU:

1. `low_cpu_mem_usage=True` (stream shards)

2. `.half()` on CPU

3. `.to(cuda:0)`



Opt-in fast path: `set LIVEEDIT_BLIP2_DEVICE_MAP=1` to try direct `device_map`.



---



## Direction check result — **PASS** (GPU, dsn=10, sen=3, steps=15)



**Output:** `doc/results/meeting_validation/step6/avbl_direction_check.json`



| Metric | Before | After | Delta |

|--------|-------:|------:|------:|

| swap expert_mass | 0.793 | 0.0003 | **−0.793** |

| hard-neg expert_mass | 0.816 | 0.100 | **−0.715** |

| reliability own_prob | 0.711 | 0.779 | +0.069 (within bound) |

| swap false_cross | 0% | 0% | — |

| hard-neg false_cross | 0% | 0% | — |



**Verdict:** AVBL pushes swap/hard-neg binding away from experts while reliability stays stable. **Proceed to short train + dsn=100 eval.**



---



## Run commands



```cmd

conda activate liveedit

cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"



python tools/meeting_validation/step6/smoke_test_avbl.py



python tools/meeting_validation/step6/verify_avbl_direction.py ^

  -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936" ^

  -dvc cuda:0 -dsn 10 -sen 3 -seed 42 --steps 15 --lr 1e-4

```



---



## Full train (after PASS)



```cmd

python train_vllm_editor.py -en liveedit -mn blip2 -dna VLKEB -bs 4 -dvc cuda:0 ^

  -lkpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936" ^

  -eps 3 -tnp VLKEB_long_avbl ^

  -dpath data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json ^

  --patch_grounding --adversarial_grounding --adv_grounding_lambda 1.0

```



Then eval: `test_grounding_eval.py` dsn=100 vs L0.



---



## Decision tree

```
smoke_test_avbl.py PASS?  ✅
  └─ verify_avbl_direction.py PASS?  ✅ (GPU)
       └─ short train + dsn=100 eval  ✅ DONE
```

---

## Short train + dsn=100 eval (2026-07-09)

**Train run:** `VLKEB_long_avbl-2026.07.09-16.28.14`  
**Resume:** epoch-1-i-125 → completed epochs 2–3  
**Final ckpt:** `epoch-3-i-375-ema_loss-8.1046`  
**Protocol:** seed 42, dsn=100, sen=10

| Metric | L0 | AVBL epoch-3 | Delta |
|--------|---:|-------------:|------:|
| Reliability | 74.00% | 48.58% | -25.4 pp |
| Swap acc | 42.42% | 42.13% | -0.3 pp |
| Hard-neg acc | 42.81% | 42.29% | -0.5 pp |
| **Swap-gap** | **31.58 pp** | **6.45 pp** | **-25.1 pp** |

**Verdict:** Swap-gap drops, but reliability collapses — not ready for dsn=500. Tune `adv_grounding_lambda` down and add reliability floor before retrain.

**Artifacts:** `doc/results/meeting_validation/step6/avbl_short_train_summary.json`

---

## Tune experiments (2026-07-09, sequential)

| Run | Reliability | Swap acc | Swap-gap | Notes |
|-----|------------:|---------:|---------:|-------|
| L0 | 74.00% | 42.42% | 31.58 pp | baseline |
| AVBL λ=1.0 epoch-2 | **79.46%** | **53.15%** | 26.31 pp | best token tradeoff so far |
| AVBL λ=1.0 epoch-3 | 48.58% | 42.13% | 6.45 pp | reliability collapse |
| AVBL λ=0.2, neg_w=2.0 (1 ep) | 78.02% | 50.60% | 27.42 pp | calibrated retrain |

**Takeaway:** Early stop (epoch-2) or lower λ preserves reliability **and** raises swap/hard-neg acc (~+8–11 pp). Do not train 3 epochs at λ=1.0.

**Code added:** `--adv_grounding_neg_weight`, `--avbl_hinge_tau`, `--avbl_contrastive`, `--log_routing` on eval.

**Summary:** `doc/results/meeting_validation/step6/avbl_tune_experiments_summary.json`

