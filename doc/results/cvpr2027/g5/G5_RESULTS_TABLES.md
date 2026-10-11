# G5 results tables — LLaVA lab + dual extract

**Gate:** G5 (second backbone) · **Decision due:** 2026-10-15  
**Glossary:** [`../../CVPR2027_GATE_LOG.md`](../../CVPR2027_GATE_LOG.md) § Glossary  
**Gate write-up (LLaVA-only):** [`G5_GATE_WRITEUP_LLAVA.md`](G5_GATE_WRITEUP_LLAVA.md)  
**Supervisor tracker:** [`../../AAAI_TO_CVPR_SUPERVISOR_PROGRESS.md`](../../AAAI_TO_CVPR_SUPERVISOR_PROGRESS.md) (R1 #8, R2 #5)

All % unless noted. Directions: total leak / stale leak / foreign leak / cross / emits stale answer ↓; correct route / abstain (on hard_negative) / reliability hit / only-own / emits partner answer ↑.

---

## 0. Extraction plan (what we still need)

| Phase | Work | Status | Where |
|---|---|---|---|
| **B1** | LLaVA VLKEB short grounding + leak taxonomy (L0 / S2 / DEK) | **Done** 2026-10-11 | §§1–3 below |
| **E2** | Pair-closed pool curve (LLaVA) — G3-style n∈{100,300,1000} | **Done** 2026-10-11 (`pool_size_curve_llava_s42.md`) | §7 |
| **E3** | Routing-rule controls on LLaVA (G2 matrix VLKEB) | **Done** 2026-10-11 (~16 min; `routing_table_llava_vlkeb_s42.md`) | §8 |
| **E4** | Multi-seed eval (5 seeds) on LLaVA VLKEB | **Done** 2026-10-11 (~30 min; `multiseed_llava_vlkeb_d100.md`) | §9 |
| **E1** | Easy-VQA short extract (LLaVA L0/S2/DEK) | **Done** 2026-10-11 (`leakage_taxonomy_llava_evqa.json`) | §6 |
| **C** | BLIP-2 re-extract (same protocol) | **Deferred** (after full LLaVA set) | §4 TBD |
| **B2** | Side-by-side LLaVA vs BLIP-2 dual table | After C | §5 |

**LLaVA-first:** finish E1–E4 (or as far as GPU time allows) before Phase C. G5 gate write-up still needs C eventually for dual extract.

### Commands (Phase C — BLIP-2 re-extract)

Use existing AAAI ckpts; same probe JSON and flags as LLaVA lab eval:

```text
PROBE=data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json

# L0
python -u -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 \
  -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_baseline-2026.04.09-08.20.07/checkpoints/epoch-20-i-25950-ema_loss-0.1201" \
  -enp g5_blip2_L0_short -dpath %PROBE% --meeting_step 6 --log_routing

# S2
python -u -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 \
  -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_avbt_short_S2-2026.07.10-11.16.19/checkpoints/epoch-1-i-125-ema_loss-4.0787" \
  -enp g5_blip2_S2_short -dpath %PROBE% --meeting_step 6 --patch_grounding --log_routing

# DEK
python -u -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 \
  -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_avbt_short_DEK-2026.07.10-18.13.18/checkpoints/epoch-1-i-125-ema_loss-4.0100" \
  -enp g5_blip2_DEK_short -dpath %PROBE% --meeting_step 6 --patch_grounding --dek_dual_gate --log_routing

python -u tools/score_leakage_taxonomy.py \
  --run <L0_run> --run <S2_run> --run <DEK_run> \
  --labels L0 S2 DEK \
  -out doc/results/cvpr2027/g5/leakage_taxonomy_blip2.json
```

Prefer home GPU if BLIP-2 weights + ckpts are local; else lab after copying ckpts/weights.

---

## 1. Protocol (shared)

| Item | Value |
|---|---|
| Probe JSON | `data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json` |
| `dsn` / `sen` / seed | 100 / 10 / 42 |
| Meeting step | 6 (`eval_results/meeting_validation/step6/…`) |
| Taxonomy | `tools/score_leakage_taxonomy.py` (binary + τ ∈ {0.05, 0.1, 0.2}) |
| Partner-in-pool (natural) | swap ~1%, hard-neg ~6% (same subsample for all LLaVA runs) |

### LLaVA training budget (disclose)

| Stage | Backbone | `dn` | Epochs | Ckpt |
|---|---|---|---|---|
| L0 | LLaVA-1.5-7B | 500 | **20** | `…/epoch-20-i-10200-ema_loss-2.2061` |
| S2 | LLaVA | 500 | 1 | `…/epoch-1-i-500-ema_loss-3.8904` |
| DEK | LLaVA | 500 | 1 | `…/epoch-1-i-500-ema_loss-3.8851` |

BLIP-2 reference L0 is full-set ×20; S2/DEK are short `dn=500`×1 (AAAI ckpts). Do not claim matched train compute across backbones.

---

## 2. LLaVA — leak taxonomy (binary counting) ✓

Source: `leakage_taxonomy_llava.json` (lab, 2026-10-11).

### 2.1 Headline (binary)

| Method | Swap total leak ↓ | Swap stale ↓ | Swap foreign ↓ | Correct route ↑ | Abstain | Hard-neg total leak ↓ | Rel hit ↑ | Only-own ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 | 100.0 | 24.0 | 76.0 | 0.0 | 0.0 | 98.0 | 79.0 | 0.0 |
| S2 | 100.0 | 41.0 | 59.0 | 0.0 | 0.0 | 100.0 | 98.0 | 0.0 |
| **DEK** | **36.0** | **1.0** | 35.0 | 0.0 | **64.0** | **46.0** | **1.0** | 1.0 |

**Readout:** DEK cuts swap total leak −64 pp and hard-neg total leak −52 pp vs L0, almost entirely via abstain; reliability hit collapses (79% → 1%). This is **not** the BLIP-2 DEK pattern (high rel + lower leak). Flag for G5 Result / gate write-up.

### 2.2 Image-swap four-way (binary)

| Method | stale_leak ↓ | foreign_leak ↓ | total_leak ↓ | correct_route ↑ | abstain |
|---|---:|---:|---:|---:|---:|
| L0 | 24.0 | 76.0 | 100.0 | 0.0 | 0.0 |
| S2 | 41.0 | 59.0 | 100.0 | 0.0 | 0.0 |
| DEK | 1.0 | 35.0 | 36.0 | 0.0 | 64.0 |

### 2.3 Hard-negative (binary)

| Method | stale_leak ↓ | foreign_leak ↓ | total_leak ↓ | abstain |
|---|---:|---:|---:|---:|
| L0 | 24.0 | 74.0 | 98.0 | 2.0 |
| S2 | 47.0 | 53.0 | 100.0 | 0.0 |
| DEK | 1.0 | 45.0 | 46.0 | 54.0 |

### 2.4 Soft-weight side check (vision-logit proxy; not fuse_coe)

| Method | Variant | Swap total leak ↓ | Hard-neg total leak ↓ | Rel hit ↑ |
|---|---|---:|---:|---:|
| L0 | τ=0.05 | 44.0 | 57.0 | 12.0 |
| L0 | τ=0.1 | 13.0 | 38.0 | 2.0 |
| L0 | τ=0.2 | 2.0 | 5.0 | 0.0 |
| S2 | τ=0.05 | 35.0 | 14.0 | 11.0 |
| S2 | τ=0.1 | 34.0 | 7.0 | 5.0 |
| S2 | τ=0.2 | 15.0 | 3.0 | 2.0 |
| DEK | τ=0.05 / 0.1 / 0.2 | 36.0 | 46.0 | 1.0 |

DEK binary ≈ all τ rows (gate already sparse). Do not put proxy τ numbers in the paper as fuse coefficients (M3).

---

## 3. LLaVA — token / answer-level metrics ✓

Source: `lab_mirror/eval_results/.../g5_llava_*/sequential_edit_10/seed_42_mean_results.json`.

| Method | Rel acc ↑ | Rel EM ↑ | Image-swap acc | emits stale (`emits_y_i`) ↓ | emits partner (`emits_y_j`) ↑ | Hard-neg acc | Text loc ↑ | Image loc ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 | 0.5115 | 0.01 | 0.4742 | 0.00 | 0.00 | 0.4747 | 1.00 | 0.9859 |
| S2 | 0.5023 | 0.01 | 0.4749 | 0.01 | 0.00 | 0.4682 | 1.00 | 0.9877 |
| DEK | 0.4876 | 0.01 | 0.4697 | 0.01 | 0.00 | 0.4739 | 1.00 | 0.9814 |

Answer-level swap rescue is **not** claimed (partner answers ~0; Rel EM ~1%).

Legacy routing false-activation (retired for paper; kept for audit): L0 swap/hn 100%/98%; S2 100%/100%; DEK **35%/45%**.

---

## 4. BLIP-2 re-extract (same protocol) — TBD

Fill after Phase C. Rows reserved for dual table:

| Method | Swap total leak ↓ | Hard-neg total leak ↓ | Rel hit ↑ | Only-own ↑ | emits_y_i ↓ | emits_y_j ↑ |
|---|---:|---:|---:|---:|---:|---:|
| L0 | — | — | — | — | — | — |
| S2 | — | — | — | — | — | — |
| DEK | — | — | — | — | — | — |

Artifact target: `leakage_taxonomy_blip2.json` + `g5_blip2_*_short` eval dirs.

*(Existing G0 VLKEB L0/DEK binary for reference only — not a substitute for Phase C: swap TL 88→20, hn 91→52, rel hit 100 / only-own 16→100. See `../g0/leakage_taxonomy_vlkeb_s42.md`.)*

---

## 5. Dual extract side-by-side — TBD after §4

| Backbone | Method | Train note | Swap TL ↓ | Hn TL ↓ | Rel hit ↑ | Only-own ↑ |
|---|---|---|---:|---:|---:|---:|
| LLaVA | L0 | short-n ×20 | 100.0 | 98.0 | 79.0 | 0.0 |
| LLaVA | DEK | short-n L0 + AVBT×1 | 36.0 | 46.0 | 1.0 | 1.0 |
| BLIP-2 | L0 | full-set ×20 | — | — | — | — |
| BLIP-2 | DEK | short S2/DEK ×1 | — | — | — | — |

**G5 question:** Does DEK reduce total leak on a second backbone without collapsing reliability? Current LLaVA answer: leak ↓ yes, reliability hit ↓ collapse — provisional **WEAK / investigate** until BLIP-2 re-extract confirms the extractor and we decide gate wording.

---

## 6. Easy-VQA (LLaVA) ✓ (E1)

Source: `leakage_taxonomy_llava_evqa.json` (lab, 2026-10-11). Protocol: `dsn=100`, `sen=10`, seed=42 (match B1). Partner-in-pool: swap 83%, hard-neg 3%.

### 6.1 Headline (binary)

| Method | Swap total leak ↓ | Swap stale ↓ | Swap foreign ↓ | Correct route ↑ | Abstain | Hard-neg total leak ↓ | Rel hit ↑ | Only-own ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 | 100.0 | 43.0 | 57.0 | 0.0 | 0.0 | 100.0 | 71.0 | 0.0 |
| S2 | 100.0 | 58.0 | 42.0 | 0.0 | 0.0 | 100.0 | 96.0 | 1.0 |
| **DEK** | **13.0** | **0.0** | 13.0 | 0.0 | **87.0** | **91.0** | **1.0** | 1.0 |

**Readout:** On Easy-VQA, DEK cuts swap total leak −87 pp (mostly abstain) but hard-neg total leak barely moves (−9 pp) and Rel hit collapses (71% → 1%) — same Rel pathology as VLKEB B1.

---

## 7. Pair-closed pool curve (LLaVA) ✓ (E2)

Source: `pool_size_curve_llava_s42.md` (lab, 2026-10-11). Partners in pool = 100% on all runs. n≥300 used `--routing_only`.

| target_n | Method | Hn TL ↓ | Swap TL ↓ | Correct route ↑ | Abstain | Rel hit ↑ |
|---|---|---:|---:|---:|---:|---:|
| 100 | L0 | 100.0 | 100.0 | 0.0 | 0.0 | 77.0 |
| 100 | DEK | **31.0** | **20.0** | 0.0 | 80.0 | **0.0** |
| 300 | L0 | 99.3 | 100.0 | 0.0 | 0.0 | 77.5 |
| 300 | DEK | **38.7** | **42.1** | 0.3 | 57.6 | **0.3** |
| 1000 | L0 | 99.8 | 100.0 | 0.0 | 0.0 | 81.5 |
| 1000 | DEK | **39.0** | **33.7** | 0.1 | 66.2 | **0.2** |

**Readout:** DEK keeps a large leak advantage vs L0 as pool grows (swap TL ~20–42% vs 100%; hn ~31–39% vs ~100%), but reliability hit stays collapsed (~0%). Matches the natural-pool B1 pattern.

---

## 8. Routing-rule controls (LLaVA) ✓ (E3)

Source: `routing_table_llava_vlkeb_s42.md` (lab, 2026-10-11). VLKEB `dsn=100`, `--routing_only`.

| Run | Rule | Swap TL ↓ | Correct route ↑ | Abstain | Hn TL ↓ | Rel hit ↑ | Only-own ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| L0 | liveedit | 100.0 | 0.0 | 0.0 | 98.0 | 79.0 | 0.0 |
| L0 | vision_only | 100.0 | 0.0 | 0.0 | 98.0 | 79.0 | 0.0 |
| L0 | text_only | 100.0 | 0.0 | 0.0 | 100.0 | 2.0 | 2.0 |
| L0 | and | **2.0** | 0.0 | 98.0 | **4.0** | **1.0** | 1.0 |
| L0 | or | 100.0 | 0.0 | 0.0 | 100.0 | 79.0 | 0.0 |
| L0 | sum | 100.0 | 0.0 | 0.0 | 98.0 | 79.0 | 0.0 |
| L0 | carml | 73.0 | 0.0 | 27.0 | 96.0 | 41.0 | 41.0 |
| L0 | always_reject | **0.0** | 0.0 | 100.0 | **0.0** | **0.0** | 0.0 |
| DEK | and | **36.0** | 0.0 | 64.0 | **46.0** | **1.0** | 1.0 |
| DEK | liveedit | 100.0 | 0.0 | 0.0 | 100.0 | 98.0 | 0.0 |

**Readout:** Same control pattern as BLIP-2 G2: `and` / `always_reject` kill leak by collapsing Rel; `carml` mid; DEK+`and` matches B1 DEK numbers; DEK+`liveedit` reverts to saturated leak.

---

## 9. Multi-seed ✓ (E4)

Source: `multiseed_llava_vlkeb_d100.md` (lab, 2026-10-11). Seeds `{42,0,1,2,3}`, `dsn=100`, `--routing_only`.

| Method | Swap TL mean±std ↓ | Hn TL mean±std ↓ | Rel hit mean±std ↑ | Only-own mean±std ↑ |
|---|---:|---:|---:|---:|
| L0 | 100.0 ± 0.0 | 99.6 ± 0.9 | 76.6 ± 5.4 | 0.0 ± 0.0 |
| S2 | 100.0 ± 0.0 | 100.0 ± 0.0 | 98.8 ± 0.8 | 0.2 ± 0.4 |
| **DEK** | **35.0 ± 7.8** | **41.2 ± 10.2** | **0.6 ± 0.5** | 0.6 ± 0.5 |

**Readout:** DEK leak win is stable across seeds; Rel hit collapse is also stable (~0–1%). Matches seed-42 B1.

---

## Artifacts

| File | Role |
|---|---|
| `leakage_taxonomy_llava.json` | LLaVA taxonomy (filled) |
| `logs/eval_g5_llava_{L0,S2,DEK}_short.log` | Eval stdout |
| `lab_mirror/eval_results/.../g5_llava_*` | mean results + routing_diagnostics |
| `leakage_taxonomy_blip2.json` | Phase C target |
| `EXPERIMENT_TRACKER.md` | Run timeline |

Run dirs (lab):

- L0: `eval_results/.../VLKEB_GROUNDING-g5_llava_L0_short-2026.10.11-12.10.05`
- S2: `.../g5_llava_S2_short-2026.10.11-12.14.11`
- DEK: `.../g5_llava_DEK_short-2026.10.11-12.22.23`
