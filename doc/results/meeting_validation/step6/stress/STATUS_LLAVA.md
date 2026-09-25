# Stress run status — LLaVA Phase 2

**Updated:** 2026-07-27  
**Plan:** [`AAAI_LLAVA_PHASE2_PLAN.md`](AAAI_LLAVA_PHASE2_PLAN.md)  
**Paper deadline:** 2026-07-28

| Gate | Status |
|------|--------|
| G-B0 | PASS |
| G-B1 L0 `dn=500`, **`eps=3`** | **DONE** (resumed after power trip; EXIT_CODE=0) |
| G-B2 S2 / DEK | **NEXT** |

## Decision (locked for submission)

| Choice | Value |
|--------|-------|
| Data | `dn=500` |
| Epochs | **`eps=3`** |
| Paper role | Directional backbone stress; disclose reduced-data L0 |
| Primary stress for 28 Jul | **EVQA multi-seed** (already done) |

## L0 run (complete)

| | |
|--|--|
| Prefix | `VLKEB_llava_L0_dn500` |
| Run dir | `VLKEB_llava_L0_dn500-2026.07.25-19.25.39` |
| Resume ckpt | `epoch-2-i-900` → finished epoch 3 |
| Log | `logs/VLKEB_llava_L0_dn500.train.log` |
| Wall time (resume segment) | ~965 min (~16 h) |

## After L0 finishes

1. Train S2 then DEK (`dn=500`, `eps=1`) from final L0 ckpt  
2. Grounding eval L0 / S2 / DEK  
3. Write `AAAI_LLAVA_STRESS_RESULTS.md` with disclosed protocol  
