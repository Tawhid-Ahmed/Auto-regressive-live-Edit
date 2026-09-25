# AAAI Phase D — Short pool-buffer gate

**Date:** 2026-07-13  
**Protocol:** `dsn=100`, `sen=10`, `seed=42`,  
`data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json`,  
`--meeting_step 6`, `--patch_grounding`, `--dek_dual_gate`, `--log_routing`

**Gate (short-first):** image-swap ≥50% **and** false-cross ≤40%.  
Only if PASS → train/eval same stack on long.

| Short | Full name |
|-------|-----------|
| L0 | LiveEdit baseline |
| S2 | AVBT (no dual-gate) |
| DEK | AVBT + Dual-Expert Keys |
| FX | False-cross mining (neighbor + refuse) |
| PB | Pool buffer |

---

## Checkpoint

| Role | Path |
|------|------|
| Init | short LiveEdit baseline (L0) `VLKEB_baseline-2026.04.09-08.20.07/.../epoch-20-i-25950-ema_loss-0.1201` |
| Pool buffer (PB) | `records/.../VLKEB_avbt_short_PB-2026.07.13-10.45.41/.../epoch-1-i-125-ema_loss-4.5631` |

**Train:** Dual-Expert Keys + AVBL (λ=0.2) + `dino_neighbor` + refuse 0.2 + **`--avbl_pool_buffer 64`**,  
`dn=500`, `eps=1`, `grounding_train_short.json`.

---

## Results vs prior short levers

| Method | Rel ↑ | Swap ↑ | Hard ↑ | FC ↓ |
|--------|------:|-------:|-------:|-----:|
| LiveEdit baseline (L0) short | 0.935 | 0.407 | — | 0.96 |
| AVBT (S2) | 0.938 | 0.478 | — | 0.96 |
| AVBT + Dual-Expert Keys (DEK) | 0.937 | 0.477 | — | **0.46** |
| False-cross mining (FX) | — | 0.472 | — | 0.43 |
| **Pool buffer (PB)** | **0.937** | **0.480** | 0.436 | **0.55** |

**Eval dir:** `eval_results/.../VLKEB_GROUNDING-aaai_PB_short-2026.07.13-11.07.15`

---

## Gate check

| Criterion | Need | Got | Status |
|-----------|------|-----|--------|
| Image-swap ≥50% | ≥0.50 | 0.480 | **FAIL** |
| False-cross ≤40% | ≤0.40 | 0.55 | **FAIL** |
| Reliability preserved | ~Dual-Expert Keys (DEK) | 0.937 | PASS |

**Overall: FAIL.**

---

## Takeaway

1. Pool-buffer mining does **not** clear the short swap/FC gate.
2. Swap is flat vs Dual-Expert Keys / False-cross mining (+0.3 pp at most); FC **regresses** (46% → 55%).
3. **Do not** burn long GPU on this stack.
4. Paper framing stays on **AVBT + Dual-Expert Keys (DEK)**: reliability preserved + FC cut (short multi-seed + fair long).

Machine-readable: `aaai_short_pb_gate.json`.
