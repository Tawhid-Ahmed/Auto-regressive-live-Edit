# AAAI Phase B — Multi-seed: Dual-Expert Keys (DEK) vs LiveEdit baseline (L0)

**Date:** 2026-07-13  
**Protocol:** `dsn=100`, `sen=10`, seeds `{42,123,456,789,1024}`,  
`grounding_eval_short.json`, `--meeting_step 6`  
**Companion JSON:** [`aaai_dek_multiseed.json`](aaai_dek_multiseed.json)

| Short | Full name |
|-------|-----------|
| L0 | LiveEdit baseline |
| DEK | AVBT + Dual-Expert Keys |

---

## Per-seed results

| Seed | Method | Rel | Swap | Hard | FC | Swap-gap |
|-----:|--------|----:|-----:|-----:|---:|---------:|
| 42 | LiveEdit baseline (L0) | 0.9347 | 0.4072 | 0.4224 | 0.96 | 0.5275 |
| 42 | Dual-Expert Keys (DEK) | 0.9372 | 0.4773 | 0.4201 | **0.46** | 0.4599 |
| 123 | LiveEdit baseline (L0) | 0.9347 | 0.4179 | 0.4232 | 0.96 | 0.5168 |
| 123 | Dual-Expert Keys (DEK) | 0.9372 | 0.4729 | 0.4306 | **0.46** | 0.4643 |
| 456 | LiveEdit baseline (L0) | 0.9347 | 0.4108 | 0.4424 | 0.96 | 0.5239 |
| 456 | Dual-Expert Keys (DEK) | 0.9372 | **0.5052** | 0.4369 | **0.46** | 0.4320 |
| 789 | LiveEdit baseline (L0) | 0.9347 | 0.4157 | 0.4351 | 0.96 | 0.5190 |
| 789 | Dual-Expert Keys (DEK) | 0.9372 | 0.4860 | 0.4248 | **0.46** | 0.4512 |
| 1024 | LiveEdit baseline (L0) | 0.9347 | 0.4097 | — | 0.96 | 0.5250 |
| 1024 | Dual-Expert Keys (DEK) | 0.9372 | 0.4895 | 0.4233 | **0.46** | 0.4477 |

---

## Mean ± std (5 seeds)

| Method | Rel | Swap | FC | Swap-gap |
|--------|----:|-----:|---:|---------:|
| LiveEdit baseline (L0) | 0.9347 ± 0 | 0.4123 ± 0.0044 | 0.960 ± 0 | 0.5224 ± 0.0044 |
| **AVBT + Dual-Expert Keys (DEK)** | 0.9372 ± 0 | **0.4862 ± 0.0125** | **0.460 ± 0** | 0.4510 ± 0.0125 |

**Swap lift:** Dual-Expert Keys (DEK) − LiveEdit baseline (L0) = **+7.39 pp** (every seed positive).  
**FC cut:** 0.96 → 0.46 (**−50 pp**, identical on all seeds).

---

## G2 gate

| Criterion | Result |
|-----------|--------|
| Dual-Expert Keys (DEK) mean FC ≤ 55% | **PASS** (46%) |
| Dual-Expert Keys (DEK) mean swap ≥ LiveEdit baseline (L0) + 5 pp | **PASS** (+7.39 pp) |
| Significance (FC or swap-gap) | FC diff constant −0.50 all seeds (degenerate Wilcoxon; effect is deterministic). Swap diffs all >0 (5/5). |

**G2: PASS.**

---

## Takeaway

- Not a seed-42 fluke: FC win and swap lift hold across all 5 seeds.
- FC is rock-stable (exactly 46% every seed) — routing gate, not noise.
- Swap has small seed variance (std ~1.3 pp); seed 456 crosses 50% short swap.

**Next:** Phase C — long-form transfer eval (no retrain first).
