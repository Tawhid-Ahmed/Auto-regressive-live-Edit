# AAAI Phase C.1 — Long-form transfer (no retrain)

**Date:** 2026-07-13  
**Protocol:** `dsn=100`, `sen=10`, `seed=42`,  
`data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json`,  
`--meeting_step 6`  
**Ckpts:** short-trained Dual-Expert Keys (DEK) + LiveEdit baseline (L0) (same as short table)

| Short | Full name |
|-------|-----------|
| L0 | LiveEdit baseline |
| DEK | AVBT + Dual-Expert Keys |

---

## Results

| Method | Rel ↑ | Swap ↑ | Hard ↑ | FC ↓ | Swap-gap | EM |
|--------|------:|-------:|-------:|-----:|---------:|---:|
| LiveEdit baseline (L0) short→long | 0.2932 | 0.4059 | 0.3929 | 0.96 | −0.1127 | 0.0 |
| Dual-Expert Keys (DEK) short→long | 0.2704 | 0.3784 | 0.3854 | **0.46** | −0.1080 | 0.0 |

**Eval dirs**
- Dual-Expert Keys (DEK): `.../VLKEB_GROUNDING-aaai_DEK_long_transfer-2026.07.13-09.59.53`
- LiveEdit baseline (L0): `.../VLKEB_GROUNDING-aaai_L0_long_transfer-2026.07.13-10.03.05`

---

## G3 check (transfer only)

| Criterion | Need | Got | Status |
|-----------|------|-----|--------|
| Long image-swap ≥55% | ≥0.55 | 0.378 | **FAIL** |
| Reliability ≥70% | ≥0.70 | 0.270 | **FAIL** |
| False-cross clearly below LiveEdit baseline (L0) | yes | 0.46 vs 0.96 | **PASS (routing only)** |

**G3 transfer: FAIL** — domain mismatch. Short-trained editors do not produce usable long-form reliability (EM=0 both). FC win transfers; the edit itself does not.

---

## Takeaway

1. **False-cross is domain-robust** (still 46% vs 96%) — supports the routing claim.
2. **Token metrics are not** — both Dual-Expert Keys (DEK) and LiveEdit baseline (L0) collapse on long targets (~27–29% Rel). This is not a Dual-Expert Keys-specific bug; short→long transfer fails for both.
3. **Plan says C.2 next:** retrain AVBT + Dual-Expert Keys on long grounding train data, then re-eval.

**Train data note:** full `grounding_train_longform.json` not present under `data/VLKEB/grounding/`; only `pilot_train_longform_10.json` + eval longform. Need to locate/build long train before C.2.

---

## Next

**Phase C.2** — build/find long train JSON → train `VLKEB_avbt_long_DEK` → eval long (dsn=100 gate, then larger if pass).
