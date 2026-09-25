# AAAI Phase C — Fair long-form (long-trained ckpts)

**Date:** 2026-07-13  
**Protocol:** `dsn=100`, `sen=10`, `seed=42`,  
`data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json`,  
`--meeting_step 6`

**Important:** This supersedes `AAAI_LONG_TRANSFER.md` (short ckpts → long eval).  
Here both models are trained on **VLKEB-long**.

| Short | Full name |
|-------|-----------|
| L0 | LiveEdit baseline |
| DEK | AVBT + Dual-Expert Keys |
| Rel / Swap / Hard / FC | Reliability / Image-swap / Hard-negative / False-cross |

---

## Checkpoints

| Role | Path |
|------|------|
| LiveEdit baseline (L0) | `records/.../VLKEB_long_baseline-2026.06.02-13.43.00/.../epoch-20-i-27800-ema_loss-0.7936` |
| Dual-Expert Keys (DEK) | `records/.../VLKEB_avbt_long_DEK-2026.07.13-10.17.25/.../epoch-1-i-125-ema_loss-7.1174` |

**Dual-Expert Keys train:** from long LiveEdit baseline (L0), `dn=500`, `eps=1`,  
`data/VLKEB/grounding/grounding_train_longform.json` (built from `train_longform_ollama.json`),  
AVBT / Dual-Expert Keys flags: λ=0.2, neg_w=2.0, hinge=0.3, contrastive, `--dek_dual_gate`.

---

## Results (fair)

| Method | Rel ↑ | Swap ↑ | Hard ↑ | FC ↓ | Swap-gap |
|--------|------:|-------:|-------:|-----:|---------:|
| LiveEdit baseline (L0) long | **0.7400** | 0.4242 | 0.4281 | 0.88 | 0.3158 |
| **AVBT + Dual-Expert Keys (DEK) long** | 0.7282 | **0.4268** | 0.4255 | **0.38** | 0.3014 |

**Eval dirs**
- LiveEdit baseline (L0): `.../VLKEB_GROUNDING-aaai_L0_long-2026.07.13-10.12.34`
- Dual-Expert Keys (DEK): `.../VLKEB_GROUNDING-aaai_DEK_long-2026.07.13-10.35.50`

---

## G3 check

| Criterion | Need | Got | Status |
|-----------|------|-----|--------|
| Long image-swap ≥55% | ≥0.55 | 0.427 | **FAIL** |
| Reliability ≥70% | ≥0.70 | 0.728 | **PASS** |
| False-cross clearly below LiveEdit baseline (L0) | yes | 0.38 vs 0.88 (−50 pp) | **PASS** |

**G3 headline swap: FAIL.**  
**G3 reliability + FC story: PASS.**

---

## Takeaway

1. With **matched long training**, Dual-Expert Keys (DEK) keeps reliability (~73% vs LiveEdit baseline 74%) and **cuts FC 88% → 38%**.
2. Token swap barely moves (+0.3 pp) — same pattern as short: novelty is **anti-leakage routing**, not raw swap SOTA.
3. Short→long transfer numbers in `AAAI_LONG_TRANSFER.md` remain useful as a negative control only.

**Next options:** more long Dual-Expert Keys train (eps/dn), Phase D pool-buffer mining, or write paper framing around FC + rel on short+long.
