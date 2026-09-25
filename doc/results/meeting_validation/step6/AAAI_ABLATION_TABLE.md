# AAAI Ablation Table — AVBT / Dual-Expert Keys (G4)

**Date:** 2026-07-13  
**Protocol (short):** `dsn=100`, `sen=10`, `seed=42`,  
`grounding_eval_short.json`, `--meeting_step 6`, `--log_routing`  
**Companion JSON:** [`aaai_ablation_table.json`](aaai_ablation_table.json)

This is the **component ablation** for the paper (LiveEdit family only).  
External editors (Fine-tuning / MEND-VL / SERAC-VL) stay in `AAAI_COMPETITOR_TABLE_SHORT.md`.

### Naming
| Short | Full name |
|-------|-----------|
| L0 | LiveEdit baseline |
| S2 | AVBT (patch grounding, no dual-gate) |
| DEK | AVBT + Dual-Expert Keys (**ours**) |
| FX | AVBT+DEK + false-cross mining |
| PB | AVBT+DEK + pool buffer |

---

## 1. Short VLKEB — component stack

| # | Method | Components | Rel ↑ | Swap ↑ | Hard ↑ | FC ↓ | Swap-gap ↓ | Role |
|---|--------|------------|------:|-------:|-------:|-----:|-----------:|------|
| 1 | LiveEdit baseline (L0) | baseline MoE | 0.9347 | 0.4072 | 0.4224 | 0.96 | 0.5275 | baseline |
| 2 | AVBT (S2) | + patch grounding + AVBL | 0.9380 | 0.4782 | 0.4259 | 0.96 | 0.4598 | +AVBT (no Dual-Expert Keys) |
| 3 | **AVBT + Dual-Expert Keys (DEK)** | AVBT (S2) + vision∧text dual-gate | **0.9372** | **0.4773** | 0.4201 | **0.46** | 0.4599 | **ours** |
| 4 | False-cross mining (FX) | Dual-Expert Keys + DINO neighbor + refuse | 0.9372 | 0.4723 | 0.4176 | **0.43** | 0.4649 | negative (no swap gain) |
| 5 | Pool buffer (PB) | FX + pool buffer 64 | 0.9372 | 0.4798 | 0.4357 | 0.55 | 0.4574 | negative (FC regresses) |

**What each row proves**

| Step | Δ vs previous | Claim |
|------|---------------|--------|
| LiveEdit baseline (L0) → AVBT (S2) | Swap +7.1 pp; FC unchanged 96% | AVBT lifts token swap; does **not** fix false-cross |
| AVBT (S2) → Dual-Expert Keys (DEK) | FC 96% → **46%**; swap flat | **Dual-Expert Keys is the FC mechanism**; swap already from AVBT |
| Dual-Expert Keys (DEK) → False-cross mining (FX) | FC 46% → 43%; swap −0.5 pp | Neighbor+refuse slightly tighter routing; **no token gain** |
| False-cross mining (FX) → Pool buffer (PB) | FC 43% → **55%**; swap flat | Cross-batch buffer **hurts** FC; abandon |

**Paper primary row:** #3 AVBT + Dual-Expert Keys (DEK).  
**Paper negatives:** #4–5 (show we tried stronger mining; swap ceiling remains).

---

## 2. Multi-seed confirmation (Dual-Expert Keys vs LiveEdit baseline)

From `AAAI_DEK_MULTISEED.md` (seeds `{42,123,456,789,1024}`):

| Method | Rel (mean) | Swap (mean±std) | FC (mean) |
|--------|-----------:|----------------:|----------:|
| LiveEdit baseline (L0) | 0.9347 | 0.4123 ± 0.009 | 0.96 |
| **AVBT + Dual-Expert Keys (DEK)** | 0.9372 | **0.4862 ± 0.0125** | **0.46** |

- Swap lift: **+7.39 pp** (all seeds positive)  
- FC: **96% → 46%** (stable)  
- **G2: PASS**

---

## 3. Fair long-form (matched training)

From `AAAI_LONG_FAIR.md` (seed 42, long train + long eval):

| Method | Rel ↑ | Swap ↑ | Hard ↑ | FC ↓ | Swap-gap ↓ |
|--------|------:|-------:|-------:|-----:|-----------:|
| LiveEdit baseline (L0) long | 0.7400 | 0.4242 | 0.4281 | 0.88 | 0.3158 |
| **AVBT + Dual-Expert Keys (DEK) long** | 0.7282 | 0.4268 | 0.4255 | **0.38** | 0.3014 |

- Rel held (~73–74%); FC **88% → 38%**  
- Swap essentially flat → **G3 swap FAIL**; **FC+rel story PASS**

---

## 4. Checkpoints (canonical)

| Method | Checkpoint |
|--------|------------|
| LiveEdit baseline (L0) short | `.../VLKEB_baseline-2026.04.09-08.20.07/.../epoch-20-i-25950-ema_loss-0.1201` |
| AVBT (S2) | `.../VLKEB_avbt_short_S2-2026.07.10-11.16.19/.../epoch-1-i-125-ema_loss-4.0787` |
| Dual-Expert Keys (DEK) | `.../VLKEB_avbt_short_DEK-2026.07.10-18.13.18/.../epoch-1-i-125-ema_loss-4.0100` |
| False-cross mining (FX) | `.../VLKEB_avbt_short_FX-2026.07.10-18.41.48/.../epoch-1-i-125-ema_loss-4.7361` |
| Pool buffer (PB) | `.../VLKEB_avbt_short_PB-2026.07.13-10.45.41/.../epoch-1-i-125-ema_loss-4.5631` |
| LiveEdit baseline (L0) long | `.../VLKEB_long_baseline-2026.06.02-13.43.00/.../epoch-20-i-27800-ema_loss-0.7936` |
| Dual-Expert Keys (DEK) long | `.../VLKEB_avbt_long_DEK-2026.07.13-10.17.25/.../epoch-1-i-125-ema_loss-7.1174` |

**Eval flags**
- LiveEdit baseline (L0): `--log_routing`
- AVBT (S2): `--patch_grounding --log_routing`
- Dual-Expert Keys (DEK) / False-cross mining (FX) / Pool buffer (PB): `--patch_grounding --dek_dual_gate --log_routing`

---

## 5. Paper framing (use this wording)

1. **AVBT** improves adversarial token swap (~+7 pp) but leaves false-cross ~96%.  
2. **Dual-Expert Keys (DEK)** cuts false-cross roughly in half (short 96→46; long 88→38) **without** collapsing reliability.  
3. Extra mining (false-cross mining, pool buffer) does not raise swap past ~48%; FX’s small FC gain is secondary — **Dual-Expert Keys is the clean method row**.

Do **not** claim G3 swap ≥55% or that FX/PB beat Dual-Expert Keys on the primary operating point.
