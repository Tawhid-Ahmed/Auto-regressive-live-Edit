# AAAI — EVQA Dataset Stress (AVBT + Dual-Expert Keys)

**Date:** 2026-07-25  
**Role:** Cross-dataset stress supporting the main VLKEB FC claim  
**Main results hub:** [`../AAAI_RESULTS_ANALYSIS.md`](../AAAI_RESULTS_ANALYSIS.md)  
**Plan:** [`../../../AAAI_STRESS_TRANSFER_PLAN.md`](../../../AAAI_STRESS_TRANSFER_PLAN.md)  
**Status:** [`STATUS.md`](STATUS.md)

---

## 0. Paper placement

| Paper section | Use |
|---------------|-----|
| **Results — cross-setting / dataset stress** | §1 pilot + §2 multi-seed |
| **Claim** | DEK cuts **false-positive retrieval (FC)**; Rel held; AVBT alone does not fix FC |
| **Do not claim** | EVQA token-swap SOTA; LiveEdit Rel/Gen/Loc leaderboard |

**Draft sentence:**

> On EasyEdit EVQA with constructed swap/hard-negative probes, Dual-Expert Keys reduce image-swap false-cross while holding reliability near the lifelong baseline across five evaluation seeds (dsn=300); AVBT alone leaves FC near the baseline — matching the VLKEB pattern.

---

## 1. Pilot (seed 42, dsn=100) — first grounding stress

**JSON:** [`aaai_stress_evqa_grounding_summary.json`](aaai_stress_evqa_grounding_summary.json)

| Method | Rel↑ | Swap acc | Hard acc | **FC swap↓** | FC hard↓ | Mean #retr (Rel / Hard) |
|--------|-----:|---------:|---------:|-------------:|---------:|-------------------------:|
| L0 | 0.936 | 0.153 | 0.206 | **0.97** | 0.92 | 3.91 / 3.66 |
| S2 (AVBT) | 0.928 | 0.153 | 0.206 | **0.97** | 0.92 | 4.27 / 3.65 |
| **DEK** | **0.928** | 0.153 | 0.206 | **0.00** | **0.00** | **1.00 / 0.06** |

Directional only (single seed / small dsn). FC=0 means **0/100** false-cross retrievals on that split.

---

## 2. Must-do reliability — multi-seed + larger dsn

**Protocol:** same L0 / S2 / DEK checkpoints; **5 seeds** `{42,0,1,2,3}`; **dsn=300** subsampled from full `grounding_eval_short.json` with `sample_seed=eval_seed`; `sen=10`; `--log_routing`.

**JSON:** [`aaai_stress_evqa_grounding_multiseed.json`](aaai_stress_evqa_grounding_multiseed.json)  
**Script:** `tools/stress/run_evqa_grounding_multiseed.py`  
**Gate:** **PASS** (Rel held; FC drop vs L0)

### Aggregate (mean ± std over 5 seeds)

| Method | Rel↑ | **FC swap↓** | **FC hard↓** | Swap acc | Hard acc |
|--------|------:|-------------:|-------------:|---------:|---------:|
| L0 | 0.950 ± 0.006 | 0.313 ± 0.381 | 0.968 ± 0.006 | 0.154 ± 0.012 | 0.179 ± 0.018 |
| S2 (AVBT) | 0.952 ± 0.006 | 0.303 ± 0.388 | 0.970 ± 0.004 | 0.155 ± 0.013 | 0.185 ± 0.021 |
| **DEK** | **0.952 ± 0.005** | **0.015 ± 0.011** | **0.027 ± 0.013** | 0.156 ± 0.011 | 0.183 ± 0.018 |

### Per-seed check (same checkpoints; only the 300-example subsample changes)

**How to read:** lower FC is better. Each row is one random eval subsample. Compare **DEK** to **L0** in the same row — S2 is AVBT-only (should look like L0).

**False-cross on image swap (FC swap ↓)**

| Seed | L0 | S2 | DEK |
|-----:|---:|---:|----:|
| 42 | 0.993 | 0.997 | **0.023** |
| 0 | 0.120 | 0.113 | **0.003** |
| 1 | 0.153 | 0.140 | **0.010** |
| 2 | 0.187 | 0.143 | **0.030** |
| 3 | 0.113 | 0.123 | **0.010** |

**False-cross on hard-negative image (FC hard ↓)**

| Seed | L0 | S2 | DEK |
|-----:|---:|---:|----:|
| 42 | 0.970 | 0.977 | **0.037** |
| 0 | 0.967 | 0.967 | **0.027** |
| 1 | 0.960 | 0.970 | **0.017** |
| 2 | 0.967 | 0.967 | **0.043** |
| 3 | 0.977 | 0.970 | **0.013** |

**Reliability still held (Rel ↑)**

| Seed | L0 | DEK |
|-----:|---:|----:|
| 42 | 0.942 | 0.950 |
| 0 | 0.949 | 0.951 |
| 1 | 0.948 | 0.948 |
| 2 | 0.957 | 0.960 |
| 3 | 0.952 | 0.951 |

**Takeaway from these rows:** DEK stays low FC on every seed (swap ≤3%, hard ≤5%). S2 ≈ L0, so binding alone does not fix FC. Seed 42’s L0 FC swap (~99%) is a hard subsample; DEK still drops it to ~2%.

**Notes for write-up:**
- Pilot (dsn=100, seed 42) had DEK FC swap/hard = **0.00**; multi-seed mean is **~1.5% / ~2.7%** — still near-floor vs L0/S2.
- Prefer **mean±std** (table above) plus “DEK FC ≤3%/≤5% on all 5 seeds.”
- Token swap/hard-acc stay ~0.15–0.20 — claim is **routing FC**, not swap SOTA.

*(Archive of an earlier run that only reshuffled a fixed first-300 prefix — not for paper: [`aaai_stress_evqa_grounding_multiseed_fixedprefix300.json`](aaai_stress_evqa_grounding_multiseed_fixedprefix300.json).)*

---

## 3. Checkpoints & data

| Run | Path |
|-----|------|
| L0 | `records/liveedit/blip2-opt-2.7b/EVQA_L0_full-2026.07.24-12.05.12/checkpoints/epoch-20-i-29400-ema_loss-0.1996` |
| S2 | `records/liveedit/blip2-opt-2.7b/EVQA_avbt_S2_gnd-2026.07.25-11.32.38/checkpoints/epoch-1-i-125-ema_loss-3.8897` |
| DEK | `records/liveedit/blip2-opt-2.7b/EVQA_avbt_DEK_gnd-2026.07.25-11.47.44/checkpoints/epoch-1-i-125-ema_loss-4.4475` |

| File | n |
|------|--:|
| `data/easy-edit-mm/grounding/grounding_train_short.json` | 800 (train `-dn 500`) |
| `data/easy-edit-mm/grounding/grounding_eval_short.json` | 2093 |

---

## 4. Contrast — plain EVQA without grounding *(do not headline)*

Without swap/hard-neg train JSON, DEK Rel dropped on standard Rel/Gen/Loc. Prefer §1–§2.

[`aaai_stress_dataset_summary.json`](aaai_stress_dataset_summary.json)

---

## 5. Link map

```text
doc/results/meeting_validation/step6/AAAI_RESULTS_ANALYSIS.md   ← main VLKEB
        └── stress/AAAI_EVQA_STRESS_RESULTS.md                  ← this file
                ├── aaai_stress_evqa_grounding_summary.json
                └── aaai_stress_evqa_grounding_multiseed.json
```
