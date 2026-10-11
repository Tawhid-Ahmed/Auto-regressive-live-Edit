# G5 gate write-up — LLaVA-only (provisional)

**Gate:** G5 (second backbone)  
**Date written:** 2026-10-11  
**Decision due:** 2026-10-15  
**Scope of this note:** **LLaVA-1.5-7B lab ladder only.** Phase C (BLIP-2 same-protocol re-extract) is **not** included; dual-backbone claim stays open until C lands.  
**Glossary:** [`../../CVPR2027_GATE_LOG.md`](../../CVPR2027_GATE_LOG.md) § Glossary  
**Number tables:** [`G5_RESULTS_TABLES.md`](G5_RESULTS_TABLES.md)

Same four blocks as the gate log: Result · Good · Bad · Carry-forward.

---

## Result

### Protocol (disclose)

| Item | Value |
|---|---|
| Backbone | LLaVA-1.5-7B (lab RTX 4090) |
| Train | L0 `dn=500` × **20** → S2 / DEK `dn=500` × **1** (AVBT; DEK + dual gate) |
| Eval | `dsn=100`, `sen=10`, seed 42 (multi-seed: `{42,0,1,2,3}`) |
| Probes | reliability, image_swap, hard_negative (+ standard Gen/Loc in B1) |
| Counting | binary `total_leak` = `stale_leak` + `foreign_leak` (headline) |
| Not claimed | Matched train compute vs BLIP-2 full-set L0; answer-level swap rescue |

### Headline numbers (binary, %)

**VLKEB natural pool (B1, seed 42)**

| Method | Swap total leak ↓ | Hard-neg total leak ↓ | Rel hit ↑ | Only-own ↑ | Abstain (swap) |
|---|---:|---:|---:|---:|---:|
| L0 | 100.0 | 98.0 | 79.0 | 0.0 | 0.0 |
| S2 | 100.0 | 100.0 | 98.0 | 0.0 | 0.0 |
| DEK | **36.0** (−64 pp) | **46.0** (−52 pp) | **1.0** | 1.0 | **64.0** |

**Easy-VQA (E1, seed 42)**

| Method | Swap total leak ↓ | Hard-neg total leak ↓ | Rel hit ↑ | Abstain (swap) |
|---|---:|---:|---:|---:|
| L0 | 100.0 | 100.0 | 71.0 | 0.0 |
| DEK | **13.0** (−87 pp) | **91.0** (−9 pp) | **1.0** | **87.0** |

**VLKEB multi-seed (E4, mean ± std, n=5)**

| Method | Swap TL ↓ | Hn TL ↓ | Rel hit ↑ |
|---|---:|---:|---:|
| L0 | 100.0 ± 0.0 | 99.6 ± 0.9 | 76.6 ± 5.4 |
| DEK | **35.0 ± 7.8** | **41.2 ± 10.2** | **0.6 ± 0.5** |

**Pair-closed curve (E2):** DEK swap/hn total leak stays well below L0 at n∈{100,300,1000}; Rel hit stays ~0–0.3% at every size.

**Routing controls (E3):** L0+`and` / `always_reject` kill leak by collapsing Rel; DEK+`and` matches B1 DEK; DEK+`liveedit` returns saturated leak — same qualitative control story as BLIP-2 G2.

### Verdict (LLaVA-only)

**G5 question:** Does Dual-Expert Keys reduce `total_leak` on a second backbone **without collapsing reliability**?

| Criterion | LLaVA evidence |
|---|---|
| Leak reduction transfers | **Yes** — VLKEB swap −64 pp, hn −52 pp; EVQA swap −87 pp; stable in 5-seed and pair-closed |
| Reliability held | **No** — Rel hit 79%→1% (VLKEB), 71%→1% (EVQA); multi-seed Rel hit 0.6±0.5% |
| Mechanism | Almost all swap gain is **abstain**, not `correct_route` (~0%) |
| Answer-level | No rescue — `emits_y_i` / `emits_y_j` ~0; Rel EM ~1% |

**Provisional gate verdict: WEAK (LLaVA-only).**

- DEK is **not BLIP-2-only for leak reduction**, but on this LLaVA short-n ladder it behaves like a **strong abstain / AND-style gate** that sacrifices reliability hit.
- Do **not** claim LLaVA reproduces the BLIP-2 G0 pattern (high Rel hit + lower leak).
- Do **not** close G5 as PASS until Phase C (BLIP-2 re-extract) is filled **or** an explicit methodology note accepts LLaVA as “leak-only / Rel-failed” second-backbone stress.
- Plan A for **BLIP-2** is unchanged by this note; LLaVA is a **stress / limitation** result, not a second primary table yet.

---

## Good

- Fresh lab L0×20 → S2 → DEK finished; paper paths are `records/lab_g5/…`, not AAAI home smoke.
- Honest taxonomy extract (B1) + Easy-VQA (E1) + pair-closed (E2) + routing controls (E3) + 5-seed (E4) are all on disk under `doc/results/cvpr2027/g5/`.
- Leak advantage is **large and stable** (multi-seed; pool sizes 100→1000).
- Stale leak on DEK image_swap is tiny (VLKEB 1%; EVQA 0%); residual is mostly **foreign leak** + abstain.
- Control table confirms DEK’s train dual gate is doing real work vs DEK+`liveedit` (100% swap TL).
- Budget disclosure is clear: LLaVA short-n L0×20 vs BLIP-2 full-set L0×20 (when C lands).

---

## Bad

- **Reliability hit collapse** is the gate-breaker for a clean second-backbone PASS. DEK Rel hit ≈ 0–1% everywhere we measured.
- **Hard-neg on Easy-VQA barely moves** (100→91) while swap collapses via abstain — dataset-dependent story; do not oversell EVQA hn.
- **`correct_route` ≈ 0%** on LLaVA for L0 and DEK; partner routing is not learned.
- **Only-own** stays ~0–1% (vs BLIP-2 DEK ~100% in G0) — no precision win to quote.
- Answer-level metrics do not support DEK (same as BLIP-2 caveat, worse here because Rel EM is already near zero).
- Soft-weight τ side check is still a **vision-logit proxy**, not `fuse_coe` (M3); do not paper those τ rows as fuse coefficients.
- Phase C missing → no apples-to-apples BLIP-2 column in the dual table yet.

---

## Carry-forward

| Item | Action |
|---|---|
| **Phase C** | Re-extract BLIP-2 L0/S2/DEK with the same scripts; fill `G5_RESULTS_TABLES.md` §4–5; then finalize G5 PASS/WEAK/FAIL in the gate log |
| **Paper** | If citing LLaVA before Rel fix: frame as **second-backbone leak stress** with explicit Rel-collapse / abstain mechanism; lead hn + four-way swap; no Rel/only-own claim for LLaVA DEK |
| **Method** | Treat LLaVA Rel collapse as a **HIGH** open issue (alongside M6 hn under pair-closed). Candidate later tries: train-time Rel constraint, softer dual gate, or backbone-specific τ — **discuss before implementing** |
| **Gate log** | Replace G5 “Result _(pending)_” with a short pointer to this file; update status board when Phase C closes |
| **Supervisor progress** | R1 #8 / R2 #5: LLaVA extract done; dual extract still due 15 Oct |

---

## Artifact index

| Artifact | Path |
|---|---|
| VLKEB taxonomy | `leakage_taxonomy_llava.json` |
| Easy-VQA taxonomy | `leakage_taxonomy_llava_evqa.json` |
| Pair-closed curve | `pool_size_curve_llava_s42.{md,json}` |
| Routing table | `routing_table_llava_vlkeb_s42.{md,json}` |
| Multi-seed | `multiseed_llava_vlkeb_d100.{md,json}` |
| Full tables | `G5_RESULTS_TABLES.md` |
| Tracker | `EXPERIMENT_TRACKER.md` |

---

*End of LLaVA-only G5 write-up. Final G5 board status waits on Phase C or an explicit Oct-15 decision to accept WEAK.*
