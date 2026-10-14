# G5 gate write-up — LLaVA second backbone

**Gate:** G5 (second backbone)  
**Decision date:** **2026-10-15** (accepted **2026-10-14**)  
**Board verdict:** **WEAK**  
**Scope:** LLaVA-1.5-7B lab ladder + root-cause follow-ups through S2-tune A. Phase C (BLIP-2 same-protocol re-extract) **not** closed.  
**Glossary:** [`../../CVPR2027_GATE_LOG.md`](../../CVPR2027_GATE_LOG.md) § Glossary  
**Root cause:** [`G5_LLAVA_DEK_ROOT_CAUSE.md`](G5_LLAVA_DEK_ROOT_CAUSE.md)  
**Number tables (historical extract):** [`G5_RESULTS_TABLES.md`](G5_RESULTS_TABLES.md)  
**Metric audit:** [`G5_METRIC_AUDIT.md`](G5_METRIC_AUDIT.md)

Same four blocks as the gate log: Result · Good · Bad · Carry-forward.

---

## Result

### G5 question

Does Dual-Expert Keys reduce `total_leak` on a second backbone **without collapsing reliability hit**?

### Board answer (accepted)

**WEAK.** LLaVA does **not** reproduce the BLIP-2 G0 pattern (held Rel + lower leak). Plan A for **BLIP-2** is unchanged. LLaVA is a **stress / limitation** result until M8 (or equivalent) closes the vision health gate.

### Why WEAK (not PASS / not FAIL)

| Criterion | Evidence | Board reading |
|---|---|---|
| Second backbone trained + extracted | Lab L0 `dn=500`×20 → S2/DEK; B1/E1–E4 on disk | Done |
| Reproduces BLIP-2 DEK (Rel held + leak ↓) | Rel hit collapses on the extracted DEK ladder; health gates fail after S2 | **No** |
| Root cause identified | Vacuous / under-negatived S2 routing (`bs=1`, then residual at `bs=2`) → open vision gate → DEK degenerates | Yes — engineering + method gap, not “DEK impossible on LLaVA” |
| Cheap fix closed the gate? | S3 `bs=2` S2 improved but still FAIL; S2-tune A (stronger AVBL) **worse** vision-ok | **No** → M8 next |
| Dual-backbone apples-to-apples? | Phase C open (BLIP-2 S2 ckpt missing locally; F4 `head` vs `subsample`) | Dual table **deferred** |

### Protocol (disclose)

| Item | Value |
|---|---|
| Backbone | LLaVA-1.5-7B (lab RTX 4090) |
| Train budget | L0 `dn=500` × **20** (`bs=2`, M7); S2/DEK targets `dn=500` × **1** |
| Known train defect (M7 / F3) | First S2/DEK ladder used **`bs=1`** → soft/hard/AVBL routing losses vacuous |
| Follow-ups | S3 S2 at `bs=2` (VRAM pack, λ=0.1); S2-tune A (retrieval margin 1.0, neg_w 4.0, contrast_m 0.3) — health-only |
| Eval | `dsn=100`, `sen=10`, seed 42 (+ multi-seed E4) |
| Health gate | Rel vision-ok % of pool ≤ 20; swap own–prot margin < 0; text top-1 ≥ L0 − 5 pp |
| Not claimed | Matched train compute vs BLIP-2 full-set L0; LLaVA DEK as a Rel-held method win; side-by-side BLIP-2 vs LLaVA until Phase C |

### Routing health (canonical for Oct 15 — do not skip)

Train-split health, `dsn=50`, seed 42. Directions: vision-ok % ↓; swap own–prot margin should be < 0; text top-1 ↑.

| Run | Rel vision-ok % ↓ | Swap own–prot margin | Rel text top-1 ↑ | Health |
|---|---:|---:|---:|---|
| M7 L0 (`bs=2`) | 12.3 | −13.7 | 70 | **PASS** |
| M7 S2 (`bs=1`) | 98.1 | +43.4 | 56 | FAIL |
| S3 S2 (`bs=2`) | 33.2 | +6.73 | 78 | FAIL |
| S2-tune A | **62.0** | **+13.11** | 74 | **FAIL** (worse than S3) |

L0 is healthy. **S2 re-opens the vision gate.** Stronger AVBL knobs do not fix it. **Do not train / quote DEK** until this gate passes.

### Historical taxonomy extract (disclose — not a method claim)

Oct 11 VLKEB binary extract on the **`bs=1` S2/DEK** ladder (open vision gate → text-argmax / abstain behavior). Useful as a stress observation; **do not** cite as “LLaVA DEK works.”

| Method | Swap `total_leak` ↓ | Hard-neg `total_leak` ↓ | Rel hit ↑ | Note |
|---|---:|---:|---:|---|
| L0 | 100.0 | 98.0 | 79.0 | |
| DEK | 36.0 | 46.0 | **1.0** | Almost all swap gain is **abstain**; `correct_route` ≈ 0% |
| Weight-aware τ=0.1 (audit) | DEK swap TL **worse** than L0 | — | — | Leak “win” does not survive real `fuse_coe` |

Full tables: `G5_RESULTS_TABLES.md`. Mechanism write-up: `G5_LLAVA_DEK_ROOT_CAUSE.md` (F1–F4).

### Verdict sentence (for board / paper footnote)

**G5 = WEAK (2026-10-15):** second-backbone DEK does not yet hold reliability on LLaVA; root cause is under-negatived S2 routing that opens the vision gate (confirmed by health gates + failed S2-tune A). BLIP-2 Plan A unchanged. Next method item: **M8** (batch-independent routing negatives). Phase C dual extract deferred.

---

## Good

- Lab L0 at `bs=2` (M7) is **routing-healthy** (vision gate PASS; text top-1 70%).
- Root cause chain documented with artifacts (health JSONs, extractor energy, proto/foreign breakdown).
- S3 showed F3 was real: `bs=2` cut vision-ok 98→33 and restored text top-1 — but not enough.
- S2-tune A exhausted cheap AVBL retuning (finite train; clear FAIL) → avoids another week of knob spinning.
- Paper can disclose a concrete limitation instead of a silent second-backbone failure.
- BLIP-2 G0/G2/G3 Plan A untouched.

---

## Bad

- **No Rel-held LLaVA DEK** suitable for a second primary table.
- Extracted DEK leak drop is **mechanism-contaminated** (open vision gate + abstain + image-conditioned prompt key); weight-aware audit reverses the swap win.
- S2 at `bs=2` still under-negatived (1 in-batch foreign vs BLIP-2’s 3); foreign inflation + own stays hot on `image_swap`.
- Phase C open → F4 protocol confound (`head` vs `subsample` partner-in-pool) still blocks fair dual tables.
- Oct 15 dual-backbone PASS is **not** available.

---

## Carry-forward

| Item | Action |
|---|---|
| **M8** | Batch-independent routing negatives (FIFO key bank into soft/hard/AVBL softmaxes). Discuss → add to methodology table → implement. Acceptance: health gate PASS after S2, then DEK. |
| **Phase C** | Recover BLIP-2 short S2 ckpt; re-extract L0/S2/DEK under `subsample`; fill `G5_RESULTS_TABLES.md` §4–5. Does not unblock LLaVA Rel. |
| **Paper** | Cite LLaVA as **WEAK / limitation**: disclose M7 `bs=1` S2/DEK defect and open vision gate; **do not** quote LLaVA DEK Rel or binary leak as a method win; lead with BLIP-2. |
| **Not to do** | DEK from current S3/tuneA S2; more AVBL λ cranking; M10 (extractor energy not dominated); side-by-side BLIP-2 vs LLaVA until Phase C. |
| **Supervisor** | R1 #8 / R2 #5: G5 **WEAK accepted**; dual extract + M8 remain open. |

---

## Artifact index

| Artifact | Path |
|---|---|
| Root cause + health tables | `G5_LLAVA_DEK_ROOT_CAUSE.md` |
| S3 S2 health (FAIL) | `routing_health_s2_s3_bs2.json` |
| L0 health (PASS) | `routing_health_l0_s3diag.json` |
| S2-tune A health (FAIL) | `routing_health_s2_tune_a.json` |
| Pre-M summary | `s3_pre_m_diag_summary.json` |
| Historical VLKEB taxonomy | `leakage_taxonomy_llava.json` / `leakage_taxonomy_llava_m7.json` |
| Metric audit | `G5_METRIC_AUDIT.md` |
| Full tables | `G5_RESULTS_TABLES.md` |

---

*End of G5 write-up. Board status: **WEAK accepted 2026-10-15**.*
