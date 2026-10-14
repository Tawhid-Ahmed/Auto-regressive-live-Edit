# CVPR main-track improvement backlog

**Purpose:** Collect every **Bad** / open-flag item from BLIP-2-OPT-2.7B gates (G0–G3) and LLaVA-1.5-7B G5 that still undermines main-track confidence, and turn each into a concrete improvement target.  
**Not a gate verdict.** Gate write-ups stay in [`CVPR2027_GATE_LOG.md`](CVPR2027_GATE_LOG.md) and [`results/cvpr2027/g5/G5_GATE_WRITEUP_LLAVA.md`](results/cvpr2027/g5/G5_GATE_WRITEUP_LLAVA.md).  
**Glossary:** same file § Glossary — use those names only (`total_leak`, `stale_leak`, `foreign_leak`, `correct_route`, `abstain`, reliability hit / only-own, `image_swap`, `hard_negative`, `emits_y_i` / `emits_y_j`).  
**Last updated:** 2026-10-11.

Confidence rule of thumb for this project:

| Tier | Meaning |
|---|---|
| **P0 — gate / claim blocker** | Without a win or an honest demotion of the claim, a main-track story is fragile |
| **P1 — strengthener** | Needed for reviewer resilience; paper can survive with a clear limitation if unfixed |
| **P2 — polish / optional** | Nice for completeness; do not block freeze |

---

## Snapshot: where confidence is thin today

| Area | BLIP-2-OPT-2.7B | LLaVA-1.5-7B |
|---|---|---|
| Leak ↓ with Rel held | **Yes** on natural VLKEB / EVQA (G0/G2) | **No** — Rel hit collapses to ~0–1% (G5 WEAK) |
| Absolute hard_negative under pair-closed | **Weak** (~84–86% DEK; M6) | Curve exists; Rel already dead — hn win not usable as main claim |
| Partner routing (`correct_route`) | Partial (VLKEB swap OK; EVQA swap ≈ abstain) | ≈ 0% |
| Only-own | Strong on BLIP-2 DEK (~100%) | ~0–1% |
| Answer-level (`emits_y_i` / `emits_y_j`) | Routing ≠ answer; DEK often worse on stale answers | Near-zero Rel EM; no rescue |
| Second-backbone transfer | Plan A holds on BLIP-2 | Leak transfers; Rel does not |
| Other editors (G4) | Not started | — |

---

## P0 — must improve (or demote the claim)

### 1. Absolute hard_negative total_leak under pair-closed pools (BLIP-2) — M6

| | |
|---|---|
| **Source** | G3 Bad; Active flag HIGH; Reviewer 1 #9 |
| **Evidence** | Pair-closed VLKEB DEK hard_negative `total_leak` **~84–86%** at n∈{100,300,1000} vs L0 100%; natural-side DEK was **44%** (partner-absent flattery) |
| **Why it hurts** | M1 says headline leakage is hard_negative `total_leak`. Absolute ~84% is a weak main-track headline even if relative vs L0 holds |
| **Improvement target** | Cut pair-closed hn `total_leak` **substantially** (direction: toward natural-side ~40% or better) **without** dropping Rel hit below ~99% |
| **Candidate tries** (discuss before implementing) | Tighter vision scope / top-K; train-tuned CARML-style; margin/abstain on hn only; hard-neg–aware training; keep DEK as Plan A for swap+Rel until a candidate beats it on hn without killing Rel |
| **Status** | Deferred — after G5 / Oct 15 |

### 2. LLaVA reliability collapse under DEK (second backbone)

| | |
|---|---|
| **Source** | G5 LLaVA write-up Bad; provisional WEAK |
| **Evidence** | VLKEB Rel hit **79%→1%**; EVQA **71%→1%**; multi-seed Rel hit **0.6±0.5%**; only-own ~0–1%; swap gain is mostly **abstain** (64–87%), `correct_route` ≈ 0% |
| **Why it hurts** | G5 question is leak↓ **without** Rel collapse. Today LLaVA is a leak-only / AND-style stress, not a BLIP-2-style second primary table |
| **Improvement target** | On LLaVA short ladder (or a disclosed stronger one): Rel hit **≫ 1%** (aim BLIP-2-like ≥90% if claiming parity) **and** keep meaningful swap/hn `total_leak` cut vs L0 |
| **Candidate tries** (discuss before implementing) | Train-time Rel constraint; softer dual gate / τ; backbone-specific fuse; longer L0/S2 budget; diagnose vision vs text gate fail mode first |
| **Status** | Open HIGH; method change needs discussion |

### 3. Answer-level does not follow routing (both backbones)

| | |
|---|---|
| **Source** | G0 Bad; Active flag HIGH; Reviewer 1 #3 / AI #1 |
| **Evidence** | BLIP-2 VLKEB image_swap: DEK `emits_y_i` **21%** vs L0 **8%** (worse); `emits_y_j` ~0 both. LLaVA: Rel EM ~1%, no partner-answer rescue |
| **Why it hurts** | Reviewers can dismiss routing wins as “internal bookkeeping” if free-gen still emits the stale fact |
| **Improvement target** | On image_swap free-gen: lower `emits_y_i` vs L0 and/or raise `emits_y_j` under a constrained decode protocol — or **never claim** answer-level rescue |
| **Status** | Deferred — later; paper must stay honest until fixed |

### 4. Dual-backbone apples-to-apples table incomplete

| | |
|---|---|
| **Source** | G5 Carry-forward Phase C; Reviewer 1 #8 / R2 #5 |
| **Evidence** | LLaVA extract done; BLIP-2 same-protocol re-extract **not** filled in `G5_RESULTS_TABLES.md` §4–5 |
| **Why it hurts** | Cannot close G5 board or put a dual table in the paper without metric-drift risk |
| **Improvement target** | Phase C: re-extract BLIP-2 L0/S2/DEK with the same scripts; fill dual columns; finalize PASS/WEAK/FAIL |
| **Status** | **Calendar — due 15 Oct** |

---

## P1 — strengthen main-track resilience

### 5. Easy-VQA image_swap win is abstain, not correct_route (BLIP-2) — M2

| | |
|---|---|
| **Source** | G0 Bad; Active flag medium |
| **Evidence** | EVQA DEK swap `total_leak` 0.7% via **99% abstain**; `correct_route` 0.3% vs L0 **87%** |
| **Why it hurts** | Looks like `always_reject` unless coverage / `correct_route` sit beside the leak number |
| **Improvement target** | Either raise EVQA swap `correct_route` under DEK, or freeze paper framing: four-outcome + coverage; never sell EVQA swap as partner-routing success |
| **Status** | Framing decided (M2); method win optional |

### 6. Residual hard_negative is mostly foreign_leak (BLIP-2)

| | |
|---|---|
| **Source** | G0/G1 Bad |
| **Evidence** | VLKEB DEK hn residual **~94% foreign**; hn still leaks on **43% of unique-prompt** probes — not a text-gate story (dup share of residual 47% < 70% bar) |
| **Why it hurts** | Absolute hn weakness (P0 #1) is vision/scope/collision territory; wrong fix wastes the freeze window |
| **Improvement target** | Instrument + method try aimed at **foreign** fires on hard_negative (aligns with M1); do not center paper on “fix the text gate” |
| **Status** | Diagnosis done; method tied to M6 |

### 7. LLaVA hard_negative on Easy-VQA barely moves

| | |
|---|---|
| **Source** | G5 Bad |
| **Evidence** | EVQA hn `total_leak` 100→**91** (−9 pp) while swap 100→13 via abstain |
| **Why it hurts** | Dataset-dependent; overselling EVQA hn on LLaVA invites review pushback |
| **Improvement target** | Same as P0 #2 if Rel is restored; until then cite VLKEB hn + four-way swap, not EVQA hn as lead |

### 8. Other editors under the same leak protocol (G4)

| | |
|---|---|
| **Source** | Reviewer 2 #2; G4 not started |
| **Evidence** | None yet |
| **Why it hurts** | Leak taxonomy may look LiveEdit-specific |
| **Improvement target** | Run same protocol on ≥1 external editor; report Rel + four-way outcomes |
| **Status** | Planned Oct 17–28 |

### 9. Multi-seed / multi-order coverage gaps

| | |
|---|---|
| **Source** | G3 Bad; AI #9; supervisor R1 #6 |
| **Evidence** | BLIP-2 pair-closed curve = one seed / one order / VLKEB only. LLaVA VLKEB 5-seed **done** (E4); EVQA pair-closed + BLIP-2 5-seed still thin |
| **Why it hurts** | Scaling + Rel claims need variance, not a single lucky order |
| **Improvement target** | Authorized 5-seed on BLIP-2 DEK_and (G2); optional EVQA pair-closed; more edit orders if GPU allows |

### 10. Soft-weight / fuse_coe story still messy for paper

| | |
|---|---|
| **Source** | G0 Bad (proxy); M3 applied in G2; G5 LLaVA τ rows are real `fuse_coe` and reverse the swap win (DEK 36 vs L0 13 at τ=0.1; [`G5_METRIC_AUDIT.md`](results/cvpr2027/g5/G5_METRIC_AUDIT.md)) |
| **Evidence** | Weight-aware at τ=0.1 shrinks BLIP-2 VLKEB swap gain to **−4 pp**; LLaVA τ rows are still not `fuse_coe` |
| **Why it hurts** | Easy for a reviewer to say binary overstates L0 |
| **Improvement target** | Paper tables: binary headline + `fuse_coe` weight-aware where logged; **never** put vision-logit proxy τ as fuse coefficients (M3) |

---

## P2 — polish / optional before freeze

| # | Item | Source | Action |
|---|---|---|---|
| 11 | Untuned CARML-style row looks weak | G2 Bad | Label “untuned” or train-tune `carml_scope_tau` before naming CARML |
| 12 | MSCKE public-code check | G2 carry; R2 #1 | ≤1 day when spare + related-work paragraph |
| 13 | G0 vs G2 number mismatch (swap 20% vs 48%) | G2 Bad | One sample protocol in paper tables; footnote subsample shift |
| 14 | DINOv2 hn difficulty terciles / human pair audit | G1 deferred; AI #7 | Optional if residual story is challenged |
| 15 | Prototype-score definition underspecified | AI #6 | Writing only — method section |
| 16 | Paper routing-spec side-by-side + leak-vs-Rel plot | G2 / R2 #4 / AI #3 | Writing — controls already on disk |
| 17 | Terminology / spelling / hyperparams in draft | AI #11; AAAI criticism | Glossary lock already; clean AuthorKit draft |

---

## Suggested improvement order (GPU-aware)

Do **not** implement method changes silently — discuss first (gate-log rule).

1. **Phase C** — BLIP-2 same-protocol re-extract → close dual table / G5 board (**calendar**).  
2. **Decide LLaVA paper role** — second-backbone **stress/limitation** vs invest in Rel-recovery try (P0 #2).  
3. **M6 method try** — absolute pair-closed hard_negative on BLIP-2 (P0 #1); this is the main Plan-A strengthener.  
4. **Answer-level** — only after routing+Rel are stable on the backbone you claim (P0 #3).  
5. **G4 other editors** — protocol transfer (P1 #8).  
6. **Seeds / EVQA closed / MSCKE / writing** — fill variance and related work (P1 #9, P2).

---

## What *not* to chase as the main fix

From G1: residual is **not** text-gate / prompt-duplication dominated (<70% bar; fired foreign shares query prompt only ~41%). Do not spend the freeze window on “fix the text gate” as the primary contribution.

From G2: `always_reject` and `L0_and` without AVBT are **controls**, not methods — they kill Rel. Any LLaVA Rel-recovery try must beat that trap.

---

## Artifact pointers

| Backbone | Bad / numbers live in |
|---|---|
| BLIP-2 | Gate log G0–G3; `doc/results/cvpr2027/g0/`, `g1/`, `g2/`, `g3/` |
| LLaVA | [`g5/G5_GATE_WRITEUP_LLAVA.md`](results/cvpr2027/g5/G5_GATE_WRITEUP_LLAVA.md), [`g5/G5_RESULTS_TABLES.md`](results/cvpr2027/g5/G5_RESULTS_TABLES.md) |
| Reviewer map | [`AAAI_TO_CVPR_SUPERVISOR_PROGRESS.md`](AAAI_TO_CVPR_SUPERVISOR_PROGRESS.md) |
| Active flags | Gate log § Active flags |

---

*End of improvement backlog. Update this file when a P0/P1 item closes or a method try is decided.*
