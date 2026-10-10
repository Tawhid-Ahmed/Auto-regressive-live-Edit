# AAAI → CVPR 2027 — Reviewer Handoff

**Date:** 2026-10-09  
**Purpose:** Map each AAAI reviewer's points to what we fixed, what remains, and before/after numbers for the CVPR 2027 resubmission.  
**Canonical terms:** Glossary in [`CVPR2027_GATE_LOG.md`](CVPR2027_GATE_LOG.md).  
**Plan source:** `cvpr_2027_resubmission_plan` (reviewer coverage matrix).  
**Living status:** [`CVPR2027_GATE_LOG.md`](CVPR2027_GATE_LOG.md) status board.

**Source note.** Full OpenReview export PDFs are not in this repo. Reviewer IDs and point lists below come from the resubmission plan's coverage matrix (forum fragments `4umz`, `xKQ6`, plus the AAAI **AI reviewer**). Point wording is paraphrased to match that matrix; if you paste the official reviews, we can align quotes 1:1.

**Current track:** Plan A (DEK = AVBT + `--routing_rule and`). Gates **G0–G3 PASS**. **G5** (LLaVA second backbone) is active on a stronger GPU. **G4 / G6 / 5-seed / M6** still open.

---

## Status at a glance

| Reviewer | Core theme | Worked | Left / deferred |
|---|---|---|---|
| **4umz** | Metric honesty, ablations, scale, scope | Taxonomy, Swap retired, routing ablations, pool-size curve, residual note | Second backbone (G5), other editors (G4), absolute pair-closed hn (M6), multi-seed |
| **xKQ6** | Novelty vs CARML/MSCKE; FPR not method-specific; Rel/Gen/Loc flat | Routing table + LiveEdit baseline row; untuned CARML-style row; leak-vs-reliability framing | MSCKE code check; G4 multi-editor FPR; paper related-work polish |
| **AI reviewer** | Conflicting targets, missing controls, novelty boundary, lifelong claim, paper nits | Taxonomy aligned with Eq. 7; `always_reject`; vision/text/AND ablations; G3 curve; hn in main numbers; glossary | Prototype-score writeup; DINOv2 terciles / 50-pair hand-check; training seeds; writing (G6) |

---

## Reviewer 1 — `4umz`

### Points → status

| ID | Reviewer point | Status | Where | Before (AAAI) | After (CVPR work) |
|---|---|---|---|---|---|
| W1 / Q1 | Own expert `e_i` firing on a mismatched image not counted as error | **Done** | G0 `stale_leak` | Not separated; folded into opaque "false activation" | Counted explicitly. VLKEB/EVQA DEK stale ≤ 9% everywhere (G0) |
| W1 / Q2 | Partner expert `e_j` counted as error even when it matches the shown image | **Done** | G0 `correct_route` | EVQA "false activation" ~96–98% included **87%** partner-correct fires as errors | Four-outcome taxonomy. EVQA L0 swap: true `total_leak` **11.3%**, `correct_route` **87%** |
| W2 / Q3 | Swap-accuracy "higher is better" was backwards / conflicting | **Done (metric)** / **open (story)** | G0 `emits_y_i` / `emits_y_j`; Swap-accuracy retired | Swap-acc used as headline | Replaced by emit rates. **Caveat:** DEK does **not** yet win at answer level (VLKEB `emits_y_i` DEK **21%** vs L0 **8%**) — HIGH FLAG deferred |
| W3 / Q4 | Components not isolated (visual-only, text-only, original LiveEdit, score fusion) | **Done** | G2 routing-rule table | Only DEK vs author baseline | Full table: `liveedit`, `vision_only`, `text_only`, `and`, `or`, `sum`, `carml`, `always_reject`, DEK_and. DEK on Pareto front (G2 PASS) |
| W3 | Binary FPR may overstate baseline leak | **Done (measured)** | G0 weight-aware + G2 `fuse_coe` | Binary-only ~96% FPR story | Weight-aware VLKEB L0 swap leak **24%** at tau=0.1; real fuse coeffs logged (M3). Headline shifted to hard-negative `total_leak` (M1) |
| W4 | 100 edits only; pool size / edit order untested | **Partial** | G3 | n=100 only | Pair-closed VLKEB curve n∈{100,300,1000}: DEK swap `total_leak` **19 / 15 / 23%** vs L0 **90 / 97 / 100%**. Extra edit orders + EVQA curve **left** |
| W4 | Repeated prompts / visual-key collisions | **Partial** | G1 | Unanalyzed | Prompt-dup explains **47%** of residual (below 70% threshold) → not text-gate story. DINOv2 terciles + 50-pair hand-check **deferred** |
| W4 / Q5 | One backbone, one editor | **In progress / not started** | G5 / G4 | BLIP-2 + LiveEdit only | G5: LLaVA load smoke PASS; S2@i500 local; DEK train moved to stronger GPU (decision due **Oct 15**). G4 (LEMoE / RECIPE / SERAC) **not started** |
| W4 | Residual leak still high (~46%) | **Partial** | G1, M6 | Opaque residual | Residual composition known (mostly **foreign** on hn). Absolute pair-closed hn DEK still **~84–86%** → **M6 HIGH FLAG deferred** |

### 4umz — verdict for handoff

Metric and ablation objections are largely closed by G0–G2. Scale objection is closed enough for a VLKEB seed-42 curve (G3). Scope (backbone + other editors) and absolute hard-negative under pair-closed partners are the live risks.

---

## Reviewer 2 — `xKQ6`

### Points → status

| ID | Reviewer point | Status | Where | Before (AAAI) | After (CVPR work) |
|---|---|---|---|---|---|
| CARML / MSCKE | Not discussed or compared | **Partial** | G2 + paper | Absent | Untuned `carml` row in G2 (leaky vs DEK). **MSCKE public-code check still not done** (≤1 day). Spec table / related work for writing (G6) |
| FPR method-specific | Metric only makes sense for LiveEdit-style retrieval | **Not started** | G4 | n/a for SERAC etc. | Plan: run taxonomy on `lemoe_vl` / `recipe_vl`; SERAC scope-classifier fallback. Window Oct 17–28 |
| Author-built baseline only | No original LiveEdit config as control | **Done** | G2 | Custom DEK baseline | `L0_liveedit` is the baseline row; `L0_and` without AVBT shown as abstain trap (rel hit 50% VLKEB) |
| No gain on Rel / Gen / Loc | Method looks null on standard metrics | **Reframed (evidence ready)** | G2 | Flat Rel story | Lead with **leak-vs-reliability**: DEK_and hn `total_leak` **44%** (VLKEB) / **9%** (EVQA) at rel hit **≥99%**, vs L0 hn **93% / 97%**. Paper plot still for G6 |
| Single BLIP-2 backbone | Same as 4umz Q5 | **In progress** | G5 | One backbone | Same as G5 row above |

### xKQ6 — verdict for handoff

Strongest remaining ask is **external comparison** (MSCKE + G4 editors) and **writing** the leak-vs-reliability claim. Method isolation and LiveEdit baseline are done.

---

## Reviewer 3 — AI reviewer

### Points → status

| ID | Reviewer point | Status | Where | Before (AAAI) | After (CVPR work) |
|---|---|---|---|---|---|
| W1 | Eq. 7 vs FPR vs Swap define conflicting targets | **Done (routing)** | G0 taxonomy; Swap retired | Conflicting directions | `stale_leak` / `foreign_leak` / `correct_route` / `abstain` + `emits_y_*`. G2b abstain sweep **not triggered** (G0 not WEAK) |
| W2 | No routing recall, FNR, prototype-selection rate, always-reject control | **Mostly done** | G2 | Missing controls | `always_reject` row; rel hit / only_own / miss in taxonomy; partner `correct_route`. Prototype-selection rate as named metric still for paper |
| W3 | Novelty boundary vs LiveEdit / CARML unclear | **Partial** | G2 + paper | Unclear | Empirical Pareto: DEK beats LiveEdit / sum / untuned carml. Written side-by-side routing spec **left for G6** |
| W4 | No DEK-without-AVBT; no visual-only / text-only | **Done** | G2 | Missing ablations | `L0_and` (AND w/o AVBT), `vision_only`, `text_only`, `DEK_liveedit` all run |
| W5 | Lifelong scaling claimed but not tested | **Done (VLKEB s42)** | G3 | Claim without curve | Pair-closed curve PASS; advantage holds to n=1000. More orders / EVQA **optional left** |
| W6 | Prototype score underspecified | **Left (writing)** | Paper | Underspecified | Concrete definition from `retrieve_moes` still needed in method section |
| Suggestion | Hard-negative difficulty strata + human verification | **Deferred** | G1 optional | — | Skipped after G1 "unexplained" branch; revisit only if spare time |
| Suggestion | Residual analysis for VLKEB vs EVQA gap | **Done (brief)** | G1 | — | Dup correlation usable in one sentence; do not center paper on it |
| Minor | Multi-seed reuses shared checkpoints | **Authorized, not run** | freeze plan | Shared ckpt seeds | 5 eval seeds authorized after G2 PASS; 3 BLIP-2 **training** seeds still pending |
| Minor | Hard-negative missing from VLKEB main tables | **Done (numbers)** | G0/G3 + M1 | Swap-only main table | hn is headline leakage number; paper table layout still G6 |
| Minor | `m`/`alpha` unreported; Eq. 6 index reuse; "Defination"; "false-cross"; Fig 3 seeds; "negative ablation" criterion | **Partial** | glossary + paper | Nits in AAAI draft | Terminology locked (glossary + `.cursor/rules/cvpr2027-terminology.mdc`). Draft cleanup **left for G6** |

### AI reviewer — verdict for handoff

Experimental control requests from W1–W5 are largely met. Remaining AI-reviewer debt is **writing clarity** (prototype score, routing-spec table, terminology nits) and **deferred** strata / training seeds.

---

## Headline results — before vs after

### A. Leakage metric (addresses 4umz W1/Q1–Q2, AI W1)

AAAI-era "false activation" mixed `stale_leak`, `foreign_leak`, and `correct_route`. Honest G0 (binary, seed 42):

| Run | swap `total_leak` ↓ | of which `correct_route` ↑ (not a leak) | hn `total_leak` ↓ | rel `only_own` ↑ |
|---|---|---|---|---|
| VLKEB L0 | 88.0 | 8.0 | 91.0 | 16.0 |
| VLKEB DEK | **20.0** (−68 pp) | 34.0 | **52.0** (−39 pp) | **100.0** |
| EVQA L0 | 11.3 | **87.0** | 96.7 | 17.3 |
| EVQA DEK | **0.7** | 0.3 (99% abstain) | **9.3** (−87 pp) | **99.0** |

**Takeaway:** DEK's real win is cutting **foreign** leak and raising routing precision; EVQA swap "96% FPR" was mostly partner-correct routing, not DEK magic.

### B. Routing ablations (addresses 4umz W3, xKQ6 baseline, AI W2/W4)

G2 VLKEB (seeded subsample; compare within-table):

| Rule | swap `total_leak` | hn `total_leak` | rel hit |
|---|---|---|---|
| L0_liveedit | 88 | 93 | 100 |
| L0_and (no AVBT) | 0 | 5 | **50** (abstain trap) |
| L0_always_reject | 0 | 0 | **0** |
| L0_sum | 68 | 71 | 100 |
| L0_carml (untuned) | 99 | 100 | 100 |
| **DEK_and** | **48** | **44** | **100** |

**Takeaway:** DEK is on the leak-vs-reliability Pareto front (G2 PASS). AND without AVBT is not a method.

### C. Lifelong pool size (addresses 4umz W4, AI W5)

Pair-closed VLKEB seed 42:

| n | L0 swap / hn `total_leak` | DEK_and swap / hn | DEK `correct_route` | DEK rel hit |
|---|---|---|---|---|
| 100 | 90 / 100 | **19 / 84** | 68 | 100 |
| 300 | 97 / 100 | **15 / 86** | 70 | 99.7 |
| 1000 | 100 / 100 | **23 / 84** | 62 | 99.2 |

**Takeaway:** Swap scaling claim **survives**. Absolute pair-closed hn (~84%) is the open method gap (**M6**).

### D. Answer-level (addresses 4umz W2 — still weak)

| Run | `emits_y_i` ↓ | `emits_y_j` ↑ |
|---|---|---|
| VLKEB L0 | 8.0 | 0.0 |
| VLKEB DEK | **21.0** (worse) | 0.0 |
| EVQA L0 / DEK | 8.3 / 9.3 | 0.7 / 1.3 |

**Takeaway:** Do not claim answer-level rescue until revisited. Routing ≠ safer freegen answers yet.

---

## Pending for CVPR review readiness

Ordered by calendar / plan priority.

### Active now

| Item | Reviewer link | Due | Notes |
|---|---|---|---|
| **G5** LLaVA DEK train + L0/S2/DEK eval | 4umz Q5, xKQ6 / AI backbone | **Oct 15** | Handoff: [`doc/results/cvpr2027/g5/README.md`](results/cvpr2027/g5/README.md). Local S2@i500 kept; DEK on stronger GPU from L0 |
| Remote GPU jobs only for long train/eval | — | ongoing | Local PC: no long runs |

### Next gates / experiments

| Item | Reviewer link | Window | Status |
|---|---|---|---|
| **G4** other editors (LEMoE / RECIPE / SERAC scope) | xKQ6 FPR transfer; 4umz Q5 | Oct 17–28 | Not started |
| **M6** pair-closed hn method try | 4umz residual; M1 headline risk | after G5 | Deferred — HIGH FLAG |
| 5-seed eval (authorized configs) | AI minor; plan freeze | after gates | Not started |
| 3 BLIP-2 training seeds | AI minor | before freeze | Not started |
| EVQA pair-closed + more edit orders | 4umz W4 | optional | Not started |
| MSCKE code availability check | xKQ6 | ≤1 day | Not done |
| CARML tau train-tune (or label "untuned") | xKQ6 / G2 carry | optional | Untuned row only |
| `emits_y_i` revisit / constrained decode | 4umz W2 | deferred | HIGH FLAG |
| G1 DINOv2 terciles + 50-pair hand-check | AI suggestion | deferred | Skip unless spare |

### Writing / paper (G6, Oct 27)

| Item | Reviewer link |
|---|---|
| Abstract + contribution list with finished-table numbers only | all |
| Leak-vs-reliability main plot from G2 | xKQ6 Rel/Gen/Loc |
| Four-outcome taxonomy + retire Swap-accuracy / FPR wording | 4umz, AI |
| Side-by-side routing spec (LiveEdit / CARML / DEK / sum) | xKQ6, AI W3 |
| Prototype score definition from `retrieve_moes` | AI W6 |
| Fix AAAI nits: Defination, false-cross, Eq. 6, `m`/`alpha`, Fig 3 seeds | AI minor |
| Limitations: single-backbone if G5 fails; absolute hn; routing ≠ answer | honesty |

### Explicitly not claiming yet

- Answer-level rescue on image-swap (`emits_y_i` / `emits_y_j`).
- Absolute hard-negative safety under pair-closed partners (M6).
- Multi-backbone until G5 decision.
- Cross-editor FPR until G4.
- CARML beat / MSCKE parity until checked and (if needed) tuned.

---

## Gate board (quick)

| Gate | Verdict | Reviewer coverage |
|---|---|---|
| G0 honest re-scoring | **PASS** | 4umz W1–W3; AI W1; hn in main numbers |
| G1 residual source | complete — unexplained / not text-gate | 4umz W4 residual; AI residual suggestion |
| G2 routing rules | **PASS** | 4umz W3; xKQ6 baseline/CARML; AI W2/W4 |
| G2b margin/abstain | not triggered | AI W1 (if WEAK) |
| G3 pool-size curve | **PASS** (+ M6 high flag) | 4umz W4; AI W5 |
| G4 other editors | not started | xKQ6 FPR; 4umz Q5 |
| G5 second backbone | **in progress** (remote) | 4umz Q5; xKQ6/AI backbone |
| G6 writing | not started (Oct 27) | all paper nits + framing |

---

## Pointers

| Artifact | Path |
|---|---|
| Gate log + glossary | `doc/CVPR2027_GATE_LOG.md` |
| G0 taxonomy | `doc/results/cvpr2027/g0/` |
| G2 routing tables | `doc/results/cvpr2027/g2/` |
| G3 pool-size curve | `doc/results/cvpr2027/g3/` |
| G5 remote handoff | `doc/results/cvpr2027/g5/README.md` |
| Terminology rule | `.cursor/rules/cvpr2027-terminology.mdc` |
| Resubmission plan | `~/.cursor/plans/cvpr_2027_resubmission_plan_be405ff4.plan.md` |

If the three official OpenReview texts are exported (HTML/PDF), drop them under `doc/reviews/aaai/` and this handoff can be updated with verbatim quotes under each reviewer ID.
