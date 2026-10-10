# AAAI → CVPR 2027 handoff

**Purpose.** One-page status for a supervisor / next session: what each AAAI reviewer asked, what we fixed, what numbers changed, what is still open.

**Canonical detail.** Gate decisions, glossary, and full tables live in [`CVPR2027_GATE_LOG.md`](CVPR2027_GATE_LOG.md). This file is the reviewer-facing checklist only.

**As of.** 2026-10-09. Current track: **G5 LLaVA on stronger GPU** (local DEK train killed early; see `doc/results/cvpr2027/g5/README.md`). Decision due **Oct 15**. Experiment freeze **Nov 5**. CVPR register **Nov 10**, submit **Nov 16**.

**Status legend.** Done = evidence in gate log / results. Deferred = known, intentionally later. Pending = not started. Partial = code/analysis done, paper claim not locked.

---

## Snapshot

| Gate | Verdict | Date |
|---|---|---|
| G0 honest re-scoring | **PASS** (Plan A / DEK continues) | Oct 4 |
| G1 residual leak source | complete — not text-gate; move on | Oct 4 |
| G2 routing ablation | **PASS** — DEK on Pareto front | Oct 4 |
| G2b margin sweep | **not triggered** (stale was low) | — |
| G3 pool-size curve | **PASS** — advantage holds; **HIGH** hn absolute ~84% (M6) | Oct 6 |
| G4 other editors | **pending** | — |
| G5 second backbone | **in progress** (remote DEK + eval) | due Oct 15 |
| G6 writing checkpoint | **pending** | due Oct 27 |

**Before (AAAI headline):** VLKEB image-swap “FPR” 96% → 46%; Swap↑ treated as a win; Rel ~94%.

**After (honest taxonomy, G0 binary):** VLKEB swap **total_leak** 88% → **20%**; hard-neg **91% → 52%**; Rel hit 100%. AAAI “FPR” mixed in `correct_route` (8% VLKEB / 87% EVQA baseline).

---

## Reviewer 4umz (rating 4)

| # | Point | Status | What we did / result | Left |
|---|---|---|---|---|
| W1 / Q1 | Own expert e_i on swapped image not counted as error (stale-edit) | **Done** | Taxonomy adds `stale_leak`. VLKEB: L0 6% / DEK 8% (stale was never the main baseline failure) | Report clearly in paper |
| W1 / Q2 | Partner e_j counted as error though image matches | **Done** | `correct_route` outcome. AAAI FPR overstated: EVQA L0 “leak” was mostly 87% correct_route | Paper four-outcome swap table |
| W2 / Q3 | Swap↑ / keep y_i is wrong direction | **Done (metric)** / **open (story)** | Swap-accuracy retired. `emits_y_i` / `emits_y_j` measured. **Routing ≠ answer:** VLKEB DEK `emits_y_i` **21% vs L0 8%** (worse) | Do not claim answer-level rescue until revisited |
| W3 / Q4 | Components not isolated (patch, text, AND) | **Done** | G2 full routing table: liveedit, vision_only, text_only, and, or, sum, carml, always_reject; L0_and vs DEK_and | 5-seed of passed rows (authorized, not run) |
| W3 | Binary FPR may overstate soft near-zero weight | **Partial** | Weight-aware + real `fuse_coe` logging (M3). Proxy tau demotes VLKEB swap gain; hn gain robust | Paper: lead with hn; don’t oversell proxy weight-aware |
| W4 | Only 100 edits; no pool size / order | **Done (VLKEB)** | G3 pair-closed n∈{100,300,1000}: DEK swap advantage holds (~71–82 pp); Rel ≥99.2% | EVQA pair-closed; more edit orders (optional) |
| W4 | Repeated prompts / visual collisions | **Partial** | G1: residual **not** text-gate dominated (47% on dup prompts, &lt;70% threshold) | DINOv2 terciles / 50-pair hand-check deferred |
| W4 / Q5 | One backbone, one editor | **Partial** | G5 LLaVA started; G4 not started | Finish G5 by Oct 15; G4 Oct 17–28 |
| W4 | 46% residual still high | **Open / HIGH** | Honest hn under pair-closed partners: DEK **~84–86%** (M6). Relative win vs L0 100%, absolute weak | Method try after G5 (tighter scope / hn-aware train) |

---

## Reviewer xKQ6 (rating 3)

| # | Point | Status | What we did / result | Left |
|---|---|---|---|---|
| Novelty | CARML / MSCKE not discussed or compared | **Partial** | G2 has untuned **CARML-style** fusion row (leaky vs DEK). Related-work rewrite not written | MSCKE code check (≤1 day); train-tune CARML tau or label “untuned”; paper RW |
| Metric | FPR only for expert-pool editors | **Pending** | — | G4: LEMoE / RECIPE / SERAC scope logging |
| Baselines | Only author LiveEdit-style | **Done** | G2 uses original LiveEdit rule as `L0_liveedit`; AND alone collapses Rel (50% VLKEB / 61% EVQA) — AVBT needed | Paper side-by-side routing spec |
| Metrics | No gain on Rel / Gen / Loc | **Reframed** | Lead with leak-vs-reliability; Rel held | Paper framing + Gen/Loc in final tables |
| Scale | Single BLIP-2 | **In progress** | G5 LLaVA smoke PASS; S2@i500 ready; DEK needs remote GPU | Remote DEK train + L0/S2/DEK eval by Oct 15 |

---

## AI Reviewer

| # | Point | Status | What we did / result | Left |
|---|---|---|---|---|
| W1 | Eq.7 / FPR / Swap↑ conflict | **Done** | Unified taxonomy; Swap↑ gone; abstain vs leak vs correct_route | Paper Sec. problem |
| W2 | No recall / FNR / always-reject | **Done** | `always_reject` + `correct_route` / abstain / Rel hit in G2 | Coverage column next to EVQA swap (M2) |
| W3 | Novelty vs LiveEdit / CARML unclear | **Partial** | Spec exists in results; not in paper draft yet | Side-by-side table in paper |
| W4 | No DEK-without-AVBT; no vision/text-only | **Done** | `L0_and` (AND w/o AVBT) + vision_only / text_only in G2 | — |
| W5 | Lifelong claim, fixed pool 100 | **Done (VLKEB)** | G3 PASS | EVQA curve optional |
| W6 | Prototype score underspecified | **Pending (paper)** | Code exists; prose not rewritten | Paper method |
| Suggestion | Hard-neg difficulty strata + human check | **Deferred** | Skipped after G1 “unexplained” | Optional if time |
| Suggestion | Residual VLKEB vs EVQA | **Done (brief)** | G1: don’t center paper on text-gate fix | One short residual paragraph |
| Minor | Multi-seed reuses ckpts | **Pending** | 5-seed authorized after G2 PASS | Run after G5 / freeze priority |
| Minor | Hard-neg missing from VLKEB main table | **Done (data)** | hn is M1 headline; G0/G2/G3 all report hn | Paper tables |
| Minor | m, α unreported; Eq.6 index; “Defination”; “false-cross”; Fig.3 seeds; “negative ablation” | **Pending (paper)** | Glossary + terminology rule in place | Copyedit in G6 |

---

## Before → after (numbers to cite)

### AAAI vs honest G0 (same seed-42 logs, binary taxonomy)

| Metric | AAAI claim | After G0 (honest) |
|---|---|---|
| VLKEB swap “FPR” / total_leak | 96% → 46% | **88% → 20%** (AAAI counted some `correct_route` as fail) |
| VLKEB hard-neg total_leak | not in main table | **91% → 52%** |
| EVQA hard-neg total_leak | 96.8% → 2.7% | **96.7% → 9.3%** (same story, honest split) |
| EVQA swap | presented as big win | Baseline already **87% correct_route**; DEK wins by **99% abstain**, not better routing |
| Stale (own expert on wrong image) | ignored by FPR | ≤9% everywhere — not the main bug |
| Rel | ~94% held | Rel **hit** 100% / 99%; DEK **only_own** 100% / 99% vs L0 ~16–17% |
| Swap↑ | +7.4 pp “win” | **Retired.** `emits_y_i`: DEK **worse** on VLKEB (21 vs 8) |

### G2 (routing isolation, seed 42) — VLKEB hn / Rel

| Rule | hn total_leak | Rel hit |
|---|---|---|
| L0 liveedit | 93% | 100% |
| L0 text_only / or / carml-style | ~100% | ~100% |
| L0 and (no AVBT) | 5% | **50%** (abstain trap) |
| L0 always_reject | 0% | **0%** |
| **DEK_and** | **44%** | **100%** |

### G3 (pair-closed pool size, VLKEB)

| n | L0 swap total | DEK swap total | L0 hn | DEK hn | DEK Rel |
|---|---|---|---|---|---|
| 100 | 90% | **19%** | 100% | **84%** | 100% |
| 300 | 97% | **15%** | 100% | **86%** | 99.7% |
| 1000 | 100% | **23%** | 100% | **84%** | 99.2% |

Scaling claim: **survives on swap**. Absolute hn under honest partners: **HIGH FLAG (M6)**.

---

## Still pending (ordered)

### Active now
1. **G5** — remote: train LLaVA DEK from L0; eval L0 / S2 / DEK + taxonomy; decide by **Oct 15**. Local 16GB DEK train was killed (~2/500 it). Handoff: `doc/results/cvpr2027/g5/README.md`.

### After G5 (deferred flags)
2. **M6** — pair-closed hn ~84%: method try (tighter vision scope / hn-aware train / margin on hn only) without killing Rel.
3. **`emits_y_i`** — routing win ≠ answer win; either constrain decode / partner route or limit paper claim.
4. **G4** — LEMoE / RECIPE / SERAC scope (answers “FPR is method-specific”).
5. **5-seed** + optional EVQA pair-closed / more edit orders.
6. **MSCKE** public-code check; CARML tau train-tune or “untuned” label.
7. **G6 (Oct 27)** — abstract + filled tables; cut claims without numbers.
8. **Paper** — taxonomy diagram, routing-spec table, RW (CARML/MSCKE/…), retire Swap-accuracy, copyedit minors; freeze **Nov 5**; register **Nov 10**; submit **Nov 16**.

### Do not claim yet
- Answer-level rescue from DEK.
- Absolute hard-neg safety under pair-closed partners as “solved”.
- EVQA swap as “better routing” (it is mostly abstain).
- CARML as a fair published baseline until tau is tuned or labeled untuned.
- Multi-editor / second-backbone generality until G4 / G5 finish.

---

## Artifact map

| Item | Path |
|---|---|
| Gate log + glossary | `doc/CVPR2027_GATE_LOG.md` |
| G0 taxonomy | `doc/results/cvpr2027/g0/leakage_taxonomy_{vlkeb,evqa}_s42.{json,md}` |
| G0 answer-level | `doc/results/cvpr2027/g0/emits_y_seed42.json` |
| G1 residual | `doc/results/cvpr2027/g1/residual_leak_source_s42.json` |
| G2 routing table | `doc/results/cvpr2027/g2/routing_table_{vlkeb,evqa}_s42.{json,md}` |
| G3 pool curve | `doc/results/cvpr2027/g3/pool_size_curve_vlkeb_s42.{json,md}` |
| G5 remote handoff | `doc/results/cvpr2027/g5/README.md` |
| Plan | `.cursor/plans/cvpr_2027_resubmission_plan_be405ff4.plan.md` |
| Terminology rule | `.cursor/rules/cvpr2027-terminology.mdc` |
