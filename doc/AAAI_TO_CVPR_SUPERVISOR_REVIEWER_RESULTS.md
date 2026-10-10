# AAAI reviewer points → work → related results (supervisor)

**Date:** 9 Oct 2026  
**Audience:** supervisor  
**Purpose:** Same reviewer task list as the brief, with an extra **Related results** column so each point sits next to its numbers.  
**Companion:** narrative overview in `doc/AAAI_TO_CVPR_SUPERVISOR_BRIEF.md`.

**Method:** Dual-Expert Keys (adversarial visual-binding training + dual vision-and-text gate).  
**Baseline:** original LiveEdit routing.

All percentages unless noted. Lower is better for leak; higher is better for correct route, only-own, and reliability hit.

---

## Snapshot

| Reviewer | Main concern | Progress |
|---|---|---|
| Reviewer 1 | Wrong metrics; missing ablations and scale | Mostly addressed; backbone / other editors / absolute hard-negative still open |
| Reviewer 2 | Related methods; metric may be LiveEdit-only; Rel/Gen/Loc flat | Ablations ready; external editors and writing still open |
| AI reviewer | Conflicting targets; missing controls; lifelong claim; writing nits | Experiments largely done; writing and multi-seed still open |

---

## Reviewer 1

| # | Reviewer point | Our work | Related results | Status |
|---|---|---|---|---|
| 1 | Own edit firing on a wrong image was not counted as an error | Split leakage into stale (own edit), foreign (other wrong edit), correct route (partner), abstain. Headline leak = stale + foreign only. | **Stale leak (own edit on wrong image):** Dual-Expert Keys ≤ 9% on VLKEB and Easy-VQA. Baseline VLKEB swap: stale 6%, foreign **82%** → total leak 88%. Problem is foreign leak, not stale. | Done |
| 2 | Partner edit firing on its own image was wrongly counted as an error | Count partner-only fire as “correct route”, not a leak. | **Easy-VQA image-swap:** baseline correct route **87%**, true total leak only **11.3%** (old “~96–98% false activation” mixed these). Dual-Expert Keys: total leak **0.7%**, but mostly by abstaining (99%), not by routing to the partner (correct route 0.3%). | Done |
| 3 | “Swap accuracy higher is better” conflicted with the safety goal | Retired swap-accuracy. Report emit-stale vs emit-partner answers on swapped images. | **Emit stale answer on image-swap:** VLKEB baseline **8%** vs Dual-Expert Keys **21%** (worse). Easy-VQA 8.3% vs 9.3%. Emit partner answer ≈ 0–1% for both. Metric fixed; answer-level win **not** shown. | Metric done; claim open |
| 4 | Components not isolated (vision-only, text-only, original LiveEdit, score fusion) | Full routing-rule comparison on frozen checkpoints. | **VLKEB hard-neg / swap leak / reliability:** original LiveEdit 93 / 88 / 100; AND alone 5 / 0 / **50**; always-reject 0 / 0 / **0**; sum 71 / 68 / 100; untuned CARML-style 100 / 99 / 100; **Dual-Expert Keys 44 / 48 / 100**. | Done |
| 5 | Binary false-activation may overstate baseline leak | Weight-aware re-score; log real fusion weights; headline = hard-negative total leak. | **Weight-aware baseline VLKEB swap leak** at soft threshold 0.1: **~24%** (vs binary 88%). Hard-negative: VLKEB 91→**52** (−39 pp); Easy-VQA 96.7→**9.3** (−87 pp) — robust to counting. | Done |
| 6 | Only 100 edits; pool size / edit order untested | Pair-closed pools (partners in pool); curve at 100 / 300 / 1000 edits on VLKEB. | **Swap total leak (baseline → Dual-Expert Keys):** 90→**19** (n=100), 97→**15** (n=300), 100→**23** (n=1000). Reliability ≥ **99%**. Partner correct route under Dual-Expert Keys ~62–70%. Extra orders / Easy-VQA curve optional. | Mostly done |
| 7 | Repeated prompts / visual collisions may explain residual leak | Stratify residual Dual-Expert Keys leaks by prompt duplication. | **~47%** of residual leaks on duplicated prompts (below 70% bar for blaming text gate). VLKEB Dual-Expert Keys hard-neg residual: **94% foreign**. Fired foreign expert shares query prompt only **41%** overall. | Partial |
| 8 | Only one backbone and one editor | LLaVA load check; Dual-Expert Keys LLaVA training on stronger GPU. Other editors not started. | **LLaVA:** load smoke passed (~14.5 GB). Training in progress; decision due **15 Oct**. Other editors: no numbers yet. | In progress / pending |
| 9 | Residual leak still looks high | Document residual mix; flag absolute hard-negative under pair-closed pools. | **Pair-closed hard-neg total leak (Dual-Expert Keys):** **84 / 86 / 84%** at n=100 / 300 / 1000 (baseline 100%). Relative win holds; absolute claim weak. Natural (non-closed) Dual-Expert Keys hard-neg was 44% — flattered by missing partners. | Open (flagged) |

---

## Reviewer 2

| # | Reviewer point | Our work | Related results | Status |
|---|---|---|---|---|
| 1 | CARML and MSCKE not discussed or compared | Untuned CARML-style row in routing table; MSCKE code check still to do; related-work for paper. | **VLKEB:** untuned CARML-style swap leak **99%**, hard-neg **100%**, vs Dual-Expert Keys **48% / 44%** at same reliability 100%. MSCKE: no result yet. | Partial |
| 2 | False-positive rate may be method-specific to LiveEdit-style retrieval | Same leak protocol planned on LEMoE / RECIPE; SERAC scope-classifier fallback. | No numbers yet (starts after backbone decision). | Pending |
| 3 | Only an author-built LiveEdit-style baseline | Original LiveEdit as baseline row; AND-without-adversarial-training as control. | **VLKEB:** original LiveEdit swap/hard-neg **88 / 93**, reliability 100. AND alone: leak near 0 but reliability **50%**. Dual-Expert Keys: **48 / 44** at reliability **100%**. | Done |
| 4 | No gain on standard Rel / Gen / Loc | Reframe to leak-versus-reliability (safety vs coverage). | **Hard-neg total leak @ reliability hit ≥99%:** Dual-Expert Keys **44%** (VLKEB) / **9%** (Easy-VQA) vs baseline **93% / 97%**. Reliability only-own: baseline ~16–17% → Dual-Expert Keys **~100%**. Rel/Gen/Loc not claimed as improved. | Evidence ready; writing pending |
| 5 | Single BLIP-2 backbone | Same LLaVA second-backbone track as Reviewer 1. | Load smoke passed; Dual-Expert Keys LLaVA train/eval pending (**15 Oct**). | In progress |

---

## AI reviewer

| # | Reviewer point | Our work | Related results | Status |
|---|---|---|---|---|
| 1 | Equation / FPR / Swap defined conflicting targets | One glossary; four-way outcomes; swap-accuracy retired; emit-stale / emit-partner defined. | Honest VLKEB: baseline swap total leak **88%** → Dual-Expert Keys **20%**; hard-neg **91%** → **52%**. Answer-level still favours baseline on emit-stale (8% vs 21%). | Routing done |
| 2 | Missing routing recall, false-negative style rates, always-reject control | Always-reject control; reliability hit / only-own / miss / partner correct route. | **Always-reject:** abstain 100%, reliability **0%**. Dual-Expert Keys Easy-VQA swap: abstain **99%**, reliability **99%** — not the same as a dead gate. VLKEB Dual-Expert Keys only-own **100%** vs baseline **16%**. | Mostly done |
| 3 | Novelty boundary vs LiveEdit / CARML unclear | Empirical routing comparison; paper routing-spec table still to write. | Dual-Expert Keys beats original LiveEdit (88/93 leak), sum (68/71), untuned CARML-style (99/100) on VLKEB swap/hard-neg at high reliability. | Experiments done; writing open |
| 4 | No Dual-Expert Keys without adversarial training; no vision-only / text-only | Ran those ablations. | Vision-only ≈ original LiveEdit (88/93). Text-only: swap leak **99%**. AND without adversarial training: reliability **50%** (VLKEB) / **61%** (Easy-VQA). Both pieces needed. | Done |
| 5 | Lifelong scaling claimed but not tested | Pool-size curve 100 / 300 / 1000, pair-closed. | Swap leak Dual-Expert Keys **19 / 15 / 23%** vs baseline **90 / 97 / 100%**. Hard-neg Dual-Expert Keys stuck ~**84–86%** vs baseline 100%. | Done on VLKEB (one seed) |
| 6 | Prototype score underspecified | Deferred to method write-up. | No new number — writing only. | Pending (writing) |
| 7 | Hard-negative difficulty strata + human check of pairs | De-prioritised after residual analysis. | No strata table yet. | Deferred |
| 8 | Explain VLKEB vs Easy-VQA residual gap | Short residual analysis. | VLKEB prompt-duplication rate **40%** with Dual-Expert Keys residual swap/hard-neg **20% / 52%**. Easy-VQA duplication **2.7%** with residual **0.7% / 9.3%**. Correlation yes; text-gate mechanism not proven. | Done (brief) |
| 9 | Multi-seed reuses the same checkpoints | Authorise five evaluation seeds; plan three training seeds. | Not run yet — no multi-seed table. | Pending |
| 10 | Hard-negative missing from VLKEB main tables | Hard-negative total leak is the headline leakage number. | VLKEB hard-neg: baseline **91%** → Dual-Expert Keys **52%** (honest); pair-closed curve Dual-Expert Keys ~**84%** (absolute weak). Numbers ready for main table. | Numbers done |
| 11 | Writing nits (spelling, undefined terms, figure seeds, hyperparameters) | Locked one glossary and naming rules. | No experimental number — draft cleanup still due. | Partial |

---

## Summary result tables (shared across points)

**How to read the flags**

| Flag | Meaning |
|---|---|
| **BEFORE (AAAI)** | What the AAAI submission used or claimed |
| **AFTER (CVPR work)** | What we measured after addressing the reviews |

---

### 1. Leakage metric — BEFORE vs AFTER side by side

| When | Dataset | Method | What was reported | Image-swap “error / leak” ↓ | Hard-negative leak ↓ | Notes |
|---|---|---|---|---|---|---|
| **BEFORE (AAAI)** | VLKEB / Easy-VQA | Dual-Expert Keys vs baseline | Opaque “false activation” / FPR (mixed partner-correct fires into the error) | Claimed ~**96–98%** fixed on Easy-VQA-style swap | Hard-negative **not** in VLKEB main table | Metric overstated the win |
| **AFTER (CVPR work)** | VLKEB | Baseline | Honest total leak (stale + foreign only) | **88.0** | **91.0** | Partner correct route 8.0% (not counted as leak) |
| **AFTER (CVPR work)** | VLKEB | Dual-Expert Keys | Same honest metric | **20.0** | **52.0** | Partner correct route 34.0%; only-own **100%** |
| **AFTER (CVPR work)** | Easy-VQA | Baseline | Same honest metric | **11.3** | **96.7** | Partner correct route **87.0%** — this was most of the old “FPR” |
| **AFTER (CVPR work)** | Easy-VQA | Dual-Expert Keys | Same honest metric | **0.7** | **9.3** | Mostly abstain (99%), not partner routing (0.3%) |

Full AFTER breakdown (same honest metric):

| When | Dataset | Method | Image-swap total leak ↓ | Partner correct route ↑ | Hard-negative total leak ↓ | Reliability only-own ↑ |
|---|---|---|---|---|---|---|
| **AFTER** | VLKEB | Baseline | 88.0 | 8.0 | 91.0 | 16.0 |
| **AFTER** | VLKEB | Dual-Expert Keys | **20.0** | 34.0 | **52.0** | **100.0** |
| **AFTER** | Easy-VQA | Baseline | 11.3 | **87.0** | 96.7 | 17.3 |
| **AFTER** | Easy-VQA | Dual-Expert Keys | **0.7** | 0.3 | **9.3** | **99.0** |

---

### 2. Routing controls (VLKEB) — whole table is AFTER

| When | What existed | Detail |
|---|---|---|
| **BEFORE (AAAI)** | Not available | Only Dual-Expert Keys vs a custom baseline; no vision-only / text-only / AND-alone / always-reject / CARML-style rows |
| **AFTER (CVPR work)** | Full comparison below | Numbers from the new routing-rule table |

| When | Rule | Image-swap total leak ↓ | Hard-negative total leak ↓ | Reliability hit ↑ |
|---|---|---|---|---|
| **AFTER** | Original LiveEdit (baseline) | 88 | 93 | 100 |
| **AFTER** | AND gate only (no adversarial training) | 0 | 5 | **50** |
| **AFTER** | Always reject (control) | 0 | 0 | **0** |
| **AFTER** | Score-sum fusion | 68 | 71 | 100 |
| **AFTER** | Untuned CARML-style | 99 | 100 | 100 |
| **AFTER** | **Dual-Expert Keys** | **48** | **44** | **100** |

---

### 3. Lifelong pool size (VLKEB, pair-closed) — whole table is AFTER

| When | What existed | Detail |
|---|---|---|
| **BEFORE (AAAI)** | Not available | Reported at ~100 edits only; no 300 / 1000 curve; partners often missing from the pool |
| **AFTER (CVPR work)** | Curve below | Pair-closed pools (every probe partner is also in the pool) |

| When | Pool size | Baseline swap / hard-neg ↓ | Dual-Expert Keys swap / hard-neg ↓ | Dual-Expert Keys correct route ↑ | Dual-Expert Keys reliability ↑ |
|---|---|---|---|---|---|
| **AFTER** | 100 | 90 / 100 | **19 / 84** | 68 | 100 |
| **AFTER** | 300 | 97 / 100 | **15 / 86** | 70 | 99.7 |
| **AFTER** | 1000 | 100 / 100 | **23 / 84** | 62 | 99.2 |

---

### 4. Answer-level on image-swap — AFTER only (new metric)

| When | What existed | Detail |
|---|---|---|
| **BEFORE (AAAI)** | Swap-accuracy (direction conflicted with safety goal) | Used as a headline; higher-is-better conflicted with “do not fire on wrong image” |
| **AFTER (CVPR work)** | Emit-stale / emit-partner rates below | Swap-accuracy retired |

| When | Dataset | Method | Emit stale answer ↓ | Emit partner answer ↑ |
|---|---|---|---|---|
| **AFTER** | VLKEB | Baseline | 8.0 | 0.0 |
| **AFTER** | VLKEB | Dual-Expert Keys | **21.0** (worse than baseline) | 0.0 |
| **AFTER** | Easy-VQA | Baseline | 8.3 | 0.7 |
| **AFTER** | Easy-VQA | Dual-Expert Keys | 9.3 | 1.3 |

**Note:** This AFTER table does **not** favour Dual-Expert Keys — keep it as a limitation, not a win.

---

## Still pending

| Item | Timing |
|---|---|
| Finish LLaVA Dual-Expert Keys train + eval | Decision by **15 Oct** |
| Other editors under the same leak protocol | After backbone decision |
| Method retry on absolute hard-negative (pair-closed) | After backbone decision |
| Five evaluation seeds (+ training seeds if time) | After configs locked |
| MSCKE code check + related-work / routing-spec write-up | Late Oct |
| Paper writing checkpoint | **27 Oct** |
| Experiment freeze | **5 Nov** |

---

**Related docs:** `doc/AAAI_TO_CVPR_SUPERVISOR_BRIEF.md` · `doc/AAAI_TO_CVPR_REVIEWER_HANDOFF.md` · `doc/CVPR2027_GATE_LOG.md`
