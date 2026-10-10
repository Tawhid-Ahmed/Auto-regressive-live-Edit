# AAAI reviews → CVPR 2027 progress (supervisor brief)

**Date:** 9 Oct 2026  
**Audience:** supervisor  
**Purpose:** For each AAAI reviewer comment — what they asked, what we did, and what the numbers show.  
**Venue target:** CVPR 2027 (register 10 Nov, submit 16 Nov 2026).

**Method under test:** Dual-Expert Keys (adversarial visual-binding training + dual vision-and-text gate).  
**Baseline:** original LiveEdit routing (no dual gate, no adversarial binding).

**Overall:** Metric honesty, controlled ablations, and lifelong pool-size evidence are largely done on BLIP-2. Still open: second backbone (in progress), other editors, multi-seed runs, paper writing, and absolute hard-negative leak under honest pair-closed pools.

---

## Snapshot

| Reviewer | Main concern | Progress |
|---|---|---|
| Reviewer 1 | Metrics were wrong; ablations and scale missing | Mostly addressed; backbone / other editors / absolute hard-negative still open |
| Reviewer 2 | Novelty vs related methods; metric may be LiveEdit-specific; standard Rel/Gen/Loc flat | Ablations and reframing ready; external editors and related-work write-up still open |
| AI reviewer | Conflicting targets; missing controls; lifelong claim untested; writing nits | Experimental controls largely done; writing and multi-seed still open |

---

## Reviewer 1

| # | Reviewer point | Our work | Result / output | Status |
|---|---|---|---|---|
| 1 | Own edit firing on a wrong image was not counted as an error | Split leakage into: stale leak (own edit fires), foreign leak (some other wrong edit fires), correct route (partner edit fires), abstain (nothing fires). Headline = stale + foreign only. | Own-edit (stale) leak is small for both methods (≤ 9%). Baseline’s real problem is foreign leak, not stale. | Done |
| 2 | Partner edit firing on its own image was wrongly counted as an error | Same four-way breakdown; partner-only fire is now “correct route”, not a leak. | On Easy-VQA image-swap, baseline “~96–98% false activation” was largely partner-correct routing (**87%**). True baseline swap leak is only **11.3%**. | Done |
| 3 | “Swap accuracy higher is better” conflicted with the safety goal | Retired swap-accuracy. Report instead: how often the model emits the stale edited answer vs the partner answer on a swapped image. | Metric fixed. Story still weak: Dual-Expert Keys emits the stale answer **more** often on VLKEB (21% vs baseline 8%). Do not claim answer-level rescue yet. | Metric done; claim open |
| 4 | Components not isolated (vision-only, text-only, original LiveEdit, score fusion) | Ran a full routing-rule comparison on frozen checkpoints: original LiveEdit, vision-only, text-only, AND gate, OR, sum, CARML-style fusion, always-reject, and Dual-Expert Keys. | Dual-Expert Keys is the only high-reliability / low-leak point. AND gate alone (without adversarial training) collapses reliability (~50%). Always-reject kills reliability to 0%. | Done |
| 5 | Binary false-activation may overstate baseline leak | Re-scored with weight-aware counting; logged real fusion weights. Shifted paper headline to hard-negative total leak. | Weight-aware baseline VLKEB swap leak falls to ~24% (vs old ~88–96% story). Hard-negative gain stays large and robust. | Done |
| 6 | Only 100 edits; pool size / edit order untested | Built pair-closed pools (every probe partner is also in the pool) and ran 100 / 300 / 1000 edits on VLKEB. | Dual-Expert Keys swap leak stays ~15–23% while baseline rises to 100% at 1000 edits. Reliability stays ≥ 99%. Extra edit orders and Easy-VQA curve still optional. | Mostly done |
| 7 | Repeated prompts / visual collisions may explain residual leak | Stratified residual Dual-Expert Keys leaks by prompt duplication. | Only ~47% of residual leaks sit on duplicated prompts — below the bar for blaming the text gate. Residual is mostly foreign fire on hard-negatives. | Partial (enough to de-prioritize) |
| 8 | Only one backbone and one editor | Second backbone (LLaVA) load checked; training moved to a stronger GPU. Other editors (LEMoE / RECIPE / SERAC) not started yet. | LLaVA load smoke passed. Dual-Expert Keys LLaVA training in progress; decision due **15 Oct**. Other-editor protocol still pending. | In progress / pending |
| 9 | Residual leak still looks high | Documented residual composition; flagged absolute hard-negative under pair-closed pools as an open method gap. | With partners honestly in-pool, Dual-Expert Keys hard-negative total leak stays ~**84–86%** (better than baseline 100%, but weak as an absolute claim). Method retry deferred until after the backbone decision. | Open (flagged) |

---

## Reviewer 2

| # | Reviewer point | Our work | Result / output | Status |
|---|---|---|---|---|
| 1 | CARML and MSCKE not discussed or compared | Added an untuned CARML-style fusion row to the routing comparison. MSCKE public-code check still to do (≤ 1 day). Related-work write-up for the paper. | Untuned CARML-style is much leakier than Dual-Expert Keys on our table. MSCKE not checked yet. | Partial |
| 2 | False-positive rate may be method-specific to LiveEdit-style retrieval | Plan: run the same leakage protocol on other pool-based editors; if full routing IDs are hard, fall back to SERAC’s scope classifier. | Not started (planned after the backbone decision). | Pending |
| 3 | Only an author-built LiveEdit-style baseline | Use original LiveEdit config as the baseline row; show AND-without-adversarial-training as a control. | Original LiveEdit is the baseline. AND alone is an abstain trap (low reliability) — shows Dual-Expert Keys needs both pieces. | Done |
| 4 | No gain on standard Rel / Gen / Loc | Reframe the paper around leak-versus-reliability (safety vs coverage), not “we improve Rel”. | Dual-Expert Keys: hard-negative total leak **44%** (VLKEB) / **9%** (Easy-VQA) with reliability hit ≥ **99%**, vs baseline hard-negative **93% / 97%**. Plot for the paper still to draw. | Evidence ready; writing pending |
| 5 | Single BLIP-2 backbone | Same second-backbone track as Reviewer 1. | In progress on stronger GPU; due 15 Oct. | In progress |

---

## AI reviewer

| # | Reviewer point | Our work | Result / output | Status |
|---|---|---|---|---|
| 1 | Equation / FPR / Swap defined conflicting targets | One glossary; four-way leakage outcomes; swap-accuracy retired; emit-stale / emit-partner answers defined. | Routing metrics now consistent with the safety goal. Answer-level still does not favour Dual-Expert Keys. | Routing done |
| 2 | Missing routing recall, false-negative style rates, always-reject control | Added always-reject control; report reliability hit, only-own precision, miss, and partner correct-route. | Always-reject: 100% abstain, 0% reliability — cleanly separates Dual-Expert Keys’ Easy-VQA abstain-heavy swap from a dead gate. | Mostly done |
| 3 | Novelty boundary vs LiveEdit / CARML unclear | Empirical routing comparison; written side-by-side routing-spec table still needed in the paper. | Dual-Expert Keys beats LiveEdit, sum, and untuned CARML-style on leak-vs-reliability. Paper table not written yet. | Experiments done; writing open |
| 4 | No Dual-Expert Keys without adversarial training; no vision-only / text-only | Ran those ablations. | Vision-only ≈ original LiveEdit. Text-only is very leaky. AND without adversarial training kills reliability. Both pieces matter. | Done |
| 5 | Lifelong scaling claimed but not tested | Pool-size curve at 100 / 300 / 1000 with pair-closed partners. | Advantage holds as the pool grows (see results table below). | Done on VLKEB (one seed) |
| 6 | Prototype score underspecified | Deferred to method write-up (concrete definition from the retrieval code). | Not written yet. | Pending (writing) |
| 7 | Hard-negative difficulty strata + human check of pairs | Optional after residual analysis; de-prioritised because residual was not text-gate dominated. | Not run; revisit only if time remains. | Deferred |
| 8 | Explain VLKEB vs Easy-VQA residual gap | Short residual analysis. | VLKEB has more duplicated prompts and higher residual; Easy-VQA is cleaner. Not causal enough to centre the paper on it. | Done (brief) |
| 9 | Multi-seed reuses the same checkpoints | Authorise five evaluation seeds for passed configs; plan three BLIP-2 training seeds before freeze. | Authorised; not run yet. | Pending |
| 10 | Hard-negative missing from VLKEB main tables | Make hard-negative total leak the headline leakage number; keep image-swap as a four-outcome second column. | Numbers exist; paper table layout still to write. | Numbers done |
| 11 | Writing nits (spelling, undefined terms, figure seed labels, hyperparameters unreported) | Locked one glossary and naming rules for all future text/tables. | Draft cleanup still needed before submission. | Partial |

---

## Before → after results

### 1. Honest leakage (replaces AAAI “false activation”)

Lower is better for total leak. Higher is better for correct route and only-own precision. All figures in %.

| Dataset | Method | Image-swap total leak | Partner correct route (not a leak) | Hard-negative total leak | Reliability only-own |
|---|---|---|---|---|---|
| VLKEB | Baseline (AAAI-era story) | counted as ~high “false activation” | mixed into the error | not in main table | not foregrounded |
| VLKEB | Baseline (honest) | 88.0 | 8.0 | 91.0 | 16.0 |
| VLKEB | Dual-Expert Keys (honest) | **20.0** | 34.0 | **52.0** | **100.0** |
| Easy-VQA | Baseline (honest) | 11.3 | **87.0** | 96.7 | 17.3 |
| Easy-VQA | Dual-Expert Keys (honest) | **0.7** | 0.3 (mostly abstains) | **9.3** | **99.0** |

**Plain reading:** Dual-Expert Keys cuts foreign leak and nearly perfects “only the right edit fires” on the correct image. The old Easy-VQA swap win was overstated because partner-correct routing was counted as failure.

### 2. Routing controls (who actually wins)

| Rule | Image-swap total leak (VLKEB) | Hard-negative total leak (VLKEB) | Reliability hit |
|---|---|---|---|
| Original LiveEdit | 88 | 93 | 100 |
| AND gate only (no adversarial training) | 0 | 5 | **50** (too much abstain) |
| Always reject | 0 | 0 | **0** |
| Score-sum fusion | 68 | 71 | 100 |
| Untuned CARML-style fusion | 99 | 100 | 100 |
| **Dual-Expert Keys** | **48** | **44** | **100** |

**Plain reading:** Low leak with collapsed reliability is not a win. Dual-Expert Keys is the useful operating point.

### 3. Lifelong pool size (pair-closed partners)

| Pool size | Baseline swap / hard-neg leak | Dual-Expert Keys swap / hard-neg leak | Dual-Expert Keys partner correct route | Dual-Expert Keys reliability |
|---|---|---|---|---|
| 100 | 90 / 100 | **19 / 84** | 68 | 100 |
| 300 | 97 / 100 | **15 / 86** | 70 | 99.7 |
| 1000 | 100 / 100 | **23 / 84** | 62 | 99.2 |

**Plain reading:** Swap leakage stays controlled as the lifelong pool grows. Absolute hard-negative leakage under honest partners remains high (~84%) — open limitation.

### 4. Answer-level on image-swap (do not over-claim)

| Dataset | Method | Emits stale edited answer (lower better) | Emits partner answer (higher better) |
|---|---|---|---|
| VLKEB | Baseline | 8.0 | 0.0 |
| VLKEB | Dual-Expert Keys | **21.0** (worse) | 0.0 |
| Easy-VQA | Baseline | 8.3 | 0.7 |
| Easy-VQA | Dual-Expert Keys | 9.3 | 1.3 |

**Plain reading:** Better routing does not yet mean safer free-form answers on a swapped image.

---

## Still pending (for CVPR)

| Item | Why it matters | Timing |
|---|---|---|
| Finish LLaVA Dual-Expert Keys train + eval | Answers “only one backbone” | Decision by **15 Oct** (stronger GPU) |
| Other editors under the same leak protocol | Answers “metric is LiveEdit-specific” | After backbone decision |
| Method retry on absolute hard-negative (pair-closed) | Headline hard-negative claim is weak in absolute terms | After backbone decision |
| Five evaluation seeds (+ training seeds if time) | Reviewer multi-seed concern | After configs are locked |
| MSCKE code check + related-work / routing-spec write-up | Reviewer 2 / AI novelty boundary | ≤ 1 day code check; writing by late Oct |
| Paper writing checkpoint | Abstract, contributions, every main table with real numbers | **27 Oct** |
| Experiment freeze | No new threads after this | **5 Nov** |

---

## What we can say today (safe claims)

1. The AAAI leakage metric mixed correct partner routing into the error rate; we fixed that openly.  
2. Dual-Expert Keys reduces foreign expert activation and raises routing precision on the correct image.  
3. Under controlled ablations, Dual-Expert Keys sits on the useful leak-vs-reliability frontier; always-reject and AND-alone do not.  
4. On VLKEB, the lifelong swap advantage holds from 100 to 1000 edits when partners are honestly in the pool.  

## What we should not say yet

1. Dual-Expert Keys improves free-form answers on swapped images.  
2. Dual-Expert Keys is strong in absolute terms on hard-negatives under pair-closed pools (~84% leak).  
3. Results transfer to a second backbone or other editors (still running / not started).  

---

**Internal detail log (gates, code names):** `doc/CVPR2027_GATE_LOG.md`  
**Engineer handoff (IDs, file paths):** `doc/AAAI_TO_CVPR_REVIEWER_HANDOFF.md`
