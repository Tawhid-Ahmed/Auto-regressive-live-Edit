# AAAI → CVPR progress (supervisor)

**Writing only?** = Yes means paper/writing work only (no new experiment needed for that point). No = needs experiment or method work.  
**What’s left** = remaining work for that point (— if nothing left).

## Review list and progress status

### Reviewer 1

| # | Review point | Status | Writing only? | What’s left |
|---|---|---|---|---|
| 1 | Own edit on wrong image not counted as error | Done | No | — |
| 2 | Partner edit on matching image wrongly counted as error | Done | No | — |
| 3 | Swap-accuracy direction conflicted with safety goal | Measured; no win yet | No | Improve answer-level (cut stale answers on swap); then we can claim it |
| 4 | Components not isolated (vision / text / LiveEdit / fusion) | Done | No | — |
| 5 | Binary false-activation may overstate baseline leak | Done | No | — |
| 6 | Only 100 edits; pool size untested | Partly done | No | Optional: more edit orders + Easy-VQA pool-size curve |
| 7 | Repeated prompts / visual collisions may explain residual | Partly done | No | Optional: hard-neg difficulty strata + small human check of pairs |
| 8 | Only one backbone and one editor | In progress | No | Finish LLaVA Dual-Expert Keys train + eval; then other editors |
| 9 | Residual leak still high | Measured; no win yet | No | Method retry to cut absolute hard-neg leak under honest pools |

### Reviewer 2

| # | Review point | Status | Writing only? | What’s left |
|---|---|---|---|---|
| 1 | CARML / MSCKE not compared | Partly done | No | MSCKE public-code check + related-work write-up |
| 2 | Leak metric may be LiveEdit-specific | Not started | No | Run same leak protocol on other editors |
| 3 | Only author-built baseline | Done | No | — |
| 4 | No gain on Rel / Gen / Loc | Partly done | Yes | Write paper section / plot leading with leak vs reliability |
| 5 | Single BLIP-2 backbone | In progress | No | Finish LLaVA Dual-Expert Keys train + eval (due 15 Oct) |

### AI reviewer

| # | Review point | Status | Writing only? | What’s left |
|---|---|---|---|---|
| 1 | Conflicting targets (FPR / Swap / equation) | Partly done | No | Answer-level still weak; do not claim swap-answer rescue until fixed |
| 2 | Missing always-reject / routing controls | Partly done | Yes | Paper polish: present controls clearly in tables/text |
| 3 | Novelty vs LiveEdit / CARML unclear | Partly done | Yes | Write side-by-side routing-spec table in the paper |
| 4 | No Dual-Expert Keys without adversarial training; no vision/text-only | Done | No | — |
| 5 | Lifelong scaling claimed but not tested | Done | No | — |
| 6 | Prototype score underspecified | Not started | Yes | Write concrete prototype-score definition in method section |
| 7 | Hard-negative difficulty strata + human check | Later | No | Optional later if time remains |
| 8 | Explain VLKEB vs Easy-VQA residual gap | Done | No | — |
| 9 | Multi-seed reuses same checkpoints | Not started | No | Run five evaluation seeds (and training seeds if time) |
| 10 | Hard-negative missing from VLKEB main tables | Done | No | — |
| 11 | Writing nits (terms, spelling, hyperparameters) | Partly done | Yes | Clean draft: spelling, terms, figure seed labels, report hyperparameters |

---

## Details — Reviewer 1

| # | Reviewer point | Our work | Related results | Status | Writing only? | What’s left |
|---|---|---|---|---|---|---|
| 1 | Own edit on wrong image not counted | Four-way leak: stale / foreign / correct route / abstain | Dual-Expert Keys stale ≤ 9%. Baseline VLKEB swap: foreign **82%** → total leak 88% | Done | No | — |
| 2 | Partner edit counted as error | Partner-only fire = correct route | Easy-VQA baseline correct route **87%**, true leak **11.3%**. Dual-Expert Keys leak **0.7%** mainly via abstain (99%) | Done | No | — |
| 3 | Swap-accuracy conflicted with safety | Stopped using swap-accuracy; report how often stale vs partner answers appear | VLKEB stale answer: baseline **8%** vs Dual-Expert Keys **21%** (worse). Partner answer ~0–1% | Measured; no win yet | No | Improve answer-level (cut stale answers on swap); then we can claim it |
| 4 | Components not isolated | Full routing-rule comparison | VLKEB hard-neg/swap/rel: LiveEdit 93/88/100; AND alone 5/0/**50**; always-reject 0/0/**0**; **Dual-Expert Keys 44/48/100** | Done | No | — |
| 5 | Binary FPR may overstate leak | Soft-weight re-score; lead with hard-neg leak | Soft-weight baseline VLKEB swap ~**24%**. Hard-neg: 91→**52** (VLKEB), 96.7→**9.3** (Easy-VQA) | Done | No | — |
| 6 | Pool size untested | Partners kept in pool; curve at 100/300/1000 | Swap leak baseline→ours: 90→**19**, 97→**15**, 100→**23**. Rel ≥ 99% | Partly done | No | Optional: more edit orders + Easy-VQA pool-size curve |
| 7 | Prompt / visual collisions | Checked whether residual leaks sit on repeated prompts | ~**47%** on repeated prompts (not enough to blame text gate). Hard-neg residual **94% foreign** | Partly done | No | Optional: hard-neg difficulty strata + small human check of pairs |
| 8 | One backbone / one editor | LLaVA train on stronger GPU; other editors later | Load check passed. Train/eval due **15 Oct**. No LLaVA Dual-Expert Keys eval numbers yet. Other editors: no numbers | In progress | No | Finish LLaVA Dual-Expert Keys train + eval; then other editors |
| 9 | Residual still high | Marked absolute hard-neg under honest pools as open gap | Honest-pool hard-neg Dual-Expert Keys ~**84–86%** (baseline 100%). Relative win only | Measured; no win yet | No | Method retry to cut absolute hard-neg leak under honest pools |

---

## Details — Reviewer 2

| # | Reviewer point | Our work | Related results | Status | Writing only? | What’s left |
|---|---|---|---|---|---|---|
| 1 | CARML / MSCKE missing | Added untuned CARML-style row; MSCKE check still needed | CARML-style VLKEB swap/hard-neg **99/100** vs Dual-Expert Keys **48/44**. MSCKE: none yet | Partly done | No | MSCKE public-code check + related-work write-up |
| 2 | Metric may be LiveEdit-only | Same leak protocol on other editors (planned) | No numbers yet | Not started | No | Run same leak protocol on other editors |
| 3 | Author-built baseline only | Original LiveEdit + AND-alone controls | LiveEdit 88/93 @ rel 100; AND alone rel **50%**; Dual-Expert Keys **48/44** @ **100%** | Done | No | — |
| 4 | No Rel/Gen/Loc gain | Numbers ready; paper should lead with leak vs reliability | Hard-neg @ high reliability: ours **44% / 9%** vs baseline **93% / 97%**. Only-own ~16%→**~100%** | Partly done | Yes | Write paper section / plot leading with leak vs reliability |
| 5 | Single backbone | LLaVA second backbone | In progress; due **15 Oct**. No LLaVA Dual-Expert Keys eval numbers yet | In progress | No | Finish LLaVA Dual-Expert Keys train + eval (due 15 Oct) |

---

## Details — AI reviewer

| # | Reviewer point | Our work | Related results | Status | Writing only? | What’s left |
|---|---|---|---|---|---|---|
| 1 | Conflicting metric targets | One naming list; four-way outcomes; swap-accuracy removed | VLKEB swap 88→**20**; hard-neg 91→**52**. Stale answers still worse (8→21) | Partly done | No | Answer-level still weak; do not claim swap-answer rescue until fixed |
| 2 | Missing always-reject / controls | Always-reject control + reliability / only-own / correct route | Always-reject: rel **0%**. Ours Easy-VQA abstain 99% but rel **99%**. Only-own 16→**100%** | Partly done | Yes | Paper polish: present controls clearly in tables/text |
| 3 | Novelty vs LiveEdit / CARML | Ran comparison; paper side-by-side table still needed | Beats LiveEdit (88/93), sum (68/71), CARML-style (99/100) | Partly done | Yes | Write side-by-side routing-spec table in the paper |
| 4 | No ablation without adversarial / vision / text | Ran those rows | Vision≈LiveEdit; text-only swap **99%**; AND alone rel **50%/61%** | Done | No | — |
| 5 | Lifelong scaling untested | Pool curve with partners in pool (VLKEB, one seed) | Swap ours **19/15/23** vs baseline **90/97/100**. Hard-neg ours ~**84–86%** | Done | No | — |
| 6 | Prototype score underspecified | Needs clear write-up in the paper | — | Not started | Yes | Write concrete prototype-score definition in method section |
| 7 | Hard-neg strata + human check | Put off after residual check | — | Later | No | Optional later if time remains |
| 8 | VLKEB vs Easy-VQA residual | Short residual check | VLKEB repeated-prompt rate 40%, residual 20/52; Easy-VQA 2.7%, residual 0.7/9.3 | Done | No | — |
| 9 | Multi-seed shared checkpoints | Five eval seeds planned | Not run yet | Not started | No | Run five evaluation seeds (and training seeds if time) |
| 10 | Hard-neg missing from main tables | Hard-neg is now the main leak number | Honest 91→**52**; honest-pool curve ~**84%** (still weak in absolute terms) | Done | No | — |
| 11 | Writing nits | Naming rules locked | Paper draft cleanup still due | Partly done | Yes | Clean draft: spelling, terms, figure seed labels, report hyperparameters |

---

## Summary results (BEFORE vs AFTER)

Lower is better for leak. Higher is better for correct route and only-own. All figures in %.

### Leakage metric

| When | Dataset | Method | Swap leak ↓ | Correct route ↑ | Hard-neg leak ↓ | Only-own ↑ |
|---|---|---|---|---|---|---|
| **BEFORE (AAAI)** | Easy-VQA-style | Dual-Expert Keys | Claimed ~96–98% “false activation” fixed (mixed correct route into error) | — | Not in VLKEB main table | — |
| **AFTER** | VLKEB | Baseline | 88.0 | 8.0 | 91.0 | 16.0 |
| **AFTER** | VLKEB | Dual-Expert Keys | **20.0** | 34.0 | **52.0** | **100.0** |
| **AFTER** | Easy-VQA | Baseline | 11.3 | **87.0** | 96.7 | 17.3 |
| **AFTER** | Easy-VQA | Dual-Expert Keys | **0.7** | 0.3 | **9.3** | **99.0** |

### Routing controls (VLKEB) — AFTER only (BEFORE: not run)

| When | Rule | Swap leak ↓ | Hard-neg leak ↓ | Rel hit ↑ |
|---|---|---|---|---|
| **AFTER** | Original LiveEdit | 88 | 93 | 100 |
| **AFTER** | AND alone | 0 | 5 | **50** |
| **AFTER** | Always reject | 0 | 0 | **0** |
| **AFTER** | Untuned CARML-style | 99 | 100 | 100 |
| **AFTER** | **Dual-Expert Keys** | **48** | **44** | **100** |

### Lifelong pool (VLKEB, partners in pool) — AFTER only (BEFORE: ~100 edits only)

| When | Pool size | Baseline swap / hard-neg ↓ | Dual-Expert Keys swap / hard-neg ↓ | Correct route ↑ | Rel ↑ |
|---|---|---|---|---|---|
| **AFTER** | 100 | 90 / 100 | **19 / 84** | 68 | 100 |
| **AFTER** | 300 | 97 / 100 | **15 / 86** | 70 | 99.7 |
| **AFTER** | 1000 | 100 / 100 | **23 / 84** | 62 | 99.2 |

### Answer-level (AFTER; BEFORE used conflicting swap-accuracy)

| When | Dataset | Method | Stale answer ↓ | Partner answer ↑ |
|---|---|---|---|---|
| **AFTER** | VLKEB | Baseline | 8.0 | 0.0 |
| **AFTER** | VLKEB | Dual-Expert Keys | **21.0** (worse) | 0.0 |
| **AFTER** | Easy-VQA | Baseline | 8.3 | 0.7 |
| **AFTER** | Easy-VQA | Dual-Expert Keys | 9.3 | 1.3 |

---

## Still pending

| Item | Timing |
|---|---|
| LLaVA Dual-Expert Keys train + eval | **15 Oct** |
| Other editors under same leak protocol | After backbone |
| Absolute hard-neg method retry | After backbone |
| Multi-seed runs | After configs locked |
| MSCKE check + paper writing | Late Oct / **27 Oct** |
| Experiment freeze | **5 Nov** |
