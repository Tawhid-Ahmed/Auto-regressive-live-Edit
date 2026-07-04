# Path to AAAI: Reviewer-Perspective Analysis & Roadmap

**Purpose:** Brutally honest mock-review of the current work, then a concrete plan to make it a *novel*, acceptable AAAI-tier paper.
**Date:** 2026-06-28
**Current artifacts:** VLKEB-long benchmark, AR-LiveEdit (chunk-wise) on BLIP2-OPT-2.7b, full eval matrix (short/long train × baseline/AR), chunk-size ablation.

---

## 1. Mock reviewer scorecard (current form)

| Criterion | Score (1–6) | Reviewer comment |
|-----------|-------------|------------------|
| Novelty / originality | **2** | Combination of two existing methods: AnyEdit's chunk-wise AR + LiveEdit's MoE editor. The paper itself says AnyEdit is "the core mechanism we adapt." This reads as A+B. |
| Technical soundness | **3** | Pipeline is correct, but the key claim rests on **+0.54 pp** with a **single seed** — within noise. |
| Significance / impact | **2** | Main positive effect (+46.8 pp) is just "train on the target distribution," which is expected. The *proposed* mechanism (AR) adds almost nothing. |
| Benchmark validity | **2.5** | VLKEB-long targets are LLM-generated paragraphs anchored only by a **lexical substring**; no human verification, no guarantee the added text is visually grounded. A reviewer will argue it measures *fluent paragraph completion*, not *visual knowledge editing*. |
| Clarity / writing | **4** | Strong, honest exposition and figures. |
| **Overall recommendation** | **Reject (current)** | Solid MSc thesis; not yet an AAAI contribution. |

### The 5 rejection triggers (each is independently fatal at AAAI)
1. **Incremental novelty** — porting AnyEdit to a VLM editor is not a new idea.
2. **Effect within noise** — +0.54 pp, one seed, no significance test.
3. **Benchmark provenance** — synthetic, lexically-anchored, unverified grounding.
4. **Single backbone, no competing baselines** — only LiveEdit-vs-itself.
5. **Self-undercutting ablation** — chunk size doesn't change token accuracy, which weakens the "chunking matters" story.

> Read these as a to-do list, not a verdict. Every one is fixable, and you already have the infrastructure.

---

## 2. The core issue: you need ONE genuine contribution

AAAI accepts papers that introduce *a new idea that works*, or *a new resource the community will use*. Right now you have neither in a defensible form. You must pick and commit to one of three identities below. Trying to be all three weakly is the most common failure mode.

---

## 3. Three viable papers (pick one as PRIMARY)

### Option A — **Method paper:** "Vision-Grounded Autoregressive Editing"
Make the *mechanism* genuinely new and multimodal, not a text port.

- **Novel core (this is the part no prior work has done):** chunk boundaries and edited-token selection are **driven by visual grounding** — place edit anchors at the tokens with the highest image cross-attention (BLIP2 Q-Former), so visually-grounded clauses get the strongest edits. Call it e.g. **VG-Edit** (Vision-Gated Editing).
- **Add causal correctness (fixes your +0.54 pp):** replace independent per-chunk edits with a **Matryoshka / causal objective (μKE-style)** — each early edit is optimized against *all* later tokens, restoring the dependency your current per-chunk loss breaks.
- **Why it can win:** combines two things that are *each* known to help (semantic chunking — AnyEdit++; causal memory — μKE) with a *new* multimodal signal (visual gating). That third piece is the originality.
- **Risk:** requires real engineering + must beat strong baselines.

### Option B — **Benchmark/Resource paper:** "VLKEB-long: A Benchmark for Long-Form Multimodal Knowledge Editing"
Lead with the *dataset* as the contribution; AR becomes one of many evaluated methods.

- **Must add:** (1) **human verification** of a substantial sample (grounding, factuality, fluency) with inter-annotator agreement; (2) **quality tiers** (visually-grounded vs lexically-anchored); (3) evaluate **many editing methods** (FT, MEND, SERAC, IKE, LiveEdit, AR) so it's a real benchmark, not a single-method test.
- **Why it can win:** the field genuinely lacks a long-form multimodal editing benchmark; MMKE-Bench/MC-MKE are free-form but not long-form-centric. A careful, human-verified resource is publishable at D&B tracks (NeurIPS D&B) and citable.
- **Risk:** needs annotation effort; "LLM-generated" must be defended with human QA.

### Option C — **Analysis paper:** "When does autoregressive editing help long-form multimodal knowledge?"
Reframe as a rigorous empirical study (less common at AAAI main, better at EMNLP/findings).

- **Must add:** many backbones, many methods, many seeds, length/forgetting sweeps, and a *predictive* takeaway (when AR helps vs not).
- **Lower novelty ceiling** — usually not enough for AAAI main alone.

**Recommendation:** **Option A as the paper, with B's human audit folded in as the benchmark section.** That gives you a *method* contribution (required for AAAI main) backed by a *credible* benchmark (defuses the provenance attack).

---

## 4. Reviewer requirements → concrete work items

### R1 — Novelty (must-have)
- [ ] Implement **vision-gated chunk anchoring** (image cross-attention selects edit tokens/boundaries). *This is the headline contribution.*
- [ ] Implement **causal/Matryoshka AR objective** (μKE-style expanding-horizon loss) to replace independent per-chunk edits.
- [ ] (Optional) **semantic Bayes-Chunk** boundaries instead of fixed 16-token windows (AnyEdit++).
- [ ] Frame contribution as a *new editing mechanism for VLMs*, not "AnyEdit applied to LiveEdit."

### R2 — Statistical validity (must-have, cheap)
- [ ] **≥5 seeds** for every headline comparison.
- [ ] **Paired significance test** (e.g., bootstrap or paired t-test) on reliability/chunk-EM deltas; report mean ± std and p-values.
- [ ] Replace "+0.54 pp" claims with "significant at p<0.05" or honestly drop them.

### R3 — Benchmark credibility (must-have for this work specifically)
- [ ] **Human audit** of 300–500 VLKEB-long samples: is the long target (a) visually grounded, (b) factually consistent with the short fact, (c) fluent? Report % and **inter-annotator agreement (κ)**.
- [ ] Define **quality tiers**; report results on the verified-grounded subset separately.
- [ ] Add **diverse-format** targets (lists, multi-sentence, reasoning) to show generality, à la AnyEdit's EditEverything.
- [ ] Document construction + release scripts (you mostly have this).

### R4 — Scope (must-have)
- [ ] **≥2 more backbones**: LLaVA-1.5 and/or MiniGPT-4 / InstructBLIP.
- [ ] **≥4 competing editors** as baselines: FT(-L), MEND, SERAC, IKE, plus LiveEdit. Show you beat *the field*, not one model.

### R5 — Metrics (must-have)
- [ ] Add **ROUGE-L** and **BERTScore** for long-form (EM=0% is uninformative).
- [ ] Add an **LLM-as-judge** faithfulness/grounding score (and validate it against the human audit).
- [ ] Keep token-acc + chunk-F1; drop chunk-EM-across-sizes comparisons.

### R6 — Theory (strongly recommended)
- [ ] Tighten the MI / chain-rule argument **for the multimodal case** (frozen image prefix V) — and connect it to *why visual gating reduces cross-chunk interference* (link to AnyEdit++ structural-independence theorem).

### R7 — Positioning / writing (must-have)
- [ ] Crisp 3-bullet contribution list led by the *method*.
- [ ] Related work must explicitly **differentiate** from AnyEdit, μKE, AnyEdit++, LiveEdit, MMKE-Bench, MC-MKE, LEMoE/MEMoE.
- [ ] Lead results with the **significant** method gain on the **verified** subset across **multiple backbones**.

---

## 5. Target experiment matrix (for the method paper)

| Axis | Settings |
|------|----------|
| Backbones | BLIP2-OPT-2.7b, LLaVA-1.5, (InstructBLIP) |
| Editors | FT, MEND, SERAC, IKE, LiveEdit (baseline), **VG-Edit (ours)** |
| Ablations | fixed vs semantic chunking; per-chunk vs Matryoshka loss; with/without visual gating |
| Seeds | ≥5 |
| Metrics | token-acc, chunk-F1, ROUGE-L, BERTScore, LLM-judge, locality, edit-time |
| Stress | edit-count sweep (1 → 1000), length bins |
| Subset | full vs human-verified-grounded |

The three ablation rows (chunking, loss, gating) are what prove **each** new component contributes — reviewers require this for a method paper.

---

## 6. Suggested timeline (part-time, ~3–4 months)

| Month | Focus |
|-------|-------|
| 1 | Vision-gated + Matryoshka implementation; multi-seed harness; ROUGE-L/BERTScore/LLM-judge |
| 2 | Human audit (300–500 samples) + quality tiers; second backbone (LLaVA) |
| 3 | Competing baselines (FT/MEND/SERAC/IKE); edit-count + length sweeps; significance tests |
| 4 | Theory tightening; writing; figures; internal review; submit |

---

## 7. Pre-submission checklist (gate before you submit)

- [ ] One clearly novel mechanism, with ablations proving each part helps
- [ ] Gains **significant** across ≥5 seeds
- [ ] ≥2 backbones, ≥4 competing editors
- [ ] Human-verified benchmark subset + agreement scores
- [ ] ROUGE-L / BERTScore / LLM-judge reported
- [ ] Related work differentiates from AnyEdit / μKE / AnyEdit++ / LiveEdit / MMKE-Bench
- [ ] Reproducibility: code + data + configs released
- [ ] Limitations section is honest (you already do this well)

---

## 7b. Revised execution order (updated 2026-06-28)

**Why revised:** On seed 42, the AR gain over baseline is small on the headline metric
(reliability acc 75.40% → 75.94%, **+0.54 pp**; chunk-EM 13.50% → 14.82%, **+1.32 pp / +9.8%
relative**). Diagnosis: the eval is **teacher-forced** (every token scored on the gold prefix)
and token-accuracy is **saturated** and dominated by the shared frozen backbone, so the regime
where AR actually helps (compounding-error suppression during generation) is hidden. The positional
`position_curve` confirms the bottleneck is the **mid/late answer** (trough ~62–65% at the 50%
position), which AR lifts slightly but which is diluted in the global mean.

**Implication for sequencing:** finalize the eval *regime* (free-running generation + sequence-level
metrics) **before** spending GPU on the multi-seed sweep, otherwise the 8 seed runs would be redone.

**New ordered plan (supersedes the A→D→E→C→B order):**

| Step | Workstream | Rationale |
|------|-----------|-----------|
| 1 (done) | **A** — positional decay diagnostic | Already produced; gives C a measurable target. |
| 2 | **E0 (NEW)** — free-running / non-teacher-forced generation path | Highest leverage: lets baseline errors compound while AR re-grounds per chunk; most likely to turn +0.5 pp into a visible gap. Not previously a task. |
| 3 | **E1/E2** — ROUGE-L, BERTScore, LLM-judge on decoded text | Non-saturated, discriminative metrics on the free-running outputs. |
| 4 | **D1/D2** — multi-seed (≥5) + paired significance, run **once** under the finalized regime | Avoids re-running seeds; `tools/significance_test.py` already written. |
| 5 | **B** — edit-count sweep (1→1000) + forgetting curves | Cheapest path to a *large* headline gap (baseline collapses, AR holds). |
| 6 | **C** — position-aware / Matryoshka loss + retrain | Core method gain; now judged under a regime that can show it, against the A-curve target. |
| 7 | **F** — competing editors (FT/MEND/SERAC/IKE) as long-form baselines | Beat the field, not just LiveEdit-vs-itself. |
| 8 | **G** — related-work positioning, charts, deck, supervisor report | Final writeup. |

**Paused:** the teacher-forced multi-seed runs (previously in progress). Seed-42 `pos_decay`
data is retained for the positional figure.

**Headline-gain principle:** chase the gain in the regime where the mechanism acts — free-running
generation, mid/late positions, long-length bins, and high edit counts — and report on
discriminative metrics (chunk-EM, ROUGE-L, BERTScore, LLM-judge), not a saturated global token-acc.

---

## 8. Honest bottom line

- **As-is:** strong MSc thesis, **reject** at AAAI (incremental + within-noise + synthetic benchmark).
- **With R1–R5:** competitive AAAI/CVPR submission **if** the vision-gated + causal mechanism produces a *significant* multi-backbone gain.
- **Highest-probability win:** method paper (Option A) whose novelty is **visual-grounding-driven editing**, validated on a **human-audited** VLKEB-long, against **real baselines**.

The single most leverage-positive next step is implementing **vision-gated chunking + Matryoshka loss** and re-running multi-seed — that simultaneously fixes novelty (R1) and effect size (R2).
