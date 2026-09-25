# AAAI / Thesis Paper Spine — AVBT + Dual-Expert Keys (DEK)

**Date:** 2026-07-13  
**Status:** Title + abstract **submitted**; outline locked to **post–Phase D** evidence · writing copy: [`AAAI_SUBMISSION_DRAFT.md`](AAAI_SUBMISSION_DRAFT.md)  
**Canonical numbers:** `AAAI_ABLATION_TABLE.md` · `AAAI_COMPETITOR_TABLE_SHORT.md` · `AAAI_DEK_MULTISEED.md` · `AAAI_LONG_FAIR.md`  
**Supervisor walkthrough (method + results):** [`THESIS_SUPERVISOR_BRIEFING.md`](THESIS_SUPERVISOR_BRIEFING.md)  
**vs LiveEdit / AnyEdit (novelty + metrics gap):** [`THESIS_VS_LIVEEDIT_ANYEDIT.md`](THESIS_VS_LIVEEDIT_ANYEDIT.md)

### Naming
| Short | Full name |
|-------|-----------|
| L0 | LiveEdit baseline |
| S2 | AVBT (patch grounding, no dual-gate) |
| DEK | AVBT + Dual-Expert Keys (**ours**) |
| FX | False-cross mining |
| PB | Pool buffer |
| FT / MEND / SERAC | Fine-tuning / MEND-VL / SERAC-VL |
| Rel / Swap / Hard / FC | Reliability / Image-swap / Hard-negative / False-cross |

---

## Submitted title (AAAI-27 OpenReview — locked)

**Dual-Expert Keys for Reducing False-Positive Retrieval in Lifelong Vision–Language Model Editing**

**Short / running head:** Dual-Expert Keys for Lifelong VLM Editing

---

## Submitted abstract (locked substance)

Lifelong vision–language editors can look correct on the image they were edited for, then still emit the new fact when the scene is only visually related. In our probes this is usually not soft overwrite of the base model. It is false-positive retrieval (false-cross): weak visual keys make a different stored edit look good enough to retrieve, so the wrong expert fires. We address it in two steps. Dual-Expert Keys let an edit fire only when the visual match and the prompt-key match both pass. Separately, adversarial visual binding training attaches each edit to its correct image and lowers expert activation when the image is swapped or a hard negative. On VLKEB-based adversarial grounding probes, Dual-Expert Keys cut swap false-positive retrieval rate from 96% to 46% over five seeds and keep reliability near 94%. Adversarial visual binding training complements Dual-Expert Keys by stabilizing edit–image binding under these adversarial probes. Against Fine-tuning, MEND-VL, and SERAC-VL on the same probes, we keep high reliability while directly reducing the wrong-expert retrieval rate.

*(Full PDF may still report FC as false-cross; title term = false-positive retrieval. Long-form FC / ablations belong in Results, not this abstract.)*

---

## Contribution bullets (use in intro)

1. **Problem:** Visual grounding leakage in lifelong VLKE — high reliability, low image-swap / hard-negative, mass **false-cross**.  
2. **Diagnosis:** Leakage tracks **false-cross retrieval** under collapsed visual keys (Steps 0–4; DINO sims).  
3. **Method:** **AVBT** (adversarial binding) + **Dual-Expert Keys (DEK)** (vision∧text dual-gate) cut false-cross ~half without collapsing reliability.

**Do not claim:** AR as main novelty; patch-retrain fixes grounding; G3 image-swap ≥55%; false-cross mining (FX) / pool buffer (PB) as primary method.

---

## Section map

| § | Title | Content source | Status |
|---|-------|----------------|--------|
| 1 | Introduction | Swap-gap + FC framing; 3 bullets | Draft below |
| 2 | Related work | LiveEdit, MMKE, MEND/SERAC/FT, AnyEdit; position as *routing grounding* | Outline |
| 3 | Problem | Step 1 probes; short vs long leakage | Ready (meeting docs) |
| 4 | Diagnosis | Q-Former bottleneck; FC vs false-fire; DINO offline | Ready |
| 5 | Method | AVBT loss + Dual-Expert Keys (+ optional FX as appendix) | Outline below |
| 6 | Experiments | Protocol, metrics (FC first), baselines | Ready |
| 7 | Results | Ablation · competitors · multi-seed · fair long | Ready tables |
| 8 | Limitations | Swap ceiling; single backbone; FC LiveEdit-only; synthetic long | Draft |
| 9 | Conclusion | FC+rel story; future work on token-level refuse | Stub |

---

## §1 Intro — key paragraphs (outline)

1. **Setup:** Lifelong VLM editing (LiveEdit MoE pool) under sequential edits.  
2. **Hook:** Reliability can be high while **wrong image + same prompt** still retrieves foreign experts → edit leakage.  
3. **Gap:** Prior VLKE papers optimize reliability / locality; few report **adversarial visual** probes or **routing false-cross**.  
4. **Ours:** Diagnose FC; AVBT + Dual-Expert Keys (DEK); evidence on short (multi-seed) + fair long.  
5. **Teaser numbers:** short FC 96→46, swap +7 pp; long FC 88→38, rel held.

---

## §5 Method — claimable components

| Component | What it is | Evidence role |
|-----------|------------|---------------|
| **AVBT (S2)** | Train to bind correct (prompt, image) and suppress adversarial images | Swap ↑, FC still 96% |
| **Patch grounding keys** | DINOv2-informed visual expert keys | Enables Dual-Expert Keys / logging |
| **Dual-Expert Keys (DEK)** | Retrieve only if **vision ∧ text** gates agree | FC halves; **primary novelty** |
| False-cross mining (FX) / Pool buffer (PB) | Neighbor mining + refuse / pool buffer | Appendix negatives |

**Architecture figure:** [`THESIS_architecture_AVBT_DEK.png`](THESIS_architecture_AVBT_DEK.png) · Mermaid source [`THESIS_architecture_AVBT_DEK.mmd`](THESIS_architecture_AVBT_DEK.mmd)

**Inference flags for “ours”:** `--patch_grounding --dek_dual_gate` (+ `--log_routing` for false-cross).

---

## §7 Results — table plan

| Paper table | Source doc |
|-------------|------------|
| T1 Ablation: LiveEdit baseline (L0) → AVBT (S2) → Dual-Expert Keys (DEK) → FX → PB | `AAAI_ABLATION_TABLE.md` |
| T2 Competitors: Fine-tuning / MEND-VL / SERAC-VL | `AAAI_COMPETITOR_TABLE_SHORT.md` |
| T3 Multi-seed: Dual-Expert Keys (DEK) vs LiveEdit baseline (L0) | `AAAI_DEK_MULTISEED.md` |
| T4 Fair long: LiveEdit baseline (L0) vs Dual-Expert Keys (DEK) | `AAAI_LONG_FAIR.md` |

**Consolidated analysis + figures:** [`results/meeting_validation/step6/AAAI_RESULTS_ANALYSIS.md`](results/meeting_validation/step6/AAAI_RESULTS_ANALYSIS.md) · [`results/meeting_validation/step6/figures/`](results/meeting_validation/step6/figures/)

**Metric order in every table:** False-cross (FC) → Image-swap (Swap) → Hard-negative (Hard) → Reliability (Rel) → Swap-gap.

---

## §8 Limitations (honest)

- Long image-swap ~43% — **not** a token-swap SOTA claim.  
- False-cross requires LiveEdit-style retrieval pool (n/a for Fine-tuning / MEND-VL / SERAC-VL).  
- BLIP2-OPT-2.7B only so far.  
- VLKEB-long provenance needs human audit (G4).  
- Teacher-forced / token-acc protocol (state standard; mention LLM-judge as future).

---

## Venue positioning

| Track | Fit |
|-------|-----|
| **MSc thesis** | Strong — problem + mechanism + multi-seed + fair long + negatives |
| **AAAI main** | Stretch — G3 swap miss; lead with **false-cross / anti-leakage**; workshop if reviewers demand ≥55% long swap |
| **Analysis / workshop** | Safe fallback emphasizing diagnosis + Dual-Expert Keys FC |

---

## Next writing tasks (after G4 stats/audit)

1. Expand §2 related-work paragraph drafts.  
2. Figures: routing bottleneck; FC bar chart short/long; ablation waterfall.  
3. Paste T1–T4 into thesis chapter / Overleaf once significance + audit checklist land.
