# AAAI G4 — Human audit plan (swap / hard-neg probes)

**Date:** 2026-07-13  
**Goal:** Credibility check that image-swap / hard-negative pairings are *visually confusable or clearly wrong-entity*, not broken JSON / identical images / unreadable targets.  
**Scope:** Short grounding eval first (`grounding_eval_short.json`); optional longform later.  
**Related:** `THESIS_AAAI_PAPER_SPINE.md` §8 · execution gate **G4**

---

## 1. What to audit (not model outputs)

This audit validates **probe construction**, not DEK accuracy.

| Probe | Field | Pass if |
|-------|-------|---------|
| Image swap | `grounding_probes.image_swap` vs `image` | Different entity/scene; same prompt should **not** still clearly depict the edit target |
| Hard-neg | `grounding_probes.hard_neg_image` vs `image` | Visually related/confusable but **not** the same instance as `image` |
| Modality consistency | `modality_consistency_image` | Same entity as `image` (rephrase view) |
| Target text | `alt` / `pred` | Grammatical, matches intended edit fact |

**Out of scope for v1:** judging whether the *model* answered correctly.

---

## 2. Sample size

| Set | N | How |
|-----|---|-----|
| Primary (short) | **150** | Stratified random from first 100 eval indices used in AAAI runs *or* uniform from full 3174 with seed 42 |
| Double-coded subset | **50** of the 150 | Second rater → Cohen’s κ |
| Optional long | 50 | Same rubric on `grounding_eval_longform.json` |

Paper report: agreement κ on probe validity; % invalid / borderline.

---

## 3. Rubric (1 row per sample × probe)

Scores (integers):

| Code | Meaning |
|------|---------|
| **2** | Clearly valid for the probe definition |
| **1** | Borderline / ambiguous |
| **0** | Invalid (same image, wrong file, unreadable, not adversarial) |
| **X** | Skip (missing file) |

Minimum fields per row:

```text
sample_idx, entity_id, prompt, target_alt,
edit_image, swap_image, hard_neg_image,
swap_score, hard_neg_score, modality_score, notes, rater_id
```

**Pass criteria for dataset quality (soft gate):**
- ≥90% of audited swaps score ≥1  
- ≥85% of hard-negs score ≥1  
- κ ≥ 0.6 on double-coded subset (substantial)

If fail → document failure modes; do **not** silently drop probes without a appendix note.

---

## 4. Rater protocol (15–20 min training)

1. Open edit image + swap image side-by-side.  
2. Read `rephrase` / conflict prompt once.  
3. Ask: “If the model answered the edit target on the *swap* image, is that unfair leakage or a bad pair?”  
4. Score swap, then hard-neg, then modality.  
5. Do **not** look at model predictions or routing FC.

Recommended UI: spreadsheet + image paths under  
`data/VLKEB/VLKEB_images/mmkb_images/`.

---

## 5. Deliverables

| Artifact | Path (planned) |
|----------|----------------|
| Sample sheet | `doc/results/meeting_validation/step6/audit/audit_samples_short_n150.csv` |
| Rater A scores | `.../audit_scores_raterA.csv` |
| Rater B (50) | `.../audit_scores_raterB.csv` |
| Summary + κ | `AAAI_HUMAN_AUDIT_RESULTS.md` (+ JSON) |

**Script to generate the sample sheet:**  
`tools/meeting_validation/step6/sample_audit_sheet.py`  
(run when ready to annotate; does not require GPU).

---

## 6. Paper wording (after results)

> We manually audited *N*=150 short swap/hard-neg pairs (50 double-coded).  
> *X*% were rated valid (≥1); Cohen’s κ=*…*. Invalid cases were … (appendix).

Until scores exist, thesis §Limitations cites this checklist as **planned / in progress**.

---

## 7. Immediate next action

Run the sampler (no GPU):

```cmd
E:\anaconda\envs\liveedit\python.exe -u tools\meeting_validation\step6\sample_audit_sheet.py
```

Then annotate 150 rows (or start with 50 if time-boxed).
