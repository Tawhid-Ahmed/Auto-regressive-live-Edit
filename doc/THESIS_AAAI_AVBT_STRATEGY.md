# AAAI Strategy: Beating SOTA via Adversarial Visual Binding Training (AVBT)

**Date:** 2026-07-09  
**Status:** Committed direction (supersedes decode-gate-only VG-Edit as headline method)  
**Audience:** Thesis + AAAI main-track planning

**Prerequisites:** [`THESIS_RESEARCH_ARC_RETROSPECTIVE.md`](THESIS_RESEARCH_ARC_RETROSPECTIVE.md) · [`POST_STEP6_GROUNDING_DECISIONS.md`](POST_STEP6_GROUNDING_DECISIONS.md) · [`THESIS_AAAI_REVIEWER_ROADMAP.md`](THESIS_AAAI_REVIEWER_ROADMAP.md)

**Work plan derived from this doc:** [`THESIS_AAAI_AVBT_WORK_PLAN.md`](THESIS_AAAI_AVBT_WORK_PLAN.md)

---

## 1. Executive decision

| Decision | Rationale |
|----------|-----------|
| **Primary paper identity** | **AVBT** (Adversarial Visual Binding Training) — not “AnyEdit + LiveEdit,” not inference-only VG-Edit |
| **What “beat SOTA” means** | Outperform FT, MEND, SERAC, IKE, and LiveEdit on **adversarial visual grounding probes** while preserving reliability — not +10 pp on reliability alone |
| **Headline metric** | **Swap-gap** and **swap/hard-neg accuracy** on Step 1 probe suite (metrics no prior VLKE paper optimizes) |
| **Supporting narrative** | Steps 0–6 diagnosis + negative ablations (L1–L4, decode gates) prove *why* prior approaches fail |
| **Defer** | AR (+0.54 pp) as main contribution; full L5 at dsn=500; more inference-only spikes |

---

## 2. Reframing SOTA for this paper

### 2.1 What reviewers will compare

| Metric | Current SOTA band (LiveEdit family) | Our gap |
|--------|-------------------------------------|---------|
| Token reliability (correct image) | 74–75% | Already competitive |
| **Image swap acc** | **~42–43%** | **Everyone fails — no method optimizes this** |
| **Hard-negative acc** | **~42–43%** | Same |
| **Swap-gap** | **~31 pp** | Stable across AR, patch, gates |

### 2.2 Honest win condition

Competing editors run on the **same grounding JSON** will likely show ~40–43% swap acc. **AVBT wins** if it:

1. Raises swap/hard-neg acc by **≥15 pp** vs LiveEdit L0  
2. Holds reliability within **≤3 pp** of baseline (~74%)  
3. Reduces swap-gap to **≤12 pp**  
4. Achieves **p < 0.05** over LiveEdit on swap-gap across **≥5 seeds**

Reliability does not need to beat 79% (epoch-2 AVBL) — stability matters more than peak.

### 2.3 Target numbers (headline table)

| Metric | L0 / expected SOTA editors | AVBT target (ours) |
|--------|---------------------------|-------------------|
| Reliability | 74–75% | **≥72%** |
| Swap acc | ~43% | **≥60%** |
| Hard-neg acc | ~43% | **≥58%** |
| Swap-gap | ~31 pp | **≤12 pp** |

**Current best (epoch-2 AVBL, dsn=100, seed 42):** rel 79.5%, swap 53.2%, gap 26.3 pp — **directionally correct, not yet headline-ready.**

Evidence: [`results/meeting_validation/step6/avbl_tune_experiments_summary.json`](results/meeting_validation/step6/avbl_tune_experiments_summary.json)

---

## 3. Why incremental tweaks will not suffice

The following are **falsified** — do not revisit as primary bets:

| Approach | Evidence | Verdict |
|----------|----------|---------|
| Higher resolution (Step 2) | Swap-gap spread < 1 pp | ❌ |
| Image alterations (Step 3) | Swap still −31 pp | ❌ |
| Inference patch spikes 4a/4b (Step 5) | False-cross → 100% | ❌ |
| Patch retrain L1–L4 (Step 6) | No win vs L0 on adversarial probes | ❌ |
| VG-Edit decode gates v1/v2 | Routing OK; token swap-gap unchanged | ❌ |
| AVBL λ=1.0 epoch-3 | Swap-gap 6.5 pp but rel 48.6% | ❌ (metric gaming) |
| Offline DINO AUC 1.0 alone | Does not transfer end-to-end | ❌ as sole method |

**Root cause (unchanged):** **88–90% false-cross** — wrong expert retrieved with DINO sim ≈ 0.93–1.0 to stored edit images. Fix requires **training-time binding** + **expert key disambiguation**, not thresholds.

Evidence: [`results/meeting_validation/step6/VG_EDIT_DINO_SIM_ANALYSIS.md`](results/meeting_validation/step6/VG_EDIT_DINO_SIM_ANALYSIS.md)

---

## 4. Method: AVBT (three components)

Rename from **AVBL** to **AVBT** in the paper to signal a full method (not a single loss term). Code flag may remain `--adversarial_grounding`.

### 4.1 Component A — Contrastive Expert Binding (CEB) [P0, training]

**Core idea:** Train routing to bind **(prompt, correct image) → own expert** and **(prompt, adversarial image) → suppress wrong experts** — not global prototype collapse.

| Mechanism | Detail |
|-----------|--------|
| Positive loss | `-log P(own expert \| prompt, correct image)` |
| Negative loss | Hinge on `expert_mass`; contrastive penalty on **max wrong-expert prob** |
| Targeted mining | For each swap probe, downweight the **specific false-cross expert** from pool (from DINO logging) |
| Trainable params | **`inpt_extractor` + `patch_grounding_module`** jointly (not patch-only) |
| Hyperparams | `λ ∈ [0.15, 0.25]`, `neg_weight=2.0`, early stop epoch 1–2 |

**Code status (2026-07-09):** Hinge + contrastive flags exist (`--avbl_hinge_tau`, `--avbl_contrastive`); targeted false-cross loss and joint `inpt_extractor` training **not yet implemented**.

**Novelty claim:** First VLKE objective that optimizes **when not to bind** under lifelong pooling with contrastive wrong-expert suppression.

### 4.2 Component B — Disambiguated Expert Keys (DEK) [P1, architecture]

**Core idea:** False-cross occurs because different facts’ edit images look alike to DINO. Disambiguate keys at **edit time**.

| Mechanism | Detail |
|-----------|--------|
| Fused key | `(patch_embedding, prompt_embedding)` instead of scalar `vis_sim.mean()` |
| Retrieval | Require visual margin **and** text-routing agreement before MoE inject |
| Pool hygiene | At edit insert: if `DINO(new_edit_img, pool_img) > 0.95`, merge or skip duplicate visual key |

**Code status:** Not implemented.

### 4.3 Component C — Decode Binding Veto (DBV) [P2, inference ablation]

**Core idea:** Lightweight gate after retrieval, before MoE inject — full-pool DINO margin (not Q-Former shortlist only).

| Rule | Inject only if own-sample expert wins **and** `sim(own) - sim(best_other) > margin` on full pool |

**Paper role:** Ablation row only. If zero token gain after CEB+DEK, omit from main method (honest negative, like VG-Edit v2).

---

## 5. Experiment matrix (credibility requirements)

From [`THESIS_AAAI_REVIEWER_ROADMAP.md`](THESIS_AAAI_REVIEWER_ROADMAP.md) R2–R4:

| Axis | Minimum | Ideal |
|------|---------|-------|
| Editors | LiveEdit L0, FT-VL, +2 of MEND/SERAC/IKE | All five |
| Backbones | BLIP2-OPT-2.7b | + LLaVA-1.5 |
| Seeds | **5** | 5 |
| Eval scale | dsn=500, sen=50 (headline) | Same |
| Metrics | Rel, swap, hard-neg, swap-gap, locality | + ROUGE-L, BERTScore |
| Subset | Full + human-verified 300 samples (κ) | Required for benchmark defense |

### 5.1 Main results table (target)

| Method | Rel ↑ | Swap ↑ | Hard-neg ↑ | Swap-gap ↓ |
|--------|------:|-------:|-----------:|-----------:|
| FT | TBD | ~40? | ~40? | ~30? |
| MEND | TBD | TBD | TBD | TBD |
| SERAC | TBD | TBD | TBD | TBD |
| IKE | TBD | TBD | TBD | TBD |
| LiveEdit (L0) | 74.7 | 43.4 | 43.4 | 31.2 |
| LiveEdit + AR | 75.5 | 42.6 | 42.4 | 32.8 |
| LiveEdit + Patch (L1) | 73.6 | 42.9 | 42.9 | 30.7 |
| **AVBT (ours)** | **≥72** | **≥60** | **≥58** | **≤12** |

### 5.2 Ablation table (method paper requirement)

| Config | Components |
|--------|------------|
| L0 | Baseline LiveEdit |
| CEB | Contrastive expert binding only |
| CEB + DEK | + disambiguated keys + pool dedup |
| CEB + DEK + DBV | + decode veto (if helps) |
| AVBL epoch-3 (negative) | Prototype-only over-suppression |

---

## 6. AAAI contribution bullets (use when experiments pass)

1. **Problem:** We identify *visual grounding leakage* in lifelong VLM editors and introduce adversarial probes where existing methods retain ~43% wrong-image accuracy (~31 pp swap-gap).

2. **Diagnosis:** We show the dominant failure is **false-cross expert retrieval** (88%), not own-expert false-fire; inference visual gates cannot fix it when wrong experts pass DINO at 0.93+ similarity.

3. **Method:** **AVBT** — contrastive adversarial binding training plus disambiguated expert keys — significantly reduces swap-gap while preserving reliability, outperforming FT, MEND, SERAC, IKE, and LiveEdit on grounding probes.

**Do not lead with:** AR (+0.54 pp), decode-only VG-Edit, offline DINO AUC, or VLKEB-long introduction alone.

---

## 7. Frozen decisions — what to stop

| Stop | Why |
|------|-----|
| More λ sweeps on prototype-only AVBL without contrastive/hinge | Epoch-3 collapse |
| dsn=500 training before 3-seed dsn=100 gate passes | Wastes GPU; repeat collapse risk |
| L2 max-pool, untrained rerank | Falsified Steps 5–6 |
| Claiming DINO offline = end-to-end win | Reviewers have Step 6 evidence |
| AAAI submit before competing editors on same probes | “Beat SOTA” unsubstantiated |
| VG-Edit decode gate as headline | Falsified for tokens |
| 3-epoch AVBL at λ=1.0 | Reliability destruction |

---

## 8. Relationship to prior documents

| Prior doc | How AVBT relates |
|-----------|------------------|
| `POST_STEP6_GROUNDING_DECISIONS.md` | Updates P0 from “decode gate first” to **CEB first**; gate demoted to DBV ablation |
| `THESIS_AAAI_REVIEWER_ROADMAP.md` | Option A retained; **VG-Edit** redefined as AVBT (training + keys), not chunk anchoring only |
| `AVBL_DIRECTION_CHECK.md` | AVBL = prototype of Component A; epoch-2 ckpt = starting checkpoint |
| `THESIS_RESEARCH_ARC_RETROSPECTIVE.md` | Historical arc; this doc = forward strategy |

---

## 9. Success criteria summary

| Gate | Criterion |
|------|-----------|
| **Development** | dsn=100, 3 seeds: swap ≥55%, rel drop ≤3 pp vs L0 |
| **Headline** | dsn=500, 5 seeds: swap ≥60%, swap-gap ≤12 pp, p<0.05 vs L0 |
| **SOTA claim** | Same probes, FT + ≥2 other editors, all worse than AVBT on swap-gap |
| **Generality** | AVBT on LLaVA shows same direction of gain |
| **Submission** | Human audit κ on 300 samples + ablation table complete |

---

*Strategy locked 2026-07-09. Implementation and schedule: [`THESIS_AAAI_AVBT_WORK_PLAN.md`](THESIS_AAAI_AVBT_WORK_PLAN.md).*
