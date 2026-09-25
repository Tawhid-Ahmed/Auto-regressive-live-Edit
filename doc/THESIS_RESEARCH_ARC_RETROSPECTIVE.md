# Research Arc Retrospective: AnyEdit → LiveEdit → Meeting Validation → AVBL

**Date:** 2026-07-09  
**Scope:** Full project arc from original thesis plan through meeting validation Steps 0–6 and current AVBL results  
**Purpose:** Chronological record of decision shifts, experiments, outcomes, accuracy assessment, and overall verdict (MSc thesis + AAAI)

**Related:** [`POST_STEP6_GROUNDING_DECISIONS.md`](POST_STEP6_GROUNDING_DECISIONS.md) · [`THESIS_AAAI_REVIEWER_ROADMAP.md`](THESIS_AAAI_REVIEWER_ROADMAP.md) · [`THESIS_PATHS.json`](THESIS_PATHS.json) · [`results/meeting_validation/step6/avbl_tune_experiments_summary.json`](results/meeting_validation/step6/avbl_tune_experiments_summary.json)

---

## 1. Executive summary

The project began as **AnyEdit-style autoregressive chunk editing applied to LiveEdit** on VLKEB-long (+0.54 pp reliability — solid MSc engineering, weak AAAI novelty). A July 2026 supervisor meeting-validation sequence discovered **visual grounding leakage** (~43% wrong-image token accuracy, ~31 pp swap-gap), traced it through systematic ablations to **Q-Former scalar routing collapse** and **false-cross expert retrieval** (88–90%), falsified inference-only fixes (patch retrain L1–L4, VG-Edit decode gates), and is now pursuing **AVBL** (training-time adversarial visual binding) as the method track. **Epoch-2 AVBL** is the best token tradeoff so far (+11 pp swap acc, reliability preserved); epoch-3 collapses reliability.

| Track | Verdict |
|-------|---------|
| **MSc thesis** | Strong — AR contribution + rigorous diagnosis + systematic negative results |
| **AAAI main** | Not submission-ready; problem/diagnosis publishable; method needs multi-seed validation |

---

## 2. Phase 0 — Original plan (AnyEdit + LiveEdit, ~April–June 2026)

### 2.1 Starting position

| Piece | Intent |
|-------|--------|
| **LiveEdit** (CVPR 2025) | Lifelong multimodal MoE editor on frozen BLIP2 |
| **AnyEdit** (ICML 2025) | Autoregressive chunk editing for long text targets |
| **Thesis goal** | First combination: **AR-LiveEdit** on **VLKEB-long** (~89-token targets) |

### 2.2 What was built and run

- Long-form benchmark pipeline (Ollama `alt` targets)
- Baseline 20-epoch train + AR `chunk_size=16` train
- Full eval: baseline **75.40%** vs AR **75.94%** reliability (+0.54 pp), chunk EM +1.32 pp, locality 100%

**Evidence:** [`THESIS_BASELINE_VS_AR_COMPARISON.md`](THESIS_BASELINE_VS_AR_COMPARISON.md) · [`THESIS_RESULTS_VLKEB_LONG.md`](THESIS_RESULTS_VLKEB_LONG.md)

### 2.3 Outcome

Solid **MSc thesis engineering** — AR works, is measurable, and is documented. For **AAAI**, the roadmap already flagged this as **incremental** and likely within noise (single seed, teacher-forced eval).

**First implicit shift:** Thesis goal was “make long-form VL editing work better.” Evidence said “it works a bit (+0.5 pp),” not “we solved a hard open problem.”

---

## 3. Phase 1 — AAAI reality check → supervisor meeting validation (July 2026)

### 3.1 Decision shift

Stop selling “AnyEdit + LiveEdit” as novelty. Pivot to a **failure mode** a reviewer would care about. Roadmap **Option A: Vision-Grounded Editing (VG-Edit)**.

**Evidence:** [`THESIS_AAAI_REVIEWER_ROADMAP.md`](THESIS_AAAI_REVIEWER_ROADMAP.md)

### 3.2 New pipeline

Steps 0–6 as **isolated patches** — legacy `test_vllm_edit.py` untouched; new probes, tools, and `eval_results/meeting_validation/`.

**Evidence:** [`THESIS_PATHS.json`](THESIS_PATHS.json)

| | Legacy | Meeting validation |
|--|--------|-------------------|
| Eval | `test_vllm_edit.py` | `test_grounding_eval.py` |
| Results | `eval_results/liveedit/...` | `eval_results/meeting_validation/step<N>/` |
| Analysis | `doc/results/` (partial) | `doc/results/meeting_validation/step<N>/` |

---

## 4. Step-by-step: what was tried, decided, and obtained

### Step 0 — Routing diagnostics (2026-07-05)

| | |
|--|--|
| **Tried** | Log MoE routing on VLKEB-long (150 samples), no retrain |
| **Found** | ~257 ViT patches → 32 Q-Former tokens → **1 scalar** routing key; 81% co-fire of correct + wrong experts |
| **Decision** | PASS → Step 1 |
| **Outcome** | Correct mechanistic starting point |

**Evidence:** [`results/meeting_validation/step0/STEP0_ROUTING_ANALYSIS.md`](results/meeting_validation/step0/STEP0_ROUTING_ANALYSIS.md) · [`THESIS_VLLM_EDITING_CHALLENGE.md`](THESIS_VLLM_EDITING_CHALLENGE.md)

---

### Step 1 — Adversarial grounding probes (2026-07-05–06)

| | |
|--|--|
| **Tried** | `image_swap`, `hard_negative`, etc. on long-form JSON (dsn=500) |
| **Found** | Reliability ~74–75%, **swap/hard-neg ~43%**, **swap-gap ~31 pp** — stable baseline vs AR. Short VLKEB: rel ~96%, swap still ~40% → leakage is **not only a long-form artifact** |
| **Decision** | Automated gate “REFRAME”; scientifically → **this is the real problem** |
| **Outcome** | **Most important discovery of the whole arc** |

**Evidence:** [`results/meeting_validation/step1/STEP1_GROUNDING_ANALYSIS.md`](results/meeting_validation/step1/STEP1_GROUNDING_ANALYSIS.md) · `step1_grounding_summary.json`

| Metric | Baseline | AR |
|--------|----------|-----|
| Reliability | 74.65% | 75.45% |
| Image swap | 43.50% | 42.63% |
| Swap-gap | 31.15 pp | 32.82 pp |

---

### Step 2 — Resolution sweep (224 / 336 / 448 px)

| | |
|--|--|
| **Tried** | More pixels into ViT |
| **Found** | Swap-gap spread **< 1 pp** across resolutions |
| **Decision** | PASS → Step 3 |
| **Outcome** | Correct negative result — bottleneck is post-Q-Former |

**Evidence:** [`results/meeting_validation/step2/STEP2_RESOLUTION_ANALYSIS.md`](results/meeting_validation/step2/STEP2_RESOLUTION_ANALYSIS.md)

---

### Step 3 — Image alterations (crop, rotate, mirror, blur)

| | |
|--|--|
| **Tried** | Same-entity appearance drift at test time |
| **Found** | Mild drops (~1–3 pp); **swap still −31 pp** |
| **Decision** | REFRAME → Step 4 |
| **Outcome** | Wrong-entity confusion dominates; texture drift does not |

**Evidence:** [`results/meeting_validation/step3/STEP3_ALTERATION_ANALYSIS.md`](results/meeting_validation/step3/STEP3_ALTERATION_ANALYSIS.md)

---

### Step 4 — Encoder offline routing (Q-Former vs EVA patches vs DINOv2)

| | |
|--|--|
| **Tried** | Inference-only routing comparison (n=50) |
| **Found** | DINOv2: **2% hard-neg false-fire**, AUC 1.0; Q-Former: **76% rel false-cross** |
| **Decision** | PASS → Step 5; **pick DINOv2** for Step 6 |
| **Outcome** | Correct offline signal — created trap: DINO “works” offline → natural next step “fuse at inference” |

**Evidence:** [`results/meeting_validation/step4/STEP4_ENCODER_ANALYSIS.md`](results/meeting_validation/step4/STEP4_ENCODER_ANALYSIS.md) · `encoder_routing_scores.json`

---

### Step 5 — Inference-only patch spikes (no retrain)

| | |
|--|--|
| **Tried** | Max-pool, rerank, vision-gated AR chunks |
| **Found** | 4a/4b **worse** (false-cross → 100%); 4c slightly better on routing only |
| **Decision** | PASS → Step 6 with reframe: need **trained patch layer** |
| **Outcome** | Correct warning; automated “PASS” was optimistic |

**Evidence:** [`results/meeting_validation/step5/STEP5_PATCH_SPIKE_ANALYSIS.md`](results/meeting_validation/step5/STEP5_PATCH_SPIKE_ANALYSIS.md)

---

### Step 6a — PatchGroundingModule retrain (L1–L4)

| | |
|--|--|
| **Tried** | 20-epoch patch retrain + ablations L0–L4 (dsn=500, sen=50) |
| **Found** | **No layer beats L0** on adversarial probes; L2 hurts; L5 OOM/crash |
| **Decision** | PARTIAL FAIL |
| **Outcome** | Important **negative ablation** — reliability-only patch objective does not fix grounding |

**Evidence:** [`results/meeting_validation/step6/STEP6_ABLATION_ANALYSIS.md`](results/meeting_validation/step6/STEP6_ABLATION_ANALYSIS.md) · `step6_ablation_summary.json`

| Layer | Rel | Hard-neg | Swap-gap |
|-------|----:|---------:|---------:|
| L0 | 74.65% | 43.36% | 31.15 pp |
| L1 | 73.6% | 42.9% | 30.7 pp |
| L2 | 73.6% | 45.8% | 28.3 pp |
| L3 | 73.2% | 42.9% | 30.4 pp |
| L4 | 73.2% | 42.0% | 31.0 pp |

---

### Step 6b — VG-Edit decode-time DINO gates (v1/v2)

| | |
|--|--|
| **Tried** | Margin gate; abs τ + margin sweeps (dsn=100) |
| **Found** | Routing direction checks **PASS**; **token swap-gap unchanged ~31 pp** |
| **DINO sim logging** | **88–90% false-cross**; wrong experts sim ≈0.93–1.0 to stored edit images |
| **Decision shift** | Gates cannot fix dominant failure mode |
| **Outcome** | **VG-Edit as inference-only method falsified** for token improvement |

**Evidence:** [`results/meeting_validation/step6/VG_EDIT_DINO_SIM_ANALYSIS.md`](results/meeting_validation/step6/VG_EDIT_DINO_SIM_ANALYSIS.md) · [`VG_EDIT_VERIFICATION_ANALYSIS.md`](results/meeting_validation/step6/VG_EDIT_VERIFICATION_ANALYSIS.md)

---

### Step 6c — AVBL (training-time adversarial visual binding)

| | |
|--|--|
| **Tried** | AVBL loss; direction check (15 steps); short train λ=1.0; tune λ=0.2 |
| **Direction check** | PASS — swap expert_mass 0.79 → 0.0003; reliability stable |
| **Outcome** | First **training-time** approach with real token movement on adversarial probes |

**Evidence:** [`results/meeting_validation/step6/AVBL_DIRECTION_CHECK.md`](results/meeting_validation/step6/AVBL_DIRECTION_CHECK.md) · [`avbl_tune_experiments_summary.json`](results/meeting_validation/step6/avbl_tune_experiments_summary.json)

| Run | Reliability | Swap acc | Swap-gap | Notes |
|-----|------------:|---------:|---------:|-------|
| L0 | 74.00% | 42.42% | 31.58 pp | baseline |
| λ=1.0 epoch-2 | **79.46%** | **53.15%** | 26.31 pp | **best tradeoff** |
| λ=1.0 epoch-3 | 48.58% | 42.13% | 6.45 pp | reliability collapse |
| λ=0.2, 1 ep | 78.02% | 50.60% | 27.42 pp | calibrated, stable |

**Key lesson:** Epoch-3 “fixes” swap-gap by **destroying reliability**, not by improving swap acc. Epoch-2 raises swap acc **+11 pp** with reliability **+5.5 pp**.

---

## 5. Decision shifts (chronological)

```
AnyEdit + LiveEdit (+0.54 pp AR)
        ↓  "Not enough for AAAI novelty"
Step 1: Discover ~43% wrong-image leakage, ~31 pp swap-gap
        ↓  "Is it pixels? texture? encoder? inference? retrain?"
Steps 2–3: NO (resolution, alterations)
Step 4: DINO wins OFFLINE → hope for end-to-end fix
Steps 5–6 L1–L4: Inference + patch retrain → NO token win
VG-Edit gates: Routing moves, tokens don't
DINO logging: false-CROSS (88%), not false-own-fire (2%)
        ↓  "Fix binding at TRAIN time, not decode filter"
AVBL: Direction PASS; epoch-2 promising; epoch-3 collapses
```

### Core intellectual shift

From “add visual signal / gates” → “the pool retrieves the **wrong** expert with **high** visual confidence; you must train **prompt–image–expert binding**, not threshold similarity.”

### Three-layer mismatch (post–Step 6)

| Layer | What | Status |
|-------|------|--------|
| **A — Offline routing** | DINOv2 discriminates well | Works |
| **B — Retrieval gate** | Q-Former fusion still permissive | Insufficient alone |
| **C — Generation** | MoE injects even when routing wrong | Needs training signal (AVBL) |

---

## 6. Accuracy assessment: was everything done correctly?

### 6.1 Done well (high confidence)

| Area | Assessment |
|------|------------|
| Isolated patch workflow | Legacy pipeline preserved; meeting validation separate — correct |
| Probe suite (Step 1) | Well-designed; quantifies real failure; short + long honest |
| Negative results (Steps 2–3, 6 L1–L4) | Run properly; conclusions match numbers |
| Step 4 offline routing | Clear; correctly motivated DINO for Step 6 |
| VG-Edit falsification | Direction checks + DINO sim logging — rigorous |
| AVBL direction check | Proper fast gate before expensive train |
| Documentation trail | POST_STEP6, step analyses, path registry — thorough |
| Honest reframing | Correctly demotes “patch retrain fixes it” and “gates fix it” |

### 6.2 Imperfections / caveats (disclose in thesis/paper)

| Issue | Impact |
|-------|--------|
| **Single seed (42) everywhere** | AR +0.54 pp and AVBL gains may be noise; AAAI blocker |
| **Teacher-forced token eval** | May overstate efficacy; free-running eval still open |
| **Step 6 L0 = Step 1 proxy** | Same ckpt, slightly different harness — fine for trends, note in appendix |
| **Step 6 L5 never completed** | Vision-gated AR chunk story incomplete |
| **Automated “PASS” gates** | Sometimes optimistic (Step 5 → 6); corrected in prose |
| **Routing log vs token eval protocol** | Full 100-edit pool routing ≠ sen=10 batches — do not mix |
| **“AVBT” in planning** | Only **AVBL** exists in code; keep naming consistent |
| **VG-Edit in POST_STEP6 bullet 3** | Decode gate alone falsified — paper should lead with AVBL + diagnosis |
| **No second backbone / competing editors** | AAAI §6 requirements unmet |
| **VLKEB-long human audit** | Benchmark validity weak for top venue |

### 6.3 Accuracy of “best result” claim

**Epoch-2 AVBL (λ=1.0)** at dsn=100/sen=10 is **accurately measured** and **the best token tradeoff in the table**.

It does **not** meet the strict gate “≥5 pp swap-gap drop with ≤5 pp reliability drop” (swap-gap −5.3 pp only; reliability improved). The **honest claim** is: **swap acc +11 pp with reliability preserved** — not “swap-gap solved.”

---

## 7. Overall verdict

### 7.1 MSc thesis — **Strong and defensible**

You have:

1. Complete AR-LiveEdit on VLKEB-long with fair baselines
2. Rigorous discovery and diagnosis of visual grounding leakage
3. Systematic elimination of wrong hypotheses (resolution, texture, inference gates, reliability-only patch retrain)
4. Emerging fix direction (AVBL) with evidence

**Recommended thesis arc:** AR contribution → grounding discovery → failed fixes → AVBL — not two disconnected projects.

### 7.2 AAAI main track — **Not submission-ready; problem is publishable**

| Dimension | Status |
|-----------|--------|
| Problem / diagnosis | **Ready** — false-cross + swap-gap + gate impossibility |
| Method novelty | **Partial** — AVBL right class; hinge/contrastive not fully validated |
| Empirical win | **Partial** — epoch-2 promising on n=100; not multi-seed / dsn=500 |
| Negative ablations | **Ready** — L1–L4, VG gates valuable baselines |
| Benchmark / rigor | **Weak** — single seed, synthetic long-form, one backbone |

**Mock reviewer (current):** weak reject → **borderline** if contribution reframed as:

> *We identify false-cross visual binding failure in lifelong VLM editing, show why inference visual gates cannot fix it, and propose AVBL training-time adversarial binding that raises adversarial probe accuracy ~10 pp while preserving reliability.*

—not as “VG-Edit decode gate solves editing.”

### 7.3 Single most important lesson

Steps 2–6 were **not wasted**. Each failure narrowed the hypothesis until the true failure mode was visible: **false-cross retrieval under visual pooling**, not “weak DINO threshold” or “need more pixels.” The risk is **over-claiming** (VG gates, swap-gap at epoch-3, offline DINO), not under-delivering.

---

## 8. Recommended next steps (priority order)

1. **Lock epoch-2** (or hinge+contrastive λ=0.2, 1 ep) as “ours” — eval dsn=100, seeds {42, 123, 456} minimum
2. **Update POST_STEP6** — AVBL primary; decode gate demoted to failed ablation
3. **Log false_cross at matched sen=10** during eval (`--log_routing`; align protocol)
4. **Do not scale dsn=500** until 3-seed dsn=100 passes reliability floor
5. **Thesis write-up** — single narrative arc (see §7.1)

---

## 9. Key artifact index

| Purpose | Path |
|---------|------|
| Path registry | `doc/THESIS_PATHS.json` |
| AR results | `doc/THESIS_BASELINE_VS_AR_COMPARISON.md` |
| AAAI pivot | `doc/THESIS_AAAI_REVIEWER_ROADMAP.md` |
| Post-Step-6 decisions | `doc/POST_STEP6_GROUNDING_DECISIONS.md` |
| Step analyses | `doc/results/meeting_validation/step{0..6}/STEP*_ANALYSIS.md` |
| AVBL / VG-Edit | `doc/results/meeting_validation/step6/AVBL_DIRECTION_CHECK.md`, `VG_EDIT_*_ANALYSIS.md` |
| AVBL tune summary | `doc/results/meeting_validation/step6/avbl_tune_experiments_summary.json` |
| **AAAI AVBT strategy (forward)** | `doc/THESIS_AAAI_AVBT_STRATEGY.md` |
| **AAAI AVBT work plan** | `doc/THESIS_AAAI_AVBT_WORK_PLAN.md` |
| Best AVBL ckpt (epoch-2) | `records/.../VLKEB_long_avbl-2026.07.09-16.28.14/checkpoints/epoch-2-i-250-ema_loss-8.0380` |
| Calibrated AVBL (λ=0.2) | `records/.../VLKEB_long_avbl_l02-2026.07.09-18.37.35/checkpoints/epoch-1-i-125-ema_loss-9.2790` |

---

*Generated from full-project retrospective analysis, 2026-07-09.*
