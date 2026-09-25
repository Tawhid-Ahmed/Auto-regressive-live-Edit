# Post–Step 6: Grounding Diagnosis, AAAI Positioning & Next Steps

This document records the **updated understanding** after Steps 0–6, reframed for an **AAAI main-track submission** (not only MSc thesis completion). It supersedes the optimistic Step 5 → Step 6 gate narrative (“retrain patch layer and grounding will follow”) with an evidence-based story aligned to [`THESIS_AAAI_REVIEWER_ROADMAP.md`](THESIS_AAAI_REVIEWER_ROADMAP.md).

**Date:** 2026-07-09  
**Target venue:** AAAI (method paper, Option A — Vision-Grounded Editing)  
**Scope:** BLIP2-OPT-2.7B + LiveEdit + DINOv2 patch grounding (Steps 0–6)  
**Canonical probe suite:** `data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json`  
**Related:** `doc/results/meeting_validation/step{0..6}/STEP*_ANALYSIS.md` · `doc/THESIS_AAAI_REVIEWER_ROADMAP.md`

---

## 1. Executive summary (AAAI lens)

| Question | Answer |
|----------|--------|
| Is there a **publishable problem**? | **Yes — and this is the paper hook.** Lifelong VLM editors leak ~40–43% token accuracy to wrong/confusable images (swap-gap ~31 pp), stable across short/long VLKEB and ablations. AAAI reviewers care about *identified failure modes*, not only +0.5 pp gains. |
| Is the **diagnosis** novel enough for AAAI? | **Partially — as setup, not as the full paper.** Steps 0–4 show Q-Former scalar routing collapse + DINOv2 offline discrimination. That motivates a *new mechanism*; diagnosis alone is an analysis paper (Option C), which AAAI main rarely accepts. |
| Did **patch retrain (Step 6 L1–L4)** solve it? | **No.** No significant adversarial gain vs L0. This is a **negative ablation** — valuable in the paper if paired with a method that *does* work. |
| **AAAI paper identity (committed)** | **Option A — Vision-Grounded Editing (VG-Edit):** decode-time DINO margin gate + adversarial grounding loss + (optionally) vision-gated AR chunks. Steps 0–6 become §3–4 (problem + failed baselines); VG-Edit is §5 (method) + §6 (results). |
| **What AAAI still requires** | ≥5 seeds, paired significance, ≥2 backbones, ≥4 competing editors, human-audited benchmark subset, ROUGE-L/BERTScore/LLM-judge — see §8. |

### 1.1 Mock reviewer take (post–Step 6)

| Criterion | Score (1–6) | Comment after Steps 0–6 |
|-----------|-------------|-------------------------|
| Novelty | **3** (↑ from 2) | Clear multimodal failure mode + patch-routing hypothesis. Still need a **working** VG-Edit mechanism, not only diagnosis. |
| Technical soundness | **4** (↑ from 3) | Rigorous probe suite, ablations L0–L4, offline routing controls. Single seed; no significance tests yet. |
| Significance | **3** | Grounding leakage is field-relevant (LiveEdit, MMKE, lifelong editing). Must show **fix** or strong benchmark resource. |
| Benchmark validity | **2.5** | Probes are strong; VLKEB-long provenance still needs human audit (R3). |
| Clarity | **4** | Step docs + figures are reviewer-ready structure. |
| **Overall (current)** | **Weak reject → borderline** | Diagnosis + negative ablations are **not sufficient alone**; decode gate + multi-seed VG-Edit results could flip to accept. |

---

## 2. AAAI contribution framing (what to claim in the paper)

### 2.1 Defensible contribution bullets (for abstract + intro)

Use exactly three bullets — AAAI reviewers expect crisp, falsifiable claims:

1. **Problem (new empirical finding):** We identify *visual grounding leakage* in lifelong VLM editors: on adversarial image-swap and hard-negative probes, token accuracy remains ~43% while correct-image reliability is ~75% (31 pp swap-gap), across short and long targets and independent of AR or patch retrain (Steps 0–1, 6).

2. **Diagnosis (mechanistic insight):** We trace leakage to **Q-Former routing collapse** (~257 ViT patches → 32 tokens → 1 scalar `vis_sim`) and show DINOv2 patch grids achieve **2% hard-negative false-fire offline** (Step 4) but **do not transfer** when fused only at retrieval without decode-time gating (Steps 5–6).

3. **Method (to implement — VG-Edit):** We propose **Vision-Grounded Editing**: a **decode-time competitive-margin gate** on DINOv2 patch similarity that blocks MoE residual injection on non-matching images, plus **adversarial grounding loss** during patch-module training. *Do not claim this bullet until Phase B/C experiments succeed.*

### 2.2 What NOT to put in the AAAI abstract

| Do not claim | Why a reviewer will reject |
|--------------|---------------------------|
| “AR improves long-form editing by X pp” | +0.54 pp, single seed — within noise (roadmap R2) |
| “Patch retrain fixes grounding” | L1–L4 ≤ L0 on adversarial probes — falsified |
| “We introduce VLKEB-long” alone | Synthetic, lexically anchored — needs human audit (R3) |
| “DINOv2 routing solves the problem” | Offline only; end-to-end unchanged |
| “AnyEdit applied to LiveEdit” | Incremental — not AAAI novelty (roadmap R1) |

### 2.3 How Steps 0–6 map to paper sections

| Paper section | Source material | Status |
|---------------|-----------------|--------|
| **§1 Introduction** | Swap-gap ~31 pp; lifelong editing + visual confusability | ✅ Ready |
| **§2 Related work** | LiveEdit, MMKE-Bench, MC-MKE, AnyEdit, μKE — differentiate explicitly | ⏳ Write |
| **§3 Problem: Visual Grounding Leakage** | Step 1 probes + short/long comparison | ✅ Ready |
| **§4 Diagnosis: Routing Bottleneck** | Steps 0, 2, 4 figures (bottleneck, encoder AUC) | ✅ Ready |
| **§5 Method: VG-Edit** | Decode gate + adversarial loss (+ optional vision-gated AR) | ❌ Implement |
| **§6 Experiments** | L0–L4 ablations as **baselines**; VG-Edit as **ours** | ⏳ Partial |
| **§7 Results** | Multi-seed, significance, 2+ backbones, 4+ editors | ❌ Required |
| **§8 Limitations** | Teacher-forced eval, single backbone so far, Windows repro | ✅ Draft |

---

## 3. Evidence synthesis (Steps 0–6) — for Results / Appendix

### 3.1 What worked (keep in paper)

| Step | Finding | Implication |
|------|---------|-------------|
| **0** | Q-Former routing co-retrieves wrong experts; sim-margin diagnostics expose permissiveness | Motivates patch-level signal |
| **1** | Adversarial probes quantify leakage: swap ~43%, hard-neg ~43%, swap-gap ~31 pp | Primary evaluation harness — keep |
| **2** | Higher ViT resolution does not fix bottleneck | Problem is Q-Former collapse, not input pixels |
| **3** | Image alterations (blur/crop) degrade reliability modestly | Leakage is not only low-level texture |
| **4** | DINOv2 offline routing: 2% hard-neg false-fire, 0% rel false-cross, AUC 1.0 | **Strong offline signal — validated hypothesis** |
| **5 (4c only)** | Vision-gated AR chunks: −4 pp hard-neg false-fire vs baseline routing | Chunk alignment helps marginally at inference |

### 3.2 What did not work (report as negative ablations — AAAI expects honesty)

| Step | Finding | Implication |
|------|---------|-------------|
| **5 (4a/4b)** | Max-pool / patch rerank **hurt** routing (50% hard-neg, 100% rel false-cross) | Do **not** use L2 max-pool or untrained rerank |
| **6 L1–L4** | Patch retrain + module: no win vs L0 on swap-gap or hard-neg | Retrain alone does not transfer Step 4 offline gains |
| **6 L4** | Smallest hard-neg (42.0%) but modality consistency drops to 67.1% | Tradeoff only — not a clear win |
| **6 L5** | Full `dsn=500` OOM in DINOv2 encode; small run unstable | Stack too heavy without memory fixes + simpler gate |

### 3.3 Key numbers (Table 1 candidate — long-form, seed 42, `sen=50`, `dsn=500`)

| Config | Reliability | Hard-neg | Swap-gap | Notes |
|--------|------------:|---------:|---------:|-------|
| L0 (Step 1 proxy, baseline ckpt) | **74.65%** | **43.36%** | **31.15 pp** | Reference |
| L1 patch module | 73.6% | 42.9% | 30.7 pp | ≤ L0 |
| L2 max-pool | 73.6% | 45.8% | 28.3 pp | **Worse** hard-neg |
| L4 verification gate | 73.2% | 42.0% | 31.0 pp | Tiny hard-neg gain, modality cost |
| Short VLKEB (Step 1) | ~96% | ~40% | ~56 pp | Leakage not long-form-only |
| L5 small (`dsn=150`, `sen=20`) | ~44.6% | ~41.8% | ~2.8 pp | Unstable — not fair full comparison |

**Caption for paper:** *Adversarial grounding probes on VLKEB-long (n=500, sequential pool size 50). Patch retrain (L1–L4) does not significantly reduce swap-gap vs Q-Former-only routing (L0). L5 incomplete — omit from main table or mark “—”.*

Sources: `doc/results/meeting_validation/step1/STEP1_GROUNDING_ANALYSIS.md`, `step6/STEP6_ABLATION_ANALYSIS.md`, `step6/step6_ablation_summary.json`.

---

## 4. Mechanistic insight — why integration failed (Discussion §4.2)

### 4.1 Three-layer mismatch

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer A: OFFLINE ROUTING (Step 4)                              │
│  "Does this image match expert k in the pool?"                  │
│  DINOv2 → excellent discrimination                              │
└────────────────────────────┬────────────────────────────────────┘
                             │  fusion in retrieve_moes()
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer B: RETRIEVAL GATE (Step 5–6)                             │
│  vis_sim > vis_sim_prot (+ optional L4 dual gate)               │
│  Still permissive; Q-Former sim dominates fusion                │
└────────────────────────────┬────────────────────────────────────┘
                             │  if gate passes
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer C: GENERATION (edit_with_moes hook)                      │
│  MoE residual injected into LM hidden states                    │
│  Prompt similarity (eqr keys) drives token changes              │
│  NO explicit "should we edit at all on this image?" gate         │
└─────────────────────────────────────────────────────────────────┘
```

**Step 4 optimized Layer A.** Steps 5–6 improved Layer B marginally. **Layer C was never vision-gated or adversarially trained** — so ~43% wrong-image token overlap persists.

### 4.2 Training objective gap

- Patch grounding training (`VLKEB_long_patch`) optimizes **reliability on the correct image** (same as baseline LiveEdit).
- Step 1 **swap / hard-negative probes were evaluation-only** — never a training loss.
- The model learns *what* to say when editing; it does not learn *when not to edit* on wrong images.

### 4.3 Sequential pool contamination

- `sequential_edit_n=50` builds a pool of 50 experts; Step 0/4 show high **false-cross** co-retrieval.
- Each new edit adds keys (`evr_pool`, `eqr_pool`, optional `evr_patch_pool`) that compete at decode time.
- L5 AR multiplies entries per fact (chunked edits) → pool bloat, memory pressure, unstable routing.

### 4.4 Long-form eval artifacts

- Teacher-forced decoding on long targets produces repetitive tokens on wrong images (`"city city"`, etc.) — **base model + decode**, not dirty dataset `alt` text.
- Sanity check (`tools/sanity_check_before_after_edit.py`): wrong-image output **unchanged** after edit on correct image → leakage is not “memory of wrong run” but **prompt-driven partial activation**.

**AAAI writing note:** Frame §4 as *“routing ≠ generation”* — a general lesson for retrieval-based VLM editors, not only LiveEdit.

---

## 5. Decisions (frozen until VG-Edit experiments)

### 5.1 Paper narrative (AAAI-safe claims only)

- **Claim:** Lifelong VLM editors exhibit **visual grounding leakage** on adversarial probes (~43% wrong-image token acc, ~31 pp swap-gap).
- **Claim:** Leakage correlates with **Q-Former scalar routing collapse**; DINOv2 patches discriminate offline but **retrieval fusion alone is insufficient** (negative ablation, Steps 5–6).
- **Claim (pending experiments):** **VG-Edit** (decode gate + adversarial loss) reduces swap-gap with bounded reliability cost.
- **Do not submit without:** multi-seed significance on headline metrics (roadmap R2).

### 5.2 Engineering — stop / deprioritize

| Item | Decision | Rationale |
|------|----------|-----------|
| L2 max-pool routing | **Stop** | Hurts hard-neg in Step 5 and Step 6 |
| L4 as default | **Deprioritize** | Tiny hard-neg gain, hurts modality consistency |
| Full L5 at `dsn=500` | **Block** until memory fixes + decode gate | Crashed twice; small run collapsed reliability |
| More inference-only spikes | **Stop** | Step 5 showed diminishing returns without training signal |
| Step 7 (new backbone) | **Defer** | Fix grounding mechanism first |

### 5.3 Engineering — pursue next (VG-Edit build order)

| Priority | Item | Rationale |
|----------|------|-----------|
| **P0** | **Decode-time DINO margin gate** | Block MoE residual when `sim(current, own) - sim(current, best_other) < margin` — directly uses Step 4 evidence |
| **P0** | **Memory hygiene** | `before_edit_ids` on CPU; optional `torch.cuda.empty_cache()` between splits; smaller probes for development |
| **P1** | **Fair small-scale ablation** | L0 vs L1 vs L4 at same `dsn=150`, `sen=20` (+ isolated pool mode) |
| **P1** | **Adversarial grounding loss** | Fine-tune patch module (or gate) with swap/hard-neg pairs from Step 1 JSON |
| **P2** | **Isolated grounding eval** | `sequential_edit_n=1` or per-sample pool reset for clean grounding metrics |
| **P2** | **Short VLKEB primary for grounding** | Cleaner metrics; long-form for qualitative / AR stress |

| Priority | Item | AAAI requirement |
|----------|------|------------------|
| **P0** | **Decode-time DINO margin gate (VG-Edit core)** | Novel mechanism (R1) — headline method component |
| **P0** | **Multi-seed harness (≥5 seeds)** | Statistical validity (R2) — mandatory before submission |
| **P1** | **Adversarial grounding loss** | Second method component; ablation: gate only vs gate+loss |
| **P1** | **Human audit 300–500 VLKEB-long samples** | Benchmark credibility (R3) |
| **P1** | **Second backbone (LLaVA-1.5)** | Scope (R4) |
| **P2** | **Competing editors: FT, MEND, SERAC, IKE** | Beat the field, not LiveEdit-vs-itself (R4) |
| **P2** | **ROUGE-L, BERTScore, LLM-judge** | Non-saturated metrics (R5) |

### 5.4 Evaluation protocol (AAAI table design)

| Metric family | Probes / metrics | Role in paper |
|---------------|------------------|---------------|
| **Primary (grounding)** | Image swap acc, hard-negative acc, **swap-gap** | Headline — Steps 0–6 already define this |
| **Secondary (edit quality)** | Reliability, locality, modality consistency | Standard editing metrics |
| **Long-form (non-saturated)** | ROUGE-L, BERTScore, LLM-judge faithfulness | Required for VLKEB-long claims (R5) |
| **Routing (diagnostic)** | False-cross rate, hard-neg false-fire (offline) | Figure / appendix — Steps 4–5 |
| **Lifelong stress** | `sen=50` sequential pool | Secondary — do not conflate with pure grounding |
| **Report format** | Mean ± std over ≥5 seeds; paired bootstrap p-values | R2 |

---

## 6. AAAI execution plan (ordered — supersedes thesis-only phases)

### Phase 0 — Paper identity locked ✅
- Commit to **Option A (VG-Edit)** per `THESIS_AAAI_REVIEWER_ROADMAP.md`.
- Steps 0–6 material → Problem + Diagnosis + Negative ablations (no new GPU).

### Phase A — VG-Edit prototype (2–4 days) — **blocks submission**

1. Implement **decode-time DINO margin gate** in `liveedit.py` (`decode_vision_gate_enabled`, `decode_vision_margin`). ✅ **Done 2026-07-09**
2. **Smoke test** (no GPU): `python tools/meeting_validation/step6/smoke_test_decode_vision_gate.py` ✅
3. **Direction verify** (small, ~5–15 min GPU): `python tools/meeting_validation/step6/verify_vg_edit_direction.py -ckpt <baseline> -dvc cuda:0 -dsn 25 -sen 5 -seed 42` — **run before any dsn=500 eval**
4. Dev eval: Step 1 probes, `dsn=150`, seeds `{42}` — gate on/off (`--decode_vision_gate` or `--ablation_layer VG`).
5. **Success gate for Phase B:** hard-neg < 35%, reliability drop < 5 pp (single seed) **after direction verify passes**.

### Phase B — Statistical validity (1–2 weeks GPU) — **blocks submission**

4. Multi-seed: **≥5 seeds** on VG-Edit vs LiveEdit vs L1 patch vs AR.
5. Run `tools/significance_test.py` (paired bootstrap) on swap-gap and hard-neg.
6. Report **mean ± std** and p-values in main table — never single-seed deltas in abstract.

### Phase C — AAAI scope expansion (3–4 weeks) — **required for main track**

7. **Human audit** 300–500 VLKEB-long samples (grounding, factuality, fluency, κ).
8. **Second backbone** (LLaVA-1.5) with same probe suite.
9. **Baselines:** FT-L, MEND, SERAC, IKE on long-form subset.
10. **Metrics:** ROUGE-L, BERTScore, LLM-judge on free-running or teacher-forced long outputs.

### Phase D — Adversarial training (if gate-only insufficient)

11. Fine-tune patch module with swap/hard-neg loss (1–3 epochs).
12. Ablation row: `gate only` | `gate + loss` | `loss only` | `L0 baseline`.

### Phase E — Writing & submission

13. Figures from `doc/results/meeting_validation/step{1,4,5,6}/` + new VG-Edit comparison figure.
14. Limitations: teacher-forced eval, probe construction, compute.
15. Release: probe JSON, eval scripts, configs (reproducibility checklist).

### Dev commands (Phase A — unchanged)

### Memory fix (engineering — run before large multi-seed jobs)

Move `before_edit_ids` to CPU in `evaluation/vllm_editor_eval.py`; reboot / increase pagefile before `dsn=500` runs.

```cmd
cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"
conda activate liveedit

REM Fair small compare (optional sanity before VG-Edit):
python -u -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 20 -dvc cuda:0 -dsn 150 -seed 42 ^
  -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936" ^
  -enp step6_l0_dsn150_sen20 --meeting_step 6 --ablation_layer L0

python -u -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 20 -dvc cuda:0 -dsn 150 -seed 42 ^
  -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_patch-2026.07.08-11.37.09/checkpoints/epoch-20-i-26250-ema_loss-1.3055" ^
  -enp step6_l1_dsn150_sen20 --meeting_step 6 --ablation_layer L1
```

```

---

## 7. Target experiment matrix (AAAI main track)

From `THESIS_AAAI_REVIEWER_ROADMAP.md` §5, updated post–Step 6:

| Axis | Settings |
|------|----------|
| **Methods** | LiveEdit (L0), LiveEdit+AR, LiveEdit+Patch (L1), **VG-Edit (ours)**, FT, MEND, SERAC, IKE |
| **Ablations** | w/o decode gate; w/o adversarial loss; w/o patch module; L2 max-pool (negative) |
| **Backbones** | BLIP2-OPT-2.7b ✅, LLaVA-1.5 ❌ |
| **Seeds** | ≥5 ❌ (only 42 so far) |
| **Primary metrics** | Swap acc, hard-neg acc, **swap-gap**, locality |
| **Long-form metrics** | ROUGE-L, BERTScore, LLM-judge ❌ |
| **Subset** | Full VLKEB-long + human-verified-grounded tier ❌ |
| **Stress** | `sen` ∈ {1, 10, 50} for grounding vs lifelong |

**Figure plan for paper:**
- Fig 1: Architecture (bottleneck + VG-Edit gate) — extend `THESIS_architecture_diagram_patch_grounding_horizontal.png`
- Fig 2: Swap-gap bar chart across methods (multi-seed error bars)
- Fig 3: Offline routing AUC (Step 4) vs end-to-end swap-gap (Step 6) — *routing ≠ generation*
- Fig 4: Ablation table heatmap (L0–L4 + VG-Edit)

---

## 8. AAAI pre-submission checklist

Cross-ref [`THESIS_AAAI_REVIEWER_ROADMAP.md`](THESIS_AAAI_REVIEWER_ROADMAP.md) §7.

| # | Requirement | Status |
|---|-------------|--------|
| R1 | One clearly **novel mechanism** (VG-Edit decode gate) with ablations | ⏳ Gate not implemented |
| R2 | Gains **significant** across ≥5 seeds | ❌ |
| R3 | Human-verified benchmark subset + κ | ❌ |
| R4 | ≥2 backbones, ≥4 competing editors | ❌ |
| R5 | ROUGE-L / BERTScore / LLM-judge | ❌ |
| R6 | Multimodal theory paragraph (optional) | ⏳ Draft in roadmap |
| R7 | Related work differentiates AnyEdit / μKE / LiveEdit / MMKE | ⏳ |
| — | Adversarial probe suite (swap, hard-neg) | ✅ Step 1 |
| — | Routing bottleneck diagnosis | ✅ Steps 0, 2, 4 |
| — | Negative ablation (patch retrain insufficient) | ✅ Step 6 L1–L4 |
| — | Reproducibility (scripts + probe JSON) | ✅ Mostly |

**Submit only when R1 + R2 are green.** R3–R5 are strongly expected for AAAI main; without them, target **AAAI Safe / workshop / Findings** or NeurIPS D&B (benchmark track).

---

## 9. Gate status (Step 6 → VG-Edit → AAAI)

| Criterion | Status |
|-----------|--------|
| L1 beats L0 on ≥2 adversarial probes | ❌ Failed |
| Locality ≥ 95% at 100+ edits | ✅ 100% (L1–L4) |
| L5 completes at full scale | ❌ OOM / unstable |
| Offline DINO routing beats Q-Former | ✅ Step 4 |
| End-to-end grounding fixed | ❌ **Reframe as open problem** |
| Proceed to Step 7 (new backbone) | ⏸ **Deferred** |
| Proceed to VG-Edit implementation | ✅ **Next critical path** |
| Proceed to AAAI submission | ❌ **Blocked** until R1–R2 (minimum) |

---

## 10. Architecture: VG-Edit (proposed method name for paper)

**Current (L1–L5):** `PatchGroundingModule` → `evr_patch` fused with Q-Former `vis_sim` in `retrieve_moes()` → threshold gate → MoE inject.

**Proposed (Phase B):** Add **hard decode gate** — if DINO margin fails, **skip all MoE residuals** for this forward pass regardless of Q-Former similarity.

```
retrieve_moes():
  vis_sim_q, vis_sim_p → fused sim → prototype threshold → candidate experts
  NEW: for each candidate, require dino_margin(own_image, query_image) ≥ τ
  if no candidate passes → return empty moe (no edit)
```

**Naming for paper:** Use **VG-Edit** (Vision-Grounded Editing) — aligns with roadmap Option A and differentiates from “AnyEdit + LiveEdit.”

Diagram (existing): `doc/THESIS_architecture_diagram_patch_grounding_horizontal.png` — add decode gate box for camera-ready.

---

## 11. Risks & fallback venues

| Risk | Mitigation |
|------|------------|
| Windows BLIP2 load ACCESS_VIOLATION | Reboot, close browsers, increase pagefile before large runs |
| DINOv2 encode RAM spike | Lazy load once; float32 preprocess; smaller `dsn` during dev |
| L5 reliability collapse | Do not stack L4+L5 until decode gate validated |
| If VG-Edit gate fails to move swap-gap | Report diagnosis + negative ablations as **analysis paper**; submit benchmark to NeurIPS D&B with human audit |
| If multi-seed gain is < 3 pp | Do not headline; focus on swap-gap + significance on adversarial probes only |
| Deadline pressure | MSc thesis can include Steps 0–6 as-is; AAAI needs Phase B minimum |

---

## 12. File index

| Artifact | Path |
|----------|------|
| AAAI roadmap | `doc/THESIS_AAAI_REVIEWER_ROADMAP.md` |
| This document | `doc/POST_STEP6_GROUNDING_DECISIONS.md` |
| Step 6 ablation | `doc/results/meeting_validation/step6/STEP6_ABLATION_ANALYSIS.md` |
| L0/L5 run log | `doc/results/meeting_validation/step6/l0_l5_run_status.md` |
| Step 1 probes | `doc/results/meeting_validation/step1/STEP1_GROUNDING_ANALYSIS.md` |
| Step 4 offline routing | `doc/results/meeting_validation/step4/STEP4_ENCODER_ANALYSIS.md` |
| Step 5 spikes | `doc/results/meeting_validation/step5/STEP5_PATCH_SPIKE_ANALYSIS.md` |
| Path registry | `doc/THESIS_PATHS.json` |

---

## 13. Changelog

| Date | Change |
|------|--------|
| 2026-07-09 | Initial post–Step 6 synthesis |
| 2026-07-09 | **VG-Edit decode gate implemented** + smoke test + `verify_vg_edit_direction.py` |
