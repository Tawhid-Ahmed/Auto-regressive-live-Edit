# AVBT Work Plan: Next Steps to AAAI-Grade Improvements

**Date:** 2026-07-09  
**Derived from:** [`THESIS_AAAI_AVBT_STRATEGY.md`](THESIS_AAAI_AVBT_STRATEGY.md)  
**Timeline:** ~6–8 weeks part-time (adjust to your availability)

**Current best checkpoint:** `records/liveedit/blip2-opt-2.7b/VLKEB_long_avbl-2026.07.09-16.28.14/checkpoints/epoch-2-i-250-ema_loss-8.0380` (rel 79.5%, swap 53.2%, dsn=100)

---

## 1. Goals and gates

### 1.1 Numeric targets

| Phase | dsn | sen | Seeds | Swap acc | Reliability | Swap-gap |
|-------|-----|-----|-------|----------|-------------|----------|
| **Gate A** (dev) | 100 | 10 | 3 | ≥55% | ≥71% (≤3 pp vs L0) | ≤20 pp |
| **Gate B** (headline) | 500 | 50 | 5 | ≥60% | ≥72% | ≤12 pp |
| **Gate C** (SOTA) | 500 | 50 | 5 | Beat all baselines on swap-gap | ≤5 pp rel drop vs best baseline | p<0.05 |

L0 reference (dsn=100): rel 74.0%, swap 42.4%, gap 31.6 pp — [`avbl_tune_experiments_summary.json`](results/meeting_validation/step6/avbl_tune_experiments_summary.json)

### 1.2 Deliverables checklist

- [ ] CEB implementation (targeted false-cross + joint `inpt_extractor` train)
- [ ] DEK implementation (fused keys + pool dedup)
- [ ] DBV ablation (optional decode veto)
- [ ] 3-seed dev eval JSON + plots
- [ ] 5-seed headline eval + bootstrap significance
- [ ] Competing editors on grounding probes (FT + 2 others minimum)
- [ ] Ablation table (L0 / CEB / CEB+DEK / CEB+DEK+DBV)
- [ ] LLaVA backbone smoke test
- [ ] Human audit 300 samples (κ)
- [ ] Paper figures: swap-gap bar chart, false-cross routing rates

---

## 2. Phase 1 — Complete CEB (Weeks 1–2)

**Objective:** Close the +7–10 pp swap gap between epoch-2 (53%) and target (60%+).

### 2.1 Engineering tasks

| ID | Task | File(s) | Priority |
|----|------|---------|----------|
| E1 | **Targeted false-cross loss:** given pool + sample id, identify top wrong expert index; add `relu(wrong_prob - margin)` or `-log(1 - wrong_prob)` | `editor/vllm_editors/liveedit/liveedit.py` | P0 |
| E2 | **Joint optimizer:** include `inpt_extractor` in AVBT train params (not only `patch_grounding_module`) | `liveedit.py` `get_a_new_optimizer` / train path | P0 |
| E3 | **Reliability anchor:** cap negative loss when `own_prob < τ_rel` or scale neg by `own_prob` | `liveedit.py` | P0 |
| E4 | Wire mining from `batch_adv_specs` + current pool indices in `_train_a_batch_impl` | `liveedit.py` | P0 |
| E5 | Extend `verify_avbl_direction.py` to report **false_cross rate** (not only expert_mass) | `tools/meeting_validation/step6/` | P1 |
| E6 | Unit smoke: `smoke_test_avbl.py` covers new loss terms | `tools/meeting_validation/step6/smoke_test_avbl.py` | P1 |

### 2.2 Hyperparameter sweep (small GPU)

Fixed: 1 epoch, dsn=500 train samples, bs=4, seed 42, grounding JSON train path.

| Run | `adv_grounding_lambda` | `neg_weight` | `avbl_hinge_tau` | `--avbl_contrastive` |
|-----|------------------------|--------------|------------------|----------------------|
| S1 | 0.15 | 2.0 | 0.0 | off |
| S2 | 0.20 | 2.0 | 0.3 | on |
| S3 | 0.25 | 2.0 | 0.3 | on |
| S4 | 0.20 | 2.0 | 0.2 | on + targeted FC (E1) |

**Base ckpt:** L0 baseline `epoch-20-i-27800-ema_loss-0.7936` (fresh fine-tune, not epoch-2 — avoid compounding).

```cmd
conda activate liveedit
cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"

python train_vllm_editor.py -en liveedit -mn blip2 -dna VLKEB -bs 4 -dvc cuda:0 -edvc 0 ^
  -lkpt "records/liveedit/blip2-opt-2.7b/VLKEB_long_baseline-2026.06.02-13.43.00/checkpoints/epoch-20-i-27800-ema_loss-0.7936" ^
  -eps 1 -tnp VLKEB_avbt_sweep_S2 -dn 500 -rs 42 -sci 125 -lpi 10 ^
  -dpath data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json ^
  --patch_grounding --adversarial_grounding --adv_grounding_lambda 0.2 ^
  --adv_grounding_neg_weight 2.0 --avbl_hinge_tau 0.3 --avbl_contrastive
```

### 2.3 Eval protocol (each sweep run)

```cmd
python -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 10 -dvc cuda:0 ^
  -dsn 100 -seed 42 -ckpt "<run_ckpt>" -enp avbt_sweep_S2_dsn100 --meeting_step 6 --log_routing
```

Repeat seeds: `42`, `123`, `456`.

**Output:** `doc/results/meeting_validation/step6/avbt_phase1_sweep_summary.json`

### 2.4 Gate A decision

| Pass | Action |
|------|--------|
| ✅ swap ≥55%, rel ≥71% on ≥2/3 seeds | Promote best config to Phase 2 |
| ❌ | Tune λ/τ; add E1 targeted loss; do **not** go to dsn=500 |

---

## 3. Phase 2 — DEK architecture (Week 3)

**Objective:** Reduce false-cross at retrieval without hurting reliability.

### 3.1 Engineering tasks

| ID | Task | Detail |
|----|------|--------|
| D1 | Fused expert key: concat or bilinear fuse patch routing rep + prompt routing rep | `liveedit.py` `_pool_vis_sim` / `retrieve_moes` |
| D2 | Pool dedup at `edit_one_piece`: DINO sim > 0.95 → skip new visual key or merge expert | `liveedit.py` |
| D3 | Eval-only flag `--pool_dedup` + `--fused_expert_key` for ablation | `test_grounding_eval.py` or editor flags |
| D4 | Log `image_swap_false_cross_rate` at **sen=10** (match token eval) | `--log_routing` protocol fix |

### 3.2 Experiments

| Run | Train | Inference |
|-----|-------|-----------|
| B1 | Best CEB ckpt from Phase 1 | Standard |
| B2 | Same | + DEK only (no retrain) |
| B3 | Retrain CEB 1 ep with DEK enabled | DEK |

Eval: dsn=100, seeds 42/123/456, `--log_routing`.

**Gate:** False-cross rate drops ≥10 pp vs CEB-only **and** swap acc does not drop.

---

## 4. Phase 3 — Statistical headline (Week 4)

**Objective:** Gate B numbers for paper Table 1.

### 4.1 Full-scale eval

- **Ckpt:** Best from Phase 1 or 2  
- **Protocol:** dsn=500, sen=50, seeds `{42, 123, 456, 789, 2024}`  
- **Baselines:** L0 same protocol (re-run if needed for fairness)

```cmd
python -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 50 -dvc cuda:0 ^
  -dsn 500 -seed <SEED> -ckpt "<avbt_ckpt>" -enp avbt_headline_sen50 --meeting_step 6 --log_routing
```

### 4.2 Analysis script

Create or extend: `tools/meeting_validation/step6/analyze_avbt_results.py`

- Mean ± std over seeds  
- Paired bootstrap on swap-gap (AVBT vs L0)  
- Write `doc/results/meeting_validation/step6/avbt_headline_summary.json`  
- Plot: `avbt_swap_gap_bar.png`

### 4.3 Ablation table

| Row | Config |
|-----|--------|
| 1 | L0 |
| 2 | CEB |
| 3 | CEB + DEK |
| 4 | CEB + DEK + DBV (if implemented) |
| 5 | AVBL epoch-3 (negative control) |

### 4.4 Gate B decision

| Pass | Action |
|------|--------|
| ✅ Gate B | Phase 4 (competing editors) |
| Partial (swap 55–60%) | Proceed to competitors anyway; report as “strong improvement” not “SOTA beat” |
| ❌ rel collapse | Roll back to epoch-2-style early stop; revisit E3 reliability anchor |

---

## 5. Phase 4 — Beat the field (Weeks 5–6)

**Objective:** Gate C — same probes, multiple editors.

### 5.1 Baselines to run

| Editor | Train needed? | Eval harness |
|--------|---------------|--------------|
| LiveEdit L0 | No (existing ckpt) | `test_grounding_eval.py` |
| LiveEdit + AR | No | same |
| FT-VL | Yes or use existing if in repo | same |
| MEND | Yes | same |
| SERAC | Yes | same |
| IKE | Yes | same |
| **AVBT (ours)** | Best ckpt | same |

**Prerequisite:** Confirm each editor supports BLIP2 + `GroundingVLKEB` loader. Add thin adapter scripts under `tools/meeting_validation/step6/baselines/` if missing.

### 5.2 Second backbone

| Backbone | Scope |
|----------|-------|
| LLaVA-1.5 | Port AVBT flags; 1-epoch CEB train; dsn=100 eval × 3 seeds |

Defer InstructBLIP unless LLaVA works.

### 5.3 Output

- `doc/results/meeting_validation/comparison/avbt_vs_editors_summary.json`  
- Table for paper §6

---

## 6. Phase 5 — Paper package (Weeks 7–8)

### 6.1 Benchmark credibility

| Task | Detail |
|------|--------|
| Human audit | 300 samples from VLKEB-long: visual grounding, factuality, fluency |
| Agreement | Report Cohen's κ |
| Tier eval | Re-run headline on verified-grounded subset only |

### 6.2 Additional metrics

- ROUGE-L, BERTScore on long-form reliability split  
- Optional LLM-judge faithfulness (validate against human audit on 50 samples)

### 6.3 Writing map

| Section | Source |
|---------|--------|
| §1 Intro | Swap-gap 31 pp hook |
| §2 Related | AnyEdit, μKE, LiveEdit, MMKE — differentiate AVBT |
| §3 Problem | Step 1 probes |
| §4 Diagnosis | Steps 0, 2–4, DINO false-cross |
| §5 Method | AVBT: CEB + DEK (+ DBV ablation) |
| §6 Experiments | Matrix §5 of strategy doc |
| §7 Results | Headline table + ablations |
| §8 Limitations | Teacher-forced, single backbone if applicable |

Update [`POST_STEP6_GROUNDING_DECISIONS.md`](POST_STEP6_GROUNDING_DECISIONS.md) contribution bullet 3 when Gate B passes.

---

## 7. DBV ablation (optional, parallel to Phase 2–3)

Only if CEB+DEK plateaus below Gate B.

| Task | Detail |
|------|--------|
| Implement full-pool margin veto before MoE inject | `liveedit.py` `edit_with_moes` hook |
| Use margin from Step 4 offline analysis (~0.35 rel, −0.67 swap) | `VG_EDIT_DINO_SIM_ANALYSIS.md` |
| Eval as ablation row 4 | Compare to CEB+DEK |

**Stop if** token swap-gap unchanged (expected from prior VG-Edit work unless DEK fixes retrieval first).

---

## 8. Risk register

| Risk | Mitigation |
|------|------------|
| Reliability collapse (epoch-3 repeat) | Early stop; hinge τ; reliability anchor E3; λ≤0.25 |
| GPU OOM on dsn=500 sen=50 | Staged BLIP2 load; `--log_routing` before eval; sen=20 for dev |
| Competing editors not in repo | Prioritize FT + MEND; document integration effort |
| Single-seed false positive | No claim until 5-seed Gate B |
| Routing vs token protocol mismatch | Always log false_cross at same sen as token eval |
| LLaVA port delays paper | BLIP2-only submission with LLaVA in appendix |

---

## 9. Immediate next actions (this week)

1. **Implement E1–E3** (targeted false-cross, joint `inpt_extractor`, reliability anchor)  
2. **Run direction check** after E1 on GPU (`verify_avbl_direction.py`)  
3. **Sweep S2–S4** (1 epoch each, ~18 min/train)  
4. **Eval 3 seeds** on best sweep → `avbt_phase1_sweep_summary.json`  
5. **Compare to epoch-2** — if sweep beats 53% swap, freeze config for Phase 2  

---

## 10. File and path conventions

| Type | Path |
|------|------|
| Strategy | `doc/THESIS_AAAI_AVBT_STRATEGY.md` |
| This plan | `doc/THESIS_AAAI_AVBT_WORK_PLAN.md` |
| Phase summaries | `doc/results/meeting_validation/step6/avbt_phase*_*.json` |
| Checkpoints | `records/liveedit/blip2-opt-2.7b/VLKEB_avbt_*` |
| Eval runs | `eval_results/meeting_validation/step6/liveedit/blip2-opt-2.7b/VLKEB_GROUNDING-avbt_*` |
| Grounding JSON | `data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json` |
| L0 baseline ckpt | `records/.../VLKEB_long_baseline-.../epoch-20-i-27800-ema_loss-0.7936` |

---

## 11. Progress log (fill as you go)

| Date | Milestone | Result |
|------|-----------|--------|
| 2026-07-09 | Strategy + work plan written | — |
| | E1–E3 implemented | |
| | Gate A passed (3 seeds) | |
| | DEK implemented | |
| | Gate B passed (5 seeds) | |
| | Competing editors evaluated | |
| | LLaVA transfer | |
| | Human audit κ | |

---

*Update this log and checklist as phases complete. Cross-link final numbers back to [`THESIS_AAAI_AVBT_STRATEGY.md`](THESIS_AAAI_AVBT_STRATEGY.md) §2.3 targets.*
