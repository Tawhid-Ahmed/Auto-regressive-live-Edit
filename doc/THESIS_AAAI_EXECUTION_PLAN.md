# AAAI Execution Plan: Novelty + Credibility (Post DEK/FX)

**Date:** 2026-07-10  
**Status:** Active — supersedes outdated Phase-1/2 items in `THESIS_AAAI_AVBT_WORK_PLAN.md` that are already done  
**Audience:** Thesis + AAAI main-track  
**Related:** `THESIS_AAAI_AVBT_STRATEGY.md` · `short_vlkeb_gate_results.json` · `THESIS_PATHS.json`

---

## 0. Where you are (freeze this)

| Asset | Path / number |
|-------|----------------|
| Short baseline | `records/.../VLKEB_baseline-2026.04.09-08.20.07/.../epoch-20-i-25950-ema_loss-0.1201` — rel 93.5%, swap 40.7% |
| Best token swap (short) | DEK infer on S2 — **48.2%** swap, FC 53% |
| Best false-cross (short) | FX — **43%** FC (was ~96%), swap 47.2% |
| Recommended method ckpt | **DEK train** `VLKEB_avbt_short_DEK-2026.07.10-18.13.18/.../epoch-1-i-125-ema_loss-4.0100` (swap≈DEK, FC 46%; cleaner than FX for ablations) |
| Long-form prior best | epoch-2 AVBL — rel 79.5%, swap 53.2% (dsn=100) — still below AAAI ≥60% |

**Falsified (do not revisit as primary bets):** λ/dn sweeps, retrieval-margin logits, legacy `--fused_expert_key`, inference-only DINO gates as main method, token-refuse-only FX for +swap.

**Novelty you already own:**
1. Problem — visual grounding leakage / swap-gap in lifelong VLKE  
2. Mechanism — **AVBT + DEK** (vision ∧ text retrieval agreement) + DINO neighbor mining  
3. Evidence — FC ~96% → ~43–46% without collapsing reliability  

**What AAAI still lacks:** competitor table, 5 seeds, long-form headline, significance, (optional) 2nd backbone.

---

## 1. Paper identity (locked — OpenReview abstract submitted)

**Title:** *Dual-Expert Keys for Reducing False-Positive Retrieval in Lifelong Vision–Language Model Editing*

**Abstract:** see [`AAAI_SUBMISSION_DRAFT.md`](AAAI_SUBMISSION_DRAFT.md) (submitted text; do not change substance before 28 July).

**Three contribution bullets (intro / full paper):**
1. **Problem:** Lifelong VLM editors leak under image-swap / hard-neg probes (~40–43% swap while reliability stays high; false-positive / false-cross retrieval ~90%).  
2. **Diagnosis:** Leakage is dominated by **wrong-expert retrieval** (false-cross), not soft overwrite; weak visual keys.  
3. **Method:** **Dual-Expert Keys** (vision∧text dual-gate) + **AVBT** (adversarial binding) cut swap false-positive retrieval **96%→46%** with reliability ~94%.

**Headline metrics (in this order):**
1. `image_swap_false_cross_rate` (routing)  
2. Swap / hard-neg token acc + **swap-gap** (= rel − swap)  
3. Reliability (must hold)

Do **not** lead with AR, patch-retrain, or “+0.5 pp reliability.”

---

## 2. AAAI gates (pass/fail)

| Gate | Protocol | Pass criteria | If fail |
|------|----------|---------------|---------|
| **G0 — Story freeze** | Docs only | FC + DEK as headline; stop short loss sweeps | — |
| **G1 — Competitor probe** | Short grounding, dsn=100, sen=10, seed=42 | Ours beats ≥3 editors on FC **and** swap-gap | Still write diagnosis paper; defer AAAI main |
| **G2 — Multi-seed** | Best method, seeds 42/123/456/789/1024 | Mean swap ≥ baseline+5 pp; FC mean ≤55%; p&lt;0.05 vs LiveEdit on FC or swap-gap | Report variance; don’t claim SOTA |
| **G3 — Long-form headline** | `grounding_eval_longform.json`, dsn=100→500, sen=10→50 | Swap ≥55% (stretch 60%); rel ≥70%; FC trending down | Thesis-first; AAAI as workshop / next cycle |
| **G4 — Ablation + audit** | Table + 100–300 human-checked probes | Clean ablations; κ reported | Required for submit |

**AAAI submit only if G1 + G2 pass and G3 is at least “directional” (swap ≥50% long-form with clear FC win).**

---

## 3. Environment & conventions

```cmd
conda activate liveedit
cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"
set PYTHONUNBUFFERED=1
```

Always use:
```cmd
E:\anaconda\envs\liveedit\python.exe -u -W ignore ...
```
(not `conda run` — tqdm visibility; avoids Windows paging issues)

**Canonical short eval:**
```cmd
E:\anaconda\envs\liveedit\python.exe -u -W ignore test_grounding_eval.py ^
  -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt "CKPT_PATH" -enp POSTFIX ^
  -dpath "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json" ^
  --meeting_step 6 --patch_grounding --dek_dual_gate --log_routing
```

**Canonical long-form eval** (same flags, change JSON + postfix):
```cmd
  -dpath "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json" ^
  -enp long_DEK ...
```

After each run, record into:
`doc/results/meeting_validation/step6/aaai_run_log.json` (create/append).

---

## 4. Phase A — Competitor table (Week 1) 【highest AAAI ROI】

**Goal:** Prove the failure mode is shared; your method is the first that attacks it.

### A.1 Editors to run (minimum 4 + ours)

| Editor CLI `-en` | Notes |
|------------------|-------|
| `liveedit` | L0 = short baseline ckpt, **no** `--patch_grounding` / no DEK |
| `ft_vl` | Fine-tune baseline |
| `mend_vl` | If ckpt/train available |
| `serac_vl` | If ckpt/train available |
| `liveedit` + DEK | Ours — DEK ckpt + `--patch_grounding --dek_dual_gate` |

If MEND/SERAC need training first, prioritize **LiveEdit L0 + FT_VL + (IKE/LTE if available) + Ours**; document missing editors as limitation.

### A.2 Commands — LiveEdit L0 (no DEK)

```cmd
E:\anaconda\envs\liveedit\python.exe -u -W ignore test_grounding_eval.py ^
  -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_baseline-2026.04.09-08.20.07/checkpoints/epoch-20-i-25950-ema_loss-0.1201" ^
  -enp aaai_L0_short ^
  -dpath "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json" ^
  --meeting_step 6 --log_routing
```

### A.3 Commands — Ours (DEK)

```cmd
E:\anaconda\envs\liveedit\python.exe -u -W ignore test_grounding_eval.py ^
  -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_avbt_short_DEK-2026.07.10-18.13.18/checkpoints/epoch-1-i-125-ema_loss-4.0100" ^
  -enp aaai_DEK_short ^
  -dpath "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json" ^
  --meeting_step 6 --patch_grounding --dek_dual_gate --log_routing
```

### A.4 Commands — FT_VL (example)

```cmd
E:\anaconda\envs\liveedit\python.exe -u -W ignore test_grounding_eval.py ^
  -en ft_vl -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt "PATH_TO_FT_CKPT_OR_none" ^
  -enp aaai_FT_short ^
  -dpath "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json" ^
  --meeting_step 6 --log_routing
```

Train FT if needed (adjust to your existing FT recipe):
```cmd
E:\anaconda\envs\liveedit\python.exe -u -W ignore train_vllm_editor.py ^
  -en FT_VL -mn blip2 -dna VLKEB -bs 4 -dvc cuda:0 -dn 500 -eps 1 -rs 42 ^
  -tnp VLKEB_ft_short_aaai -dpath "data/VLKEB/train.json"
```

### A.5 Deliverable table

Write `doc/results/meeting_validation/step6/aaai_competitor_short.json`:

| Method | Rel ↑ | Swap ↑ | Hard-neg ↑ | FC ↓ | Swap-gap ↓ |
|--------|------:|-------:|-----------:|-----:|-----------:|
| FT_VL | | | | | |
| MEND / SERAC | | | | | |
| LiveEdit L0 | | | | | |
| **AVBT+DEK (ours)** | | | | | |

**G1 pass:** Ours best FC and best (or tied-best) swap-gap among all.

---

## 5. Phase B — Multi-seed credibility (Week 1–2)

**Goal:** Show DEK is stable, not a seed-42 fluke.

```cmd
for %S in (42 123 456 789 1024) do (
  E:\anaconda\envs\liveedit\python.exe -u -W ignore test_grounding_eval.py ^
    -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed %S ^
    -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_avbt_short_DEK-2026.07.10-18.13.18/checkpoints/epoch-1-i-125-ema_loss-4.0100" ^
    -enp aaai_DEK_s%S ^
    -dpath "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json" ^
    --meeting_step 6 --patch_grounding --dek_dual_gate --log_routing
)
```

Also run L0 on the same 5 seeds (no DEK flags) for paired comparison.

**Aggregate** into `doc/results/meeting_validation/step6/aaai_dek_multiseed.json`:
- mean ± std: rel, swap, hard-neg, FC, swap-gap  
- paired bootstrap or Wilcoxon on FC and swap-gap vs L0  

**G2 pass:** mean FC ≤ 55%; mean swap ≥ L0 + 5 pp; significance on at least one of FC / swap-gap.

---

## 6. Phase C — Long-form headline (Week 2–3)

**Goal:** Match the AAAI table domain (long targets), not only short VLKEB.

### C.1 Eval DEK/FX on long-form (no retrain first)

```cmd
E:\anaconda\envs\liveedit\python.exe -u -W ignore test_grounding_eval.py ^
  -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_avbt_short_DEK-2026.07.10-18.13.18/checkpoints/epoch-1-i-125-ema_loss-4.0100" ^
  -enp aaai_DEK_long_transfer ^
  -dpath "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_longform.json" ^
  --meeting_step 6 --patch_grounding --dek_dual_gate --log_routing
```

Expect domain mismatch risk (short-trained → long eval). If rel collapses, **retrain on long grounding**.

### C.2 Retrain AVBT+DEK on long (if transfer fails)

Use your existing long grounding train JSON (or build from long train if needed):

```cmd
E:\anaconda\envs\liveedit\python.exe -u -W ignore train_vllm_editor.py ^
  -en LiveEdit -mn blip2 -dna VLKEB -bs 4 -dvc cuda:0 -dn 500 -eps 2 -sci 125 -rs 42 ^
  -tnp VLKEB_avbt_long_DEK ^
  -lkpt "PATH_TO_LONG_OR_SHORT_BASELINE" ^
  -dpath "data/VLKEB/grounding/grounding_train_longform.json" ^
  --patch_grounding --adversarial_grounding ^
  --adv_grounding_lambda 0.2 --adv_grounding_neg_weight 2.0 ^
  --avbl_hinge_tau 0.3 --avbl_contrastive ^
  --dek_dual_gate --dek_text_alpha 1.0
```

Then eval dsn=100 (gate) → dsn=500 sen=50 (headline) with `--dek_dual_gate --log_routing`.

**G3 pass (realistic):** long-form swap ≥55%, rel ≥70%, FC clearly below L0.  
**G3 stretch (strategy doc):** swap ≥60%, swap-gap ≤12 pp.

---

## 7. Phase D — Optional last technical bet (only if G3 &lt; 55% swap)

**Do not** retune λ. Implement **pool-scale hard-negative mining**:

| Item | Spec |
|------|------|
| Idea | Maintain a frozen buffer of recent edit (image, eqr, moe) outside the batch of 4; mine DINO-nearest non-own expert from buffer for AVBL neighbor + refuse |
| Why | Batch-4 mining ≠ eval false-cross at sen=10 / pool=100 |
| Files | `liveedit.py` (`_mine_dino_false_cross_neighbor`, train buffer) |
| Flag | e.g. `--avbl_pool_buffer 64` |
| Gate | Short seed-42: swap ≥50% **and** FC ≤40% before long retrain |

Skip if G1–G2 already strong and G3 ≥50% — go write the paper.

---

## 8. Phase E — Ablations, figures, audit (Week 3–4)

### E.1 Ablation rows (same protocol)

| Row | Flags / ckpt |
|-----|----------------|
| L0 | baseline, no patch |
| AVBT only (S2) | S2 ckpt + `--patch_grounding` only |
| AVBT+DEK | DEK ckpt + `--dek_dual_gate` |
| AVBT+DEK+FX | FX ckpt + `--dek_dual_gate` |
| Negative | `--fused_expert_key` (show collapse) |

### E.2 Figures (paper-ready)
1. Swap-gap bar: L0 vs competitors vs DEK  
2. False-cross rates (routing)  
3. Reliability vs swap scatter (seeds)  
4. Architecture: DEK dual-gate diagram (you already have mmd/png drafts)

### E.3 Human audit
- Sample 100–300 long-form probes; 2 annotators: “is swap image a fair confusable?”  
- Report agreement κ in appendix  

---

## 9. Week-by-week schedule

| Week | Focus | Exit criterion |
|------|--------|----------------|
| **1** | Phase A competitors + start Phase B seeds | G1 table drafted |
| **2** | Finish 5-seed; long-form transfer eval | G2; know if C.2 retrain needed |
| **3** | Long DEK train/eval **or** pool-buffer bet | G3 directional |
| **4** | Ablations, figures, audit, draft §§3–6 | Paper skeleton |
| **5–6** | Related work, limitations, 2nd backbone smoke (optional) | AAAI draft |

---

## 10. Stop / go decision tree

```text
G1 competitors: DEK wins FC + swap-gap?
  NO  → Thesis + workshop; do not claim AAAI SOTA
  YES → G2 multi-seed significant?
           NO  → Fix variance / more data; delay submit
           YES → G3 long-form swap ≥55%?
                    NO  → Phase D pool-buffer once; else thesis-first
                    YES → Prepare AAAI submit (ablations + audit + 2nd backbone if time)
```

---

## 11. Immediate next commands (do today)

**Step 1 — Confirm DEK short numbers for AAAI naming:**
```cmd
E:\anaconda\envs\liveedit\python.exe -u -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_avbt_short_DEK-2026.07.10-18.13.18/checkpoints/epoch-1-i-125-ema_loss-4.0100" -enp aaai_DEK_short -dpath "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json" --meeting_step 6 --patch_grounding --dek_dual_gate --log_routing
```

**Step 2 — L0 paired baseline with routing:**
```cmd
E:\anaconda\envs\liveedit\python.exe -u -W ignore test_grounding_eval.py -en liveedit -mn blip2 -sen 10 -dvc cuda:0 -dsn 100 -seed 42 -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_baseline-2026.04.09-08.20.07/checkpoints/epoch-20-i-25950-ema_loss-0.1201" -enp aaai_L0_short -dpath "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json" --meeting_step 6 --log_routing
```

**Step 3 — Start FT_VL or next available competitor on the same JSON.**

**Step 4 — Append results to** `doc/results/meeting_validation/step6/aaai_competitor_short.json`.

---

## 12. Novelty checklist (reviewer lens)

| Claim | Evidence you must show | Status |
|-------|------------------------|--------|
| New failure mode | Swap-gap + FC across editors | Partial (need competitors) |
| Mechanism | DEK dual-gate + AVBT | Done |
| Beats prior editors | Table + significance | Missing |
| Not a fluke | 5 seeds | Missing |
| Not short-only artifact | Long-form | Missing |
| Honest negatives | fused_expert_key, L1–L4, decode gates | Mostly done |

---

## 13. What “done for AAAI” looks like

One main results table (long-form, 5-seed mean):

| Method | Rel | Swap | Hard-neg | FC | Swap-gap |
|--------|----:|-----:|---------:|---:|---------:|
| FT | | | | | |
| MEND | | | | | |
| SERAC | | | | | |
| LiveEdit | ~75 | ~43 | ~43 | ~88 | ~31 |
| **AVBT+DEK** | ≥72 | ≥55 (ideal ≥60) | ≥53 | ≤50 | ≤20 (ideal ≤12) |

Plus ablation + routing figure + limitations (teacher-forced eval, single primary backbone if LLaVA not ready).

---

*End of plan. Execute Phase A next; do not start another short λ sweep.*
