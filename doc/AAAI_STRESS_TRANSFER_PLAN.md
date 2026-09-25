# AAAI Stress Plan — Other Models & Datasets (AVBT + Dual-Expert Keys)

**Date:** 2026-07-22  
**Status:** Ready to execute (assets partially missing — see Phase 0)  
**Primary claim to protect:** false-positive retrieval (FC) ↓ with reliability held — **not** token-swap SOTA  
**Canonical main result (already done):** BLIP2-OPT-2.7B + VLKEB short grounding, FC **96% → 46%**, Rel ~**94%**  
**Related:** [`THESIS_AAAI_EXECUTION_PLAN.md`](THESIS_AAAI_EXECUTION_PLAN.md) · [`POST_STEP6_GROUNDING_DECISIONS.md`](POST_STEP6_GROUNDING_DECISIONS.md) · [`THESIS_PATHS.json`](THESIS_PATHS.json)

---

## 0. Goal & scope

### 0.1 Why run this

Reviewers will ask: *Does Dual-Expert Keys only work on BLIP2 + your custom grounding JSON?*  
This plan answers with two stress axes:

| Axis | Target | Paper use |
|------|--------|-----------|
| **Dataset stress** | EVQA / EIC (MMEdit) + official VLKEB `eval.json` | Same method, new data |
| **Backbone stress** | **LLaVA-1.5-7B** on VLKEB short grounding | Same probes, new model |
| **Optional** | MiniGPT-4; `sen=50` lifelong; EVQA grounding probes | Only if Phase 1–2 pass |

### 0.2 What is in / out of scope

| In scope | Out of scope (do not block AAAI on these) |
|----------|-------------------------------------------|
| L0 vs S2 (AVBT) vs DEK tables | Beating LiveEdit full Rel/Gen/Loc leaderboard |
| FC + Rel as primary metrics | Token-swap ≥55% |
| One solid 2nd backbone (LLaVA) | Full 5-seed on every setting |
| Directional EVQA Rel/Gen/Loc | Short→long transfer as headline (already failed on BLIP2) |

### 0.3 Success gates (stop / go)

| Gate | Criterion | If fail |
|------|-----------|---------|
| **G-D0** | Models + MMEdit data on disk | Fix downloads; do not train |
| **G-D1** | EVQA: DEK Rel within ~3–5 pp of L0; no collapse | Tune λ / shorter train; delay LLaVA |
| **G-B1** | LLaVA L0 trains + loads grounding eval | Fix OOM / paths |
| **G-B2** | LLaVA DEK FC clearly &lt; L0/S2; Rel held | Debug DEK flags on LLaVA; report as limitation if stuck |
| **G-DONE** | One dataset table + one backbone table drafted | Enough for AAAI stress paragraph |

---

## 1. Task breakdown (checklist)

### Phase 0 — Prerequisites

- [ ] **T0.1** Confirm `ROOT_PATH` in `utils/GLOBAL.py`
- [ ] **T0.2** Confirm BLIP2 path resolves
- [ ] **T0.3** Download LLaVA-1.5-7B → `models/llava-v1.5-7b-hf`
- [ ] **T0.4** Download MMEdit / EasyEdit-MM → `data/easy-edit-mm/`
- [ ] **T0.5** (Optional) MiniGPT-4 weights
- [ ] **T0.6** Smoke: path existence script (below)
- [ ] **T0.7** Create results folder + empty JSON summary

### Phase 1 — Dataset stress (BLIP2, new data)

- [ ] **T1.1** Official VLKEB `eval.json` Rel/Gen/Loc: L0 vs DEK
- [ ] **T1.2** Train EVQA LiveEdit L0 (or reuse existing EVQA ckpt)
- [ ] **T1.3** Train EVQA AVBT-only (S2)
- [ ] **T1.4** Train EVQA AVBT+DEK
- [ ] **T1.5** Eval EVQA Rel/Gen/Loc (L0 / S2 / DEK)
- [ ] **T1.6** (Optional) Same for EIC
- [ ] **T1.7** Fill `aaai_stress_dataset_summary.json` + short MD note
- [ ] **T1.8** Pass **G-D1** before Phase 2

### Phase 2 — Backbone stress (LLaVA + VLKEB grounding)

**Execution plan (full L0):** [`results/meeting_validation/step6/stress/AAAI_LLAVA_PHASE2_PLAN.md`](results/meeting_validation/step6/stress/AAAI_LLAVA_PHASE2_PLAN.md) · status [`STATUS_LLAVA.md`](results/meeting_validation/step6/stress/STATUS_LLAVA.md)

- [ ] **T2.1** Train LLaVA LiveEdit L0 on VLKEB **`dn=500`, `eps=3`** (deadline-aware; disclose short L0) — `tools/stress/start_llava_l0.ps1` · plan [`results/meeting_validation/step6/stress/AAAI_LLAVA_PHASE2_PLAN.md`](results/meeting_validation/step6/stress/AAAI_LLAVA_PHASE2_PLAN.md)
- [ ] **T2.2** Train LLaVA S2 (AVBT, no dual-gate, `dn=500` grounding)
- [ ] **T2.3** Train LLaVA DEK (AVBT + `--dek_dual_gate`, `dn=500`)
- [ ] **T2.4** Grounding eval L0 / S2 / DEK on short probes + `--log_routing`
- [ ] **T2.5** Extract Rel / Swap / Hard / FC
- [ ] **T2.6** Fill `aaai_stress_backbone_summary.json` + comparison table vs BLIP2
- [ ] **T2.7** Pass **G-B2**

### Phase 3 — Optional (only if G-B2 passes)

- [ ] **T3.1** MiniGPT-4 smoke L0→DEK (dsn=50 first)
- [ ] **T3.2** LLaVA 3–5 eval seeds (if one seed is strong)
- [ ] **T3.3** `sen=50` lifelong stress (secondary)
- [ ] **T3.4** Adapt grounding builder for EVQA (only if time)

### Phase 4 — Paper integration

- [ ] **T4.1** Add “Cross-setting stress” subsection / table to Results
- [ ] **T4.2** One honest Limitations sentence (single primary backbone if LLaVA weak)
- [ ] **T4.3** Update `AAAI_SUBMISSION_DRAFT.md` / `.tex`

---

## 2. Recommended calendar (time-boxed)

| Day | Focus | Exit |
|-----|--------|------|
| **0** | Downloads + path smoke | G-D0 |
| **1** | EVQA L0 train start; VLKEB official eval L0/DEK | T1.1 underway |
| **2** | EVQA S2 + DEK train | ckpts on disk |
| **3** | EVQA eval + write Phase 1 table | **G-D1** |
| **4–5** | LLaVA L0 train | G-B1 |
| **6** | LLaVA S2 + DEK train | ckpts |
| **7** | LLaVA grounding eval + table | **G-B2 / G-DONE** |
| **+** | Optional Phase 3 | only if spare |

If AAAI full-paper deadline is tight: **skip EIC and MiniGPT**; do EVQA Rel/Gen/Loc + LLaVA short grounding only.

---

## 3. Paths & naming conventions

**Repo root:**

```text
E:\MSC\MSc thesis project\LiveEdit\LiveEdit
```

**Python:**

```text
E:\anaconda\envs\liveedit\python.exe
```

**Results (create once):**

```text
doc/results/meeting_validation/step6/stress/
  aaai_stress_run_manifest.json
  aaai_stress_dataset_summary.json
  aaai_stress_backbone_summary.json
  AAAI_STRESS_ANALYSIS.md
```

**Eval name prefixes (`-enp`):**

| Run | `-enp` |
|-----|--------|
| BLIP2 VLKEB official L0 | `stress_blip2_L0_vlkeb_eval` |
| BLIP2 VLKEB official DEK | `stress_blip2_DEK_vlkeb_eval` |
| BLIP2 EVQA L0 | `stress_blip2_L0_evqa` |
| BLIP2 EVQA S2 | `stress_blip2_S2_evqa` |
| BLIP2 EVQA DEK | `stress_blip2_DEK_evqa` |
| LLaVA grounding L0 | `stress_llava_L0_short` |
| LLaVA grounding S2 | `stress_llava_S2_short` |
| LLaVA grounding DEK | `stress_llava_DEK_short` |

**Train name prefixes (`-tnp`):**

| Run | `-tnp` |
|-----|--------|
| EVQA L0 | `EVQA_L0_stress` |
| EVQA S2 | `EVQA_avbt_S2_stress` |
| EVQA DEK | `EVQA_avbt_DEK_stress` |
| LLaVA L0 | `VLKEB_llava_L0_stress` |
| LLaVA S2 | `VLKEB_llava_S2_stress` |
| LLaVA DEK | `VLKEB_llava_DEK_stress` |

**CLI model flags (`-mn`):** use short names — `blip2`, `llava`, `minigpt` (resolved in `utils/__init__.py`).

---

## 4. Step-by-step guide

### Phase 0 — Prerequisites

#### T0.6 Path smoke

```cmd
cd /d "E:\MSC\MSc thesis project\LiveEdit\LiveEdit"

E:\anaconda\envs\liveedit\python.exe -c "from utils.GLOBAL import model_path_map, ROOT_PATH; import os; print('ROOT', ROOT_PATH); print({k: (v, os.path.isdir(v)) for k,v in model_path_map.items()}); print('VLKEB train', os.path.isfile('data/VLKEB/train.json')); print('easy-edit-mm', os.path.isdir('data/easy-edit-mm')); print('grounding short', os.path.isfile('data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json'))"
```

#### T0.3 LLaVA download (example)

Place Hugging Face `llava-hf/llava-1.5-7b-hf` under:

```text
E:\MSC\MSc thesis project\LiveEdit\LiveEdit\models\llava-v1.5-7b-hf
```

#### T0.4 MMEdit / EasyEdit-MM

Per LiveEdit README: download MMEdit assets into `data/easy-edit-mm/` so these exist:

```text
data/easy-edit-mm/vqa/vqa_train.json
data/easy-edit-mm/caption/caption_train_edit.json
data/easy-edit-mm/images/
```

#### T0.7 Create stress folder

```cmd
mkdir doc\results\meeting_validation\step6\stress
```

**Gate G-D0:** LLaVA dir = True **or** you explicitly defer backbone stress; `easy-edit-mm` = True for dataset stress.

---

### Phase 1 — Dataset stress (BLIP2)

#### T1.1 Official VLKEB Rel/Gen/Loc (quick)

Known BLIP2 ckpts:

```text
L0:  records/liveedit/blip2-opt-2.7b/VLKEB_baseline-2026.04.09-08.20.07/checkpoints/epoch-20-i-25950-ema_loss-0.1201
DEK: records/liveedit/blip2-opt-2.7b/VLKEB_avbt_short_DEK-2026.07.10-18.13.18/checkpoints/epoch-1-i-125-ema_loss-4.0100
```

```cmd
E:\anaconda\envs\liveedit\python.exe -u -W ignore test_vllm_edit.py ^
  -en liveedit -mn blip2 -dn VLKEB -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_baseline-2026.04.09-08.20.07/checkpoints/epoch-20-i-25950-ema_loss-0.1201" ^
  -enp stress_blip2_L0_vlkeb_eval

E:\anaconda\envs\liveedit\python.exe -u -W ignore test_vllm_edit.py ^
  -en liveedit -mn blip2 -dn VLKEB -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt "records/liveedit/blip2-opt-2.7b/VLKEB_avbt_short_DEK-2026.07.10-18.13.18/checkpoints/epoch-1-i-125-ema_loss-4.0100" ^
  -enp stress_blip2_DEK_vlkeb_eval
```

> If DEK eval needs patch modules at load time and crashes, re-run with the same enable flags your training used (check `test_vllm_edit.py` / editor load path). Prefer grounding eval for DEK FC; use this run for Rel/Gen/Loc parity.

#### T1.2 Train EVQA L0 (full train — AAAI protocol)

Use **full EVQA** (6345; omit `-dn`), same spirit as VLKEB L0 on full 5000.  
Logged / resumable wrapper: `tools/stress/start_evqa_l0.ps1` → `resume_evqa_l0.ps1`.

```cmd
E:\anaconda\envs\liveedit\python.exe -u train_vllm_editor.py ^
  -en liveedit -mn blip2 -dna EVQA -bs 4 -dvc cuda:0 -edvc 0 ^
  -eps 20 -tnp EVQA_L0_full -rs 42 -sci 400
```

Record final ckpt path under `records/liveedit/blip2-opt-2.7b/EVQA_L0_full-*/checkpoints/`.  
S2/DEK below keep `-dn 500` to match the main-paper DEK fine-tune stage.

#### T1.3 Train EVQA S2 (AVBT only)

```cmd
E:\anaconda\envs\liveedit\python.exe -u train_vllm_editor.py ^
  -en liveedit -mn blip2 -dna EVQA -bs 4 -dvc cuda:0 -edvc 0 ^
  -lkpt "<EVQA_L0_CKPT>" -eps 1 -tnp EVQA_avbt_S2_stress -dn 500 -rs 42 -sci 125 ^
  --patch_grounding --adversarial_grounding --adv_grounding_lambda 0.3 ^
  --adv_grounding_neg_weight 2.5 --avbl_hinge_tau 0.3 --avbl_contrastive
```

#### T1.4 Train EVQA DEK

```cmd
E:\anaconda\envs\liveedit\python.exe -u train_vllm_editor.py ^
  -en liveedit -mn blip2 -dna EVQA -bs 4 -dvc cuda:0 -edvc 0 ^
  -lkpt "<EVQA_L0_CKPT>" -eps 1 -tnp EVQA_avbt_DEK_stress -dn 500 -rs 42 -sci 125 ^
  --patch_grounding --adversarial_grounding --adv_grounding_lambda 0.3 ^
  --adv_grounding_neg_weight 2.5 --avbl_hinge_tau 0.3 --avbl_contrastive ^
  --dek_dual_gate
```

> If `--dek_dual_gate` is train-time only via a different flag name in your branch, match the exact flags used for `VLKEB_avbt_short_DEK` (see `RESUME_SHORT_VLKEB_GATE.md` / train logs).

#### T1.5 Eval EVQA

```cmd
E:\anaconda\envs\liveedit\python.exe -u -W ignore test_vllm_edit.py ^
  -en liveedit -mn blip2 -dn EVQA -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt "<EVQA_L0_CKPT>" -enp stress_blip2_L0_evqa

E:\anaconda\envs\liveedit\python.exe -u -W ignore test_vllm_edit.py ^
  -en liveedit -mn blip2 -dn EVQA -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt "<EVQA_S2_CKPT>" -enp stress_blip2_S2_evqa

E:\anaconda\envs\liveedit\python.exe -u -W ignore test_vllm_edit.py ^
  -en liveedit -mn blip2 -dn EVQA -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt "<EVQA_DEK_CKPT>" -enp stress_blip2_DEK_evqa
```

#### T1.7 Record Phase 1 results

Fill:

```text
doc/results/meeting_validation/step6/stress/aaai_stress_dataset_summary.json
```

Minimal schema:

```json
{
  "protocol": {"dsn": 100, "sen": 10, "seed": 42, "backbone": "blip2-opt-2.7b"},
  "vlkeb_official": {"L0": {}, "DEK": {}},
  "evqa": {"L0": {}, "S2": {}, "DEK": {}},
  "gate_G_D1": "pass|fail",
  "notes": ""
}
```

**Gate G-D1:** DEK Rel not collapsed vs L0. Then start Phase 2.

---

### Phase 2 — Backbone stress (LLaVA)

#### T2.1 Train LLaVA L0 on VLKEB (**dn=500** — deadline-aware)

Match S2/DEK sample count; **disclose** reduced-data L0 in paper (primary full-data result stays BLIP-2).  
Logged / resumable: `tools/stress/start_llava_l0.ps1` → `resume_llava_l0.ps1`.  
Detail: [`results/meeting_validation/step6/stress/AAAI_LLAVA_PHASE2_PLAN.md`](results/meeting_validation/step6/stress/AAAI_LLAVA_PHASE2_PLAN.md).

```cmd
E:\anaconda\envs\liveedit\python.exe -u train_vllm_editor.py ^
  -en liveedit -mn llava -dna VLKEB -bs 1 -dvc cuda:0 -edvc 0 ^
  -eps 3 -tnp VLKEB_llava_L0_dn500 -dn 500 -rs 42 -sci 100 ^
  -dpath data/VLKEB/train.json
```

Prefer **dn=500 over dn=1000** (fairer vs S2/DEK, ~2× faster). Full-5000 L0 aborted for July 28 deadline.

#### T2.2 Train LLaVA S2

Prefer grounding train JSON (matches BLIP2 AVBT recipe):

```cmd
E:\anaconda\envs\liveedit\python.exe -u train_vllm_editor.py ^
  -en liveedit -mn llava -dna VLKEB -bs 1 -dvc cuda:0 -edvc 0 ^
  -lkpt "<LLAVA_L0_CKPT>" -eps 1 -tnp VLKEB_llava_S2_stress -dn 500 -rs 42 -sci 125 ^
  -dpath data/VLKEB/grounding/grounding_train_short.json ^
  --patch_grounding --adversarial_grounding --adv_grounding_lambda 0.3 ^
  --adv_grounding_neg_weight 2.5 --avbl_hinge_tau 0.3 --avbl_contrastive
```

If `grounding_train_short.json` is missing, fall back to `data/VLKEB/train.json` and note it in the manifest.

#### T2.3 Train LLaVA DEK

```cmd
E:\anaconda\envs\liveedit\python.exe -u train_vllm_editor.py ^
  -en liveedit -mn llava -dna VLKEB -bs 1 -dvc cuda:0 -edvc 0 ^
  -lkpt "<LLAVA_L0_CKPT>" -eps 1 -tnp VLKEB_llava_DEK_stress -dn 500 -rs 42 -sci 125 ^
  -dpath data/VLKEB/grounding/grounding_train_short.json ^
  --patch_grounding --adversarial_grounding --adv_grounding_lambda 0.3 ^
  --adv_grounding_neg_weight 2.5 --avbl_hinge_tau 0.3 --avbl_contrastive ^
  --dek_dual_gate
```

#### T2.4 Grounding eval (same probes as main paper)

```cmd
set PROBE=data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json

E:\anaconda\envs\liveedit\python.exe -u -W ignore test_grounding_eval.py ^
  -en liveedit -mn llava -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt "<LLAVA_L0_CKPT>" -enp stress_llava_L0_short ^
  -dpath "%PROBE%" --meeting_step 6 --log_routing

E:\anaconda\envs\liveedit\python.exe -u -W ignore test_grounding_eval.py ^
  -en liveedit -mn llava -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt "<LLAVA_S2_CKPT>" -enp stress_llava_S2_short ^
  -dpath "%PROBE%" --meeting_step 6 --patch_grounding --log_routing

E:\anaconda\envs\liveedit\python.exe -u -W ignore test_grounding_eval.py ^
  -en liveedit -mn llava -sen 10 -dvc cuda:0 -dsn 100 -seed 42 ^
  -ckpt "<LLAVA_DEK_CKPT>" -enp stress_llava_DEK_short ^
  -dpath "%PROBE%" --meeting_step 6 --patch_grounding --dek_dual_gate --log_routing
```

#### T2.5–T2.6 Extract metrics & compare

From each run’s `seed_42_mean_results.json` (and routing logs), record:

| Method | Rel↑ | Swap | Hard | FC↓ |
|--------|-----:|-----:|-----:|----:|
| LLaVA L0 | | | | |
| LLaVA S2 | | | | |
| LLaVA DEK | | | | |
| BLIP2 DEK (reference) | ~93.7 | ~47.7 | ~42.0 | **46** |

Write:

```text
doc/results/meeting_validation/step6/stress/AAAI_STRESS_ANALYSIS.md
doc/results/meeting_validation/step6/stress/aaai_stress_backbone_summary.json
```

**Gate G-B2:** LLaVA DEK FC &lt; LLaVA L0 and ≤ LLaVA S2; Rel not collapsed. Exact 96→46 not required.

---

### Phase 3 — Optional

#### T3.1 MiniGPT smoke (only after G-B2)

Same recipe as LLaVA with `-mn minigpt`, `dsn=50` first eval, then `dsn=100` if stable.

#### T3.2 Multi-seed LLaVA

Re-run DEK vs L0 grounding eval for seeds `{0,1,2,42,123}` only if seed-42 DEK looks strong.

#### T3.4 EVQA grounding probes (later)

Adapt `tools/build_grounding_eval_json.py` only if you need FC on EVQA for the paper. Not required for first stress paragraph.

---

### Phase 4 — Paper text (after G-DONE)

Add a short Results paragraph, e.g.:

> To check that Dual-Expert Keys are not tied to a single backbone or probe file, we repeat the L0 / AVBT / DEK comparison on LLaVA-1.5-7B with the same VLKEB short grounding probes, and report Rel/Gen/Loc on EVQA under BLIP-2. In both settings, Dual-Expert Keys reduce false-positive retrieval (or preserve high reliability on EVQA) relative to the lifelong baseline, consistent with the main BLIP-2 result.

Keep Limitations honest if LLaVA FC gain is smaller than BLIP2.

---

## 5. Known risks & mitigations

| Risk | Mitigation |
|------|------------|
| LLaVA OOM | `bs=1`, smaller `dn`, close other GPU jobs |
| DEK flags crash on LLaVA | Confirm patch module + image size path; smoke 5 samples first |
| EVQA has no swap JSON | Report Rel/Gen/Loc first; FC only if routing logs exist |
| `grounding_train_short.json` missing | Use `data/VLKEB/train.json` + note in manifest |
| Deadline pressure | Drop EIC + MiniGPT; keep EVQA + LLaVA only |
| Over-claiming | Never say “works everywhere”; say “stress transfer is directional” |

---

## 6. Reference ckpts (BLIP2 — already done)

| Role | Path |
|------|------|
| L0 | `records/liveedit/blip2-opt-2.7b/VLKEB_baseline-2026.04.09-08.20.07/checkpoints/epoch-20-i-25950-ema_loss-0.1201` |
| S2 | `records/liveedit/blip2-opt-2.7b/VLKEB_avbt_short_S2-2026.07.10-11.16.19/checkpoints/epoch-1-i-125-ema_loss-4.0787` |
| DEK | `records/liveedit/blip2-opt-2.7b/VLKEB_avbt_short_DEK-2026.07.10-18.13.18/checkpoints/epoch-1-i-125-ema_loss-4.0100` |

Main probe JSON:

```text
data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json
```

---

## 7. Deliverables checklist

- [x] EVQA grounding train/eval JSON + L0/S2/DEK FC table (pilot)  
- [x] EVQA multi-seed dsn=300 reliability package (seeded subsample, 5 seeds)  
- [x] Results write-up linked from main analysis: [`doc/results/meeting_validation/step6/stress/AAAI_EVQA_STRESS_RESULTS.md`](results/meeting_validation/step6/stress/AAAI_EVQA_STRESS_RESULTS.md)  
- [ ] LLaVA backbone table (Phase 2)  
- [ ] Paper “Cross-setting stress” paragraph drafted  

- [ ] `stress/aaai_stress_run_manifest.json` (all ckpt + eval dirs)
- [ ] `stress/aaai_stress_dataset_summary.json`
- [ ] `stress/aaai_stress_backbone_summary.json`
- [ ] `stress/AAAI_STRESS_ANALYSIS.md` (1–2 pages: tables + gates)
- [ ] Results paragraph in `AAAI_SUBMISSION_DRAFT.md` / `.tex`
- [ ] Optional figure: BLIP2 vs LLaVA FC bars

---

## 8. Immediate next actions (today)

1. Run **T0.6** path smoke.  
2. Start **T0.3** LLaVA download + **T0.4** EasyEdit-MM download in parallel.  
3. When MMEdit is ready, launch **T1.2** EVQA L0 overnight.  
4. Do **not** start LLaVA training until **G-D0** (weights on disk) and preferably **G-D1**.

---

*End of plan. Update checkbox status in this file as you complete each task.*
