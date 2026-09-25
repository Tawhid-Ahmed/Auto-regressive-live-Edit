# AAAI Phase 2 — LLaVA Backbone Stress (execution plan)

**Date:** 2026-07-25  
**Status:** IN PROGRESS — L0 **`dn=500`, `eps=3`** (deadline-aware)  
**Paper deadline:** 2026-07-28  
**Parent plan:** [`../../../AAAI_STRESS_TRANSFER_PLAN.md`](../../../AAAI_STRESS_TRANSFER_PLAN.md)  
**EVQA (Phase 1) done:** [`AAAI_EVQA_STRESS_RESULTS.md`](AAAI_EVQA_STRESS_RESULTS.md) · Gate **G-D1 PASS**  
**Run status:** [`STATUS_LLAVA.md`](STATUS_LLAVA.md)

---

## 0. Goal

Show Dual-Expert Keys are **not BLIP-2-only**: L0 → S2 (AVBT) → DEK on **LLaVA-1.5-7B** + VLKEB short grounding probes.

| Gate | Criterion |
|------|-----------|
| **G-B0** | LLaVA weights + VLKEB paths OK; smoke load works |
| **G-B1** | LLaVA L0 (`dn=500`) finishes + grounding eval loads |
| **G-B2** | DEK FC &lt; L0 and ≤ S2; Rel held (exact 96→46 **not** required) |

**Claim to protect:** FC ↓ with Rel held — not token-swap SOTA.

---

## 1. Protocol (deadline-aware — **chosen**)

| Stage | Data | Size | Epochs | Notes |
|-------|------|------|--------|--------|
| **L0 (baseline)** | `data/VLKEB/train.json` | **`dn=500`** | **`3`** | Same *n* as S2/DEK; short L0 — **disclose** |
| **S2 (AVBT)** | `grounding_train_short.json` | **`dn=500`** | **1** | From L0; no `--dek_dual_gate` |
| **DEK** | same grounding short | **`dn=500`** | **1** | From L0 + `--dek_dual_gate` |
| **Eval** | `grounding_eval_short.json` | `dsn=100` first | — | Rel / FC swap / FC hard |

### Why `dn=500`, not `dn=1000` or full 5000

| Option | Wall-clock (≈23 s/it after fuse) | Verdict for 28 Jul |
|--------|----------------------:|--------------------|
| Full 5000 × 20 | ~weeks | Aborted |
| `dn=500` × 20 | ~2.5–3 days | Too slow |
| **`dn=500` × 3`** | **~9–10 h** | **Chosen** — directional L0; disclose |
| `dn=1000` × 20 | ~8–10 days | Rejected |

**Paper framing (required):** LLaVA uses short LiveEdit L0 (`n=500`, **3 epochs**) under compute limits; results are **directional** backbone stress. Primary evidence remains BLIP-2 (full L0) and EVQA multi-seed.

**Hardware:** `bs=1`, fp16, TF32 on; save every 100 steps.

**Aborted full run (do not use for paper):** `VLKEB_llava_L0_full-2026.07.25-18.41.57`

---

## 2. Paths

| Asset | Path |
|-------|------|
| LLaVA weights | `...\LiveEdit\models\llava-1.5-7b-hf` |
| Config | `configs/liveedit/llava-v1.5-7b.yaml` (`llm_layer_tmp: model.language_model.layers.{}`) |
| Records | `records/liveedit/llava-v1.5-7b/` |
| Logs | `doc/results/meeting_validation/step6/stress/logs/` |

**Train name prefixes**

| Stage | `-tnp` |
|-------|--------|
| L0 | `VLKEB_llava_L0_dn500` |
| S2 | `VLKEB_llava_S2_gnd` |
| DEK | `VLKEB_llava_DEK_gnd` |

---

## 3. Checklist

### Phase 2A — Prerequisites

- [x] G-B0 code fixes (fp16, layer path, processor, merge, lm_head)
- [x] Stopped full-5000 L0

### Phase 2B — L0 `dn=500`

- [ ] **T2.1** `powershell -File tools\stress\start_llava_l0.ps1` (`dn=500`, `eps=3`)
- [ ] **T2.1r** Resume: `tools\stress\resume_llava_l0.ps1`
- [ ] **T2.1done** Record final ckpt in `STATUS_LLAVA.md`

### Phase 2C–D — S2 / DEK / eval (after L0)

Same flags as BLIP2 short; then fill `aaai_stress_backbone_summary.json` + `AAAI_LLAVA_STRESS_RESULTS.md`.

### If L0 not done by 28 Jul

- Ship paper with **EVQA stress** + BLIP2 main tables  
- LLaVA → Limitations / “ongoing”  
- Continue L0 → S2 → DEK for camera-ready / arXiv update if allowed  

---

## 4. Commands

```text
# L0
powershell -File tools\stress\start_llava_l0.ps1

# Monitor
Get-Content doc\results\meeting_validation\step6\stress\logs\VLKEB_llava_L0_dn500.train.log -Wait -Tail 40
```

S2/DEK (after L0 ckpt):

```text
train_vllm_editor.py -en liveedit -mn llava -dna VLKEB -bs 1 -dvc cuda:0 -edvc 0
  -lkpt <L0_CKPT> -eps 1 -tnp VLKEB_llava_S2_gnd -dn 500 -rs 42 -sci 125
  -dpath data/VLKEB/grounding/grounding_train_short.json
  --patch_grounding --adversarial_grounding --adv_grounding_lambda 0.3
  --adv_grounding_neg_weight 2.5 --avbl_hinge_tau 0.3 --avbl_contrastive
# DEK: same + --dek_dual_gate -tnp VLKEB_llava_DEK_gnd
```
