# G5 — LLaVA DEK root cause and fix plan

**Status:** findings **complete**; S2-tune A **FAIL**; **G5 board WEAK accepted 2026-10-15** ([`G5_GATE_WRITEUP_LLAVA.md`](G5_GATE_WRITEUP_LLAVA.md)); **M8 proposed** (discuss before implement); Phase C deferred; no DEK  





**Scope:** why the M7 LLaVA ladder (L0 `bs=2` → S2/DEK `bs=1`) still does not reproduce the BLIP-2 DEK pattern (lower `total_leak` with Rel held)  
**Inputs:** per-probe `routing_diagnostics.json` of the three M7 eval dirs (`vis_sim`, `vis_sim_prot`, `txt_sim`, `vision_ok_indices`, `text_ok_indices`, `pool_fuse_coe`); BLIP-2 AAAI L0/DEK dirs (July 10; no `txt_sim` logged); `leakage_taxonomy_llava_m7.json`; `leakage_taxonomy_vlkeb_s42.json`  
**Related:** [`G5_LLAVA_M7_BS_RETRAIN.md`](G5_LLAVA_M7_BS_RETRAIN.md) (results), [`G5_LLAVA_L0_DIAGNOSIS.md`](G5_LLAVA_L0_DIAGNOSIS.md) (L0 `bs=1` bug), [`G5_METRIC_AUDIT.md`](G5_METRIC_AUDIT.md)  
**Glossary:** [`../../CVPR2027_GATE_LOG.md`](../../CVPR2027_GATE_LOG.md) § Glossary  

Pool = 100 experts, n = 100 per probe, VLKEB short, seed 42. "Own" = `expected_indices`. Vision gate = `vis_sim > vis_sim_prot`; text gate (`routing_rule=and`, `dek_text_margin=0`) = prompt-key argmax.

---

## Part A — Findings

### F1. After S2/DEK the vision gate accepts the whole pool, so DEK degenerates to text-only routing

| Run | Probe | Own vision top-1 ↑ | Experts passing vision gate (of 100) ↓ | mean(own `vis_sim` − `vis_sim_prot`) |
|---|---|---:|---:|---:|
| BLIP-2 L0 | reliability | 98% | 4.0 | +10.0 |
| BLIP-2 L0 | image_swap | 3% | 3.9 | **−6.3** |
| BLIP-2 DEK | reliability | 99% | 7.6 | +6.9 |
| BLIP-2 DEK | image_swap | 3% | 10.4 | **−2.0** |
| LLaVA M7 L0 | reliability | **25%** | 15.6 | +17.1 |
| LLaVA M7 L0 | image_swap | 1% | 17.3 | **−10.0** |
| LLaVA M7 S2 | reliability | 41% | **98.1** | +56.4 |
| LLaVA M7 S2 | image_swap | 2% | **97.0** | **+43.4** |
| LLaVA M7 DEK | reliability | 40% | **99.9** | +64.3 |
| LLaVA M7 DEK | image_swap | 4% | **99.7** | **+46.9** |

- BLIP-2 and LLaVA **L0** both flip the sign of the own-vs-prototype margin between `reliability` and `image_swap` — the gate discriminates own image from swapped image.
- After LLaVA **S2/DEK** every expert, including own on a swapped image, beats the prototype by ~45–65. Logged `vision_ok_indices` has mean length 99.9. The gate is open.
- With `and` + margin 0, `text_ok` is a single expert (the prompt-key argmax). Open vision gate ∧ argmax = **argmax alone**. Hence DEK Rel hit 45% ≡ text own-top-1 45%, and swap / hard-negative outcomes ≡ whatever the text argmax picks.
- Confidence: **high** (directly logged).

### F2. The prompt-key channel is weak, image-conditioned, and degrades along the ladder

| Run | Own text top-1, `reliability` ↑ | Own text top-1, `image_swap` | Text argmax unchanged between `reliability` and `image_swap` of the same edit (same prompt, other image) | Spearman(`vis_sim`, `txt_sim`) |
|---|---:|---:|---:|---:|
| LLaVA M7 L0 | 70% | 23% | 35% | 0.38 |
| LLaVA M7 S2 | 56% | 15% | 31% | 0.29 |
| LLaVA M7 DEK | 45% | 13% | 36% | 0.27 |

- Changing only the image moves the prompt-key argmax in ~65% of probes, and it lands on a **foreign** expert (partner is in the pool for only 1% of swaps, F4) → `foreign_leak` 85% in the DEK binary table.
- Mechanism: the key is extracted from layer-21 question-token states that attend over 576 image tokens; on LLaVA the "text" key is a second vision-dominated key, not an independent channel. BLIP-2 (32 Q-Former tokens) is expected to be less image-conditioned, but the July logs have no `txt_sim`; Phase C re-extract is needed for the BLIP-2 column.
- Own text top-1 falls 70 → 56 → 45% from L0 to DEK: S2/DEK made the key they depend on worse.
- Confidence: **high** for LLaVA numbers; **medium** for the mechanism (BLIP-2 comparison column missing).

### F3. Cause of F1/F2: VRAM-forced batch sizes starve every contrastive routing objective

All routing losses in `train_a_batch` draw negatives from the batch:

| Objective | `bs=4` (BLIP-2) | `bs=2` (LLaVA L0) | `bs=1` (LLaVA S2/DEK) |
|---|---|---|---|
| Soft routing relative loss (`rela_sim` [b,b] softmax) | 3 negatives | 1 | **none — loss ≡ 0** (same bug as L0 diagnosis) |
| Soft routing abstract loss (`abs_sim.roll(1,1)`) | 3 | 1 | **none — positive ≡ negative** |
| Hard routing neighbor loss (softmax over [b experts, prototype]) | 3 + prototype | 1 + prototype | **prototype only** |
| AVBL binding softmax (`evr_pool = batch`; `wrong_idx` = other batch experts; `avbl_targeted_false_cross`, `avbl_contrastive`) | 3 foreign + prototype | 1 + prototype | **prototype only; `wrong_idx` empty → no foreign suppression possible** |
| `--avbl_pool_buffer` (existing cross-batch buffer) | DINO neighbor mining + refuse loss only; **not** in the routing/binding softmax | same | same |

- In S2/DEK the only learnable quantity was the own-vs-prototype margin; with LLaVA features it moved the wrong way (prototype collapsed, F1). The prompt-key path received no contrastive gradient at all (F2 decline).
- M7 fixed L0 but **re-introduced vacuous routing in the two stages that define DEK**. S2 at `bs=2` with AVBL was attempted and OOMed on the 4090.
- Confidence: **high** that the losses are degenerate at `bs=1` (follows from the code); **medium** that this alone explains the prototype collapse (needs the `bs≥2` S2/DEK run in Part B to confirm).

### F3b. Secondary (unverified): LLaVA's L0 vision key is already weak

Own vision top-1 25% vs 98% on BLIP-2, mean own rank 8.1 vs 1.0. The L0 hidden-state diagnosis shows persistent massive-activation channels at layer 21 (1512, 2789; mean |·| ≈ 26 vs p99 ≈ 2.4) that plausibly dominate the dot-product routing similarity after the extractor. Not measured on extractor outputs; needs a lab run.

### F4. Protocol confound in the BLIP-2 vs LLaVA comparison (inflates BLIP-2; does not explain LLaVA)

| | BLIP-2 AAAI (July) | LLaVA M7 |
|---|---|---|
| Row sampling (scorer mapping check) | `head` (`rows[:n]`) | `subsample` (post-July seeded loader) |
| `swap_partner_in_pool_rate` | **0.73** | **0.01** |
| `hard_neg_partner_in_pool_rate` | 0.69 | 0.06 |
| `swap_prompt_duplicated_rate` | 0.40 | 0.22 |

With the partner in the pool, routing the swapped image to its true owner is `correct_route` (34% of BLIP-2 DEK swaps). On LLaVA that outcome is unreachable by construction; any non-own activation is `foreign_leak`. The two backbones cannot share a table until BLIP-2 is re-extracted with the current loader (Phase C). Expect the BLIP-2 DEK swap headline to move when `correct_route` disappears — same issue as the open `blip2-tau-check` P0.

### Verdict

Not "DEK does not work on LLaVA", and not residual L0 under-training. The chain is: `bs=1` S2/DEK → degenerate routing losses → open vision gate + weaker prompt key → `and` gate becomes text-argmax-only → text argmax is image-conditioned and lands on foreign experts.

---

## Part B — Fix plan (proposed; each step needs a go-ahead before implementation)

Order is by expected information per GPU-hour. Steps 1–3 are **engineering / protocol fixes** (no method change). Steps 4–6 are **method changes** (new M-items; discuss first, add to the gate log on approval).

| # | Step | Fixes | What | Acceptance gate (checked before the next step) | Cost |
|---|---|---|---|---|---|
| 1 | **Routing health gate in the pipeline** (diagnostic only) — **done 2026-10-12** | — | `tools/diagnose_llava_l0.py --health_gate`: reliability + `image_swap` on train split; wired after S2 and DEK in `tools/lab_g5_llava_m7_bs_pipeline.sh` (`run_routing_health_gate`). Checks: mean vision-ok % of pool ≤ `--max_vision_ok_pct` (default 20); mean own-vs-prototype margin on `image_swap` < 0; text top-1 ≥ L0 baseline + `--text_top1_min_delta_pp` (default −5). Exit code 3 on fail. Text-argmax stability logged as info only. | Pass count on `reliability` ≤ 20/100; own margin < 0 on `image_swap`; own text top-1 after S2/DEK ≥ L0 − 5 pp | CPU + one ~5 min probe per stage |
| 2 | **Fit S2/DEK at real `bs≥2`** (engineering) — **done 2026-10-12** | F3 | Lab smoke: **`bs=4` still OOM**; **`bs=2` OK** with the VRAM pack below. Soft routing is non-vacuous (TB: relative loss max ~3.5e-5 > 0; abstract pos≠neg). Changes: (a) AVBL probe mid-forwards under `torch.no_grad()`; (b) `--avbl_max_adv_images 1` (swap-only); (c) locality teacher logits under `no_grad`; (d) `--no_patch_hard_routing`; (e) `--train_max_gen_types 2` (was OOM on simultaneous gen mid graphs, not AVBL). Defaults wired in `lab_g5_llava_fresh_pipeline.py`. Smoke log: `logs/s2_step2_smoke_outer_20261012_094316.log`. | S2 smoke at `bs=2` without OOM; soft routing relative loss > 0 | ~30 min smoke |
| 3 | **Re-run S2 → DEK from the M7 L0 at `bs≥2`**, then same-protocol extract + taxonomy; **Phase C** BLIP-2 L0/S2/DEK re-extract with the current `subsample` loader so F4 disappears | F1, F3, F4 | Same `dn=500 × 1`, AVBT, `--dek_dual_gate`; same eval (`dsn=100`, `sen=10`, seed 42). **S2 retry done 2026-10-12** (`lab_g5_llava_S2_bs2_dn500_e1_s42_20261012_103239`, `adv_lambda=0.1`, AVBL clamps): finite loss (EMA≈8.43 @250, AVBL≈4e-5); health **FAIL** — vision-ok 33.2% (>20), swap own–prot margin +6.73 (need < 0); text top-1 **78% PASS**. vs M7 `bs=1` S2: vision-ok 98→33, swap margin +43→+7, text 56→78. DEK aborted by fail-fast. Artifact: `routing_health_s2_s3_bs2.json`. | Step-1 gate passes; DEK Rel hit ≥ 80%; hard-negative `total_leak` ≤ L0 − 20 pp **under the same loader on both backbones** | S2+DEK ≈ 1.5–2.5 h lab; Phase C ≈ 45 min local |
| 4 | **M8 — batch-independent negatives for routing** (method) | F3 permanently | Keep a FIFO memory bank of detached routing keys (`evr`, `eqr`, patch keys) from recent batches and add them as extra columns in the soft-routing, hard-routing-neighbor, and AVBL softmaxes (extends the existing `--avbl_pool_buffer`, which today only feeds DINO mining / refuse loss). Negatives then no longer depend on what fits in VRAM. Apply to both backbones for parity. | At `bs=1` the relative loss is non-zero and text own top-1 on train split ≥ 70%; no BLIP-2 regression on G2/G3 | ~1 day code + L0 retrain if applied to L0 (≈ 2 h) |
| 5 | **M9 — image-independent prompt key** (method; only if F2 persists after step 3) | F2 | Extract the prompt key from question-token states **before** image attention: a text-only forward of the prompt (no image) at the same layer, or a layer below the image-attention onset, cached once per edit. Dual gate then has two genuinely independent channels. | Text-argmax stability under image swap ≥ 80%; own text top-1 ≥ 85% on `reliability`; DEK `stale_leak` on `image_swap` stays ≤ L0 | ~1 day code; S2/DEK retrain ≈ 1.5 h |
| 6 | **M10 — outlier-channel normalization in the vision key** (method; only if own vision top-1 stays < 60% after step 3) | F3b | Measure extractor-output channel energy on the lab; if a handful of channels dominate, apply per-channel standardization or clipping before the routing dot product (both extractors, both backbones). | Own vision top-1 on `reliability` ≥ 80%; BLIP-2 unchanged within ±2 pp | half-day diag + L0 retrain ≈ 2 h |

### Decision points

- **After step 3:** if the step-1 gate passes and DEK beats L0 on hard-negative `total_leak` with Rel ≥ 80%, LLaVA G5 can be re-judged (PASS / WEAK) with no method change; steps 4–6 become optional robustness work. If the vision gate is still open at `bs≥2`, F3 was not the whole story → step 6 moves up. If the gate closes but text stays image-conditioned → step 5.
- **2026-10-12 step-3 S2 result:** vision gate still open at `bs=2` (though much tighter than `bs=1`).
- **2026-10-12 pre-M diag** (`routing_health_l0_s3diag.json`, `extractor_energy_s2_s3.json`, `s3_proto_foreign_breakdown.json`, `s3_pre_m_diag_summary.json`):
  - Same-probe L0 health **PASS** (vision-ok 12.3%, swap margin −13.7, text 70%). S2 **FAIL** is therefore an **S2 regression**, not leftover L0 weakness.
  - S2 open gate = **foreign inflation** (≈15.6 foreign experts beat prot on Rel; prot itself healthy mean ≈9.9) + **own stays hot on swap** (68% own still vision-ok under `image_swap`).
  - Extractor / pool / layer-21 L1 **not dominated** (top1 share ≪ 5% on all tensors; ch 1512 is top hidden but ~0.65% of L1). **M10 not justified** by energy — postpone unless a different metric shows harm.
  - Next preference: **M8** (batch-independent negatives; `bs=2` still has only 1 foreign). **M9** still optional. **Do not train DEK** until vision health passes. Phase C independent for F4.
- **2026-10-14 S2-tune A** (`lab_g5_llava_S2_…_tuneA_20261014_094604`, λ=0.1, `neg_w=4.0`, `contrast_m=0.3`, `retrieval_m=1.0`, finite loss): health **FAIL** — Rel vision-ok **62.0%** (S3 was 33.2%), swap margin **+13.11** (S3 +6.73), text top-1 74% PASS, text-argmax stability 62% (up from 50%). Cranking existing AVBL knobs **did not close the gate** and made vision openness worse → treat S2-tune as exhausted; **M8 is the remaining structural fix**. Artifact: `routing_health_s2_tune_a.json`.
- **Oct 15 G5 decision** should be taken on step-3 numbers, not on the current `bs=1` S2/DEK ladder.
- **Paper framing meanwhile:** disclose M7 and the `bs=1` S2/DEK limitation; do not quote LLaVA DEK leak numbers from the current ladder; do not quote BLIP-2 vs LLaVA side by side until Phase C.

### Step 3 S2 health (retry) vs prior ladders

Train-split health gate, `dsn=50`, seed 42 (same probe for L0 diag + S3 S2). Directions: vision-ok % of pool ↓; swap own–prot margin should be < 0; text top-1 ↑.

| Run | Rel vision-ok % ↓ | Swap own–prot margin | Rel text top-1 ↑ | Rel own retrieved ↑ | Health |
|---|---:|---:|---:|---:|---|
| LLaVA M7 L0 (`bs=2`, same probe) | **12.3** | **−13.7** | 70 | 72 | **PASS** |
| LLaVA M7 S2 (`bs=1`, eval n=100) | **98.1** | **+43.4** | 56 | 41 | FAIL (open gate) |
| LLaVA S3 S2 retry (`bs=2`, λ=0.1) | **33.2** | **+6.73** | **78** | **98** | **FAIL** (still open; text PASS) |
| LLaVA S2-tune A (`bs=2`, ret_m=1, neg_w=4) | **62.0** | **+13.11** | **74** | (see JSON) | **FAIL** (worse vision than S3) |

S3 ckpt: `…/lab_g5_llava_S2_bs2_dn500_e1_s42_20261012_103239-…/epoch-1-i-200-ema_loss-6.7143`. Tune A: `…/lab_g5_llava_S2_…_tuneA_20261014_094604-…/epoch-1-i-200-ema_loss-10.5129`.

### Pre-M diag verdict (2026-10-12)

| Question | Answer |
|---|---|
| Is L0 already broken under the health gate? | **No** — PASS |
| Did S2 reopen the gate? | **Yes** — vision-ok 12→33; swap margin −14→+7; foreign_mean_n ≈15.6 on Rel |
| Prototype collapsed? | **No** — Rel prot mean ≈9.9; problem is foreign experts also beat it |
| Own rejects under `image_swap`? | **Weak** — 68% still vision-ok; margin drops −24 but lands +6.7 |
| Extractor channel-dominated (M10)? | **No** — top1 L1 share <1% on `evr`/`ivr`; defer M10 |
| Preferred next method step | **M8** (S2-tune A exhausted); M9 optional; not M10 |
| S2-tune A (2026-10-14) | **FAIL** — vision-ok 33→62, swap margin +6.7→+13; do not keep cranking AVBL |

### Not to do

- More epochs at `bs=1` for S2/DEK (same vacuous objective).
- Raising `dek_text_margin` to "fix" Rel: it widens `text_ok` without closing the vision gate, so leak rises with Rel.
- Reporting `only_own` 45% as a precision win: it is an artifact of argmax-only routing.
- Comparing against the July BLIP-2 taxonomy under `head` sampling.

---

## Artifacts

| Item | Where |
|---|---|
| M7 eval dirs (`routing_diagnostics.json`) | lab `eval_results/meeting_validation/step6/liveedit/llava-v1.5-7b/VLKEB_GROUNDING-g5_llava_m7_{L0,S2,DEK}_short-2026.10.11-*` |
| BLIP-2 reference dirs | local `eval_results/meeting_validation/step6/liveedit/blip2-opt-2.7b/VLKEB_GROUNDING-aaai_{L0,DEK}_short-2026.07.10-*` |
| Taxonomy JSON (metadata rows used in F4) | `leakage_taxonomy_llava_m7.json`, `../g0/leakage_taxonomy_vlkeb_s42.json` |
| L0 hidden-state diagnosis (F3b) | `llava_l0_diagnosis_m7.json` |
| Step-3 S2 retry ckpt (`bs=2`) | lab `records/lab_g5/.../lab_g5_llava_S2_bs2_dn500_e1_s42_20261012_103239-.../checkpoints/epoch-1-i-200-ema_loss-6.7143` |
| Step-3 S2 health (FAIL) | `routing_health_s2_s3_bs2.json`; outer `logs/s3_s2dek_bs2_outer_20261012_103238.log` |
| Pre-M L0 health (PASS) | `routing_health_l0_s3diag.json` |
| Extractor energy L0/S2 | `routing_health_l0_s3diag.json` (`extractor_energy`), `extractor_energy_s2_s3.json` |
| Proto/foreign breakdown | `s3_proto_foreign_breakdown.json`; summary `s3_pre_m_diag_summary.json` |
| S2-tune A health (FAIL) | `routing_health_s2_tune_a.json`; outer `logs/s2_tune_a_outer_20261014_094548.log` |
