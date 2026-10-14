# G5 — LLaVA L0 diagnosis

**Status:** complete (2026-10-11, lab RTX 4090)  
**Parent audit:** [`G5_METRIC_AUDIT.md`](G5_METRIC_AUDIT.md) §3.4  
**Ckpt:** `records/lab_g5/liveedit/llava-v1.5-7b/lab_g5_llava_L0_dn500_e20_s42_20261010_231536-2026.10.10-23.15.41/checkpoints/epoch-20-i-10200-ema_loss-2.2061`  
**Config:** [`configs/liveedit/llava-v1.5-7b.yaml`](../../../configs/liveedit/llava-v1.5-7b.yaml) (`lr_cut_it: [10000]`)  
**Train cmd:** `-bs 1 -dn 500 -eps 20` (see [`EXPERIMENT_TRACKER.md`](EXPERIMENT_TRACKER.md))  
**Raw JSON:** [`llava_l0_diagnosis.json`](llava_l0_diagnosis.json)  
**Glossary:** [`../../CVPR2027_GATE_LOG.md`](../../CVPR2027_GATE_LOG.md) § Glossary  

No retrain and no pipeline change in this note. Any fix is an M-item — discuss first.

---

## Checks

| # | Check | Question | Status |
|---|---|---|---|
| 1 | Loss curve | Did `ema_loss` plateau near 2.2 or still fall? Note `lr_cut_it: [10000]` | **done** |
| 2 | Train-split routing | Own text top-1 ~1% on ~50 **train** edits too? | **done** — 4% |
| 3 | Hidden-state scale | Norm / max / outliers at edit layer 21; `ivr_eqe_std`; `sim_scale` | **done** |
| 4 | Query slice | Tokens in `vt_range[1]:query_end` are the real question? | **done** — yes |

---

## 1. Loss curve

Source: TensorBoard under the L0 run `logs/events.out.tfevents.*` (10,400 steps = 20 × 500).

| Iter | `ema_loss` | Notes |
|---:|---:|---|
| 1 | 2.60 | EMA init |
| 100 | 14.65 | early spike |
| 1000 | 11.20 | |
| 3000 | 3.76 | first clear drop |
| 5000 | 6.39 | noisy |
| 8000 | 5.15 | |
| 9000 | 4.09 | |
| 9900 | 3.66 | just before LR cut |
| 10000 | 4.55 | `lr` 1e-4 → **1e-5** at i=10001 |
| **10200** | **2.21** | paper ckpt |
| 10300 | 3.47 | |
| 10400 | 2.69 | run end |

- `ema_loss` is **noisy and still moving** at the end (global min ≈ 1.85 at i=10208), not a flat stuck plateau at 2.2. It never reaches BLIP-2 L0 (~0.12).
- Token losses **did** train: reliability / generality → ~0 by mid-run.
- **Soft routing did not train.** Across all 10,400 steps:
  - `MoE soft routing relative loss` = **0** (expected under `bs=1`: softmax over a 1×1 matrix).
  - `MoE soft routing mean abstract positive similarity` ≡ `… negative similarity` on **every** step (also expected under `bs=1`: `abs_sim.roll(1,1)` is a no-op on a 1×1 matrix).
  - Abstract soft loss sits at **≈ ln(4) ≈ 1.386** whenever abs_pos ≈ 0.5 — the Bernoulli entropy of a vacuous positive=negative pair, not a real contrast.
- Hard routing is the only discriminative routing signal; neighbor/prototype losses move, which matches the working vision gate on eval.

**Reading:** not “needs a few more epochs.” The prompt-key (soft) objective was **mathematically inactive** for the whole L0 run because lab training used `-bs 1` (VRAM). BLIP-2 L0 used `-bs 4`.

---

## 2. Train-split routing

Command (lab): `tools/diagnose_llava_l0.py` on **50** edits from `grounding_train_short.json` (same file as L0 train), seed 42, reliability probe = exact edit prompt + edit image.

| Metric (reliability, train split) | Value |
|---|---:|
| Own in vision gate | **66%** |
| Own retrieved | **66%** |
| Own in text gate / text top-1 is own | **4%** |
| Mean own text rank (pool=50) | 25.7 (chance ≈ 25.5) |
| Mean own `txt_sim` | 0.088 |
| Mean max `fuse_coe` | 0.084 |
| Mean `ivr_eqe_std` | 0.013 |
| Mean experts retrieved | 12.2 |

Eval-split reliability (audit §3.1) had text top-1 own **1–2%**. Train-split is **4%** — same failure mode, not overfitting to the eval pool. Vision gate works on a majority of train edits; prompt-key routing is near chance.

---

## 3. Hidden-state scale

Edit layer **21** (`model.language_model.layers.21`). `module_dim=1024` → `sim_scale = 1/√1024 = 0.03125` (same formula and value as BLIP-2 config). Lab has **no BLIP-2 weights**; BLIP-2 `ivr_eqe_std` below is from the G5 audit’s existing `routing_diagnostics.json`.

| Quantity | LLaVA L0 (this diag, n=5 detail) | BLIP-2 (audit / G2–G3 logs) |
|---|---|---|
| `sim_scale` | 0.03125 | 0.03125 |
| `ivr_eqe_std` (mean, train Rel) | **0.013** | **~0.43** |
| `query_reps` mean token L2 | ~57–66 | *(not re-measured here)* |
| `query_reps` max \|·\| | ~34–39 | |
| `vision_reps` max \|·\| | **~160–203** | |
| Persistent outlier channels (query) | **1512, 2789** (mean \|·\| ≫ others) | |

LLaMA-family massive activations are present (vision max_abs ≫ 100; channel 1512 dominates query). That alone does not explain the dead text gate: `sim_scale` matches BLIP-2, and the soft-routing loss never saw a real negative. Collapsed `ivr_eqe_std` is consistent with an untrained / near-constant extractor output on the prompt-key path, not with a wrong query slice (see §4).

---

## 4. Query slice

From the live forward (`probe_routing` / `_forward_query_reps`):

| Backbone | `vision_token_n` | `query_token_n` (reliability) |
|---|---:|---:|
| LLaVA L0 (train, this diag) | **576** | **11–17** (mean 13.0) |
| BLIP-2 (audit / prior logs) | ~32 (Q-Former) | **41–45** |

Decoded `vt_range[1]:query_end` (HF processor, same string as `prompts_imgs_target_to_xym` + auto `<image>\n`):

```text
tokens ≈ ['▁', '<0x0A>', 'What', '▁city', '▁is', '▁dep', 'icted', '▁in', '▁this', '▁image', '?']
decoded ≈ "\nWhat city is depicted in this image?"
```

The slice is the **actual question** (plus the newline after `<image>`). No USER/ASSISTANT template residue. Shorter than BLIP-2 because LLaVA’s prompt wrap is shorter, not because the question was dropped.

---

## Verdict for Oct 15 G5 decision

| Branch | When | Action |
|---|---|---|
| Under-converged | Loss still falling / train Rel weak but gates partly work | Longer or full-set LLaVA L0 (discuss first) |
| Scale / slice / config bug | Diagnostics show pipeline mismatch | Pipeline fix (discuss first; M-item) |
| Unclear / not fixable by Oct 14 | No clear cheap fix | Disclose LLaVA as L0-failed limitation; BLIP-2 primary only |

**Decision (recommendation for discussion):** **config / training bug — `bs=1` made soft routing vacuous**, so the prompt-key path never learned. Query slice is fine; hidden-state scale is large but secondary. This is **not** “DEK Rel collapse” and **not** plain under-training.

**M7 decided 2026-10-11:** retrain LLaVA L0 with `bs≥2` (prefer 4; fall back 2 on OOM). Results write-up: [`G5_LLAVA_M7_BS_RETRAIN.md`](G5_LLAVA_M7_BS_RETRAIN.md). Do **not** treat longer `eps` at `bs=1` as a fix.
