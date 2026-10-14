# G5 — M7 LLaVA L0 retrain (`bs≥2`) results

**Status:** **complete** (2026-10-11) — L0 gate PASS; S2/DEK trained; VLKEB extract + taxonomy on disk  
**M-item:** M7 (decided 2026-10-11) — see [`../../CVPR2027_GATE_LOG.md`](../../CVPR2027_GATE_LOG.md) § Proposed methodology changes  
**Diagnosis root cause:** [`G5_LLAVA_L0_DIAGNOSIS.md`](G5_LLAVA_L0_DIAGNOSIS.md) (`bs=1` made soft routing vacuous)  
**Parent audit:** [`G5_METRIC_AUDIT.md`](G5_METRIC_AUDIT.md)  
**Glossary:** [`../../CVPR2027_GATE_LOG.md`](../../CVPR2027_GATE_LOG.md) § Glossary  

This file is the **only** results write-up for the M7 ladder. Do not mix with the failed `bs=1` G5 numbers in [`G5_RESULTS_TABLES.md`](G5_RESULTS_TABLES.md) / [`G5_GATE_WRITEUP_LLAVA.md`](G5_GATE_WRITEUP_LLAVA.md) without an explicit `bs=1` vs M7 label.

---

## Protocol

| Item | Value |
|---|---|
| Backbone | LLaVA-1.5-7B (lab RTX 4090) |
| L0 | `dn=500` × `eps=20`, seed 42, **`bs=2`** (`bs=4` OOM; `-dbs 1`) |
| S2 / DEK | From M7 L0; `dn=500` × 1; AVBT; DEK + dual gate; **`bs=1`** (AVBL+`bs=2` OOM) |
| Eval | VLKEB short, `dsn=100`, `sen=10`, seed 42, `--log_routing` |
| Taxonomy | `tools/score_leakage_taxonomy.py` (binary + τ; weights = logged `fuse_coe`) |
| Success gate (after L0) | Train-split own text top-1 ≥ 25% — **met (70%)** |

---

## Run log

| Stage | `bs` | Status | Ckpt / artifact |
|---|---:|---|---|
| L0 smoke | 4 then 2 | **done** — 4 OOM; 2 OK | `m7_smoke_bs2` |
| L0 full | 2 | **done** (~122 min) | `…191244…/epoch-20-i-5000-ema_loss-6.4182` |
| L0 success gate | — | **PASS** — text top-1 **70%** | `llava_l0_diagnosis_m7.json` |
| S2 | 2 | **OOM** (AVBL) | — |
| S2 | **1** | **done** (~31 min) | `…S2_bs1…/epoch-1-i-500-ema_loss-4.7048` |
| DEK | **1** | **done** (~32 min) | `…DEK_bs1…/epoch-1-i-500-ema_loss-4.7985` |
| VLKEB extract + taxonomy | — | **done** | `leakage_taxonomy_llava_m7.{json,md}` |

---

## 1. Soft-routing during L0

M7 L0 used real `bs=2`, so soft relative / abstract losses are no longer vacuous (unlike `bs=1` diagnosis). Exact TensorBoard curves not pasted here; the train-split gate (§2) is the acceptance check.

---

## 2. L0 success gate (train-split routing)

Source: `llava_l0_diagnosis_m7.json` (n=50 train edits, reliability = exact edit prompt + image).

| Metric | `bs=1` (failed G5 L0) | M7 L0 (`bs=2`) |
|---|---:|---:|
| Own in vision gate | 66% | **72%** |
| Own text top-1 | 4% | **70%** |
| Mean max `fuse_coe` | 0.084 | **0.797** |
| Mean `ivr_eqe_std` | 0.013 | **0.380** |

**Reading:** M7 fixed the dead prompt-key path. Soft routing now trains; fuse weights are peaked, not flat.

---

## 3. Leak taxonomy (binary) — VLKEB short

Source: `leakage_taxonomy_llava_m7.md` (mapping check 0/100 mismatches).

| Method | Swap total leak ↓ | Hard-neg total leak ↓ | Rel hit ↑ | Only-own ↑ | Abstain (swap) |
|---|---:|---:|---:|---:|---:|
| L0 | 98.0 | 98.0 | **81.0** | 0.0 | 2.0 |
| S2 | 100.0 | 100.0 | **100.0** | 0.0 | 0.0 |
| DEK | 98.0 | 100.0 | **45.0** | **45.0** | 1.0 |

Image-swap four-way (binary):

| Method | stale ↓ | foreign ↓ | total ↓ | correct_route ↑ | abstain |
|---|---:|---:|---:|---:|---:|
| L0 | 34.0 | 64.0 | 98.0 | 0.0 | 2.0 |
| S2 | 98.0 | 2.0 | 100.0 | 0.0 | 0.0 |
| DEK | 13.0 | 85.0 | 98.0 | 1.0 | 1.0 |

---

## 4. Weight-aware (`fuse_coe`, τ=0.1)

| Method | Swap total leak ↓ | Hard-neg total leak ↓ | Rel hit ↑ |
|---|---:|---:|---:|
| L0 | 84.0 | 80.0 | 70.0 |
| S2 | 21.0 | 19.0 | 31.0 |
| DEK | 93.0 | 91.0 | 44.0 |

Under real `fuse_coe`, S2’s binary “100% leak” mostly has tiny weights (τ=0.1 cuts swap total leak to 21% but Rel hit falls to 31%). DEK does **not** beat L0 on leak at τ=0.1.

---

## 5. Verdict vs G5 question

G5: Does DEK reduce `total_leak` on a second backbone **without** collapsing reliability?

| Criterion | M7 LLaVA evidence |
|---|---|
| L0 editor trains (prompt keys) | **Yes** — text top-1 70%; Rel hit 81% (vs collapsed Rel EM / ~1% Rel hit on failed `bs=1` DEK ladder) |
| DEK leak ↓ vs L0 (binary) | **No** — swap 98→98, hn 98→100 |
| Reliability held under DEK | **No** — Rel hit 81%→45% (better than old 1%, still a large drop) |
| Only-own | Improves 0%→45% (precision), but not a leak win |
| vs failed `bs=1` G5 write-up | Old “leak↓ via abstain, Rel~1%” was an untrained editor. M7 replaces that story |

**Provisional G5 (LLaVA) after M7: still not PASS.** Soft-routing fix was necessary and succeeded; Dual-Expert Keys on this LLaVA short ladder still do **not** reproduce the BLIP-2 pattern (high Rel + lower `total_leak`). Treat LLaVA as a **limitation / open method gap** until a further try (or accept WEAK with this honest table).

**Paper framing if citing now:** disclose M7 (`L0 bs=2`, `S2/DEK bs=1` for VRAM); lead with L0 gate recovery; do **not** claim LLaVA DEK leak reduction from the binary headline.

---

## 6. Root cause (summary)

Full findings and the fix plan: [`G5_LLAVA_DEK_ROOT_CAUSE.md`](G5_LLAVA_DEK_ROOT_CAUSE.md).

- **F1** After S2/DEK the vision gate passes **99.9/100** experts on every probe (L0: 15.6; BLIP-2 DEK: 7.6), so `routing_rule=and` with margin 0 reduces to the prompt-key argmax alone. DEK Rel hit 45% equals text own-top-1 45%.
- **F2** The prompt key is image-conditioned (same prompt, swapped image: argmax changes in ~65% of probes, landing on a foreign expert) and degrades 70 -> 56 -> 45% own-top-1 along L0 -> S2 -> DEK.
- **F3** Cause: at `bs=1` every routing loss (soft relative/abstract, hard neighbor, AVBL binding softmax) has no in-batch negatives; S2/DEK could only move the own-vs-prototype margin and moved it the wrong way. M7 fixed L0 but re-introduced the vacuous objective in S2/DEK.
- **F4** Protocol confound: BLIP-2 July runs used `head` sampling (`swap_partner_in_pool_rate` 0.73, `correct_route` 34%); LLaVA uses `subsample` (0.01). Phase C re-extract required before any shared table.

---

## Contrast: failed `bs=1` ladder (do not mix)

| Method (old) | Swap TL ↓ | Hn TL ↓ | Rel hit ↑ |
|---|---:|---:|---:|
| L0 | 100 | 98 | 79 |
| DEK | 36 (mostly abstain) | 46 | **1** |

Old DEK “leak win” was abstain from a non-functional text gate — superseded by M7.

---

## Artifacts

| File | Role |
|---|---|
| `llava_l0_diagnosis_m7.json` | Post-L0 train-split gate |
| `leakage_taxonomy_llava_m7.{json,md}` | VLKEB taxonomy |
| `logs/m7_s2dek_bs1_*.log` | S2/DEK/eval outer log |
| `EXPERIMENT_TRACKER.md` | Run timeline |
| Eval dirs | `…/g5_llava_m7_{L0,S2,DEK}_short-2026.10.11-…` |
