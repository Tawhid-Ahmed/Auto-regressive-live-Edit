# G5 metric audit — code vs glossary (LLaVA + BLIP-2 reference)

**Date:** 2026-10-11  
**Why:** AAAI reviewers flagged mismatches between metric definitions, code and tables. This audit checks that every number in the G5 docs is computed the way the Glossary in [`../../CVPR2027_GATE_LOG.md`](../../CVPR2027_GATE_LOG.md) defines it, and whether the LLaVA numbers mean what the G5 write-up says they mean.  
**Inputs:** LLaVA VLKEB short runs `g5_llava_{L0,S2,DEK}_short` (lab mirror `routing_diagnostics.json`); BLIP-2 reference `g3_vlkeb_{L0,DEK}_nat_n100_s42` (home).

---

## 1. Summary

| # | Check | Result |
|---|---|---|
| A | Outcome definitions in `tools/score_leakage_taxonomy.py` vs Glossary (`stale_leak`, `foreign_leak`, `correct_route`, `abstain`, `total_leak`, `hit`, `only_own`, `cross`, `miss`) | **Match** |
| B | Sample → probe-row mapping (own expert lookup) | **Valid** — 0 mismatches on every G0 / G5 run (checked on lab against `seed_42_results.json`) |
| C | Edit-time vs probe-time query slice (`_pack_edit_forward` vs `probe_routing`) | **Consistent** — same slice, same real target |
| D | Probe naming (M4 `image_swap` clobber) | **Fixed** in source; LLaVA logs carry correct `probe` |
| E | Weight-aware (`tau_*`) source label | **Mismatch (fixed in code)** — scorer used real `fuse_coe`, but the output JSON said "softmax over vis_sim"; G5 docs called the rows a proxy |
| F | What LLaVA Rel hit / leak numbers actually mean | **Interpretation error** — see §3 |

The metric code is computing the glossary definitions correctly. The problems are one mislabel (E) and a wrong reading of the LLaVA results (F).

---

## 2. Weight-aware label mismatch (E)

`active_experts` prefers `fuse_coe` → `pool_fuse_coe` → vis-sim softmax proxy. LLaVA logs have `fuse_coe` on every non-empty probe:

| Run | `fuse_coe` | `none_retrieved` | proxy |
|---|---:|---:|---:|
| LLaVA L0 | 298 | 2 | 0 |
| LLaVA S2 | 300 | 0 | 0 |
| LLaVA DEK | 138 | 162 | 0 |

Before this audit the output field `weight_definition` was a hard-coded string ("softmax over vis_sim of retrieved experts"), so **every taxonomy JSON built from G2 / G3 / G5 logs mislabels its `tau_*` rows**. Numbers are unaffected; the label was wrong.

**Code fix (2026-10-11):** `score_leakage_taxonomy.py` now writes `weight_sources` (per-run branch counts) and a `weight_definition` that states the preference order. Re-scoring the LLaVA VLKEB logs reproduces the stored `leakage_taxonomy_llava.json` exactly.

**Doc fixes:** `G5_RESULTS_TABLES.md` §2.4, `G5_GATE_WRITEUP_LLAVA.md` Bad, `CVPR2027_MAIN_TRACK_IMPROVEMENTS.md` P1 #10 no longer call the LLaVA τ rows a vision-logit proxy.

### Consequence: LLaVA swap win reverses under real weights

| Method | Swap total leak, binary ↓ | Swap total leak, τ=0.1 ↓ | Hard-neg total leak, binary ↓ | Hard-neg total leak, τ=0.1 ↓ | Rel hit, τ=0.1 ↑ |
|---|---:|---:|---:|---:|---:|
| L0 | 100.0 | **13.0** | 98.0 | 38.0 | 2.0 |
| S2 | 100.0 | 34.0 | 100.0 | 7.0 | 5.0 |
| DEK | 36.0 | **36.0** | 46.0 | 46.0 | 1.0 |

L0 retrieves ~20 experts per probe with max `fuse_coe` ≈ 0.05, so most binary L0 "leaks" have almost no residual weight. The headline "DEK −64 pp swap" on LLaVA exists **only under binary counting**. This is the same concern AAAI Reviewer 1 #5 raised for BLIP-2 (binary may overstate baseline leak).

---

## 3. What the LLaVA numbers actually measure (F)

The G5 write-up reads LLaVA as "DEK cuts leak but collapses Rel". Gate-level logs show the **LLaVA editor's prompt-key channel never learned**, already in L0. DEK's AND gate just exposes it.

### 3.1 Gate decomposition, reliability probe (exact edit prompt + edit image)

| Run | Experts retrieved (mean) | Own in vision gate | Own in text gate | Own retrieved | Max `pool_fuse_coe` (median) |
|---|---:|---:|---:|---:|---:|
| BLIP-2 L0 | 3.1 | 100 / 100 | **100 / 100** | 100 / 100 | **0.99** |
| BLIP-2 DEK | 1.0 | 100 / 100 | **100 / 100** | 100 / 100 | **0.99** |
| LLaVA L0 | 24.4 | 79 / 100 | **2 / 100** | 79 / 100 | ~0.01 |
| LLaVA S2 | 47.2 | 98 / 100 | **1 / 100** | 98 / 100 | ~0.01 |
| LLaVA DEK | 0.6 | 98 / 100 | **1 / 100** | 1 / 100 | ~0.01 |

- **Vision gate works on LLaVA** (own expert median vision rank 1 for S2/DEK).
- **Text gate (prompt key) is non-functional on LLaVA.** Top-1 prompt-key expert is the own expert only 1–2% of the time on the *exact edit prompt*; DEK's text gate picks the same hub experts repeatedly (pool #96 on 31%, #20 on 23% of reliability probes). Own-expert text similarity is often negative.
- **Fuse coefficients are flat** (~1/pool): the residual mix weights each retrieved expert ≈ equally and tiny.
- Extractor output is near-constant: `ivr_eqe_std` LLaVA **0.009–0.012** vs BLIP-2 **0.43**.

### 3.2 Token-level metrics agree: no LLaVA method edits

| Method | Rel acc ↑ | Rel EM ↑ | Image-swap acc |
|---|---:|---:|---:|
| L0 | 0.51 | 0.01 | 0.47 |
| S2 | 0.50 | 0.01 | 0.47 |
| DEK | 0.49 | 0.01 | 0.47 |

Rel token accuracy is the same for L0 (routing Rel hit 79%) and DEK (routing Rel hit 1%), and ≈ image-swap accuracy. The edit residual has almost no effect on LLaVA outputs for **any** method. LLaVA L0 training also ended at `ema_loss` 2.21 vs BLIP-2 L0 0.12.

### 3.3 Corrected reading

| Claim in G5 write-up | Corrected |
|---|---|
| "DEK collapses Rel hit 79% → 1%" | L0's 79% Rel hit is the vision gate firing ~24 experts; L0 does not edit either (Rel EM 1%). Root cause is the **untrained LLaVA prompt-key / fuse path**, not DEK. |
| "Leak reduction transfers to LLaVA" | Only under binary counting; under real `fuse_coe` (τ=0.1) DEK swap total leak is worse than L0. |
| "Stable over 5 seeds / pool sizes / EVQA" | Stable, but it is the same broken-L0 artifact each time. |

**Status of G5 (LLaVA):** the extract does **not** test DEK on a working second-backbone editor. Neither PASS nor a method WEAK is supported; this is a **LLaVA L0 training failure** to fix before G5 can be decided on LLaVA. Gate verdict changes are **not** applied here — discuss first.

### 3.4 Candidate causes — see separate diagnosis doc

Lab checks (loss curve, train-split routing, hidden-state scale, query slice) and the Oct-15 retrain-vs-disclose recommendation live in:

**[`G5_LLAVA_L0_DIAGNOSIS.md`](G5_LLAVA_L0_DIAGNOSIS.md)** (complete 2026-10-11; root cause: `bs=1` vacuous soft routing — discuss before any M-item fix).

---

## 4. Glossary wording conflicts (flagged, not changed)

Per `.cursor/rules/cvpr2027-terminology.mdc`, these are flagged for a decision rather than silently edited:

| Where | Wording | Conflict |
|---|---|---|
| `G5_RESULTS_TABLES.md` §3; this audit §3.2 | "Image-swap acc", "Hard-neg acc" | Glossary: Swap-accuracy is **retired** and must not appear in new text |
| `G5_RESULTS_TABLES.md` §3; write-up | "Rel acc", "Rel EM" | Not defined in the Glossary — add (token accuracy / exact match on reliability) or rename |
| `G5_RESULTS_TABLES.md` header | "abstain (on hard_negative) ↑" | Glossary direction for `abstain` is "context", not higher-is-better |
| `G5_RESULTS_TABLES.md` §5, §7–9; write-up | "Swap TL", "Hn TL" | Third variant of "total leak"; tables should say "image-swap total leak" / "hard-negative total leak" |
| Taxonomy JSON `legacy_summary` | `*_false_activation_rate`, `*_false_cross_rate` | Retired AAAI terms; acceptable only as labelled legacy audit fields |

---

## 5. Reproduce

```text
python tools/score_leakage_taxonomy.py \
  --run <lab_mirror>/VLKEB_GROUNDING-g5_llava_L0_short-2026.10.11-12.10.05 \
  --run <lab_mirror>/VLKEB_GROUNDING-g5_llava_S2_short-2026.10.11-12.14.11 \
  --run <lab_mirror>/VLKEB_GROUNDING-g5_llava_DEK_short-2026.10.11-12.22.23 \
  --labels L0 S2 DEK -out <tmp>.json
```

Gate decomposition in §3.1 reads `vision_ok_indices`, `text_ok_indices`, `pool_fuse_coe`, `ivr_eqe_std` straight from each `routing_diagnostics.json`.
