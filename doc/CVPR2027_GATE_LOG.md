# CVPR 2027 Gate Log

Running record for the gated resubmission plan (`cvpr_2027_resubmission_plan`). One entry per gate, written the day the numbers exist. Each entry has the same four blocks so later gates can be read against earlier ones:

- **Result**: the numbers and the gate verdict against the plan's written thresholds.
- **Good**: what the step strengthened. Carried forward as claims we are allowed to make.
- **Bad**: what the step weakened or exposed. Carried forward as caveats, limitations, or open questions.
- **Carry-forward**: concrete consequences for the next step(s). Anything that would change the planned methodology goes into "Proposed methodology changes" and is **discussed before implementation**; it is not applied silently.

Rules (from the plan): no 5-seed run for an unpassed config; a 1/10 smoke test before any GPU job over 6 h; gates decided the day the numbers exist; Nov 5 experiment freeze.

Wording rule (lesson from AAAI): every metric, outcome, probe and gate has exactly one name, defined once in the Glossary below, used identically in code identifiers, JSON keys, log fields, tables and paper prose. New terms are added to the glossary in the same change that introduces them. Enforced by `.cursor/rules/cvpr2027-terminology.mdc`.

---

## Glossary (canonical terms)

Direction: "lower" = lower is better, "higher" = higher is better. "Code" is the exact identifier / JSON key.

### Probes (one query type each; names match `tools/log_grounding_diagnostics.py`)

| Term | Code | Definition |
|---|---|---|
| reliability probe | `reliability` | Edited prompt p_i with edited image x_i. |
| image-swap probe | `image_swap` | Edited prompt p_i with the swap partner's image x_j (different entity, same prompt skeleton). |
| hard-negative probe | `hard_negative` | Edited prompt p_i with a visually confusable partner image x_hn. |
| modality-consistency probe | `modality_consistency` | Edited prompt p_i with a rephrased image of the same entity. |
| prompt-conflict probe | `prompt_conflict` | Misleading prompt prefix on the edited image. |
| image-rephrase probe | `image_rephrase` | Standard VLKEB image generality probe. |
| image / text locality probe | `image_loc`, `text_loc` | Standard locality probes. |

### Routing outcomes (mutually exclusive, assigned in priority order; `tools/score_leakage_taxonomy.py`)

| Term | Code | Applies to | Definition | Direction |
|---|---|---|---|---|
| stale leak | `stale_leak` | image_swap, hard_negative | The edited expert e_i fired on a mismatched image (alone or with others). | lower |
| foreign leak | `foreign_leak` | image_swap, hard_negative | e_i did not fire, but some expert e_k with k not in {i, j} fired. | lower |
| correct route | `correct_route` | image_swap only | Only the swap partner's expert e_j fired. | higher |
| abstain | `abstain` | image_swap, hard_negative | No expert fired. Safe on hard_negative; safe but zero-coverage on image_swap. | context |
| total leak | `total_leak` | image_swap, hard_negative | `stale_leak + foreign_leak`. Headline leakage number. **Replaces AAAI "false activation rate" / "FPR"**, which also counted `correct_route` as an error. | lower |

Non-exclusive flags, reported alongside: `own_fired`, `partner_fired`, `foreign_fired`, `any_fired`, `hn_partner_fired`.

### Reliability routing (reliability probe)

| Term | Code | Definition | Direction |
|---|---|---|---|
| reliability hit | `hit` | e_i is among the active experts. | higher |
| only-own | `only_own` | Active set is non-empty and contains only e_i. Routing precision. | higher |
| cross | `cross` | Some expert other than e_i is active. **Replaces AAAI "false cross" / "false-cross"**. | lower |
| miss | `miss` | e_i not active. | lower |

### Counting variants

| Term | Code | Definition |
|---|---|---|
| binary counting | `binary` | An expert is active iff it is in `retrieved_indices`. |
| weight-aware counting | `tau_<value>` | An expert is active iff its soft weight exceeds tau. G0 used a vision-logit proxy; from G2 use `pool_fuse_coe` / `fuse_coe` from `get_moe_fuse_coe` (M3). |
| fuse coefficient | `fuse_coe`, `pool_fuse_coe` | LiveEdit residual mix weights: softmax(prompt-key sim) × sigmoid(prompt-key sim). `fuse_coe` aligns with `retrieved_indices`; `pool_fuse_coe` is the same formula over the full pool. |

### Answer-level metrics (pending G0 re-run)

| Term | Code | Definition | Direction |
|---|---|---|---|
| emits stale answer | `emits_y_i` | On image_swap, the model outputs the edited answer y_i. | lower |
| emits partner answer | `emits_y_j` | On image_swap, the model outputs the partner's answer y_j. | higher |

"Swap-accuracy" (AAAI) is retired and must not appear in new text.

### Methods and gates

| Term | Code / label | Definition |
|---|---|---|
| baseline | `L0` | Original LiveEdit routing, no patch key, no AVBT. |
| AVBT | `AVBT` | Adversarial visual-binding training (training-time component). |
| DEK | `DEK` | Dual-Expert Keys: AVBT + AND dual gate (vision gate AND prompt-key agreement). |
| DEK_and | `DEK_and` | DEK checkpoint evaluated with `--routing_rule and` (G2 winner; used as the G3 method). |
| routing rule | `--routing_rule` | One of `liveedit, vision_only, text_only, and, or, sum, carml, always_reject` (G2). |
| gate | `G0`..`G6`, `G2b` | Decision checkpoints of the plan. |
| methodology change | `M1`, `M2`, ... | Proposed change to the planned method, tracked below. |

### Pool construction (G3 / M5)

| Term | Code / label | Definition |
|---|---|---|
| pair-closed sampling | `pair_closed` | Every kept edit's `image_swap` and `hard_negative` partners are also in the pool; partner indices are remapped to the closed file. Built by `tools/build_pair_closed_pool.py`. |
| pool size | `sen` (= `dsn` for G3) | Number of edits in the lifelong pool for the curve. G3 targets `{100, 300, 1000}`; closed sets may be slightly larger (`actual_n`). |
| natural partner-absent | `natural` (side run) | Subsample of the original grounding JSON without closing partners; reports the natural partner-absent fraction for the supplementary. |

---

## Status board

| Gate | Window | Status | Verdict | Entry |
|---|---|---|---|---|
| G0 honest re-scoring | Oct 4 - Oct 7 | **complete (Oct 4)** | PASS; answer-level wired; caveats below | [G0](#g0-honest-re-scoring-of-existing-logs) |
| G1 residual leak source | Oct 7 - Oct 10 | **complete (Oct 4)** | unexplained / not text-gate; move on | [G1](#g1-where-do-residual-leaks-come-from) |
| G2 routing rule ablation | Oct 8 - Oct 17 | **complete (Oct 4)** | **PASS** — DEK on Pareto front; 5-seed authorized | [G2](#g2-which-routing-rule-actually-wins) |
| G2b margin / abstain sweep | Oct 11 - Oct 16 | **not triggered** (G0 was PASS, not WEAK) | - | [G2b](#g2b-can-stale-leakage-be-fixed-at-inference) |
| G3 pool-size curve | Oct 14 - Oct 24 | **complete (Oct 6)** | **PASS** — DEK advantage holds; **HIGH FLAG** pair-closed hn ~84% (M6) | [G3](#g3-does-the-lifelong-scaling-claim-survive) |
| G4 other editors | Oct 17 - Oct 28 | not started | - | [G4](#g4-does-the-protocol-transfer-to-other-editors) |
| G5 second backbone | Oct 15 (fixed) | **lab 4090 (Oct 10)** — CVPR strong LLaVA L0 (`dn=500`×`eps=20`) then S2/DEK; dual extract with BLIP-2 | - | [G5](#g5-second-backbone-decision) |
| G6 writing checkpoint | Oct 27 (fixed) | not started | - | [G6](#g6-writing-checkpoint) |

Current track: **G5 on lab RTX 4090 for CVPR main track** (strong LLaVA L0 → S2 → DEK → eval; then re-extract BLIP-2 under the same protocol; decision due Oct 15). Plan A (DEK) unchanged for BLIP-2. **Deferred → later**: M6, `emits_y_i`, EVQA pair-closed, 5-seed, G4, MSCKE, CARML tau.

---

## Proposed methodology changes (discuss before implementing)

Nothing here is implemented until it is marked *decided*. Decided items stay in the table with the decision and date; open items still need discussion.

| # | From | Evidence | Decision (Oct 4) | Status |
|---|---|---|---|---|
| M1 | G0 | Weight-aware tau=0.1 shrinks VLKEB swap gain to -4 pp; hard-negative gain unchanged (-39 / -87 pp). | Paper **headline leakage number is hard-negative `total_leak`**; image-swap is the second column with the full four-outcome breakdown. Do not drop image-swap. | **decided** |
| M2 | G0 | EVQA DEK abstains on 99.0% of image-swap; `correct_route` 0.3% vs baseline 87.0%. | Report image-swap as a four-outcome breakdown. **`abstain` is safe (not a leak) but not success**; pair it with `correct_route` and a coverage/recall@partner column so `always_reject` cannot look like a win. | **decided** |
| M3 | G0 | Weight-aware side check uses a vision-logit proxy, not `get_moe_fuse_coe`. | In G2, log the **actual per-expert fuse coefficient** in `probe_routing` and recompute weight-aware taxonomy before accepting the side-check demotion of image-swap. Do not put proxy weight-aware numbers in the paper. | **decided, applied Oct 4** (`fuse_coe` / `pool_fuse_coe` in `probe_routing`) |
| M4 | G0 | `image_swap` stored as `image_rephrase` when classification clobbers `probe`. | Fix key order in `tools/log_grounding_diagnostics.py`; keep scorer repair for old logs. | **decided, applied Oct 4** |
| M5 | G0 | Partner-absent rates flatten abstain-heavy methods (VLKEB 27–31% absent; EVQA `_sub_` hard-neg ~98% absent). | For G3 (and any new pool builds), use **pair-closed sampling** so every edit's image-swap and hard-negative partners are also in the pool. Report the natural partner-absent fraction once as a supplementary side experiment. | **implemented** (Oct 6; `tools/build_pair_closed_pool.py`; used for G3) |
| M6 | G3 | Pair-closed VLKEB DEK_and hard-negative `total_leak` stays **~84–86%** at n∈{100,300,1000} (vs L0 100%; vs natural-side DEK 44%). G3 still PASS on relative advantage / swap scaling, but absolute hn under honest partners is weak for a M1 headline. | **HIGH FLAG**: treat pair-closed hn as an open method gap. Candidate next tries (later): tighter vision scope / top-K, train-tuned CARML-style, margin/abstain on hn only, or hard-neg–aware training. Keep DEK as Plan A for swap+reliability until a candidate beats DEK on hn **without** killing rel. | **deferred — later** (Oct 6; G5 first) |

---

## Active flags (read before next GPU / paper claim)

| Flag | Level | Source | Status | What it blocks or forces |
|---|---|---|---|---|
| Pair-closed DEK hn ~84% | **HIGH** | G3 / M6 | **deferred — later** | Absolute hn under M5; method try after G5. |
| Routing ≠ answer (`emits_y_i`) | **HIGH** | G0 | **deferred — later** | Do not claim answer-level rescue until revisited. |
| EVQA swap = abstain, not route | medium | G0 / M2 | later | Coverage/`correct_route` beside EVQA swap. |
| Residual not text-gate | medium | G1 | later | Do not center paper on text-gate fix. |
| G2 number mismatch vs G0 head | low | G2 | later | One sample protocol in paper tables. |
| Untuned CARML-style row | low | G2 | later | Label or train-tune before naming CARML. |
| MSCKE public-code check | low | G2 carry | later | ≤1 day when spare. |
| G3 single seed / VLKEB only | medium | G3 | later | EVQA pair-closed + more orders optional. |
| G5 due Oct 15 | calendar | plan | **active now** | Second-backbone setup → decision. |

## G0: honest re-scoring of existing logs

**Dates**: Oct 4, 2026 (planned Oct 4 - Oct 7; finished day 1).
**Tool**: `tools/score_leakage_taxonomy.py`.
**Artifacts**: `doc/results/cvpr2027/g0/leakage_taxonomy_vlkeb_s42.{json,md}`, `doc/results/cvpr2027/g0/leakage_taxonomy_evqa_s42.{json,md}` (per-probe records included); `doc/results/cvpr2027/g0/emits_y_seed42.json` (answer-level freegen).
**Inputs**: `VLKEB_GROUNDING-aaai_L0_short` vs `aaai_DEK_short` (100 edits, seed 42); `evqa_gnd_ms_L0_s42_d300` vs `evqa_gnd_ms_DEK_s42_d300` (300 edits, seed 42). Sample-to-row mapping validated against `seed_42_results.json` with 0 mismatches on all four runs. Answer-level re-runs: `g0_emits_{L0,DEK}_short`, `g0_emits_evqa_{L0,DEK}_s42_d300`.

### Result

Binary counting (an expert is active if it is in `retrieved_indices`), all numbers in %:

| Run | swap stale | swap foreign | **swap total** | swap correct_route | swap abstain | hn stale | hn foreign | **hn total** | rel hit | rel only-own |
|---|---|---|---|---|---|---|---|---|---|---|
| VLKEB L0 | 6.0 | 82.0 | **88.0** | 8.0 | 4.0 | 9.0 | 82.0 | **91.0** | 100.0 | 16.0 |
| VLKEB DEK | 8.0 | 12.0 | **20.0** | 34.0 | 46.0 | 3.0 | 49.0 | **52.0** | 100.0 | 100.0 |
| EVQA L0 | 0.0 | 11.3 | **11.3** | 87.0 | 1.7 | 4.7 | 92.0 | **96.7** | 100.0 | 17.3 |
| EVQA DEK | 0.3 | 0.3 | **0.7** | 0.3 | 99.0 | 5.7 | 3.7 | **9.3** | 99.0 | 99.0 |

Weight-aware (softmax over vision logits of the retrieved experts, active iff weight > tau):

| Run | tau | swap total | hn total | rel hit |
|---|---|---|---|---|
| VLKEB L0 | 0.05 / 0.1 / 0.2 | 24 / 24 / 23 | 91 / 91 / 91 | 100 / 99 / 99 |
| VLKEB DEK | any | 20 | 52 | 100 |
| EVQA L0 | 0.05 / 0.1 / 0.2 | 2.7 / 2.7 / 2.7 | 96.7 / 96.3 / 96.3 | 100 / 95 / 89.7 |
| EVQA DEK | any | 0.7 | 9.3 | 99 |

DEK is unaffected by tau because it retrieves at most one expert.

Answer-level on image_swap (free-running decode; matcher: exact / first-sentence / whole-word len>=4, skip empty and `not <target>`). Swap-accuracy is retired:

| Run | emits_y_i (lower) | emits_y_j (higher) | neither | legacy swap acc (retired) | rel acc |
|---|---|---|---|---|---|
| VLKEB L0 | **8.0** | 0.0 | 92.0 | 43.4 | 95.2 |
| VLKEB DEK | **21.0** | 0.0 | 79.0 | 54.2 | 95.4 |
| EVQA L0 | **8.3** | 0.7 | 91.7 | 15.6 | 94.2 |
| EVQA DEK | **9.3** | 1.3 | 90.7 | 15.9 | 95.0 |

**Verdict against thresholds**

- PASS condition: DEK `total_leak` >= 20 pp below baseline on VLKEB or EVQA, and DEK EVQA `stale_leak` < 15%. VLKEB swap: -68 pp. EVQA stale: 0.3%. **PASS.**
- WEAK condition (VLKEB stale >= 50%): DEK VLKEB stale is 8%. Not WEAK. **G2b not triggered.**
- Side check (baseline weight-aware leak at tau = 0.1 below 60%): VLKEB 24%, EVQA 2.7%. **Triggered.** Per the plan, weight-aware numbers become primary in every table, subject to M3 below.

**Decision (Oct 4)**: G0 = PASS. Plan A continues. Proceed to G1 and G2 coding. G2b is skipped.

### Good

- The honest metric does not kill DEK. On VLKEB, swap total leak 88 -> 20 and hard-negative 91 -> 52 with reliability hit unchanged at 100%.
- Hard-negative is the strongest and most robust DEK result: -39 pp on VLKEB, -87 pp on EVQA, and insensitive to the weight-aware counting. This is also the probe the AI reviewer asked to see in the main table.
- DEK's routing precision on reliability is near perfect: `only_own` 100% (VLKEB) / 99% (EVQA) vs 16% / 17% for the baseline. This is a clean, reviewer-proof statement about the gate, independent of the leak definition.
- The reviewers' objection (4umz Q2: e_j firing on its own image counted as an error) is confirmed and quantified: the AAAI 96% / 98% "false activation" contained 8% / 87% `correct_route`. Correcting it ourselves is a credibility gain.
- Stale leakage (4umz Q1) turns out to be small for both methods (<= 9% everywhere). The baseline's problem is foreign leakage, not stale.
- The scorer reconstructs sample order from the seed and validates it against the eval results, so old logs are reusable without re-running GPU jobs. The "re-run if logs lack fields" early-warning branch was not needed.

### Bad

- **EVQA swap was never the big problem.** The baseline's true swap leak is 11.3% because it already routes to the partner 87% of the time. DEK's 0.7% is reached by abstaining on 99% of swaps (`correct_route` 0.3%). On EVQA swap, DEK does not route better, it refuses. See M2.
- **Weight-aware counting demotes the VLKEB swap result.** Baseline swap leak at tau = 0.1 is 24%, so the DEK advantage is -4 pp rather than -68 pp. The plan's side-check rule now makes weight-aware numbers primary, which would remove swap as a headline. The weight used is a proxy (vision logits, not the real fuse coefficient), so this must be re-measured with the real weights in G2 before it is accepted. See M1, M3.
- **The residual DEK leak on VLKEB is almost all foreign (12 of 20 pp) and the residual hard-negative leak is 49% foreign.** Hard-negative is the place where DEK still fails half the time on VLKEB. G1 must explain this residual, not the swap residual.
- A non-trivial fraction of partners are outside the pool (VLKEB 27% swap / 31% hard-neg; EVQA `_sub_` 98% of hard-neg partners absent). For those probes `correct_route` is unattainable and `abstain` is the only safe outcome, which flatters abstain-heavy methods. See M5.
- Logging bug: `image_swap` probes were stored as `image_rephrase` in all existing logs. Scorer repairs old logs; source fixed Oct 4 (M4).
- **Answer-level does not favor DEK.** On VLKEB free-gen image_swap, DEK `emits_y_i` is **21% vs baseline 8%** (worse). `emits_y_j` is ~0 for both methods — freegen rarely produces the short partner entity string (chatty continuations / neither). EVQA is similar (DEK 9.3 vs L0 8.3 on `emits_y_i`; `emits_y_j` 1.3 vs 0.7). So the routing win does not currently translate into a cleaner answer on the swapped image; most answers are `neither`. Do not claim answer-level rescue from DEK until a gate recovers partner routing or answer format is constrained.

### Carry-forward

- **G1**: focus the residual analysis on (a) DEK's foreign leaks on VLKEB image-swap (12%) and hard-negative (49%), (b) whether these fall on duplicated prompts (40% of VLKEB image-swap probes share a prompt with another pool entry vs 2.7% on EVQA; per-probe records already carry `n_pool_sharing_prompt`), (c) the partner-in-pool split, (d) whether DEK's higher `emits_y_i` co-occurs with residual `stale_leak` / own-expert fire on image_swap.
- **G2**: implement M3 (log real per-expert fuse coefficient) before the first routing-rule run. Add `correct_route`, `abstain`, and partner-routing recall so `always_reject` is visibly 100% abstain / 0% `correct_route` (M2). Report `emits_y_i` / `emits_y_j` on the same free-gen protocol for every routing-rule row. M4 is already applied.
- **G3**: implement M5 (pair-closed partner sampling) before building 100 / 300 / 1000 pools; keep one natural-fraction side run for the supplementary.
- **Paper** (M1, M2): lead with hard-negative `total_leak` and reliability `only_own`; present image-swap as a four-outcome breakdown; retire Swap-accuracy; report `emits_y_i` / `emits_y_j` honestly (routing ≠ answer yet).
- **Plan B** is not triggered; G1, G3, G4 run regardless as the plan requires.

### Open G0 items

- [x] Add `target_partner` to the probe JSON and `emits_y_i` / `emits_y_j` free-gen scoring; seed-42 re-run VLKEB + EVQA L0/DEK (Oct 4). Artifacts under `doc/results/cvpr2027/g0/emits_y_seed42.json`.
- [x] Decide M1–M5 (Oct 4).

**G0 complete.**

---

## G1: where do residual leaks come from

**Dates**: Oct 4, 2026 (planned Oct 7 - Oct 10; finished early from G0 artifacts).
**Tool**: `tools/analyze_residual_leak_source.py`.
**Artifacts**: `doc/results/cvpr2027/g1/residual_leak_source_s42.json`.
**Inputs**: G0 taxonomy records (`leakage_taxonomy_{vlkeb,evqa}_s42.json`) + original DEK `routing_diagnostics.json` for foreign-expert prompt match.

### Result

Prompt-duplication stratification (binary counting; `n_pool_sharing_prompt > 1` = duplicated query prompt in the pool):

| Run | Probe | total_leak | dup_prompt rate | share of residual leaks on dup | leak \| dup | leak \| unique |
|---|---|---|---|---|---|---|
| VLKEB DEK | image_swap | 20% | 40% | 40% | 20% | 20% |
| VLKEB DEK | hard_negative | 52% | 40% | 50% | 65% | 43% |
| EVQA DEK | image_swap | 0.7% | 2.7% | (n_leak=2) | — | — |
| EVQA DEK | hard_negative | 9.3% | 2.7% | 29% | 100% | 6.8% |

Combined VLKEB DEK residual (swap+hn leaks): **47.2%** fall on duplicated query prompts.

Foreign expert vs query prompt (does the *fired* expert share `p_i`?):

| Run | Probe | n foreign | same-prompt sibling fired |
|---|---|---|---|
| VLKEB DEK | image_swap | 12 | 3 (25%) |
| VLKEB DEK | hard_negative | 49 | 22 (45%) |
| VLKEB DEK | overall | 61 | 25 (41%) |

Hypothesis check (dataset-level): VLKEB dup 40% + high DEK residual vs EVQA dup 2.7% + low residual — **correlation supported**. Mechanism (text gate admits prompt siblings) — **not supported** as the residual story.

**Deferred (not in existing logs):** DINOv2 hard-neg terciles; per-channel vision/text pass bits (infer only: DEK retrieve ⇒ both passed). Optional short GPU later; not required for the decision below.

**Emits cross-check:** not joined — `g0_emits_*` used seeded subsample; `aaai_*` taxonomy used head[:n]+shuffle. Standalone: VLKEB `emits_y_i` DEK 21% vs L0 8%. Re-join in G2 with `--log_routing` on the same eval.

**Verdict against plan thresholds**

- ≥70% of residual on duplicated prompts ⇒ text gate weak: **47.2% — fail this branch.**
- <50% ⇒ do not build residual story on duplication: **triggered.**
- **Decision (Oct 4): unexplained / not text-gate-dominated. Report briefly; move on to G2.** Do not make G2b text-margin the mandatory center (G2b already not triggered by G0). Optional text-margin row in G2 is fine as an ablation, not the method.

### Good

- Dataset-level story is usable in one sentence: VLKEB’s templated prompts co-occur with higher residual; EVQA’s sparse prompts co-occur with near-clean DEK swap — without claiming causal text-gate collapse.
- Residual composition is clear: VLKEB DEK hard-negative is **94% foreign**; image-swap residual is 60% foreign / 40% stale. G2 should optimize for foreign on hard-negative first (aligns with M1 headline).
- Image-swap leak rate under DEK is **identical** on dup vs unique prompts (20%/20%) — so prompt hygiene alone will not fix swap residual.

### Bad

- The plan’s preferred residual narrative (text gate + duplicated prompts) **does not clear 70%**, and fired foreign experts share the query prompt only 41% of the time overall (25% on image_swap). Spending a section on “fix the text gate” would overclaim.
- Hard-negative still leaks on **43% of unique-prompt** probes under DEK — vision / scope / collision territory, not templates.
- Partner-absent image_swap under DEK has **higher** leak (33% vs 15% when partner in pool), reinforcing M5 (pair-closed pools) before G3.
- DINOv2 terciles and gate-channel bits still missing; if a reviewer asks “visual collision?”, we need the deferred GPU pass or G2 instrumentation.

### Carry-forward

- **G2**: proceed with routing-rule ablation; include `always_reject` and partner recall (M2). Log real fuse coeffs (M3) and optional text-margin / top-K text-set size as **ablation rows**, not as the fix for a text-gate residual story. Prioritize hard-negative foreign leak in the Pareto plot.
- **Paper**: at most one short residual paragraph (dataset dup correlation + “fired expert usually not a prompt sibling”); do not center contributions on residual diagnosis.
- **Skip** investing in G1 human 50-pair hand-check / DINOv2 tercile rebuild unless G2 leaves time — plan said move on if unexplained.
- **Emits↔routing join** when the next eval runs with `--log_routing`.

**G1 complete.**

---

## G2: which routing rule actually wins

**Dates**: coding + full seed-42 table Oct 4, 2026.
**Status**: **complete**. Smoke PASS; full table 20/20 jobs exit 0; taxonomy scored.
**Artifacts**: `doc/results/cvpr2027/g2/routing_table_{vlkeb,evqa}_s42.{json,md}`, `routing_table_manifest.json`, `tools/run_g2_routing_table.py`.
**Note**: these runs use the current seeded-subsample loader (`sample_seed=42`), so absolute % differ slightly from G0 `aaai_*` head[:n] logs; comparisons **within** this table are valid.

### Implementation (Oct 4)

- `--routing_rule {liveedit, vision_only, text_only, and, or, sum, carml, always_reject}` → `LiveEdit.set_routing_rule`.
- **M3**: `fuse_coe` / `pool_fuse_coe` logged in `probe_routing`.
- Smoke PASS on L0 and DEK ckpts.

### Result (binary taxonomy, %)

**VLKEB (dsn=100)**

| run | swap stale | swap foreign | swap total | correct_route | abstain | hn total | rel hit | only_own |
|---|---|---|---|---|---|---|---|---|
| L0_liveedit | 9 | 79 | **88** | 0 | 12 | **93** | 100 | 21 |
| L0_vision_only | 9 | 79 | 88 | 0 | 12 | 93 | 100 | 21 |
| L0_text_only | 37 | 62 | 99 | 1 | 0 | 100 | 100 | 100 |
| L0_and | 0 | 0 | 0 | 1 | 99 | 5 | **50** | 50 |
| L0_or | 39 | 61 | 100 | 0 | 0 | 100 | 100 | 21 |
| L0_sum | 21 | 47 | 68 | 1 | 31 | 71 | 100 | 34 |
| L0_carml | 35 | 64 | 99 | 1 | 0 | 100 | 100 | 100 |
| L0_always_reject | 0 | 0 | 0 | 0 | 100 | 0 | **0** | 0 |
| **DEK_and** | 15 | 33 | **48** | 1 | 51 | **44** | **100** | **100** |
| DEK_liveedit | 28 | 58 | 86 | 0 | 14 | 98 | 100 | 8 |

**EVQA (dsn=300)**

| run | swap total | correct_route | abstain | hn total | rel hit | only_own |
|---|---|---|---|---|---|---|
| L0_liveedit | 14.3 | 85.0 | 0.7 | 97.0 | 99.3 | 31.0 |
| L0_text_only | 99.0 | 1.0 | 0.0 | 100 | 99.3 | 99.3 |
| L0_and | 0.0 | 0.0 | 100 | 2.0 | **61.3** | 61.3 |
| L0_sum | 17.3 | 8.3 | 74.3 | 50.0 | 100 | 65.7 |
| L0_carml | 73.7 | 26.3 | 0.0 | 100 | 99.3 | 99.3 |
| L0_always_reject | 0.0 | 0.0 | 100 | 0.0 | **0** | 0 |
| **DEK_and** | **1.7** | 1.7 | 96.7 | **9.0** | **99.0** | **99.0** |
| DEK_liveedit | 14.7 | 85.0 | 0.3 | 98.0 | 100 | 29.0 |

**Verdict against plan thresholds**

- PASS: DEK on Pareto front (no rule has both lower `total_leak` and reliability ≥ DEK). **PASS.** `L0_and` / `always_reject` have lower leak but collapse reliability (50% / 0% VLKEB; 61% / 0% EVQA) — they do not dominate.
- PARTIAL (`text_only` within 5 pp of `and` on leak and reliability): **not met** (text_only ≈99% swap leak).
- FAIL (`carml` or `sum` dominates DEK): **not met**. `sum` helps vs liveedit but loses to DEK_and; approximate `carml` is leaky here.

**Decision (Oct 4): G2 = PASS. Plan A continues. 5-seed runs of this table are authorized.** Keep DEK (`AVBT + and`) as the method. Do not cut patch-key material (PARTIAL not triggered). Do not demote DEK to an ablation row.

### Good

- DEK_and is the only high-reliability low-leak point on both datasets for hard-negative (VLKEB 44%, EVQA 9%) with rel hit ≥99%.
- `always_reject` is a clean control: 100% abstain, 0% `correct_route`, 0% rel hit — separates DEK’s EVQA abstain-heavy swap from a dead gate (M2).
- Vision gate is doing real work: `vision_only` ≡ `liveedit`; `text_only` / `or` / approximate `carml` are much leakier on swap. AVBT matters: `L0_and` kills reliability; `DEK_and` restores it.
- `sum` is a useful ablation (better than liveedit, worse than DEK) — good paper row.

### Bad

- `L0_and` without AVBT is an abstain trap (VLKEB rel 50%, EVQA 61%) — must be shown so reviewers do not think AND alone is enough.
- Approximate CARML (fixed `carml_scope_tau=0.05`, no train-split tuning) looks weak; before naming it “CARML” in the paper, either tune tau on train only or label the row “CARML-style fusion (untuned)”.
- DEK_and VLKEB swap total 48% is higher than G0’s 20% on the old head sample — subsample shift + freegen-era eval; report this table’s numbers consistently, do not mix with G0 head logs without a note.
- MSCKE public-code check still not done (≤1 day budget).

### Carry-forward

- **Authorized**: 5-seed runs for passed configs (DEK_and primary; keep L0_liveedit, L0_sum, always_reject, L0_and as controls).
- **G3**: pool-size curve with best gate = DEK_and; implement M5 pair-closed partners first.
- **G5**: second-backbone decision now unlocked (G0+G2 PASS); due Oct 15.
- **Paper**: lead with leak-vs-reliability plot from this table; method = AVBT+AND; side-by-side routing spec including untuned CARML-style and `sum`.
- Optional: retune `carml_scope_tau` on train split only if time before freeze.

**G2 complete (PASS).**

---

## G2b: can stale leakage be fixed at inference

**Status**: not triggered (G0 = PASS, VLKEB stale 8% < 50%). Revisit only if G2's real-weight re-measurement moves the stale number.

---

## G3: does the lifelong-scaling claim survive

**Dates**: coding + VLKEB seed-42 curve Oct 6, 2026 (plan window Oct 14 - Oct 24).
**Status**: **complete**. M5 pair-closed pools; L0_liveedit vs DEK_and at target pool sizes 100 / 300 / 1000; natural n=100 side run.
**Artifacts**: `tools/build_pair_closed_pool.py`, `tools/run_g3_pool_size_curve.py`; pools in `data/VLKEB/.../grounding/pair_closed/`; `doc/results/cvpr2027/g3/pool_size_curve_vlkeb_s42.{json,md}`.
**Note**: DEK n=1000 used `--routing_only` (routing taxonomy only) after an autograd-graph OOM on 16GB; fix is `torch.no_grad()` in `collect_grounding_routing_diagnostics`. Pair-closed partner-in-pool rates are 100% on all closed runs.

### Result

Pair-closed VLKEB seed 42 (binary counting). Headline = hard-negative `total_leak`; swap shown as four-outcome.

| target_n | method | hn total_leak ↓ | swap total_leak ↓ | correct_route ↑ | abstain | rel hit ↑ |
|---|---|---|---|---|---|---|
| 100 | L0 | 100.0 | 90.0 | 10.0 | 0.0 | 100.0 |
| 100 | DEK_and | **84.0** | **19.0** | **68.0** | 13.0 | **100.0** |
| 300 | L0 | 100.0 | 97.0 | 3.0 | 0.0 | 100.0 |
| 300 | DEK_and | **85.8** | **15.2** | **69.5** | 15.2 | **99.7** |
| 1000 | L0 | 100.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| 1000 | DEK_and | **83.6** | **23.3** | **61.8** | 14.9 | **99.2** |

Natural (non-closed) n=100 side run: partners in pool ~1% swap / 6% hard-neg; DEK hn `total_leak` 44%, swap 48% with 51% abstain (matches the G2-scale flattery M5 was meant to remove).

- PASS: DEK `total_leak` advantage vs L0 **holds** as pool size grows (hn advantage ~16 / 14 / 16 pp; swap advantage ~71 / 82 / 77 pp). Reliability does not collapse (≥99.2%). **PASS.**
- FAIL (advantage shrinks toward L0): **not met.**

### Good

- Lifelong claim survives on VLKEB seed 42 under honest pair-closed partners: L0 swap `total_leak` → 100% by n=1000; DEK stays ~15–23% with `correct_route` ≈ 62–70%.
- M5 works: closed pools make `correct_route` attainable; natural side run reproduces abstain-heavy swap (51%).
- Reliability stays high for DEK_and across the curve.

### Bad

- **HIGH FLAG (M6):** pair-closed hard-negative `total_leak` for DEK is **~84–86%** at every pool size (vs natural-side 44%). With partners present, non-own fires count as `foreign_leak`. This is honest, and it means DEK is **not** a strong absolute hn method under M5 — only relatively better than saturated L0. May need another method try before freezing the hn headline.
- L0 hn is saturated at 100% for all closed sizes, so the hn gap cannot “grow”; the scaling story that is solid today is primarily **swap** + reliability hold.
- Single seed / single edit order / VLKEB only so far; EVQA + more orders still open.
- DEK n=1000 is routing-only (no sequential token metrics in that run).

### Carry-forward

- **M6 (high):** **deferred — later** (G5 first). Revisit pair-closed hn method try after Oct 15 backbone decision.
- **Paper**: plot pair-closed hn + swap `total_leak` vs pool size; put natural partner-absent as a supplementary footnote/table; flag absolute hn as open limitation until M6 closes.
- **Next optional (later)**: more edit orders; EVQA pair-closed curve; authorized 5-seed on DEK_and.
- **G5**: active now (setup Oct 6 → decision Oct 15).
- **Eng**: keep `torch.no_grad()` around eval-time `edit_one_piece` / `probe_routing` for large pools.

---

## G4: does the protocol transfer to other editors

**Dates**: planned Oct 17 - Oct 28.
**Status**: not started.

### Result
_(pending)_

### Good
_(pending)_

### Bad
_(pending)_

### Carry-forward
_(pending)_

---

## G5: second backbone decision

**Date**: decision due **Oct 15, 2026** (fixed). Setup started Oct 6; **CVPR budget locked Oct 10**.
**Status**: **lab training in progress**. Preconditions: G0 PASS (met), G2 PASS (met).
**Artifacts**: `doc/results/cvpr2027/g5/` (README + `EXPERIMENT_TRACKER.md` + logs). Paper numbers use **lab** runs under `records/lab_g5/`, not home/USB AAAI LLaVA ckpts.
**Ladder** (CVPR): LLaVA-1.5-7B on lab RTX 4090 → strong L0 → S2 → DEK → dual extract with BLIP-2. Qwen2-VL-2B fallback only if LLaVA blocked.

### CVPR training + extraction plan (Oct 10)

| Phase | Work | Budget / note |
|---|---|---|
| A | Fresh lab LLaVA L0 → S2 → DEK | L0: `dn=500`, **`eps=20`**; S2/DEK: `dn=500`, **`eps=1`** |
| B | LLaVA grounding eval + leak taxonomy | Lab ckpts only |
| C | **Re-extract BLIP-2** L0 / S2 / DEK | Same eval scripts + probes as B (existing AAAI ckpts; no silent metric drift) |
| D | G5 gate write-up | Disclose: BLIP-2 full-set L0×20 vs LLaVA short-n L0×20 |

AAAI-era LLaVA `dn=500×3` was a deadline shortcut. Oct 10 `eps=1` lab L0 is **smoke only**, not paper L0.

### Setup inventory

| Asset | Status |
|---|---|
| Lab weights `/.../models/llava-1.5-7b-hf` | **present** (validated safetensors Oct 10) |
| Config `configs/liveedit/llava-v1.5-7b.yaml` | present |
| Home L0 `VLKEB_llava_L0_dn500` epoch-3 | archive only — **not** for CVPR G5 numbers |
| Lab LLaVA L0 / S2 / DEK (`records/lab_g5/`) | **in progress** — strong L0 next |
| Qwen2-VL | not wired yet — fallback only |

### Smoke (Oct 6 / Oct 10)

- Oct 6 home: `tools/smoke_g5_llava_load.py` → **PASS** (thin 16 GB headroom).
- Oct 10 lab: model load OK after weight repair; detached `eps=1` L0 used for env/data path check only.

### Result
_(pending — strong L0 + S2/DEK + dual extract)_

### Good
- Lab 4090 available; git sync on `lab/g5-code-only`; tracker + results pull scripts ready.
- CVPR budget explicit: L0×20 at `dn=500`, then same-protocol BLIP-2 re-extract.

### Bad
- Home/USB LLaVA shards were corrupt once — lab now uses validated download.
- Strong L0 (~20×500 steps) still to finish before S2/DEK.

### Carry-forward
- **This week:** finish Phase A–D (see `doc/results/cvpr2027/g5/README.md`); decide by **Oct 15**.
- Qwen2-VL-2B only if LLaVA blocked on the lab box.
- M6 and other flags stay **deferred — later**.

---

## G6: writing checkpoint

**Date**: Oct 27 (fixed).
**Status**: not started.

### Result
_(pending)_
