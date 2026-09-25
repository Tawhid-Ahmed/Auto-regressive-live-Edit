# Do we beat LiveEdit & AnyEdit? — Novelty, performance, missing metrics

**Date:** 2026-07-13  
**For:** Supervisor discussion  
**Papers compared:**
1. **LiveEdit** — Chen et al., CVPR 2025: *Lifelong Knowledge Editing for Vision Language Models with Low-Rank Mixture-of-Experts*
2. **AnyEdit** — Jiang et al., ICML 2025: *Edit Any Knowledge Encoded in Language Models*  
**Ours:** **AVBT + Dual-Expert Keys (DEK)** on top of LiveEdit (BLIP2-OPT-2.7B)

---

## Short answer (say this first)

| Question | Honest answer |
|----------|----------------|
| Do we **beat LiveEdit on novelty**? | **Partially yes** — we introduce a *new failure mode + fix* (false-cross / adversarial image-swap) that LiveEdit does **not** study. We do **not** replace LiveEdit; we **extend** it. |
| Do we **beat LiveEdit on performance**? | **On our probes: yes** (FC 96%→46%, swap +7 pp, Rel held). **On LiveEdit’s original Rel/Gen/Loc leaderboard: not claimed** — we have not re-published a full matched “beat Table X in CVPR paper” sweep. |
| Do we **beat AnyEdit**? | **Wrong comparison.** AnyEdit is a **text-LLM**, long-form locate-then-edit method. Different modality, different problem. We should **cite** it (AR / long targets), not claim to outperform it. |
| Is this AAAI-ready “SOTA over both”? | **No.** Strong **thesis** story; AAAI needs clearer “new problem + mechanism” framing and a few extra standard metrics (below). |

---

## 1. What each paper actually contributes

### LiveEdit (CVPR 2025) — our base system
| Aspect | Their claim |
|--------|-------------|
| **Problem** | Lifelong **VLLM** editing (continuous MoE experts) |
| **Method** | Expert generator + **hard visual filter** + **soft text routing** |
| **Metrics** | Reliability, text/modal **generality**, text/modal **locality** |
| **Backbones** | LLaVA-1.5 7B, MiniGPT-4 7B, BLIP2-OPT 2.7B |
| **Benchmarks** | E-VQA, E-IC, **VLKEB** |

**They already do vision+text routing** — but as “filter by vision, fuse by text,” optimized for Rel/Gen/Loc.  
They do **not** measure: image-swap leakage, hard-negatives, or **false-cross** (wrong expert retrieved).

### AnyEdit (ICML 2025) — related but different field
| Aspect | Their claim |
|--------|-------------|
| **Problem** | Long-form / diverse-format knowledge in **pure LLMs** |
| **Method** | Autoregressive **chunk-wise** locate-then-edit (multi-token) |
| **Metrics** | Editing efficacy on CounterFact, ZsRE, UnKE, **EditEverything** |
| **Modality** | **Text only** (no images, no lifelong VLM MoE pool) |

**Takeaway:** AnyEdit motivates *long targets / chunking*; it is **not** a competitor on VLKEB grounding.

### Ours — AVBT + Dual-Expert Keys
| Aspect | Our claim |
|--------|-----------|
| **Problem** | **Visual grounding leakage** under lifelong MoE: high Rel, leaky swap, **false-cross ~90%+** |
| **Method** | **AVBT** (adversarial bind/suppress) + **DEK** (retrieve only if **vision ∧ text**) |
| **Evidence** | Ablation L0→S2→DEK; 5 seeds; FT/MEND/SERAC; fair long FC cut |

---

## 2. Novelty scorecard (relative)

| Novelty dimension | vs LiveEdit | vs AnyEdit | Notes |
|-------------------|-------------|------------|-------|
| Lifelong VLM MoE editor | ❌ They invented it | N/A | We build **on** LiveEdit |
| Hard/soft routing idea | ❌ Prior art (theirs) | N/A | DEK is a **stricter dual-gate**, not a from-scratch editor |
| Adversarial image-swap / hard-neg probes | ✅ **Ours** | N/A | LiveEdit paper does not define these |
| **False-cross** as primary metric | ✅ **Ours** | N/A | New diagnostic + optimization target |
| Adversarial visual binding train (AVBT) | ✅ **Ours** | N/A | They train Rel/Gen/Loc, not swap negatives |
| Dual-gate **vision ∧ text** agreement | ✅ **Incremental but real** | N/A | Same family as their routing; different rule + objective |
| Autoregressive long-form text editing | ❌ | ❌ They own this | Do **not** lead with AnyEdit-style AR |
| New VLM backbone / new benchmark resource | ❌ Weak so far | ❌ | Optional upgrades |

**Novelty verdict for supervisor:**  
Not “we invent lifelong VLM editing” (LiveEdit did).  
Yes “we identify and mitigate a **grounding leakage / false-cross** failure in lifelong VLM editors that standard Rel/Gen/Loc miss.”  
That is a **valid thesis contribution**; for AAAI main it is **borderline / stretch** unless framed tightly and backed by more standard metrics (next section).

---

## 3. Performance scorecard (relative)

### 3.1 vs LiveEdit baseline (same code, our probes) — **fair**

| Metric | LiveEdit baseline (L0) | AVBT + Dual-Expert Keys (DEK) | Win? |
|--------|------------------------|-------------------------------|------|
| Reliability (short) | 93.5% | 93.7% | ≈ tie (held) |
| Image-swap (short) | 40.7% | ~47.7–48.6% | **+7 pp** (modest) |
| False-cross (short) | 96% | **46%** | **large win** |
| Reliability (fair long) | 74.0% | 72.8% | ≈ held |
| Image-swap (fair long) | 42.4% | 42.7% | ≈ tie |
| False-cross (fair long) | 88% | **38%** | **large win** |

**Verdict:** We **beat LiveEdit-as-baseline on the leakage problem**. We do **not** claim a huge swap SOTA.

### 3.2 vs FT / MEND / SERAC (our short table)

- We **win on operating point**: high Rel (~94%) + FC cut.  
- FT/MEND look higher on swap only because Rel collapses (~71–73%) — different failure mode.

### 3.3 vs LiveEdit **published** CVPR numbers — **not yet a fair “we beat the paper” claim**

Their tables use Rel / T-Gen / M-Gen / T-Loc / M-Loc at 1/10/100/1000 edits on multiple VLMs.  
We have **not** reproduced that full matrix for DEK and shown superiority on **their** metrics.

**Do not tell supervisor:** “We beat Chen et al. CVPR on VLKEB.”  
**Do say:** “On the same LiveEdit stack, we cut false-cross roughly in half on adversarial visual probes while preserving reliability.”

### 3.4 vs AnyEdit — **not comparable head-to-head**

| | AnyEdit | Ours |
|--|---------|------|
| Domain | Text LLM editing | Lifelong **VLM** MoE |
| Goal | Long / diverse text efficacy | Visual grounding / anti-leakage |
| Metric | Edit success on long text | Rel, swap, FC |

**Verdict:** No win/loss — cite as related long-form editing work only.

---

## 4. Do we need more data / metrics?

### Must-have before a strong paper claim (recommended)

| Metric / data | Why | Status |
|---------------|-----|--------|
| **LiveEdit-native** T-Gen, M-Gen, T-Loc, M-Loc for L0 vs DEK | Prove DEK doesn’t break what LiveEdit optimized | ⚠️ Partially available in eval; **need a clean side-by-side table** |
| **Image locality** under our probes | Show we don’t over-refuse correct related images | Check current eval outputs; report explicitly |
| **Human audit** of swap/hard-neg pairs (sheet ready, n=150) | Credibility of probes | Planned, not scored |
| Keep **FC + Swap + Rel** | Our novelty metrics | ✅ Done |

### Nice-to-have (AAAI stretch)

| Item | Why |
|------|-----|
| Second backbone (e.g. LLaVA) | LiveEdit showed multi-VLM; we are BLIP2-only |
| Larger `dsn` / `sen` (e.g. 500 / 50) | Closer to lifelong stress |
| IKE / LTE (if runnable) | Broader editor set |
| ROUGE-L / BERTScore / EM on long | Complement token-acc |
| Matched re-eval of **published LiveEdit protocol** Rel/Gen/Loc | Only if you want “beats Table X” language |

### Do **not** need (low ROI now)

- More λ / pool-buffer / FX sweeps for +swap  
- Claiming AnyEdit EditEverything numbers  
- Treating AR chunking as the main novelty  

---

## 5. Positioning diagram (for the meeting)

```text
AnyEdit (text LLM, long-form AR)
        │  cite only
        ▼
LiveEdit (lifelong VLM MoE + hard/soft routing)  ← base system
        │  extend
        ▼
Ours: AVBT + Dual-Expert Keys
        • new problem: visual leakage / false-cross
        • new train: adversarial binding
        • new rule: vision ∧ text dual-gate
        • win: FC ↓↓, Rel held, swap ↑ modest
```

---

## 6. Recommended supervisor wording

**Safe & accurate:**
> LiveEdit established lifelong VLM MoE editing under Rel/Gen/Loc. We show that under **adversarial image-swap**, LiveEdit still suffers massive **false-cross** retrieval. **AVBT + Dual-Expert Keys** cut false-cross roughly in half while preserving reliability. AnyEdit is related for long-form text editing but is not a multimodal lifelong competitor.

**Avoid:**
> We outperform LiveEdit and AnyEdit on all benchmarks / we are SOTA multimodal editors.

---

## 7. Bottom line

| | Novelty | Performance |
|--|---------|--------------|
| **vs LiveEdit** | New **problem+mechanism** on their stack (not a from-scratch editor) | **Win on FC / leakage**; Rel held; swap modest; **not** a full CVPR-table takeover yet |
| **vs AnyEdit** | Orthogonal | Not comparable |
| **Thesis** | ✅ Enough if framed as grounding leakage | ✅ |
| **AAAI main** | Stretch — need Gen/Loc parity table + audit (+ ideally 2nd backbone) | Don’t lead with swap SOTA |

**Next extraction priority:** one clean table of **Rel / T-Gen / M-Gen / T-Loc / M-Loc / Swap / FC** for LiveEdit baseline vs Dual-Expert Keys (same short protocol) — that answers “did we hurt LiveEdit’s original metrics?”
