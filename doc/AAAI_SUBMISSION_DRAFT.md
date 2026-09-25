# AAAI-27 Submission Draft (Anonymous Writing Copy)

**Venue:** The 41st AAAI Conference on Artificial Intelligence (**AAAI-27**), Montréal, Canada, 16–23 February 2027  
**Purpose:** Working manuscript text in AAAI-style section order. Paste into the official **AAAI-27 author kit** (LaTeX/Word) for the PDF.  
**Track:** Main Technical Track (double-blind).  
**Official instructions:** [Submission Instructions](https://aaai.org/conference/aaai/aaai-27/submission-instructions/) · [Main Track CFP](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/) · Author kit on the AAAI-27 site  
**Status:** **Title + Abstract submitted to OpenReview (locked substance).** Full paper due **28 July 2026**. Remaining sections are scaffolds.  
**LaTeX twin (Overleaf-ready):** `[AAAI_SUBMISSION_DRAFT.tex](AAAI_SUBMISSION_DRAFT.tex)` · BibTeX `[aaai_refs.bib](aaai_refs.bib)` · also copied into `[../AuthorKit27/](../AuthorKit27/)`

### Submitted OpenReview registration (do not change claim)

**Title:** Dual-Expert Keys for Reducing False-Positive Retrieval in Lifelong Vision–Language Model Editing  

**Abstract (submitted):**  
Lifelong vision–language editors can look correct on the image they were edited for, then still emit the new fact when the scene is only visually related. In our probes this is usually not soft overwrite of the base model. It is false-positive retrieval (false-cross): weak visual keys make a different stored edit look good enough to retrieve, so the wrong expert fires. We address it in two steps. Dual-Expert Keys let an edit fire only when the visual match and the prompt-key match both pass. Separately, adversarial visual binding training attaches each edit to its correct image and lowers expert activation when the image is swapped or a hard negative. On VLKEB-based adversarial grounding probes, Dual-Expert Keys cut swap false-positive retrieval rate from 96% to 46% over five seeds and keep reliability near 94%. Adversarial visual binding training complements Dual-Expert Keys by stabilizing edit–image binding under these adversarial probes. Against Fine-tuning, MEND-VL, and SERAC-VL on the same probes, we keep high reliability while directly reducing the wrong-expert retrieval rate.

**Terminology note for the full PDF:** Title uses *false-positive retrieval*; tables may still report **FC / false-cross** as the same quantity (wrong-expert retrieval rate). Keep that synonym explicit once in Intro/Problem.

### AAAI-27 author timetable (UTC-12 / AoE)


| Milestone                     | Date                              |
| ----------------------------- | --------------------------------- |
| OpenReview registration opens | 17 June 2026                      |
| Paper submission site opens   | 30 June 2026                      |
| **Abstracts due**             | **21 July 2026**, 11:59 PM UTC-12 |
| **Full papers due**           | **28 July 2026**, 11:59 PM UTC-12 |
| Supplementary / code due      | 31 July 2026                      |
| Phase-1 rejection notice      | 24 September 2026                 |
| Author feedback               | 19–25 October 2026                |
| Final accept/reject           | 30 November 2026                  |
| Camera-ready                  | 14 December 2026                  |


**Page budget (Main Technical Track):** up to **7 pages** of content + **pages 8–9 for references only** (≤ **9 pages** total in the review PDF). Acknowledgments omitted for review; ethics statements (if any) count inside the 7 content pages. Supplementary material is optional and reviewers are not required to read it.

**Critical:** Abstract registration must be a **complete** title + abstract (no placeholders). Do not change the abstract/title into a qualitatively different paper between the abstract and full-paper deadlines.

---



## AAAI-27 formatting checklist (do before PDF export)


| Rule             | Action                                                                                                    |
| ---------------- | --------------------------------------------------------------------------------------------------------- |
| Double-blind     | No names, affiliations, or acknowledgments in the PDF                                                     |
| Abstract         | No citations inside the abstract; same substance as OpenReview abstract                                   |
| Style            | Official **AAAI-27** two-column US Letter author kit                                                      |
| Fonts            | Do not shrink body font / line spacing                                                                    |
| Figures          | Readable in B&W; captions below figures / above tables                                                    |
| Claim discipline | Lead with **false-positive retrieval / false-cross (FC)** + reliability; do **not** claim token-swap SOTA |


**Internal note (not for the paper):** Lifelong MoE retrieval editors are the experimental *testbed*. The paper claims a **new failure mode + mechanism**, not an incremental add-on to any single prior system.

---



# Submitted title (locked)

**Dual-Expert Keys for Reducing False-Positive Retrieval in Lifelong Vision–Language Model Editing**

**Short / running head:** `Dual-Expert Keys for Lifelong VLM Editing`

Earlier title options (A–F) are archived; do not switch to a qualitatively different problem statement before the full-paper deadline.

---



# Dual-Expert Keys for Reducing False-Positive Retrieval in Lifelong Vision–Language Model Editing

*Anonymous Author(s)*  
*Paper ID: [OpenReview — submitted]*  
*Keywords:* lifelong model editing, vision–language models, false-positive retrieval, false-cross retrieval, multimodal knowledge editing, visual grounding

---



## Abstract

Lifelong vision–language editors can look correct on the image they were edited for, then still emit the new fact when the scene is only visually related. In our probes this is usually not soft overwrite of the base model. It is false-positive retrieval (false-cross): weak visual keys make a different stored edit look good enough to retrieve, so the wrong expert fires. We address it in two steps. Dual-Expert Keys let an edit fire only when the visual match and the prompt-key match both pass. Separately, adversarial visual binding training attaches each edit to its correct image and lowers expert activation when the image is swapped or a hard negative. On VLKEB-based adversarial grounding probes, Dual-Expert Keys cut swap false-positive retrieval rate from 96% to 46% over five seeds and keep reliability near 94%. Adversarial visual binding training complements Dual-Expert Keys by stabilizing edit–image binding under these adversarial probes. Against Fine-tuning, MEND-VL, and SERAC-VL on the same probes, we keep high reliability while directly reducing the wrong-expert retrieval rate.

*(Submitted to OpenReview for AAAI-27. Long-form FC may be added in the full PDF before 28 July if desired; do not change the core claim.)*

---



## Writing notes (authors only — delete before camera-ready PDF)

**Submitted abstract choices**

- Title term: false-positive retrieval; body alias: false-cross.
- Long-form omitted; optional in full paper only.
- Adversarial visual binding framed as complement (binding), not as false-positive ablation in the abstract.
- Neighbor mining / pool-buffer negatives → Results section, not abstract.

**Why this framing is strong**

- Opens with a community problem (lifelong VLM edit leakage), not “we improve X.”
- Names a measurable failure (false-positive / false-cross retrieval).
- Method is **ours** (Dual-Expert Keys + adversarial visual binding).
- Numbers lead with FC + reliability.

**What we deliberately avoid**

- “Building on / extension of” any single prior editor.
- Treating image-swap token rates as a grounding win.
- Positioning AnyEdit as a direct competitor.

**Optional insert before July 28 (same substance):**  
`… and from 88% to 38% under matched long-form training (reliability ~73%).`

---



## 1. Introduction

Large vision–language models (VLMs) now answer questions about photographs, documents, and everyday scenes, and they are often treated as soft knowledge bases in interactive systems (Li et al. 2023).
Facts encoded in these models drift: places change names, entities are mislabeled, and new visual knowledge appears after pre-training ends.
Full fine-tuning is costly and can disturb unrelated skills, so *model editing* has emerged as a practical alternative—apply a local update that corrects a target fact while preserving behaviour elsewhere (Mitchell et al. 2022a; 2022b; Cheng et al. 2023).
Multimodal benchmarks such as MMEdit and VLKEB standardize this goal around reliability, generality, and locality (Cheng et al. 2023; Huang et al. 2024).
When edits must arrive one after another, the problem becomes *lifelong* editing: each new fact joins a growing set of updates that must remain available without overwriting earlier ones (Chen et al. 2025).

Retrieval- and mixture-of-experts (MoE) editors are a natural fit for that setting.
They store each edit as an expert (or memory entry) and, at test time, retrieve which expert(s) to inject for a query (Mitchell et al. 2022b; Chen et al. 2025).
On the image used for the edit, these systems can look excellent: reliability often sits in the low-to-mid 90s.
The harder question is whether the edit is *visually grounded*.
Consider a city-entity edit with prompt *p* (“What city in Illinois is depicted in the image?”) and target answer “Kankakee.”
With the edited photograph, the post-edit model answers correctly.
With the *same* prompt but a swapped skyline or a confusable hard-negative scene, a retrieval editor can still surface “Kankakee” (Figure 1).
That behaviour is especially damaging in lifelong pools: the more experts accumulate, the more chances a weak visual key has to match the wrong past edit.

Teaser: reliability vs false-positive retrieval under a shared prompt; Dual-Expert Keys require vision∧text agreement.

**Figure 1.** Top: shared prompt with the edited image (reliability), an image-swap, and a hard-negative. Bottom: Dual-Expert Keys fire an edit only when vision and text gates both pass; adversarial visual binding strengthens edit–image attachment under these probes.

We find that this failure is usually not soft overwrite of the frozen base VLM.
On our adversarial grounding probes, routing logs show that a *foreign* stored expert is retrieved for the swapped image—often at rates above 90%.
We call this *false-positive retrieval* (and *false-cross* when we score it from expert identities): the visual key for edit *i* is similar enough to query image *j* ≠ *i* that the wrong expert passes the retrieval threshold and injects the wrong fact.
This is a retrieval-safety problem, not merely an answer-quality problem.
Yet the metrics that dominate multimodal editing papers—reliability on the edited image, text/image generality, and locality on unrelated prompts (Cheng et al. 2023; Huang et al. 2024; Chen et al. 2025)—do not report which expert fired under image-swap or hard-negative probes.
Lifelong VLM editors such as LiveEdit already combine visual filtering with text routing (Chen et al. 2025), but those designs are optimized for Rel/Gen/Loc, not for measuring or blocking wrong-expert activation on adversarial images.
DualEdit studies *where* to place modality-specific adapters for reliability–locality trade-offs (Shi et al. 2025); it does not address lifelong expert-pool leakage.
Weight-updating editors (Fine-tuning, MEND, SERAC) (Mitchell et al. 2022a; 2022b) can rewrite answers, but when the competing system is a growing expert memory they leave false-positive retrieval unmeasured.

**Key insight.**
If leakage is dominated by wrong-expert selection, then improving only the binding of an edit to its own image is not enough: the retrieval rule itself must reject candidates unless *both* modalities agree.
A vision-only gate can still admit a foreign edit whose visual key collides with the query; a text-only gate can still admit that edit when the prompt matches.
Requiring vision ∧ text agreement turns retrieval into a dual check: the image must look like the edited scene *and* the prompt must match the stored edit key.

**Our approach.**
We study this failure on a lifelong MoE-style editor (Chen et al. 2025) with BLIP-2 (Li et al. 2023) as the frozen backbone—the setting where continuous expert pools make false-positive retrieval observable.
We introduce two complementary pieces.
(1) *Dual-Expert Keys*: at inference, an expert may fire only when its visual gate and its prompt-key (text) gate both pass.
(2) *Adversarial visual binding training*: during editing, we attach each expert to its correct image and explicitly suppress fused expert activation on swap and hard-negative images.
Dual-Expert Keys are the primary anti-leakage mechanism; adversarial visual binding stabilizes edit–image attachment so the dual-gate is not fighting a weakly grounded expert.

**Empirical preview.**
On VLKEB-based adversarial grounding probes (Huang et al. 2024), Dual-Expert Keys cut the swap false-positive retrieval rate from **96% to 46%** over five evaluation seeds while keeping reliability near **94%**.
Standard Rel/Gen/Loc stay essentially on parity with the lifelong baseline.
Ablations show that binding training alone leaves false-cross near 96%; adding Dual-Expert Keys is what halves wrong-expert retrieval.
Under a matched long-form train/eval protocol, false-positive retrieval similarly falls from 88% to 38% with reliability held near 73%.
Against Fine-tuning, MEND-VL, and SERAC-VL on the same probes, our method is the one that keeps high reliability while directly reducing wrong-expert retrieval.
We emphasize the claim carefully: we target *anti-leakage routing*, not token-level image-swap state of the art.

**Contributions.** This paper makes three contributions:

1. **Problem.** We characterize false-positive retrieval (false-cross) as a dominant visual grounding failure in lifelong multimodal editors under image-swap and hard-negative probes, and show that it is under-reported by Rel/Gen/Loc alone.
2. **Method.** We propose Dual-Expert Keys, which require vision–text agreement at retrieval time, together with adversarial visual binding training that attaches edits to their images and suppresses adversarial activation.
3. **Evidence.** On VLKEB grounding probes we show large false-positive cuts with reliability preserved, ablations that isolate Dual-Expert Keys as the FC mechanism, multi-seed stability, and comparisons to Fine-tuning, MEND-VL, and SERAC-VL.

The rest of the paper situates related work (Section 2), formalizes probes and false-cross (Section 3), presents adversarial visual binding and Dual-Expert Keys (Section 4), then reports the experimental protocol and results (Sections 5–6). We close with limitations and conclusions.

---



## 2. Related Work

**Knowledge editing for language models.**
Modern LMs store large amounts of factual knowledge, but that knowledge can be wrong, outdated, or incomplete.
*Model editing* aims to inject a local correction without full retraining and without broadly damaging unrelated behaviour (De Cao et al. 2021; Yao et al. 2023).
The literature is usually grouped into three families.
*Locate-then-edit* methods first identify parameters associated with a fact and then rewrite them; ROME does this for single edits, and MEMIT scales the idea to many facts (Meng et al. 2022; 2023).
*Meta-learning* editors such as MEND train a hypernetwork to turn edit gradients into low-rank weight updates (Mitchell et al. 2022a).
*Memory / retrieval* editors keep the base model frozen and store edits externally: SERAC retrieves a counterfactual memory and routes through a scope classifier (Mitchell et al. 2022b), GRACE maintains lifelong discrete key–value adaptors (Hartvigsen et al. 2023), and in-context editors such as IKE place edit demonstrations in the prompt (Zheng et al. 2023).
EasyEdit unifies many of these methods behind a common interface, which has made fair comparison easier (Wang et al. 2024).
Beyond single-fact reliability, later evaluation work asks whether an edit propagates consistently to related facts (Cohen et al. 2024) and whether editors can handle long or diverse-format targets (Jiang et al. 2025).
AnyEdit is the closest long-form *text* line: it edits chunk by chunk under an autoregressive locate-then-edit view (Jiang et al. 2025).
We cite it for that contribution, but it is not a competitor on VLKEB visual grounding: it has no image modality and no lifelong multimodal expert pool.

**Multimodal knowledge editing and benchmarks.**
Editing vision–language models is harder than editing text-only LMs because the fact is tied to an image as well as a prompt (Cheng et al. 2023).
MMEdit established early multimodal reliability / generality / locality protocols for VQA- and caption-style edits (Cheng et al. 2023).
VLKEB improves visual quality by building edits from multimodal knowledge graphs and adds portability-oriented checks (Huang et al. 2024).
More recent benchmarks push different axes: MC-MKE emphasizes modality consistency across fine-grained edit formats (Zhang et al. 2025), MMKE-Bench covers diverse visual semantics and free-form knowledge (Du et al. 2025), and ComprehendEdit aggregates multiple tasks with generalization / preservation indices (Ma et al. 2025).
On the method side, DualEdit studies *where* to edit: it updates text and vision at modality-sensitive layers and uses a gate to protect original behaviour (Shi et al. 2025).
That gating is about reliability–locality trade-offs under layer choice, not about blocking wrong-expert retrieval in a lifelong MoE pool.
Across this literature, success is still mostly reported as: the new answer appears on the edited (or rephrased) input, and unrelated inputs stay stable.
What is usually missing is an explicit account of *which stored edit* fired when the prompt is held fixed and the image is swapped or made confusable.

**Lifelong retrieval editors and visual keys.**
Lifelong editing requires many updates to coexist over time (Hartvigsen et al. 2023; Chen et al. 2025).
LiveEdit is the closest system to our setting: it generates a low-rank MoE expert per edit, stores experts in a growing repository, and at inference uses hard visual filtering followed by soft text routing to fuse relevant experts into a frozen VLM such as BLIP-2 (Chen et al. 2025; Li et al. 2023).
Related multimodal backbones used in editing studies include LLaVA and InstructBLIP (Liu et al. 2023; Dai et al. 2023).
Because retrieval decides which residual is injected, routing quality becomes part of editing safety as the pool grows—an issue that sparse MoE selection has long faced in other domains (Shazeer et al. 2017).
Strong visual encoders can help in principle: CLIP-style vision–language similarity (Radford et al. 2021) and DINOv2 features (Oquab et al. 2024) often separate images well in offline probes.
But offline discrimination does not automatically transfer to *edit retrieval* when the editor compresses vision into a few routing keys and scores them against a large lifelong pool.
Weight-updating multimodal adaptations of Fine-tuning, MEND, and SERAC remain important baselines (Mitchell et al. 2022a; 2022b; Cheng et al. 2023); they can change answers, yet they do not expose a growing expert identity that we can score for false-cross.

**Positioning and distinction from close neighbours.**
Our focus is an under-measured lifelong multimodal failure: *false-positive retrieval* (false-cross) under image-swap and hard-negative probes.
Dual-Expert Keys require vision ∧ text agreement before an expert may fire; adversarial visual binding training attaches each edit to its image and suppresses activation on adversarial scenes.
This differs from DualEdit, which chooses modality-specific layers and a Rel/Loc gate (Shi et al. 2025), and from AnyEdit, which targets long-form text locate-then-edit (Jiang et al. 2025).
It also differs from benchmark papers that expand Rel/Gen/Loc, portability, or modality consistency without measuring wrong-expert retrieval rates (Huang et al. 2024; Zhang et al. 2025; Du et al. 2025; Ma et al. 2025).
We use a LiveEdit-style lifelong MoE editor as the experimental *testbed* because continuous expert pools make false-cross observable and actionable.
The scientific claim is not “higher Rel/Gen/Loc than LiveEdit on the original leaderboard,” but a diagnosis and mechanism for anti-leakage retrieval under adversarial images.

---

\section{3. Problem: Visual Grounding Leakage via False-Positive Retrieval}

Lifelong multimodal editors that store one expert per edit are attractive because the base VLM can stay frozen. As edits accumulate, though, the system is only as safe as its retrieval step. Write $\mathcal{E}=e_1,\ldots,e_m$ for the pool; each $e_i$ remembers an edit $(p_i,x_i,y_i)$. On a query $(p,x)$ the editor scores the pool and injects residuals into the frozen model. On the photograph that was actually edited this can look excellent—Rel/Gen/Loc in the literature are often already high. The awkward case is the one those scores largely ignore: keep the prompt fixed, replace the image, and ask whether the pool still hands you an expert that belongs to a different scene.

To make that concrete we use adversarial grounding probes around the same edit prompt $p_i$ when we can. **Reliability** puts the edited image $x_i$ back; **image-swap** feeds another edit's image $x_j$; **hard-negative** keeps $p_i$ but uses a near-neighbour distractor that was never an edit image. Text rephrases and locality checks stay as controls—if rephrase tracks reliability but swap/hard-negative collapse, the trouble is unlikely to be "language ignored the picture." We also log which pool experts fired, not only token overlap with $y_i$.

Answer-level swap accuracy tells you *what was said*. We care at least as much about *which memory was opened*. **False-positive retrieval** means that, for an adversarial query aimed at edit $i$, retrieval activates at least one expert $e_k$ with $k\neq i$. That rate (lower is better) is our main leakage number—**FPR** in tables. A non-zero token score on a wrong image can happen for many reasons; high FPR points at the lifelong pool selecting a foreign expert.

Standard Rel/Gen/Loc never ask, under a shared prompt and an adversarial image, whether the retrieved expert matches the scene. So a system can post strong Rel/Gen/Loc while colliding visual keys still admit foreign edits. On our VLKEB evaluation split with adversarial probes and a lifelong MoE baseline, reliability sits near 94% while swap FPR climbs to **96%**: solid on its own photograph, yet almost always retrieving someone else's expert under swap. Binding the edit more tightly to its own image is not enough by itself—if foreign experts keep clearing the bar, the rule has to change: fire only when image evidence and prompt key both agree. That motivates Dual-Expert Keys and adversarial visual binding (Section 4).

---



## 4. Method

We target false-positive retrieval directly with two coupled mechanisms. **Dual-Expert Keys** change the inference rule: a stored edit may fire only when vision and prompt-key evidence both agree. **Adversarial visual binding** attaches each edit to its correct image and suppresses swap/hard-negative activation. Binding helps adversarial answers somewhat; Dual-Expert Keys cut FPR roughly in half. We instantiate this on a frozen VLM with a growing residual pool because wrong-expert retrieval is observable there; the claim is the retrieval-safety mechanism, not a Rel/Gen/Loc leaderboard entry for a prior lifelong editor.

### 4.1 Lifelong editor and keys

Edit $(p_i,x_i,y_i)$ writes expert $e_i$, Q-Former key $\mathbf{k}^v_i$, DINOv2+PatchGrounding key $\mathbf{k}^p_i$, prompt key $\mathbf{k}^t_i$ into pool $\mathcal{E}_t$ (Eq. pool). Query keys $(\mathbf{q}^v,\mathbf{q}^p,\mathbf{q}^t)$; vision logit mixes Q-Former and patch sims with fuse $\alpha$; text logit from prompt keys; prototype vision score is the no-edit baseline.

### 4.2 Dual-Expert Keys

- Vision gate: $s^v_i > s^v_{\mathrm{prot}}$
- Text gate: $s^t_i \ge \max_j s^t_j - m$
- Retain: $r_i = \mathrm{visionok}_i \wedge \mathrm{textok}_i$
Survivors soft-fuse into MoE residual. Primary FPR mechanism on VLKEB: 96%→46% (fair extended-answer check: 88%→38%).



### 4.3 Adversarial visual binding

$\mathcal{L}*{\mathrm{pos}}=-\log(\sum*{i\in\mathcal{O}}\pi_i)$ on correct image;  
$\mathcal{L}*{\mathrm{neg}}=\mathrm{ReLU}(m*{\mathrm{exp}}-\tau)(-\log\pi_{\mathrm{prot}})+\mathcal{L}*{\mathrm{aux}}$ on swap/hard-neg;*  
*$\mathcal{L}=\mathcal{L}*{\mathrm{edit}}+\lambda\mathcal{L}_{\mathrm{bind}}$ with $\lambda{=}0.3$, $\beta{=}2.5$, $\tau{=}0.3$.  
Binding alone leaves FPR ≈96% — supporting training, not the false-positive fix.

### 4.4 Not the method row

Neighbor mining / pool buffer: secondary ablations only.

---



## 5. Experiments

We want the evaluation to answer one question cleanly: under adversarial images, does Dual-Expert Keys reduce false-positive retrieval without giving up reliability on the edited photograph? Everything below is organized around that question rather than around chasing the highest token-swap score.

### Backbone and editor testbed

All primary runs use BLIP-2 OPT-2.7B as a frozen vision--language backbone. We edit it with a LiveEdit-style lifelong MoE editor: each training edit writes a low-rank residual expert into a growing pool, and inference retrieves which expert(s) to inject. We keep this architecture as the experimental *testbed* because continuous expert pools make wrong-expert retrieval observable; the scientific claim is about anti-leakage routing, not about beating LiveEdit on the original Rel/Gen/Loc leaderboard. Weight-updating multimodal editors (Fine-tuning, MEND-VL, SERAC-VL) appear as external comparisons. They can change answers, but they do not expose a growing expert identity that we can score for FPR, so FPR is reported only for pool-based systems.

### Datasets and probes

Our main benchmark is the original VLKEB multimodal editing dataset. Each VLKEB example already provides an edit triple $(p_i,x_i,y_i)$ plus text rephrase, image rephrase, and text / multimodal locality. We keep those fields unchanged. VLKEB does *not* test whether retrieval still fires the right expert when the prompt is fixed and the image is wrong, so we build an adversarial overlay on the original VLKEB splits rather than a separate benchmark.

**Construction.** From the original VLKEB train/eval JSONs and image root we attach two probes to every retained edit:

- **Image-swap:** keep $p_i,y_i$, replace the image with $x_j$ from another edit $j\neq i$ in a *different* VLKEB entity folder.
- **Hard-negative:** keep $p_i,y_i$, replace the image with a confusable different-entity distractor, preferring partners whose prompts share a coarse question skeleton with $p_i$.

Missing image files skip to the next partner; edits without both probes are dropped. Rel/Gen/Loc stay on the original VLKEB content; only the adversarial images are ours.

**Training vs evaluation.** Lifelong baseline trains on original VLKEB training edits. Adversarial visual binding and Dual-Expert Keys also use the probe-augmented training split for the binding loss. Primary tables evaluate the probe-augmented VLKEB eval split. Cross-dataset check: same pairing on EasyEdit EVQA.

Probe suite: reliability, image-swap, hard-negative, plus VLKEB text/image rephrase and locality.

### Metrics

Standard Rel / T-Gen / M-Gen / T-Loc / M-Loc for context. On adversarial probes: swap and hard-negative accuracy, reliability--swap gap. Primary novelty metric: **false-positive retrieval (FPR)** from routing logs (lower better). Table reading order: FPR first, then Rel, then swap/hard/gap as secondary behaviour.

### Baselines, comparisons, and ablations

- **Lifelong baseline:** LiveEdit-style MoE retrieval (no binding training, no Dual-Expert Keys).
- **Binding only:** adversarial visual binding with patch grounding; original retrieval rule.
- **Dual-Expert Keys (ours):** binding training plus vision ∧ text retrieval gate.
- **Neighbor mining / pool buffer:** Dual-Expert Keys plus either false-cross neighbor mining or a larger pool buffer (negative / secondary only).
- **Fine-tuning / MEND-VL / SERAC-VL:** weight-updating multimodal editors (no false-positive retrieval score).

Unless noted: binding loss $\lambda{=}0.3$, $\beta{=}2.5$, $\tau{=}0.3$, contrastive aux on; Dual-Expert Keys conjunction gate with fixed text margin $m$.

### Evaluation protocol

Main VLKEB tables: $100$ edits, $10$ generations per example. Headline ablation/competitors at seed 42; multi-seed credibility on seeds $42,0,1,2,3$. EVQA stress uses the same three-way contrast (lifelong baseline, binding only, Dual-Expert Keys) with a $300$-edit multi-seed subsample when reporting means±std. Routing identities are logged whenever false-positive retrieval is claimed. Token-level image-swap is *not* a submission gate: the intended claim is false-positive retrieval ↓ with reliability held.

---



## 6. Results

Unless noted otherwise, VLKEB uses the probe-augmented original eval split (100 edits, 10 generations). Lead metric: **FPR ↓ at Rel parity**.

### Main VLKEB result


| Method               | Rel↑      | T-Gen↑    | M-Gen↑    | T-Loc↑   | M-Loc↑   | FPR↓    |
| -------------------- | --------- | --------- | --------- | -------- | -------- | ------- |
| Lifelong baseline    | 93.5%     | 94.3%     | 92.1%     | 100%     | 100%     | 96%     |
| Binding only         | 93.8%     | 94.5%     | 93.8%     | 100%     | 100%     | 96%     |
| **Dual-Expert Keys** | **93.7%** | **94.5%** | **92.0%** | **100%** | **100%** | **46%** |




### Ablation

Figure: `Figures/fig_results_ablation.png`. Binding lifts swap (~+7 pp) but FPR stays 96%; Dual-Expert Keys cut FPR to 46%. Neighbor mining / pool buffer are negatives (pool buffer regresses FPR to 55%).

### Multi-seed + significance

Figure: `Figures/fig_results_multiseed.png`. Five seeds: FPR 96%→46% every seed (−50 pp); swap +7.4 pp ($p=3.4\times10^{-4}$); Rel held.

### Competitors

Figure: `Figures/fig_results_competitors.png`. Fine-tuning / MEND-VL / SERAC-VL: lower Rel, near-zero swap-gap, FPR n/a. Dual-Expert Keys uniquely keep ~94% Rel and cut FPR.

### EVQA cross-dataset stress

Figure: `Figures/fig_results_evqa.png`. Five seeds × 300 edits: Rel ~95%; Dual-Expert Keys FPR swap 1.5%±1.1%, hard 2.7%±1.3% (baseline hard ~96.8%). No EVQA swap-SOTA claim.

**Claim:** Dual-Expert Keys deliver a large, stable FPR cut with Rel held; binding training supports but does not replace the dual gate.

---



## 7. Limitations

The results establish a clear win on the metric we set out to improve: Dual-Expert Keys cut false-positive retrieval while preserving high reliability. What follows only situates that win.

Our primary success criterion is retrieval safety rather than adversarial token accuracy. Remaining image-swap answer headroom is a separate generation problem for the frozen decoder—not evidence that the dual gate failed. Stopping the wrong expert from firing is already a substantial gain even when the model does not always refuse the edited answer string on a mismatched photograph.

False-positive retrieval is scoped to lifelong editors with an identifiable expert pool, where leakage is observable and actionable. Fine-tuning / MEND-VL / SERAC-VL remain useful operating-point baselines but do not admit the same expert-identity score; marking FPR as n/a is a property of the metric, not a Dual-Expert Keys shortcoming.

Finally, the main evidence uses BLIP-2 OPT-2.7B on VLKEB, complemented by a completed EVQA cross-dataset stress that reproduces the same FPR pattern—covering both the primary editing benchmark and an out-of-distribution probe set under the same dual-gate rule.

---



## 8. Conclusion

Lifelong multimodal editors can look correct on the edited image and still retrieve the wrong stored fact when the scene changes. We identify that as *false-positive retrieval*—missed by Rel/Gen/Loc alone. Dual-Expert Keys fix the decision rule (vision ∧ text); adversarial visual binding strengthens edit–image attachment in training.

Empirically the combination works: on original VLKEB, Dual-Expert Keys cut FPR **96%→46%** with Rel near **94%**, stable across five seeds, strong at a high-reliability operating point vs Fine-tuning / MEND-VL / SERAC-VL, and reproduced under EVQA stress. Binding alone does not deliver the FPR win; the dual gate does. Lifelong multimodal editing can be made substantially safer at retrieval time without sacrificing the reliability existing benchmarks already reward.

---



## References

*[To be compiled in AAAI BibTeX.]*

---



## Reproducibility Checklist

*[Fill from AAAI-26 checklist PDF when preparing final PDF.]*

---



## Appendix (optional / supplementary)

- Intro teaser: `figures/aaai/fig1_teaser.png` (regenerate: `tools/make_aaai_fig1_teaser.py`)
- Architecture diagram: `figures/aaai/architecture_avbt_dek.png` (for Method)
- Negative ablations: neighbor mining / pool buffer  
- Extra extended-answer / seed tables if page budget allows
- Binding-layer expert_mass tables for adversarial visual binding  
- Human audit protocol: `AAAI_HUMAN_AUDIT_PLAN.md`

