# Direction A: Long-Form Lifelong VLM Editing — Full Plan

## 1. One-Sentence Thesis Claim

> Existing lifelong VLM editors (LiveEdit) fail on long-form edit targets due to the single-step efficacy barrier, and existing long-form editors (AnyEdit) lack multimodal and lifelong support; we propose **[Method Name]**, the first framework that enables lifelong editing of long-form multimodal knowledge in vision-language models, with provable chunk-wise consistency and bounded locality.

---

## 2. Problem Statement

### 2.1 What exists

| Paper | Venue | Lifelong | Multimodal | Long-form |
|-------|-------|----------|------------|-----------|
| LiveEdit (Chen et al.) | CVPR 2025 | Yes (1000+ edits) | Yes (VLMs) | No (targets <=16 tokens) |
| AnyEdit (Jiang et al.) | ICML 2025 | No | No (LLM only) | Yes (up to 458 tokens) |
| DualEdit (Kim et al.) | COLM 2025 | No | Yes | No |
| MEMOIR (Wang et al.) | NeurIPS 2025 | Yes (15k edits) | No (LLM only) | No |
| RILKE (Liu et al.) | arXiv 2025 | Yes | No (LLM only) | Unstructured text |
| KDKE | ICLR 2026 sub | No | Yes (MLLMs) | No |

### 2.2 The gap

Nobody has solved **lifelong + multimodal + long-form** together. Both LiveEdit and AnyEdit explicitly flag this as future work:
- AnyEdit (Section 7): "not optimized for lifelong editing" and "confined to textual knowledge editing; lacks support for multimodal"
- LiveEdit: all benchmarks (E-VQA, E-IC, VLKEB) have targets <=18 tokens

### 2.3 Why this matters

Real-world VLM corrections often require long, detailed answers:
- Medical image reports (50-200 tokens)
- Detailed visual descriptions / captions (30-100 tokens)
- Visual reasoning chains (40-150 tokens)
- Correcting hallucinated narratives (50-300 tokens)

Current VLM editors can only correct short factual answers (e.g., "Paris" → "London").

---

## 3. Proposed Method (High-Level Design)

### 3.1 Architecture overview

Build on LiveEdit's MoE framework but extend the expert generation and routing to handle chunk-wise editing:

```
Input: (image, prompt, long_target)
                    |
          [Chunk long_target into K chunks]
                    |
    For each chunk k = 1, ..., K:
        |
        [Generate chunk-aware expert (U_k, V_k) using LiveEdit's feg]
        |
        [Extract routing features for chunk k]
        |
        [Store in expert repository with chunk metadata]
                    |
    Inference: route to relevant experts per-chunk, fuse, generate
```

### 3.2 Key innovations (pick 1-2, not all)

**Innovation 1: Vision-Aware Chunking**
- Instead of fixed-size sliding window (AnyEdit) or sentence boundaries, use the image's visual attention map to determine chunk boundaries
- Chunks that describe the same visual region should stay together
- This is multimodal-specific and cannot be done in text-only AnyEdit

**Innovation 2: Cross-Modal Chunk Consistency Loss**
- When editing chunk k, add a loss term that penalizes degradation of visual grounding for chunks 1..k-1
- Ensures that editing later chunks doesn't break earlier visual understanding
- Formally: L_consistency = KL(f_θ(o_{1..k-1} | v, p) || f_θ'(o_{1..k-1} | v, p)) summed over previous chunks

**Innovation 3: Chunk-Aware Expert Routing**
- Extend LiveEdit's hard/soft routing to include chunk position information
- Different chunks of the same edit may need different routing weights
- Early chunks (visual description) may route more on visual similarity
- Later chunks (textual reasoning) may route more on textual similarity

### 3.3 Theoretical grounding

AnyEdit's MI chain rule proof (Theorem B.1) conditions on hidden states h'_k:

```
I(X; Y | h'_1, ..., h'_K) = Σ_k I(X, Y_1, ..., Y_{k-1}; Y_k | h'_k)
```

For VLMs, extend to condition on image representation V:

```
I(X, V; Y | h'_1, ..., h'_K) = Σ_k I(X, V, Y_1, ..., Y_{k-1}; Y_k | h'_k)
```

**Key argument:** Since the image encoder is frozen and image tokens enter as a fixed prefix, V acts as a constant conditioning variable. The decomposition holds with V absorbed into the input X' = (X, V). This requires a 1-2 paragraph argument (sufficient for thesis) or a 1-page formal extension (for journal).

---

## 4. Benchmark: Long-Form VLM Editing Dataset

### 4.1 Construction strategy

**Option A (recommended): Extend VLKEB**
- Use existing VLKEB images + questions
- Replace short `alt` answers with long-form answers (50-150 tokens)
- Generation: Use GPT-4V/GPT-4o with the image + original question to produce detailed answers
- Manual verification: check a random 10-20% subset for factual grounding
- Keep original short `alt` as a "short-form" split for regression testing

**Option B: Adapt existing long-answer VQA**
- A-OKVQA, OK-VQA have some longer answers but still mostly short
- Would require significant curation

**Option C: Build from scratch**
- Most effort, but full control over quality
- Use diverse image sources (medical, scene description, document understanding)

### 4.2 Dataset specifications

| Property | Value | Rationale |
|----------|-------|-----------|
| Total eval samples | 300-500 | Sufficient for statistical significance with 3+ seeds |
| Total train samples | 1000-2000 | Enough to train LiveEdit's expert generator |
| Target length bins | Short (<=16), Medium (17-64), Long (65-150) | Matches AnyEdit's analysis range |
| Min samples per bin | 80-100 in eval | Needed for reliable bin-level metrics |
| Image source | VLKEB images (reuse) | Avoids licensing issues |
| Answer generation | GPT-4V + manual QA | Balances scale and quality |
| Splits | Train / Eval (no overlap in images) | Standard practice |

### 4.3 Dataset fields (per sample)

```json
{
  "src": "What is happening in this image?",
  "alt_short": "A birthday party",
  "alt_long": "The image shows a birthday celebration in a backyard setting. Several children are gathered around a rectangular table covered with a blue tablecloth. In the center of the table is a two-tier chocolate cake decorated with colorful sprinkles and six lit candles. The birthday child, wearing a red party hat, is leaning forward to blow out the candles while other children watch with excited expressions. Balloons in various colors are tied to the fence posts in the background.",
  "rephrase": "Can you describe the scene in the picture?",
  "image": "m.027l4q/google_12.jpg",
  "image_rephrase": "m.027l4q/yahoo_4.jpg",
  "loc": "when did prison break season 4 come out",
  "loc_ans": "September 1, 2008",
  "m_loc": "m.0mkp7/bing_5.jpg",
  "m_loc_q": "What county is shown?",
  "m_loc_a": "Outagamie County",
  "target_token_count_short": 3,
  "target_token_count_long": 89
}
```

---

## 5. Evaluation Design

### 5.1 Metrics

| Metric | Short targets | Long targets | Purpose |
|--------|--------------|--------------|---------|
| Exact Match (EM) | Primary | Report but expect low | Backward-compatible with LiveEdit |
| Token Accuracy | Primary | Primary | Per-token correctness |
| ROUGE-L | Optional | Primary | Sequence-level similarity |
| BERTScore | Optional | Primary | Semantic similarity |
| Chunk EM | Report | Primary | AnyEdit-style per-chunk EM |
| Chunk F1 | Report | Primary | Already in your eval code |
| Edit Time (s) | Report | Report | Efficiency comparison |
| Locality (text + image) | Primary | Primary | Must stay bounded |
| Generality (text + image) | Primary | Primary | Must generalize to rephrase |

### 5.2 Experiment table (what the results section looks like)

**Table 1: Short-form VLKEB (regression test)**

| Method | Rel. Acc | Rel. EM | T-Gen | M-Gen | T-Loc | M-Loc |
|--------|----------|---------|-------|-------|-------|-------|
| LiveEdit (baseline) | - | - | - | - | - | - |
| AR-LiveEdit (ours) | - | - | - | - | - | - |
| FT-L | - | - | - | - | - | - |

**Table 2: Long-form VLKEB-Long (main result)**

| Method | Rel. Acc | ROUGE-L | BERTScore | Chunk EM | Chunk F1 | T-Loc | M-Loc |
|--------|----------|---------|-----------|----------|----------|-------|-------|
| LiveEdit | - | - | - | - | - | - | - |
| AR-LiveEdit (ours) | - | - | - | - | - | - | - |
| FT-L | - | - | - | - | - | - | - |
| AnyEdit-adapted* | - | - | - | - | - | - | - |

**Table 3: Length-bin breakdown (key evidence)**

| Method | Bin | Count | EM | ROUGE-L | Chunk F1 |
|--------|-----|-------|-----|---------|----------|
| LiveEdit | <=16 | - | - | - | - |
| LiveEdit | 17-64 | - | - | - | - |
| LiveEdit | 65+ | - | - | - | - |
| Ours | <=16 | - | - | - | - |
| Ours | 17-64 | - | - | - | - |
| Ours | 65+ | - | - | - | - |

**Table 4: Ablation**

| Config | ROUGE-L (long) | Locality | What it tests |
|--------|----------------|----------|---------------|
| Full method | - | - | Complete system |
| - chunk-wise (single step) | - | - | Is chunking needed? |
| - vision-aware chunking (fixed window) | - | - | Does vision-aware help? |
| - consistency loss | - | - | Does cross-chunk consistency help? |
| - chunk-aware routing (standard routing) | - | - | Does chunk routing help? |
| chunk_size=8 | - | - | Chunk size sensitivity |
| chunk_size=16 | - | - | |
| chunk_size=32 | - | - | |

### 5.3 Experimental protocol

- **Seeds:** 3 different random seeds (e.g., 42, 123, 456), report mean ± std
- **Edit counts:** Test at 1, 10, 100, 500 edits (lifelong scalability)
- **Same eval split** for all methods (fixed seed shuffle)
- **Matched training:** Same epochs, learning rate, batch size for all methods

---

## 6. Implementation Phases

### Phase 1: Validation (1-2 weeks)
- Run pre-implementation checklist (see separate document)
- Confirm GPU can handle long targets
- Confirm pipeline handles long targets end-to-end
- Generate 10-20 pilot long-form samples

### Phase 2: Dataset Construction (2-3 weeks)
- Generate long-form answers for VLKEB images using GPT-4V
- Quality-check 10-20% manually
- Split into train/eval
- Verify token length distribution covers all bins

### Phase 3: Baseline Experiments (1-2 weeks)
- Train LiveEdit on long-form data (standard, no AR)
- Evaluate → expect poor long-form performance (this is your "gap" evidence)
- Train FT-L baseline on same data
- Document failure modes qualitatively

### Phase 4: Method Implementation (2-3 weeks)
- Implement chosen innovation(s)
- Train on long-form data
- Evaluate and compare

### Phase 5: Ablations + Analysis (1-2 weeks)
- Chunk size sweep
- Ablate each component
- Qualitative examples (5-10 success + 5-10 failure cases)
- Lifelong scaling curve (1 → 500 edits)

### Phase 6: Writing (2-3 weeks)
- Thesis chapters / paper draft
- Figures: method diagram, length-bin bar charts, scaling curves
- Tables: fill in all experiment tables above

**Total estimated: 9-15 weeks**

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GPU OOM on long targets | High | Blocks training | Gradient accumulation, reduce batch size, cap target at 100 tokens |
| GPT-4V generates hallucinated answers | Medium | Bad benchmark | Manual verification of 10-20% + consistency checks |
| LiveEdit already works well on long targets | Low | No gap to fill | Unlikely given single-step design, but if so → pivot to scalability (Direction B) |
| AR gives no improvement over baseline | Medium | Weak results | This itself is a valid finding if you show WHY (analyze failure modes). Add the innovation (vision-aware chunking or consistency loss) to address it |
| Cannot adapt AnyEdit to BLIP2 for baseline | High | Missing baseline | Document why (encoder-decoder vs decoder-only), exclude with justification, use FT-L and LiveEdit as baselines instead |

---

## 8. Target Venues (ordered by fit)

| Venue | Type | Deadline (typical) | Fit |
|-------|------|---------------------|-----|
| ACL / EMNLP / NAACL | NLP conference | Feb/Jun/Oct | Strong (knowledge editing is a hot topic) |
| CVPR / ECCV | Vision conference | Nov/Mar | Good (VLM editing) |
| NeurIPS / ICML / ICLR | ML conference | May/Jan/Sep | Good if theory is strong |
| IEEE TPAMI / IJCV | Q1 journal | Rolling | Good for extended version |
| Pattern Recognition | Q1 journal | Rolling | Good for applied ML |
| Information Fusion | Q1 journal | Rolling | Good if multi-modal fusion angle |
| Neural Networks | Q1 journal | Rolling | Solid Q1, lower bar than TPAMI |

---

## 9. Related Work Map (must-cite papers)

### Knowledge editing for LLMs
- ROME (Meng et al., NeurIPS 2022)
- MEMIT (Meng et al., ICLR 2023)
- AlphaEdit (Fang et al., ICLR 2025)
- GRACE (Hartvigsen et al., NeurIPS 2023)
- WISE (Wang et al., 2024)
- MEMOIR (Wang et al., NeurIPS 2025)
- RILKE (Liu et al., 2025)

### Knowledge editing for VLMs
- MMEdit (Cheng et al., EMNLP 2023) — first VLM editing benchmark
- VLKEB (Huang et al., 2024) — VLM editing benchmark with real images
- VisEdit (Chen et al., 2024) — attribution-based single-shot VLM editing
- LiveEdit (Chen et al., CVPR 2025) — lifelong VLM editing with MoE
- DualEdit (Kim et al., COLM 2025) — modality-specific VLM editing
- KDKE (under review, ICLR 2026) — dynamic module editing for MLLMs

### Long-form / unstructured editing
- AnyEdit (Jiang et al., ICML 2025) — autoregressive chunk-wise editing
- UnKE (Deng et al., ICLR 2025) — unstructured knowledge editing
- AKEW (Wu et al., EMNLP 2024) — long-form editing benchmark

### Surveys
- "A Comprehensive Study of Knowledge Editing for LLMs" (Zhang et al., 2024)
- "A Dual-Axis Taxonomy of Knowledge Editing for LLMs" (2025)
- "Editing Across Languages: A Survey of MKE" (ACL 2025)
