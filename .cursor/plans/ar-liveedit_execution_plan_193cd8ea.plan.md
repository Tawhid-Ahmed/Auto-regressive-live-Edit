---
name: AR-LiveEdit Execution Plan
overview: Execute AR-LiveEdit as the primary thesis direction with a staged implementation workflow and publication-grade evaluation pipeline grounded in your current LiveEdit codebase.
todos:
  - id: baseline-lock
    content: Reproduce and freeze LiveEdit baseline metrics on EVQA/EIC/VLKEB
    status: pending
  - id: chunk-datapath
    content: Implement token chunk generation and reconstruction checks in data path
    status: pending
  - id: ar-train-loop
    content: Add autoregressive chunk-step training path in LiveEdit
    status: pending
  - id: ar-infer-loop
    content: Implement chunk-aware inference with mode switch and retrieval tracing
    status: pending
  - id: eval-metrics-extend
    content: Add chunk-level and length-bin metrics to evaluation pipeline
    status: pending
  - id: run-ablation-set
    content: Execute core ablations (chunk size, routing, memory variant) with 3 seeds
    status: pending
  - id: prepare-publication-assets
    content: Generate final tables/plots and thesis-paper narrative from validated runs
    status: pending
isProject: false
---

# AR-LiveEdit Thesis: Concrete Working Procedure and Evaluation Plan

## Objective

Implement and validate **AR-LiveEdit**: integrate AnyEdit-style autoregressive chunked editing into LiveEdit’s multimodal lifelong MoE framework, then demonstrate gains on long-form multimodal editing with controlled side effects.

## Scope Locked (Primary)

- Primary novelty: **Autoregressive chunked multimodal lifelong editing**.
- Secondary (after stable baseline): **serial overwrite robustness**.
- Out of scope for first milestone: region-level scope modeling and safety alignment.

## Codebase Anchors

- Training entry: [E:/MSC/MSc thesis project/LiveEdit/LiveEdit/train_vllm_editor.py](E:/MSC/MSc thesis project/LiveEdit/LiveEdit/train_vllm_editor.py)
- Core editor: [E:/MSC/MSc thesis project/LiveEdit/LiveEdit/editor/vllm_editors/liveedit/liveedit.py](E:/MSC/MSc thesis project/LiveEdit/LiveEdit/editor/vllm_editors/liveedit/liveedit.py)
- Training framework: [E:/MSC/MSc thesis project/LiveEdit/LiveEdit/editor/vllm_editors/base.py](E:/MSC/MSc thesis project/LiveEdit/LiveEdit/editor/vllm_editors/base.py)
- Dataset structure: [E:/MSC/MSc thesis project/LiveEdit/LiveEdit/dataset/vllm.py](E:/MSC/MSc thesis project/LiveEdit/LiveEdit/dataset/vllm.py)
- Evaluation logic: [E:/MSC/MSc thesis project/LiveEdit/LiveEdit/evaluation/vllm_editor_eval.py](E:/MSC/MSc thesis project/LiveEdit/LiveEdit/evaluation/vllm_editor_eval.py)
- Eval runner: [E:/MSC/MSc thesis project/LiveEdit/LiveEdit/test_vllm_edit.py](E:/MSC/MSc thesis project/LiveEdit/LiveEdit/test_vllm_edit.py)

## Working Procedure (Implementation Phases)

### Phase 0 (Day 1-2): Reproducible Baseline Lock

- Reproduce current LiveEdit on EVQA/EIC/VLKEB with fixed seeds.
- Save baseline metrics and logs as immutable references.
- Output: `baseline_metrics.json` per dataset + run notes.

### Phase 1 (Week 1): AR Data and Chunk Pipeline

- Add chunking strategy for `target_new` (token-based fixed-size chunking first).
- Extend dataset preprocessing to generate per-sample chunk sequences.
- Ensure fallback behavior for short targets (single chunk == current behavior).
- Output: unit-level sanity checks for chunk boundaries and reconstruction.

### Phase 2 (Week 2): AR Expert Generation and Training Path

- Modify training flow to process chunk steps autoregressively.
- At each chunk step, generate chunk-conditioned expert residuals.
- Preserve existing losses (reliability/generality/locality/routing) and add chunk-step aggregation.
- Output: trainable AR-LiveEdit checkpoint on small subset.

### Phase 3 (Week 3): AR Inference and Routing

- Add chunk-aware inference loop with MoE retrieval at each step.
- Introduce chunk-time index in routing (simple positional encoding or step-conditioned gate).
- Keep original LiveEdit mode toggle for ablation (`mode = liveedit | ar_liveedit`).
- Output: deterministic inference traces and per-step retrieval logs.

### Phase 4 (Week 4): Evaluation Integration

- Extend evaluation to long-form metrics and chunk-level scoring.
- Add CSV/JSON reporters for per-length-bin performance.
- Add scriptable experiment matrix for 1/10/100/1000 sequential edits.
- Output: first full comparison table (LiveEdit vs AR-LiveEdit).

### Phase 5 (Week 5-6): Ablations + Robustness

- Chunk size ablation: 8/16/32 tokens (or model-token equivalents).
- Routing ablation: no chunk gate vs chunk gate.
- Memory ablation: per-chunk experts vs merged experts.
- Optional start: serial overwrite scenario prototype.
- Output: publishable ablation figures and error analysis.

### Phase 6 (Week 7+): Paper-Ready Hardening

- Runtime/memory profiling, confidence intervals, significance tests.
- Failure taxonomy (hallucination, stale edit retrieval, locality break).
- Prepare thesis chapter + conference/journal draft figures.

## Evaluation Process (Performance Metrics)

### Core Metrics (Must Report)

- **Reliability**: post-edit accuracy on edited target prompts.
- **Generality**: accuracy on text/image rephrase variants.
- **Locality**: preservation on non-target facts (token-level consistency).
- **Edit Time**: per-edit latency.

### AR-Specific Metrics (New)

- **Chunk EM**: exact match per chunk.
- **Chunk F1**: token F1 per chunk and macro average.
- **Long-Form EM/F1 by length bin**: 0-16, 17-32, 33-64, 65-128, 128+ tokens.
- **Step Stability**: performance drop from early chunks to late chunks.
- **Retrieval Precision@k (optional if logged)**: selected experts relevance quality.

### Lifelong/Scalability Metrics

- Sequential edit performance at: **1 / 10 / 100 / 1000** edits.
- **Forgetting Index**: degradation of earlier edits after later edits.
- **Memory Growth**: number of stored experts / storage size over time.

### Secondary (Phase 5+)

- **Overwrite Success Rate** (serial updates to same fact).
- **Recency Correctness**: latest version wins under repeated edits.

## Experimental Matrix

- Datasets: EVQA, EIC, VLKEB.
- Methods:
  - LiveEdit (baseline)
  - AR-LiveEdit (full)
  - AR-LiveEdit w/o chunk routing
  - AR-LiveEdit merged-memory variant
- Length bins: short/medium/long/very-long targets.
- Seeds: at least 3 for stable reporting.

## Success Criteria (Go/No-Go Gates)

- Gate A (Week 2): AR training runs stably without NaNs; loss decreases.
- Gate B (Week 4): AR-LiveEdit improves long-form bin (65+ tokens) by meaningful margin over LiveEdit with <= small locality drop.
- Gate C (Week 6): ablations confirm chunked AR and chunk-aware routing each contribute.
- Gate D (submission-ready): results reproducible across seeds and datasets with clear compute report.

## Immediate Start Checklist (Today)

- Freeze baseline run configs and random seeds.
- Define first chunking policy (fixed token length + overlap 0).
- Create minimal AR mode flag and logging schema.
- Run small-split smoke training and verify end-to-end pipeline.

## Workflow Diagram

```mermaid
flowchart LR
baselineLock --> chunkPipeline
chunkPipeline --> arTraining
arTraining --> arInference
arInference --> evalIntegration
evalIntegration --> ablations
ablations --> paperReady
```



## Notes for Thesis Positioning

- Claim as first practical bridge between **long-form autoregressive editing** and **multimodal lifelong expert routing**.
- Keep novelty evidence centered on long-form bins + sequential edit robustness, not only average scores.

