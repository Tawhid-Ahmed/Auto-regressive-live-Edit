# AR-LiveEdit Execution Plan

Version: `v1.0.0`  
Date: `2026-02-15`  
Scope: General AR-LiveEdit roadmap (multi-dataset, multi-model ready)

## Objective

Implement and validate **AR-LiveEdit**: integrate AnyEdit-style autoregressive chunked editing into LiveEdit's multimodal lifelong MoE framework, then demonstrate gains on long-form multimodal editing with controlled side effects.

## Scope Locked (Primary)

- Primary novelty: **Autoregressive chunked multimodal lifelong editing**.
- Secondary (after stable baseline): **serial overwrite robustness**.
- Out of scope for first milestone: region-level scope modeling and safety alignment.

## Codebase Anchors

- Training entry: `train_vllm_editor.py`
- Core editor: `editor/vllm_editors/liveedit/liveedit.py`
- Training framework: `editor/vllm_editors/base.py`
- Dataset structure: `dataset/vllm.py`
- Evaluation logic: `evaluation/vllm_editor_eval.py`
- Eval runner: `test_vllm_edit.py`

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

### Core Metrics

- Reliability
- Generality
- Locality
- Edit Time

### AR-Specific Metrics

- Chunk EM
- Chunk F1
- Long-Form EM/F1 by length bin: 0-16, 17-32, 33-64, 65-128, 128+ tokens
- Step Stability
- Retrieval Precision@k (optional)

### Lifelong/Scalability Metrics

- Sequential edit performance: 1/10/100/1000 edits
- Forgetting Index
- Memory Growth

### Secondary Metrics

- Overwrite Success Rate
- Recency Correctness

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

