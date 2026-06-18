# Baseline vs AR — full comparison (VLKEB-long)

**Baseline:** LiveEdit (no AR) — `VLKEB_long_baseline_full-2026.06.04-17.39.13`  
**AR runs:** chunk_size 8 / 16 / 32 / 64 — 2026-06-18 full evals  
**JSON:** `eval_results/comparison/baseline_vs_ar_all_chunks.json`

---

## Main comparison table

| Method | Rel acc | Δ vs baseline | Chunk F1 | Δ vs baseline | Text rephrase | Image rephrase | Locality |
|--------|---------|---------------|----------|---------------|---------------|----------------|----------|
| **Baseline** | **75.40%** | — | **76.28%** | — | **75.21%** | **71.49%** | **100%** |
| AR (cs=8) | 75.94% | +0.54 pp | 76.48% | +0.20 pp | 75.63% | 72.51% | 100% |
| **AR (cs=16)** | **75.94%** | **+0.54 pp** | **76.90%** | **+0.62 pp** | **75.63%** | **72.51%** | **100%** |
| AR (cs=32) | 75.94% | +0.54 pp | 77.30% | +1.02 pp | 75.63% | 72.51% | 100% |
| AR (cs=64) | 75.94% | +0.54 pp | 77.88% | +1.60 pp | 75.63% | 72.51% | 100% |

*pp = percentage points. n = 3150 edits, seed 42.*

---

## Fair chunk-level comparison (cs=16 only)

Baseline and AR both use a **16-token chunk window** for chunk EM/F1 in eval:

| Metric | Baseline | AR (cs=16) | Δ |
|--------|----------|------------|---|
| Token accuracy | 75.40% | 75.94% | **+0.54 pp** |
| Chunk EM | 13.50% | 14.82% | **+1.32 pp** |
| Chunk F1 | 76.28% | 76.90% | **+0.62 pp** |
| 65+ bin acc | 75.40% | 75.95% | **+0.55 pp** |

---

## Results paragraph (paste-ready)

On the full VLKEB-long eval split (3150 sequential edits, seed 42), AR-LiveEdit outperformed standard LiveEdit on token-level reliability regardless of inference chunk size: **75.94% vs 75.40%** (+0.54 pp). Generality improved consistently (+0.42 pp text rephrase, +1.02 pp image rephrase), and locality remained at 100% for both text and image probes. For the fairest chunk-level comparison (16-token eval windows, matching AR training), AR at chunk_size=16 improved chunk EM from 13.5% to 14.8% (+1.3 pp) and chunk F1 from 76.3% to 76.9% (+0.6 pp). Larger inference chunk sizes (32, 64) did not further improve token accuracy but yielded slightly higher chunk F1 scores; we attribute part of that gap to the eval metric using wider chunk windows, not necessarily better editing. **AR with chunk_size=16** is the recommended configuration: it beats baseline on all primary metrics and matches the training setup.

---

## Key takeaways

1. **AR beats baseline on token accuracy** — all four chunk sizes (+0.54 pp).
2. **Generality improves** with AR (+0.4–1.0 pp); **locality unchanged** (100%).
3. **Best apples-to-apples chunk comparison:** AR cs=16 vs baseline (same 16-token metric window).
4. **Chunk size ablation:** inference chunk size does not change token acc among AR runs; cs16 is the thesis default.
5. **Do not over-interpret** chunk F1 gains at cs32/cs64 vs baseline without noting different chunk windows.

---

## Related files

| File | Content |
|------|---------|
| `liveedit_vs_ar_seed42_longform.json` | Baseline vs AR cs=16 (June 4 runs) |
| `ar_chunk_size_sweep.json` | AR-only chunk ablation (June 18 runs) |
| `baseline_vs_ar_all_chunks.json` | This combined comparison |
