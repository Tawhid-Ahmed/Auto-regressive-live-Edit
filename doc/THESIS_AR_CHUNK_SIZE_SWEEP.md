# AR chunk size sweep (VLKEB-long)

**Runs:** 2026-06-18 full eval, seed 42, sequential_edit_n=50, n=3150 edits  
**Checkpoint:** `VLKEB_long_ar` epoch-20 (trained with chunk_size=16)  
**Comparison JSON:** `eval_results/comparison/ar_chunk_size_sweep.json`

---

## Table — inference chunk size ablation

| chunk_size | rel acc | chunk EM | chunk F1 | 65+ acc | 65+ chunk EM | text rephrase | image rephrase | locality min | n edits |
|------------|---------|----------|----------|---------|--------------|---------------|----------------|--------------|---------|
| 8 | 75.94% | 26.26% | 76.48% | 75.95% | 26.26% | 75.63% | 72.51% | 100.00% | 3150 |
| **16** | **75.94%** | **14.82%** | **76.90%** | **75.95%** | **14.82%** | **75.63%** | **72.51%** | **100.00%** | **3150** |
| 32 | 75.94% | 8.11% | 77.30% | 75.95% | 8.11% | 75.63% | 72.51% | 100.00% | 3150 |
| 64 | 75.94% | 6.14% | 77.88% | 75.95% | 6.14% | 75.63% | 72.51% | 100.00% | 3150 |

---

## Results paragraph (paste-ready)

We ablated AR inference chunk size (8, 16, 32, 64 tokens) on the full VLKEB-long eval split using the same AR-LiveEdit checkpoint (trained with chunk_size=16). Reliability token accuracy was **identical** at 75.94% for all settings; generality (text rephrase 75.63%, image rephrase 72.51%) and locality (100% on both probes) were also unchanged. Chunk-level exact match **decreases** with larger chunk windows (26.3% at cs8 vs 6.1% at cs64), but this metric is **not directly comparable across chunk sizes** because each run scores EM over windows of that size. Chunk F1 showed a modest upward trend with larger chunks (76.5% → 77.9%). We therefore report **chunk_size=16** as the thesis default because it matches training and achieves the same token-level reliability as other sizes without requiring more sequential edit steps than smaller chunks.

---

## Discussion bullets

1. **No token-level gain from changing inference chunk size** — all four settings reach the same reliability and generality on this benchmark.
2. **Do not rank by chunk EM across sizes** — shorter windows (cs8) artificially inflate chunk exact match.
3. **Chunk F1** rises slightly with larger chunks (77.88% at cs64 vs 76.48% at cs8), but the effect is small (~1.4 pp).
4. **Training–inference alignment:** cs16 matches the training configuration; no evidence that eval-time retuning to cs32/cs64 helps on VLKEB-long.
5. **Efficiency:** larger chunks mean fewer AR edit steps per target (~2 edits at cs64 vs ~11 at cs8 for ~89-token targets), with no measured quality loss here.

---

## Source paths

| chunk_size | Run folder |
|------------|------------|
| 8 | `VLKEB-ar-VLKEB_long_ar_cs8_full-2026.06.18-13.14.37` |
| 16 | `VLKEB-ar-VLKEB_long_ar_cs16_full-2026.06.18-14.22.51` |
| 32 | `VLKEB-ar-VLKEB_long_ar_cs32_full-2026.06.18-14.56.55` |
| 64 | `VLKEB-ar-VLKEB_long_ar_cs64_full-2026.06.18-15.20.16` |
