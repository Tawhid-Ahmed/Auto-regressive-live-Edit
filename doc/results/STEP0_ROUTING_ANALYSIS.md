# Step 0 — Routing Diagnostic Analysis

**Date:** 2026-07-05  
**Protocol:** seed 42, n=150 sequential edits, VLKEB-long (`eval_longform_ollama.json`)  
**Checkpoints:** baseline `epoch-20-i-27800-ema_loss-0.7936`, AR `epoch-20-i-26600-ema_loss-1.5124`  
**Raw logs:** [`routing_diagnostics_baseline.json`](routing_diagnostics_baseline.json), [`routing_diagnostics_ar.json`](routing_diagnostics_ar.json)

---

## Summary

| Metric | Baseline | AR |
|--------|----------|-----|
| Experts retrieved / query (reliability) | 3.8 | 2.6 |
| Only correct expert fires | 18% | 28% |
| Correct + wrong co-fire | 81% | 72% |
| Only wrong experts (reliability) | 0% | 0% |
| Image rephrase miss | 6% | 2.7% |
| Image rephrase only-wrong | 4% | 0.7% |
| Locality false fire | 0% | 0% |

## Interpretation

1. **Correct expert is usually present** (>99% reliability miss baseline 0.7%) but almost always **co-fires with 2–3 wrong experts** from the lifelong pool.
2. **Image rephrase** is the stress probe: 10% clear routing failure (miss + only-wrong), vs text probes ~1%.
3. **Locality routing is clean** — unrelated images never pass the prototype gate (0% false fire), even though output locality is 100%.
4. **AR** reduces co-retrieval (2.6 vs 3.8 experts) and improves clean routing (28% vs 18% only-correct).

## Cross-check (prior full eval, n=3150)

| Probe | Token accuracy |
|-------|----------------|
| Reliability | 75.4% |
| Image rephrase | 71.5% (−4.0 pp) |
| Text rephrase | 75.2% |

Source: `eval_results/liveedit/blip2-opt-2.7b/VLKEB-VLKEB_long_baseline_full-2026.06.04-17.39.13/sequential_edit_50/seed_42_mean_results.json`

## Gate

**PASS → Step 1.** Visual routing bottleneck + image-rephrase failures documented.

## Re-analyze

```powershell
python tools/analyze_routing_diagnostics.py doc/results/routing_diagnostics_baseline.json doc/results/routing_diagnostics_ar.json --labels baseline ar
python tools/plot_routing_diagnostics.py --routing-json doc/results/routing_diagnostics_baseline.json
```

## Figures

| Figure | File |
|--------|------|
| Vision bottleneck | [`results_vision_bottleneck.png`](results_vision_bottleneck.png) |
| Generality gap (full eval) | [`results_generality_routing_gap.png`](results_generality_routing_gap.png) |
| Routing failure rates | [`results_routing_probe_rates.png`](results_routing_probe_rates.png) |
| Routing margin histogram | [`results_routing_sim_margin.png`](results_routing_sim_margin.png) |
| Positional decay (secondary) | [`results_positional_decay.png`](results_positional_decay.png) |

## Narrative memo

[`doc/THESIS_VLLM_EDITING_CHALLENGE.md`](../THESIS_VLLM_EDITING_CHALLENGE.md)
