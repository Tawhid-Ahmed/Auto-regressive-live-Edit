# Stress run status — EVQA grounding

**Updated:** 2026-07-25  
**Pilot + must-do multi-seed:** COMPLETE · Gate **PASS**  
**Next:** LLaVA Phase 2 → [`STATUS_LLAVA.md`](STATUS_LLAVA.md) · [`AAAI_LLAVA_PHASE2_PLAN.md`](AAAI_LLAVA_PHASE2_PLAN.md)

## Docs for paper

- **Write-up:** [`AAAI_EVQA_STRESS_RESULTS.md`](AAAI_EVQA_STRESS_RESULTS.md)
- **Linked from:** [`../AAAI_RESULTS_ANALYSIS.md`](../AAAI_RESULTS_ANALYSIS.md) (§1 takeaway #6, §9)
- **Pilot JSON:** [`aaai_stress_evqa_grounding_summary.json`](aaai_stress_evqa_grounding_summary.json)
- **Multi-seed JSON:** [`aaai_stress_evqa_grounding_multiseed.json`](aaai_stress_evqa_grounding_multiseed.json)

## Headline (multi-seed, dsn=300, n=5)

| | Rel (mean±std) | FC swap (mean±std) |
|--|----------------:|-------------------:|
| L0 | 0.950±0.006 | 0.313±0.381 |
| S2 | 0.952±0.006 | 0.303±0.388 |
| **DEK** | **0.952±0.005** | **0.015±0.011** |
