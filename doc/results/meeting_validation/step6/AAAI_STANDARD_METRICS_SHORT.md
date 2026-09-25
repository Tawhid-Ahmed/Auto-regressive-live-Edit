# AAAI — Standard Rel/Gen/Loc on short grounding (main protocol)

**Date:** 2026-07-19  
**Protocol:** `grounding_eval_short.json`, dsn=100, sen=10, seed=42, BLIP2-OPT-2.7B  
**JSON:** [`aaai_standard_metrics_short.json`](aaai_standard_metrics_short.json)

Name mapping (LiveEdit / DualEdit style):
| Paper name | Field in eval |
|------------|---------------|
| Rel | reliability.acc |
| T-Gen | text_rephrase |
| M-Gen | image_rephrase |
| T-Loc | text_loc |
| M-Loc | image_loc |
| FC | image_swap_false_cross_rate |

## Table (parity + novelty)

| Method | Rel↑ | T-Gen↑ | M-Gen↑ | T-Loc↑ | M-Loc↑ | FC↓ |
|--------|-----:|-------:|-------:|-------:|-------:|----:|
| Lifelong baseline (L0) | 93.5% | 94.3% | 92.1% | 100% | 100% | 96% |
| AVBT only (S2) | 93.8% | 94.5% | 93.8% | 100% | 100% | 96% |
| **Dual-Expert Keys (DEK)** | **93.7%** | **94.5%** | **92.0%** | **100%** | **100%** | **46%** |

**Δ (DEK − L0):** Rel +0.25 pp · T-Gen +0.17 pp · M-Gen -0.16 pp · FC -50.0 pp.

## Claim for paper

Dual-Expert Keys preserve Rel/T-Gen/M-Gen/T-Loc/M-Loc near the lifelong baseline while cutting swap false-cross 96%->46%.

Adversarial probes (context only; not Rel/Gen/Loc):
| Method | Swap token | Hard-neg token |
|--------|----------:|---------------:|
| L0 | 40.7% | 42.2% |
| S2 | 47.8% | 42.6% |
| DEK | 47.7% | 42.0% |
