# AAAI — Paired significance: Dual-Expert Keys (DEK) vs LiveEdit baseline (L0)

**Date:** 2026-07-13  
**Source:** `aaai_dek_multiseed.json`  
**Test:** paired Student *t* over seeds `{42,123,456,789,1024}` (same ckpts, different eval seeds)  
**Companion:** [`aaai_dek_significance.json`](aaai_dek_significance.json)

| Short | Full name |
|-------|-----------|
| L0 | LiveEdit baseline |
| DEK | AVBT + Dual-Expert Keys |
| Rel | Reliability |
| Swap | Image-swap accuracy |
| FC | False-cross rate ↓ |
| Swap-gap | Reliability − image-swap |

---

## Results

| Metric | LiveEdit baseline (L0) | Dual-Expert Keys (DEK) | Δ (DEK−L0) | *t*(df=4) | *p* (two-sided) | *p* (one-sided) | Sig @0.05 |
|--------|-----------------------:|-----------------------:|-----------:|----------:|----------------:|----------------:|:---------:|
| False-cross (FC) | 0.9600 | 0.4600 | -0.5000 | ∞ | 0 | 0 | YES |
| Image-swap (Swap) | 0.4123 | 0.4862 | +0.0739 | 11.408 | 0.0003368 | 0.0001684 | YES |
| Swap-gap | 0.5224 | 0.4510 | -0.0714 | -11.023 | 0.0003851 | 0.0001925 | YES |
| Reliability (Rel) | 0.9347 | 0.9372 | +0.0025 | ∞ | 0 | 0 | YES |

---

## Interpretation for the paper

- **Image-swap (Swap):** mean +7.39 pp; two-sided *p*=0.0003368 (significant).
- **False-cross (FC):** mean −50.0 pp; identical on every seed → paired *t* degenerate (∞), *p*=0 under this logger.
- **Swap-gap:** mean -7.14 pp; *p*=0.0003851 (significant).
- **Reliability (Rel):** mean +0.25 pp (held; not a win claim).

**Claimable:** multi-seed swap lift and large FC cut vs LiveEdit baseline (L0).  
**Caveat:** *n*=5 seeds; FC has zero within-method seed variance here.
