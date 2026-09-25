# Step 6 — Layer ablation (motivation → impact)

| Layer | Component | Motivation | Rel acc | Hard-neg acc | Swap gap (pp) | Locality |
|-------|-----------|------------|--------:|-------------:|--------------:|---------:|
| L0 | Baseline LiveEdit | Reference — Q-Former routing only | 74.7% | 43.4% | 31.2 | 100.0% |
| L1 | PatchGroundingModule | Q-Former bottleneck; hard negatives | 73.6% | 42.9% | 30.7 | 100.0% |
| L2 | Patch routing (no mean-pool) | Scalar key collapse | 73.6% | 45.8% | 28.3 | 100.0% |
| L3 | Cross-attn expert gen | Expert should depend on region | 73.2% | 42.9% | 30.4 | 100.0% |
| L4 | Verification gate | MC-MKE consistency — dual Q-Former + patch thres… | 73.2% | 42.0% | 31.0 | 100.0% |
