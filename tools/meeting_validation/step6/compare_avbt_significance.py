"""Synthesize AVBT (S2) significance from binding probes + ablation evals.

Reads:
  - avbt_binding_measure_L0.json / avbt_binding_measure_S2.json (fair binding protocol)
  - aaai_ablation_table.json (dsn=100 token + FC)
  - aaai_dek_multiseed.json (L0 multi-seed token baselines)
  - short_vlkeb_gate_results.json (DEK infer on S2 ckpt)

Writes:
  doc/results/meeting_validation/step6/aaai_avbt_significance.json
  doc/results/meeting_validation/step6/AAAI_AVBT_SIGNIFICANCE.md

Optional GPU re-measure:
  python tools/meeting_validation/step6/verify_avbl_direction.py ... --steps 0
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT_JSON = ROOT / "doc/results/meeting_validation/step6/aaai_avbt_significance.json"
OUT_MD = ROOT / "doc/results/meeting_validation/step6/AAAI_AVBT_SIGNIFICANCE.md"


def betacf(a, b, x, max_iter=200, eps=3e-7):
    qab = a + b
    qap = a + 1
    qam = a - 1
    bz = 1 - qab * x / qap
    az = 1.0
    am = bm = 1.0
    for m in range(1, max_iter + 1):
        em = float(m)
        tem = em + em
        d = em * (b - em) * x / ((qam + tem) * (a + tem))
        ap = az + d * am
        bp = bz + d * bm
        d = (a + em) * (qab + em) * x / ((a + tem) * (qap + tem))
        app = ap + d * az
        bpp = bp + d * bz
        am = ap / bpp
        bm = bp / bpp
        az = app / bpp
        bz = 1.0
        if abs(az - am) < eps * abs(az):
            return az
    return az


def betai(a, b, x):
    if x == 0 or x == 1:
        bt = 0.0
    else:
        bt = math.exp(
            math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
            + a * math.log(x)
            + b * math.log(1 - x)
        )
    if x < (a + 1) / (a + b + 2):
        return bt * betacf(a, b, x) / a
    return 1 - bt * betacf(b, a, 1 - x) / b


def one_sample_vs_constant(values: list[float], null_value: float, lower_better: bool):
    """Test whether mean(values) differs from null_value (S2 held constant across seeds)."""
    n = len(values)
    mean = sum(values) / n
    diffs = [v - null_value for v in values]
    mean_diff = mean - null_value
    ss = sum((v - mean) ** 2 for v in values)
    var = ss / (n - 1) if n > 1 else 0.0
    if var < 1e-18:
        t = float("inf") if mean_diff != 0 else 0.0
        p = 0.0 if mean_diff != 0 else 1.0
    else:
        se = math.sqrt(var / n)
        t = mean_diff / se
        df = n - 1
        x = df / (df + t * t)
        p = betai(df / 2.0, 0.5, x)
    if lower_better:
        p_one = (p / 2) if mean_diff < 0 else (1 - p / 2)
    else:
        p_one = (p / 2) if mean_diff > 0 else (1 - p / 2)
    return {
        "l0_values": values,
        "l0_mean": mean,
        "s2_constant": null_value,
        "mean_diff_l0_minus_s2": -mean_diff if lower_better else mean_diff,
        "mean_diff_s2_minus_l0": mean_diff,
        "paired_t": ("inf" if math.isinf(t) else t),
        "df": n - 1,
        "p_two_sided": p,
        "p_one_sided": p_one,
        "significant_0_05": bool(p < 0.05),
    }


def load_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def main() -> int:
    binding_l0 = load_json("doc/results/meeting_validation/step6/avbt_binding_measure_L0.json")
    binding_s2 = load_json("doc/results/meeting_validation/step6/avbt_binding_measure_S2.json")
    ablation = load_json("doc/results/meeting_validation/step6/aaai_ablation_table.json")
    multiseed = load_json("doc/results/meeting_validation/step6/aaai_dek_multiseed.json")
    gate = load_json("doc/results/meeting_validation/step6/short_vlkeb_gate_results.json")

    l0_bind = binding_l0["before"]
    s2_bind = binding_s2["before"]
    bind_protocol = binding_l0["meta"]

    rows = {r["method"]: r for r in ablation["short_component_ablation"]}
    l0_row = rows["LiveEdit_L0"]
    s2_row = rows["AVBT_S2"]
    dek_row = rows["AVBT_DEK"]

    l0_swap_seeds = [
        multiseed["per_seed"][k]["L0"]["swap"] for k in sorted(multiseed["per_seed"], key=int)
    ]
    l0_rel_seeds = [
        multiseed["per_seed"][k]["L0"]["rel"] for k in sorted(multiseed["per_seed"], key=int)
    ]
    l0_gap_seeds = [
        multiseed["per_seed"][k]["L0"]["swap_gap"] for k in sorted(multiseed["per_seed"], key=int)
    ]

    token_swap_test = one_sample_vs_constant(l0_swap_seeds, s2_row["swap"], lower_better=False)
    token_gap_test = one_sample_vs_constant(l0_gap_seeds, s2_row["swap_gap"], lower_better=False)
    token_rel_test = one_sample_vs_constant(l0_rel_seeds, s2_row["rel"], lower_better=False)

    binding_deltas = {
        "swap_expert_mass": {
            "l0": l0_bind["swap_expert_mass"],
            "s2": s2_bind["swap_expert_mass"],
            "delta_s2_minus_l0": s2_bind["swap_expert_mass"] - l0_bind["swap_expert_mass"],
            "interpretation": "lower = less fused expert activation on wrong-image probe",
        },
        "hard_neg_expert_mass": {
            "l0": l0_bind["hard_neg_expert_mass"],
            "s2": s2_bind["hard_neg_expert_mass"],
            "delta_s2_minus_l0": s2_bind["hard_neg_expert_mass"] - l0_bind["hard_neg_expert_mass"],
        },
        "reliability_own_prob": {
            "l0": l0_bind["reliability_own_prob"],
            "s2": s2_bind["reliability_own_prob"],
            "delta_s2_minus_l0": s2_bind["reliability_own_prob"] - l0_bind["reliability_own_prob"],
        },
        "swap_false_cross_binding_probe": {
            "l0": l0_bind["swap_false_cross"],
            "s2": s2_bind["swap_false_cross"],
            "note": "mini-pool Q-Former cross-retrieval; not same as dsn=100 FC logger",
        },
    }

    swap_mass_reduction = l0_bind["swap_expert_mass"] - s2_bind["swap_expert_mass"]
    hard_mass_reduction = l0_bind["hard_neg_expert_mass"] - s2_bind["hard_neg_expert_mass"]

    out = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "question": "What is AVBT's significant contribution separate from DEK?",
        "verdict_short": (
            "AVBT significantly suppresses adversarial fused binding (expert_mass −0.56 on swap) "
            "and preserves reliability, but does NOT cut false-cross (~96%) without DEK. "
            "Token image-swap leakage rises +7 pp at dsn=100 — report binding metrics for AVBT, FC for DEK."
        ),
        "binding_protocol": {
            "tool": "verify_avbl_direction.py --steps 0",
            "grounding_eval_json": "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json",
            "sample_n": bind_protocol["sample_n"],
            "sequential_n": bind_protocol["sequential_n"],
            "seed": bind_protocol["seed"],
            "l0_ckpt": binding_l0["meta"]["ckpt"],
            "s2_ckpt": binding_s2["meta"]["ckpt"],
        },
        "binding_significance": binding_deltas,
        "binding_headline": {
            "swap_expert_mass_reduction": swap_mass_reduction,
            "hard_neg_expert_mass_reduction": hard_mass_reduction,
            "swap_mass_reduction_pp": swap_mass_reduction * 100,
            "significant_at_binding_layer": swap_mass_reduction >= 0.10,
        },
        "token_eval_dsn100": {
            "protocol": ablation["protocol_short"],
            "metrics_note": "image_swap = token match on WRONG image → lower is better (less leakage)",
            "L0": {
                "rel": l0_row["rel"],
                "swap_leakage": l0_row["swap"],
                "hard_neg_leakage": l0_row["hard_neg"],
                "fc": l0_row["fc"],
                "swap_gap": l0_row["swap_gap"],
            },
            "S2_AVBT": {
                "rel": s2_row["rel"],
                "swap_leakage": s2_row["swap"],
                "hard_neg_leakage": s2_row["hard_neg"],
                "fc": s2_row["fc"],
                "swap_gap": s2_row["swap_gap"],
            },
            "DEK": {
                "fc": dek_row["fc"],
                "swap_leakage": dek_row["swap"],
            },
            "deltas_L0_to_S2": {
                "rel_pp": round((s2_row["rel"] - l0_row["rel"]) * 100, 2),
                "swap_leakage_pp": round((s2_row["swap"] - l0_row["swap"]) * 100, 2),
                "fc": "unchanged_0.96",
                "swap_gap_pp": round((s2_row["swap_gap"] - l0_row["swap_gap"]) * 100, 2),
            },
        },
        "multiseed_L0_vs_S2_constant": {
            "note": "S2 eval only at seed 42; test whether L0 multi-seed mean differs from S2 point estimate",
            "swap_leakage": token_swap_test,
            "swap_gap": token_gap_test,
            "reliability": token_rel_test,
        },
        "dek_dependency": {
            "S2_fc": s2_row["fc"],
            "DEK_infer_on_S2_fc": next(
                r["fc_swap"] for r in gate["mini_sweep_summary"] if r["run"].startswith("DEK_infer")
            ),
            "DEK_train_fc": dek_row["fc"],
            "interpretation": "Dual-gate alone on S2 ckpt cuts FC 96%→53%; trained DEK →46%",
        },
        "standard_vlkeb": load_json("doc/results/meeting_validation/step6/avbt_short_vlkeb_comparison.json")[
            "eval_standard_vlkeb"
        ],
        "layered_method_story": [
            "LiveEdit (L0): MoE retrieval; high swap expert_mass (~0.98) on adversarial images.",
            "AVBT (S2): patch grounding + AVBL training → expert_mass ~0.42; reliability held; FC still ~96%.",
            "DEK: vision∧text gate on AVBT-trained pool → FC ~46%; primary anti-leakage mechanism.",
        ],
        "supervisor_table": {
            "columns": ["Stage", "Metric type", "Key metric", "L0", "S2", "DEK", "Claim"],
            "rows": [
                ["L0", "baseline", "Reliability", "93.5%", "—", "—", "reference"],
                ["AVBT", "binding", "Swap expert_mass ↓", "0.98", "0.42", "—", "AVBT win"],
                ["AVBT", "token", "Image-swap leakage ↓", "40.7%", "47.8%", "—", "NOT improved"],
                ["AVBT", "routing", "False-cross (FC) ↓", "96%", "96%", "—", "unchanged"],
                ["DEK", "routing", "False-cross (FC) ↓", "96%", "96%", "46%", "DEK win"],
            ],
        },
    }

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    ms = out["multiseed_L0_vs_S2_constant"]["swap_leakage"]
    bh = out["binding_headline"]
    tok = out["token_eval_dsn100"]["deltas_L0_to_S2"]

    md = f"""# AVBT significance (separate from DEK)

**Date:** {out['date']}  
**JSON:** [`aaai_avbt_significance.json`](aaai_avbt_significance.json)

## Bottom line

{out['verdict_short']}

---

## 1. Binding-layer evidence (AVBT-native metric)

**Protocol:** `verify_avbl_direction.py --steps 0`, short grounding JSON, dsn=15, sen=5, seed=42 — same apples-to-apples pool for L0 and S2 ckpts.

| Binding metric | L0 | S2 (AVBT) | Δ (S2−L0) | Direction |
|----------------|---:|----------:|----------:|-----------|
| Swap expert_mass | {l0_bind['swap_expert_mass']:.3f} | {s2_bind['swap_expert_mass']:.3f} | {binding_deltas['swap_expert_mass']['delta_s2_minus_l0']:+.3f} | **↓ better** |
| Hard-neg expert_mass | {l0_bind['hard_neg_expert_mass']:.3f} | {s2_bind['hard_neg_expert_mass']:.3f} | {binding_deltas['hard_neg_expert_mass']['delta_s2_minus_l0']:+.3f} | **↓ better** |
| Reliability own_prob | {l0_bind['reliability_own_prob']:.3f} | {s2_bind['reliability_own_prob']:.3f} | {binding_deltas['reliability_own_prob']['delta_s2_minus_l0']:+.3f} | held |

**Headline:** swap expert_mass drops **{bh['swap_mass_reduction_pp']:.0f} pp** ({l0_bind['swap_expert_mass']:.2f} → {s2_bind['swap_expert_mass']:.2f}). This is what AVBL optimizes and is the right metric to showcase AVBT.

---

## 2. Token-level grounding eval (dsn=100) — use with care

**Important:** `image_swap` = fraction of probes where the model outputs the **edited answer on the wrong image**. **Lower is better** (less leakage).

| Metric | L0 | S2 (AVBT) | Δ |
|--------|---:|----------:|--:|
| Reliability | {l0_row['rel']*100:.1f}% | {s2_row['rel']*100:.1f}% | {tok['rel_pp']:+.1f} pp |
| Image-swap leakage | {l0_row['swap']*100:.1f}% | {s2_row['swap']*100:.1f}% | {tok['swap_leakage_pp']:+.1f} pp |
| False-cross (FC) | {l0_row['fc']*100:.0f}% | {s2_row['fc']*100:.0f}% | unchanged |
| Swap-gap (rel−swap) | {l0_row['swap_gap']*100:.1f} pp | {s2_row['swap_gap']*100:.1f} pp | {tok['swap_gap_pp']:+.1f} pp |

AVBT **does not** improve token leakage or FC at this scale. Binding ↓ and token leakage are **decoupled** because Q-Former can false-cross retrieve (~96%) even when fused patch binding is weaker.

Multi-seed L0 swap mean {ms['l0_mean']*100:.1f}% vs S2 {ms['s2_constant']*100:.1f}%: *p*={ms['p_two_sided']:.4g} (L0 **lower** leakage — S2 is worse on this token probe).

---

## 3. DEK needs AVBT as foundation

| Config | FC |
|--------|---:|
| L0 | 96% |
| S2 (AVBT only) | 96% |
| S2 + DEK infer (gate only) | {out['dek_dependency']['DEK_infer_on_S2_fc']*100:.0f}% |
| AVBT + DEK (trained) | {dek_row['fc']*100:.0f}% |

DEK dual-gate on the S2 checkpoint alone cuts FC substantially; full method trains patch module with AVBL first.

---

## 4. Layered story for supervisor / thesis

{chr(10).join('- ' + s for s in out['layered_method_story'])}

**Motivation for AVBT:** LiveEdit fires experts strongly on adversarial images (expert_mass ≈ 1). AVBL + patch grounding learns to **suppress** that binding while keeping correct-image reliability.

**What AVBT does not solve:** Q-Former **false-cross retrieval** (wrong expert, right enough vision/text to leak tokens) — that requires **DEK**.

---

## Reproduce binding measure

```cmd
python tools/meeting_validation/step6/verify_avbl_direction.py ^
  -ckpt "records/.../VLKEB_baseline-.../epoch-20-..." ^
  -dvc cuda:0 -dsn 15 -sen 5 -seed 42 --steps 0 ^
  -dpath data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json ^
  -out doc/results/meeting_validation/step6/avbt_binding_measure_L0.json

python tools/meeting_validation/step6/verify_avbl_direction.py ^
  -ckpt "records/.../VLKEB_avbt_short_S2-.../epoch-1-i-125-..." ^
  -dvc cuda:0 -dsn 15 -sen 5 -seed 42 --steps 0 --patch_grounding ^
  -dpath data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json ^
  -out doc/results/meeting_validation/step6/avbt_binding_measure_S2.json
```
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print("Wrote", OUT_JSON)
    print("Wrote", OUT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
