"""Summarize pre-M diag: L0 health vs S3 S2 + extractor energy."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G5 = ROOT / "doc/results/cvpr2027/g5"


def _keys(d, ks):
    return {k: d.get(k) for k in ks}


def _energy_line(label, block):
    rows = []
    for k, v in (block or {}).items():
        if isinstance(v, dict) and "top1_share" in v:
            top = [(c["channel"], round(c["share_of_l1"], 4)) for c in (v.get("top_channels") or [])[:3]]
            rows.append(
                {
                    "tensor": k,
                    "top1_share": v["top1_share"],
                    "top8_share": v.get("top8_share"),
                    "dominated": v.get("dominated"),
                    "top3": top,
                }
            )
    return {"ckpt": label, "tensors": rows}


def main() -> None:
    l0 = json.loads((G5 / "routing_health_l0_s3diag.json").read_text(encoding="utf-8"))
    s2e = json.loads((G5 / "extractor_energy_s2_s3.json").read_text(encoding="utf-8"))
    s2h = json.loads((G5 / "routing_health_s2_s3_bs2.json").read_text(encoding="utf-8"))
    proto = json.loads((G5 / "s3_proto_foreign_breakdown.json").read_text(encoding="utf-8"))

    metric_ks = [
        "mean_vision_ok_pct_of_pool",
        "mean_own_vis_margin",
        "text_top1_is_own_pct",
        "own_retrieved_pct",
        "mean_n_vision_ok",
        "own_vis_margin_neg_pct",
    ]
    out = {
        "l0_health_gate": l0.get("health_gate"),
        "comparison": {
            "L0_reliability": _keys(l0["routing_health"]["reliability"], metric_ks),
            "L0_image_swap": _keys(l0["routing_health"]["image_swap"], metric_ks),
            "S2_reliability": _keys(s2h["routing_health"]["reliability"], metric_ks),
            "S2_image_swap": _keys(s2h["routing_health"]["image_swap"], metric_ks),
            "text_argmax_stable": {
                "L0": l0["routing_health"].get("text_argmax_stable_under_swap_pct"),
                "S2": s2h["routing_health"].get("text_argmax_stable_under_swap_pct"),
            },
        },
        "extractor_energy": {
            "L0": _energy_line("L0", l0.get("extractor_energy")),
            "S2": _energy_line("S2", s2e.get("extractor_energy")),
        },
        "proto_foreign": {
            "reliability_foreign_mean_n": proto["reliability"]["vision_ok"]["foreign_mean_n"],
            "reliability_prot_mean": proto["reliability"]["vis_sim_prot"]["mean"],
            "swap_own_vision_ok_both_pct": proto["paired"]["own_vision_ok_both_pct"],
            "swap_mean_margin_drop": proto["paired"]["mean_margin_drop_swap_minus_rel"],
        },
        "verdict": {
            "M10_extractor_dominated": False,
            "M10_note": "Extractor + layer-21 top1 L1 shares are all <1%; channel 1512 is top hidden but not mass-dominant. M10 not justified by energy.",
            "L0_vs_S2": "L0 health PASS; S2 FAIL — S2 training re-opens the vision gate (foreign inflation), not residual L0 weakness.",
            "prefer_next": "M8 (more negatives) and/or S2 objective tuning; not M10. M9 still optional for text stability ~50%.",
        },
    }
    path = G5 / "s3_pre_m_diag_summary.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
