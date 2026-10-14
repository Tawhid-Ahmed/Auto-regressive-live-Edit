"""Offline prototype/foreign breakdown from routing_health_s2_s3_bs2.json."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "doc/results/cvpr2027/g5/routing_health_s2_s3_bs2.json"
OUT = ROOT / "doc/results/cvpr2027/g5/s3_proto_foreign_breakdown.json"


def analyze(rows: list[dict], name: str) -> dict:
    prot = [float(r["vis_sim_prot"]) for r in rows if r.get("vis_sim_prot") is not None]
    margin = [float(r["own_vis_margin"]) for r in rows if r.get("own_vis_margin") is not None]
    own_vis = [m + pr for m, pr in zip(margin, prot)]
    vok = [int(r["n_vision_ok"]) for r in rows]
    vok_pct = [float(r["vision_ok_pct_of_pool"]) for r in rows]
    foreign_ok = [
        int(r["n_vision_ok"]) - (1 if r.get("own_in_vision_ok") else 0) for r in rows
    ]
    buckets = {}
    for lo, hi, lab in [(-999, -5, "prot_lt_-5"), (-5, 5, "prot_near_0"), (5, 15, "prot_5_15"), (15, 999, "prot_gt_15")]:
        sub = [(pr, m, v) for pr, m, v in zip(prot, margin, vok_pct) if lo <= pr < hi]
        if not sub:
            continue
        buckets[lab] = {
            "n": len(sub),
            "mean_margin": round(float(st.mean(x[1] for x in sub)), 3),
            "mean_vok_pct": round(float(st.mean(x[2] for x in sub)), 2),
        }
    return {
        "n": len(rows),
        "vis_sim_prot": {
            "mean": round(float(st.mean(prot)), 3),
            "median": round(float(st.median(prot)), 3),
            "p10": round(float(sorted(prot)[len(prot) // 10]), 3),
            "p90": round(float(sorted(prot)[9 * len(prot) // 10]), 3),
        },
        "own_vis_sim": {
            "mean": round(float(st.mean(own_vis)), 3),
            "median": round(float(st.median(own_vis)), 3),
        },
        "own_margin": {
            "mean": round(float(st.mean(margin)), 3),
            "neg_pct": round(100.0 * sum(m < 0 for m in margin) / len(margin), 2),
            "pos_pct": round(100.0 * sum(m > 0 for m in margin) / len(margin), 2),
        },
        "vision_ok": {
            "mean_n": round(float(st.mean(vok)), 2),
            "mean_pct": round(float(st.mean(vok_pct)), 2),
            "foreign_mean_n": round(float(st.mean(foreign_ok)), 2),
        },
        "own_retrieved_pct": round(100.0 * sum(bool(r["own_retrieved"]) for r in rows) / len(rows), 2),
        "own_in_vision_ok_pct": round(
            100.0 * sum(bool(r["own_in_vision_ok"]) for r in rows) / len(rows), 2
        ),
        "corr_prot_vok_pct": round(float(np.corrcoef(prot, vok_pct)[0, 1]), 3),
        "corr_own_vis_vok_pct": round(float(np.corrcoef(own_vis, vok_pct)[0, 1]), 3),
        "prot_buckets": buckets,
    }


def main() -> None:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    rel = d["all_reliability_rows"]
    swp = d["all_image_swap_rows"]
    rel_m = {r["sample_id"]: r for r in rel}
    swp_m = {r["sample_id"]: r for r in swp}
    ids = sorted(set(rel_m) & set(swp_m))
    own_hot_both = sum(
        1
        for i in ids
        if rel_m[i]["own_in_vision_ok"] and swp_m[i]["own_in_vision_ok"]
    )
    own_flip_off = sum(
        1
        for i in ids
        if rel_m[i]["own_in_vision_ok"] and not swp_m[i]["own_in_vision_ok"]
    )
    margin_drop = [
        float(swp_m[i]["own_vis_margin"]) - float(rel_m[i]["own_vis_margin"]) for i in ids
    ]
    out = {
        "source": str(SRC.as_posix()),
        "reliability": analyze(rel, "reliability"),
        "image_swap": analyze(swp, "image_swap"),
        "paired": {
            "n": len(ids),
            "own_vision_ok_both_pct": round(100.0 * own_hot_both / len(ids), 2),
            "own_vision_ok_flips_off_under_swap_pct": round(100.0 * own_flip_off / len(ids), 2),
            "mean_margin_drop_swap_minus_rel": round(float(st.mean(margin_drop)), 3),
            "text_argmax_stable_under_swap_pct": d["routing_health"].get(
                "text_argmax_stable_under_swap_pct"
            ),
        },
        "verdict_hints": {
            "prototype_collapsed_if": "rel vis_sim_prot mean near/below 0 while foreign_vision_ok high",
            "foreign_inflation_if": "foreign_mean_n high with healthy prot",
            "own_stays_hot_on_swap_if": "own_vision_ok_both_pct high and mean_margin_drop small",
        },
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
