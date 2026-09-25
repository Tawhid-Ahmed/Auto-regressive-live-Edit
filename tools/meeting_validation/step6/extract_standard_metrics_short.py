"""Extract Rel/Gen/Loc (+ FC) from short AAAI grounding mean_results into paper tables."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STEP6 = ROOT / "doc/results/meeting_validation/step6"
EVAL = ROOT / "eval_results/meeting_validation/step6/liveedit/blip2-opt-2.7b"

RUNS = {
    "L0": EVAL / "VLKEB_GROUNDING-aaai_L0_short-2026.07.10-19.25.37/sequential_edit_10/seed_42_mean_results.json",
    "S2": EVAL / "VLKEB_GROUNDING-aaai_S2_short-2026.07.10-19.27.38/sequential_edit_10/seed_42_mean_results.json",
    "DEK": EVAL / "VLKEB_GROUNDING-aaai_DEK_short-2026.07.10-19.30.27/sequential_edit_10/seed_42_mean_results.json",
}


def extract(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    tm = d["total_mean"]
    rd = d.get("routing_diagnostics") or {}
    return {
        "rel": tm["reliability"]["acc"],
        "t_gen": tm["generality"]["text_rephrase"]["acc"],
        "m_gen": tm["generality"]["image_rephrase"]["acc"],
        "t_loc": tm["locality"]["text_loc"]["acc"],
        "m_loc": tm["locality"]["image_loc"]["acc"],
        "swap": tm["generality"]["image_swap"]["acc"],
        "hard": tm["generality"]["hard_negative"]["acc"],
        "modality_consistency": tm["generality"]["modality_consistency"]["acc"],
        "prompt_conflict": tm["generality"]["prompt_conflict"]["acc"],
        "em": tm["reliability"].get("em"),
        "fc": rd.get("image_swap_false_cross_rate"),
        "hard_fc": rd.get("hard_negative_false_cross_rate"),
        "source": str(path.relative_to(ROOT)).replace("\\", "/"),
    }


def pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.1f}\\%"


def main() -> int:
    rows = {k: extract(p) for k, p in RUNS.items()}
    out = {
        "date": "2026-07-19",
        "protocol": {
            "dataset": "data/VLKEB/meeting_validation/step1/grounding/grounding_eval_short.json",
            "dsn": 100,
            "sen": 10,
            "seed": 42,
            "backbone": "BLIP2-OPT-2.7B",
            "note": (
                "Same short AAAI protocol as ablation/competitor tables. "
                "T-Gen=text_rephrase, M-Gen=image_rephrase, T-Loc=text_loc, M-Loc=image_loc "
                "(LiveEdit-style names). Swap/hard = adversarial probes (higher token match "
                "on wrong image = more leakage)."
            ),
        },
        "methods": {
            "L0": {"name": "Lifelong baseline", "row": rows["L0"]},
            "S2": {"name": "AVBT (binding only)", "row": rows["S2"]},
            "DEK": {"name": "Dual-Expert Keys (ours)", "row": rows["DEK"]},
        },
        "deltas_L0_to_DEK": {
            "rel_pp": round((rows["DEK"]["rel"] - rows["L0"]["rel"]) * 100, 2),
            "t_gen_pp": round((rows["DEK"]["t_gen"] - rows["L0"]["t_gen"]) * 100, 2),
            "m_gen_pp": round((rows["DEK"]["m_gen"] - rows["L0"]["m_gen"]) * 100, 2),
            "t_loc_pp": round((rows["DEK"]["t_loc"] - rows["L0"]["t_loc"]) * 100, 2),
            "m_loc_pp": round((rows["DEK"]["m_loc"] - rows["L0"]["m_loc"]) * 100, 2),
            "fc_pp": round((rows["DEK"]["fc"] - rows["L0"]["fc"]) * 100, 2),
        },
        "verdict": (
            "Dual-Expert Keys preserve Rel/T-Gen/M-Gen/T-Loc/M-Loc near the lifelong baseline "
            "while cutting swap false-cross 96%->46%."
        ),
    }

    json_path = STEP6 / "aaai_standard_metrics_short.json"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = f"""# AAAI — Standard Rel/Gen/Loc on short grounding (main protocol)

**Date:** {out['date']}  
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
| Lifelong baseline (L0) | {rows['L0']['rel']*100:.1f}% | {rows['L0']['t_gen']*100:.1f}% | {rows['L0']['m_gen']*100:.1f}% | {rows['L0']['t_loc']*100:.0f}% | {rows['L0']['m_loc']*100:.0f}% | {rows['L0']['fc']*100:.0f}% |
| AVBT only (S2) | {rows['S2']['rel']*100:.1f}% | {rows['S2']['t_gen']*100:.1f}% | {rows['S2']['m_gen']*100:.1f}% | {rows['S2']['t_loc']*100:.0f}% | {rows['S2']['m_loc']*100:.0f}% | {rows['S2']['fc']*100:.0f}% |
| **Dual-Expert Keys (DEK)** | **{rows['DEK']['rel']*100:.1f}%** | **{rows['DEK']['t_gen']*100:.1f}%** | **{rows['DEK']['m_gen']*100:.1f}%** | **{rows['DEK']['t_loc']*100:.0f}%** | **{rows['DEK']['m_loc']*100:.0f}%** | **{rows['DEK']['fc']*100:.0f}%** |

**Δ (DEK − L0):** Rel {out['deltas_L0_to_DEK']['rel_pp']:+.2f} pp · T-Gen {out['deltas_L0_to_DEK']['t_gen_pp']:+.2f} pp · M-Gen {out['deltas_L0_to_DEK']['m_gen_pp']:+.2f} pp · FC {out['deltas_L0_to_DEK']['fc_pp']:+.1f} pp.

## Claim for paper

{out['verdict']}

Adversarial probes (context only; not Rel/Gen/Loc):
| Method | Swap token | Hard-neg token |
|--------|----------:|---------------:|
| L0 | {rows['L0']['swap']*100:.1f}% | {rows['L0']['hard']*100:.1f}% |
| S2 | {rows['S2']['swap']*100:.1f}% | {rows['S2']['hard']*100:.1f}% |
| DEK | {rows['DEK']['swap']*100:.1f}% | {rows['DEK']['hard']*100:.1f}% |
"""
    (STEP6 / "AAAI_STANDARD_METRICS_SHORT.md").write_text(md, encoding="utf-8")
    print("Wrote", json_path)
    print("Wrote", STEP6 / "AAAI_STANDARD_METRICS_SHORT.md")
    for k, r in rows.items():
        print(k, "rel", r["rel"], "t_gen", r["t_gen"], "m_gen", r["m_gen"], "fc", r["fc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
