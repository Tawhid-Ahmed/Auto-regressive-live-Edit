"""Analyze VG-Edit DINO sim logs: separability + abs-threshold sweep.

Usage:
  python tools/meeting_validation/step6/analyze_vg_edit_dino_sims.py \\
    doc/results/meeting_validation/step6/vg_edit_dino_sims_d100_s10.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _probe_rows(data: Dict[str, Any], probe: str) -> List[Dict[str, Any]]:
    return [e for e in data.get("probes", []) if e.get("probe") == probe]


def _threshold_sweep(
    probes: List[Dict[str, Any]],
    thresholds: np.ndarray,
) -> List[Dict[str, Any]]:
    """Simulate abs-only gate on own-expert Q-Former candidates."""
    rows = []
    for tau in thresholds:
        rel_miss = []
        swap_block = []
        hard_block = []
        swap_fire = []
        hard_fire = []
        for e in probes:
            probe = e["probe"]
            own_sims = [
                c["dino_sim"] for c in e.get("dino_candidates", [])
                if c.get("is_own_expert")
            ]
            if probe == "reliability":
                if not own_sims:
                    rel_miss.append(True)
                else:
                    rel_miss.append(max(own_sims) < float(tau))
            elif probe in ("image_swap", "hard_negative"):
                fired = bool(e.get("false_fire_own_expert"))
                if probe == "image_swap":
                    swap_fire.append(fired)
                else:
                    hard_fire.append(fired)
                if fired and own_sims:
                    blocked = max(own_sims) < float(tau)
                else:
                    blocked = not fired
                if probe == "image_swap":
                    swap_block.append(blocked)
                else:
                    hard_block.append(blocked)
        row = {
            "abs_threshold": float(tau),
            "reliability_miss_rate": float(np.mean(rel_miss)) if rel_miss else None,
            "swap_false_fire_rate": float(np.mean(swap_fire)) if swap_fire else None,
            "swap_blocked_rate": float(np.mean(swap_block)) if swap_block else None,
            "hard_neg_false_fire_rate": float(np.mean(hard_fire)) if hard_fire else None,
            "hard_neg_blocked_rate": float(np.mean(hard_block)) if hard_block else None,
        }
        rows.append(row)
    return rows


def _best_threshold(sweep: List[Dict[str, Any]], max_rel_miss: float = 0.05) -> Optional[Dict[str, Any]]:
    feasible = [
        r for r in sweep
        if r["reliability_miss_rate"] is not None and r["reliability_miss_rate"] <= max_rel_miss
    ]
    if not feasible:
        return None
    return max(
        feasible,
        key=lambda r: (r.get("swap_blocked_rate") or 0.0) + (r.get("hard_neg_blocked_rate") or 0.0),
    )


def analyze(path: str, plot_path: Optional[str] = None) -> Dict[str, Any]:
    data = _load(path)
    probes = data.get("probes", [])
    thresholds = np.round(np.arange(0.30, 0.96, 0.05), 2)

    rel_rows = _probe_rows(data, "reliability")
    swap_rows = _probe_rows(data, "image_swap")
    hard_rows = _probe_rows(data, "hard_negative")

    def _vals(rows: List[Dict[str, Any]], key: str) -> np.ndarray:
        return np.array([float(e[key]) for e in rows if e.get(key) is not None], dtype=np.float64)

    rel_own = _vals(rel_rows, "dino_sim_query_vs_own_edit_img")
    swap_own = _vals(swap_rows, "dino_sim_query_vs_own_edit_img")
    hard_own = _vals(hard_rows, "dino_sim_query_vs_own_edit_img")
    rel_margin = _vals(rel_rows, "dino_margin_own_vs_best_wrong")
    swap_margin = _vals(swap_rows, "dino_margin_own_vs_best_wrong")
    hard_margin = _vals(hard_rows, "dino_margin_own_vs_best_wrong")
    swap_wrong_ret = _vals(swap_rows, "retrieved_wrong_dino_max")

    sweep = _threshold_sweep(probes, thresholds)
    best = _best_threshold(sweep)

    out = {
        "input": path,
        "meta": data.get("meta", {}),
        "summary": data.get("summary", {}),
        "routing_leak_profile": {
            "swap_false_cross_rate": data.get("summary", {}).get("swap_false_cross_rate"),
            "swap_false_fire_own_rate": data.get("summary", {}).get("swap_false_fire_rate"),
            "interpretation": (
                "Token swap-gap is driven mainly by false_cross (wrong expert retrieval), "
                "not own-expert false-fire on Q-Former routing."
            ),
        },
        "separability": {
            "reliability_own_sim": _array_stats(rel_own),
            "swap_own_sim": _array_stats(swap_own),
            "hard_neg_own_sim": _array_stats(hard_own),
            "swap_minus_rel_mean_gap": (
                float(swap_own.mean() - rel_own.mean())
                if len(swap_own) and len(rel_own) else None
            ),
            "reliability_margin_own_vs_best_wrong": _array_stats(rel_margin),
            "swap_margin_own_vs_best_wrong": _array_stats(swap_margin),
            "hard_neg_margin_own_vs_best_wrong": _array_stats(hard_margin),
            "swap_retrieved_wrong_dino_max": _array_stats(swap_wrong_ret),
        },
        "threshold_sweep": sweep,
        "best_threshold_max_5pct_rel_miss": best,
    }

    if plot_path:
        _plot_histograms(rel_own, swap_own, hard_own, sweep, plot_path)
        out["plot_path"] = plot_path

    return out


def _array_stats(a: np.ndarray) -> Optional[Dict[str, float]]:
    if len(a) == 0:
        return None
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _plot_histograms(
    rel_own: np.ndarray,
    swap_own: np.ndarray,
    hard_own: np.ndarray,
    sweep: List[Dict[str, Any]],
    plot_path: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    bins = np.linspace(0.25, 1.0, 31)
    if len(rel_own):
        ax.hist(rel_own, bins=bins, alpha=0.6, label="reliability (own)", density=True)
    if len(swap_own):
        ax.hist(swap_own, bins=bins, alpha=0.6, label="image_swap (own)", density=True)
    if len(hard_own):
        ax.hist(hard_own, bins=bins, alpha=0.6, label="hard_neg (own)", density=True)
    ax.set_xlabel("DINO sim(query, own expert edit img)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title("Own-expert DINO sim by probe")

    ax2 = axes[1]
    taus = [r["abs_threshold"] for r in sweep]
    ax2.plot(taus, [r["reliability_miss_rate"] for r in sweep], label="rel miss", marker="o", ms=3)
    ax2.plot(taus, [r["swap_blocked_rate"] for r in sweep], label="swap blocked", marker="o", ms=3)
    ax2.plot(taus, [r["hard_neg_blocked_rate"] for r in sweep], label="hard-neg blocked", marker="o", ms=3)
    ax2.set_xlabel("abs threshold τ")
    ax2.set_ylabel("rate")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(fontsize=8)
    ax2.set_title("Threshold sweep (abs-only)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze VG-Edit DINO sim logs")
    ap.add_argument("dino_sims_json", type=str)
    ap.add_argument(
        "--plot", type=str,
        default="doc/results/meeting_validation/step6/vg_edit_dino_sims_hist.png",
    )
    ap.add_argument("-out", "--output_json", type=str, default=None)
    args = ap.parse_args()

    plot_path = args.plot
    if plot_path and not os.path.isabs(plot_path):
        plot_path = os.path.join(ROOT, plot_path)

    result = analyze(args.dino_sims_json, plot_path=plot_path)
    out_path = args.output_json or args.dino_sims_json.replace(".json", "_analysis.json")
    if not os.path.isabs(out_path):
        out_path = os.path.join(ROOT, out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Wrote %s" % out_path)
    if plot_path:
        print("Plot: %s" % plot_path)
    sep = result["separability"]
    print("\n=== Separability (DINO sim: query vs own expert edit image) ===")
    for k in ("reliability_own_sim", "swap_own_sim", "hard_neg_own_sim"):
        print("  %s: %s" % (k, sep.get(k)))
    print("  swap_minus_rel_mean_gap: %.4f" % (sep.get("swap_minus_rel_mean_gap") or 0))
    print("\n=== Margin own vs best wrong (full pool) ===")
    for k in ("reliability_margin_own_vs_best_wrong", "swap_margin_own_vs_best_wrong", "swap_retrieved_wrong_dino_max"):
        print("  %s: %s" % (k, sep.get(k)))
    leak = result.get("routing_leak_profile", {})
    print("\n=== Routing leak profile ===")
    print("  swap_false_cross_rate: %s" % leak.get("swap_false_cross_rate"))
    print("  swap_false_fire_own_rate: %s" % leak.get("swap_false_fire_own_rate"))
    best = result.get("best_threshold_max_5pct_rel_miss")
    print("\n=== Best τ (rel miss <= 5%%) ===")
    print(json.dumps(best, indent=2) if best else "  None feasible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
