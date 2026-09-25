"""Plot Step 2 resolution sweep curves from analyze_resolution_results JSON.

Outputs:
  doc/results/meeting_validation/step2/step2_resolution_curve.png

Usage:
  python tools/meeting_validation/step2/plot_resolution_results.py \\
    --analysis-json doc/results/meeting_validation/step2/step2_resolution_summary.json
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.analyze_grounding_results import GROUNDING_PROBES
from tools.meeting_validation.step2.analyze_resolution_results import discover_mean_results, gate_verdict
from tools.analyze_grounding_results import analyze_mean


def _runs_from_analysis(path: str) -> List[Dict[str, Any]]:
    import json

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["runs"]


def plot_resolution_curve(runs: List[Dict[str, Any]], out_path: str, gate: str = "") -> None:
    runs = sorted(runs, key=lambda r: (r.get("image_size") is None, r.get("image_size") or 0))
    sizes = [r.get("image_size") or 0 for r in runs]
    if not any(sizes):
        sizes = list(range(len(runs)))

    metrics = {
        "Reliability": [(r.get("reliability_acc") or 0) * 100 for r in runs],
        "Image swap": [(r.get("probe_acc", {}).get("image_swap") or 0) * 100 for r in runs],
        "Hard negative": [(r.get("hard_negative_false_activation_rate") or 0) * 100 for r in runs],
        "Modality consistency": [(r.get("modality_consistency_acc") or 0) * 100 for r in runs],
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    colors = ["#2ecc71", "#e74c3c", "#9b59b6", "#3498db"]
    for i, (name, vals) in enumerate(metrics.items()):
        ax.plot(sizes, vals, marker="o", linewidth=2, label=name, color=colors[i % len(colors)])
    ax.set_xlabel("Input resolution (px)")
    ax.set_ylabel("Token accuracy (%)")
    ax.set_title("Step 2 — Grounding metrics vs resolution")
    ax.set_xticks(sizes)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(35, 80)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    swap_gaps = [r.get("swap_gap_pp") or 0 for r in runs]
    ax2.plot(sizes, swap_gaps, marker="s", linewidth=2, color="#e67e22", label="Swap-gap (rel − swap)")
    ax2.axhline(31.15, color="#95a5a6", linestyle="--", linewidth=1, label="Step 1 baseline @224 (~31 pp)")
    ax2.set_xlabel("Input resolution (px)")
    ax2.set_ylabel("Swap-gap (pp)")
    ax2.set_title("Step 2 — Swap-gap vs resolution")
    ax2.set_xticks(sizes)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    if gate:
        fig.suptitle(f"Gate: {gate}", fontsize=9, y=1.02)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot Step 2 resolution sweep figures")
    ap.add_argument("--analysis-json", type=str, default=None)
    ap.add_argument("--eval-dir", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="doc/results/meeting_validation/step2")
    args = ap.parse_args()

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    gate = ""

    if args.analysis_json:
        path = args.analysis_json if os.path.isabs(args.analysis_json) else os.path.join(ROOT, args.analysis_json)
        import json

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        runs = data["runs"]
        gate = data.get("gate", "")
    elif args.eval_dir:
        eval_dir = args.eval_dir if os.path.isabs(args.eval_dir) else os.path.join(ROOT, args.eval_dir)
        paths = discover_mean_results(eval_dir)
        runs = [analyze_mean(p, f"res{i}") for i, p in enumerate(paths)]
        gate = gate_verdict(runs)
    else:
        ap.error("Provide --analysis-json or --eval-dir")

    plot_resolution_curve(runs, os.path.join(out_dir, "step2_resolution_curve.png"), gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
