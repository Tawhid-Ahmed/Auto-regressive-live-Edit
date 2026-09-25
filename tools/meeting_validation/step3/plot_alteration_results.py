"""Plot Step 3 image-alteration probe results.

Outputs: doc/results/meeting_validation/step3/step3_alteration_robustness.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset.image_alterations import ALTERATION_LABELS
from tools.meeting_validation.step3.analyze_alteration_results import (
    ALTERATION_PROBES,
    analyze_mean,
)


def plot_alteration_robustness(runs: List[Dict[str, Any]], out_path: str, gate: str = "") -> None:
    run = runs[0]
    rel = (run.get("reliability_acc") or 0) * 100
    img_rep = (run.get("image_rephrase_acc") or 0) * 100

    labels = []
    vals = []
    gaps = []
    for probe in ALTERATION_PROBES:
        alt_key = probe.replace("alter_", "", 1)
        labels.append(ALTERATION_LABELS.get(alt_key, probe))
        vals.append((run.get("probe_acc", {}).get(probe) or 0) * 100)
        gaps.append(run.get("alteration_gap_pp", {}).get(probe) or 0)

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.bar(x, vals, color="#3498db", alpha=0.9, label="Alteration probes")
    ax.axhline(rel, color="#2ecc71", linestyle="-", linewidth=2, label="Reliability (original)")
    ax.axhline(img_rep, color="#9b59b6", linestyle="--", linewidth=1.5, label="Image rephrase")
    for i, v in enumerate(vals):
        ax.annotate("%.1f%%" % v, (i, v + 1), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Token accuracy (%)")
    ax.set_title("Accuracy by alteration type")
    ax.set_ylim(0, max(rel + 5, max(vals) + 8, img_rep + 5))
    ax.legend(loc="upper right", fontsize=7)

    ax2 = axes[1]
    colors = ["#e74c3c" if g > 0 else "#95a5a6" for g in gaps]
    ax2.bar(x, gaps, color=colors, alpha=0.9)
    ax2.axhline(0, color="#333", linewidth=0.8)
    for i, g in enumerate(gaps):
        ax2.annotate("%.1f pp" % g, (i, g + 0.15 if g >= 0 else g - 0.6), ha="center", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Gap vs reliability (pp)")
    ax2.set_title("Individual alteration effect (rel - probe)")

    fig.suptitle("Step 3 - Same-image appearance change", fontsize=11, y=1.02)
    if gate:
        fig.text(0.5, 1.06, "Gate: %s" % gate, ha="center", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote %s" % out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot Step 3 alteration figures")
    ap.add_argument("--analysis-json", type=str, default=None)
    ap.add_argument("--mean-results", type=str, default=None)
    ap.add_argument("--label", type=str, default="baseline")
    ap.add_argument("--out-dir", type=str, default="doc/results/meeting_validation/step3")
    args = ap.parse_args()

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    gate = ""

    if args.analysis_json:
        path = args.analysis_json if os.path.isabs(args.analysis_json) else os.path.join(ROOT, args.analysis_json)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        runs = data["runs"]
        gate = data.get("gate", "")
    elif args.mean_results:
        mp = args.mean_results if os.path.isabs(args.mean_results) else os.path.join(ROOT, args.mean_results)
        runs = [analyze_mean(mp, args.label)]
    else:
        ap.error("Provide --analysis-json or --mean-results")

    plot_alteration_robustness(
        runs, os.path.join(out_dir, "step3_alteration_robustness.png"), gate
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
