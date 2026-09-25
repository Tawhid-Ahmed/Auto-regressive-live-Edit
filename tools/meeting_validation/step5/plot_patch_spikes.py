"""Plot Step 5 patch spike comparison figure.

Outputs: doc/results/meeting_validation/step5/step5_spike_comparison.png
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

from tools.meeting_validation.step5.analyze_patch_spikes import SPIKE_LABELS

SPIKE_ORDER = ("baseline", "4a_max", "4a_attn", "4b_rerank", "4c_vision_gate")


def plot_spike_comparison(data: Dict[str, Any], out_path: str) -> None:
    summaries = data.get("summaries", {})
    spikes = [s for s in SPIKE_ORDER if s in summaries]
    if not spikes:
        raise ValueError("No spike summaries in JSON")

    labels = [SPIKE_LABELS.get(s, s) for s in spikes]
    ff = [
        (summaries[s]["summary"].get("hard_negative_false_activation_rate") or 0) * 100
        for s in spikes
    ]
    fc = [
        (summaries[s]["summary"].get("reliability_false_cross_rate") or 0) * 100
        for s in spikes
    ]
    miss = [
        (summaries[s]["summary"].get("reliability_miss_rate") or 0) * 100 for s in spikes
    ]

    x = np.arange(len(spikes))
    colors = ["#8172B3", "#4C72B0", "#64B5CD", "#55A868", "#C44E52"][: len(spikes)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    for ax, vals, title, ylabel in zip(
        axes,
        [ff, fc, miss],
        [
            "Hard-negative false activation",
            "Reliability false-cross edit",
            "Reliability routing miss",
        ],
        ["Rate (%)", "Rate (%)", "Rate (%)"],
    ):
        bars = ax.bar(x, vals, color=colors, alpha=0.9)
        for i, v in enumerate(vals):
            ax.annotate("%.1f%%" % v, (i, v + 0.6), ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [l.replace(" — ", "\n") for l in labels], rotation=0, ha="center", fontsize=8
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, max(vals + [5]) + 10)

    gate = data.get("gate", "")
    fig.suptitle("Step 5 — Patch + cross-attention inference spikes", fontsize=11, y=1.02)
    if gate:
        fig.text(0.5, 1.06, gate[:130], ha="center", fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote %s" % out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot Step 5 spike comparison")
    ap.add_argument(
        "--comparison-json",
        type=str,
        default="doc/results/meeting_validation/step5/patch_spike_comparison.json",
    )
    ap.add_argument("--out-dir", type=str, default="doc/results/meeting_validation/step5")
    args = ap.parse_args()

    json_path = (
        args.comparison_json
        if os.path.isabs(args.comparison_json)
        else os.path.join(ROOT, args.comparison_json)
    )
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    plot_spike_comparison(data, os.path.join(out_dir, "step5_spike_comparison.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
