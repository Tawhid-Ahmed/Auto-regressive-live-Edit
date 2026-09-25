"""Plot Step 4 encoder routing AUC and false-fire comparison.

Outputs: doc/results/meeting_validation/step4/step4_encoder_auc.png
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


ENCODER_LABELS = {
    "qformer": "A: Q-Former (current)",
    "eva_patches": "B: EVA patches",
    "dinov2": "C: DINOv2",
    "siglip": "C: SigLIP",
}


def plot_encoder_auc(data: Dict[str, Any], out_path: str) -> None:
    summaries = data.get("summaries", {})
    encoders = [e for e in ("qformer", "eva_patches", "dinov2", "siglip") if e in summaries]
    if not encoders:
        raise ValueError("No encoder summaries in JSON")

    labels = [ENCODER_LABELS.get(e, e) for e in encoders]
    ff_rates = [(summaries[e].get("hard_neg_false_fire_rate") or 0) * 100 for e in encoders]
    aucs = [summaries[e].get("hard_neg_auc_margin") or summaries[e].get("hard_neg_auc_own_expert") or 0 for e in encoders]
    miss = [(summaries[e].get("reliability_miss_rate") or 0) * 100 for e in encoders]

    x = np.arange(len(encoders))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    colors = ["#8172B3", "#4C72B0", "#55A868", "#C44E52"][: len(encoders)]

    ax = axes[0]
    bars = ax.bar(x, ff_rates, color=colors, alpha=0.9)
    for i, v in enumerate(ff_rates):
        ax.annotate("%.1f%%" % v, (i, v + 0.8), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("A: ", "").replace("B: ", "").replace("C: ", "") for e in labels], rotation=15, ha="right")
    ax.set_ylabel("Hard-negative false-fire rate (%)")
    ax.set_title("False activation on confusable image")
    ax.set_ylim(0, max(ff_rates + [5]) + 8)

    ax2 = axes[1]
    ax2.bar(x, aucs, color=colors, alpha=0.9)
    for i, v in enumerate(aucs):
        ax2.annotate("%.3f" % v, (i, v + 0.02), ha="center", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.split(": ")[-1][:12] for l in labels], rotation=15, ha="right")
    ax2.set_ylabel("AUC (own-expert score)")
    ax2.set_title("Hard-negative discrimination AUC")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0.5, color="#999", linestyle="--", linewidth=0.8)

    ax3 = axes[2]
    ax3.bar(x, miss, color=colors, alpha=0.9)
    for i, v in enumerate(miss):
        ax3.annotate("%.1f%%" % v, (i, v + 0.5), ha="center", fontsize=9)
    ax3.set_xticks(x)
    ax3.set_xticklabels([l.split(": ")[-1][:12] for l in labels], rotation=15, ha="right")
    ax3.set_ylabel("Reliability miss rate (%)")
    ax3.set_title("Routing miss on correct image")
    ax3.set_ylim(0, max(miss + [5]) + 6)

    gate = data.get("gate", "")
    fig.suptitle("Step 4 — Encoder offline routing probe", fontsize=11, y=1.02)
    if gate:
        fig.text(0.5, 1.06, "Gate: %s" % gate[:120], ha="center", fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote %s" % out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot Step 4 encoder AUC figure")
    ap.add_argument(
        "--scores-json",
        type=str,
        default="doc/results/meeting_validation/step4/encoder_routing_scores.json",
    )
    ap.add_argument("--out-dir", type=str, default="doc/results/meeting_validation/step4")
    args = ap.parse_args()

    json_path = args.scores_json if os.path.isabs(args.scores_json) else os.path.join(ROOT, args.scores_json)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(ROOT, args.out_dir)
    plot_encoder_auc(data, os.path.join(out_dir, "step4_encoder_auc.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
