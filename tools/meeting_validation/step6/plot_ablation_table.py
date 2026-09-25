"""Plot Step 6 layer ablation bar chart.

Usage:
  python tools/meeting_validation/step6/plot_ablation_table.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary-json",
        type=str,
        default="doc/results/meeting_validation/step6/step6_ablation_summary.json",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="doc/results/meeting_validation/step6/step6_layer_ablation.png",
    )
    args = ap.parse_args()

    summary_path = args.summary_json if os.path.isabs(args.summary_json) else os.path.join(ROOT, args.summary_json)
    if not os.path.isfile(summary_path):
        print("ERROR: summary not found:", summary_path)
        return 1

    import matplotlib.pyplot as plt

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    rows = summary.get("layers", [])
    if not rows:
        print("No layer rows to plot")
        return 1

    labels = [r["layer"] for r in rows]
    rel = [(r.get("reliability_acc") or 0) * 100 for r in rows]
    hn = [(r.get("hard_negative_acc") or 0) * 100 for r in rows]

    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - w / 2 for i in x], rel, width=w, label="Reliability acc", color="#5a8f4e")
    ax.bar([i + w / 2 for i in x], hn, width=w, label="Hard-negative acc", color="#c2410c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Token accuracy (%)")
    ax.set_title("Step 6 — Patch-grounding layer ablation")
    ax.legend()
    ax.set_ylim(0, 100)
    fig.tight_layout()
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print("Wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
