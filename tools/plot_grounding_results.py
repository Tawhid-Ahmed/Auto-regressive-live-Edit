"""Plot Step 1 grounding probe metrics from analyze_grounding_results JSON or mean_results files.

Outputs:
  doc/results/step1_swap_gap.png
  doc/results/step1_probe_rates.png

Usage:
  python tools/plot_grounding_results.py \\
    --mean-results eval_results/.../seed_42_mean_results.json \\
    --label baseline

  python tools/plot_grounding_results.py \\
    --analysis-json doc/results/step1_grounding_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools.analyze_grounding_results import GROUNDING_PROBES, analyze_mean


def _runs_from_analysis(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["runs"]


def plot_swap_gap(runs: List[Dict[str, Any]], out_path: str) -> None:
    labels = [r["label"] for r in runs]
    x = np.arange(len(labels))
    width = 0.35

    rel = [r.get("reliability_acc", 0) or 0 for r in runs]
    swap = [r.get("probe_acc", {}).get("image_swap", 0) or 0 for r in runs]
    gaps = [r.get("swap_gap_pp", 0) or 0 for r in runs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, [v * 100 for v in rel], width, label="Reliability", color="#2ecc71")
    ax.bar(x + width / 2, [v * 100 for v in swap], width, label="Image swap", color="#e74c3c")
    for i, g in enumerate(gaps):
        ax.annotate(f"Δ {g:.1f} pp", (i, max(rel[i], swap[i]) * 100 + 2),
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Token accuracy (%)")
    ax.set_title("Step 1 — Counterfactual image swap gap")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_probe_rates(runs: List[Dict[str, Any]], out_path: str) -> None:
    probe_labels = {
        "image_swap": "Image swap",
        "hard_negative": "Hard negative",
        "modality_consistency": "Modality consistency",
        "prompt_conflict": "Prompt conflict",
        "text_rephrase": "Text rephrase",
        "image_rephrase": "Image rephrase",
    }
    probes = list(GROUNDING_PROBES) + ["text_rephrase", "image_rephrase"]
    run_labels = [r["label"] for r in runs]
    n_runs = len(runs)
    n_probes = len(probes)
    x = np.arange(n_probes)
    width = 0.8 / max(n_runs, 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#3498db", "#9b59b6", "#1abc9c", "#e67e22"]
    for ri, run in enumerate(runs):
        vals = []
        for probe in probes:
            if probe in GROUNDING_PROBES:
                v = run.get("probe_acc", {}).get(probe)
            else:
                v = None
                mean_path = run.get("path")
                if mean_path and os.path.isfile(mean_path):
                    with open(mean_path, encoding="utf-8") as f:
                        mean = json.load(f)
                    tm = mean.get("total_mean", mean)
                    v = tm.get("generality", {}).get(probe, {}).get("acc")
            vals.append((v or 0) * 100)
        offset = (ri - (n_runs - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=run["label"],
               color=colors[ri % len(colors)], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([probe_labels.get(p, p) for p in probes], rotation=15, ha="right")
    ax.set_ylabel("Token accuracy (%)")
    ax.set_title("Step 1 — Grounding probe suite")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot Step 1 grounding figures")
    ap.add_argument("--analysis-json", type=str, default=None)
    ap.add_argument("--mean-results", type=str, action="append", default=[])
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--out-dir", type=str, default="doc/results")
    ap.add_argument("--prefix", type=str, default="step1",
                    help="Filename prefix, e.g. step1_short → step1_short_swap_gap.png")
    args = ap.parse_args()

    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(ROOT, out_dir)

    if args.analysis_json:
        path = args.analysis_json
        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)
        runs = _runs_from_analysis(path)
    elif args.mean_results:
        labels = args.labels or [f"run{i}" for i in range(len(args.mean_results))]
        runs = []
        for p, label in zip(args.mean_results, labels):
            mp = p if os.path.isabs(p) else os.path.join(ROOT, p)
            runs.append(analyze_mean(mp, label))
    else:
        ap.error("Provide --analysis-json or --mean-results")

    prefix = args.prefix.rstrip("_")
    plot_swap_gap(runs, os.path.join(out_dir, f"{prefix}_swap_gap.png"))
    plot_probe_rates(runs, os.path.join(out_dir, f"{prefix}_probe_rates.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
