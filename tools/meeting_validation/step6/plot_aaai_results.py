"""Generate AAAI / thesis result figures from locked JSON tables.

Outputs under doc/results/meeting_validation/step6/figures/:
  aaai_ablation_short.png
  aaai_competitors_short.png
  aaai_multiseed_dek.png
  aaai_long_fair.png
  aaai_results_panel.png

Usage:
  E:\\anaconda\\envs\\liveedit\\python.exe -u tools/meeting_validation/step6/plot_aaai_results.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STEP6 = ROOT / "doc/results/meeting_validation/step6"
OUT = STEP6 / "figures"


def _load(name: str) -> dict:
    return json.loads((STEP6 / name).read_text(encoding="utf-8"))


def _style():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_ablation(ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _load("aaai_ablation_table.json")["short_component_ablation"]
    labels = [
        "LiveEdit\nbaseline (L0)",
        "AVBT (S2)",
        "Dual-Expert\nKeys (DEK)",
        "False-cross\nmining (FX)",
        "Pool buffer\n(PB)",
    ]
    rel = [r["rel"] * 100 for r in rows]
    swap = [r["swap"] * 100 for r in rows]
    fc = [r["fc"] * 100 for r in rows]

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(11, 5.0))
    else:
        fig = ax.figure

    x = np.arange(len(labels))
    w = 0.27
    b1 = ax.bar(x - w, rel, w, label="Reliability (Rel)", color="#5a8f4e")
    b2 = ax.bar(x, swap, w, label="Image-swap (Swap)", color="#2b6cb0")
    b3 = ax.bar(x + w, fc, w, label="False-cross (FC) ↓", color="#c2410c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title("Short VLKEB — component ablation (seed 42)")
    ax.legend(loc="upper right", frameon=False)
    ax.axhline(50, color="#999", lw=0.8, ls="--", alpha=0.6)
    if own:
        fig.tight_layout()
        path = OUT / "aaai_ablation_short.png"
        fig.savefig(path)
        plt.close(fig)
        return path
    return None


def plot_competitors(ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _load("aaai_competitor_short.json")["rows"]
    # Keep display order: LiveEdit family then externals
    order = ["LiveEdit_L0", "AVBT_S2", "AVBT_DEK", "FT_VL", "MEND_VL", "SERAC_VL"]
    by = {r["method"]: r for r in rows}
    labels, rel, swap, gap = [], [], [], []
    for m in order:
        r = by[m]
        labels.append(
            {
                "LiveEdit_L0": "LiveEdit\nbaseline (L0)",
                "AVBT_S2": "AVBT (S2)",
                "AVBT_DEK": "Dual-Expert\nKeys (DEK)",
                "FT_VL": "Fine-tuning\n(FT)",
                "MEND_VL": "MEND-VL\n(MEND)",
                "SERAC_VL": "SERAC-VL\n(SERAC)",
            }[m]
        )
        rel.append(r["rel"] * 100)
        swap.append(r["swap"] * 100)
        gap.append(r["swap_gap"] * 100)

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(11, 5.0))
    else:
        fig = ax.figure

    x = np.arange(len(labels))
    w = 0.27
    ax.bar(x - w, rel, w, label="Reliability (Rel)", color="#5a8f4e")
    ax.bar(x, swap, w, label="Image-swap (Swap)", color="#2b6cb0")
    ax.bar(x + w, gap, w, label="Swap-gap ↓", color="#7c3aed")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.set_ylim(-5, 105)
    ax.set_title("Short VLKEB — competitors (seed 42)")
    ax.legend(loc="upper right", frameon=False)
    ax.annotate(
        "False-cross (FC) n/a for Fine-tuning / MEND-VL / SERAC-VL",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=8,
        color="#666",
    )
    if own:
        fig.tight_layout()
        path = OUT / "aaai_competitors_short.png"
        fig.savefig(path)
        plt.close(fig)
        return path
    return None


def plot_multiseed(ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    d = _load("aaai_dek_multiseed.json")
    seeds = sorted(d["per_seed"], key=int)
    l0_swap = [d["per_seed"][s]["L0"]["swap"] * 100 for s in seeds]
    dek_swap = [d["per_seed"][s]["DEK"]["swap"] * 100 for s in seeds]
    l0_fc = [d["per_seed"][s]["L0"]["fc"] * 100 for s in seeds]
    dek_fc = [d["per_seed"][s]["DEK"]["fc"] * 100 for s in seeds]

    own = ax is None
    if own:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    else:
        # panel mode: ax is a single axes; draw swap only with error bars
        fig = ax.figure
        axes = None

    if axes is not None:
        ax0, ax1 = axes
        x = np.arange(len(seeds))
        ax0.plot(x, l0_swap, "o--", color="#888", label="LiveEdit baseline (L0)", lw=1.5)
        ax0.plot(x, dek_swap, "o-", color="#2b6cb0", label="Dual-Expert Keys (DEK)", lw=1.8)
        ax0.set_xticks(x)
        ax0.set_xticklabels(seeds)
        ax0.set_xlabel("Eval seed")
        ax0.set_ylabel("Image-swap (Swap) %")
        ax0.set_ylim(35, 55)
        ax0.set_title("Image-swap across seeds")
        ax0.legend(frameon=False, fontsize=8)

        ax1.plot(x, l0_fc, "o--", color="#888", label="LiveEdit baseline (L0)", lw=1.5)
        ax1.plot(x, dek_fc, "o-", color="#c2410c", label="Dual-Expert Keys (DEK)", lw=1.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(seeds)
        ax1.set_xlabel("Eval seed")
        ax1.set_ylabel("False-cross (FC) %")
        ax1.set_ylim(0, 105)
        ax1.set_title("False-cross across seeds")
        ax1.legend(frameon=False, fontsize=8)
        fig.suptitle(
            "Multi-seed: Dual-Expert Keys (DEK) vs LiveEdit baseline (L0)", y=1.02
        )
        fig.tight_layout()
        path = OUT / "aaai_multiseed_dek.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    # compact mean±std bars for panel
    import numpy as np

    means = [
        d["L0_summary"]["swap"]["mean"] * 100,
        d["DEK_summary"]["swap"]["mean"] * 100,
        d["L0_summary"]["fc"]["mean"] * 100,
        d["DEK_summary"]["fc"]["mean"] * 100,
    ]
    stds = [
        d["L0_summary"]["swap"]["std"] * 100,
        d["DEK_summary"]["swap"]["std"] * 100,
        d["L0_summary"]["fc"]["std"] * 100,
        d["DEK_summary"]["fc"]["std"] * 100,
    ]
    labels = [
        "Baseline (L0)\nSwap",
        "Dual-Expert (DEK)\nSwap",
        "Baseline (L0)\nFC",
        "Dual-Expert (DEK)\nFC",
    ]
    colors = ["#9ca3af", "#2b6cb0", "#9ca3af", "#c2410c"]
    x = np.arange(4)
    ax.bar(x, means, yerr=stds, color=colors, capsize=4, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title("Multi-seed mean ± std (n=5)")
    return None


def plot_long_fair(ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _load("aaai_long_fair.json")["rows"]
    labels = ["LiveEdit baseline\n(L0) long", "Dual-Expert Keys\n(DEK) long"]
    rel = [r["rel"] * 100 for r in rows]
    swap = [r["swap"] * 100 for r in rows]
    fc = [r["fc_swap"] * 100 for r in rows]

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
    else:
        fig = ax.figure

    x = np.arange(len(labels))
    w = 0.27
    ax.bar(x - w, rel, w, label="Reliability (Rel)", color="#5a8f4e")
    ax.bar(x, swap, w, label="Image-swap (Swap)", color="#2b6cb0")
    ax.bar(x + w, fc, w, label="False-cross (FC) ↓", color="#c2410c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.set_title("Fair long-form (matched training, seed 42)")
    ax.legend(loc="upper right", frameon=False)
    ax.annotate(
        "G3 image-swap≥55%: FAIL   |   False-cross + reliability: PASS",
        xy=(0.5, -0.18),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
        color="#444",
    )
    if own:
        fig.tight_layout()
        path = OUT / "aaai_long_fair.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    return None


def plot_panel():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    # reuse internals with axes
    # Ablation
    rows = _load("aaai_ablation_table.json")["short_component_ablation"]
    import numpy as np

    labels = ["Baseline\n(L0)", "AVBT\n(S2)", "Dual-Expert\n(DEK)", "FC mining\n(FX)", "Pool buf.\n(PB)"]
    x = np.arange(5)
    w = 0.27
    ax = axes[0, 0]
    ax.bar(x - w, [r["rel"] * 100 for r in rows], w, color="#5a8f4e", label="Rel")
    ax.bar(x, [r["swap"] * 100 for r in rows], w, color="#2b6cb0", label="Swap")
    ax.bar(x + w, [r["fc"] * 100 for r in rows], w, color="#c2410c", label="FC↓")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_title("A. Component ablation (short)")
    ax.legend(fontsize=8, frameon=False)

    # Competitors
    crow = _load("aaai_competitor_short.json")["rows"]
    by = {r["method"]: r for r in crow}
    order = ["LiveEdit_L0", "AVBT_DEK", "FT_VL", "MEND_VL", "SERAC_VL"]
    labs = ["Baseline\n(L0)", "Dual-Expert\n(DEK)", "Fine-tune\n(FT)", "MEND-VL", "SERAC-VL"]
    x = np.arange(5)
    ax = axes[0, 1]
    ax.bar(x - w, [by[m]["rel"] * 100 for m in order], w, color="#5a8f4e", label="Rel")
    ax.bar(x, [by[m]["swap"] * 100 for m in order], w, color="#2b6cb0", label="Swap")
    ax.bar(x + w, [by[m]["swap_gap"] * 100 for m in order], w, color="#7c3aed", label="Gap↓")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylim(-5, 105)
    ax.set_title("B. Competitors (short)")
    ax.legend(fontsize=8, frameon=False)

    # Multiseed
    ax = axes[1, 0]
    plot_multiseed(ax=ax)

    # Long fair
    ax = axes[1, 1]
    lrows = _load("aaai_long_fair.json")["rows"]
    x = np.arange(2)
    ax.bar(x - w, [r["rel"] * 100 for r in lrows], w, color="#5a8f4e", label="Rel")
    ax.bar(x, [r["swap"] * 100 for r in lrows], w, color="#2b6cb0", label="Swap")
    ax.bar(x + w, [r["fc_swap"] * 100 for r in lrows], w, color="#c2410c", label="FC↓")
    ax.set_xticks(x)
    ax.set_xticklabels(["LiveEdit baseline\n(L0) long", "Dual-Expert Keys\n(DEK) long"], fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_title("C. Fair long-form")
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle(
        "AVBT + Dual-Expert Keys (DEK) — AAAI / thesis results panel",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout()
    path = OUT / "aaai_results_panel.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_ablation(),
        plot_competitors(),
        plot_multiseed(),
        plot_long_fair(),
        plot_panel(),
    ]
    for p in paths:
        print("Wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
