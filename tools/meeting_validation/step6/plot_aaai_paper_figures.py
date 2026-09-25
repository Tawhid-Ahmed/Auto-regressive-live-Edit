"""Paper-facing AAAI result figures (no internal L0/S2 codes; FPR wording).

Writes PNGs to AuthorKit27/Figures/ and doc/figures/aaai/.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STEP6 = ROOT / "doc/results/meeting_validation/step6"
OUTS = [
    ROOT / "AuthorKit27/Figures",
    ROOT / "doc/figures/aaai",
]


def _load(name: str) -> dict:
    return json.loads((STEP6 / name).read_text(encoding="utf-8"))


def _style():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig, name: str):
    import matplotlib.pyplot as plt

    for out in OUTS:
        out.mkdir(parents=True, exist_ok=True)
        path = out / name
        fig.savefig(path, bbox_inches="tight")
        print("wrote", path)
    plt.close(fig)


def plot_ablation_fpr():
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _load("aaai_ablation_table.json")["short_component_ablation"]
    labels = [
        "Lifelong\nbaseline",
        "Binding\nonly",
        "Dual-Expert\nKeys (ours)",
        "+ Neighbor\nmining",
        "+ Pool\nbuffer",
    ]
    rel = [r["rel"] * 100 for r in rows]
    fpr = [r["fc"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, rel, w, label="Reliability ↑", color="#2f6f4e")
    bars = ax.bar(x + w / 2, fpr, w, label="False-positive retrieval ↓", color="#c2410c")
    # Highlight our method
    bars[2].set_color("#9a3412")
    bars[2].set_edgecolor("#7c2d12")
    bars[2].set_linewidth(1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.set_ylim(0, 110)
    ax.set_title("Component ablation on VLKEB: Dual-Expert Keys cut FPR at Rel parity")
    ax.legend(loc="upper right", frameon=False)
    ax.annotate(
        "96% → 46%",
        xy=(2 + w / 2, 46),
        xytext=(2.55, 62),
        fontsize=10,
        color="#9a3412",
        arrowprops=dict(arrowstyle="->", color="#9a3412"),
    )
    ax.axhline(50, color="#999", lw=0.8, ls="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, "fig_results_ablation.png")


def plot_multiseed_fpr():
    import matplotlib.pyplot as plt
    import numpy as np

    d = _load("aaai_dek_multiseed.json")
    seeds = sorted(d["per_seed"], key=int)
    base_fpr = [d["per_seed"][s]["L0"]["fc"] * 100 for s in seeds]
    ours_fpr = [d["per_seed"][s]["DEK"]["fc"] * 100 for s in seeds]
    base_swap = [d["per_seed"][s]["L0"]["swap"] * 100 for s in seeds]
    ours_swap = [d["per_seed"][s]["DEK"]["swap"] * 100 for s in seeds]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    x = np.arange(len(seeds))

    axes[0].plot(x, base_fpr, "o--", color="#9ca3af", label="Lifelong baseline", lw=1.6)
    axes[0].plot(x, ours_fpr, "o-", color="#c2410c", label="Dual-Expert Keys", lw=2.0)
    axes[0].fill_between(x, ours_fpr, base_fpr, color="#c2410c", alpha=0.12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(seeds)
    axes[0].set_xlabel("Evaluation seed")
    axes[0].set_ylabel("False-positive retrieval (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("FPR across five seeds (−50 pp every seed)")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].plot(x, base_swap, "o--", color="#9ca3af", label="Lifelong baseline", lw=1.6)
    axes[1].plot(x, ours_swap, "o-", color="#1d4ed8", label="Dual-Expert Keys", lw=2.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(seeds)
    axes[1].set_xlabel("Evaluation seed")
    axes[1].set_ylabel("Image-swap accuracy (%)")
    axes[1].set_ylim(35, 55)
    axes[1].set_title(r"Image-swap lift (+7.4 pp, $p=3.4\times10^{-4}$)")
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("Multi-seed VLKEB stability", y=1.02, fontsize=12)
    fig.tight_layout()
    _save(fig, "fig_results_multiseed.png")


def plot_competitors_rel_fpr():
    import matplotlib.pyplot as plt
    import numpy as np

    # Pool editors: Rel + FPR; externals: Rel only (FPR n/a)
    methods = [
        ("Lifelong\nbaseline", 93.5, 96.0, True),
        ("Binding\nonly", 93.8, 96.0, True),
        ("Dual-Expert\nKeys (ours)", 93.7, 46.0, True),
        ("Fine-\ntuning", 73.0, None, False),
        ("MEND-VL", 71.3, None, False),
        ("SERAC-VL", 38.6, None, False),
    ]
    labels = [m[0] for m in methods]
    rel = [m[1] for m in methods]
    fpr = [m[2] if m[2] is not None else 0 for m in methods]
    has_fpr = [m[3] for m in methods]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, rel, w, label="Reliability ↑", color="#2f6f4e")
    fpr_vals = [f if h else np.nan for f, h in zip(fpr, has_fpr)]
    ax.bar(x + w / 2, [0 if np.isnan(v) else v for v in fpr_vals], w,
           label="False-positive retrieval ↓", color="#c2410c")
    for i, h in enumerate(has_fpr):
        if not h:
            ax.text(i + w / 2, 8, "n/a", ha="center", fontsize=8, color="#666")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.set_ylim(0, 110)
    ax.set_title("VLKEB operating point: only Dual-Expert Keys keep high Rel and cut FPR")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    _save(fig, "fig_results_competitors.png")


def plot_evqa_stress():
    import matplotlib.pyplot as plt
    import numpy as np

    # From AAAI_EVQA_STRESS_RESULTS.md multi-seed aggregate
    labels = ["Lifelong\nbaseline", "Binding\nonly", "Dual-Expert\nKeys (ours)"]
    rel = [95.0, 95.2, 95.2]
    fpr_swap = [31.3, 30.3, 1.5]
    fpr_hard = [96.8, 97.0, 2.7]
    err_swap = [38.1, 38.8, 1.1]
    err_hard = [0.6, 0.4, 1.3]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(labels))
    w = 0.28
    ax.bar(x - w, rel, w, label="Reliability ↑", color="#2f6f4e",
           yerr=[0.6, 0.6, 0.5], capsize=3)
    ax.bar(x, fpr_swap, w, label="FPR (image-swap) ↓", color="#ea580c",
           yerr=err_swap, capsize=3)
    ax.bar(x + w, fpr_hard, w, label="FPR (hard-neg.) ↓", color="#9a3412",
           yerr=err_hard, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.set_ylim(0, 120)
    ax.set_title("Cross-dataset stress on EVQA (mean ± std, 5 seeds, 300 edits)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_results_evqa.png")


def main():
    _style()
    plot_ablation_fpr()
    plot_multiseed_fpr()
    plot_competitors_rel_fpr()
    plot_evqa_stress()


if __name__ == "__main__":
    main()
