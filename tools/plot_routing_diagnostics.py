"""Step 0: plot routing-diagnostic figures for the VLLM editing challenge memo.

Produces up to three PNGs under doc/results/:
  1. results_vision_bottleneck.png — patch → Q-Former → mean-pooled routing key
  2. results_routing_probe_rates.png — false-fire / miss rates from routing JSON
  3. (optional) copies positional decay if --positional-decay path given

Usage:
  python tools/plot_routing_diagnostics.py --architecture-only
  python tools/plot_routing_diagnostics.py --routing-json doc/results/routing_diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def plot_vision_bottleneck(arch: dict, out_path: str) -> None:
    """Sankey-style diagram of visual compression in BLIP2 + LiveEdit routing."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.3, 2.0, 2.0, 1.4, f"EVA-ViT patches\n~{arch.get('vit_patch_tokens', 257)} tokens",
         "#4C72B0"),
        (3.0, 2.0, 2.0, 1.4, f"Q-Former output\n{arch.get('qformer_query_tokens', 32)} tokens",
         "#55A868"),
        (5.7, 2.0, 2.0, 1.4,
         f"QVExtractor\n{arch.get('editor_routing_tokens', 32)} keys",
         "#C44E52"),
        (8.4, 2.0, 1.5, 1.4, "mean(2)\n→ scalar\nvis_sim", "#8172B3"),
    ]
    for x, y, w, h, label, color in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                              linewidth=1.5, edgecolor="black", facecolor=color, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")

    for x1, x2 in [(2.3, 3.0), (5.0, 5.7), (7.7, 8.4)]:
        ax.annotate("", xy=(x2, 2.7), xytext=(x1, 2.7),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#333333"))

    ax.text(5.0, 0.55,
            f"Compression: {arch.get('vit_patch_tokens', 257)} patches → "
            f"{arch.get('qformer_query_tokens', 32)} LLM tokens → "
            f"1 scalar routing score  "
            f"(ratio {arch.get('compression_ratio_patches_to_routing', '?')}×)",
            ha="center", fontsize=11, style="italic")
    ax.text(5.0, 4.5, "VLLM editing routes in a bottlenecked visual space",
            ha="center", fontsize=13, fontweight="bold")

    discarded = FancyBboxPatch((3.0, 0.9), 2.0, 0.55, boxstyle="round,pad=0.02",
                               linewidth=1, edgecolor="#999", facecolor="#EEEEEE", linestyle="--")
    ax.add_patch(discarded)
    ax.text(4.0, 1.17, "image_embeds discarded\n(not seen by editor)", ha="center", va="center",
            fontsize=8, color="#555")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_path}")


def plot_routing_probe_rates(summary: dict, out_path: str, title_suffix: str = "") -> None:
    """Bar chart of routing failure modes by probe type."""
    metrics = [
        ("Reliability miss", summary.get("reliability_miss_rate")),
        ("Reliability\nfalse cross-edit", summary.get("reliability_false_cross_rate")),
        ("Image rephrase\nmiss", summary.get("image_rephrase_miss_rate")),
        ("Image rephrase\nfalse cross-edit", summary.get("image_rephrase_false_cross_rate")),
        ("Locality\nfalse fire", summary.get("locality_false_fire_rate")),
    ]
    labels, values = zip(*[(l, v * 100 if v is not None else 0) for l, v in metrics])
    colors = ["#C44E52" if v > 0 else "#55A868" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"LiveEdit routing failure modes{title_suffix}")
    ax.set_ylim(0, max(max(values) * 1.25, 5))
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


def plot_sim_margin(probes: list, out_path: str) -> None:
    """Histogram of vis_sim − prototype margin for image_rephrase vs reliability."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for probe, color in [("reliability", "#4C72B0"), ("image_rephrase", "#C44E52")]:
        margins = []
        for p in probes:
            if p.get("probe") != probe:
                continue
            sims = p.get("vis_sim") or []
            prot = p.get("vis_sim_prot", 0.0)
            if sims:
                margins.append(max(sims) - prot)
        if margins:
            ax.hist(margins, bins=20, alpha=0.55, label=probe.replace("_", " "), color=color)
    ax.set_xlabel("max(vis_sim) − prototype threshold")
    ax.set_ylabel("Count")
    ax.set_title("Routing margin distribution (higher = more experts fire)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


def plot_generality_routing_gap(out_path: str) -> None:
    """Proxy for visual-routing stress: reliability vs image-rephrase gap from VLKEB-long eval."""
    baseline = os.path.join(
        ROOT,
        "eval_results/liveedit/blip2-opt-2.7b/"
        "VLKEB-VLKEB_long_baseline_full-2026.06.04-17.39.13/"
        "sequential_edit_50/seed_42_mean_results.json",
    )
    if not os.path.isfile(baseline):
        return
    with open(baseline, "r", encoding="utf-8") as f:
        data = json.load(f)
    tm = data["total_mean"]
    metrics = [
        ("Reliability\n(same image)", tm["reliability"]["acc"] * 100, "#4C72B0"),
        ("Text rephrase", tm["generality"]["text_rephrase"]["acc"] * 100, "#55A868"),
        ("Image rephrase\n(routing stress)", tm["generality"]["image_rephrase"]["acc"] * 100, "#C44E52"),
    ]
    labels, vals, colors = zip(*metrics)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Token accuracy (%)")
    ax.set_title("Visual routing stress: image-rephrase gap (VLKEB-long baseline)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    gap = metrics[0][1] - metrics[2][1]
    ax.annotate(f"Δ = {gap:.1f} pp", xy=(2, metrics[2][1]), xytext=(2.3, metrics[2][1] + 8),
                arrowprops=dict(arrowstyle="->", color="#333"), fontsize=10)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--architecture-only", action="store_true")
    ap.add_argument("--routing-json", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="doc/results")
    args = ap.parse_args()

    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(ROOT, out_dir)

    if args.architecture_only or not args.routing_json:
        from tools.log_routing_diagnostics import _architecture_summary
        arch = _architecture_summary()
        plot_vision_bottleneck(arch, os.path.join(out_dir, "results_vision_bottleneck.png"))
        plot_generality_routing_gap(os.path.join(out_dir, "results_generality_routing_gap.png"))
        if args.architecture_only:
            return 0

    if not args.routing_json:
        print("No --routing-json provided; only architecture figure generated.")
        return 0

    json_path = args.routing_json
    if not os.path.isabs(json_path):
        json_path = os.path.join(ROOT, json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    arch = data.get("architecture") or {}
    plot_vision_bottleneck(arch, os.path.join(out_dir, "results_vision_bottleneck.png"))

    summary = data.get("summary", {})
    suffix = f" (n={summary.get('n_samples', '?')}, pool={summary.get('n_pool_entries', '?')})"
    plot_routing_probe_rates(summary, os.path.join(out_dir, "results_routing_probe_rates.png"), suffix)

    probes = data.get("probes", [])
    if probes:
        plot_sim_margin(probes, os.path.join(out_dir, "results_routing_sim_margin.png"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
