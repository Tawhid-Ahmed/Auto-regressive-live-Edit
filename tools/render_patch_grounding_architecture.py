"""Render proposed PatchGrounding architecture diagram (horizontal).

Usage:
  python tools/render_patch_grounding_architecture.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "doc" / "THESIS_architecture_diagram_patch_grounding_horizontal.png"


def box(ax, xy, w, h, title, lines, face, edge, title_size=9, line_size=7.5):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=title_size, fontweight="bold")
    ty = y + h - 0.09
    for line in lines:
        ax.text(x + w / 2, ty, line, ha="center", va="top", fontsize=line_size)
        ty -= 0.055
    return patch


def cylinder(ax, xy, w, h, label, face, edge):
    x, y = xy
    body = FancyBboxPatch(
        (x, y),
        w,
        h * 0.72,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
    )
    top = mpatches.Ellipse((x + w / 2, y + h * 0.72), w, h * 0.18, facecolor=face, edgecolor=edge, linewidth=1.2)
    bot = mpatches.Ellipse((x + w / 2, y), w, h * 0.18, facecolor=face, edgecolor=edge, linewidth=1.2)
    ax.add_patch(body)
    ax.add_patch(top)
    ax.add_patch(bot)
    ax.text(x + w / 2, y + h * 0.36, label, ha="center", va="center", fontsize=7.5)
    return body


def arrow(ax, p1, p2, text=None, style="-|>", color="#334155", rad=0.0, lw=1.2):
    a = FancyArrowPatch(
        p1,
        p2,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.02, text, ha="center", va="bottom", fontsize=6.5, color=color)


def main() -> None:
    fig, ax = plt.subplots(figsize=(18, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Colors aligned with original thesis diagram
    c_input = ("#e8f4fc", "#2b6cb0")
    c_chunk = ("#fef3c7", "#d97706")
    c_vlm = ("#ede9fe", "#6d28d9")
    c_patch = ("#ffedd5", "#c2410c")  # NEW branch
    c_editor = ("#d6ecd2", "#5a8f4e")
    c_route = ("#dfe7f5", "#5a78b0")
    c_out = ("#fce7f3", "#be185d")

    # Row y positions
    y_top = 0.58
    y_mid = 0.30
    y_bot = 0.06

    # --- Inputs ---
    box(ax, (0.02, y_top), 0.11, 0.34, "VLKEB-long sample", ["Image v", "Prompt p", "Long target y*"], *c_input)

    # --- AR planner ---
    box(
        ax,
        (0.16, y_bot),
        0.13,
        0.28,
        "AR chunk planner",
        ["split chunks s=16", "c1 → c2 → cK"],
        *c_chunk,
    )
    box(
        ax,
        (0.31, y_bot),
        0.14,
        0.28,
        "Vision-gated chunk k",
        ["Q-Former cross-attn peaks", "(Step 5 spike 4c)"],
        *c_patch,
    )

    # --- BLIP2 path (unchanged) ---
    box(
        ax,
        (0.16, y_top),
        0.20,
        0.34,
        "Frozen BLIP2-OPT-2.7b",
        ["EVA-ViT → Q-Former (32 tok)", "LLM forward", "Hidden h^le"],
        *c_vlm,
    )

    # --- NEW Patch grounding branch ---
    box(
        ax,
        (0.38, y_top),
        0.20,
        0.34,
        "Patch grounding branch (NEW)",
        [
            "Frozen DINOv2 → patch grid",
            "PatchGroundingModule",
            "cross-attn(query, patches)",
            "evr_patch routing key",
        ],
        *c_patch,
    )

    # --- Editor ---
    box(ax, (0.60, y_mid), 0.10, 0.20, "Expert gen", ["f_eg"], *c_editor)
    box(ax, (0.60, y_bot + 0.02), 0.10, 0.18, "Routing Q", ["φ_legacy", "(QVExtractor)"], *c_editor)
    box(ax, (0.72, y_bot + 0.02), 0.10, 0.18, "Routing patch", ["φ_patch", "(margin route)"], *c_patch)
    cylinder(ax, (0.84, y_bot + 0.04), 0.08, 0.16, "Pool E_t\n+ patch keys", *c_editor)

    box(ax, (0.60, y_top + 0.02), 0.08, 0.14, "Fused\nhard route", [], *c_route)
    box(ax, (0.70, y_top + 0.02), 0.08, 0.14, "Soft route", ["ψ"], *c_route)
    box(ax, (0.80, y_top + 0.02), 0.08, 0.14, "Chunk\ngate k", [], *c_route)
    box(ax, (0.90, y_top + 0.02), 0.08, 0.14, "Residual", ["update"], *c_editor)

    # --- Output ---
    box(ax, (0.90, y_mid), 0.08, 0.20, "Output", ["Long-form ŷ", "Metrics"], *c_out)

    # Arrows — inputs
    arrow(ax, (0.13, 0.78), (0.16, 0.78))  # v -> BLIP2
    arrow(ax, (0.13, 0.72), (0.38, 0.78), color="#c2410c")  # v -> DINO branch
    arrow(ax, (0.13, 0.66), (0.26, 0.78))  # p -> BLIP2
    arrow(ax, (0.13, 0.66), (0.48, 0.72), color="#c2410c")  # p -> PGM
    arrow(ax, (0.075, 0.58), (0.225, 0.20))  # y* -> chunk planner

    # BLIP2 internal flow implied; hidden to routing
    arrow(ax, (0.36, 0.70), (0.60, 0.16), rad=0.15)  # h^le -> routing Q
    arrow(ax, (0.36, 0.70), (0.48, 0.68), color="#c2410c", rad=-0.1)  # h^le -> PGM query

    # Patch branch to routing
    arrow(ax, (0.58, 0.66), (0.72, 0.16), color="#c2410c", rad=0.2)  # evr_patch -> routing patch

    # AR / vision gate
    arrow(ax, (0.38, 0.20), (0.65, 0.12))  # chunk k -> expert gen
    arrow(ax, (0.45, 0.20), (0.84, 0.72))  # chunk k -> chunk gate

    # Editor flow
    arrow(ax, (0.70, 0.40), (0.88, 0.14))  # f_eg -> pool
    arrow(ax, (0.65, 0.16), (0.84, 0.12))  # routing Q -> pool
    arrow(ax, (0.77, 0.16), (0.64, 0.72))  # routing patch -> fused hard
    arrow(ax, (0.88, 0.20), (0.64, 0.76))  # pool -> fused hard
    arrow(ax, (0.68, 0.72), (0.74, 0.72))  # hard -> soft
    arrow(ax, (0.78, 0.72), (0.84, 0.72))  # soft -> chunk gate
    arrow(ax, (0.94, 0.72), (0.94, 0.50))  # residual -> output

    # Edit layer loop
    arrow(ax, (0.94, 0.50), (0.30, 0.86), text="edit layer", rad=0.25, color="#6d28d9", lw=1.4)
    arrow(ax, (0.30, 0.86), (0.94, 0.58), rad=0.25, color="#6d28d9", lw=1.4)

    # Legend
    ax.text(
        0.02,
        0.98,
        "Proposed: DINOv2 patch keys parallel to Q-Former routing — BLIP2 generation path unchanged",
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    legend_items = [
        mpatches.Patch(facecolor=c_vlm[0], edgecolor=c_vlm[1], label="Frozen VLM (unchanged)"),
        mpatches.Patch(facecolor=c_patch[0], edgecolor=c_patch[1], label="NEW patch grounding"),
        mpatches.Patch(facecolor=c_editor[0], edgecolor=c_editor[1], label="LiveEdit MoE editor"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8, frameon=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
