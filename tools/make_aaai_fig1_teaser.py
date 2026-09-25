"""Compose AAAI Fig.1 teaser: reliability vs false-positive retrieval + DEK fix.

Uses Times New Roman (AAAI body font) so embedded text matches the paper.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "doc" / "figures" / "supervisor_samples"
OUTS = [
    ROOT / "doc" / "figures" / "aaai" / "fig1_teaser.png",
    ROOT / "AuthorKit27" / "Figures" / "fig1_teaser.png",
]


def _aaai_fonts():
    """Match AAAI-27 body style: Times New Roman + STIX math."""
    mpl.rcParams.update(
        {
            # Force Times New Roman (Windows times.ttf); do not fall back to DejaVu.
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rgb(name: str, size=(320, 240)):
    im = Image.open(SAMPLES / name).convert("RGB")
    im.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (245, 245, 245))
    ox = (size[0] - im.width) // 2
    oy = (size[1] - im.height) // 2
    canvas.paste(im, (ox, oy))
    return canvas


def main():
    _aaai_fonts()

    edit = load_rgb("sampleA_edit_kankakee.jpg")
    swap = load_rgb("sampleA_swap.jpg")
    hard = load_rgb("sampleA_hardneg.jpg")

    # AAAI textwidth ≈ 7in; body ≈ 10pt Times
    fig = plt.figure(figsize=(7.0, 3.55), dpi=300)
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.55, 1.0], hspace=0.22, left=0.04, right=0.98, top=0.92, bottom=0.06
    )

    # Point sizes chosen to read like caption/body text when the figure is full-width.
    FS_TITLE = 10.0
    FS_PANEL = 9.0
    FS_NOTE = 8.0
    FS_CAPTION = 8.0
    FS_BOX = 8.5
    FS_MARK = 13.0

    # ---- Top: problem probes ----
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title(
        "Lifelong edit looks correct on the edited image, yet can fire on a wrong scene",
        fontsize=FS_TITLE,
        pad=6,
        fontweight="bold",
        fontfamily="Times New Roman",
    )

    panels = [
        (0.25, edit, "Edited image", "Reliability\nexpect: Kankakee", "#1b7f4e", "✓"),
        (3.45, swap, "Image-swap", "False-positive\nretrieval risk", "#b33a3a", "✗"),
        (6.65, hard, "Hard-negative", "Confusable\nnon-match", "#b33a3a", "✗"),
    ]
    for x0, im, title, note, color, mark in panels:
        ax.imshow(im, extent=[x0, x0 + 2.9, 1.15, 3.55], aspect="auto", zorder=2)
        ax.add_patch(
            FancyBboxPatch(
                (x0 - 0.05, 1.05),
                3.0,
                2.6,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.4,
                edgecolor=color,
                facecolor="none",
                zorder=3,
            )
        )
        ax.text(
            x0 + 1.45,
            3.85,
            title,
            ha="center",
            va="bottom",
            fontsize=FS_PANEL,
            fontweight="bold",
            fontfamily="Times New Roman",
        )
        ax.text(
            x0 + 1.45,
            0.55,
            note,
            ha="center",
            va="center",
            fontsize=FS_NOTE,
            color=color,
            linespacing=1.15,
            fontfamily="Times New Roman",
        )
        ax.text(
            x0 + 2.55,
            3.25,
            mark,
            ha="center",
            va="center",
            fontsize=FS_MARK,
            color=color,
            fontweight="bold",
            zorder=4,
            # Times lacks ✓/✗; keep marks readable without changing body text font.
            fontfamily="DejaVu Sans",
        )

    ax.text(
        5.0,
        0.08,
        r"Shared prompt $p$: ``What city in Illinois is depicted in the image?''  "
        r"Weak visual keys $\Rightarrow$ wrong stored expert can still retrieve.",
        ha="center",
        va="bottom",
        fontsize=FS_CAPTION,
        color="#333333",
        fontfamily="Times New Roman",
    )

    # ---- Bottom: Dual-Expert Keys ----
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 2.6)
    ax2.axis("off")
    ax2.set_title(
        r"Our fix: Dual-Expert Keys (vision $\wedge$ text) + adversarial visual binding",
        fontsize=FS_TITLE,
        pad=4,
        fontweight="bold",
        fontfamily="Times New Roman",
    )

    def box(x, y, w, h, text, fc, ec):
        ax2.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.2,
                edgecolor=ec,
                facecolor=fc,
                zorder=2,
            )
        )
        ax2.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=FS_BOX,
            zorder=3,
            fontfamily="Times New Roman",
        )

    box(0.3, 0.85, 2.1, 1.15, "Query\n$(p', v')$", "#eef3fb", "#3a5f8a")
    box(3.0, 1.45, 2.0, 0.75, "Vision gate", "#e8f5e9", "#2e7d32")
    box(3.0, 0.45, 2.0, 0.75, "Text gate", "#e8f5e9", "#2e7d32")
    box(5.7, 0.85, 1.9, 1.15, r"vision $\wedge$ text", "#fff8e1", "#f9a825")
    box(8.1, 0.85, 1.6, 1.15, "Fire edit\nexpert", "#f3e5f5", "#6a1b9a")

    for (x1, y1, x2, y2) in [
        (2.4, 1.42, 3.0, 1.82),
        (2.4, 1.42, 3.0, 0.82),
        (5.0, 1.82, 5.7, 1.55),
        (5.0, 0.82, 5.7, 1.25),
        (7.6, 1.42, 8.1, 1.42),
    ]:
        ax2.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1.0,
                color="#555555",
                zorder=1,
            )
        )

    ax2.text(
        5.0,
        0.12,
        "Adversarial visual binding training attaches each edit to its image and "
        "suppresses activation on swap / hard-negative scenes.",
        ha="center",
        va="bottom",
        fontsize=FS_CAPTION,
        color="#333333",
        fontfamily="Times New Roman",
    )

    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
