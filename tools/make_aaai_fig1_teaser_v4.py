"""AAAI Fig.1 teaser v4: same layout/aspect as v2, larger fonts for column-width use.

Does NOT overwrite fig1_teaser.png / v2 / v3.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "doc" / "figures" / "supervisor_samples"
OUTS = [
    ROOT / "doc" / "figures" / "aaai" / "fig1_teaser_v4.png",
    ROOT / "AuthorKit27" / "Figures" / "fig1_teaser_v4.png",
]


def _aaai_fonts():
    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.antialiased": True,
            "figure.dpi": 400,
            "savefig.dpi": 400,
        }
    )


def load_rgb(name: str, size=(480, 360)):
    """Higher-res panel images so the figure stays sharp at 400 dpi."""
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

    # Slightly taller canvas so notes + shared-prompt never collide.
    fig = plt.figure(figsize=(7.0, 4.05), dpi=400)
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.85, 1.0],
        hspace=0.28,
        left=0.035,
        right=0.985,
        top=0.93,
        bottom=0.05,
    )

    # Same canvas as v2. Fonts ~1.85× so after scale to \columnwidth (~0.47×)
    # they still read near AAAI caption/body size.
    FS_TITLE = 18.5
    FS_PANEL = 16.5
    FS_NOTE = 15.0
    FS_CAPTION = 14.5
    FS_BOX = 15.5
    FS_MARK_OK = 17.0
    FS_MARK_X = 16.0

    # ---- Top: problem probes ----
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.0)  # extra bottom room for notes + caption
    ax.axis("off")
    ax.set_title(
        "Lifelong edit looks correct on the edited image, yet can fire on a wrong scene",
        fontsize=FS_TITLE,
        pad=8,
        fontweight="bold",
        fontfamily="Times New Roman",
    )

    # Image band: y in [2.05, 4.35]; notes sit clearly below; caption at bottom.
    panels = [
        (0.25, edit, "Edited image", "Reliability\nexpect: Kankakee", "#1b7f4e", True),
        (3.45, swap, "Image-swap", "False-positive\nretrieval risk", "#b33a3a", False),
        (6.65, hard, "Hard-negative", "Confusable\nnon-match", "#b33a3a", False),
    ]
    for x0, im, title, note, color, is_ok in panels:
        ax.imshow(im, extent=[x0, x0 + 2.9, 2.05, 4.35], aspect="auto", zorder=2)
        ax.add_patch(
            FancyBboxPatch(
                (x0 - 0.05, 1.95),
                3.0,
                2.55,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.4,
                edgecolor=color,
                facecolor="none",
                zorder=3,
            )
        )
        ax.text(
            x0 + 1.45,
            4.62,
            title,
            ha="center",
            va="bottom",
            fontsize=FS_PANEL,
            fontweight="bold",
            fontfamily="Times New Roman",
        )
        ax.text(
            x0 + 1.45,
            1.35,
            note,
            ha="center",
            va="center",
            fontsize=FS_NOTE,
            color=color,
            linespacing=1.2,
            fontfamily="Times New Roman",
            zorder=5,
        )
        # Corner mark in display-stable text (line plots stretch under unequal axes aspect).
        mx, my = x0 + 2.55, 4.05
        ax.text(
            mx,
            my,
            "✓" if is_ok else "X",
            ha="center",
            va="center",
            fontsize=FS_MARK_OK if is_ok else FS_MARK_X,
            color=color,
            fontweight="bold",
            zorder=4,
            fontfamily="DejaVu Sans",
        )

    # Shared prompt on its own band (no overlap with the three notes).
    ax.text(
        5.0,
        0.35,
        r"Shared prompt $p$: ``What city in Illinois is depicted in the image?''"
        "\n"
        r"Weak visual keys $\Rightarrow$ wrong stored expert can still retrieve.",
        ha="center",
        va="center",
        fontsize=FS_CAPTION,
        color="#222222",
        linespacing=1.35,
        fontfamily="Times New Roman",
        zorder=5,
    )

    # ---- Bottom: Dual-Expert Keys ----
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 2.75)
    ax2.axis("off")
    ax2.set_title(
        r"Our fix: Dual-Expert Keys (vision $\wedge$ text) + adversarial visual binding",
        fontsize=FS_TITLE,
        pad=6,
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

    box(0.3, 0.95, 2.1, 1.15, "Query\n$(p', v')$", "#eef3fb", "#3a5f8a")
    box(3.0, 1.55, 2.0, 0.75, "Vision gate", "#e8f5e9", "#2e7d32")
    box(3.0, 0.55, 2.0, 0.75, "Text gate", "#e8f5e9", "#2e7d32")
    box(5.7, 0.95, 1.9, 1.15, r"vision $\wedge$ text", "#fff8e1", "#f9a825")
    box(8.1, 0.95, 1.6, 1.15, "Fire edit\nexpert", "#f3e5f5", "#6a1b9a")

    for (x1, y1, x2, y2) in [
        (2.4, 1.52, 3.0, 1.92),
        (2.4, 1.52, 3.0, 0.92),
        (5.0, 1.92, 5.7, 1.65),
        (5.0, 0.92, 5.7, 1.35),
        (7.6, 1.52, 8.1, 1.52),
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
        0.18,
        "Adversarial visual binding training attaches each edit to its image and "
        "suppresses activation on swap / hard-negative scenes.",
        ha="center",
        va="center",
        fontsize=FS_CAPTION,
        color="#222222",
        fontfamily="Times New Roman",
    )

    for out in OUTS:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            out,
            dpi=400,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            pad_inches=0.04,
        )
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
