"""AAAI Fig.1 teaser — single-column readable version (v3).

Designed at ~AAAI column width so body text stays ~8–10pt in the PDF.
Does NOT overwrite fig1_teaser.png or fig1_teaser_v2.png.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "doc" / "figures" / "supervisor_samples"
OUTS = [
    ROOT / "doc" / "figures" / "aaai" / "fig1_teaser_v3.png",
    ROOT / "AuthorKit27" / "Figures" / "fig1_teaser_v3.png",
]

# AAAI letter two-column text column ≈ 3.3–3.4 in
COL_W = 3.35


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


def load_rgb(name: str, size=(280, 210)):
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

    # Tall single-column figure: fonts here ≈ final PDF size at width=\columnwidth.
    fig = plt.figure(figsize=(COL_W, 5.35), dpi=400)
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[2.35, 1.35],
        hspace=0.22,
        left=0.04,
        right=0.96,
        top=0.95,
        bottom=0.04,
    )

    # Readable at column width (AAAI body ~10pt; figure text a bit smaller).
    FS_TITLE = 9.0
    FS_PANEL = 8.0
    FS_NOTE = 7.5
    FS_CAPTION = 7.0
    FS_BOX = 8.0
    FS_MARK = 9.0

    # ---- Top: problem probes ----
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.3)
    ax.axis("off")
    ax.set_title(
        "Lifelong edit looks correct on the edited image,\nyet can fire on a wrong scene",
        fontsize=FS_TITLE,
        pad=4,
        fontweight="bold",
        fontfamily="Times New Roman",
        linespacing=1.15,
    )

    panels = [
        (0.2, edit, "Edited image", "Reliability\nexpect: Kankakee", "#1b7f4e", True),
        (3.4, swap, "Image-swap", "False-positive\nretrieval risk", "#b33a3a", False),
        (6.6, hard, "Hard-negative", "Confusable\nnon-match", "#b33a3a", False),
    ]
    for x0, im, title, note, color, is_ok in panels:
        ax.imshow(im, extent=[x0, x0 + 3.0, 2.15, 4.45], aspect="auto", zorder=2)
        ax.add_patch(
            FancyBboxPatch(
                (x0 - 0.05, 2.05),
                3.1,
                2.55,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=1.2,
                edgecolor=color,
                facecolor="none",
                zorder=3,
            )
        )
        ax.text(
            x0 + 1.5,
            4.72,
            title,
            ha="center",
            va="bottom",
            fontsize=FS_PANEL,
            fontweight="bold",
            fontfamily="Times New Roman",
        )
        ax.text(
            x0 + 1.5,
            1.45,
            note,
            ha="center",
            va="center",
            fontsize=FS_NOTE,
            color=color,
            linespacing=1.15,
            fontfamily="Times New Roman",
            zorder=5,
        )
        ax.text(
            x0 + 2.65,
            4.15,
            "✓" if is_ok else "X",
            ha="center",
            va="center",
            fontsize=FS_MARK,
            color=color,
            fontweight="bold",
            zorder=4,
            fontfamily="DejaVu Sans",
        )

    ax.text(
        5.0,
        0.42,
        r"Shared prompt $p$: ``What city in Illinois"
        "\n"
        r"is depicted in the image?''"
        "\n"
        r"Weak visual keys $\Rightarrow$ wrong stored expert can still retrieve.",
        ha="center",
        va="center",
        fontsize=FS_CAPTION,
        color="#222222",
        linespacing=1.3,
        fontfamily="Times New Roman",
        zorder=5,
    )

    # ---- Bottom: Dual-Expert Keys ----
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 3.05)
    ax2.axis("off")
    ax2.set_title(
        r"Our fix: Dual-Expert Keys (vision $\wedge$ text)"
        "\n"
        r"+ adversarial visual binding",
        fontsize=FS_TITLE,
        pad=3,
        fontweight="bold",
        fontfamily="Times New Roman",
        linespacing=1.15,
    )

    def box(x, y, w, h, text, fc, ec):
        ax2.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=1.1,
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
            linespacing=1.1,
        )

    # Compact flowchart for narrow column
    box(0.25, 1.05, 2.2, 1.2, "Query\n$(p', v')$", "#eef3fb", "#3a5f8a")
    box(3.0, 1.7, 2.05, 0.8, "Vision gate", "#e8f5e9", "#2e7d32")
    box(3.0, 0.65, 2.05, 0.8, "Text gate", "#e8f5e9", "#2e7d32")
    box(5.65, 1.05, 2.0, 1.2, r"vision $\wedge$" "\n" r"text", "#fff8e1", "#f9a825")
    box(8.15, 1.05, 1.6, 1.2, "Fire edit\nexpert", "#f3e5f5", "#6a1b9a")

    for (x1, y1, x2, y2) in [
        (2.45, 1.65, 3.0, 2.1),
        (2.45, 1.65, 3.0, 1.05),
        (5.05, 2.1, 5.65, 1.85),
        (5.05, 1.05, 5.65, 1.35),
        (7.65, 1.65, 8.15, 1.65),
    ]:
        ax2.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.95,
                color="#555555",
                zorder=1,
            )
        )

    ax2.text(
        5.0,
        0.22,
        "Adversarial visual binding attaches each edit to its image\n"
        "and suppresses activation on swap / hard-negative scenes.",
        ha="center",
        va="center",
        fontsize=FS_CAPTION,
        color="#222222",
        linespacing=1.25,
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
            pad_inches=0.03,
        )
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
