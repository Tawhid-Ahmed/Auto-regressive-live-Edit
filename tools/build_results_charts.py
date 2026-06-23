"""Generate thesis result charts — colorblind-friendly palette, axis labels, captions."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "doc" / "results"
W, H = 1280, 760

# Tol / IBM-inspired — blue vs amber/teal, not red vs orange
C_BASELINE = "#3366CC"      # steel blue
C_AR = "#CC6600"            # amber (pairs well with blue)
C_SINGLE = "#757575"        # neutral gray
C_SHORT = "#8B5A2B"         # brown (warning / negative control)
C_CHUNK_EM = "#4338CA"      # indigo
C_CHUNK_F1 = "#0D9488"      # teal (not red/green pair)
C_DELTA = "#3366CC"         # unified blue for delta bars
C_LINE = "#3366CC"

MARGIN = dict(left=118, right=108, top=88, bottom=118)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    bold_candidates = [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    for path in (bold_candidates if bold else candidates):
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def _plot_area() -> tuple[int, int, int, int]:
    return MARGIN["left"], MARGIN["top"], W - MARGIN["right"], H - MARGIN["bottom"]


def _new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    return Image.new("RGB", (W, H), "white"), ImageDraw.Draw(Image.new("RGB", (W, H), "white"))


def _draw_rotated_text(img: Image.Image, text: str, center_x: int, center_y: int, size: int = 13, bold: bool = True) -> None:
    f = _font(size, bold=bold)
    tmp = Image.new("RGBA", (400, 40), (255, 255, 255, 0))
    td = ImageDraw.Draw(tmp)
    tw = td.textlength(text, font=f)
    td.text((0, 0), text, fill="#111827", font=f)
    rotated = tmp.crop((0, 0, int(tw) + 4, 22)).rotate(90, expand=True)
    img.paste(rotated, (center_x - rotated.width // 2, center_y - rotated.height // 2), rotated)


def _draw_header(draw: ImageDraw.ImageDraw, title: str, caption: str) -> None:
    ft = _font(22, bold=True)
    fc = _font(12)
    tw = draw.textlength(title, font=ft)
    draw.text(((W - tw) / 2, 14), title, fill="#111827", font=ft)
    cw = draw.textlength(caption, font=fc)
    draw.text(((W - cw) / 2, 44), caption, fill="#4B5563", font=fc)


def _draw_axis_labels(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    xlabel: str,
    ylabel: str,
) -> None:
    fl = _font(13, bold=True)
    xw = draw.textlength(xlabel, font=fl)
    draw.text(((x0 + x1) / 2 - xw / 2, y1 + 52), xlabel, fill="#111827", font=fl)
    _draw_rotated_text(img, ylabel, center_x=34, center_y=(y0 + y1) // 2)


def _draw_y_axis(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    y_min: float,
    y_max: float,
    y_ticks: Sequence[float],
    tick_fmt: str = "{:g}",
) -> None:
    draw.line([(x0, y1), (x1, y1)], fill="#111827", width=2)
    draw.line([(x0, y0), (x0, y1)], fill="#111827", width=2)
    ft = _font(11)
    for tick in y_ticks:
        t = (tick - y_min) / (y_max - y_min)
        y = y1 - t * (y1 - y0)
        draw.line([(x0 - 5, y), (x0, y)], fill="#6B7280", width=1)
        draw.line([(x0, y), (x1, y)], fill="#E5E7EB", width=1)
        label = tick_fmt.format(tick)
        draw.text((x0 - 8 - draw.textlength(label, font=ft), y - 7), label, fill="#374151", font=ft)


def _y_to_px(y: float, y_min: float, y_max: float, y0: int, y1: int) -> float:
    return y1 - (y - y_min) / (y_max - y_min) * (y1 - y0)


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    items: Sequence[tuple[str, str]],
    x: int,
    y: int,
    ncol: int = 1,
) -> None:
    f = _font(11)
    box = 14
    pad = 10
    row_h = 22
    col_w = max(draw.textlength(label, font=f) for _, label in items) + box + 22
    rows = math.ceil(len(items) / ncol) if ncol > 1 else len(items)
    cols = ncol if ncol > 1 else 1
    w = cols * col_w + pad * 2
    h = rows * row_h + pad * 2
    draw.rectangle([x, y, x + w, y + h], fill="#FAFAFA", outline="#CBD5E1", width=1)
    for i, (color, label) in enumerate(items):
        if ncol > 1:
            col, row = i % cols, i // cols
        else:
            col, row = 0, i
        lx = x + pad + col * col_w
        ly = y + pad + row * row_h
        draw.rectangle([lx, ly + 3, lx + box, ly + 3 + box], fill=color, outline="#374151", width=1)
        draw.text((lx + box + 6, ly + 1), label, fill="#111827", font=f)


def _save(img: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / name, "PNG", optimize=True)


def chart_efficacy_vs_length() -> None:
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    _draw_header(
        draw,
        "Editing efficacy vs target length (concept)",
        "Single-pass control weakens on long targets; AR chunk editing stays stable.",
    )
    x0, y0, x1, y1 = _plot_area()
    _draw_y_axis(draw, x0, y0, x1, y1, 0, 100, range(0, 101, 10))
    _draw_axis_labels(img, draw, x0, y0, x1, y1, "Target length (tokens)", "Efficacy (%)")

    xs = [10, 30, 60, 100, 150]
    single, ar = [95, 80, 55, 30, 18], [94, 90, 86, 83, 81]

    def xpx(v: float) -> float:
        return x0 + (v - 10) / 140 * (x1 - x0)

    for vals, color, marker in [(single, C_SINGLE, "o"), (ar, C_AR, "s")]:
        pts = [(xpx(x), _y_to_px(y, 0, 100, y0, y1)) for x, y in zip(xs, vals)]
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=color, width=3)
        for x, y in zip(xs, vals):
            px, py = xpx(x), _y_to_px(y, 0, 100, y0, y1)
            if marker == "o":
                draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=color, outline="#374151", width=1)
            else:
                draw.rectangle([px - 6, py - 6, px + 6, py + 6], fill=color, outline="#374151", width=1)
            draw.text((px - 8, y1 + 12), str(x), fill="#374151", font=_font(11))

    _draw_legend(draw, [(C_SINGLE, "Single-pass editing"), (C_AR, "Autoregressive chunk editing")], x1 - 310, y0 + 8)
    _save(img, "results_efficacy_vs_length.png")


def chart_baseline_vs_ar() -> None:
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    _draw_header(
        draw,
        "Long baseline vs AR cs=16 (VLKEB-long)",
        "AR improves every metric slightly; largest relative gain on chunk EM (+1.3 pp).",
    )
    x0, y0, x1, y1 = _plot_area()
    _draw_y_axis(draw, x0, y0, x1, y1, 0, 100, range(0, 101, 10))
    _draw_axis_labels(img, draw, x0, y0, x1, y1, "Evaluation metric", "Score (%)")

    metrics = ["Rel acc", "Chunk EM", "Chunk F1", "Text gen", "Image gen"]
    baseline = [75.40, 13.50, 76.28, 75.21, 71.49]
    ar_vals = [75.94, 14.82, 76.90, 75.63, 72.51]
    n = len(metrics)
    gw = (x1 - x0) / n
    bw = gw * 0.28

    for i, (m, b, a) in enumerate(zip(metrics, baseline, ar_vals)):
        cx = x0 + (i + 0.5) * gw
        bx, ax_ = cx - bw - 2, cx + 2
        by, ay = _y_to_px(b, 0, 100, y0, y1), _y_to_px(a, 0, 100, y0, y1)
        draw.rectangle([bx, by, bx + bw, y1], fill=C_BASELINE, outline="#1E40AF", width=1)
        draw.rectangle([ax_, ay, ax_ + bw, y1], fill=C_AR, outline="#92400E", width=1)
        draw.text((cx - draw.textlength(m, font=_font(11)) / 2, y1 + 12), m, fill="#374151", font=_font(11))
        draw.text((bx, by - 15), f"{b:.1f}", fill=C_BASELINE, font=_font(9, bold=True))
        draw.text((ax_, ay - 15), f"{a:.1f}", fill=C_AR, font=_font(9, bold=True))

    _draw_legend(draw, [(C_BASELINE, "Long baseline"), (C_AR, "AR cs=16")], x1 - 210, y0 + 8)
    _save(img, "results_baseline_vs_ar.png")


def chart_ar_delta() -> None:
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    _draw_header(
        draw,
        "AR gain over long baseline",
        "Absolute improvement in percentage points — modest but consistent across probes.",
    )
    x0, y0, x1, y1 = _plot_area()
    _draw_y_axis(draw, x0, y0, x1, y1, 0, 1.6, [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
    _draw_axis_labels(img, draw, x0, y0, x1, y1, "Evaluation metric", "Gain (pp)")

    metrics = ["Rel acc", "Chunk EM", "Chunk F1", "Text gen", "Image gen"]
    deltas = [0.54, 1.32, 0.62, 0.42, 1.02]
    shades = ["#93C5FD", "#3366CC", "#60A5FA", "#2563EB", "#1D4ED8"]
    n = len(metrics)
    bw = (x1 - x0) / n * 0.52

    for i, (m, d, c) in enumerate(zip(metrics, deltas, shades)):
        cx = x0 + (i + 0.5) * (x1 - x0) / n
        bx = cx - bw / 2
        by = _y_to_px(d, 0, 1.6, y0, y1)
        draw.rectangle([bx, by, bx + bw, y1], fill=c, outline="#1E3A8A", width=1)
        draw.text((cx - draw.textlength(m, font=_font(11)) / 2, y1 + 12), m, fill="#374151", font=_font(11))
        draw.text((cx - 24, by - 18), f"+{d:.2f}", fill="#111827", font=_font(10, bold=True))

    _draw_legend(draw, [(shades[1], "AR − baseline (pp)")], x1 - 200, y0 + 8)
    _save(img, "results_ar_delta.png")


def chart_training_effect() -> None:
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    _draw_header(
        draw,
        "Reliability on VLKEB-long (token accuracy)",
        "Long-form training is essential: short-trained checkpoint fails on long eval (−47 pp).",
    )
    x0, y0, x1, y1 = _plot_area()
    _draw_y_axis(draw, x0, y0, x1, y1, 0, 100, range(0, 101, 10))
    _draw_axis_labels(img, draw, x0, y0, x1, y1, "Training / eval regime", "Token accuracy (%)")

    labels = ["Short-trained\n(on long eval)", "Long baseline", "AR cs=16"]
    values = [28.57, 75.40, 75.94]
    colors = [C_SHORT, C_BASELINE, C_AR]
    n = len(labels)
    bw = (x1 - x0) / n * 0.42

    for i, (lab, val, col) in enumerate(zip(labels, values, colors)):
        cx = x0 + (i + 0.5) * (x1 - x0) / n
        bx = cx - bw / 2
        by = _y_to_px(val, 0, 100, y0, y1)
        draw.rectangle([bx, by, bx + bw, y1], fill=col, outline="#374151", width=1)
        draw.text((cx - 22, by - 20), f"{val:.1f}%", fill="#111827", font=_font(11, bold=True))
        for j, line in enumerate(lab.split("\n")):
            draw.text((cx - draw.textlength(line, font=_font(11)) / 2, y1 + 12 + j * 15), line, fill="#374151", font=_font(11))

    _draw_legend(
        draw,
        [(C_SHORT, "Short-trained"), (C_BASELINE, "Long-trained baseline"), (C_AR, "Long-trained AR")],
        x0 + 8, y0 + 8, ncol=1,
    )
    _save(img, "results_training_effect.png")


def chart_chunk_sweep() -> None:
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    _draw_header(
        draw,
        "AR inference chunk-size ablation",
        "Chunk EM falls with larger windows (metric effect); chunk F1 rises slightly; same checkpoint.",
    )

    x0, y0, x1, y1 = MARGIN["left"], MARGIN["top"], W - MARGIN["right"] - 62, H - MARGIN["bottom"]
    draw.line([(x0, y1), (x1, y1)], fill="#111827", width=2)
    draw.line([(x0, y0), (x0, y1)], fill="#111827", width=2)
    draw.line([(x1, y0), (x1, y1)], fill="#111827", width=2)

    chunk_sizes = [8, 16, 32, 64]
    chunk_em = [26.26, 14.82, 8.11, 6.14]
    chunk_f1 = [76.48, 76.90, 77.30, 77.88]
    em_min, em_max, f1_min, f1_max = 0, 35, 74, 80

    def xpx(v: float) -> float:
        return x0 + (v - 8) / 56 * (x1 - x0)

    ft = _font(11)
    for tick in range(0, 36, 5):
        y = _y_to_px(tick, em_min, em_max, y0, y1)
        draw.line([(x0, y), (x1, y)], fill="#EEF2FF", width=1)
        draw.text((x0 - 28, y - 7), f"{tick}", fill=C_CHUNK_EM, font=ft)
    for tick in range(74, 81, 2):
        y = y1 - (tick - f1_min) / (f1_max - f1_min) * (y1 - y0)
        draw.text((x1 + 10, y - 7), f"{tick}", fill=C_CHUNK_F1, font=ft)

    em_pts = [(xpx(x), _y_to_px(y, em_min, em_max, y0, y1)) for x, y in zip(chunk_sizes, chunk_em)]
    f1_pts = [(xpx(x), y1 - (y - f1_min) / (f1_max - f1_min) * (y1 - y0)) for x, y in zip(chunk_sizes, chunk_f1)]

    for i in range(len(em_pts) - 1):
        draw.line([em_pts[i], em_pts[i + 1]], fill=C_CHUNK_EM, width=3)
        draw.line([f1_pts[i], f1_pts[i + 1]], fill=C_CHUNK_F1, width=3)
    for px, py in em_pts:
        draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=C_CHUNK_EM, outline="#312E81")
    for px, py in f1_pts:
        draw.rectangle([px - 6, py - 6, px + 6, py + 6], fill=C_CHUNK_F1, outline="#115E59")
    for x in chunk_sizes:
        draw.text((xpx(x) - 6, y1 + 12), str(x), fill="#374151", font=ft)

    xl = "Inference chunk size"
    xw = draw.textlength(xl, font=_font(13, bold=True))
    draw.text(((x0 + x1) / 2 - xw / 2, y1 + 52), xl, fill="#111827", font=_font(13, bold=True))
    _draw_rotated_text(img, "Chunk EM (%)", center_x=52, center_y=(y0 + y1) // 2 - 20)
    _draw_rotated_text(img, "Chunk F1 (%)", center_x=W - 38, center_y=(y0 + y1) // 2 - 20, size=12)

    _draw_legend(draw, [(C_CHUNK_EM, "Chunk EM (left)"), (C_CHUNK_F1, "Chunk F1 (right)")], x0 + 8, y0 + 8)
    _save(img, "results_chunk_sweep.png")


def chart_reliability_sweep() -> None:
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    _draw_header(
        draw,
        "Token accuracy vs inference chunk size",
        "Reliability is flat at 75.94% — changing inference granularity does not move token accuracy.",
    )
    x0, y0, x1, y1 = _plot_area()
    _draw_y_axis(draw, x0, y0, x1, y1, 75, 76.5, [75, 75.5, 76, 76.5], tick_fmt="{:.1f}")
    _draw_axis_labels(img, draw, x0, y0, x1, y1, "Inference chunk size", "Reliability (%)")

    chunk_sizes = [8, 16, 32, 64]
    rel = [75.94] * 4

    def xpx(v: float) -> float:
        return x0 + (v - 8) / 56 * (x1 - x0)

    pts = [(xpx(x), _y_to_px(y, 75, 76.5, y0, y1)) for x, y in zip(chunk_sizes, rel)]
    draw.line([pts[0], pts[-1]], fill=C_LINE, width=3)
    for i, (px, py) in enumerate(pts):
        draw.ellipse([px - 7, py - 7, px + 7, py + 7], fill=C_LINE, outline="#1E40AF", width=2)
        if i == 1:
            draw.text((px - 28, py - 30), f"{rel[0]:.2f}%", fill=C_LINE, font=_font(10, bold=True))
        draw.text((px - 6, y1 + 12), str(chunk_sizes[i]), fill="#374151", font=_font(11))

    _draw_legend(draw, [(C_LINE, "Reliability (token accuracy)")], x1 - 260, y0 + 8)
    _save(img, "results_reliability_sweep.png")


def main() -> None:
    chart_efficacy_vs_length()
    chart_baseline_vs_ar()
    chart_ar_delta()
    chart_training_effect()
    chart_chunk_sweep()
    chart_reliability_sweep()
    print(f"Wrote 6 charts to {OUT}")


if __name__ == "__main__":
    main()
