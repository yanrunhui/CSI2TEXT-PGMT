#!/usr/bin/env python3
"""Render two grouped line panels for leave-one-Nf-out degradation trends."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from generate_leave_one_nf_heatmap import HELD_OUT_NF, ROWS, degradation


OUT_DIR = Path("output/figures")
PNG_OUT = OUT_DIR / "leave_one_nf_out_grouped_lines.png"
PDF_OUT = OUT_DIR / "leave_one_nf_out_grouped_lines.pdf"
FULL_PAPER_OUT_DIR = Path("tmp/pdfs/full_paper/figures")
FULL_PAPER_PNG_OUT = FULL_PAPER_OUT_DIR / "leave_one_nf_out_line_panels.png"
FULL_PAPER_PDF_OUT = FULL_PAPER_OUT_DIR / "leave_one_nf_out_line_panels.pdf"

W, H = 2500, 1360
SCALE = 2

FONT_REGULAR = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"

INK = "#20242A"
MUTED = "#5D6470"
GRID = "#D5DAE0"
AXIS = "#858B94"
WHITE = "#FFFFFF"
ANNOTATION = "#155F83"

SERIES_STYLE = {
    "LoS/NLoS accuracy ↑": ("#0072B2", "circle", "LoS/NLoS"),
    "Text factuality ↑": ("#009E73", "square", "Factuality"),
    "Numeric accuracy ↑": ("#E69F00", "triangle", "Numeric"),
    "K-factor MAE ↓": ("#CC79A7", "diamond", "K-factor"),
    "First-angle MAE ↓": ("#D55E00", "circle", "Angle"),
    "First-delay MAE ↓": ("#B2182B", "square", "Delay"),
    "First-power MAE ↓": ("#56B4E9", "triangle", "Power"),
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size * SCALE)


def xy(text_scale_tuple: tuple[int, int]) -> tuple[int, int]:
    return text_scale_tuple[0] * SCALE, text_scale_tuple[1] * SCALE


def centered(
    draw: ImageDraw.ImageDraw,
    center: tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str = INK,
) -> None:
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text(
        (center[0] * SCALE - (box[2] - box[0]) / 2, center[1] * SCALE - (box[3] - box[1]) / 2),
        text,
        font=fnt,
        fill=fill,
    )


def text(draw: ImageDraw.ImageDraw, pos: tuple[int, int], value: str, size: int, fill: str = INK, bold: bool = False) -> None:
    draw.text(xy(pos), value, font=font(size, bold), fill=fill)


def lookup(label: str) -> list[float]:
    row = next(row for row in ROWS if row["label"] == label)
    return [
        degradation(value, row["baseline"], row["higher_is_better"])
        for value in row["values"]
    ]


def draw_marker(draw: ImageDraw.ImageDraw, x: float, y: float, color: str, shape: str, radius: int = 15) -> None:
    r = radius * SCALE
    cx, cy = x, y
    outline_width = 4 * SCALE
    if shape == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color, outline=WHITE, width=outline_width)
    elif shape == "square":
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=color, outline=WHITE, width=outline_width)
    elif shape == "triangle":
        points = [(cx, cy - r * 1.15), (cx - r * 1.05, cy + r * 0.9), (cx + r * 1.05, cy + r * 0.9)]
        draw.polygon(points, fill=color)
        draw.line(points + [points[0]], fill=WHITE, width=outline_width, joint="curve")
    elif shape == "diamond":
        points = [(cx, cy - r * 1.1), (cx - r * 1.1, cy), (cx, cy + r * 1.1), (cx + r * 1.1, cy)]
        draw.polygon(points, fill=color)
        draw.line(points + [points[0]], fill=WHITE, width=outline_width, joint="curve")


def draw_legend(draw: ImageDraw.ImageDraw, left: int, top: int, labels: list[str], columns: int) -> None:
    item_w = 330
    item_h = 58
    for index, label in enumerate(labels):
        col = index % columns
        row = index // columns
        x = (left + col * item_w) * SCALE
        y = (top + row * item_h) * SCALE
        color, shape, display = SERIES_STYLE[label]
        draw.line((x, y + 25 * SCALE, x + 70 * SCALE, y + 25 * SCALE), fill=color, width=9 * SCALE)
        draw_marker(draw, x + 35 * SCALE, y + 25 * SCALE, color, shape, radius=12)
        draw.text((x + 85 * SCALE, y - 3 * SCALE), display, font=font(39), fill=MUTED)


def draw_right_brace(
    draw: ImageDraw.ImageDraw,
    x: float,
    y_top: float,
    y_bottom: float,
    color: str = ANNOTATION,
) -> None:
    w = 26 * SCALE
    width = 7 * SCALE
    mid = (y_top + y_bottom) / 2
    draw.line((x, y_top, x + w, y_top), fill=color, width=width)
    draw.line((x + w, y_top, x + w, mid - 14 * SCALE), fill=color, width=width)
    draw.line((x + w, mid - 14 * SCALE, x + 2 * w, mid), fill=color, width=width)
    draw.line((x + 2 * w, mid, x + w, mid + 14 * SCALE), fill=color, width=width)
    draw.line((x + w, mid + 14 * SCALE, x + w, y_bottom), fill=color, width=width)
    draw.line((x, y_bottom, x + w, y_bottom), fill=color, width=width)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = ANNOTATION,
) -> None:
    sx, sy = start
    ex, ey = end
    draw.line((sx, sy, ex, ey), fill=color, width=7 * SCALE)
    vx, vy = ex - sx, ey - sy
    length = (vx * vx + vy * vy) ** 0.5
    if length == 0:
        return
    ux, uy = vx / length, vy / length
    px, py = -uy, ux
    arrow_len = 28 * SCALE
    arrow_w = 16 * SCALE
    points = [
        (ex, ey),
        (ex - ux * arrow_len + px * arrow_w, ey - uy * arrow_len + py * arrow_w),
        (ex - ux * arrow_len - px * arrow_w, ey - uy * arrow_len - py * arrow_w),
    ]
    draw.polygon(points, fill=color)


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    labels: list[str],
    title: str,
    y_limits: tuple[float, float],
    y_ticks: list[float],
    legend_columns: int,
    annotate_delay: bool = False,
    annotation: str | None = None,
) -> None:
    left, top, right, bottom = [v * SCALE for v in box]
    plot_left = left + 125 * SCALE
    plot_right = right - 42 * SCALE
    plot_top = top + 230 * SCALE
    plot_bottom = bottom - 100 * SCALE

    draw.text((left, top), title, font=font(36, True), fill=INK)
    draw_legend(draw, box[0] + 155, box[1] + 78, labels, legend_columns)

    y_min, y_max = y_limits

    def x_pos(index: int) -> float:
        return plot_left + index / (len(HELD_OUT_NF) - 1) * (plot_right - plot_left)

    def y_pos(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    for tick in y_ticks:
        y = y_pos(tick)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=2 * SCALE)
        label = f"{tick:g}"
        tick_box = draw.textbbox((0, 0), label, font=font(36))
        draw.text((plot_left - 24 * SCALE - (tick_box[2] - tick_box[0]), y - 18 * SCALE), label, font=font(36), fill=MUTED)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=AXIS, width=3 * SCALE)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=AXIS, width=3 * SCALE)
    zero_y = y_pos(0.0)
    if plot_top <= zero_y <= plot_bottom:
        draw.line((plot_left, zero_y, plot_right, zero_y), fill="#777777", width=4 * SCALE)

    for index, nf in enumerate(HELD_OUT_NF):
        x = x_pos(index)
        draw.line((x, plot_bottom, x, plot_bottom + 12 * SCALE), fill=AXIS, width=3 * SCALE)
        centered(draw, (round(x / SCALE), round(plot_bottom / SCALE) + 54), f"N{nf}", font(38), fill=MUTED)

    for label in labels:
        color, shape, _ = SERIES_STYLE[label]
        values = lookup(label)
        points = [(x_pos(index), y_pos(value)) for index, value in enumerate(values)]
        draw.line(points, fill=color, width=13 * SCALE, joint="curve")
        for x, y in points:
            draw_marker(draw, x, y, color, shape)
        if annotate_delay and label == "First-delay MAE ↓":
            for point_index, (x, y, value) in enumerate(
                zip([point[0] for point in points], [point[1] for point in points], values)
            ):
                label_text = f"{value:+.1f}%"
                label_box = draw.textbbox((0, 0), label_text, font=font(29, True))
                label_x = x - (label_box[2] - label_box[0]) / 2
                label_y = y - 50 * SCALE
                if point_index == 0:
                    label_x = x + 22 * SCALE
                    label_y = y - 62 * SCALE
                elif point_index == len(points) - 1:
                    label_x = x - (label_box[2] - label_box[0]) - 12 * SCALE
                    label_y = y - 70 * SCALE
                draw.text(
                    (label_x, label_y),
                    label_text,
                    font=font(32, True),
                    fill=color,
                )

    draw.text((plot_left, plot_top - 55 * SCALE), "Performance degradation (%)", font=font(34), fill=INK)

    if annotation == "left_stable":
        draw.multiline_text(
            (x_pos(2) - 35 * SCALE, y_pos(7.75)),
            "remain within\nsingle-digit",
            font=font(35, True),
            fill=INK,
            spacing=8 * SCALE,
        )
    elif annotation == "right_delay":
        target = (x_pos(2), y_pos(137.7))
        label_pos = (target[0] + 65 * SCALE, y_pos(88.0))
        draw_arrow(
            draw,
            (target[0] + 30 * SCALE, target[1] + 18 * SCALE),
            (label_pos[0] + 18 * SCALE, label_pos[1] + 5 * SCALE),
        )
        draw.multiline_text(
            label_pos,
            "largest-degradation\nattribute",
            font=font(32, True),
            fill=ANNOTATION,
            spacing=8 * SCALE,
        )


def main() -> None:
    img = Image.new("RGB", (W * SCALE, H * SCALE), WHITE)
    draw = ImageDraw.Draw(img)

    centered(draw, (W // 2, 72), "Leave-one-Nf-out cross-configuration generalization", font(66, True))
    centered(
        draw,
        (W // 2, 140),
        "Performance degradation relative to the seen reference (%)",
        font(40),
        fill=MUTED,
    )

    draw_panel(
        draw,
        (65, 245, 1190, 1180),
        [
            "LoS/NLoS accuracy ↑",
            "Text factuality ↑",
            "Numeric accuracy ↑",
            "K-factor MAE ↓",
        ],
        "(a) Accuracy, factuality, numeric accuracy, and K-factor",
        (-1.5, 9.0),
        [-1, 0, 2, 4, 6, 8],
        legend_columns=2,
        annotation="left_stable",
    )
    draw_panel(
        draw,
        (1285, 245, 2410, 1180),
        ["First-angle MAE ↓", "First-delay MAE ↓", "First-power MAE ↓"],
        "(b) First-path angle, delay, and power",
        (0.0, 175.0),
        [0, 50, 100, 150],
        legend_columns=2,
        annotate_delay=True,
        annotation="right_delay",
    )
    img = img.resize((W, H), Image.Resampling.LANCZOS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FULL_PAPER_OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(PNG_OUT, dpi=(300, 300), optimize=True)
    img.save(PDF_OUT, "PDF", resolution=300.0)
    img.save(FULL_PAPER_PNG_OUT, dpi=(300, 300), optimize=True)
    img.save(FULL_PAPER_PDF_OUT, "PDF", resolution=300.0)
    print(PNG_OUT.resolve())
    print(PDF_OUT.resolve())
    print(FULL_PAPER_PNG_OUT.resolve())
    print(FULL_PAPER_PDF_OUT.resolve())


if __name__ == "__main__":
    main()
