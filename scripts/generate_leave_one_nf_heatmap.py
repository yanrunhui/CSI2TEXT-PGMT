#!/usr/bin/env python3
"""Generate a leave-one-Nf-out generalization heatmap from reported results."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path("output/figures/leave_one_nf_out_generalization_heatmap.png")
W, H = 3300, 2350

FONT_REGULAR = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"

INK = "#20242A"
MUTED = "#5D6470"
GRID = "#B8BDC5"
WHITE = "#FFFFFF"

HELD_OUT_NF = [64, 96, 128, 192, 256]

# Values transcribed from the leave-one-Nf-out table supplied by the user.
ROWS = [
    {
        "label": "LoS/NLoS accuracy ↑",
        "baseline_label": "seen reference: 99.58%",
        "baseline": 99.58,
        "higher_is_better": True,
        "values": [99.88, 99.84, 99.83, 99.76, 99.81],
    },
    {
        "label": "Text factuality ↑",
        "baseline_label": "seen reference: 96.43%",
        "baseline": 96.43,
        "higher_is_better": True,
        "values": [96.54, 96.32, 95.67, 96.55, 95.74],
    },
    {
        "label": "Numeric accuracy ↑",
        "baseline_label": "seen reference: 82.61%",
        "baseline": 82.61,
        "higher_is_better": True,
        "values": [79.88, 78.83, 78.64, 78.37, 76.39],
    },
    {
        "label": "First-angle MAE ↓",
        "baseline_label": "seen reference: 16.61°",
        "baseline": 16.61,
        "higher_is_better": False,
        "values": [17.73, 18.23, 18.48, 19.21, 20.11],
    },
    {
        "label": "First-delay MAE ↓",
        "baseline_label": "seen reference: 21.91 ns",
        "baseline": 21.91,
        "higher_is_better": False,
        "values": [44.00, 48.60, 52.09, 50.28, 57.11],
    },
    {
        "label": "First-power MAE ↓",
        "baseline_label": "seen reference: 4.34 dB",
        "baseline": 4.34,
        "higher_is_better": False,
        "values": [5.07, 5.44, 6.00, 6.17, 6.28],
    },
    {
        "label": "K-factor MAE ↓",
        "baseline_label": "seen reference: 2.15 dB",
        "baseline": 2.15,
        "higher_is_better": False,
        "values": [2.16, 2.28, 2.17, 2.16, 2.19],
    },
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def degradation(value: float, baseline: float, higher_is_better: bool) -> float:
    if higher_is_better:
        return (baseline - value) / baseline * 100.0
    return (value - baseline) / baseline * 100.0


def cell_style(value: float) -> tuple[str, str]:
    """Discrete diverging scale; negative values are improvements."""
    if value <= -1.0:
        return "#2166AC", WHITE
    if value < 0.0:
        return "#92C5DE", INK
    if value < 2.0:
        return "#F7F7F7", INK
    if value < 5.0:
        return "#FEE8C8", INK
    if value < 20.0:
        return "#FDBB84", INK
    if value < 50.0:
        return "#FC8D59", INK
    return "#B2182B", WHITE


def centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt, fill=INK) -> None:
    box = draw.textbbox((0, 0), text, font=fnt)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=fnt, fill=fill)


def main() -> None:
    img = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(img)

    centered(draw, (W / 2, 82), "Leave-one-Nf-out Cross-Configuration Generalization", font(78, True))
    centered(
        draw,
        (W / 2, 158),
        "Cell values: performance degradation relative to the available seen Nf = 128 reference",
        font(38),
        fill=MUTED,
    )

    label_w = 720
    cell_w = 430
    table_left = (W - label_w - cell_w * len(HELD_OUT_NF)) / 2
    table_top = 355
    header_h = 185
    row_h = 190
    table_right = table_left + label_w + cell_w * len(HELD_OUT_NF)

    centered(draw, (table_left + label_w / 2, table_top - 66), "Metric and seen reference", font(42, True))
    centered(draw, ((table_left + label_w + table_right) / 2, table_top - 64), "Held-out test Nf", font(45, True))

    # Column headers.
    draw.rectangle((table_left, table_top, table_left + label_w, table_top + header_h), fill="#ECEFF3", outline=GRID, width=3)
    centered(draw, (table_left + label_w / 2, table_top + header_h / 2), "Evaluation metric", font(42, True))
    for col, nf in enumerate(HELD_OUT_NF):
        x0 = table_left + label_w + col * cell_w
        x1 = x0 + cell_w
        draw.rectangle((x0, table_top, x1, table_top + header_h), fill="#ECEFF3", outline=GRID, width=3)
        centered(draw, ((x0 + x1) / 2, table_top + 64), str(nf), font(56, True))
        centered(draw, ((x0 + x1) / 2, table_top + 126), "train on other four Nf", font(27), fill=MUTED)

    # Heatmap rows.
    for row_idx, row in enumerate(ROWS):
        y0 = table_top + header_h + row_idx * row_h
        y1 = y0 + row_h
        draw.rectangle((table_left, y0, table_left + label_w, y1), fill=WHITE, outline=GRID, width=3)
        centered(draw, (table_left + label_w / 2, y0 + 67), row["label"], font(40, True))
        centered(draw, (table_left + label_w / 2, y0 + 127), row["baseline_label"], font(30), fill=MUTED)

        for col_idx, raw_value in enumerate(row["values"]):
            drop = degradation(raw_value, row["baseline"], row["higher_is_better"])
            fill, text_fill = cell_style(drop)
            x0 = table_left + label_w + col_idx * cell_w
            x1 = x0 + cell_w
            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=WHITE, width=5)
            centered(draw, ((x0 + x1) / 2, (y0 + y1) / 2 - 5), f"{drop:+.1f}%", font(47, True), fill=text_fill)

    # Discrete legend.
    legend_y = 2025
    legend_items = [
        ("#2166AC", "≤−1"),
        ("#92C5DE", "−1–0"),
        ("#F7F7F7", "0–2"),
        ("#FEE8C8", "2–5"),
        ("#FDBB84", "5–20"),
        ("#FC8D59", "20–50"),
        ("#B2182B", "≥50"),
    ]
    draw.text((table_left, legend_y + 12), "Degradation (%)", font=font(33, True), fill=INK)
    lx = table_left + 390
    for color, label in legend_items:
        draw.rectangle((lx, legend_y, lx + 105, legend_y + 58), fill=color, outline=GRID, width=2)
        centered(draw, (lx + 52, legend_y + 86), label, font(27), fill=MUTED)
        lx += 165
    draw.text((lx + 25, legend_y + 13), "blue = improvement", font=font(30), fill=MUTED)

    formula = (
        "Accuracy/quality: (seen − held-out) / seen.  MAE: (held-out − seen) / seen.  "
        "Positive values indicate worse held-out performance."
    )
    centered(draw, (W / 2, 2265), formula, font(31), fill=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, dpi=(300, 300), optimize=True)
    print(OUT.resolve())


if __name__ == "__main__":
    main()
