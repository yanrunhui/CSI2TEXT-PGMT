#!/usr/bin/env python3
"""Compose the three NLoS selective-prediction plots and the all-metric heatmap."""

from pathlib import Path

import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


ROOT = Path("/Users/yanrunhui/CSI_model")
SOURCE = ROOT / "tmp/pdfs/full_paper/figures/nlos_selective_column.png"
HEATMAP = Path("/Users/yanrunhui/Downloads/Codex 图像 2026年9月10日 22_32_53.png")
OUTPUT = ROOT / "tmp/pdfs/full_paper/figures/selective_prediction_four_panel.png"

CELL_W = 1800
TOP_H = 1500
BOTTOM_H = 1880
CANVAS = (2 * CELL_W, TOP_H + BOTTOM_H)
PADDING = 42

REGULAR_FONT = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
BOLD_FONT = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"
CALLOUT_COLOR = (180, 22, 38)


def draw_callout(draw: ImageDraw.ImageDraw, point: tuple[int, int], label_xy: tuple[int, int],
                 text: str, font: ImageFont.FreeTypeFont) -> None:
    """Draw a compact publication-style arrow annotation and emphasize its node."""
    left, top = label_xy
    bounds = draw.textbbox((left, top), text, font=font)
    pad = 8
    draw.rectangle(
        (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad),
        fill="white",
    )
    draw.text(label_xy, text, font=font, fill=CALLOUT_COLOR)
    start = ((bounds[0] + bounds[2]) // 2, bounds[3] + pad)
    draw.line((start, point), fill=CALLOUT_COLOR, width=7)
    angle = math.atan2(point[1] - start[1], point[0] - start[0])
    arrow_len = 19
    arrow_half = 8
    back_x = point[0] - arrow_len * math.cos(angle)
    back_y = point[1] - arrow_len * math.sin(angle)
    perp_x = arrow_half * math.sin(angle)
    perp_y = -arrow_half * math.cos(angle)
    draw.polygon(
        [point, (back_x + perp_x, back_y + perp_y), (back_x - perp_x, back_y - perp_y)],
        fill=CALLOUT_COLOR,
    )
    radius = 17
    draw.ellipse(
        (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
        outline=CALLOUT_COLOR,
        width=7,
    )


def enhance_line_panels(image: Image.Image) -> Image.Image:
    """Thicken the colored line work and label the key node in panel (a)."""
    rgb = image.convert("RGB")
    values = np.asarray(rgb).copy()
    top = values[: rgb.height // 2]
    target_colors = ((31, 119, 180), (255, 127, 14), (44, 160, 44))
    for color in target_colors:
        color_array = np.asarray(color, dtype=np.int16)
        distance = np.max(np.abs(top.astype(np.int16) - color_array), axis=2)
        mask = Image.fromarray((distance <= 28).astype(np.uint8) * 255)
        expanded = np.asarray(mask.filter(ImageFilter.MaxFilter(5))) > 0
        top[expanded] = color
    values[: rgb.height // 2] = top
    enhanced = Image.fromarray(values, mode="RGB")

    draw = ImageDraw.Draw(enhanced)
    font = ImageFont.truetype(BOLD_FONT, 55)
    # Panel (a): retaining 80% coverage lowers NLoS delay risk from about 31 to 16.7 ns.
    draw_callout(draw, (950, 652), (590, 465), "20% deferred: -46% MAE", font)
    return enhanced


def trim_white(image: Image.Image, margin: int = 22) -> Image.Image:
    """Remove white margins while retaining a small publication-safe border."""
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, "white")
    difference = ImageChops.difference(rgb, background).convert("L")
    mask = difference.point(lambda value: 255 if value > 10 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return rgb
    left, top, right, bottom = bbox
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(rgb.width, right + margin)
    bottom = min(rgb.height, bottom + margin)
    return rgb.crop((left, top, right, bottom))


def fit(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def build_normalized_error_uncertainty_panel(width: int = 1600, height: int = 1120) -> Image.Image:
    """Plot cross-metric error against uncertainty rank on percentage axes.

    Each row below is the cumulative MAE after retaining the least-uncertain
    samples, normalized by that metric's full-coverage MAE.  Differencing the
    cumulative risks yields the MAE within successive uncertainty-percentile
    bins; the plotted curve is the mean over delay, angle, power, and K-factor.
    """
    cumulative_risk = np.asarray(
        [
            [36.9, 39.9, 42.9, 47.8, 58.0, 100.0],
            [20.7, 31.6, 45.7, 61.2, 77.3, 100.0],
            [89.8, 90.0, 90.2, 91.6, 94.8, 100.0],
            [54.5, 61.1, 68.7, 74.9, 82.7, 100.0],
        ],
        dtype=float,
    )
    coverages = np.asarray([0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    bin_risk = np.empty_like(cumulative_risk)
    bin_risk[:, 0] = cumulative_risk[:, 0]
    for index in range(1, len(coverages)):
        bin_risk[:, index] = (
            coverages[index] * cumulative_risk[:, index]
            - coverages[index - 1] * cumulative_risk[:, index - 1]
        ) / (coverages[index] - coverages[index - 1])
    mean_risk = np.mean(bin_risk, axis=0)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(BOLD_FONT, 62)
    subtitle_font = ImageFont.truetype(REGULAR_FONT, 40)
    axis_font = ImageFont.truetype(REGULAR_FONT, 48)
    tick_font = ImageFont.truetype(REGULAR_FONT, 40)
    note_font = ImageFont.truetype(BOLD_FONT, 42)

    left, right, top, bottom = 170, width - 65, 185, height - 165
    y_max = 350.0

    draw.text((width / 2, 28), "(b) Error vs. uncertainty rank", font=title_font,
              fill=(25, 25, 25), anchor="ma")
    draw.text((width / 2, 100), "Mean across delay, angle, power, and K-factor",
              font=subtitle_font, fill=(80, 80, 80), anchor="ma")

    def xcoord(index: int) -> float:
        return left + index * (right - left) / (len(mean_risk) - 1)

    def ycoord(value: float) -> float:
        return bottom - value / y_max * (bottom - top)

    for value in (0, 100, 200, 300):
        y = ycoord(value)
        draw.line((left, y, right, y), fill=(218, 222, 228), width=3)
        draw.text((left - 20, y), str(value), font=tick_font, fill=(55, 55, 55), anchor="rm")
    baseline_y = ycoord(100.0)
    draw.line((left, baseline_y, right, baseline_y), fill=(90, 90, 90), width=4)
    draw.text((right - 8, baseline_y - 13), "Full-coverage MAE = 100%",
              font=subtitle_font, fill=(70, 70, 70), anchor="rs")

    draw.line((left, top, left, bottom), fill=(40, 40, 40), width=5)
    draw.line((left, bottom, right, bottom), fill=(40, 40, 40), width=5)

    labels = ["0-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
    points = [(round(xcoord(index)), round(ycoord(value))) for index, value in enumerate(mean_risk)]
    draw.line(points, fill=(0, 114, 178), width=17, joint="curve")
    for point in points:
        radius = 16
        draw.ellipse((point[0] - radius, point[1] - radius,
                      point[0] + radius, point[1] + radius),
                     fill=(0, 114, 178), outline="white", width=4)
    for index, label in enumerate(labels):
        draw.text((xcoord(index), bottom + 25), label, font=tick_font,
                  fill=(45, 45, 45), anchor="ma")

    draw.text(((left + right) / 2, height - 58), "Uncertainty percentile bin (%)",
              font=axis_font, fill=(30, 30, 30), anchor="mm")
    vertical = Image.new("RGBA", (620, 70), (255, 255, 255, 0))
    vertical_draw = ImageDraw.Draw(vertical)
    vertical_draw.text((310, 35), "Normalized MAE (%)", font=axis_font,
                       fill=(30, 30, 30, 255), anchor="mm")
    vertical = vertical.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(vertical, (28, round((top + bottom - vertical.height) / 2)), vertical)

    final_point = points[-1]
    label = f"Top 10%: {mean_risk[-1]:.0f}% error"
    draw_callout(draw, final_point, (900, 225), label, note_font)
    return image


def paste_centered(canvas: Image.Image, image: Image.Image, cell_x: int, cell_y: int,
                   cell_height: int, top_offset: int = 0) -> None:
    available_w = CELL_W - 2 * PADDING
    available_h = cell_height - 2 * PADDING - top_offset
    resized = fit(image, available_w, available_h)
    x = cell_x + (CELL_W - resized.width) // 2
    y = cell_y + PADDING + top_offset + (available_h - resized.height) // 2
    canvas.paste(resized, (x, y))


def relocate_panel_c_legend(panel: Image.Image) -> Image.Image:
    """Move the bar-chart legend above the title and restore the covered bars."""
    panel = panel.copy()
    draw = ImageDraw.Draw(panel)

    # Remove the original in-axes legend. The legend covered the tops of the two
    # right orange bars, so redraw those small bar segments after restoring the grid.
    draw.rectangle((1080, 150, 1650, 270), fill="white")
    draw.line((1080, 209, 1650, 209), fill=(205, 205, 205), width=2)
    orange = (255, 127, 14)
    draw.rectangle((1167, 184, 1290, 270), fill=orange)
    draw.rectangle((1489, 184, 1612, 270), fill=orange)

    # Use one compact horizontal line in the otherwise empty top margin.
    legend_font = ImageFont.truetype(REGULAR_FONT, 48)
    y = 4
    x = 385
    draw.rectangle((x, y + 9, x + 62, y + 38), fill=(31, 119, 180))
    draw.text((x + 70, y), "Ensemble uncertainty", fill=(20, 20, 20), font=legend_font)
    x = 1120
    draw.rectangle((x, y + 9, x + 62, y + 38), fill=orange)
    draw.text((x + 70, y), "Random rejection", fill=(20, 20, 20), font=legend_font)
    return panel


def interpolate_color(value: float) -> tuple[int, int, int]:
    """Approximate Matplotlib's RdYlGn_r palette without a plotting dependency."""
    stops = [
        (0, (0, 104, 55)),
        (10, (26, 152, 80)),
        (20, (102, 189, 99)),
        (30, (166, 217, 106)),
        (40, (217, 239, 139)),
        (50, (255, 255, 191)),
        (60, (254, 224, 139)),
        (70, (253, 174, 97)),
        (80, (244, 109, 67)),
        (90, (215, 48, 39)),
        (100, (165, 0, 38)),
    ]
    value = min(100.0, max(0.0, value))
    for (low, c0), (high, c1) in zip(stops, stops[1:]):
        if value <= high:
            fraction = (value - low) / (high - low)
            return tuple(round(a + fraction * (b - a)) for a, b in zip(c0, c1))
    return stops[-1][1]


def centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str,
                  font: ImageFont.FreeTypeFont, fill: tuple[int, int, int]) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        ((left + right - width) / 2, (top + bottom - height) / 2 - bounds[1]),
        text,
        font=font,
        fill=fill,
    )


def build_square_heatmap(width: int, height: int) -> Image.Image:
    """Rebuild panel (d) with square cells and larger, publication-readable values."""
    # Values are copied verbatim from the user-supplied six-coverage heatmap.
    uncertainty = [
        [36.9, 39.9, 42.9, 47.8, 58.0, 100.0],
        [20.7, 31.6, 45.7, 61.2, 77.3, 100.0],
        [89.8, 90.0, 90.2, 91.6, 94.8, 100.0],
        [54.5, 61.1, 68.7, 74.9, 82.7, 100.0],
    ]
    oracle = [
        [12.6, 16.2, 20.4, 26.1, 36.3, 100.0],
        [5.0, 8.7, 17.6, 30.5, 51.5, 100.0],
        [28.3, 35.7, 44.5, 55.4, 70.9, 100.0],
        [23.8, 29.7, 37.5, 47.8, 64.3, 100.0],
    ]
    metrics = ["First-path delay", "First-path angle", "First-path power", "K-factor"]
    coverages = [50, 60, 70, 80, 90, 100]

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    heading_font = ImageFont.truetype(BOLD_FONT, 42)
    label_font = ImageFont.truetype(REGULAR_FONT, 36)
    value_font = ImageFont.truetype(REGULAR_FONT, 54)
    value_bold = ImageFont.truetype(BOLD_FONT, 54)
    tick_font = ImageFont.truetype(REGULAR_FONT, 36)
    axis_font = ImageFont.truetype(REGULAR_FONT, 40)
    colorbar_font = ImageFont.truetype(REGULAR_FONT, 30)

    cell = 180
    grid_x = 300
    first_y = 50
    second_y = 850
    separator = 4

    def draw_table(values: list[list[float]], y: int, heading: str) -> None:
        draw.text((grid_x, y - 48), heading, font=heading_font, fill=(25, 25, 25))
        for row, (metric, row_values) in enumerate(zip(metrics, values)):
            cell_y = y + row * cell
            bounds = draw.textbbox((0, 0), metric, font=label_font)
            draw.text(
                (grid_x - 18 - (bounds[2] - bounds[0]), cell_y + (cell - (bounds[3] - bounds[1])) / 2 - bounds[1]),
                metric,
                font=label_font,
                fill=(25, 25, 25),
            )
            for col, value in enumerate(row_values):
                cell_x = grid_x + col * cell
                color = interpolate_color(value)
                draw.rectangle(
                    (cell_x + separator // 2, cell_y + separator // 2,
                     cell_x + cell - separator // 2, cell_y + cell - separator // 2),
                    fill=color,
                )
                luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
                text_color = (255, 255, 255) if luminance < 120 else (25, 25, 25)
                font = value_bold if col == len(row_values) - 1 else value_font
                centered_text(
                    draw,
                    (cell_x, cell_y, cell_x + cell, cell_y + cell),
                    f"{value:.1f}",
                    font,
                    text_color,
                )

    draw_table(uncertainty, first_y, "Uncertainty ranking")
    draw_table(oracle, second_y, "Oracle ranking")

    grid_bottom = second_y + 4 * cell
    for col, coverage in enumerate(coverages):
        box = (grid_x + col * cell, grid_bottom + 4, grid_x + (col + 1) * cell, grid_bottom + 46)
        centered_text(draw, box, str(coverage), tick_font, (25, 25, 25))
    centered_text(
        draw,
        (grid_x, grid_bottom + 45, grid_x + 6 * cell, height - 2),
        "Coverage (%)",
        axis_font,
        (25, 25, 25),
    )

    # Compact color bar leaves the heatmap itself as the dominant visual element.
    bar_x, bar_y, bar_w, bar_h = 1420, 255, 48, 1050
    for offset in range(bar_h):
        value = 100.0 * (bar_h - 1 - offset) / (bar_h - 1)
        draw.line((bar_x, bar_y + offset, bar_x + bar_w, bar_y + offset), fill=interpolate_color(value))
    for value in range(0, 101, 20):
        y = bar_y + bar_h - round(value / 100 * bar_h)
        draw.line((bar_x + bar_w, y, bar_x + bar_w + 10, y), fill=(30, 30, 30), width=2)
        label = str(value)
        bounds = draw.textbbox((0, 0), label, font=colorbar_font)
        draw.text((bar_x + bar_w + 16, y - (bounds[3] - bounds[1]) / 2 - bounds[1]), label,
                  font=colorbar_font, fill=(25, 25, 25))

    vertical = Image.new("RGBA", (760, 58), (255, 255, 255, 0))
    vertical_draw = ImageDraw.Draw(vertical)
    vertical_draw.text(
        (0, 0),
        "Normalized MAE (% of full-coverage MAE)  Lower is better",
        font=axis_font,
        fill=(25, 25, 25, 255),
    )
    vertical = vertical.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(vertical, (1560, (height - vertical.height) // 2), vertical)
    return image


def main() -> None:
    source = enhance_line_panels(Image.open(SOURCE).convert("RGB"))
    width, height = source.size

    # The original Figure 7 is a 2+1 layout. Split it into three independent panels.
    panel_a = trim_white(source.crop((0, 0, width // 2, height // 2)))
    panel_b = trim_white(build_normalized_error_uncertainty_panel())
    panel_c_raw = source.crop((width // 5, height // 2, 4 * width // 5, height))
    panel_c = trim_white(relocate_panel_c_legend(panel_c_raw))

    # Validate that the requested replacement image is present, then rebuild it from
    # its displayed values so the cells can be truly square rather than bitmap-stretched.
    with Image.open(HEATMAP) as supplied_heatmap:
        supplied_heatmap.verify()
    heat = build_square_heatmap(CELL_W - 2 * PADDING, BOTTOM_H - 2 * PADDING - 105)

    canvas = Image.new("RGB", CANVAS, "white")
    paste_centered(canvas, panel_a, 0, 0, TOP_H)
    paste_centered(canvas, panel_b, CELL_W, 0, TOP_H)
    paste_centered(canvas, panel_c, 0, TOP_H, BOTTOM_H)

    title_height = 105
    paste_centered(canvas, heat, CELL_W, TOP_H, BOTTOM_H, top_offset=title_height)
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(
        BOLD_FONT, 70
    )
    draw.text(
        (CELL_W + PADDING, TOP_H + 28),
        "(d) All-sample normalized MAE",
        fill=(25, 25, 25),
        font=title_font,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, dpi=(300, 300), optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
