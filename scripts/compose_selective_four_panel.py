#!/usr/bin/env python3
"""Compose the aggregate selective-prediction plots and the all-metric heatmap."""

from pathlib import Path

import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


ROOT = Path("/Users/yanrunhui/CSI_model")
SOURCE = ROOT / "tmp/pdfs/full_paper/figures/nlos_selective_column.png"
HEATMAP = Path("/Users/yanrunhui/Downloads/Codex 图像 2026年9月10日 22_32_53.png")
OUTPUT = ROOT / "tmp/pdfs/full_paper/figures/selective_prediction_four_panel.png"

CELL_W = 1800
CELL_H = 1280
CANVAS = (2 * CELL_W, 2 * CELL_H)
PADDING = 42

REGULAR_FONT = "/System/Library/Fonts/Supplemental/Times New Roman.ttf"
BOLD_FONT = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"
CALLOUT_COLOR = (180, 22, 38)

COVERAGES = np.asarray([0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
UNCERTAINTY_RISK = np.asarray(
    [
        [36.9, 39.9, 42.9, 47.8, 58.0, 100.0],
        [20.7, 31.6, 45.7, 61.2, 77.3, 100.0],
        [89.8, 90.0, 90.2, 91.6, 94.8, 100.0],
        [54.5, 61.1, 68.7, 74.9, 82.7, 100.0],
    ],
    dtype=float,
)
ORACLE_RISK = np.asarray(
    [
        [12.6, 16.2, 20.4, 26.1, 36.3, 100.0],
        [5.0, 8.7, 17.6, 30.5, 51.5, 100.0],
        [28.3, 35.7, 44.5, 55.4, 70.9, 100.0],
        [23.8, 29.7, 37.5, 47.8, 64.3, 100.0],
    ],
    dtype=float,
)

METRIC_LABELS = ["First-path delay", "First-path angle", "First-path power", "K-factor"]
METRIC_STYLES = [
    ((0, 114, 178), "circle"),
    ((213, 94, 0), "square"),
    ((0, 158, 115), "triangle"),
    ((204, 121, 167), "diamond"),
]


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


def draw_boxed_arrow(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    label_xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int],
    anchor: str = "la",
    circle: bool = False,
) -> None:
    """Draw a compact boxed annotation with an arrow pointing to a data feature."""
    left, top = label_xy
    bounds = draw.multiline_textbbox((left, top), text, font=font, spacing=4, anchor=anchor)
    pad_x, pad_y = 12, 8
    draw.rounded_rectangle(
        (bounds[0] - pad_x, bounds[1] - pad_y, bounds[2] + pad_x, bounds[3] + pad_y),
        radius=8,
        fill="white",
    )
    draw.multiline_text(label_xy, text, font=font, fill=color, spacing=4, anchor=anchor)
    start = ((bounds[0] + bounds[2]) // 2, bounds[3] + pad_y)
    if point[1] < bounds[1]:
        start = ((bounds[0] + bounds[2]) // 2, bounds[1] - pad_y)
    draw.line((start, point), fill=color, width=7)
    angle = math.atan2(point[1] - start[1], point[0] - start[0])
    arrow_len = 22
    arrow_half = 10
    back_x = point[0] - arrow_len * math.cos(angle)
    back_y = point[1] - arrow_len * math.sin(angle)
    perp_x = arrow_half * math.sin(angle)
    perp_y = -arrow_half * math.cos(angle)
    draw.polygon(
        [point, (back_x + perp_x, back_y + perp_y), (back_x - perp_x, back_y - perp_y)],
        fill=color,
    )
    if circle:
        radius = 18
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                     outline=color, width=7)


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
    # Panel (a): synthetic helper retained for old single-metric diagnostics.
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
    bin_risk = np.empty_like(UNCERTAINTY_RISK)
    bin_risk[:, 0] = UNCERTAINTY_RISK[:, 0]
    for index in range(1, len(COVERAGES)):
        bin_risk[:, index] = (
            COVERAGES[index] * UNCERTAINTY_RISK[:, index]
            - COVERAGES[index - 1] * UNCERTAINTY_RISK[:, index - 1]
        ) / (COVERAGES[index] - COVERAGES[index - 1])
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


def build_overall_risk_coverage_panel(width: int = 1600, height: int = 1120) -> Image.Image:
    """Plot cross-metric normalized MAE as grouped bars at selected coverage levels."""
    coverage_pct = COVERAGES * 100.0
    uncertainty = np.mean(UNCERTAINTY_RISK, axis=0)
    oracle = np.mean(ORACLE_RISK, axis=0)
    random = np.full_like(uncertainty, 100.0)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(BOLD_FONT, 62)
    subtitle_font = ImageFont.truetype(REGULAR_FONT, 40)
    axis_font = ImageFont.truetype(REGULAR_FONT, 48)
    tick_font = ImageFont.truetype(REGULAR_FONT, 40)
    legend_font = ImageFont.truetype(REGULAR_FONT, 36)

    left, right, top, bottom = 190, width - 70, 185, height - 165
    y_max = 112.0

    draw.text((width / 2, 28), "(a) Overall risk-coverage", font=title_font,
              fill=(25, 25, 25), anchor="ma")
    draw.text((width / 2, 100), "Mean normalized MAE across delay, angle, power, and K-factor",
              font=subtitle_font, fill=(80, 80, 80), anchor="ma")

    def ycoord(value: float) -> float:
        return bottom - value / y_max * (bottom - top)

    for value in (0, 25, 50, 75, 100):
        y = ycoord(value)
        draw.line((left, y, right, y), fill=(218, 222, 228), width=3)
        draw.text((left - 20, y), str(value), font=tick_font, fill=(55, 55, 55), anchor="rm")
    draw.line((left, top, left, bottom), fill=(40, 40, 40), width=5)
    draw.line((left, bottom, right, bottom), fill=(40, 40, 40), width=5)

    colors = [
        (0, 114, 178),
        (0, 158, 115),
        (230, 159, 0),
    ]
    series = [
        ("Ensemble-uncertainty", uncertainty),
        ("Oracle-ranking", oracle),
        ("Random-rejection", random),
    ]
    group_w = (right - left) / len(coverage_pct)
    bar_w = group_w * 0.22
    offsets = [-bar_w * 1.15, 0.0, bar_w * 1.15]
    for group_index, coverage in enumerate(coverage_pct):
        center = left + group_w * (group_index + 0.5)
        for series_index, (_, values) in enumerate(series):
            value = float(values[group_index])
            x0 = center + offsets[series_index] - bar_w / 2
            x1 = center + offsets[series_index] + bar_w / 2
            y = ycoord(value)
            draw.rectangle((x0, y, x1, bottom), fill=colors[series_index])
            draw.rectangle((x0, y, x1, bottom), outline="white", width=3)
        draw.text((center, bottom + 25), f"{int(coverage)}", font=tick_font,
                  fill=(45, 45, 45), anchor="ma")

    draw.text(((left + right) / 2, height - 58), "Coverage (%)",
              font=axis_font, fill=(30, 30, 30), anchor="mm")

    vertical = Image.new("RGBA", (690, 70), (255, 255, 255, 0))
    vertical_draw = ImageDraw.Draw(vertical)
    vertical_draw.text((345, 35), "Normalized MAE (%)", font=axis_font,
                       fill=(30, 30, 30, 255), anchor="mm")
    vertical = vertical.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(vertical, (30, round((top + bottom - vertical.height) / 2)), vertical)

    legend_x, legend_y = left + 15, top + 12
    items = [(color, label) for color, (label, _) in zip(colors, series)]
    legend_box = (
        legend_x - 18,
        legend_y - 32,
        legend_x + 1520,
        legend_y + 45,
    )
    draw.rectangle(legend_box, fill="white")
    for index, (color, label) in enumerate(items):
        x = legend_x + index * 500
        draw.rectangle((x, legend_y - 17, x + 58, legend_y + 17), fill=color)
        draw.text((x + 76, legend_y - 20), label, font=legend_font, fill=(25, 25, 25))

    note_font = ImageFont.truetype(BOLD_FONT, 40)
    coverage_index = 3  # 80% retained means the top 20% most uncertain samples are deferred.
    center = left + group_w * (coverage_index + 0.5)
    point = (round(center + offsets[0]), round(ycoord(float(uncertainty[coverage_index]))))
    draw_boxed_arrow(
        draw,
        point,
        (round(center + 95), round(ycoord(float(uncertainty[coverage_index])) - 155)),
        "Top 20% deferred:\n31% lower MAE",
        note_font,
        CALLOUT_COLOR,
        circle=True,
    )
    return image


def build_metric_line_panel(
    values: np.ndarray,
    title: str,
    subtitle: str,
    width: int = 1600,
    height: int = 1120,
) -> Image.Image:
    """Plot one ranking strategy as four per-metric normalized risk-coverage curves."""
    coverage_pct = COVERAGES * 100.0
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(BOLD_FONT, 62)
    subtitle_font = ImageFont.truetype(REGULAR_FONT, 40)
    axis_font = ImageFont.truetype(REGULAR_FONT, 48)
    tick_font = ImageFont.truetype(REGULAR_FONT, 40)
    legend_font = ImageFont.truetype(REGULAR_FONT, 36)

    left, right, top, bottom = 205, width - 95, 275, height - 165
    y_max = 112.0
    draw.text((width / 2, 28), title, font=title_font, fill=(25, 25, 25), anchor="ma")
    draw.text((width / 2, 100), subtitle, font=subtitle_font, fill=(80, 80, 80), anchor="ma")

    legend_x, legend_y = left + 15, 165
    legend_gap_x = 560
    legend_gap_y = 48
    for index, (metric, (color, shape)) in enumerate(zip(METRIC_LABELS, METRIC_STYLES)):
        x = legend_x + (index % 2) * legend_gap_x
        y = legend_y + (index // 2) * legend_gap_y
        draw.line((x, y, x + 68, y), fill=color, width=10)
        # Use the same marker language as the plotted line.
        marker_point = (x + 34, y)
        radius = 13
        if shape == "circle":
            draw.ellipse((marker_point[0] - radius, marker_point[1] - radius,
                          marker_point[0] + radius, marker_point[1] + radius),
                         fill=color, outline="white", width=3)
        elif shape == "square":
            draw.rectangle((marker_point[0] - radius, marker_point[1] - radius,
                            marker_point[0] + radius, marker_point[1] + radius),
                           fill=color, outline="white", width=3)
        elif shape == "triangle":
            points = [
                (marker_point[0], marker_point[1] - radius - 2),
                (marker_point[0] - radius - 2, marker_point[1] + radius),
                (marker_point[0] + radius + 2, marker_point[1] + radius),
            ]
            draw.polygon(points, fill=color)
            draw.line(points + [points[0]], fill="white", width=3)
        else:
            points = [
                (marker_point[0], marker_point[1] - radius - 2),
                (marker_point[0] - radius - 2, marker_point[1]),
                (marker_point[0], marker_point[1] + radius + 2),
                (marker_point[0] + radius + 2, marker_point[1]),
            ]
            draw.polygon(points, fill=color)
            draw.line(points + [points[0]], fill="white", width=3)
        draw.text((x + 84, y - 18), metric, font=legend_font, fill=(25, 25, 25))

    def xcoord(value: float) -> float:
        return left + (value - 50.0) / 50.0 * (right - left)

    def ycoord(value: float) -> float:
        return bottom - value / y_max * (bottom - top)

    for value in (0, 25, 50, 75, 100):
        y = ycoord(value)
        draw.line((left, y, right, y), fill=(218, 222, 228), width=3)
        draw.text((left - 20, y), str(value), font=tick_font, fill=(55, 55, 55), anchor="rm")
    baseline_y = ycoord(100.0)
    draw.line((left, baseline_y, right, baseline_y), fill=(90, 90, 90), width=4)
    draw.text((right - 8, baseline_y - 18), "Full coverage = 100%",
              font=subtitle_font, fill=(70, 70, 70), anchor="rs")

    draw.line((left, top, left, bottom), fill=(40, 40, 40), width=5)
    draw.line((left, bottom, right, bottom), fill=(40, 40, 40), width=5)

    def draw_marker(point: tuple[int, int], color: tuple[int, int, int], shape: str) -> None:
        x, y = point
        radius = 15
        if shape == "circle":
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=4)
        elif shape == "square":
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=4)
        elif shape == "triangle":
            points = [(x, y - radius - 2), (x - radius - 2, y + radius), (x + radius + 2, y + radius)]
            draw.polygon(points, fill=color)
            draw.line(points + [points[0]], fill="white", width=4, joint="curve")
        else:
            points = [(x, y - radius - 2), (x - radius - 2, y), (x, y + radius + 2), (x + radius + 2, y)]
            draw.polygon(points, fill=color)
            draw.line(points + [points[0]], fill="white", width=4, joint="curve")

    all_points: list[list[tuple[int, int]]] = []
    for metric_index, (metric, (color, shape)) in enumerate(zip(METRIC_LABELS, METRIC_STYLES)):
        row = values[metric_index]
        points = [(round(xcoord(x)), round(ycoord(y))) for x, y in zip(coverage_pct, row)]
        all_points.append(points)
        draw.line(points, fill=color, width=12, joint="curve")
        for point in points:
            draw_marker(point, color, shape)

    note_font = ImageFont.truetype(BOLD_FONT, 35)
    note_color = CALLOUT_COLOR
    if title.startswith("(c)"):
        draw_boxed_arrow(
            draw,
            all_points[2][3],
            (round(xcoord(56)), round(ycoord(111))),
            "Power remains\nhardest to reduce",
            note_font,
            note_color,
        )
    elif title.startswith("(d)"):
        draw_boxed_arrow(
            draw,
            all_points[1][0],
            (round(xcoord(57)), round(ycoord(88))),
            "Oracle ranking\nseparates low-risk\nsamples clearly",
            note_font,
            note_color,
        )

    for value in coverage_pct:
        draw.text((xcoord(value), bottom + 25), f"{int(value)}", font=tick_font,
                  fill=(45, 45, 45), anchor="ma")
    draw.text(((left + right) / 2, height - 58), "Coverage (%)",
              font=axis_font, fill=(30, 30, 30), anchor="mm")

    vertical = Image.new("RGBA", (1040, 70), (255, 255, 255, 0))
    vertical_draw = ImageDraw.Draw(vertical)
    vertical_draw.text((520, 35), "Normalized MAE (% of full-coverage MAE)",
                       font=axis_font, fill=(30, 30, 30, 255), anchor="mm")
    vertical = vertical.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(vertical, (58, round((top + bottom - vertical.height) / 2)), vertical)
    return image


def build_overall_deferral_panel(width: int = 1600, height: int = 1120) -> Image.Image:
    """Summarize overall error after deferring the highest-uncertainty bins."""
    deferred = 100.0 - COVERAGES[::-1] * 100.0
    uncertainty = np.mean(UNCERTAINTY_RISK, axis=0)[::-1]
    oracle = np.mean(ORACLE_RISK, axis=0)[::-1]
    random = np.full_like(uncertainty, 100.0)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(BOLD_FONT, 62)
    subtitle_font = ImageFont.truetype(REGULAR_FONT, 40)
    axis_font = ImageFont.truetype(REGULAR_FONT, 48)
    tick_font = ImageFont.truetype(REGULAR_FONT, 40)
    note_font = ImageFont.truetype(BOLD_FONT, 42)
    legend_font = ImageFont.truetype(REGULAR_FONT, 36)

    left, right, top, bottom = 185, width - 70, 185, height - 165
    y_max = 112.0
    draw.text((width / 2, 28), "(c) Overall deferral effect", font=title_font,
              fill=(25, 25, 25), anchor="ma")
    draw.text((width / 2, 100), "Highest-uncertainty samples are deferred first",
              font=subtitle_font, fill=(80, 80, 80), anchor="ma")

    def xcoord(value: float) -> float:
        return left + value / 50.0 * (right - left)

    def ycoord(value: float) -> float:
        return bottom - value / y_max * (bottom - top)

    for value in (0, 25, 50, 75, 100):
        y = ycoord(value)
        draw.line((left, y, right, y), fill=(218, 222, 228), width=3)
        draw.text((left - 20, y), str(value), font=tick_font, fill=(55, 55, 55), anchor="rm")
    baseline_y = ycoord(100.0)
    draw.line((left, baseline_y, right, baseline_y), fill=(90, 90, 90), width=4)
    baseline_label = "No deferral = 100%"
    baseline_label_xy = (right - 8, baseline_y - 48)
    baseline_bounds = draw.textbbox(baseline_label_xy, baseline_label, font=subtitle_font, anchor="rs")
    draw.rectangle((baseline_bounds[0] - 8, baseline_bounds[1] - 6,
                    baseline_bounds[2] + 8, baseline_bounds[3] + 6), fill="white")
    draw.text(baseline_label_xy, baseline_label,
              font=subtitle_font, fill=(70, 70, 70), anchor="rs")
    draw.line((left, top, left, bottom), fill=(40, 40, 40), width=5)
    draw.line((left, bottom, right, bottom), fill=(40, 40, 40), width=5)

    def draw_curve(values: np.ndarray, color: tuple[int, int, int], width_px: int) -> list[tuple[int, int]]:
        points = [(round(xcoord(x)), round(ycoord(y))) for x, y in zip(deferred, values)]
        draw.line(points, fill=color, width=width_px, joint="curve")
        for point in points:
            radius = 14
            draw.ellipse((point[0] - radius, point[1] - radius,
                          point[0] + radius, point[1] + radius),
                         fill=color, outline="white", width=4)
        return points

    points = draw_curve(uncertainty, (0, 114, 178), 15)
    draw_curve(random, (230, 159, 0), 9)
    draw_curve(oracle, (0, 158, 115), 12)

    for value in deferred:
        draw.text((xcoord(value), bottom + 25), f"{int(value)}", font=tick_font,
                  fill=(45, 45, 45), anchor="ma")
    draw.text(((left + right) / 2, height - 58), "Deferred highest-uncertainty samples (%)",
              font=axis_font, fill=(30, 30, 30), anchor="mm")
    vertical = Image.new("RGBA", (690, 70), (255, 255, 255, 0))
    vertical_draw = ImageDraw.Draw(vertical)
    vertical_draw.text((345, 35), "Normalized MAE (%)", font=axis_font,
                       fill=(30, 30, 30, 255), anchor="mm")
    vertical = vertical.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.paste(vertical, (30, round((top + bottom - vertical.height) / 2)), vertical)

    # The lower-left region is below both curves; the upper-left covered them.
    legend_x, legend_y = left + 35, bottom - 175
    legend_box = (
        legend_x - 18,
        legend_y - 32,
        legend_x + 520,
        legend_y + 2 * 48 + 35,
    )
    draw.rectangle(legend_box, fill="white")
    for index, (color, label) in enumerate([
        ((0, 114, 178), "Ensemble uncertainty"),
        ((230, 159, 0), "Random rejection"),
        ((0, 158, 115), "Oracle ranking"),
    ]):
        y = legend_y + index * 48
        draw.line((legend_x, y, legend_x + 70, y), fill=color, width=11)
        draw.text((legend_x + 86, y - 18), label, font=legend_font, fill=(25, 25, 25))

    point_20 = points[2]
    draw_callout(draw, point_20, (900, 330),
                 f"20% deferred:\n{uncertainty[2]:.0f}% normalized MAE", note_font)
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
    panel_a = build_overall_risk_coverage_panel(width=1600, height=1120)
    panel_b = build_normalized_error_uncertainty_panel(width=1600, height=1120)
    panel_c = build_metric_line_panel(
            UNCERTAINTY_RISK,
            "(c) Uncertainty ranking",
            "Per-metric normalized MAE after retaining the least-uncertain samples",
            width=1600,
            height=1120,
    )
    panel_d = build_metric_line_panel(
            ORACLE_RISK,
            "(d) Oracle ranking",
            "Per-metric normalized MAE under realized-error oracle ordering",
            width=1600,
            height=1120,
    )
    canvas = Image.new("RGB", CANVAS, "white")
    paste_centered(canvas, trim_white(panel_a), 0, 0, CELL_H)
    paste_centered(canvas, trim_white(panel_b), CELL_W, 0, CELL_H)
    paste_centered(canvas, trim_white(panel_c), 0, CELL_H, CELL_H)
    paste_centered(canvas, trim_white(panel_d), CELL_W, CELL_H, CELL_H)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, dpi=(300, 300), optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
