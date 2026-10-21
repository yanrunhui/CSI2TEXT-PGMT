#!/usr/bin/env python3
"""Create a publication-ready 2x2 regression scatter figure without matplotlib."""

from __future__ import annotations

import csv
import html
import json
import math
import random
from pathlib import Path


SOURCE = Path("/Users/yanrunhui/Downloads/regression_pairs.csv")
OUTPUT_SVG = Path("output/figures/regression_prediction_scatter_500.svg")
SAMPLE_SIZE = 500
SEED = 20260909

WIDTH = 2200
HEIGHT = 2160

LOS_COLOR = "#2F6FB0"
NLOS_COLOR = "#E4572E"
TEXT_COLOR = "#20242A"
MUTED_COLOR = "#5D6470"
GRID_COLOR = "#D9DEE7"
FRAME_COLOR = "#9CA3AF"
DEVIATION_COLOR = "#C62828"


METRICS = (
    {
        "key": "delay",
        "title": "(a) First-path delay",
        "gt": "first_path_delay_ns_ground_truth",
        "pred": "first_path_delay_ns_prediction",
        "unit": "ns",
        "circular": False,
        "nlos_deviation_box": (200.0, 700.0, 700.0, 1050.0),
        "nlos_deviation_polygon": None,
    },
    {
        "key": "angle",
        "title": "(b) First-path angle",
        "gt": "first_path_angle_deg_ground_truth",
        "pred": "first_path_angle_deg_prediction",
        "unit": "°",
        "circular": True,
        "nlos_deviation_box": (-120.0, 150.0, -165.0, -110.0),
        "nlos_deviation_polygon": None,
    },
    {
        "key": "power",
        "title": "(c) First-path power",
        "gt": "first_path_power_dbw_ground_truth",
        "pred": "first_path_power_dbw_prediction",
        "unit": "dBW",
        "circular": False,
        "nlos_deviation_box": None,
        "nlos_deviation_polygon": (
            (-225.0, -200.0),
            (-135.0, -110.0),
            (-125.0, -120.0),
            (-215.0, -210.0),
        ),
    },
    {
        "key": "kfactor",
        "title": "(d) K-factor",
        "gt": "k_factor_db_ground_truth",
        "pred": "k_factor_db_prediction",
        "unit": "dB",
        "circular": False,
        "nlos_deviation_box": None,
        "nlos_deviation_polygon": None,
    },
)


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        parsed = {
            "sample_index": int(row["sample_index"]),
            "los_status": row["los_status"].lower(),
        }
        for metric in METRICS:
            parsed[metric["gt"]] = float(row[metric["gt"]])
            parsed[metric["pred"]] = float(row[metric["pred"]])
        result.append(parsed)
    return result


def stratified_sample(rows: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    groups = {
        status: [row for row in rows if row["los_status"] == status]
        for status in ("los", "nlos")
    }
    n_los = round(SAMPLE_SIZE * len(groups["los"]) / len(rows))
    sampled = rng.sample(groups["los"], n_los)
    sampled += rng.sample(groups["nlos"], SAMPLE_SIZE - n_los)
    return sorted(sampled, key=lambda row: row["sample_index"])


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for pos in range(start, end):
            ranks[order[pos]] = rank
        start = end
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    dx = [value - mean_x for value in x]
    dy = [value - mean_y for value in y]
    numerator = sum(a * b for a, b in zip(dx, dy))
    denominator = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    return numerator / denominator


def metric_stats(rows: list[dict], metric: dict) -> tuple[float, float, float]:
    gt = [row[metric["gt"]] for row in rows]
    pred = [row[metric["pred"]] for row in rows]
    if metric["circular"]:
        errors = [abs((p - g + 180.0) % 360.0 - 180.0) for g, p in zip(gt, pred)]
    else:
        errors = [abs(p - g) for g, p in zip(gt, pred)]
    return (
        sum(errors) / len(errors),
        pearson(gt, pred),
        pearson(average_ranks(gt), average_ranks(pred)),
    )


def nice_step(span: float, target_ticks: int = 5) -> float:
    rough = span / max(target_ticks, 1)
    exponent = math.floor(math.log10(max(rough, 1e-12)))
    fraction = rough / (10**exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10**exponent)


def nice_domain(values: list[float], is_angle: bool) -> tuple[float, float, float]:
    low, high = min(values), max(values)
    span = max(high - low, 1.0)
    low -= span * 0.045
    high += span * 0.045
    if is_angle:
        low = min(low, -180.0)
        high = max(high, 180.0)
    step = nice_step(high - low)
    low = math.floor(low / step) * step
    high = math.ceil(high / step) * step
    return low, high, step


def tick_label(value: float, step: float) -> str:
    if step >= 1:
        return f"{value:.0f}"
    decimals = max(1, -math.floor(math.log10(step)))
    return f"{value:.{decimals}f}"


def svg_text(x: float, y: float, text: str, **attrs) -> str:
    attributes = " ".join(f'{key.replace("_", "-")}="{html.escape(str(value))}"' for key, value in attrs.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {attributes}>{html.escape(text)}</text>'


def draw_panel(rows: list[dict], all_rows: list[dict], metric: dict, panel_x: int, panel_y: int) -> list[str]:
    panel_w = 1060
    plot_size = 760
    plot_x = panel_x + 145
    plot_y = panel_y + 125
    title_y = panel_y + 32

    all_values = [row[metric["gt"]] for row in rows] + [row[metric["pred"]] for row in rows]
    low, high, step = nice_domain(all_values, metric["key"] == "angle")

    def sx(value: float) -> float:
        return plot_x + (value - low) / (high - low) * plot_size

    def sy(value: float) -> float:
        return plot_y + plot_size - (value - low) / (high - low) * plot_size

    mae, pearson_r, spearman_rho = metric_stats(all_rows, metric)
    out = [
        svg_text(panel_x + panel_w / 2, title_y, metric["title"],
                 fill=TEXT_COLOR, font_size=48, font_weight=600, text_anchor="middle"),
        svg_text(plot_x + 12, panel_y + 72, f'MAE = {mae:.2f} {metric["unit"]}',
                 fill=MUTED_COLOR, font_size=36),
        svg_text(plot_x + 12, panel_y + 108, f'Pearson r = {pearson_r:.3f}    Spearman ρ = {spearman_rho:.3f}',
                 fill=MUTED_COLOR, font_size=36),
    ]

    tick = math.ceil(low / step) * step
    while tick <= high + step * 1e-8:
        px, py = sx(tick), sy(tick)
        out.append(f'<line x1="{px:.1f}" y1="{plot_y}" x2="{px:.1f}" y2="{plot_y + plot_size}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        out.append(f'<line x1="{plot_x}" y1="{py:.1f}" x2="{plot_x + plot_size}" y2="{py:.1f}" stroke="{GRID_COLOR}" stroke-width="1"/>')
        label = tick_label(tick, step)
        out.append(svg_text(px, plot_y + plot_size + 43, label, fill=MUTED_COLOR, font_size=36, text_anchor="middle"))
        out.append(svg_text(plot_x - 18, py + 11, label, fill=MUTED_COLOR, font_size=36, text_anchor="end"))
        tick += step

    out.append(f'<rect x="{plot_x}" y="{plot_y}" width="{plot_size}" height="{plot_size}" fill="none" stroke="{FRAME_COLOR}" stroke-width="1.4"/>')
    diag_low = max(low, low)
    diag_high = min(high, high)
    out.append(f'<line x1="{sx(diag_low):.1f}" y1="{sy(diag_low):.1f}" x2="{sx(diag_high):.1f}" y2="{sy(diag_high):.1f}" stroke="#111111" stroke-width="2.2" stroke-dasharray="12 9"/>')

    for status, color in (("nlos", NLOS_COLOR), ("los", LOS_COLOR)):
        for row in rows:
            if row["los_status"] != status:
                continue
            out.append(
                f'<circle cx="{sx(row[metric["gt"]]):.1f}" cy="{sy(row[metric["pred"]]):.1f}" '
                f'r="4.5" fill="{color}" fill-opacity="0.64" stroke="#FFFFFF" stroke-opacity="0.70" stroke-width="0.8"/>'
            )

    deviation_box = metric["nlos_deviation_box"]
    if deviation_box is not None:
        x_min, x_max, y_min, y_max = deviation_box
        box_x = sx(x_min)
        box_y = sy(y_max)
        box_w = sx(x_max) - box_x
        box_h = sy(y_min) - box_y
        out.append(
            f'<rect x="{box_x:.1f}" y="{box_y:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" '
            f'fill="none" stroke="{DEVIATION_COLOR}" stroke-width="5.5"/>'
        )

    deviation_polygon = metric["nlos_deviation_polygon"]
    if deviation_polygon is not None:
        polygon_points = " ".join(
            f"{sx(x_value):.1f},{sy(y_value):.1f}"
            for x_value, y_value in deviation_polygon
        )
        out.append(
            f'<polygon points="{polygon_points}" fill="none" '
            f'stroke="{DEVIATION_COLOR}" stroke-width="5.5" stroke-linejoin="round"/>'
        )

    out.append(svg_text(plot_x + plot_size / 2, plot_y + plot_size + 87,
                        f'Ground truth ({metric["unit"]})', fill=TEXT_COLOR,
                        font_size=42, text_anchor="middle"))
    out.append(
        f'<text x="{plot_x - 112}" y="{plot_y + plot_size / 2}" fill="{TEXT_COLOR}" font-size="42" '
        f'text-anchor="middle" transform="rotate(-90 {plot_x - 112} {plot_y + plot_size / 2})">'
        f'Prediction ({html.escape(metric["unit"])})</text>'
    )
    return out


def build_svg(rows: list[dict], all_rows: list[dict]) -> str:
    los_count = sum(row["los_status"] == "los" for row in rows)
    nlos_count = len(rows) - los_count
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<g font-family="Arial, Helvetica, sans-serif">',
        svg_text(45, 58, "Prediction vs. Ground Truth", fill=TEXT_COLOR, font_size=58, font_weight=600),
        f'<rect x="1040" y="25" width="30" height="24" fill="none" stroke="{DEVIATION_COLOR}" stroke-width="4"/>',
        '<text x="1084" y="50" fill="#20242A" font-size="38">NLoS deviation samples</text>',
        f'<circle cx="1575" cy="39" r="12" fill="{LOS_COLOR}"/><text x="1601" y="50" fill="{TEXT_COLOR}" font-size="38">LoS</text>',
        f'<circle cx="1710" cy="39" r="12" fill="{NLOS_COLOR}"/><text x="1736" y="50" fill="{TEXT_COLOR}" font-size="38">NLoS</text>',
        '<line x1="1870" y1="38" x2="1930" y2="38" stroke="#111111" stroke-width="2.6" stroke-dasharray="12 9"/>',
        '<text x="1945" y="50" fill="#20242A" font-size="38">y = x</text>',
    ]
    positions = ((10, 80), (1080, 80), (10, 1080), (1080, 1080))
    for metric, (panel_x, panel_y) in zip(METRICS, positions):
        elements.extend(draw_panel(rows, all_rows, metric, panel_x, panel_y))
    note = (
        f"Stratified sample: {len(rows)} points ({los_count} LoS, {nlos_count} NLoS), seed {SEED}. "
        f"Metrics use all {len(all_rows)} valid samples; angle MAE uses circular error."
    )
    elements.append(svg_text(WIDTH / 2, 2140, note, fill=MUTED_COLOR, font_size=28, text_anchor="middle"))
    elements.extend(("</g>", "</svg>"))
    return "\n".join(elements)


def main() -> None:
    all_rows = read_rows(SOURCE)
    sampled_rows = stratified_sample(all_rows)
    OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SVG.write_text(build_svg(sampled_rows, all_rows), encoding="utf-8")
    print(json.dumps({
        "source_rows": len(all_rows),
        "plotted_rows": len(sampled_rows),
        "los": sum(row["los_status"] == "los" for row in sampled_rows),
        "nlos": sum(row["los_status"] == "nlos" for row in sampled_rows),
        "svg": str(OUTPUT_SVG.resolve()),
    }))


if __name__ == "__main__":
    main()
