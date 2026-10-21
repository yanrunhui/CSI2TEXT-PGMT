#!/usr/bin/env python3
"""Typeset the reported physics-head ablation as two manuscript tables."""

from __future__ import annotations

from pathlib import Path

from pdf2image import convert_from_path
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "output/pdf/physics_ablation_two_tables.pdf"
PNG = ROOT / "output/figures/physics_ablation_two_tables.png"

pdfmetrics.registerFont(TTFont("TableTimes", "/System/Library/Fonts/Supplemental/Times New Roman.ttf"))
pdfmetrics.registerFont(TTFont("TableTimes-Bold", "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"))

PAGE_W, PAGE_H = 515, 350
LEFT, RIGHT = 7.5, 507.5
BLACK = HexColor("#161616")
SUBTLE = HexColor("#52606A")
FULL_BG = HexColor("#E1EEF8")
SUMMARY_BG = HexColor("#F2F3F4")
POSITIVE = HexColor("#A4232C")
NEGATIVE = HexColor("#475463")

VARIANTS = ["Full multitask", "No CSI–text", "No text–prototype", "Physics only"]

TABLES = [
    {
        "title": "A. Delay and channel estimates",
        "top": 291,
        "first_col": 120,
        "headers": [
            ("LoS delay", "MAE ↓"),
            ("NLoS delay", "MAE ↓"),
            ("Delay spread", "MAE ↓"),
            ("K-factor", "MAE ↓"),
        ],
        "values": [
            ["17.8347", "20.9726", "22.1455", "21.4466"],
            ["58.8105", "59.2215", "59.8201", "58.8984"],
            ["17.8460", "17.9922", "18.2041", "17.7470"],
            ["3.2551", "3.3279", "3.3932", "3.3915"],
        ],
    },
    {
        "title": "B. Angular and path-structure estimates",
        "top": 139,
        "first_col": 120,
        "headers": [
            ("LoS angle", "MAE ↓"),
            ("NLoS angle", "MAE ↓"),
            ("Reflection", "count MAE ↓"),
            ("Path count", "MAE ↓"),
            ("Azimuth spread", "MAE ↓"),
        ],
        "values": [
            ["3.7504", "3.7504", "3.9118", "4.3980"],
            ["41.4775", "40.6324", "42.2838", "40.3674"],
            ["1.2724", "1.3182", "1.2924", "1.2746"],
            ["0.1935", "0.2051", "0.2505", "0.3267"],
            ["7.869", "9.6889", "8.1116", "8.269"],
        ],
    },
]


def text(canvas: Canvas, x: float, y: float, value: str, size: float = 9.5,
         bold: bool = False, color=BLACK, align: str = "left") -> None:
    font = "TableTimes-Bold" if bold else "TableTimes"
    canvas.setFont(font, size)
    canvas.setFillColor(color)
    if align == "center":
        canvas.drawCentredString(x, y, value)
    elif align == "right":
        canvas.drawRightString(x, y, value)
    else:
        canvas.drawString(x, y, value)


def rule(canvas: Canvas, y: float, width: float) -> None:
    canvas.setStrokeColor(BLACK)
    canvas.setLineWidth(width)
    canvas.line(LEFT, y, RIGHT, y)


def draw_table(canvas: Canvas, table: dict) -> None:
    top = table["top"]
    title_y = top + 8
    text(canvas, LEFT + 2, title_y, table["title"], 10.2, bold=True)
    rule(canvas, top, 0.9)

    n = len(table["headers"])
    centers = [LEFT + table["first_col"] + (j + 0.5) * (500 - table["first_col"]) / n for j in range(n)]
    text(canvas, LEFT + 3, top - 19, "Variant", 9.3, bold=True)
    for x, (line1, line2) in zip(centers, table["headers"]):
        text(canvas, x, top - 13, line1, 8.8, bold=True, align="center")
        text(canvas, x, top - 24, line2, 8.8, bold=True, align="center")
    rule(canvas, top - 28, 0.5)

    row_h = 18
    for row_index, variant in enumerate(VARIANTS):
        row_top = top - 28 - row_index * row_h
        row_bottom = row_top - row_h
        if row_index == 0:
            canvas.setFillColor(FULL_BG)
            canvas.rect(LEFT, row_bottom, RIGHT - LEFT, row_h, fill=1, stroke=0)
        baseline = row_bottom + 5.2
        text(canvas, LEFT + 3, baseline, variant, 9.6, bold=(row_index == 0))
        for j, col in enumerate(table["values"]):
            value = col[row_index]
            lowest = min(float(v) for v in col)
            text(canvas, centers[j], baseline, value, 9.7,
                 bold=abs(float(value) - lowest) < 1e-10, align="center")

    summary_top = top - 28 - 4 * row_h - 3
    rule(canvas, summary_top + 3, 0.55)
    summary_bottom = summary_top - 20
    canvas.setFillColor(SUMMARY_BG)
    canvas.rect(LEFT, summary_bottom, RIGHT - LEFT, 20, fill=1, stroke=0)
    text(canvas, LEFT + 3, summary_bottom + 5.1, "Improvement vs. physics only", 8.65, bold=True)
    for j, col in enumerate(table["values"]):
        full, physics = float(col[0]), float(col[3])
        improvement = 100.0 * (physics - full) / physics
        color = POSITIVE if improvement >= 0 else NEGATIVE
        text(canvas, centers[j], summary_bottom + 5.1, f"{improvement:+.1f}%", 9.7,
             bold=True, color=color, align="center")
    rule(canvas, summary_bottom, 1.1)


def main() -> None:
    PDF.parent.mkdir(parents=True, exist_ok=True)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    canvas = Canvas(str(PDF), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    canvas.setFillColor(HexColor("#FFFFFF"))
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    text(canvas, LEFT + 1, 335,
         "Multitask and semantic-alignment ablation. All entries are MAEs (lower is better).", 9.7)
    text(canvas, LEFT + 1, 323,
         "LoS/NLoS delay and angle refer to the first path; bold marks minima; negative improvement denotes higher error.", 9.7)
    for table in TABLES:
        draw_table(canvas, table)
    canvas.showPage()
    canvas.save()

    image = convert_from_path(str(PDF), dpi=300, first_page=1, last_page=1, fmt="png")[0]
    image.save(PNG)
    print(PDF)
    print(PNG)


if __name__ == "__main__":
    main()
