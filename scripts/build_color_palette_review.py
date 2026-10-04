#!/usr/bin/env python3
"""Build a two-page PDF placing academic palette swatches beside each figure."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor, Color
from reportlab.lib.pagesizes import landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "tmp" / "pdfs" / "color_review"
OUT = ROOT / "output" / "pdf" / "academic_palette_beside_figures.pdf"

PAGE = landscape((800, 1400))
W, H = PAGE


PALETTE_1 = [
    ("STRUCTURE", None, None),
    ("Panel A background", "#F7F7F5", "neutral warm gray"),
    ("Panel B background", "#F3F7F8", "cool blue gray"),
    ("Panel C background", "#F8F6F2", "neutral warm gray"),
    ("Card fill", "#FFFFFF", "border #CBD5DF"),
    ("Main headings", "#17365D", "navy"),
    ("SEMANTIC ROLES", None, None),
    ("CSI representation", "#2A7F7F", "deep teal"),
    ("Text representation", "#4C78A8", "steel blue"),
    ("Prototype representation", "#B07A2A", "muted ochre"),
    ("PATHS", None, None),
    ("Inference / encoding", "#56616B", "solid"),
    ("Semantic alignment", "#C56A1A", "dashed"),
    ("Backpropagation", "#B23A48", "solid"),
    ("Body text", "#263238", "charcoal"),
]


PALETTE_2 = [
    ("MODALITIES", None, None),
    ("CSI fill", "#DCEBE8", "border #2A7F7F"),
    ("CSI node Zc", "#4F9A94", "border #246963"),
    ("Text fill", "#EAF0F7", "border #4C78A8"),
    ("Text node Zt", "#5B86B3", "border #315F8A"),
    ("Prototype fill", "#F4EEE2", "border #B07A2A"),
    ("Prototype node Zp", "#C49A4A", "border #87651F"),
    ("PATHS", None, None),
    ("Encoding / inference", "#56616B", "solid"),
    ("Semantic alignment", "#C56A1A", "dashed"),
    ("PROTOTYPE CLASSES", None, None),
    ("P1 / class 1", "#4C78A8", "light #B7CEE2"),
    ("P2 / class 2", "#C7794F", "light #E3B7A3"),
    ("P3 / class 3", "#7A9E6D", "light #C5D8BD"),
    ("Neutral cards", "#F4F5F6", "border #AAB7C4"),
    ("Body text", "#263238", "charcoal"),
]


def fit_image(path: Path, x: float, y: float, box_w: float, box_h: float):
    from PIL import Image

    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(box_w / iw, box_h / ih)
    dw, dh = iw * scale, ih * scale
    return x + (box_w - dw) / 2, y + (box_h - dh) / 2, dw, dh


def draw_palette(c: canvas.Canvas, rows, x: float, y_top: float, width: float):
    y = y_top
    for label, color, note in rows:
        if color is None:
            y -= 8
            c.setFillColor(HexColor("#17365D"))
            c.setFont("Helvetica-Bold", 12)
            c.drawString(x, y, label)
            y -= 22
            c.setStrokeColor(HexColor("#D6DEE6"))
            c.setLineWidth(0.7)
            c.line(x, y + 13, x + width, y + 13)
            continue

        sw = 34
        c.setFillColor(HexColor(color))
        c.setStrokeColor(HexColor("#9DAAB5"))
        c.roundRect(x, y - 23, sw, 23, 3, fill=1, stroke=1)

        c.setFillColor(HexColor("#263238"))
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(x + 45, y - 8, label)
        c.setFont("Helvetica", 9.2)
        c.setFillColor(HexColor("#56616B"))
        c.drawString(x + 45, y - 21, f"{color}  |  {note}")
        y -= 39


def draw_page(c: canvas.Canvas, image: Path, page_title: str, rows, source_name: str):
    c.setFillColor(Color(1, 1, 1))
    c.rect(0, 0, W, H, fill=1, stroke=0)

    c.setFillColor(HexColor("#17365D"))
    c.setFont("Helvetica-Bold", 20)
    c.drawString(44, H - 43, page_title)
    c.setFillColor(HexColor("#667085"))
    c.setFont("Helvetica", 9.5)
    c.drawRightString(W - 42, H - 40, f"Source: {source_name}")

    fig_x, fig_y, fig_w, fig_h = 42, 72, 985, 655
    c.setFillColor(HexColor("#FFFFFF"))
    c.setStrokeColor(HexColor("#D7DEE5"))
    c.roundRect(fig_x, fig_y, fig_w, fig_h, 8, fill=1, stroke=1)
    ix, iy, iw, ih = fit_image(image, fig_x + 12, fig_y + 12, fig_w - 24, fig_h - 24)
    c.drawImage(ImageReader(str(image)), ix, iy, iw, ih, preserveAspectRatio=True, mask="auto")

    side_x = 1062
    c.setFillColor(HexColor("#F7F9FA"))
    c.setStrokeColor(HexColor("#D7DEE5"))
    c.roundRect(side_x - 18, 72, 314, 655, 8, fill=1, stroke=1)
    c.setFillColor(HexColor("#17365D"))
    c.setFont("Helvetica-Bold", 16)
    c.drawString(side_x, 698, "Suggested academic palette")
    c.setFillColor(HexColor("#667085"))
    c.setFont("Helvetica", 9.5)
    c.drawString(side_x, 681, "Use the same color for the same scientific role.")
    draw_palette(c, rows, side_x, 656, 270)

    c.setFillColor(HexColor("#667085"))
    c.setFont("Helvetica-Oblique", 8.5)
    c.drawString(44, 39, "Recommendation: retain viridis for heatmaps; reduce saturation in frames, cards, and arrows.")
    c.showPage()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=PAGE, pageCompression=1)
    c.setTitle("Academic palette beside figures")
    draw_page(
        c,
        IMG_DIR / "csi-1.png",
        "Figure 1 - CSI2TEXT-PGMT color replacement guide",
        PALETTE_1,
        "CSI2TEXT-PGMT-16x10.pdf",
    )
    draw_page(
        c,
        IMG_DIR / "figurew-1.png",
        "Figure 2 - Semantic alignment color replacement guide",
        PALETTE_2,
        "figurew.pdf",
    )
    c.save()
    print(OUT)


if __name__ == "__main__":
    main()
