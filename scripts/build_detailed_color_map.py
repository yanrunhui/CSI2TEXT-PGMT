#!/usr/bin/env python3
"""Create object-by-object fill/border/text color maps beside two figures."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from reportlab.lib.colors import HexColor, Color
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp" / "pdfs" / "color_review"
OUT = ROOT / "output" / "pdf" / "academic_palette_object_by_object.pdf"
PAGE_W, PAGE_H = 1600, 900


F1 = [
    ("A. TOKENIZATION", None, None, None),
    ("A outer panel", "#F7F7F5", "-", "#17365D"),
    ("Preprocessing group", "#F1F3F5", "-", "#17365D"),
    ("Phase / Resample / FFT / Patch", "#FFFFFF", "#CBD5DF", "#17365D"),
    ("Config group", "#EEF3F6", "#C2CED8", "#17365D"),
    ("Raw power / Power Structure", "#F4F6F7", "#CBD5DF", "#17365D"),
    ("Token blocks X", "#DCEAF2", "#8CB6CC", "#2F6F8F"),
    ("LoS emphasis", "#EAF0F7", "#4C78A8", "#315F8A"),
    ("NLoS emphasis", "#F7ECE5", "#C56A1A", "#8A4B18"),
    ("B. SHARED ENCODER", None, None, None),
    ("B outer panel", "#F3F7F8", "-", "#17365D"),
    ("Transformer container", "#E8EEF4", "#C1CEDA", "#17365D"),
    ("Norm / Project / FiLM / Attention", "#FFFFFF", "#CBD5DF", "#17365D"),
    ("Semantic representation area", "#EDF4F1", "#C6D9D3", "#17365D"),
    ("CSI embedding", "#DDEBE8", "#2A7F7F", "#263238"),
    ("Text representation card", "#EAF0F7", "#4C78A8", "#263238"),
    ("Prototype representation card", "#F4EEE2", "#B07A2A", "#263238"),
    ("Loss node", "#FFFFFF", "#AAB7C4", "#17365D"),
    ("C. PHYSICS TO TEXT", None, None, None),
    ("C outer panel", "#F8F6F2", "-", "#17365D"),
    ("Temporal / Angular / Power", "#FFFFFF", "#CBD5DF", "#17365D"),
    ("Physics MLP", "#E9EEF3", "#AAB7C4", "#17365D"),
    ("Correction / Verbalizer", "#F2F4F6", "#AAB7C4", "#17365D"),
    ("Report", "#F4F4F2", "#C8C2B8", "#17365D"),
    ("CONNECTORS", None, None, None),
    ("Ordinary connector", "-", "#7B8794", "-"),
    ("Inference / encoding", "-", "#56616B", "-"),
    ("Semantic alignment (dashed)", "-", "#C56A1A", "-"),
    ("Backpropagation / warning", "-", "#B23A48", "#B23A48"),
]


F2 = [
    ("CSI BRANCH", None, None, None),
    ("CSI sample frame", "#E8F1F0", "#2A7F7F", "#17365D"),
    ("Shared CSI Transformer", "#E8F1F0", "#2A7F7F", "#205F5F"),
    ("CSI embedding label", "#DCEBE8", "#2A7F7F", "#205F5F"),
    ("TEXT BRANCH", None, None, None),
    ("Text transformer", "#EAF0F7", "#4C78A8", "#17365D"),
    ("Description outer frame", "#F2F5F9", "#4C78A8", "#17365D"),
    ("Description sentence cards", "#E3EDF5", "#C4D3E1", "#263238"),
    ("SEMANTIC NODES", None, None, None),
    ("Zc - CSI node", "#4F9A94", "#246963", "#FFFFFF"),
    ("Zt - text node", "#5B86B3", "#315F8A", "#FFFFFF"),
    ("Zp - prototype node", "#C49A4A", "#87651F", "#FFFFFF"),
    ("PATHS AND LABELS", None, None, None),
    ("Encoding / inference arrow", "-", "#56616B", "-"),
    ("Alignment arrows (dashed)", "-", "#C56A1A", "-"),
    ("Alignment label text", "-", "-", "#7A4A16"),
    ("PROTOTYPE BANK", None, None, None),
    ("Prototype bank container", "#FAF8F3", "#8B6B2E", "#6F5422"),
    ("Prototype embedding", "#F4EEE2", "#B07A2A", "#6F5422"),
    ("P1 center", "#4C78A8", "#315F8A", "#FFFFFF"),
    ("P2 center", "#C7794F", "#91462B", "#FFFFFF"),
    ("P3 center", "#7A9E6D", "#4F6E45", "#FFFFFF"),
    ("P1 sample dots", "#B7CEE2", "#4C78A8", "#263238"),
    ("P2 sample dots", "#E3B7A3", "#C7794F", "#263238"),
    ("P3 sample dots", "#C5D8BD", "#7A9E6D", "#263238"),
    ("OTHER BOXES", None, None, None),
    ("Similarity equation box", "#F4F5F6", "#6F8091", "#263238"),
    ("Legend", "#FFFFFF", "#AAB7C4", "#263238"),
    ("Explanation note", "#F2F4F6", "#D7DEE5", "#465564"),
]


def swatch(c, x, y, value, *, line=False):
    if not value or value == "-":
        c.setFillColor(HexColor("#8B96A3"))
        c.setFont("Helvetica", 7.5)
        c.drawString(x, y + 2, "-")
        return
    col = HexColor(value)
    if line:
        c.setStrokeColor(col)
        c.setLineWidth(3)
        c.line(x, y + 7, x + 26, y + 7)
    else:
        c.setFillColor(col)
        c.setStrokeColor(HexColor("#98A4AF"))
        c.roundRect(x, y, 24, 15, 2, fill=1, stroke=1)
    c.setFillColor(HexColor("#4B5563"))
    c.setFont("Helvetica", 7.2)
    c.drawString(x + 31, y + 3, value)


def draw_table(c, rows, x, y_top, width, row_h=24):
    obj_w = 220
    col_w = (width - obj_w) / 3
    c.setFillColor(HexColor("#E9EEF3"))
    c.rect(x, y_top - 23, width, 23, fill=1, stroke=0)
    c.setFillColor(HexColor("#17365D"))
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(x + 7, y_top - 15, "OBJECT / SHAPE")
    c.drawString(x + obj_w + 7, y_top - 15, "FILL")
    c.drawString(x + obj_w + col_w + 7, y_top - 15, "BORDER / LINE")
    c.drawString(x + obj_w + 2 * col_w + 7, y_top - 15, "TEXT")
    y = y_top - 28

    for label, fill, border, text in rows:
        if fill is None:
            c.setFillColor(HexColor("#17365D"))
            c.setFont("Helvetica-Bold", 8.3)
            c.drawString(x + 5, y - 1, label)
            c.setStrokeColor(HexColor("#CCD5DE"))
            c.setLineWidth(0.5)
            c.line(x, y - 5, x + width, y - 5)
            y -= 18
            continue
        c.setFillColor(HexColor("#263238"))
        c.setFont("Helvetica", 7.8)
        c.drawString(x + 5, y + 3, label)
        swatch(c, x + obj_w + 6, y - 2, fill)
        swatch(c, x + obj_w + col_w + 6, y - 2, border, line=(fill == "-"))
        swatch(c, x + obj_w + 2 * col_w + 6, y - 2, text)
        y -= row_h


def fit(path, x, y, w, h):
    with Image.open(path) as im:
        iw, ih = im.size
    s = min(w / iw, h / ih)
    return x + (w - iw * s) / 2, y + (h - ih * s) / 2, iw * s, ih * s


def draw_page(c, image, title, rows, source):
    c.setFillColor(Color(1, 1, 1))
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(HexColor("#17365D"))
    c.setFont("Helvetica-Bold", 21)
    c.drawString(36, PAGE_H - 42, title)
    c.setFillColor(HexColor("#667085"))
    c.setFont("Helvetica", 9)
    c.drawRightString(PAGE_W - 35, PAGE_H - 40, f"Source: {source}")

    c.setFillColor(HexColor("#FFFFFF"))
    c.setStrokeColor(HexColor("#D7DEE5"))
    c.roundRect(30, 75, 910, 755, 8, fill=1, stroke=1)
    ix, iy, iw, ih = fit(image, 43, 88, 884, 729)
    c.drawImage(ImageReader(str(image)), ix, iy, iw, ih, preserveAspectRatio=True, mask="auto")

    c.setFillColor(HexColor("#F8FAFB"))
    c.roundRect(958, 75, 612, 755, 8, fill=1, stroke=1)
    c.setFillColor(HexColor("#17365D"))
    c.setFont("Helvetica-Bold", 15)
    c.drawString(976, 805, "Object-by-object color specification")
    c.setFillColor(HexColor("#56616B"))
    c.setFont("Helvetica", 8.5)
    c.drawString(976, 790, "Pale fills define containers; saturated colors identify semantic roles.")
    draw_table(c, rows, 976, 770, 574, row_h=22)

    c.setFillColor(HexColor("#263238"))
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(36, 42, "Rule:")
    c.setFont("Helvetica", 8.5)
    c.drawString(64, 42, "container = very pale fill; semantic node = saturated fill; path meaning = line color + line style.")
    c.showPage()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("Academic palette object-by-object guide")
    draw_page(c, SRC / "csi-1.png", "Figure 1 - exact fill, border, text, and path colors", F1, "CSI2TEXT-PGMT-16x10.pdf")
    draw_page(c, SRC / "figurew-1.png", "Figure 2 - exact modality, node, prototype, and path colors", F2, "figurew.pdf")
    c.save()
    print(OUT)


if __name__ == "__main__":
    main()
