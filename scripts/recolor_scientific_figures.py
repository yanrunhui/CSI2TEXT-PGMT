#!/usr/bin/env python3
"""Apply the approved academic palette directly to two rendered figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp" / "pdfs" / "recolor_hd"
OUT_IMG = ROOT / "output" / "figures" / "academic_recolored"
OUT_PDF = ROOT / "output" / "pdf" / "academic_recolored_figures.pdf"


def rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.int16)


def apply_map(image: Image.Image, mapping: list[tuple[str, str, int]]) -> tuple[Image.Image, np.ndarray]:
    src = np.asarray(image.convert("RGB"), dtype=np.int16)
    out = src.copy()
    for old, new, tolerance in mapping:
        delta = src - rgb(old)
        mask = np.max(np.abs(delta), axis=2) <= tolerance
        out[mask] = rgb(new)
    return Image.fromarray(out.astype(np.uint8), "RGB"), src


FIG1 = [
    # Large panels and containers.
    ("#FEF5ED", "#F8F6F2", 3),
    ("#F5F3ED", "#F7F7F5", 3),
    ("#E7F3E5", "#EDF4F1", 3),
    ("#F1FBF8", "#F3F7F8", 3),
    ("#DCE6F2", "#E8EEF4", 3),
    ("#E6F0F0", "#EEF3F6", 3),
    ("#F5F7FA", "#FFFFFF", 3),
    ("#FBEBDD", "#FFFFFF", 3),
    ("#F2EBDE", "#F2F4F6", 3),
    ("#F3EBDF", "#F4F4F2", 3),
    ("#ECE6DF", "#F4F5F6", 3),
    ("#F6EEE3", "#F2F4F6", 3),
    ("#F8EFE4", "#FFFFFF", 3),
    ("#ECF1EF", "#F4F6F7", 3),
    ("#F5EDE1", "#F4F4F2", 3),
    ("#F5EDE2", "#F2F4F6", 3),
    ("#D7E4BD", "#DDEBE8", 4),
    ("#EBF1DE", "#F4F5F6", 3),
    ("#EFE8DA", "#F4F4F2", 3),
    ("#F2F2F2", "#F4F5F6", 3),
    ("#F5F6F9", "#FFFFFF", 3),
    ("#F7E9DA", "#F2F4F6", 3),
    ("#FAFBFF", "#FFFFFF", 3),
    ("#E8F6FC", "#EAF0F7", 3),
    ("#F9F4F4", "#F4EEE2", 3),
    # Typography, semantic colors, and paths.
    ("#06346C", "#17365D", 4),
    ("#17375E", "#263238", 4),
    ("#984807", "#56616B", 4),
    ("#D99694", "#B8C3CC", 4),
    ("#B7DEE8", "#DCEAF2", 4),
    ("#558ED5", "#4C78A8", 5),
    ("#8EB4E3", "#B7CEE2", 5),
    ("#C0504D", "#B23A48", 5),
    ("#FF0000", "#B23A48", 5),
    ("#4F81BD", "#4C78A8", 5),
    ("#4CAA48", "#2A7F7F", 5),
    ("#DC142D", "#B23A48", 5),
    ("#CCC1DA", "#E3B7A3", 5),
    ("#E46C0A", "#7A9E6D", 5),
    ("#8064A2", "#C7794F", 5),
    ("#6EAF45", "#7A9E6D", 5),
    ("#A9CF55", "#C5D8BD", 5),
    ("#2E8447", "#2A7F7F", 5),
    ("#05B053", "#2A7F7F", 5),
    ("#D6E36A", "#C5D8BD", 5),
    ("#9BBB59", "#7A9E6D", 5),
    ("#1F497D", "#315F8A", 5),
    ("#376092", "#315F8A", 5),
    ("#953735", "#91462B", 5),
]


FIG2 = [
    # Neutral containers.
    ("#F5F3F1", "#FAF8F3", 3),
    ("#E8E8E8", "#F4F5F6", 3),
    ("#EFE7F9", "#F2F4F6", 3),
    # CSI branch fills.
    ("#D9F2D0", "#E8F1F0", 3),
    ("#E0F1DD", "#DCEBE8", 3),
    ("#DEEFD9", "#DCEBE8", 3),
    ("#DAF2D0", "#E8F1F0", 3),
    # Text branch fills.
    ("#E8F2FD", "#F2F5F9", 3),
    ("#C1E5F5", "#E3EDF5", 3),
    ("#D0E3FE", "#EAF0F7", 3),
    ("#CAE1FD", "#E3EDF5", 3),
    ("#DCEAF7", "#EAF0F7", 3),
    # Typography and borders.
    ("#215F9A", "#17365D", 5),
    ("#042433", "#263238", 5),
    ("#0B1E31", "#263238", 5),
    ("#3B7D23", "#246963", 5),
    ("#0B6822", "#205F5F", 5),
    ("#2F6FB0", "#315F8A", 5),
    # Semantic nodes.
    ("#8ED973", "#4F9A94", 5),
    ("#4E95D9", "#5B86B3", 5),
    ("#D86ECC", "#C49A4A", 5),
    ("#50164A", "#87651F", 5),
    # Alignment and prototype styling.
    ("#F07E15", "#C56A1A", 5),
    ("#C00071", "#7A4A16", 5),
    ("#7F2ABC", "#8B6B2E", 5),
    ("#78206E", "#6F5422", 5),
    ("#AB74D3", "#6F5422", 6),
    ("#A870CE", "#806A45", 5),
    ("#A1659E", "#806A6A", 5),
    ("#D1B3E7", "#D8C7A7", 5),
    ("#E6D9EA", "#EAE2D4", 5),
    ("#CBACDE", "#D1B98E", 5),
    ("#0070C0", "#4C78A8", 5),
    ("#738C3A", "#7A9E6D", 5),
    ("#83CBEB", "#B7CEE2", 5),
    ("#E3B8B7", "#E3B7A3", 5),
    ("#D1E4B9", "#C5D8BD", 5),
    ("#47D45A", "#7A9E6D", 5),
    ("#8DB6E0", "#B7CEE2", 5),
    # Current brown inference paths are made graphite; P2 centers are restored below.
    ("#C04F15", "#56616B", 5),
    ("#C04F14", "#56616B", 5),
]


def recolor_figure2_p2(image: Image.Image, original: np.ndarray) -> Image.Image:
    out = np.asarray(image).copy()
    h, w = original.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    xn, yn = xx / w, yy / h
    old1 = np.max(np.abs(original - rgb("#C04F15")), axis=2) <= 6
    old2 = np.max(np.abs(original - rgb("#C04F14")), axis=2) <= 6
    p2_regions = (((xn > 0.17) & (xn < 0.25) & (yn > 0.67) & (yn < 0.82)) |
                  ((xn > 0.64) & (xn < 0.74) & (yn > 0.61) & (yn < 0.80)))
    out[(old1 | old2) & p2_regions] = rgb("#C7794F").astype(np.uint8)

    # The prototype-bank heading uses a soft purple glow made from many
    # antialias colors. Recompose that localized purple text/glow in ochre.
    bank_title_region = (xn > 0.10) & (xn < 0.40) & (yn > 0.64) & (yn < 0.78)
    purpleish = (original[..., 0] > original[..., 1] * 1.12) & (original[..., 2] > original[..., 1] * 1.18)
    title_mask = bank_title_region & purpleish
    background = rgb("#FAF8F3").astype(float)
    source_main = rgb("#7F2ABC").astype(float)
    target_main = rgb("#6F5422").astype(float)
    denom = np.linalg.norm(source_main - background)
    alpha = np.clip(np.linalg.norm(original.astype(float) - background, axis=2) / denom, 0.0, 1.0)
    remapped = background[None, None, :] * (1.0 - alpha[..., None]) + target_main[None, None, :] * alpha[..., None]
    out[title_mask] = remapped[title_mask].astype(np.uint8)
    return Image.fromarray(out, "RGB")


def save_pdf(images: list[Path]) -> None:
    sizes = []
    for p in images:
        with Image.open(p) as im:
            sizes.append(im.size)
    c = None
    for index, (path, (w, h)) in enumerate(zip(images, sizes)):
        if c is None:
            c = canvas.Canvas(str(OUT_PDF), pagesize=(w, h), pageCompression=1)
            c.setTitle("Academic recolored scientific figures")
        else:
            c.setPageSize((w, h))
        c.drawImage(ImageReader(str(path)), 0, 0, width=w, height=h, preserveAspectRatio=True)
        c.showPage()
    assert c is not None
    c.save()


def main() -> None:
    OUT_IMG.mkdir(parents=True, exist_ok=True)
    fig1, _ = apply_map(Image.open(SRC / "figure1.png"), FIG1)
    fig2, original2 = apply_map(Image.open(SRC / "figure2.png"), FIG2)
    fig2 = recolor_figure2_p2(fig2, original2)

    p1 = OUT_IMG / "figure1_academic_recolored.png"
    p2 = OUT_IMG / "figure2_academic_recolored.png"
    fig1.save(p1, optimize=True, dpi=(300, 300))
    fig2.save(p2, optimize=True, dpi=(300, 300))
    save_pdf([p1, p2])
    print(p1)
    print(p2)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
