#!/usr/bin/env python3
"""Create a transparent CSI propagation and angle-of-arrival schematic."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.mplconfig").resolve()))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch, Wedge
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
USER_ICON = ROOT / "assets" / "figure_components" / "user_walking.png"
BS_ICON = ROOT / "assets" / "figure_components" / "base_station.png"
OUT = ROOT / "output" / "figures" / "csi_propagation_schematic"
OUT.mkdir(parents=True, exist_ok=True)


def wavy_path(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    amplitude: float,
    cycles: float,
    offset: float,
    phase: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a tapered sinusoid around a straight propagation path."""
    t = np.linspace(0.0, 1.0, 900)
    start_v = np.asarray(start, dtype=float)
    end_v = np.asarray(end, dtype=float)
    direction = end_v - start_v
    normal = np.asarray([-direction[1], direction[0]])
    normal /= np.linalg.norm(normal)
    base = start_v[None, :] + t[:, None] * direction[None, :]
    taper = np.sin(np.pi * t) ** 0.42
    displacement = offset + amplitude * taper * np.sin(2 * np.pi * cycles * t + phase)
    points = base + displacement[:, None] * normal[None, :]
    return points[:, 0], points[:, 1]


def load_icon(
    path: Path,
    *,
    remove_white: bool = False,
    crop: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> np.ndarray:
    """Load an icon with clean alpha edges and transparent padding."""
    rgba = np.asarray(Image.open(path).convert("RGBA")).copy()
    left, top, right, bottom = crop
    rgba = rgba[top : rgba.shape[0] - bottom, left : rgba.shape[1] - right].copy()
    if remove_white:
        rgb = rgba[..., :3].astype(float)
        # Convert the source's white/off-white card into transparency while
        # preserving antialiased dark-blue icon edges.
        distance_from_white = 255.0 - rgb.min(axis=2)
        # A slightly firm threshold removes the screenshot's off-white paper
        # texture and faint frame without eroding the dark teal glyph.
        white_alpha = np.clip((distance_from_white - 28.0) / 36.0, 0.0, 1.0)
        rgba[..., 3] = np.minimum(rgba[..., 3], (255 * white_alpha).astype(np.uint8))
    # Transparent padding prevents interpolation from repeating opaque border pixels.
    return np.pad(rgba, ((4, 4), (4, 4), (0, 0)), mode="constant")


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.transparent": True,
        }
    )

    navy = "#173F5F"
    blue = "#285F9E"
    blue_mid = "#4D87B8"
    teal = "#58A6A6"
    plane = "#4B7FB3"
    red = "#D63E36"
    grey = "#A4ADB5"
    dark = "#20252A"

    fig, ax = plt.subplots(figsize=(10.0, 5.4), facecolor="none")
    ax.set_facecolor("none")
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 650)
    ax.axis("off")

    # The supplied raster snippets contain one-pixel capture borders; crop only
    # those borders, not the scientific content.
    user = load_icon(USER_ICON, crop=(0, 3, 0, 9))
    tower = load_icon(BS_ICON, remove_white=True, crop=(5, 5, 5, 5))
    ax.imshow(user, extent=(70, 245, 70, 388), interpolation="bilinear", zorder=7)
    ax.imshow(tower, extent=(930, 1115, 285, 505), interpolation="bilinear", zorder=7)

    # Reference plane / array aperture.
    plane_x = 385
    ax.plot([plane_x, plane_x], [100, 550], color=plane, lw=4.0, zorder=5)
    ax.text(plane_x, 575, "reference plane", ha="center", va="bottom", color=plane, fontsize=12, fontweight="semibold")

    # Propagation geometry.
    start = (940, 415)  # base-station emission point
    end = (235, 258)    # effective UE receive point
    dx, dy = end[0] - start[0], end[1] - start[1]
    t_cross = (plane_x - start[0]) / dx
    y_cross = start[1] + t_cross * dy

    paths = [
        (navy, 7.5, 13.5, 0.0, 0.2, 2.5),
        (blue_mid, 8.5, 11.5, 25.0, 1.2, 2.2),
        (teal, 7.0, 10.0, -24.0, 2.0, 2.1),
    ]
    for color, amp, cycles, offset, phase, lw in paths:
        xs, ys = wavy_path(start, end, amplitude=amp, cycles=cycles, offset=offset, phase=phase)
        ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=4)
        # Small directional arrow at the UE end.
        ax.add_patch(
            FancyArrowPatch(
                (xs[-24], ys[-24]),
                (xs[-1], ys[-1]),
                arrowstyle="-|>",
                mutation_scale=12,
                lw=0,
                color=color,
                zorder=6,
            )
        )

    # A quiet centre line clarifies that the oscillations encode a propagating signal.
    ax.plot([start[0], end[0]], [start[1], end[1]], color=grey, lw=1.0, ls=(0, (4, 4)), alpha=0.6, zorder=1)

    # Angle of arrival is measured from the plane normal toward the direct component.
    ax.plot([205, plane_x], [y_cross, y_cross], color=dark, lw=1.2, zorder=5)
    angle_deg = np.degrees(np.arctan2(end[1] - y_cross, end[0] - plane_x))
    if angle_deg < 0:
        angle_deg += 360
    ax.add_patch(Wedge((plane_x, y_cross), 71, 180, angle_deg, width=13, facecolor=red, alpha=0.18, edgecolor="none", zorder=2))
    ax.add_patch(Arc((plane_x, y_cross), 142, 142, theta1=180, theta2=angle_deg, color=red, lw=3.0, zorder=6))
    ax.text(318, y_cross + 62, r"$\theta$", color=red, fontsize=30, fontweight="bold", ha="center", va="center")
    ax.text(294, y_cross + 34, "angle of arrival", color=red, fontsize=10.5, ha="center", va="center")

    # Labels and emission/receive markers.
    ax.scatter([start[0]], [start[1]], s=46, color=navy, edgecolor="white", linewidth=0.9, zorder=9)
    ax.scatter([end[0]], [end[1]], s=46, color=navy, edgecolor="white", linewidth=0.9, zorder=9)
    ax.text(158, 40, "User terminal", color=dark, fontsize=13, fontweight="semibold", ha="center")
    ax.text(1022, 250, "Base station", color=dark, fontsize=13, fontweight="semibold", ha="center")
    ax.text(655, 500, "CSI propagation", color=navy, fontsize=17, fontweight="bold", ha="center")
    ax.text(655, 470, r"multipath wavefronts  $H(f)$", color=blue, fontsize=11.5, ha="center")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    stem = OUT / "csi_wave_propagation"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
