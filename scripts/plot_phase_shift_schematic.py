#!/usr/bin/env python3
"""Create a polished, transparent phase-shift schematic."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.mplconfig").resolve()))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "figures" / "phase_shift_schematic"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "legend.frameon": False,
            "savefig.transparent": True,
        }
    )

    blue = "#1758B7"
    red = "#9E1B1B"
    navy = "#274C77"
    gold = "#E49A17"
    purple = "#6D4BB3"
    dark = "#202020"
    guide = "#A7ADB4"

    phi = np.pi / 6
    x_ref = np.linspace(0, 2 * np.pi, 900)
    x_shift = np.linspace(0, 2 * np.pi + phi, 980)
    y_ref = np.sin(x_ref)
    y_shift = np.sin(x_shift - phi)

    fig, ax = plt.subplots(figsize=(7.15, 4.55), facecolor="none")
    ax.set_facecolor("none")

    # Quiet amplitude guides.
    for y in (-1, 1):
        ax.hlines(y, 0, 2 * np.pi + phi, color=guide, lw=1.0, ls=(0, (5, 4)), alpha=0.72, zorder=0)

    # Semi-transparent fills retain the original diagram's signal-region logic.
    ax.fill_between(x_ref, 0, y_ref, color=blue, alpha=0.11, zorder=1)
    ax.fill_between(x_shift, 0, y_shift, color=red, alpha=0.11, zorder=1)
    ax.plot(x_ref, y_ref, color=blue, lw=4.0, label=r"Reference: $\sin\theta$", zorder=4)
    ax.plot(x_shift, y_shift, color=red, lw=3.7, label=r"Shifted: $\sin(\theta-\phi)$", zorder=4)

    # Clean custom axes.
    ax.annotate(
        "",
        xy=(2 * np.pi + phi + 0.27, 0),
        xytext=(-0.08, 0),
        arrowprops={"arrowstyle": "-|>", "lw": 1.25, "color": dark, "mutation_scale": 9},
        zorder=6,
    )
    ax.plot([0, 0], [-1.06, 1.04], color=dark, lw=1.25, zorder=6)
    ax.text(2 * np.pi + phi + 0.31, -0.025, r"phase, $\theta$", ha="left", va="center", color=dark, fontsize=8.5)

    ticks = [np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    labels = [r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    for xpos, label in zip(ticks, labels):
        ax.plot([xpos, xpos], [-0.045, 0.045], color=dark, lw=1.0, zorder=7)
        ax.text(xpos, 0.10, label, ha="center", va="bottom", color=dark, fontsize=9)

    # Zero-crossing markers.
    zero_points = [(0, 0), (0, -np.sin(phi)), (2 * np.pi, 0), (2 * np.pi + phi, 0)]
    for xpos, ypos in zero_points:
        ax.scatter([xpos], [ypos], s=48, facecolor="#70757A", edgecolor=dark, linewidth=1.0, zorder=8)

    # Peak-to-peak phase separation.
    peak_ref = np.pi / 2
    peak_shift = np.pi / 2 + phi
    for xpos in (peak_ref, peak_shift):
        ax.vlines(xpos, -1.02, 1.31, color=navy, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.annotate(
        "",
        xy=(peak_shift, 1.34),
        xytext=(peak_ref, 1.34),
        arrowprops={"arrowstyle": "<->", "lw": 1.8, "color": gold, "mutation_scale": 9},
        zorder=8,
    )
    ax.text(
        (peak_ref + peak_shift) / 2,
        1.39,
        r"phase offset  $\phi$",
        ha="center",
        va="bottom",
        color=purple,
        fontsize=9.2,
        fontweight="bold",
    )

    # Endpoint separation makes the 2π+φ interpretation explicit.
    bracket_y = 0.30
    ax.vlines([2 * np.pi, 2 * np.pi + phi], 0.02, bracket_y, color=dark, lw=1.0, zorder=6)
    ax.annotate(
        "",
        xy=(2 * np.pi + phi, bracket_y),
        xytext=(2 * np.pi, bracket_y),
        arrowprops={"arrowstyle": "<->", "lw": 1.15, "color": dark, "mutation_scale": 8},
        zorder=8,
    )
    ax.text(
        2 * np.pi + phi,
        bracket_y + 0.12,
        r"$2\pi+\phi$",
        ha="center",
        va="bottom",
        color=dark,
        fontsize=9.0,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.28", "fc": "#FFD55C", "ec": "none", "alpha": 0.95},
        zorder=9,
    )

    # Compact phase label at the initial zero-crossing separation.
    left_y = -0.72
    ax.vlines([0, phi], -0.52, left_y, color=dark, lw=0.9, zorder=6)
    ax.annotate(
        "",
        xy=(phi, left_y),
        xytext=(0, left_y),
        arrowprops={"arrowstyle": "<->", "lw": 1.1, "color": dark, "mutation_scale": 8},
        zorder=8,
    )
    ax.text(
        phi / 2,
        left_y - 0.08,
        r"$\phi$",
        ha="center",
        va="top",
        color=purple,
        fontsize=9.0,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.20", "fc": "#FFD55C", "ec": "none", "alpha": 0.95},
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=7.5,
        handlelength=2.4,
        labelspacing=0.5,
    )

    ax.set_xlim(-0.28, 2 * np.pi + phi + 0.82)
    ax.set_ylim(-1.30, 1.58)
    ax.axis("off")
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.045, top=0.985)

    stem = OUT / "phase_shift_schematic"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
