#!/usr/bin/env python3
"""Draw a transparent scientific schematic of sinusoidal phase alignment."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.mplconfig").resolve()))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "figures" / "phase_alignment"
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

    blue = "#2F66A8"
    red = "#A33A32"
    red_soft = "#C98F88"
    purple = "#7B2CBF"
    gold = "#D39A16"
    dark = "#252525"

    phi = np.pi / 6
    x = np.linspace(0, 2 * np.pi, 1200)
    y_ref = np.sin(x)
    y_shifted = np.sin(x - phi)
    y_aligned = np.sin(x)

    fig, ax = plt.subplots(figsize=(7.2, 3.35), facecolor="none")
    ax.set_facecolor("none")

    # Original shifted signal stays visible as a quiet before-alignment reference.
    ax.plot(
        x,
        y_shifted,
        color=red_soft,
        lw=1.8,
        ls=(0, (4, 3)),
        alpha=0.62,
        label=r"Before alignment: $\sin(\theta-\phi)$",
        zorder=2,
    )

    # The aligned signal is drawn over the reference using a dashed line so both remain legible.
    ax.plot(x, y_ref, color=blue, lw=4.0, label=r"Reference: $\sin\theta$", zorder=3)
    ax.plot(
        x,
        y_aligned,
        color=red,
        lw=2.5,
        ls=(0, (7, 4)),
        label=r"After alignment: $y_{\rm aligned}(\theta)$",
        zorder=4,
    )

    # Custom axes keep the background fully transparent.
    ax.annotate(
        "",
        xy=(2 * np.pi + 0.38, 0),
        xytext=(-0.12, 0),
        arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": dark, "mutation_scale": 9},
        zorder=5,
    )
    ax.annotate(
        "",
        xy=(0, 1.30),
        xytext=(0, -1.30),
        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": dark, "mutation_scale": 8},
        zorder=5,
    )

    ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    labels = ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    for xpos, label in zip(ticks, labels):
        ax.plot([xpos, xpos], [-0.035, 0.035], color=dark, lw=0.9, zorder=6)
        ax.text(xpos, -0.105, label, ha="center", va="top", color=dark, fontsize=8)
    ax.text(2 * np.pi + 0.42, -0.02, r"phase, $\theta$", ha="left", va="center", color=dark, fontsize=8.5)

    # Show the geometric correction from the shifted peak to the reference peak.
    x_ref_peak = np.pi / 2
    x_shift_peak = np.pi / 2 + phi
    ax.scatter([x_ref_peak], [1], s=38, color=blue, edgecolor="white", linewidth=0.8, zorder=7)
    ax.scatter([x_shift_peak], [1], s=42, facecolor="white", edgecolor=red_soft, linewidth=1.3, zorder=7)
    ax.annotate(
        "",
        xy=(x_ref_peak, 1.16),
        xytext=(x_shift_peak, 1.16),
        arrowprops={"arrowstyle": "-|>", "lw": 1.7, "color": purple, "mutation_scale": 9},
        zorder=8,
    )
    ax.text(
        (x_ref_peak + x_shift_peak) / 2,
        1.205,
        r"phase compensation: shift by $-\phi$",
        ha="center",
        va="bottom",
        color=purple,
        fontsize=8.2,
        fontweight="bold",
    )

    # Highlight the result without hiding the overlaid curves.
    target_x = 5.18
    target_y = np.sin(target_x)
    ax.annotate(
        r"Phase aligned: $\Delta\phi=0$",
        xy=(target_x, target_y),
        xytext=(4.25, -1.20),
        ha="left",
        va="center",
        color=purple,
        fontsize=8.1,
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-|>",
            "color": purple,
            "lw": 1.1,
            "mutation_scale": 8,
            "connectionstyle": "arc3,rad=-0.12",
        },
        bbox={"boxstyle": "round,pad=0.25", "fc": "#FFF3BF", "ec": gold, "lw": 0.9, "alpha": 0.95},
        zorder=9,
    )

    ax.text(
        0.10,
        -1.20,
        r"$y_{\rm aligned}(\theta)=y_{\rm shifted}(\theta+\phi)=\sin\theta$",
        ha="left",
        va="center",
        color=red,
        fontsize=8.2,
        fontweight="semibold",
    )

    handles, legend_labels = ax.get_legend_handles_labels()
    order = [1, 0, 2]
    ax.legend(
        [handles[i] for i in order],
        [legend_labels[i] for i in order],
        loc="upper right",
        bbox_to_anchor=(0.995, 0.94),
        fontsize=7.2,
        handlelength=2.7,
        labelspacing=0.45,
    )

    ax.set_xlim(-0.18, 2 * np.pi + 0.84)
    ax.set_ylim(-1.38, 1.40)
    ax.axis("off")
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.05, top=0.98)

    stem = OUT / "phase_alignment"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
