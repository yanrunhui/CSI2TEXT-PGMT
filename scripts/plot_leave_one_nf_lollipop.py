#!/usr/bin/env python3
"""Render faceted lollipop plots for leave-one-Nf-out degradation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from generate_leave_one_nf_heatmap import HELD_OUT_NF, ROWS, degradation


OUT_DIR = Path("output/figures")
IMPROVEMENT = "#0072B2"
DEGRADATION = "#D55E00"
NEUTRAL = "#707070"
GRID = "#D8D8D8"

SHORT_LABELS = [
    "LoS/NLoS accuracy",
    "Text factuality",
    "Numeric accuracy",
    "First-angle MAE",
    "First-delay MAE",
    "First-power MAE",
    "K-factor MAE",
]


def degradation_matrix() -> np.ndarray:
    return np.asarray(
        [
            [
                degradation(value, row["baseline"], row["higher_is_better"])
                for value in row["values"]
            ]
            for row in ROWS
        ],
        dtype=float,
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 18,
            "axes.titlesize": 19,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
        }
    )

    values = degradation_matrix()
    y = np.arange(len(SHORT_LABELS))
    fig, axes_grid = plt.subplots(3, 2, figsize=(7.2, 8.5), sharex=True, sharey=True)
    axes = axes_grid.ravel()

    for col, (ax, nf) in enumerate(zip(axes, HELD_OUT_NF)):
        panel = values[:, col]
        colors = [IMPROVEMENT if value < 0 else DEGRADATION for value in panel]

        ax.axvline(0, color=NEUTRAL, linewidth=1.25, zorder=1)
        for row, value in enumerate(panel):
            ax.hlines(
                row,
                min(0.0, value),
                max(0.0, value),
                color=colors[row],
                linewidth=3.2,
                alpha=0.72,
                zorder=2,
            )
            ax.scatter(
                value,
                row,
                s=78 if abs(value) < 50 else 98,
                color=colors[row],
                edgecolor="white",
                linewidth=1.0,
                zorder=3,
            )

            text_y = row
            vertical_align = "center"
            if value >= 145:
                text_x, align = value, "center"
                text_y = row - 0.18
                vertical_align = "bottom"
            elif value >= 0:
                text_x, align = value + 4.0, "left"
            else:
                text_x, align = 4.0, "left"
            ax.text(
                text_x,
                text_y,
                f"{value:+.1f}%",
                ha=align,
                va=vertical_align,
                color=colors[row],
                fontsize=13.8,
                weight="bold" if abs(value) >= 50 else "normal",
            )

        ax.set_xlabel(f"Held-out $N_f$ = {nf}", labelpad=8, weight="bold", fontsize=16)
        ax.set_xlim(-18, 181)
        ax.set_xticks([0, 50, 100, 150])
        ax.grid(axis="x", color=GRID, linewidth=0.9, alpha=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#9A9A9A")

        if col % 2 == 0:
            ax.set_yticks(y, SHORT_LABELS)
        else:
            ax.tick_params(labelleft=False)

    axes[0].invert_yaxis()
    axes[-1].axis("off")
    fig.suptitle(
        "Leave-one-$N_f$-out cross-configuration generalization",
        fontsize=19,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        "Positive values indicate worse held-out performance; negative values indicate improvement.",
        ha="center",
        fontsize=14,
        color="#555555",
    )

    legend = [
        Line2D([0], [0], color=IMPROVEMENT, marker="o", linewidth=2.6, label="Improvement"),
        Line2D([0], [0], color=DEGRADATION, marker="o", linewidth=2.6, label="Degradation"),
    ]
    fig.legend(
        handles=legend,
        loc="center",
        bbox_to_anchor=(0.74, 0.19),
        ncol=1,
        frameon=False,
        fontsize=14,
        handlelength=2.4,
        labelspacing=0.9,
    )
    fig.subplots_adjust(left=0.30, right=0.985, top=0.905, bottom=0.075, wspace=0.16, hspace=0.46)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "leave_one_nf_out_lollipop.png"
    pdf = OUT_DIR / "leave_one_nf_out_lollipop.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png.resolve())
    print(pdf.resolve())


if __name__ == "__main__":
    main()
