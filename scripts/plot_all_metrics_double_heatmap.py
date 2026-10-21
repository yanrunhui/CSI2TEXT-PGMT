#!/usr/bin/env python3
"""Render compact all-sample uncertainty/oracle risk--coverage heatmaps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_all_metrics_normalized_risk_coverage import COVERAGE, CURVES, normalized


OUT_DIR = Path("output/figures")
SELECTED_INDICES = np.array([0, 2, 4, 6, 8, 10])
SELECTED_COVERAGE = COVERAGE[SELECTED_INDICES]
METRIC_LABELS = [
    "First-path delay",
    "First-path angle",
    "First-path power",
    "K-factor",
]


def matrix(strategy: str) -> np.ndarray:
    return np.vstack(
        [normalized(CURVES[label][strategy])[SELECTED_INDICES] for label in METRIC_LABELS]
    )


def contrasting_text_color(rgba: tuple[float, float, float, float]) -> str:
    r, g, b, _ = rgba
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.48 else "#202020"


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
        }
    )

    uncertainty = matrix("uncertainty")
    oracle = matrix("oracle")
    cmap = plt.get_cmap("RdYlGn_r")
    norm = plt.Normalize(vmin=0.0, vmax=100.0)

    fig, axes = plt.subplots(2, 1, figsize=(12.2, 6.7), sharex=True)
    panels = [
        (axes[0], uncertainty, "(a) Uncertainty ranking"),
        (axes[1], oracle, "(b) Oracle ranking"),
    ]

    for ax, values, title in panels:
        image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(title, loc="left", pad=9, weight="bold")
        ax.set_yticks(np.arange(len(METRIC_LABELS)), METRIC_LABELS)
        ax.set_xticks(
            np.arange(len(SELECTED_COVERAGE)),
            [f"{int(x)}" for x in SELECTED_COVERAGE],
        )
        ax.tick_params(axis="both", length=0)

        # White separators make individual values easy to scan without a heavy grid.
        ax.set_xticks(np.arange(-0.5, len(SELECTED_COVERAGE), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(METRIC_LABELS), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2.2)
        ax.tick_params(which="minor", bottom=False, left=False)

        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                rgba = cmap(norm(value))
                ax.text(
                    col,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9.2,
                    weight="bold" if col == values.shape[1] - 1 else "normal",
                    color=contrasting_text_color(rgba),
                )

        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[1].set_xlabel("Coverage (%)")

    fig.subplots_adjust(left=0.17, right=0.86, top=0.90, bottom=0.12, hspace=0.35)
    colorbar_axis = fig.add_axes([0.89, 0.20, 0.018, 0.62])
    colorbar = fig.colorbar(
        image,
        cax=colorbar_axis,
        orientation="vertical",
        ticks=np.arange(0, 101, 20),
    )
    colorbar.set_label("Normalized MAE (% of full-coverage MAE)\nLower is better", labelpad=12)
    colorbar.outline.set_visible(False)

    fig.suptitle(
        "All-sample selective prediction across physical metrics",
        fontsize=18,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.022,
        "Every row is normalized by its MAE at 100% coverage; random rejection is the 100% reference.",
        ha="center",
        fontsize=10.5,
        color="#4D4D4D",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "all_metrics_double_heatmap.png"
    pdf = OUT_DIR / "all_metrics_double_heatmap.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png.resolve())
    print(pdf.resolve())


if __name__ == "__main__":
    main()
