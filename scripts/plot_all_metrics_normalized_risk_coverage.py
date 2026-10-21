#!/usr/bin/env python3
"""Combine the four all-sample risk--coverage curves on one normalized axis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


OUT_DIR = Path("output/figures")
COVERAGE = np.arange(50, 101, 5, dtype=float)

# All-sample curves represented in the supplied risk--coverage figure.
# The final entry is the full-coverage MAE used to normalize each metric.
CURVES = {
    "First-path delay": {
        "unit": "ns",
        "color": "#0072B2",
        "marker": "o",
        "uncertainty": [6.56, 7.01, 7.11, 7.39, 7.63, 7.96, 8.51, 9.34, 10.33, 11.76, 17.80],
        "oracle": [2.24, 2.57, 2.88, 3.24, 3.64, 4.10, 4.64, 5.49, 6.46, 8.81, 17.80],
    },
    "First-path angle": {
        "unit": "deg",
        "color": "#D55E00",
        "marker": "s",
        "uncertainty": [3.67, 4.54, 5.59, 6.86, 8.09, 9.39, 10.83, 12.31, 13.69, 15.09, 17.70],
        "oracle": [0.884, 1.114, 1.548, 2.161, 3.107, 4.078, 5.407, 6.992, 9.113, 12.052, 17.70],
    },
    "First-path power": {
        "unit": "dBW",
        "color": "#009E73",
        "marker": "^",
        "uncertainty": [3.95, 3.96, 3.96, 3.96, 3.97, 3.99, 4.03, 4.08, 4.17, 4.27, 4.40],
        "oracle": [1.247, 1.388, 1.569, 1.738, 1.960, 2.168, 2.438, 2.738, 3.119, 3.566, 4.40],
    },
    "K-factor": {
        "unit": "dB",
        "color": "#CC79A7",
        "marker": "D",
        "uncertainty": [0.856, 0.909, 0.959, 1.022, 1.079, 1.125, 1.176, 1.232, 1.298, 1.384, 1.570],
        "oracle": [0.374, 0.418, 0.467, 0.529, 0.588, 0.664, 0.751, 0.862, 1.010, 1.185, 1.570],
    },
}


def normalized(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return 100.0 * arr / arr[-1]


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(10.8, 6.7))

    for label, spec in CURVES.items():
        uncertainty = normalized(spec["uncertainty"])
        oracle = normalized(spec["oracle"])
        ax.plot(
            COVERAGE,
            uncertainty,
            color=spec["color"],
            marker=spec["marker"],
            markersize=5.5,
            linewidth=2.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=3,
        )
        ax.plot(
            COVERAGE,
            oracle,
            color=spec["color"],
            linestyle=(0, (2, 2)),
            linewidth=1.8,
            alpha=0.72,
            zorder=2,
        )

    ax.axhline(100.0, color="#333333", linestyle="--", linewidth=1.7, zorder=1)
    ax.text(50.4, 101.5, "Random rejection / full-coverage baseline", color="#333333", fontsize=10.5)

    ax.set_xlim(49, 101)
    ax.set_ylim(0, 108)
    ax.set_xticks(np.arange(50, 101, 10))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Normalized MAE (% of full-coverage MAE)")
    ax.set_title("All-sample selective prediction across physical metrics", pad=18, weight="bold")
    ax.grid(True, color="#D8D8D8", linewidth=0.8, alpha=0.65)
    ax.set_axisbelow(True)

    metric_handles = [
        Line2D(
            [0],
            [0],
            color=spec["color"],
            marker=spec["marker"],
            markersize=6,
            linewidth=2.5,
            label=label,
        )
        for label, spec in CURVES.items()
    ]
    strategy_handles = [
        Line2D([0], [0], color="#444444", linewidth=2.5, label="Uncertainty ranking"),
        Line2D([0], [0], color="#444444", linestyle=(0, (2, 2)), linewidth=1.8, label="Oracle ranking"),
        Line2D([0], [0], color="#333333", linestyle="--", linewidth=1.7, label="Random / baseline"),
    ]
    first_legend = ax.legend(
        handles=metric_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.13),
        ncol=2,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.7,
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=strategy_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, -0.13),
        ncol=1,
        frameon=False,
        handlelength=2.8,
    )

    fig.text(
        0.5,
        0.015,
        "Each metric is divided by its own MAE at 100% coverage; lower is better.",
        ha="center",
        fontsize=10.5,
        color="#555555",
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.90, bottom=0.29)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "all_metrics_normalized_risk_coverage.png"
    pdf = OUT_DIR / "all_metrics_normalized_risk_coverage.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(png.resolve())
    print(pdf.resolve())


if __name__ == "__main__":
    main()
