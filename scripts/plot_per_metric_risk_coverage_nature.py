#!/usr/bin/env python3
"""Create Nature-style per-metric risk--coverage panels."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Mandatory publication settings: sans-serif text and editable SVG labels.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


OUT_DIR = Path("output/figures/per_metric_risk_coverage_nature")
STEM = OUT_DIR / "per_metric_risk_coverage_nature"
CAPTION = (
    "(c,d) Per-metric normalized MAE across retained coverage under (c) "
    "ensemble-uncertainty ranking and (d) oracle ranking. Each metric is "
    "normalized to its full-coverage MAE (100%); lower values indicate better "
    "selective prediction."
)

COVERAGE = np.asarray([50, 60, 70, 80, 90, 100], dtype=float)
METRICS = [
    "First-path delay",
    "First-path angle",
    "First-path power",
    "K-factor",
]

UNCERTAINTY_RISK = np.asarray(
    [
        [36.9, 39.9, 42.9, 47.8, 58.0, 100.0],
        [20.7, 31.6, 45.7, 61.2, 77.3, 100.0],
        [89.8, 90.0, 90.2, 91.6, 94.8, 100.0],
        [54.5, 61.1, 68.7, 74.9, 82.7, 100.0],
    ],
    dtype=float,
)
ORACLE_RISK = np.asarray(
    [
        [12.6, 16.2, 20.4, 26.1, 36.3, 100.0],
        [5.0, 8.7, 17.6, 30.5, 51.5, 100.0],
        [28.3, 35.7, 44.5, 55.4, 70.9, 100.0],
        [23.8, 29.7, 37.5, 47.8, 64.3, 100.0],
    ],
    dtype=float,
)

METRIC_STYLE = {
    "First-path delay": {"color": "#355C7D", "marker": "o"},
    "First-path angle": {"color": "#C58B18", "marker": "s"},
    "First-path power": {"color": "#3F8F7A", "marker": "^"},
    "K-factor": {"color": "#8B6B9C", "marker": "D"},
}


def apply_publication_style() -> None:
    """Apply compact, editable, print-oriented Matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.0,
            "axes.titlesize": 7.2,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "savefig.facecolor": "white",
        }
    )


def write_source_data() -> None:
    """Export every plotted metric, method and coverage value."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    methods = {
        "Ensemble uncertainty": UNCERTAINTY_RISK,
        "Oracle": ORACLE_RISK,
    }
    with (OUT_DIR / "source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "metric", "coverage_percent", "normalized_mae_percent"],
        )
        writer.writeheader()
        for method, values in methods.items():
            for metric, metric_values in zip(METRICS, values):
                for coverage, value in zip(COVERAGE, metric_values):
                    writer.writerow(
                        {
                            "method": method,
                            "metric": metric,
                            "coverage_percent": int(coverage),
                            "normalized_mae_percent": f"{value:.1f}",
                        }
                    )
    (OUT_DIR / "caption.txt").write_text(CAPTION + "\n", encoding="utf-8")


def draw_panel(
    ax: plt.Axes,
    values: np.ndarray,
    panel_label: str,
    title: str,
) -> list[plt.Line2D]:
    """Draw one aligned risk--coverage panel."""
    lines: list[plt.Line2D] = []
    for metric, metric_values in zip(METRICS, values):
        style = METRIC_STYLE[metric]
        (line,) = ax.plot(
            COVERAGE,
            metric_values,
            color=style["color"],
            marker=style["marker"],
            markevery=[0, 1, 2, 3, 4],
            markersize=3.9,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.55,
            linewidth=1.45,
            label=metric,
            zorder=3,
        )
        lines.append(line)

    # All metrics converge at the full-coverage reference; one neutral marker
    # avoids implying that one metric visually dominates the shared endpoint.
    ax.scatter(
        [100],
        [100],
        s=18,
        facecolor="white",
        edgecolor="#555555",
        linewidth=0.7,
        zorder=5,
    )
    ax.axhline(100, color="#8A8A8A", linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)

    ax.set_xlim(48.5, 101.5)
    ax.set_ylim(0, 107)
    ax.set_xticks(COVERAGE.astype(int))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Coverage (%)", labelpad=4)
    ax.tick_params(axis="both", direction="out", length=2.5, width=0.65, pad=2.2)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    ax.text(-0.14, 1.09, panel_label, transform=ax.transAxes, fontsize=8.5, fontweight="bold")
    ax.text(0.0, 1.09, title, transform=ax.transAxes, fontsize=7.2, fontweight="bold")
    return lines


def main() -> None:
    apply_publication_style()
    write_source_data()

    # Nature double-column width (183 mm), with matched panels for direct comparison.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(183 / 25.4, 78 / 25.4),
        sharex=True,
        sharey=True,
    )
    ax_uncertainty, ax_oracle = axes
    lines = draw_panel(ax_uncertainty, UNCERTAINTY_RISK, "c", "Uncertainty ranking")
    draw_panel(ax_oracle, ORACLE_RISK, "d", "Oracle ranking")

    ax_uncertainty.set_ylabel("Normalized MAE (% of full coverage)", labelpad=5)
    ax_uncertainty.text(
        0.02,
        0.04,
        "Lower is better",
        transform=ax_uncertainty.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.5,
        color="#666666",
    )
    ax_oracle.text(
        0.98,
        0.965,
        "Full coverage = 100%",
        transform=ax_oracle.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        color="#666666",
    )

    fig.legend(
        lines,
        METRICS,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.885),
        ncol=4,
        fontsize=6.0,
        handlelength=1.7,
        handletextpad=0.45,
        columnspacing=1.25,
    )

    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.19, top=0.74, wspace=0.18)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(STEM.with_suffix(".svg"))
    fig.savefig(STEM.with_suffix(".pdf"))
    fig.savefig(STEM.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(STEM.with_suffix(".png"), dpi=600)
    plt.close(fig)

    for suffix in (".svg", ".pdf", ".tiff", ".png"):
        print(STEM.with_suffix(suffix).resolve())
    print((OUT_DIR / "source_data.csv").resolve())
    print((OUT_DIR / "caption.txt").resolve())


if __name__ == "__main__":
    main()
