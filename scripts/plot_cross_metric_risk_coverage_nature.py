#!/usr/bin/env python3
"""Create a Nature-style cross-metric normalized risk--coverage panel."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Mandatory publication settings: sans-serif text and editable SVG labels.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


OUT_DIR = Path("output/figures/cross_metric_risk_coverage_nature")
STEM = OUT_DIR / "cross_metric_risk_coverage_nature"
CAPTION = (
    "(a) Cross-metric normalized risk–coverage curves at discrete coverage levels "
    "under ensemble-uncertainty, oracle and random-rejection ranking."
)

COVERAGE = np.asarray([50, 60, 70, 80, 90, 100], dtype=float)
METRICS = ["First-path delay", "First-path angle", "First-path power", "K-factor"]

# Each row is one metric; columns follow COVERAGE. Values are normalized by the
# corresponding full-coverage MAE, so 100% is the full-coverage reference.
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
RANDOM_RISK = np.full_like(UNCERTAINTY_RISK, 100.0)

SERIES_STYLE = {
    "Ensemble uncertainty": {"color": "#355C7D", "marker": "o", "linestyle": "-"},
    "Oracle": {"color": "#5D9D82", "marker": "s", "linestyle": "-"},
    "Random rejection": {"color": "#929292", "marker": None, "linestyle": (0, (3, 2))},
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


def plotted_series() -> dict[str, np.ndarray]:
    """Return cross-metric means used in the panel."""
    return {
        "Ensemble uncertainty": UNCERTAINTY_RISK.mean(axis=0),
        "Oracle": ORACLE_RISK.mean(axis=0),
        "Random rejection": RANDOM_RISK.mean(axis=0),
    }


def write_source_data() -> None:
    """Export metric-level values and the plotted cross-metric mean."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    methods = {
        "Ensemble uncertainty": UNCERTAINTY_RISK,
        "Oracle": ORACLE_RISK,
        "Random rejection": RANDOM_RISK,
    }
    with (OUT_DIR / "source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "coverage_percent", *METRICS, "cross_metric_mean"],
        )
        writer.writeheader()
        for method, values in methods.items():
            for column, coverage in enumerate(COVERAGE):
                metric_values = values[:, column]
                row: dict[str, str | int] = {
                    "method": method,
                    "coverage_percent": int(coverage),
                    "cross_metric_mean": f"{metric_values.mean():.6f}",
                }
                row.update(
                    {metric: f"{value:.1f}" for metric, value in zip(METRICS, metric_values)}
                )
                writer.writerow(row)
    (OUT_DIR / "caption.txt").write_text(CAPTION + "\n", encoding="utf-8")


def main() -> None:
    apply_publication_style()
    write_source_data()
    series = plotted_series()

    # Compact panel suitable for a two-column multi-panel figure.
    fig, ax = plt.subplots(figsize=(89 / 25.4, 66 / 25.4))

    # Draw the random-rejection reference first so the empirical curves remain dominant.
    for name in ("Random rejection", "Oracle", "Ensemble uncertainty"):
        style = SERIES_STYLE[name]
        ax.plot(
            COVERAGE,
            series[name],
            label=name,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=1.35 if name == "Random rejection" else 1.55,
            markersize=4.0,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.55,
            zorder=1 if name == "Random rejection" else 3,
        )

    ax.set_xlim(48.5, 101.5)
    ax.set_ylim(0, 108)
    ax.set_xticks(COVERAGE.astype(int))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Coverage (%)", labelpad=4)
    ax.set_ylabel("Normalized MAE (%)", labelpad=5)
    ax.tick_params(axis="both", direction="out", length=2.5, width=0.65, pad=2.2)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    # Nature-style panel label, short title and unobtrusive statistic strip.
    ax.text(-0.17, 1.13, "a", transform=ax.transAxes, fontsize=8.5, fontweight="bold")
    ax.text(
        -0.04,
        1.13,
        "Cross-metric risk–coverage",
        transform=ax.transAxes,
        fontsize=7.2,
        fontweight="bold",
    )
    ax.text(
        0.5,
        1.035,
        "Mean across delay, angle, power and K-factor",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#555555",
    )

    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(name) for name in ("Ensemble uncertainty", "Oracle", "Random rejection")]
    ax.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=3,
        borderaxespad=0,
        handlelength=1.7,
        columnspacing=1.0,
        handletextpad=0.45,
        fontsize=5.55,
    )

    coverage_index = int(np.where(COVERAGE == 80)[0][0])
    point_y = series["Ensemble uncertainty"][coverage_index]
    reduction = 100.0 - point_y
    ax.annotate(
        f"Defer 20%\n{reduction:.0f}% lower risk",
        xy=(80, point_y),
        xytext=(69.0, 86.0),
        ha="left",
        va="center",
        fontsize=5.7,
        color=SERIES_STYLE["Ensemble uncertainty"]["color"],
        fontweight="bold",
        arrowprops={
            "arrowstyle": "-",
            "color": SERIES_STYLE["Ensemble uncertainty"]["color"],
            "linewidth": 0.8,
            "shrinkA": 1.5,
            "shrinkB": 3,
        },
    )

    ax.text(
        0.015,
        0.035,
        "Lower is better",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.5,
        color="#666666",
    )

    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.78)
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
