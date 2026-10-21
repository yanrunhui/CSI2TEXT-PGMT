#!/usr/bin/env python3
"""Create a Nature-style bar chart of error by uncertainty-percentile bin."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Mandatory publication settings: sans-serif text and editable SVG labels.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


OUT_DIR = Path("output/figures/uncertainty_bin_error_bars_nature")
STEM = OUT_DIR / "uncertainty_bin_error_bars_nature"
CAPTION = (
    "(b) Cross-metric normalized MAE within successive uncertainty-percentile bins. "
    "Bars show the mean across first-path delay, angle, power and K-factor; 100% "
    "denotes the corresponding full-coverage MAE."
)

COVERAGE = np.asarray([0.50, 0.60, 0.70, 0.80, 0.90, 1.00], dtype=float)
BIN_LABELS = ["0–50", "50–60", "60–70", "70–80", "80–90", "90–100"]
METRICS = ["First-path delay", "First-path angle", "First-path power", "K-factor"]

# Cumulative normalized risk after retaining the least-uncertain samples.
UNCERTAINTY_RISK = np.asarray(
    [
        [36.9, 39.9, 42.9, 47.8, 58.0, 100.0],
        [20.7, 31.6, 45.7, 61.2, 77.3, 100.0],
        [89.8, 90.0, 90.2, 91.6, 94.8, 100.0],
        [54.5, 61.1, 68.7, 74.9, 82.7, 100.0],
    ],
    dtype=float,
)


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
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )


def percentile_bin_risk() -> np.ndarray:
    """Recover the risk in each disjoint percentile bin from cumulative risks."""
    bin_risk = np.empty_like(UNCERTAINTY_RISK)
    bin_risk[:, 0] = UNCERTAINTY_RISK[:, 0]
    for index in range(1, len(COVERAGE)):
        bin_risk[:, index] = (
            COVERAGE[index] * UNCERTAINTY_RISK[:, index]
            - COVERAGE[index - 1] * UNCERTAINTY_RISK[:, index - 1]
        ) / (COVERAGE[index] - COVERAGE[index - 1])
    return bin_risk


def write_source_data(bin_risk: np.ndarray) -> None:
    """Export the metric-level bin risks and their plotted mean."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["uncertainty_percentile_bin", *METRICS, "cross_metric_mean"],
        )
        writer.writeheader()
        for index, label in enumerate(BIN_LABELS):
            metric_values = bin_risk[:, index]
            row: dict[str, str] = {
                "uncertainty_percentile_bin": label,
                "cross_metric_mean": f"{metric_values.mean():.6f}",
            }
            row.update(
                {metric: f"{value:.1f}" for metric, value in zip(METRICS, metric_values)}
            )
            writer.writerow(row)
    (OUT_DIR / "caption.txt").write_text(CAPTION + "\n", encoding="utf-8")


def main() -> None:
    apply_publication_style()
    bin_risk = percentile_bin_risk()
    mean_risk = bin_risk.mean(axis=0)
    write_source_data(bin_risk)

    # Match the single-panel dimensions used by the companion risk--coverage panel.
    fig, ax = plt.subplots(figsize=(89 / 25.4, 66 / 25.4))
    x = np.arange(len(BIN_LABELS))
    colors = ["#557A95"] * (len(BIN_LABELS) - 1) + ["#A9233F"]
    bars = ax.bar(
        x,
        mean_risk,
        width=0.66,
        color=colors,
        edgecolor="none",
        zorder=2,
    )

    ax.axhline(100, color="#7A7A7A", linewidth=0.75, linestyle=(0, (3, 2)), zorder=0)
    ax.text(
        -0.45,
        104,
        "Full coverage",
        ha="left",
        va="bottom",
        fontsize=5.5,
        color="#666666",
    )

    ax.set_xlim(-0.60, len(BIN_LABELS) - 0.40)
    ax.set_ylim(0, 330)
    ax.set_xticks(x, BIN_LABELS)
    ax.set_yticks([0, 100, 200, 300])
    ax.set_xlabel("Uncertainty percentile bin (%)", labelpad=4)
    ax.set_ylabel("Normalized MAE (%)", labelpad=5)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", direction="out", length=2.5, width=0.65, pad=2.2)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    ax.text(-0.17, 1.13, "b", transform=ax.transAxes, fontsize=8.5, fontweight="bold")
    ax.text(
        -0.04,
        1.13,
        "Error by uncertainty rank",
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

    for index, (bar, value) in enumerate(zip(bars, mean_risk)):
        color = colors[index]
        ax.annotate(
            f"{value:.0f}%",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.2,
            fontweight="bold" if index == len(bars) - 1 else "normal",
            color=color,
        )

    ax.text(
        0.02,
        0.965,
        "Lower is better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.5,
        color="#666666",
    )

    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.21, top=0.79)
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
