#!/usr/bin/env python3
"""Create a Nature-style LoS-versus-NLoS MAE-ratio figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Mandatory publication settings: sans-serif text and editable SVG labels.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


OUT_DIR = Path("output/figures/los_nlos_gap_nature")
STEM = OUT_DIR / "los_nlos_mae_ratio_nature"

METRICS = [
    {
        "name": "First delay",
        "unit": "ns",
        "los_mean": 8.174,
        "los_sd": 0.156,
        "nlos_mean": 35.820,
        "nlos_sd": 0.599,
        "color": "#557A95",
    },
    {
        "name": "First angle",
        "unit": "°",
        "los_mean": 1.680,
        "los_sd": 0.043,
        "nlos_mean": 35.272,
        "nlos_sd": 0.748,
        "color": "#A9233F",
    },
    {
        "name": "First power",
        "unit": "dB",
        "los_mean": 2.973,
        "los_sd": 0.638,
        "nlos_mean": 6.966,
        "nlos_sd": 0.122,
        "color": "#7D98A1",
    },
]


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
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )


def ratio(metric: dict[str, float | str]) -> float:
    """Calculate the NLoS-to-LoS MAE ratio from the reported means."""
    return float(metric["nlos_mean"]) / float(metric["los_mean"])


def write_source_data() -> None:
    """Write the source means, standard deviations, and derived ratios."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "unit",
                "los_mean",
                "los_sd",
                "nlos_mean",
                "nlos_sd",
                "nlos_to_los_ratio",
            ],
        )
        writer.writeheader()
        for metric in METRICS:
            writer.writerow(
                {
                    "metric": metric["name"],
                    "unit": metric["unit"],
                    "los_mean": f"{metric['los_mean']:.3f}",
                    "los_sd": f"{metric['los_sd']:.3f}",
                    "nlos_mean": f"{metric['nlos_mean']:.3f}",
                    "nlos_sd": f"{metric['nlos_sd']:.3f}",
                    "nlos_to_los_ratio": f"{ratio(metric):.6f}",
                }
            )


def main() -> None:
    apply_publication_style()
    write_source_data()

    # Nature single-column width (89 mm). Text remains readable at final size.
    fig, ax = plt.subplots(figsize=(89 / 25.4, 72 / 25.4))
    x = np.arange(len(METRICS))
    values = np.asarray([ratio(metric) for metric in METRICS])
    colors = [str(metric["color"]) for metric in METRICS]

    bars = ax.bar(
        x,
        values,
        width=0.56,
        color=colors,
        edgecolor="none",
        zorder=2,
    )

    # A ratio of 1 denotes equal LoS and NLoS error; it is the scientific reference.
    ax.axhline(1.0, color="#8A8A8A", linewidth=0.65, linestyle=(0, (2.2, 2.2)), zorder=1)
    ax.text(
        -0.52,
        1.25,
        "parity",
        ha="left",
        va="bottom",
        fontsize=5.5,
        color="#707070",
    )

    ax.set_ylabel("NLoS / LoS MAE ratio", labelpad=5)
    ax.set_ylim(0, 23.8)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_xlim(-0.58, 2.58)
    ax.set_xticks(x, [str(metric["name"]) for metric in METRICS], fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", direction="out", length=2.5, width=0.65, pad=2.2)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    for index, (bar, metric, value) in enumerate(zip(bars, METRICS, values)):
        value_label = f"{value:.1f}×" if value >= 10 else f"{value:.2f}×"
        ax.annotate(
            value_label,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
            color=str(metric["color"]),
        )

        unit = str(metric["unit"])
        unit_prefix = "" if unit == "°" else " "
        summary = (
            f"LoS  {metric['los_mean']:.3f} ± {metric['los_sd']:.3f}{unit_prefix}{unit}\n"
            f"NLoS  {metric['nlos_mean']:.3f} ± {metric['nlos_sd']:.3f}{unit_prefix}{unit}"
        )
        ax.text(
            index,
            -0.17,
            summary,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=5.25,
            color="#404040",
            linespacing=1.22,
            clip_on=False,
        )

    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.31, top=0.95)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(STEM.with_suffix(".svg"))
    fig.savefig(STEM.with_suffix(".pdf"))
    fig.savefig(STEM.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(STEM.with_suffix(".png"), dpi=600)
    plt.close(fig)

    for suffix in (".svg", ".pdf", ".tiff", ".png"):
        print(STEM.with_suffix(suffix).resolve())
    print((OUT_DIR / "source_data.csv").resolve())


if __name__ == "__main__":
    main()
