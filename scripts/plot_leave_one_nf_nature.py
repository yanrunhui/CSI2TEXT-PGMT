#!/usr/bin/env python3
"""Create a Nature-style leave-one-Nf-out generalization figure.

The source values are the reported held-out results and the seen-reference
baselines used by the original leave-one-Nf-out analysis.  The script exports
editable vector artwork, manuscript-ready raster files, and a tidy source-data
table for traceability.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Mandatory publication settings: sans-serif text and editable SVG labels.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


OUT_DIR = Path("output/figures/leave_one_nf_nature")
STEM = OUT_DIR / "leave_one_nf_generalization_nature"
HELD_OUT_NF = np.asarray([64, 96, 128, 192, 256], dtype=int)

# Raw values and baselines from the reported leave-one-Nf-out table.
# For accuracy/quality, degradation = (baseline - held out) / baseline.
# For MAE, degradation = (held out - baseline) / baseline.
SERIES = {
    "LoS/NLoS accuracy": {
        "baseline": 99.58,
        "values": [99.88, 99.84, 99.83, 99.76, 99.81],
        "higher_is_better": True,
        "panel": "a",
        "color": "#355C7D",
        "marker": "o",
    },
    "Text factuality": {
        "baseline": 96.43,
        "values": [96.54, 96.32, 95.67, 96.55, 95.74],
        "higher_is_better": True,
        "panel": "a",
        "color": "#3F8F7A",
        "marker": "s",
    },
    "Numeric accuracy": {
        "baseline": 82.61,
        "values": [79.88, 78.83, 78.64, 78.37, 76.39],
        "higher_is_better": True,
        "panel": "a",
        "color": "#C58B18",
        "marker": "^",
    },
    "K-factor MAE": {
        "baseline": 2.15,
        "values": [2.16, 2.28, 2.17, 2.16, 2.19],
        "higher_is_better": False,
        "panel": "a",
        "color": "#8B6B9C",
        "marker": "D",
    },
    "First-path angle MAE": {
        "baseline": 16.61,
        "values": [17.73, 18.23, 18.48, 19.21, 20.11],
        "higher_is_better": False,
        "panel": "b",
        "color": "#C76C3D",
        "marker": "o",
    },
    "First-path delay MAE": {
        "baseline": 21.91,
        "values": [44.00, 48.60, 52.09, 50.28, 57.11],
        "higher_is_better": False,
        "panel": "b",
        "color": "#A9233F",
        "marker": "s",
    },
    "First-path power MAE": {
        "baseline": 4.34,
        "values": [5.07, 5.44, 6.00, 6.17, 6.28],
        "higher_is_better": False,
        "panel": "b",
        "color": "#4B8DB8",
        "marker": "^",
    },
}


def degradation(values: list[float], baseline: float, higher_is_better: bool) -> np.ndarray:
    """Return degradation in percent; positive values always mean worse."""
    values_array = np.asarray(values, dtype=float)
    if higher_is_better:
        return (baseline - values_array) / baseline * 100.0
    return (values_array - baseline) / baseline * 100.0


def apply_publication_style() -> None:
    """Set compact, editable, print-oriented Matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.3,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.3,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "savefig.facecolor": "white",
        }
    )


def tidy_axes(ax: plt.Axes) -> None:
    """Apply restrained Nature-style axis treatment."""
    ax.tick_params(axis="both", which="major", direction="out", length=2.5, width=0.65, pad=2.2)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.axhline(0, color="#8A8A8A", linewidth=0.8, linestyle=(0, (2.2, 2.2)), zorder=0)
    ax.set_xlim(-0.08, 4.88)
    ax.set_xticks(np.arange(len(HELD_OUT_NF)), [str(value) for value in HELD_OUT_NF])
    ax.set_xlabel(r"Held-out frequency samples, $N_f$", labelpad=4)


def draw_series(ax: plt.Axes, names: list[str]) -> dict[str, np.ndarray]:
    """Plot each metric with color and marker-shape redundancy."""
    plotted: dict[str, np.ndarray] = {}
    x = np.arange(len(HELD_OUT_NF))
    for name in names:
        item = SERIES[name]
        values = degradation(item["values"], item["baseline"], item["higher_is_better"])
        plotted[name] = values
        ax.plot(
            x,
            values,
            color=item["color"],
            marker=item["marker"],
            markersize=5.0,
            markerfacecolor=item["color"],
            markeredgecolor="white",
            markeredgewidth=0.75,
            linewidth=2.0 if "delay" not in name.lower() else 2.35,
            zorder=3 if "delay" not in name.lower() else 4,
        )
    return plotted


def end_label(
    ax: plt.Axes,
    name: str,
    values: np.ndarray,
    label: str,
    dy: float = 0.0,
    weight: str = "semibold",
) -> None:
    """Label a line directly at its terminal value."""
    ax.annotate(
        label,
        xy=(4, values[-1]),
        xytext=(6, dy),
        textcoords="offset points",
        ha="left",
        va="center",
        color=SERIES[name]["color"],
        fontsize=7.0,
        fontweight=weight,
        annotation_clip=False,
    )


def write_source_data() -> None:
    """Export a tidy table containing raw, baseline, and derived values."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "panel",
                "metric",
                "held_out_nf",
                "seen_reference",
                "held_out_value",
                "higher_is_better",
                "degradation_percent",
            ],
        )
        writer.writeheader()
        for name, item in SERIES.items():
            drops = degradation(item["values"], item["baseline"], item["higher_is_better"])
            for nf, raw_value, drop in zip(HELD_OUT_NF, item["values"], drops):
                writer.writerow(
                    {
                        "panel": item["panel"],
                        "metric": name,
                        "held_out_nf": int(nf),
                        "seen_reference": f"{item['baseline']:.2f}",
                        "held_out_value": f"{raw_value:.2f}",
                        "higher_is_better": item["higher_is_better"],
                        "degradation_percent": f"{drop:.6f}",
                    }
                )


def main() -> None:
    apply_publication_style()
    write_source_data()

    # Double-column Nature width (183 mm); compact height preserves manuscript economy.
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(183 / 25.4, 78 / 25.4),
        gridspec_kw={"width_ratios": [1.00, 1.08]},
    )
    ax_left, ax_right = axes

    left_names = [
        "LoS/NLoS accuracy",
        "Text factuality",
        "Numeric accuracy",
        "K-factor MAE",
    ]
    left = draw_series(ax_left, left_names)
    tidy_axes(ax_left)
    ax_left.set_ylim(-1.0, 8.45)
    ax_left.set_yticks([0, 2, 4, 6, 8])
    ax_left.set_ylabel("Degradation relative to seen reference (%)", labelpad=5)
    ax_left.text(-0.14, 1.075, "a", transform=ax_left.transAxes, fontsize=8.5, fontweight="bold")
    ax_left.text(0.0, 1.075, "Task-level metrics", transform=ax_left.transAxes, fontsize=7.5, fontweight="bold")

    end_label(ax_left, "Numeric accuracy", left["Numeric accuracy"], "Numeric  7.5%", dy=0)
    end_label(ax_left, "K-factor MAE", left["K-factor MAE"], "K-factor  1.9%", dy=0)
    end_label(ax_left, "Text factuality", left["Text factuality"], "Factuality  0.7%", dy=1)
    end_label(ax_left, "LoS/NLoS accuracy", left["LoS/NLoS accuracy"], "LoS/NLoS  -0.2%", dy=-3)
    ax_left.text(
        0.98,
        0.96,
        "Task-level degradation remains <8%",
        transform=ax_left.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color=SERIES["Numeric accuracy"]["color"],
        fontweight="bold",
    )
    k_peak_index = int(np.argmax(left["K-factor MAE"]))
    ax_left.annotate(
        f"K-factor peak\n{left['K-factor MAE'][k_peak_index]:.1f}% at $N_f$={HELD_OUT_NF[k_peak_index]}",
        xy=(k_peak_index, left["K-factor MAE"][k_peak_index]),
        xytext=(0.26, 7.25),
        ha="left",
        va="center",
        fontsize=6.1,
        fontweight="semibold",
        color=SERIES["K-factor MAE"]["color"],
        arrowprops={
            "arrowstyle": "-|>",
            "color": SERIES["K-factor MAE"]["color"],
            "lw": 0.8,
            "mutation_scale": 6,
            "connectionstyle": "angle3,angleA=0,angleB=90",
            "shrinkA": 2,
            "shrinkB": 3,
        },
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.94},
        zorder=6,
    )
    los_nlos_max = np.max(np.abs(left["LoS/NLoS accuracy"]))
    ax_left.text(
        0.08,
        -0.78,
        f"LoS/NLoS remains stable  (|Δ| ≤ {los_nlos_max:.1f}%)",
        ha="left",
        va="center",
        fontsize=5.8,
        color=SERIES["LoS/NLoS accuracy"]["color"],
    )

    right_names = [
        "First-path angle MAE",
        "First-path delay MAE",
        "First-path power MAE",
    ]
    right = draw_series(ax_right, right_names)
    tidy_axes(ax_right)
    ax_right.set_ylim(0, 176)
    ax_right.set_yticks([0, 50, 100, 150])
    ax_right.text(-0.13, 1.075, "b", transform=ax_right.transAxes, fontsize=8.5, fontweight="bold")
    ax_right.text(0.0, 1.075, "First-path attributes", transform=ax_right.transAxes, fontsize=7.5, fontweight="bold")

    end_label(
        ax_right,
        "First-path delay MAE",
        right["First-path delay MAE"],
        "Delay  160.7%",
        dy=0,
        weight="bold",
    )
    end_label(ax_right, "First-path power MAE", right["First-path power MAE"], "Power  44.7%", dy=0)
    end_label(ax_right, "First-path angle MAE", right["First-path angle MAE"], "Angle  21.1%", dy=0)
    for index in [0, 2]:
        ax_right.annotate(
            f"{right['First-path delay MAE'][index]:.1f}%",
            xy=(index, right["First-path delay MAE"][index]),
            xytext=(4 if index == 0 else 0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.6,
            fontweight="bold",
            color=SERIES["First-path delay MAE"]["color"],
        )
    ax_right.text(
        0.03,
        0.96,
        "Delay dominates\n100.8–160.7% degradation",
        transform=ax_right.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color=SERIES["First-path delay MAE"]["color"],
        fontweight="bold",
        linespacing=1.15,
    )

    fig.subplots_adjust(left=0.09, right=0.915, bottom=0.20, top=0.86, wspace=0.34)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    for suffix in (".svg", ".pdf", ".tiff", ".png"):
        print(STEM.with_suffix(suffix).resolve())
    print((OUT_DIR / "source_data.csv").resolve())


if __name__ == "__main__":
    main()
