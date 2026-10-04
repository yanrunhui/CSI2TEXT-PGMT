#!/usr/bin/env python3
"""Create a Nature-style two-panel AWGN robustness figure.

The figure emphasizes two manuscript claims:
1. NLoS delay is already strongly affected at moderate noise levels.
2. K-factor error dominates the severe-noise (0 dB) regime.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output/.mplconfig").resolve()))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "figures" / "awgn_robustness_nature"
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = ["Clean", "30 dB", "20 dB", "10 dB", "0 dB"]
X = np.arange(len(CONDITIONS), dtype=float)

DELAY = {
    "First-path": {
        "mean": np.array([25.97, 72.70, 86.11, 100.23, 209.29]),
        "sd": np.array([0.34, 0.40, 0.96, 1.73, 12.31]),
        "color": "#38558C",
        "marker": "o",
    },
    "LoS": {
        "mean": np.array([8.72, 20.30, 25.00, 34.10, 231.85]),
        "sd": np.array([0.49, 2.71, 2.43, 1.32, 28.52]),
        "color": "#D95A45",
        "marker": "s",
    },
    "NLoS": {
        "mean": np.array([43.22, 125.10, 147.22, 166.36, 186.73]),
        "sd": np.array([0.27, 3.05, 2.52, 4.31, 6.54]),
        "color": "#138F83",
        "marker": "^",
    },
}

PHYSICAL = {
    "Angle": {
        "mean": np.array([18.84, 23.10, 24.29, 25.12, 32.30]),
        "sd": np.array([0.40, 0.84, 0.71, 0.79, 1.92]),
        "color": "#38558C",
        "marker": "o",
    },
    "Power": {
        "mean": np.array([5.12, 4.88, 4.79, 5.38, 17.69]),
        "sd": np.array([0.35, 0.37, 0.37, 0.58, 0.08]),
        "color": "#D95A45",
        "marker": "s",
    },
    "K-factor": {
        "mean": np.array([2.084, 2.142, 2.612, 5.978, 19.855]),
        "sd": np.array([0.124, 0.125, 0.118, 0.582, 0.659]),
        "color": "#138F83",
        "marker": "^",
    },
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "semibold",
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def plot_series(ax: plt.Axes, series: dict[str, dict[str, object]]) -> None:
    for label, spec in series.items():
        mean = np.asarray(spec["mean"], dtype=float)
        sd = np.asarray(spec["sd"], dtype=float)
        ax.errorbar(
            X,
            mean,
            yerr=sd,
            label=label,
            color=str(spec["color"]),
            marker=str(spec["marker"]),
            lw=1.8,
            ms=4.8,
            mfc=str(spec["color"]),
            mec="white",
            mew=0.55,
            capsize=2.1,
            capthick=0.75,
            elinewidth=0.75,
            zorder=3,
        )


def export_source_data() -> None:
    rows: list[dict[str, object]] = []
    for metric, spec in DELAY.items():
        for condition, mean, sd in zip(CONDITIONS, spec["mean"], spec["sd"]):
            rows.append(
                {
                    "panel": "a",
                    "metric": metric,
                    "condition": condition,
                    "mean": float(mean),
                    "sd": float(sd),
                    "plotted_value": float(mean),
                    "plotted_sd": float(sd),
                    "normalization": "none",
                }
            )
    for metric, spec in PHYSICAL.items():
        clean = float(spec["mean"][0])
        for condition, mean, sd in zip(CONDITIONS, spec["mean"], spec["sd"]):
            rows.append(
                {
                    "panel": "b",
                    "metric": metric,
                    "condition": condition,
                    "mean": float(mean),
                    "sd": float(sd),
                    "plotted_value": float(mean) / clean,
                    "plotted_sd": float(sd) / clean,
                    "normalization": "condition mean / clean-condition mean",
                }
            )
    with (OUT / "source_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_figure() -> plt.Figure:
    configure_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.20, 3.15))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.19, top=0.79, wspace=0.27)

    # Panel a: absolute delay MAE. The pale band marks the moderate-noise regime.
    ax_a.axvspan(0.55, 3.45, color="#EAF4F1", zorder=0)
    plot_series(ax_a, DELAY)
    ax_a.set_xlim(-0.12, 4.12)
    ax_a.set_ylim(0, 278)
    ax_a.set_yticks([0, 50, 100, 150, 200, 250])
    ax_a.set_xticks(X, CONDITIONS)
    ax_a.set_ylabel("Delay MAE (ns)")
    ax_a.set_xlabel("Noise condition (SNR)")
    ax_a.set_title("Delay error under noise", loc="left", pad=33)
    ax_a.text(-0.16, 1.22, "a", transform=ax_a.transAxes, fontsize=10, fontweight="bold", va="top")
    ax_a.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.035),
        ncol=3,
        handlelength=1.45,
        columnspacing=1.05,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    ax_a.text(
        2.0,
        265,
        "moderate-noise regime",
        ha="center",
        va="top",
        fontsize=6.5,
        color="#58736E",
    )

    nlos_rise = (DELAY["NLoS"]["mean"][1] / DELAY["NLoS"]["mean"][0] - 1.0) * 100.0
    ax_a.annotate(
        f"NLoS is noise-sensitive\n+{nlos_rise:.0f}% at 30 dB",
        xy=(1, DELAY["NLoS"]["mean"][1]),
        xytext=(0.55, 207),
        color=DELAY["NLoS"]["color"],
        fontsize=7.0,
        fontweight="semibold",
        ha="left",
        va="center",
        arrowprops={
            "arrowstyle": "-|>",
            "color": DELAY["NLoS"]["color"],
            "lw": 0.9,
            "mutation_scale": 7,
            "shrinkA": 3,
            "shrinkB": 3,
            "connectionstyle": "angle3,angleA=0,angleB=90",
        },
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "none", "alpha": 0.94},
        zorder=5,
    )
    ax_a.annotate(
        "231.9 ± 28.5 ns",
        xy=(4, DELAY["LoS"]["mean"][4]),
        xytext=(3.60, 260),
        color=DELAY["LoS"]["color"],
        fontsize=7.0,
        fontweight="semibold",
        ha="right",
        va="bottom",
        arrowprops={"arrowstyle": "-", "color": DELAY["LoS"]["color"], "lw": 0.9},
    )

    # Panel b: errors normalized to their respective clean-condition means.
    normalized: dict[str, dict[str, object]] = {}
    for metric, spec in PHYSICAL.items():
        clean = float(spec["mean"][0])
        normalized[metric] = {
            **spec,
            "mean": np.asarray(spec["mean"], dtype=float) / clean,
            "sd": np.asarray(spec["sd"], dtype=float) / clean,
        }
    ax_b.axvspan(3.65, 4.12, color="#FAECE8", zorder=0)
    plot_series(ax_b, normalized)
    ax_b.axhline(1.0, color="#777777", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax_b.text(
        0.10,
        0.55,
        "clean baseline",
        color="#666666",
        fontsize=6.2,
        va="top",
        bbox={"boxstyle": "square,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.9},
    )
    ax_b.set_xlim(-0.12, 4.12)
    ax_b.set_ylim(0, 10.45)
    ax_b.set_yticks([0, 2, 4, 6, 8, 10])
    ax_b.set_xticks(X, CONDITIONS)
    ax_b.set_ylabel("MAE relative to clean (×)")
    ax_b.set_xlabel("Noise condition (SNR)")
    ax_b.set_title("Relative physical error", loc="left", pad=33)
    ax_b.text(-0.16, 1.22, "b", transform=ax_b.transAxes, fontsize=10, fontweight="bold", va="top")
    ax_b.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.035),
        ncol=3,
        handlelength=1.45,
        columnspacing=1.05,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    k_ratio = normalized["K-factor"]["mean"][4]
    power_ratio = normalized["Power"]["mean"][4]
    angle_ratio = normalized["Angle"]["mean"][4]
    ax_b.scatter(
        [4],
        [k_ratio],
        s=76,
        facecolors="none",
        edgecolors="#B21F2D",
        linewidths=1.15,
        zorder=5,
    )
    ax_b.annotate(
        f"0 dB: K-factor {k_ratio:.2f}×",
        xy=(4, k_ratio),
        xytext=(2.72, 8.56),
        color="#B21F2D",
        fontsize=7.0,
        fontweight="semibold",
        ha="right",
        va="center",
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#B21F2D",
            "lw": 0.9,
            "mutation_scale": 7,
            "shrinkA": 3,
            "shrinkB": 3,
        },
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": "none", "alpha": 0.94},
        zorder=5,
    )
    ax_b.text(
        3.93,
        4.18,
        f"Power  {power_ratio:.2f}×\nAngle   {angle_ratio:.2f}×",
        ha="right",
        va="bottom",
        fontsize=6.4,
        color="#5C5C5C",
        linespacing=1.35,
    )

    for ax in (ax_a, ax_b):
        ax.tick_params(direction="out", pad=2.5)
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")
        ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.035,
        "Points show the mean ± s.d. across three trained model seeds; each seed is averaged over three test-noise realizations.",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#5A5A5A",
    )
    return fig


def main() -> None:
    export_source_data()
    fig = build_figure()
    stem = OUT / "awgn_robustness_nature"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(stem)


if __name__ == "__main__":
    main()
