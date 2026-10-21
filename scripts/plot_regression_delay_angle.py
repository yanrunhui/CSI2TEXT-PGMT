#!/usr/bin/env python3
"""Plot only first-path delay and angle prediction against ground truth."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from plot_regression_pairs_static import (
    DEVIATION_COLOR,
    LOS_COLOR,
    METRICS,
    MUTED_COLOR,
    NLOS_COLOR,
    SOURCE,
    metric_stats,
    read_rows,
    stratified_sample,
)


ROOT = Path(__file__).resolve().parents[1]
STEM = ROOT / "output/figures/prediction_vs_ground_truth_ab"
PDF = ROOT / "output/pdf/prediction_vs_ground_truth_ab.pdf"


def draw_panel(ax: plt.Axes, sample: list[dict], all_rows: list[dict], metric: dict, letter: str) -> None:
    unit = metric["unit"]
    limits = (-500, 2000) if metric["key"] == "delay" else (-200, 200)
    ticks = [-500, 0, 500, 1000, 1500, 2000] if metric["key"] == "delay" else [-200, -100, 0, 100, 200]

    for status, color in (("nlos", NLOS_COLOR), ("los", LOS_COLOR)):
        points = [row for row in sample if row["los_status"] == status]
        ax.scatter(
            [row[metric["gt"]] for row in points],
            [row[metric["pred"]] for row in points],
            s=12,
            c=color,
            alpha=0.64,
            linewidths=0.32,
            edgecolors="white",
            zorder=3,
        )

    ax.plot(limits, limits, color="#252525", linewidth=1.0, linestyle=(0, (5, 4)), zorder=2)
    x_min, x_max, y_min, y_max = metric["nlos_deviation_box"]
    ax.add_patch(
        Rectangle(
            (x_min, y_min), x_max-x_min, y_max-y_min,
            facecolor="none", edgecolor=DEVIATION_COLOR, linewidth=1.7, zorder=4,
        )
    )
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_box_aspect(1)
    ax.grid(color="#D9DEE7", linewidth=0.5, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#A7AFBB")
        spine.set_linewidth(0.65)
    ax.tick_params(length=0, labelsize=10, colors="#5D6470", pad=3)
    ax.set_xlabel(f"Ground truth ({unit})", fontsize=11, labelpad=4)
    ax.set_ylabel(f"Prediction ({unit})", fontsize=11, labelpad=3)

    mae, pearson_r, spearman_rho = metric_stats(all_rows, metric)
    ax.text(0.5, 1.19, f"({letter}) {metric['title'][4:]}", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=15, fontweight="bold", clip_on=False)
    ax.text(0.02, 1.11, f"MAE = {mae:.2f} {unit}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10.4, color=MUTED_COLOR, clip_on=False)
    ax.text(0.02, 1.045, f"Pearson r = {pearson_r:.3f}   Spearman ρ = {spearman_rho:.3f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10.2,
            color=MUTED_COLOR, clip_on=False)


def main() -> None:
    all_rows = read_rows(SOURCE)
    sample = stratified_sample(all_rows)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.8))
    fig.subplots_adjust(left=0.075, right=0.982, bottom=0.145, top=0.72, wspace=0.18)
    for ax, metric, letter in zip(axes, METRICS[:2], "ab"):
        draw_panel(ax, sample, all_rows, metric, letter)

    fig.text(0.075, 0.972, "Prediction vs. Ground Truth", ha="left", va="top",
             fontsize=20, fontweight="bold", color="#20242A")
    handles = [
        Patch(facecolor="none", edgecolor=DEVIATION_COLOR, linewidth=1.5,
              label="NLoS deviation samples"),
        Line2D([], [], marker="o", linestyle="none", markersize=6.5,
               markerfacecolor=LOS_COLOR, markeredgecolor="none", label="LoS"),
        Line2D([], [], marker="o", linestyle="none", markersize=6.5,
               markerfacecolor=NLOS_COLOR, markeredgecolor="none", label="NLoS"),
        Line2D([], [], color="#252525", linestyle=(0, (5, 4)), linewidth=1.0, label="y = x"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.923),
               frameon=False, ncol=4, fontsize=14.0, handlelength=1.6,
               handletextpad=0.5, columnspacing=1.45, borderaxespad=0)

    n_los = sum(row["los_status"] == "los" for row in sample)
    fig.text(
        0.5, 0.038,
        f"Stratified sample: {len(sample)} points ({n_los} LoS, {len(sample)-n_los} NLoS), "
        f"seed 20260909. Metrics use all {len(all_rows)} valid samples; angle MAE uses circular error.",
        ha="center", va="bottom", fontsize=9.0, color=MUTED_COLOR,
    )

    STEM.parent.mkdir(parents=True, exist_ok=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(PDF, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(STEM.with_suffix(".png"), dpi=350, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(STEM.with_suffix(".svg"))
    print(PDF)
    print(STEM.with_suffix(".png"))


if __name__ == "__main__":
    main()
