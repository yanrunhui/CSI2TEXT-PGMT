#!/usr/bin/env python3
"""Evaluate three checkpoints and analyse per-sample ensemble uncertainty.

The script deliberately reuses ``scripts/evaluate.py --save-regression-pairs``
so that every prediction has exactly the same definition as the paper's
reported metrics.  It can run inference for several seed directories and then
produce risk--coverage statistics, random-rejection baselines, uncertainty
bins, and publication-ready figures.

Typical usage::

    python scripts/evaluate_ensemble_uncertainty.py \
        --seed-dirs /path/to/full_multitask/seed_0 \
                    /path/to/full_multitask/seed_1 \
                    /path/to/full_multitask/seed_2 \
        --data-path /path/to/test.pt \
        --output-dir /path/to/uncertainty_results

Pass ``--skip-inference`` to rebuild the analysis from the cached
``predictions/seed_*.pt`` payloads without evaluating the models again.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]

METRICS = (
    "first_path_delay_ns",
    "first_path_angle_deg",
    "first_path_power_dbw",
    "k_factor_db",
)

METRIC_LABELS = {
    "first_path_delay_ns": "First-path delay",
    "first_path_angle_deg": "First-path angle",
    "first_path_power_dbw": "First-path power",
    "k_factor_db": "K-factor",
}

METRIC_UNITS = {
    "first_path_delay_ns": "ns",
    "first_path_angle_deg": "deg",
    "first_path_power_dbw": "dBW",
    "k_factor_db": "dB",
}

SUBSETS = ("all", "los", "nlos")


def _checkpoint_epoch(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return -1


def discover_checkpoint(seed_path: Path, checkpoint_epoch: int | None) -> Path:
    """Resolve a checkpoint file from a seed directory or explicit file."""
    if seed_path.is_file():
        return seed_path
    if not seed_path.is_dir():
        raise FileNotFoundError(f"Seed directory does not exist: {seed_path}")

    if checkpoint_epoch is not None:
        preferred = seed_path / f"checkpoint_epoch_{checkpoint_epoch}.pt"
        if preferred.is_file():
            return preferred
        raise FileNotFoundError(
            f"Requested checkpoint epoch {checkpoint_epoch} is missing: {preferred}"
        )

    last = seed_path / "checkpoint_last.pt"
    if last.is_file():
        return last

    epoch_checkpoints = sorted(
        seed_path.glob("checkpoint_epoch_*.pt"),
        key=_checkpoint_epoch,
    )
    if epoch_checkpoints:
        return epoch_checkpoints[-1]
    raise FileNotFoundError(
        f"No checkpoint_last.pt or checkpoint_epoch_*.pt found under {seed_path}"
    )


def unique_seed_names(seed_paths: list[Path]) -> list[str]:
    names: list[str] = []
    used: set[str] = set()
    for index, path in enumerate(seed_paths):
        base = path.parent.name if path.is_file() else path.name
        base = base or f"seed_{index}"
        name = base
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        names.append(name)
        used.add(name)
    return names


def run_inference(
    *,
    seed_paths: list[Path],
    seed_names: list[str],
    data_path: Path,
    output_dir: Path,
    batch_size: int,
    checkpoint_epoch: int | None,
) -> list[Path]:
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    payload_paths: list[Path] = []

    for seed_path, seed_name in zip(seed_paths, seed_names):
        checkpoint = discover_checkpoint(seed_path, checkpoint_epoch)
        csv_path = prediction_dir / f"{seed_name}.csv"
        log_path = prediction_dir / f"{seed_name}_evaluate.log"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate.py"),
            "--data-path",
            str(data_path),
            "--checkpoint",
            str(checkpoint),
            "--batch-size",
            str(batch_size),
            "--physical-caption-examples",
            "0",
            "--save-regression-pairs",
            str(csv_path),
        ]
        print(f"[{seed_name}] checkpoint={checkpoint}", flush=True)
        print(f"[{seed_name}] log={log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_handle.write(line)
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"Evaluation failed for {seed_name}; see {log_path}"
            )
        payload_path = csv_path.with_suffix(".pt")
        if not payload_path.is_file():
            raise FileNotFoundError(
                f"Evaluation completed without the expected payload: {payload_path}"
            )
        payload_paths.append(payload_path)
    return payload_paths


def cached_payload_paths(output_dir: Path, seed_names: list[str]) -> list[Path]:
    paths = [output_dir / "predictions" / f"{name}.pt" for name in seed_names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "--skip-inference was used, but cached prediction payloads are missing:\n"
            + "\n".join(missing)
        )
    return paths


def load_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "sample_count",
        "sample_index",
        "group_ids",
        "config_keys",
        "los_status",
        "predictions",
        "ground_truth",
        "valid_masks",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{path} is missing payload fields: {sorted(missing)}")
    return payload


def as_numpy(value, *, dtype=None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def validate_alignment(payloads: list[dict], paths: list[Path]) -> None:
    if len(payloads) < 2:
        raise ValueError("At least two seed prediction payloads are required.")
    reference = payloads[0]
    reference_count = int(reference["sample_count"])
    reference_index = as_numpy(reference["sample_index"], dtype=np.int64)
    reference_status = [str(value).lower() for value in reference["los_status"]]

    for payload, path in zip(payloads[1:], paths[1:]):
        if int(payload["sample_count"]) != reference_count:
            raise ValueError(
                f"Sample count mismatch: {paths[0]} has {reference_count}, "
                f"but {path} has {payload['sample_count']}."
            )
        if not np.array_equal(
            as_numpy(payload["sample_index"], dtype=np.int64), reference_index
        ):
            raise ValueError(f"Sample-index order differs between {paths[0]} and {path}.")
        if list(payload["group_ids"]) != list(reference["group_ids"]):
            raise ValueError(f"group_ids differ between {paths[0]} and {path}.")
        if list(payload["config_keys"]) != list(reference["config_keys"]):
            raise ValueError(f"config_keys differ between {paths[0]} and {path}.")
        status = [str(value).lower() for value in payload["los_status"]]
        if status != reference_status:
            raise ValueError(f"LoS/NLoS labels differ between {paths[0]} and {path}.")

        for metric in METRICS:
            if metric not in payload["predictions"]:
                raise ValueError(f"{path} has no predictions for {metric}.")
            ref_target = as_numpy(reference["ground_truth"][metric], dtype=np.float64)
            target = as_numpy(payload["ground_truth"][metric], dtype=np.float64)
            if not np.allclose(ref_target, target, rtol=0.0, atol=1e-6, equal_nan=True):
                raise ValueError(
                    f"Ground truth for {metric} differs between {paths[0]} and {path}."
                )


def circular_difference_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return (first - second + 180.0) % 360.0 - 180.0


def ensemble_prediction_and_uncertainty(
    seed_predictions: np.ndarray,
    *,
    circular: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ensemble mean and sample standard deviation along the seed axis."""
    if circular:
        radians = np.deg2rad(seed_predictions)
        prediction = np.rad2deg(
            np.arctan2(np.mean(np.sin(radians), axis=0), np.mean(np.cos(radians), axis=0))
        )
        residuals = circular_difference_deg(seed_predictions, prediction[None, :])
        uncertainty = np.sqrt(
            np.sum(np.square(residuals), axis=0) / (seed_predictions.shape[0] - 1)
        )
        return prediction, uncertainty
    return (
        np.mean(seed_predictions, axis=0),
        np.std(seed_predictions, axis=0, ddof=1),
    )


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if len(first) < 2:
        return float("nan")
    first = first - np.mean(first)
    second = second - np.mean(second)
    denominator = math.sqrt(float(first @ first) * float(second @ second))
    if denominator <= 0.0:
        return float("nan")
    return float((first @ second) / denominator)


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    return pearson(average_ranks(first), average_ranks(second))


def subset_mask(status: np.ndarray, subset: str) -> np.ndarray:
    if subset == "all":
        return np.ones(len(status), dtype=bool)
    return status == subset


def coverage_values(
    minimum: float,
    step: float,
    rejection_rates: Iterable[float],
) -> list[float]:
    values: list[float] = []
    current = minimum
    while current < 1.0 - 1e-10:
        values.append(round(current, 10))
        current += step
    values.append(1.0)
    values.extend(1.0 - rate for rate in rejection_rates)
    return sorted({min(1.0, max(minimum, value)) for value in values})


def bootstrap_mean_interval(
    values: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if len(values) <= 1 or repeats <= 0:
        mean = float(np.mean(values))
        return mean, mean
    means = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        means[index] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def compute_risk_coverage(
    errors: np.ndarray,
    uncertainty: np.ndarray,
    coverages: list[float],
    random_repeats: int,
    rng: np.random.Generator,
) -> list[dict]:
    count = len(errors)
    keep_counts = sorted(
        {max(1, min(count, int(round(coverage * count)))) for coverage in coverages}
    )
    uncertainty_order = np.lexsort((np.arange(count), uncertainty))
    oracle_order = np.lexsort((np.arange(count), errors))
    uncertainty_cumsum = np.cumsum(errors[uncertainty_order], dtype=np.float64)
    oracle_cumsum = np.cumsum(errors[oracle_order], dtype=np.float64)

    random_risks = np.empty((random_repeats, len(keep_counts)), dtype=np.float64)
    for repeat in range(random_repeats):
        random_cumsum = np.cumsum(errors[rng.permutation(count)], dtype=np.float64)
        for column, keep_count in enumerate(keep_counts):
            random_risks[repeat, column] = random_cumsum[keep_count - 1] / keep_count

    rows: list[dict] = []
    for column, keep_count in enumerate(keep_counts):
        if random_repeats > 0:
            random_mean = float(np.mean(random_risks[:, column]))
            random_low, random_high = np.quantile(
                random_risks[:, column], [0.025, 0.975]
            )
        else:
            random_mean = float(np.mean(errors))
            random_low = random_high = random_mean
        rows.append(
            {
                "coverage": keep_count / count,
                "retained_count": keep_count,
                "rejected_count": count - keep_count,
                "uncertainty_risk": float(uncertainty_cumsum[keep_count - 1] / keep_count),
                "random_risk_mean": random_mean,
                "random_risk_low95": float(random_low),
                "random_risk_high95": float(random_high),
                "oracle_risk": float(oracle_cumsum[keep_count - 1] / keep_count),
            }
        )
    return rows


def compute_uncertainty_bins(
    errors: np.ndarray,
    uncertainty: np.ndarray,
    bin_count: int,
    bootstrap_repeats: int,
    rng: np.random.Generator,
) -> list[dict]:
    order = np.lexsort((np.arange(len(errors)), uncertainty))
    chunks = np.array_split(order, min(bin_count, len(order)))
    rows: list[dict] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_errors = errors[chunk]
        low, high = bootstrap_mean_interval(chunk_errors, bootstrap_repeats, rng)
        rows.append(
            {
                "bin": index,
                "count": len(chunk),
                "uncertainty_mean": float(np.mean(uncertainty[chunk])),
                "uncertainty_min": float(np.min(uncertainty[chunk])),
                "uncertainty_max": float(np.max(uncertainty[chunk])),
                "error_mean": float(np.mean(chunk_errors)),
                "error_low95": low,
                "error_high95": high,
            }
        )
    return rows


def normalized_aurc(rows: list[dict]) -> float:
    rows = sorted(rows, key=lambda row: row["coverage"])
    coverage = np.asarray([row["coverage"] for row in rows], dtype=np.float64)
    risk = np.asarray([row["uncertainty_risk"] for row in rows], dtype=np.float64)
    width = float(coverage[-1] - coverage[0])
    if width <= 0.0:
        return float(risk[-1])
    return float(np.trapz(risk, coverage) / width)


def row_at_coverage(rows: list[dict], desired: float) -> dict:
    return min(rows, key=lambda row: abs(row["coverage"] - desired))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyse(
    *,
    payload_paths: list[Path],
    seed_names: list[str],
    output_dir: Path,
    minimum_coverage: float,
    coverage_step: float,
    rejection_rates: list[float],
    random_repeats: int,
    bootstrap_repeats: int,
    random_seed: int,
    bin_count: int,
    primary_metric: str,
    primary_subset: str,
) -> None:
    payloads = [load_payload(path) for path in payload_paths]
    validate_alignment(payloads, payload_paths)
    reference = payloads[0]
    sample_indices = as_numpy(reference["sample_index"], dtype=np.int64)
    status = np.asarray([str(value).lower() for value in reference["los_status"]])
    group_ids = [str(value) for value in reference["group_ids"]]
    config_keys = [str(value) for value in reference["config_keys"]]
    coverages = coverage_values(minimum_coverage, coverage_step, rejection_rates)

    sample_rows: list[dict] = []
    risk_rows: list[dict] = []
    bin_rows: list[dict] = []
    summary_rows: list[dict] = []
    results: dict[tuple[str, str], dict] = {}

    for metric_index, metric in enumerate(METRICS):
        predictions = np.stack(
            [
                as_numpy(payload["predictions"][metric], dtype=np.float64)
                for payload in payloads
            ],
            axis=0,
        )
        target = as_numpy(reference["ground_truth"][metric], dtype=np.float64)
        valid = np.logical_and.reduce(
            [
                as_numpy(payload["valid_masks"][metric], dtype=bool)
                for payload in payloads
            ]
        )
        valid &= np.isfinite(target) & np.isfinite(predictions).all(axis=0)
        circular = metric == "first_path_angle_deg"
        ensemble_prediction, uncertainty = ensemble_prediction_and_uncertainty(
            predictions,
            circular=circular,
        )
        if circular:
            errors = np.abs(circular_difference_deg(ensemble_prediction, target))
        else:
            errors = np.abs(ensemble_prediction - target)

        for sample_position in np.flatnonzero(valid):
            row = {
                "sample_index": int(sample_indices[sample_position]),
                "group_id": group_ids[sample_position],
                "config_key": config_keys[sample_position],
                "los_status": status[sample_position],
                "metric": metric,
                "ground_truth": float(target[sample_position]),
                "ensemble_prediction": float(ensemble_prediction[sample_position]),
                "uncertainty": float(uncertainty[sample_position]),
                "absolute_error": float(errors[sample_position]),
            }
            for seed_name, seed_predictions in zip(seed_names, predictions):
                row[f"prediction_{seed_name}"] = float(seed_predictions[sample_position])
            sample_rows.append(row)

        for subset_index, subset in enumerate(SUBSETS):
            selected = valid & subset_mask(status, subset)
            selected_errors = errors[selected]
            selected_uncertainty = uncertainty[selected]
            if len(selected_errors) < 2:
                print(
                    f"warning: skipping metric={metric}, subset={subset}; "
                    f"only {len(selected_errors)} valid samples",
                    file=sys.stderr,
                )
                continue

            rng_offset = metric_index * 100_000 + subset_index * 10_000
            risk = compute_risk_coverage(
                selected_errors,
                selected_uncertainty,
                coverages,
                random_repeats,
                np.random.default_rng(random_seed + rng_offset),
            )
            bins = compute_uncertainty_bins(
                selected_errors,
                selected_uncertainty,
                bin_count,
                bootstrap_repeats,
                np.random.default_rng(random_seed + rng_offset + 1),
            )
            for row in risk:
                risk_rows.append({"metric": metric, "subset": subset, **row})
            for row in bins:
                bin_rows.append({"metric": metric, "subset": subset, **row})

            full = row_at_coverage(risk, 1.0)
            summary = {
                "metric": metric,
                "subset": subset,
                "sample_count": len(selected_errors),
                "full_coverage_mae": full["uncertainty_risk"],
                "spearman_uncertainty_error": spearman(
                    selected_uncertainty, selected_errors
                ),
                "normalized_aurc": normalized_aurc(risk),
            }
            for rejection_rate in rejection_rates:
                desired_coverage = 1.0 - rejection_rate
                coverage_row = row_at_coverage(risk, desired_coverage)
                suffix = f"reject_{int(round(rejection_rate * 100))}pct"
                summary[f"{suffix}_actual_coverage"] = coverage_row["coverage"]
                summary[f"{suffix}_mae"] = coverage_row["uncertainty_risk"]
                summary[f"{suffix}_relative_improvement_pct"] = (
                    100.0
                    * (full["uncertainty_risk"] - coverage_row["uncertainty_risk"])
                    / full["uncertainty_risk"]
                    if full["uncertainty_risk"] > 0.0
                    else float("nan")
                )
                summary[f"{suffix}_random_mae"] = coverage_row["random_risk_mean"]
            summary_rows.append(summary)
            results[(metric, subset)] = {"risk": risk, "bins": bins}

    write_csv(output_dir / "per_sample_uncertainty.csv", sample_rows)
    write_csv(output_dir / "risk_coverage.csv", risk_rows)
    write_csv(output_dir / "uncertainty_bins.csv", bin_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    plot_results(
        results=results,
        output_dir=output_dir,
        rejection_rates=rejection_rates,
        primary_metric=primary_metric,
        primary_subset=primary_subset,
    )

    print(f"saved_per_sample={output_dir / 'per_sample_uncertainty.csv'}")
    print(f"saved_risk_coverage={output_dir / 'risk_coverage.csv'}")
    print(f"saved_uncertainty_bins={output_dir / 'uncertainty_bins.csv'}")
    print(f"saved_summary={output_dir / 'summary.csv'}")
    primary_summary = next(
        (
            row
            for row in summary_rows
            if row["metric"] == primary_metric and row["subset"] == primary_subset
        ),
        None,
    )
    if primary_summary is not None:
        print(
            "primary_result="
            + json.dumps(primary_summary, ensure_ascii=False, allow_nan=True)
        )


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib. Install it with: pip install matplotlib"
        ) from exc
    return plt


def _save_figure(fig, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")


def plot_results(
    *,
    results: dict[tuple[str, str], dict],
    output_dir: Path,
    rejection_rates: list[float],
    primary_metric: str,
    primary_subset: str,
) -> None:
    plt = _load_matplotlib()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    primary_key = (primary_metric, primary_subset)
    if primary_key not in results:
        raise ValueError(
            f"No valid result for primary metric/subset: {primary_metric}/{primary_subset}"
        )
    primary = results[primary_key]
    risk = sorted(primary["risk"], key=lambda row: row["coverage"])
    bins = primary["bins"]
    unit = METRIC_UNITS[primary_metric]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    coverage_pct = np.asarray([row["coverage"] * 100.0 for row in risk])
    axes[0].plot(
        coverage_pct,
        [row["uncertainty_risk"] for row in risk],
        marker="o",
        linewidth=2.2,
        label="Ensemble uncertainty",
    )
    axes[0].plot(
        coverage_pct,
        [row["random_risk_mean"] for row in risk],
        linestyle="--",
        linewidth=2.0,
        label="Random rejection",
    )
    axes[0].fill_between(
        coverage_pct,
        [row["random_risk_low95"] for row in risk],
        [row["random_risk_high95"] for row in risk],
        alpha=0.16,
    )
    axes[0].plot(
        coverage_pct,
        [row["oracle_risk"] for row in risk],
        linestyle=":",
        linewidth=2.0,
        label="Oracle (reference)",
    )
    axes[0].set_xlabel("Coverage (%)")
    axes[0].set_ylabel(f"MAE / risk ({unit})")
    axes[0].set_title("(a) Risk–coverage")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(alpha=0.25)

    uncertainty_mean = np.asarray([row["uncertainty_mean"] for row in bins])
    error_mean = np.asarray([row["error_mean"] for row in bins])
    error_low = np.asarray([row["error_low95"] for row in bins])
    error_high = np.asarray([row["error_high95"] for row in bins])
    axes[1].errorbar(
        uncertainty_mean,
        error_mean,
        yerr=np.vstack(
            [
                np.maximum(error_mean - error_low, 0.0),
                np.maximum(error_high - error_mean, 0.0),
            ]
        ),
        marker="o",
        linewidth=2.2,
        capsize=3,
    )
    axes[1].set_xlabel(f"Predicted uncertainty ({unit})")
    axes[1].set_ylabel(f"Mean absolute error ({unit})")
    axes[1].set_title("(b) Error vs. uncertainty")
    axes[1].grid(alpha=0.25)

    plot_rejections = sorted(set([0.0, *rejection_rates]))
    selected_rows = [row_at_coverage(risk, 1.0 - rate) for rate in plot_rejections]
    x = np.arange(len(plot_rejections))
    width = 0.38
    axes[2].bar(
        x - width / 2,
        [row["uncertainty_risk"] for row in selected_rows],
        width,
        label="Ensemble uncertainty",
    )
    axes[2].bar(
        x + width / 2,
        [row["random_risk_mean"] for row in selected_rows],
        width,
        label="Random rejection",
    )
    axes[2].set_xticks(
        x,
        [f"{int(round(rate * 100))}%" for rate in plot_rejections],
    )
    axes[2].set_xlabel("Rejected highest-uncertainty samples")
    axes[2].set_ylabel(f"MAE ({unit})")
    axes[2].set_title("(c) MAE after rejection")
    axes[2].legend(frameon=False, fontsize=9)
    axes[2].grid(axis="y", alpha=0.25)

    subset_title = {"all": "All", "los": "LoS", "nlos": "NLoS"}[primary_subset]
    fig.suptitle(
        f"{subset_title} {METRIC_LABELS[primary_metric]}: selective prediction",
        fontsize=14,
    )
    fig.tight_layout()
    _save_figure(fig, figure_dir / f"primary_{primary_subset}_{primary_metric}")
    plt.close(fig)

    fig, axes = plt.subplots(
        len(METRICS),
        len(SUBSETS),
        figsize=(13.5, 12.0),
        squeeze=False,
        sharex=True,
    )
    for row_index, metric in enumerate(METRICS):
        for column_index, subset in enumerate(SUBSETS):
            axis = axes[row_index][column_index]
            result = results.get((metric, subset))
            if result is None:
                axis.set_axis_off()
                continue
            rows = sorted(result["risk"], key=lambda row: row["coverage"])
            x_values = [row["coverage"] * 100.0 for row in rows]
            axis.plot(
                x_values,
                [row["uncertainty_risk"] for row in rows],
                marker="o",
                markersize=3,
                label="Uncertainty",
            )
            axis.plot(
                x_values,
                [row["random_risk_mean"] for row in rows],
                linestyle="--",
                label="Random",
            )
            axis.plot(
                x_values,
                [row["oracle_risk"] for row in rows],
                linestyle=":",
                label="Oracle",
            )
            axis.set_title(
                f"{METRIC_LABELS[metric]} — "
                f"{ {'all': 'All', 'los': 'LoS', 'nlos': 'NLoS'}[subset] }"
            )
            axis.set_ylabel(f"MAE ({METRIC_UNITS[metric]})")
            axis.grid(alpha=0.22)
            if row_index == len(METRICS) - 1:
                axis.set_xlabel("Coverage (%)")
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, figure_dir / "risk_coverage_all_metrics")
    plt.close(fig)


def parse_fraction_list(values: list[float], name: str, *, allow_zero: bool) -> list[float]:
    lower = 0.0 if allow_zero else 0.0
    for value in values:
        if value < lower or value >= 1.0 or (not allow_zero and value <= 0.0):
            comparator = "[0, 1)" if allow_zero else "(0, 1)"
            raise ValueError(f"{name} values must lie in {comparator}; got {value}")
    return sorted(set(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple seed checkpoints on one test set and evaluate whether "
            "per-sample ensemble disagreement predicts regression error."
        )
    )
    parser.add_argument(
        "--seed-dirs",
        nargs="+",
        required=True,
        help="Seed checkpoint directories, or explicit checkpoint .pt files.",
    )
    parser.add_argument("--data-path", required=True, help="Shared test-set .pt path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        help=(
            "Use checkpoint_epoch_N.pt in every seed directory. By default, "
            "checkpoint_last.pt is preferred, otherwise the largest epoch is used."
        ),
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Reuse output-dir/predictions/seed_name.pt files and only rebuild analysis.",
    )
    parser.add_argument("--minimum-coverage", type=float, default=0.50)
    parser.add_argument("--coverage-step", type=float, default=0.05)
    parser.add_argument(
        "--rejection-rates",
        nargs="+",
        type=float,
        default=(0.05, 0.10, 0.20),
        help="Discrete rejection rates highlighted in the summary and main figure.",
    )
    parser.add_argument("--random-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=20260909)
    parser.add_argument("--uncertainty-bins", type=int, default=10)
    parser.add_argument(
        "--primary-metric",
        choices=METRICS,
        default="first_path_delay_ns",
    )
    parser.add_argument(
        "--primary-subset",
        choices=SUBSETS,
        default="nlos",
    )
    args = parser.parse_args()

    seed_paths = [Path(path).expanduser().resolve() for path in args.seed_dirs]
    if len(seed_paths) < 2:
        parser.error("--seed-dirs requires at least two checkpoints; three are recommended.")
    data_path = Path(args.data_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not 0.0 < args.minimum_coverage <= 1.0:
        parser.error("--minimum-coverage must lie in (0, 1].")
    if args.coverage_step <= 0.0:
        parser.error("--coverage-step must be positive.")
    if args.random_repeats <= 0:
        parser.error("--random-repeats must be positive.")
    if args.bootstrap_repeats < 0:
        parser.error("--bootstrap-repeats cannot be negative.")
    if args.uncertainty_bins <= 0:
        parser.error("--uncertainty-bins must be positive.")
    try:
        rejection_rates = parse_fraction_list(
            list(args.rejection_rates),
            "--rejection-rates",
            allow_zero=True,
        )
    except ValueError as exc:
        parser.error(str(exc))

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_names = unique_seed_names(seed_paths)
    if args.skip_inference:
        payload_paths = cached_payload_paths(output_dir, seed_names)
    else:
        if not data_path.is_file():
            raise FileNotFoundError(f"Test data does not exist: {data_path}")
        payload_paths = run_inference(
            seed_paths=seed_paths,
            seed_names=seed_names,
            data_path=data_path,
            output_dir=output_dir,
            batch_size=args.batch_size,
            checkpoint_epoch=args.checkpoint_epoch,
        )

    analyse(
        payload_paths=payload_paths,
        seed_names=seed_names,
        output_dir=output_dir,
        minimum_coverage=args.minimum_coverage,
        coverage_step=args.coverage_step,
        rejection_rates=rejection_rates,
        random_repeats=args.random_repeats,
        bootstrap_repeats=args.bootstrap_repeats,
        random_seed=args.random_seed,
        bin_count=args.uncertainty_bins,
        primary_metric=args.primary_metric,
        primary_subset=args.primary_subset,
    )


if __name__ == "__main__":
    main()
