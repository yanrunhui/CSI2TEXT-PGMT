from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream_cp_selection.data_io import CPCorpus  # noqa: E402
from downstream_cp_selection.pretrained import PHYSICS_FEATURE_ORDER  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose CP-class intervals, error directions, and likely causes."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--feature-cache",
        type=Path,
        help="Frozen feature cache used by Full+Physics; enables physics-error analysis.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--boundary-margin",
        type=float,
        default=0.002,
        help=(
            "An energy-ratio distance from the 1%% labeling threshold considered "
            "boundary-ambiguous; 0.002 means 0.2 percentage points."
        ),
    )
    return parser.parse_args()


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else math.nan


def finite_mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(prediction) & np.isfinite(target)
    if not bool(valid.any()):
        return math.nan
    return float(np.mean(np.abs(prediction[valid] - target[valid])))


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "indices",
            "predicted_class",
            "target_class",
            "predicted_probabilities",
            "all_cp_outside_energy_ratios",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"Prediction file is missing fields: {sorted(missing)}")
        return {key: payload[key] for key in payload.files}


def load_physics_cache(
    path: Path | None,
    count: int,
) -> tuple[np.ndarray | None, list[str]]:
    if path is None:
        return None, []
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    physics = torch.as_tensor(payload.get("physics"), dtype=torch.float32)
    order = [str(value) for value in payload.get("physics_order", PHYSICS_FEATURE_ORDER)]
    if physics.ndim != 2 or physics.shape[0] != count:
        raise ValueError(
            f"Physics cache must be [{count}, features], got {tuple(physics.shape)}"
        )
    if physics.shape[1] != len(order):
        raise ValueError("physics_order does not match cached physics width")
    return physics.numpy(), order


def physics_column(
    physics: np.ndarray,
    order: list[str],
    candidates: tuple[str, ...],
) -> np.ndarray:
    for name in candidates:
        if name in order:
            return physics[:, order.index(name)]
    raise ValueError(f"None of {candidates} appears in physics_order={order}")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.boundary_margin < 0.0:
        raise ValueError("boundary-margin must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpus = CPCorpus.load(args.data)
    prediction_payload = load_predictions(args.predictions)
    indices = np.asarray(prediction_payload["indices"], dtype=np.int64)
    predicted = np.asarray(prediction_payload["predicted_class"], dtype=np.int64)
    target = np.asarray(prediction_payload["target_class"], dtype=np.int64)
    probabilities = np.asarray(
        prediction_payload["predicted_probabilities"], dtype=np.float64
    )
    outside = np.asarray(
        prediction_payload["all_cp_outside_energy_ratios"], dtype=np.float64
    )
    cp_ns = corpus.cp_lengths_ns.numpy().astype(np.float64)
    class_count = len(cp_ns)
    sample_count = len(indices)
    expected_test = corpus.split_indices["test"].numpy()

    if not (
        predicted.shape == target.shape == indices.shape == (sample_count,)
        and probabilities.shape == outside.shape == (sample_count, class_count)
    ):
        raise ValueError("Prediction arrays have inconsistent shapes")
    if not np.array_equal(np.sort(indices), np.sort(expected_test)):
        raise ValueError("Prediction indices do not match the corpus test split")
    if not np.array_equal(target, corpus.labels[indices].numpy()):
        raise ValueError("Prediction target_class does not match the CP manifest")
    if not np.allclose(outside, corpus.outside_energy_ratios[indices].numpy()):
        raise ValueError("Prediction outside-energy table does not match the CP manifest")

    rows = np.arange(sample_count)
    correct = predicted == target
    under = predicted < target
    over = predicted > target
    adjacent = np.abs(predicted - target) == 1
    confidence = probabilities.max(axis=1)
    true_probability = probabilities[rows, target]
    selected_outside = outside[rows, predicted]
    threshold = float(corpus.max_outside_energy_ratio)

    safe_slack = threshold - outside[rows, target]
    previous_excess = np.full(sample_count, np.nan, dtype=np.float64)
    has_previous = target > 0
    previous_excess[has_previous] = (
        outside[rows[has_previous], target[has_previous] - 1] - threshold
    )
    hard_boundary = safe_slack <= args.boundary_margin
    hard_boundary[has_previous] |= (
        previous_excess[has_previous] <= args.boundary_margin
    )

    true_n_paths = corpus.path_count[indices].numpy().astype(np.float64)
    true_energy_delay = corpus.energy_delay_ns[indices].numpy().astype(np.float64)
    true_max_excess = corpus.max_excess_delay_ns[indices].numpy().astype(np.float64)
    true_delay_spread = np.asarray(
        [float(corpus.samples[int(index)].delay_spread_s) * 1e9 for index in indices],
        dtype=np.float64,
    )
    true_reflection_paths = np.asarray(
        [
            float(getattr(corpus.samples[int(index)], "reflection_path_count", math.nan))
            for index in indices
        ],
        dtype=np.float64,
    )

    cached_physics, physics_order = load_physics_cache(
        args.feature_cache, len(corpus.samples)
    )
    predicted_delay_spread = predicted_n_paths = predicted_reflection_paths = None
    if cached_physics is not None:
        test_physics = cached_physics[indices]
        predicted_delay_spread = 200.0 * physics_column(
            test_physics,
            physics_order,
            ("predicted_delay_spread_ns/200",),
        )
        predicted_n_paths = 10.0 * physics_column(
            test_physics,
            physics_order,
            ("predicted_n_paths/10",),
        )
        predicted_reflection_paths = 10.0 * physics_column(
            test_physics,
            physics_order,
            (
                "predicted_reflection_path_count/10",
                "predicted_reflection_count/10",
            ),
        )

    interval_rows: list[dict[str, Any]] = []
    for class_index, value in enumerate(cp_ns):
        mask = target == class_index
        count = int(mask.sum())
        if count == 0:
            continue
        selected = selected_outside[mask]
        row: dict[str, Any] = {
            "class_index": class_index,
            "cp_ns": float(value),
            "sample_count": count,
            "sample_share_pct": float(100.0 * count / sample_count),
            "recall_pct": float(100.0 * correct[mask].mean()),
            "under_rate_pct": float(100.0 * under[mask].mean()),
            "over_rate_pct": float(100.0 * over[mask].mean()),
            "adjacent_error_rate_pct": float(
                100.0 * (adjacent[mask] & ~correct[mask]).mean()
            ),
            "mean_confidence_pct": float(100.0 * confidence[mask].mean()),
            "mean_true_class_probability_pct": float(
                100.0 * true_probability[mask].mean()
            ),
            "boundary_hard_rate_pct": float(100.0 * hard_boundary[mask].mean()),
            "mean_safe_slack_pct_point": float(100.0 * safe_slack[mask].mean()),
            "mean_previous_cp_excess_pct_point": float(
                100.0 * finite_mean(previous_excess[mask])
            ),
            "mean_selected_outside_energy_pct": float(100.0 * selected.mean()),
            "p95_selected_outside_energy_pct": float(
                100.0 * np.percentile(selected, 95)
            ),
            "mean_true_n_paths": finite_mean(true_n_paths[mask]),
            "mean_true_delay_spread_ns": finite_mean(true_delay_spread[mask]),
            "mean_energy_delay_ns": finite_mean(true_energy_delay[mask]),
            "mean_max_excess_delay_ns": finite_mean(true_max_excess[mask]),
        }
        if cached_physics is not None:
            assert predicted_delay_spread is not None
            assert predicted_n_paths is not None
            assert predicted_reflection_paths is not None
            row.update(
                {
                    "delay_spread_mae_ns": finite_mae(
                        predicted_delay_spread, true_delay_spread, mask
                    ),
                    "n_paths_mae": finite_mae(predicted_n_paths, true_n_paths, mask),
                    "reflection_path_count_mae": finite_mae(
                        predicted_reflection_paths, true_reflection_paths, mask
                    ),
                }
            )
        interval_rows.append(row)

    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(confusion, (target, predicted), 1)
    confusion_rows: list[dict[str, Any]] = []
    for true_class in range(class_count):
        denominator = max(int(confusion[true_class].sum()), 1)
        for predicted_class in range(class_count):
            count = int(confusion[true_class, predicted_class])
            confusion_rows.append(
                {
                    "true_cp_ns": float(cp_ns[true_class]),
                    "predicted_cp_ns": float(cp_ns[predicted_class]),
                    "count": count,
                    "row_pct": float(100.0 * count / denominator),
                    "direction": (
                        "correct"
                        if true_class == predicted_class
                        else "under"
                        if predicted_class < true_class
                        else "over"
                    ),
                }
            )

    physics_error_summary: dict[str, Any] = {}
    if cached_physics is not None:
        assert predicted_delay_spread is not None
        assert predicted_n_paths is not None
        assert predicted_reflection_paths is not None
        for name, pred_values, true_values in (
            ("delay_spread_ns", predicted_delay_spread, true_delay_spread),
            ("n_paths", predicted_n_paths, true_n_paths),
            (
                "reflection_path_count",
                predicted_reflection_paths,
                true_reflection_paths,
            ),
        ):
            physics_error_summary[name] = {
                "mae_all": finite_mae(pred_values, true_values, np.ones(sample_count, bool)),
                "mae_correct": finite_mae(pred_values, true_values, correct),
                "mae_incorrect": finite_mae(pred_values, true_values, ~correct),
                "mae_under": finite_mae(pred_values, true_values, under),
                "mae_over": finite_mae(pred_values, true_values, over),
            }

    error_pairs = sorted(
        [row for row in confusion_rows if row["direction"] != "correct"],
        key=lambda row: row["count"],
        reverse=True,
    )
    worst_interval = min(interval_rows, key=lambda row: row["recall_pct"])
    most_unsafe_interval = max(interval_rows, key=lambda row: row["under_rate_pct"])
    hardest_boundary_interval = max(
        interval_rows, key=lambda row: row["boundary_hard_rate_pct"]
    )
    likely_causes = [
        {
            "finding": "lowest_recall_interval",
            "cp_ns": worst_interval["cp_ns"],
            "recall_pct": worst_interval["recall_pct"],
            "sample_count": worst_interval["sample_count"],
            "interpretation": (
                "Check class scarcity and its dominant confusion pairs; low count is an "
                "association with difficulty, not proof of causation."
            ),
        },
        {
            "finding": "highest_underprediction_interval",
            "cp_ns": most_unsafe_interval["cp_ns"],
            "under_rate_pct": most_unsafe_interval["under_rate_pct"],
            "interpretation": "This interval contributes the greatest class-conditional safety risk.",
        },
        {
            "finding": "most_boundary_ambiguous_interval",
            "cp_ns": hardest_boundary_interval["cp_ns"],
            "boundary_hard_rate_pct": hardest_boundary_interval[
                "boundary_hard_rate_pct"
            ],
            "interpretation": (
                "Many labels lie close to the 1% energy threshold, so small CSI/physics "
                "estimation errors can change the exact CP class."
            ),
        },
    ]
    if error_pairs:
        likely_causes.append(
            {
                "finding": "largest_confusion_pair",
                **error_pairs[0],
                "interpretation": "This is the single largest source of exact-classification errors.",
            }
        )

    overall = {
        "sample_count": sample_count,
        "accuracy_pct": float(100.0 * correct.mean()),
        "underprediction_rate_pct": float(100.0 * under.mean()),
        "overprediction_rate_pct": float(100.0 * over.mean()),
        "boundary_hard_rate_pct": float(100.0 * hard_boundary.mean()),
        "mean_confidence_pct": float(100.0 * confidence.mean()),
        "mean_confidence_correct_pct": float(100.0 * finite_mean(confidence[correct])),
        "mean_confidence_incorrect_pct": float(100.0 * finite_mean(confidence[~correct])),
        "mean_selected_outside_energy_pct": float(100.0 * selected_outside.mean()),
        "p95_selected_outside_energy_pct": float(
            100.0 * np.percentile(selected_outside, 95)
        ),
    }

    save_csv(args.output_dir / "interval_diagnostics.csv", interval_rows)
    save_csv(args.output_dir / "confusion_pairs.csv", confusion_rows)
    save_csv(args.output_dir / "top_error_pairs.csv", error_pairs)
    report = {
        "data": str(args.data.resolve()),
        "predictions": str(args.predictions.resolve()),
        "feature_cache": str(args.feature_cache.resolve()) if args.feature_cache else None,
        "cp_lengths_ns": cp_ns.tolist(),
        "threshold": threshold,
        "boundary_margin": args.boundary_margin,
        "overall": overall,
        "intervals": interval_rows,
        "physics_error_summary": physics_error_summary,
        "likely_causes": likely_causes,
        "causality_note": (
            "These diagnostics identify associations. Confirming causes requires controlled "
            "ablations such as balanced sampling, oracle-physics substitution, or boundary removal."
        ),
    }
    with (args.output_dir / "diagnostics.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=True)

    labels = [f"{value:g} ns" for value in cp_ns]
    recall = [row["recall_pct"] for row in interval_rows]
    under_rate = [row["under_rate_pct"] for row in interval_rows]
    over_rate = [row["over_rate_pct"] for row in interval_rows]
    counts = [row["sample_count"] for row in interval_rows]
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    x = np.arange(len(labels))
    width = 0.25
    axes[0].bar(x - width, recall, width, label="Recall")
    axes[0].bar(x, under_rate, width, label="Under")
    axes[0].bar(x + width, over_rate, width, label="Over")
    axes[0].set_xticks(x, labels, rotation=30)
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_title("Per-interval outcomes")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, counts, color="#4C78A8")
    axes[1].set_xticks(x, labels, rotation=30)
    axes[1].set_ylabel("Test samples")
    axes[1].set_title("Class support")
    axes[1].grid(axis="y", alpha=0.25)

    normalized = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    image = axes[2].imshow(normalized * 100.0, cmap="Blues", vmin=0.0, vmax=100.0)
    axes[2].set_xticks(x, labels, rotation=30)
    axes[2].set_yticks(x, labels)
    axes[2].set_xlabel("Predicted CP")
    axes[2].set_ylabel("True CP")
    axes[2].set_title("Row-normalized confusion (%)")
    for row_index in range(class_count):
        for column_index in range(class_count):
            value = 100.0 * normalized[row_index, column_index]
            axes[2].text(
                column_index,
                row_index,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 50.0 else "black",
            )
    figure.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(args.output_dir / "interval_diagnostics.png", dpi=220)
    plt.close(figure)

    print("=== Overall ===")
    print(json.dumps(overall, indent=2, ensure_ascii=False))
    print("\n=== Per CP interval ===")
    for row in interval_rows:
        print(
            f"CP={row['cp_ns']:g}ns n={row['sample_count']} "
            f"recall={row['recall_pct']:.2f}% under={row['under_rate_pct']:.2f}% "
            f"over={row['over_rate_pct']:.2f}% "
            f"boundary_hard={row['boundary_hard_rate_pct']:.2f}%"
        )
    print("\n=== Evidence-backed likely causes ===")
    print(json.dumps(likely_causes, indent=2, ensure_ascii=False))
    print(f"\noutputs={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
