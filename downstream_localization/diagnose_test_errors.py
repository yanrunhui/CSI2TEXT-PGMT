from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LINK_RE = re.compile(r"map_(\d+)-source_(\d+)-rx_(\d+)$")
PREDICTION_SUFFIX = "_test_predictions.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose LoS/NLoS localization errors using saved test predictions."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory containing *_test_predictions.npz files.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional model filename prefixes, e.g. scratch scratch_physics.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def load_corpus(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or payload.get("format") != "csi2text_localization_v1":
        raise ValueError(f"Unsupported localization dataset: {path}")
    samples = payload["samples"]
    tx = torch.as_tensor(payload["tx_xyz_m"]).cpu().numpy()
    rx = torch.as_tensor(payload["rx_xyz_m"]).cpu().numpy()
    test_indices = torch.as_tensor(payload["split_indices"]["test"]).cpu().numpy()
    if tx.shape != (len(samples), 3) or rx.shape != (len(samples), 3):
        raise ValueError("TX/RX coordinate arrays do not match the sample count")
    if test_indices.ndim != 1 or len(np.unique(test_indices)) != len(test_indices):
        raise ValueError("Test indices must be a unique one-dimensional array")
    if np.any(test_indices < 0) or np.any(test_indices >= len(samples)):
        raise ValueError("Test indices are outside the dataset")
    return {"samples": samples, "tx": tx, "rx": rx, "test_indices": test_indices}


def discover_predictions(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.models:
        paths = [
            (name, args.predictions_dir / f"{name}{PREDICTION_SUFFIX}")
            for name in dict.fromkeys(args.models)
        ]
    else:
        paths = [
            (path.name[: -len(PREDICTION_SUFFIX)], path)
            for path in sorted(args.predictions_dir.glob(f"*{PREDICTION_SUFFIX}"))
        ]
    if not paths:
        raise FileNotFoundError(
            f"No *{PREDICTION_SUFFIX} files in {args.predictions_dir}"
        )
    for _, path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def load_predictions(path: Path, corpus: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"indices", "predicted_rx_xy_m", "true_rx_xy_m", "error_m"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        indices = np.asarray(archive["indices"])
        predicted_rx = np.asarray(archive["predicted_rx_xy_m"], dtype=np.float64)
        saved_true_rx = np.asarray(archive["true_rx_xy_m"], dtype=np.float64)
        saved_error = np.asarray(archive["error_m"], dtype=np.float64)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{path}: indices must be one-dimensional integers")
    indices = indices.astype(np.int64, copy=False)
    expected = corpus["test_indices"]
    if len(indices) != len(expected) or not np.array_equal(np.sort(indices), np.sort(expected)):
        raise ValueError(
            f"{path}: prediction indices do not exactly match this dataset's test split"
        )
    if predicted_rx.shape != (len(indices), 2) or saved_true_rx.shape != (len(indices), 2):
        raise ValueError(f"{path}: predicted/true RX arrays must be [test_count, 2]")
    if saved_error.shape != (len(indices),):
        raise ValueError(f"{path}: error_m must have one value per test prediction")
    true_rx = corpus["rx"][indices, :2].astype(np.float64)
    if not np.allclose(saved_true_rx, true_rx, rtol=0, atol=1e-3):
        raise ValueError(f"{path}: saved true RX coordinates differ from the dataset")
    errors = np.linalg.norm(predicted_rx - true_rx, axis=1)
    if not np.isfinite(errors).all():
        raise ValueError(f"{path}: non-finite localization errors")
    if not np.allclose(saved_error, errors, rtol=1e-4, atol=1e-3):
        raise ValueError(f"{path}: saved error_m differs from recomputed RX error")
    return indices, predicted_rx


def sample_los_status(sample: Any) -> str:
    semantic_key = getattr(sample, "semantic_key", None)
    status = str(getattr(semantic_key, "los_status", "")).strip().lower()
    if status not in {"los", "nlos"}:
        raise ValueError(
            f"Missing LoS/NLoS label for group_id={getattr(sample, 'group_id', '?')}"
        )
    return status


def metric_row(model: str, group: str, errors: np.ndarray) -> dict[str, Any]:
    if len(errors) == 0:
        raise ValueError(f"No test samples for model={model}, group={group}")
    return {
        "model": model,
        "group": group,
        "count": len(errors),
        "mean_m": float(np.mean(errors)),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.percentile(errors, 90)),
        "within_5m_pct": float(100 * np.mean(errors <= 5)),
        "within_10m_pct": float(100 * np.mean(errors <= 10)),
        "over_50m_pct": float(100 * np.mean(errors > 50)),
        "over_100m_pct": float(100 * np.mean(errors > 100)),
    }


def sample_row(
    model: str,
    index: int,
    prediction: np.ndarray,
    corpus: dict[str, Any],
) -> dict[str, Any]:
    sample = corpus["samples"][index]
    tx = corpus["tx"][index]
    rx = corpus["rx"][index]
    match = LINK_RE.search(str(sample.group_id))
    return {
        "model": model,
        "sample_index": int(index),
        "group_id": str(sample.group_id),
        "map_id": int(match.group(1)) if match else "",
        "source_id": int(match.group(2)) if match else "",
        "rx_id": int(match.group(3)) if match else "",
        "los_status": sample_los_status(sample),
        "error_m": float(np.linalg.norm(prediction - rx[:2])),
        "tx_x_m": float(tx[0]),
        "tx_y_m": float(tx[1]),
        "true_rx_x_m": float(rx[0]),
        "true_rx_y_m": float(rx[1]),
        "pred_rx_x_m": float(prediction[0]),
        "pred_rx_y_m": float(prediction[1]),
        "n_paths": getattr(sample, "n_paths", ""),
        "first_path_delay_ns": (
            float(sample.first_path_delay_s) * 1e9
            if hasattr(sample, "first_path_delay_s")
            else ""
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnose_model(
    model: str,
    indices: np.ndarray,
    predictions: np.ndarray,
    corpus: dict[str, Any],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    samples = corpus["samples"]
    true_rx = corpus["rx"][indices, :2].astype(np.float64)
    errors = np.linalg.norm(predictions - true_rx, axis=1)
    statuses = np.asarray([sample_los_status(samples[int(index)]) for index in indices])
    if not np.any(statuses == "los") or not np.any(statuses == "nlos"):
        raise ValueError("Test split must contain both LoS and NLoS samples")
    metrics = [
        metric_row(model, group, errors[statuses == group] if group != "all" else errors)
        for group in ("all", "los", "nlos")
    ]
    worst_positions = np.argsort(-errors, kind="stable")[:top_k]
    top_rows = [
        {"rank": rank, **sample_row(model, int(indices[position]), predictions[position], corpus)}
        for rank, position in enumerate(worst_positions, start=1)
    ]
    tail_cutoff = float(np.percentile(errors, 90))
    tail_mask = errors >= tail_cutoff
    tail_count = int(np.sum(tail_mask))
    nlos_tail_count = int(np.sum(tail_mask & (statuses == "nlos")))
    tail = {
        "model": model,
        "test_count": len(errors),
        "test_nlos_pct": float(100 * np.mean(statuses == "nlos")),
        "p90_cutoff_m": tail_cutoff,
        "tail_count": tail_count,
        "tail_nlos_count": nlos_tail_count,
        "tail_nlos_pct": float(100 * nlos_tail_count / tail_count),
        "tail_los_count": tail_count - nlos_tail_count,
        "top_k": len(top_rows),
        "top_k_nlos_count": sum(row["los_status"] == "nlos" for row in top_rows),
    }
    return metrics, tail, top_rows


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    corpus = load_corpus(args.data)
    paths = discover_predictions(args)
    output_dir = args.output_dir or args.predictions_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict[str, Any]] = []
    all_tails: list[dict[str, Any]] = []
    all_top: list[dict[str, Any]] = []
    for model, path in paths:
        indices, predictions = load_predictions(path, corpus)
        metrics, tail, top_rows = diagnose_model(
            model, indices, predictions, corpus, args.top_k
        )
        all_metrics.extend(metrics)
        all_tails.append(tail)
        all_top.extend(top_rows)
        print(
            f"{model}: P90={tail['p90_cutoff_m']:.2f} m; "
            f"worst 10% NLoS={tail['tail_nlos_pct']:.1f}% "
            f"(test baseline NLoS={tail['test_nlos_pct']:.1f}%)"
        )
        for row in metrics:
            print(
                f"  {row['group']:<4} n={row['count']:>5} "
                f"mean={row['mean_m']:>7.2f} m "
                f"median={row['median_m']:>7.2f} m "
                f"P90={row['p90_m']:>7.2f} m "
                f"<=10m={row['within_10m_pct']:>6.2f}%"
            )

    write_csv(output_dir / "los_nlos_metrics.csv", all_metrics)
    write_csv(output_dir / "tail_composition.csv", all_tails)
    write_csv(output_dir / "worst_samples.csv", all_top)
    (output_dir / "diagnostics.json").write_text(
        json.dumps(
            {
                "data": str(args.data.resolve()),
                "predictions_dir": str(args.predictions_dir.resolve()),
                "metrics": all_metrics,
                "tails": all_tails,
                "worst_samples": all_top,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved={output_dir.resolve()}")


if __name__ == "__main__":
    main()
