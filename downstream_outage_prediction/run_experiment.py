from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import (  # noqa: E402
    PHYSICS_TARGET_NAMES,
    PHYSICS_TARGET_OFFSETS,
    PHYSICS_TARGET_SCALES,
    PreprocessedCSIDataset,
    collate_fn,
)
from scripts.benchmark_full_model import build_model  # noqa: E402
from scripts.evaluate import build_tokenizer, move_batch  # noqa: E402


DEFAULT_REPO = Path("/home/yrh/CSI_model/CSI_model")
DEFAULT_DATA = DEFAULT_REPO / "downstream_outage_prediction/artifacts/seed23421/test.pt"
DEFAULT_CHECKPOINT = (
    DEFAULT_REPO
    / "artifacts/pretrain_d2los_50k_upa8x8_bw100m_nf128_total_power_seed0/checkpoint_epoch_100.pt"
)
DEFAULT_OUTPUT_DIR = DEFAULT_REPO / "downstream_outage_prediction/outputs/seed23421"
TOTAL_POWER_NAME = "total_received_power_dbw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict total receive power from CSI, derive SNR from a fixed receiver "
            "noise floor, and apply a deterministic outage rule."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bandwidth-hz", type=float, default=100e6)
    parser.add_argument("--noise-figure-db", type=float, default=7.0)
    parser.add_argument("--thermal-noise-density-dbm-hz", type=float, default=-174.0)
    parser.add_argument("--required-snr-db", type=float, default=5.0)
    parser.add_argument(
        "--safety-margin-db",
        type=float,
        default=0.0,
        help=(
            "Nonnegative conservative margin added only to the predicted outage "
            "decision threshold. Ground-truth outage always uses required-snr-db."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--allow-missing-targets",
        action="store_true",
        help="Evaluate only rows with a finite total-power target instead of failing.",
    )
    parser.add_argument(
        "--allow-bandwidth-mismatch",
        action="store_true",
        help="Do not reject samples whose stored bandwidth differs from bandwidth-hz.",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def finite_sample_bandwidth_hz(sample) -> float | None:
    try:
        bandwidth_hz = float(getattr(sample, "bandwidth_hz", math.nan))
    except (TypeError, ValueError):
        bandwidth_hz = math.nan
    if math.isfinite(bandwidth_hz) and bandwidth_hz > 0.0:
        return bandwidth_hz
    try:
        spacing_hz = float(sample.subcarrier_spacing_hz)
        source_n_freq = int(getattr(sample, "source_n_freq", 0))
        if source_n_freq <= 0:
            source_n_freq = int(sample.tokens.shape[-1])
        bandwidth_hz = spacing_hz * source_n_freq
    except (AttributeError, TypeError, ValueError):
        return None
    return bandwidth_hz if math.isfinite(bandwidth_hz) and bandwidth_hz > 0.0 else None


def validate_sample_bandwidths(
    samples: list,
    expected_hz: float,
    allow_mismatch: bool,
) -> dict[str, Any]:
    available = [
        value
        for sample in samples
        if (value := finite_sample_bandwidth_hz(sample)) is not None
    ]
    mismatched = [
        value
        for value in available
        if not math.isclose(value, expected_hz, rel_tol=1e-6, abs_tol=1.0)
    ]
    if mismatched and not allow_mismatch:
        unique_mhz = sorted({round(value / 1e6, 6) for value in mismatched})
        raise ValueError(
            f"Found {len(mismatched)} samples whose bandwidth differs from "
            f"{expected_hz / 1e6:g} MHz: {unique_mhz[:10]}. Use "
            "--allow-bandwidth-mismatch only when intentional."
        )
    return {
        "available_count": len(available),
        "missing_count": len(samples) - len(available),
        "mismatch_count": len(mismatched),
        "unique_bandwidth_mhz": sorted({round(value / 1e6, 6) for value in available}),
    }


def require_total_power_checkpoint(checkpoint: dict[str, Any]) -> None:
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint is missing model_state.")
    if not any(name.startswith("total_received_power_head.") for name in state):
        raise ValueError(
            "Checkpoint does not contain a trained total_received_power_head. "
            "Use the checkpoint produced after adding and training the total-power label."
        )
    if TOTAL_POWER_NAME not in PHYSICS_TARGET_NAMES:
        raise RuntimeError(f"Current code does not define {TOTAL_POWER_NAME}.")


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def power_metrics(
    prediction_dbw: np.ndarray,
    target_dbw: np.ndarray,
) -> dict[str, float | int]:
    error = prediction_dbw - target_dbw
    absolute_error = np.abs(error)
    return {
        "count": int(error.size),
        "mae_db": float(np.mean(absolute_error)),
        "rmse_db": float(np.sqrt(np.mean(np.square(error)))),
        "bias_db": float(np.mean(error)),
        "median_absolute_error_db": float(np.median(absolute_error)),
        "p90_absolute_error_db": float(np.percentile(absolute_error, 90)),
        "p95_absolute_error_db": float(np.percentile(absolute_error, 95)),
        "pearson": safe_pearson(prediction_dbw, target_dbw),
        "target_mean_dbw": float(np.mean(target_dbw)),
        "target_std_db": float(np.std(target_dbw)),
        "prediction_mean_dbw": float(np.mean(prediction_dbw)),
        "prediction_std_db": float(np.std(prediction_dbw)),
    }


def outage_metrics(
    predicted_outage: np.ndarray,
    true_outage: np.ndarray,
) -> dict[str, float | int]:
    predicted_outage = predicted_outage.astype(bool)
    true_outage = true_outage.astype(bool)
    tp = int(np.sum(predicted_outage & true_outage))
    tn = int(np.sum(~predicted_outage & ~true_outage))
    fp = int(np.sum(predicted_outage & ~true_outage))
    fn = int(np.sum(~predicted_outage & true_outage))
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if math.isfinite(precision)
        and math.isfinite(recall)
        and precision + recall > 0.0
        else math.nan
    )
    return {
        "count": int(true_outage.size),
        "true_outage_count": int(np.sum(true_outage)),
        "true_normal_count": int(np.sum(~true_outage)),
        "predicted_outage_count": int(np.sum(predicted_outage)),
        "predicted_normal_count": int(np.sum(~predicted_outage)),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy_pct": 100.0 * safe_ratio(tp + tn, true_outage.size),
        "outage_precision_pct": 100.0 * precision,
        "outage_recall_pct": 100.0 * recall,
        "outage_f1_pct": 100.0 * f1,
        "missed_outage_rate_pct": 100.0 * safe_ratio(fn, tp + fn),
        "false_alarm_rate_pct": 100.0 * safe_ratio(fp, tn + fp),
        "specificity_pct": 100.0 * specificity,
        "balanced_accuracy_pct": 50.0 * (recall + specificity),
        "true_outage_fraction_pct": 100.0 * safe_ratio(np.sum(true_outage), true_outage.size),
        "predicted_outage_fraction_pct": 100.0
        * safe_ratio(np.sum(predicted_outage), predicted_outage.size),
    }


def breakdown_metrics(
    prediction_dbw: np.ndarray,
    target_dbw: np.ndarray,
    predicted_outage: np.ndarray,
    true_outage: np.ndarray,
    los_status: np.ndarray,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for status in ("los", "nlos"):
        mask = los_status == status
        if not bool(np.any(mask)):
            continue
        result[status] = {
            "power": power_metrics(prediction_dbw[mask], target_dbw[mask]),
            "outage": outage_metrics(predicted_outage[mask], true_outage[mask]),
        }
    return result


def save_predictions_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(
    output_dir: Path,
    prediction_dbw: np.ndarray,
    target_dbw: np.ndarray,
    predicted_snr_db: np.ndarray,
    true_snr_db: np.ndarray,
    predicted_outage: np.ndarray,
    true_outage: np.ndarray,
    required_snr_db: float,
    metrics: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = np.where(true_outage, "#D55E00", "#0072B2")
    lower = float(min(target_dbw.min(), prediction_dbw.min()))
    upper = float(max(target_dbw.max(), prediction_dbw.max()))
    padding = max((upper - lower) * 0.04, 1.0)
    fig, axis = plt.subplots(figsize=(6.5, 6.0))
    axis.scatter(target_dbw, prediction_dbw, s=12, c=colors, alpha=0.45, edgecolors="none")
    axis.plot([lower - padding, upper + padding], [lower - padding, upper + padding], "k--", lw=1.3)
    axis.set_xlim(lower - padding, upper + padding)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_xlabel("True total received power (dBW)")
    axis.set_ylabel("Predicted total received power (dBW)")
    axis.set_title("Total received power prediction")
    axis.grid(alpha=0.2)
    power = metrics["power"]
    axis.text(
        0.03,
        0.97,
        f"MAE = {power['mae_db']:.3f} dB\nRMSE = {power['rmse_db']:.3f} dB\nBias = {power['bias_db']:.3f} dB",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#BBBBBB"},
    )
    fig.tight_layout()
    fig.savefig(output_dir / "power_scatter.png", dpi=220)
    plt.close(fig)

    outage = metrics["outage"]
    matrix = np.asarray(
        [
            [outage["true_negative"], outage["false_positive"]],
            [outage["false_negative"], outage["true_positive"]],
        ],
        dtype=np.int64,
    )
    fig, axis = plt.subplots(figsize=(5.6, 5.0))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks([0, 1], ["Normal", "Outage"])
    axis.set_yticks([0, 1], ["Normal", "Outage"])
    axis.set_xlabel("Predicted link status")
    axis.set_ylabel("True link status")
    axis.set_title("Outage confusion matrix")
    cutoff = float(matrix.max()) / 2.0 if matrix.size else 0.0
    for row in range(2):
        for col in range(2):
            axis.text(
                col,
                row,
                str(int(matrix[row, col])),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > cutoff else "black",
                fontsize=13,
            )
    fig.tight_layout()
    fig.savefig(output_dir / "outage_confusion_matrix.png", dpi=220)
    plt.close(fig)

    bins = np.linspace(
        float(min(true_snr_db.min(), predicted_snr_db.min())),
        float(max(true_snr_db.max(), predicted_snr_db.max())),
        50,
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.hist(true_snr_db, bins=bins, alpha=0.55, label="True SNR", color="#0072B2")
    axis.hist(predicted_snr_db, bins=bins, alpha=0.45, label="Predicted SNR", color="#E69F00")
    axis.axvline(required_snr_db, color="#D55E00", linestyle="--", linewidth=1.8, label="Required SNR")
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Samples")
    axis.set_title("True and predicted SNR distributions")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "snr_histogram.png", dpi=220)
    plt.close(fig)


@torch.inference_mode()
def infer_total_power(
    model,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> dict[str, Any]:
    target_index = PHYSICS_TARGET_NAMES.index(TOTAL_POWER_NAME)
    scale = float(PHYSICS_TARGET_SCALES[target_index])
    offset = float(PHYSICS_TARGET_OFFSETS[target_index])
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    group_ids: list[str] = []
    config_keys: list[str] = []
    los_statuses: list[str] = []

    for batch_number, cpu_batch in enumerate(loader, start=1):
        targets.append(cpu_batch["physics_raw_targets"][:, target_index].float())
        masks.append(cpu_batch["physics_target_mask"][:, target_index].bool())
        group_ids.extend(str(value) for value in cpu_batch["group_ids"])
        config_keys.extend(str(value) for value in cpu_batch["config_keys"])
        los_statuses.extend(str(key.los_status).lower() for key in cpu_batch["semantic_keys"])
        batch = move_batch(cpu_batch, device)
        with torch.autocast(device_type=device.type, enabled=amp):
            csi_features = model.encode_csi(
                batch["tokens"],
                batch["beam_positions"],
                batch["token_mask"],
                batch["freq_bin"],
                batch["bw_bin"],
                batch["subcarrier_spacing"],
                normalize=False,
                config_features=batch.get("config_features"),
                antenna_coordinates=batch.get("antenna_coordinates"),
                antenna_mask=batch.get("antenna_mask"),
            )
            power_context = model.encode_power_context(
                batch["tokens"],
                batch["token_mask"],
                delay_power_map=batch.get("delay_power_map"),
                delay_power_profile=batch.get("delay_power_profile"),
            )
            components = model.predict_physics_components(
                csi_features,
                power_context=power_context,
            )
            normalized_prediction = components["total_received_power_prediction"]
            raw_prediction = normalized_prediction * scale + offset
        predictions.append(raw_prediction.float().cpu())
        if batch_number % 25 == 0 or batch_number == len(loader):
            print(f"inference_batches={batch_number}/{len(loader)}", flush=True)

    return {
        "prediction_dbw": torch.cat(predictions).numpy(),
        "target_dbw": torch.cat(targets).numpy(),
        "valid_mask": torch.cat(masks).numpy().astype(bool),
        "group_ids": np.asarray(group_ids, dtype=str),
        "config_keys": np.asarray(config_keys, dtype=str),
        "los_status": np.asarray(los_statuses, dtype=str),
    }


def main() -> None:
    args = parse_args()
    if args.bandwidth_hz <= 0.0:
        raise ValueError("--bandwidth-hz must be positive.")
    if args.safety_margin_db < 0.0:
        raise ValueError("--safety-margin-db must be nonnegative.")
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers nonnegative.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when set.")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    device = resolve_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    require_total_power_checkpoint(checkpoint)
    dataset = PreprocessedCSIDataset.from_pt(str(args.data))
    if args.limit is not None:
        dataset = PreprocessedCSIDataset(dataset.samples[: args.limit])
    bandwidth_summary = validate_sample_bandwidths(
        dataset.samples,
        expected_hz=args.bandwidth_hz,
        allow_mismatch=args.allow_bandwidth_mismatch,
    )
    tokenizer = build_tokenizer(dataset.samples, checkpoint)
    model = build_model(checkpoint, device)
    if not bool(getattr(model, "use_power_branch", False)):
        raise ValueError(
            "Checkpoint has use_power_branch=False; absolute receive power would not "
            "have access to the raw CSI amplitude features."
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_caption_len=48),
    )
    inference = infer_total_power(model, loader, device, amp)
    valid_mask = (
        inference["valid_mask"]
        & np.isfinite(inference["target_dbw"])
        & np.isfinite(inference["prediction_dbw"])
    )
    invalid_count = int(valid_mask.size - np.sum(valid_mask))
    if invalid_count and not args.allow_missing_targets:
        raise ValueError(
            f"Found {invalid_count} samples without finite total-power targets. "
            "Regenerate preprocessing data or pass --allow-missing-targets explicitly."
        )
    if not bool(np.any(valid_mask)):
        raise ValueError("No valid total-power prediction/target pairs remain.")
    for key in inference:
        if isinstance(inference[key], np.ndarray):
            inference[key] = inference[key][valid_mask]

    prediction_dbw = inference["prediction_dbw"].astype(np.float64)
    target_dbw = inference["target_dbw"].astype(np.float64)
    prediction_dbm = prediction_dbw + 30.0
    target_dbm = target_dbw + 30.0
    noise_power_dbm = (
        args.thermal_noise_density_dbm_hz
        + 10.0 * math.log10(args.bandwidth_hz)
        + args.noise_figure_db
    )
    noise_power_dbw = noise_power_dbm - 30.0
    true_snr_db = target_dbw - noise_power_dbw
    predicted_snr_db = prediction_dbw - noise_power_dbw
    true_outage = true_snr_db < args.required_snr_db
    predicted_decision_threshold_db = args.required_snr_db + args.safety_margin_db
    predicted_outage = predicted_snr_db < predicted_decision_threshold_db

    overall_power = power_metrics(prediction_dbw, target_dbw)
    overall_outage = outage_metrics(predicted_outage, true_outage)
    boundary_distance_db = np.abs(true_snr_db - args.required_snr_db)
    metrics: dict[str, Any] = {
        "format": "csi_power_rule_outage_v1",
        "data": str(args.data.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "amp": amp,
        "input_sample_count": int(valid_mask.size),
        "valid_sample_count": int(np.sum(valid_mask)),
        "invalid_sample_count": invalid_count,
        "bandwidth_hz": args.bandwidth_hz,
        "noise_figure_db": args.noise_figure_db,
        "thermal_noise_density_dbm_hz": args.thermal_noise_density_dbm_hz,
        "noise_power_dbm": noise_power_dbm,
        "noise_power_dbw": noise_power_dbw,
        "required_snr_db": args.required_snr_db,
        "safety_margin_db": args.safety_margin_db,
        "predicted_decision_threshold_db": predicted_decision_threshold_db,
        "bandwidth_validation": bandwidth_summary,
        "power": overall_power,
        "outage": overall_outage,
        "los_nlos_breakdown": breakdown_metrics(
            prediction_dbw,
            target_dbw,
            predicted_outage,
            true_outage,
            inference["los_status"],
        ),
        "boundary": {
            "within_1db_count": int(np.sum(boundary_distance_db <= 1.0)),
            "within_2db_count": int(np.sum(boundary_distance_db <= 2.0)),
            "within_3db_count": int(np.sum(boundary_distance_db <= 3.0)),
            "within_1db_accuracy_pct": 100.0
            * safe_ratio(
                np.sum(predicted_outage[boundary_distance_db <= 1.0] == true_outage[boundary_distance_db <= 1.0]),
                np.sum(boundary_distance_db <= 1.0),
            ),
            "within_2db_accuracy_pct": 100.0
            * safe_ratio(
                np.sum(predicted_outage[boundary_distance_db <= 2.0] == true_outage[boundary_distance_db <= 2.0]),
                np.sum(boundary_distance_db <= 2.0),
            ),
            "within_3db_accuracy_pct": 100.0
            * safe_ratio(
                np.sum(predicted_outage[boundary_distance_db <= 3.0] == true_outage[boundary_distance_db <= 3.0]),
                np.sum(boundary_distance_db <= 3.0),
            ),
        },
        "config_histogram": dict(Counter(inference["config_keys"].tolist())),
        "los_status_histogram": dict(Counter(inference["los_status"].tolist())),
    }

    rows: list[dict[str, Any]] = []
    error_db = prediction_dbw - target_dbw
    for index in range(len(prediction_dbw)):
        rows.append(
            {
                "sample_index": index,
                "group_id": inference["group_ids"][index],
                "config_key": inference["config_keys"][index],
                "los_status": inference["los_status"][index],
                "true_total_rx_power_dbw": target_dbw[index],
                "predicted_total_rx_power_dbw": prediction_dbw[index],
                "true_total_rx_power_dbm": target_dbm[index],
                "predicted_total_rx_power_dbm": prediction_dbm[index],
                "power_error_db": error_db[index],
                "absolute_power_error_db": abs(error_db[index]),
                "noise_power_dbm": noise_power_dbm,
                "true_snr_db": true_snr_db[index],
                "predicted_snr_db": predicted_snr_db[index],
                "required_snr_db": args.required_snr_db,
                "predicted_decision_threshold_db": predicted_decision_threshold_db,
                "true_link_status": "outage" if true_outage[index] else "normal",
                "predicted_link_status": "outage" if predicted_outage[index] else "normal",
                "correct": bool(predicted_outage[index] == true_outage[index]),
                "missed_outage": bool(true_outage[index] and not predicted_outage[index]),
                "false_alarm": bool(not true_outage[index] and predicted_outage[index]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    predictions_csv_path = args.output_dir / "predictions.csv"
    predictions_npz_path = args.output_dir / "predictions.npz"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    save_predictions_csv(predictions_csv_path, rows)
    np.savez_compressed(
        predictions_npz_path,
        group_id=inference["group_ids"],
        config_key=inference["config_keys"],
        los_status=inference["los_status"],
        true_total_rx_power_dbw=target_dbw,
        predicted_total_rx_power_dbw=prediction_dbw,
        power_error_db=error_db,
        noise_power_dbm=np.asarray(noise_power_dbm),
        true_snr_db=true_snr_db,
        predicted_snr_db=predicted_snr_db,
        true_outage=true_outage,
        predicted_outage=predicted_outage,
    )
    if not args.no_plots:
        save_plots(
            args.output_dir,
            prediction_dbw,
            target_dbw,
            predicted_snr_db,
            true_snr_db,
            predicted_outage,
            true_outage,
            args.required_snr_db,
            metrics,
        )

    print(f"noise_power_dbm={noise_power_dbm:.4f}")
    print(f"noise_power_dbw={noise_power_dbw:.4f}")
    print(f"power_MAE_db={overall_power['mae_db']:.4f}")
    print(f"power_RMSE_db={overall_power['rmse_db']:.4f}")
    print(f"power_bias_db={overall_power['bias_db']:.4f}")
    print(f"power_pearson={overall_power['pearson']:.6f}")
    for name in (
        "accuracy_pct",
        "outage_recall_pct",
        "outage_f1_pct",
        "missed_outage_rate_pct",
        "false_alarm_rate_pct",
        "true_outage_fraction_pct",
        "predicted_outage_fraction_pct",
    ):
        print(f"{name}={overall_outage[name]:.4f}")
    print(f"metrics={metrics_path}")
    print(f"predictions_csv={predictions_csv_path}")
    print(f"predictions_npz={predictions_npz_path}")


if __name__ == "__main__":
    main()

