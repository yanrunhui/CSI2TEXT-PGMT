from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


C_M_PER_S = 299_792_458.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match BUPT first-path detections and predictions to TX/RX GNSS, "
            "then estimate a single fixed system delay from manual LOS points "
            "or a provisional lower-envelope subset."
        )
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--rx-gnss", type=Path, required=True)
    parser.add_argument("--tx-gnss", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--los-timestamps-csv",
        type=Path,
        help=(
            "Optional CSV with a timestamp column and, optionally, an is_los column. "
            "When omitted, a lower-envelope subset is used only as a provisional estimate."
        ),
    )
    parser.add_argument("--utc-offset-hours", type=float, default=8.0)
    parser.add_argument("--max-gnss-time-error-s", type=float, default=0.2)
    parser.add_argument("--lower-envelope-quantile", type=float, default=0.2)
    parser.add_argument("--minimum-calibration-points", type=int, default=3)
    return parser.parse_args()


def read_table_flexible(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for reader in (pd.read_excel, pd.read_csv):
        try:
            return reader(path)
        except Exception as exc:  # pragma: no cover - only the final error is surfaced
            errors.append(f"{reader.__name__}: {exc}")
    raise ValueError(f"Could not read {path}: {'; '.join(errors)}")


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def timestamp_to_utc_seconds(timestamp: str, utc_offset_hours: float) -> float:
    text = str(timestamp).strip()
    if len(text) < 14 or not text[:14].isdigit():
        raise ValueError(f"Unsupported timestamp {timestamp!r}; expected YYYYMMDDHHMMSSfff")
    hour = int(text[8:10])
    minute = int(text[10:12])
    second = int(text[12:14])
    fractional_digits = "".join(character for character in text[14:] if character.isdigit())
    fraction = float(f"0.{fractional_digits}") if fractional_digits else 0.0
    return (hour * 3600.0 + minute * 60.0 + second + fraction - utc_offset_hours * 3600.0) % 86400.0


def circular_seconds_difference(values: np.ndarray, target: float) -> np.ndarray:
    return np.abs((values - target + 43200.0) % 86400.0 - 43200.0)


def haversine_distance_m(
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: float,
    lon2_deg: float,
) -> float:
    radius_m = 6_371_008.8
    lat1 = math.radians(lat1_deg)
    lat2 = math.radians(lat2_deg)
    delta_lat = lat2 - lat1
    delta_lon = math.radians(lon2_deg - lon1_deg)
    a = math.sin(delta_lat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2.0) ** 2
    return 2.0 * radius_m * math.asin(min(1.0, math.sqrt(a)))


def prediction_means(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"timestamp": str})
    required = {"timestamp", "pred_first_path_delay_ns"}
    require_columns(frame, required, "Predictions CSV")
    frame["timestamp"] = frame["timestamp"].str.strip()
    return (
        frame.groupby("timestamp", as_index=False)["pred_first_path_delay_ns"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "pred_first_path_delay_mean_ns",
                "std": "pred_first_path_delay_std_ns",
                "count": "prediction_snapshot_count",
            }
        )
    )


def processed_first_paths(directory: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            sample_rate_mhz = float(np.asarray(data["sampleRateMHz"]).reshape(-1)[0])
            rows.append(
                {
                    "timestamp": path.stem,
                    "processed_file": str(path),
                    "first_path_tap": int(np.asarray(data["first_path_tap"]).reshape(-1)[0]),
                    "detected_first_delay_uncalibrated_ns": float(
                        np.asarray(data["first_path_delay_uncalibrated_ns"]).reshape(-1)[0]
                    ),
                    "sample_rate_mhz": sample_rate_mhz,
                    "sample_period_ns": 1000.0 / sample_rate_mhz,
                }
            )
    if not rows:
        raise ValueError(f"No NPZ files found in {directory}")
    return pd.DataFrame(rows)


def read_manual_los(path: Path) -> set[str]:
    frame = pd.read_csv(path, dtype={"timestamp": str})
    require_columns(frame, {"timestamp"}, "Manual LOS CSV")
    if "is_los" in frame:
        accepted = {"1", "true", "yes", "y", "los"}
        frame = frame[frame["is_los"].astype(str).str.strip().str.lower().isin(accepted)]
    return set(frame["timestamp"].astype(str).str.strip())


def finite_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(reference) & np.isfinite(prediction)
    reference = reference[valid]
    prediction = prediction[valid]
    if reference.size == 0:
        return {"count": 0}
    error = prediction - reference
    correlation = (
        float(np.corrcoef(reference, prediction)[0, 1])
        if reference.size >= 2 and np.std(reference) > 0.0 and np.std(prediction) > 0.0
        else float("nan")
    )
    return {
        "count": int(reference.size),
        "reference_mean_ns": float(np.mean(reference)),
        "prediction_mean_ns": float(np.mean(prediction)),
        "mae_ns": float(np.mean(np.abs(error))),
        "bias_ns": float(np.mean(error)),
        "pearson_r": correlation,
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.lower_envelope_quantile <= 0.5:
        raise ValueError("lower-envelope-quantile must be in (0, 0.5]")
    if args.max_gnss_time_error_s < 0.0:
        raise ValueError("max-gnss-time-error-s must be non-negative")
    if args.minimum_calibration_points <= 0:
        raise ValueError("minimum-calibration-points must be positive")

    detected = processed_first_paths(args.processed_dir)
    predicted = prediction_means(args.predictions_csv)
    result = detected.merge(predicted, on="timestamp", how="left", validate="one_to_one")

    rx = pd.read_csv(args.rx_gnss)
    require_columns(rx, {"UTC_Seconds", "Lat_Deg", "Lon_Deg", "Alt_M"}, "RX GNSS")
    rx = rx.sort_values("UTC_Seconds").reset_index(drop=True)
    rx_seconds = rx["UTC_Seconds"].to_numpy(dtype=float)
    tx = read_table_flexible(args.tx_gnss)
    require_columns(tx, {"Lat_Deg", "Lon_Deg", "Alt_M"}, "TX GNSS")
    tx_lat = float(tx["Lat_Deg"].median())
    tx_lon = float(tx["Lon_Deg"].median())
    tx_alt = float(tx["Alt_M"].median())

    gnss_rows: list[dict[str, float]] = []
    for timestamp in result["timestamp"]:
        utc_seconds = timestamp_to_utc_seconds(timestamp, args.utc_offset_hours)
        differences = circular_seconds_difference(rx_seconds, utc_seconds)
        nearest = rx.iloc[int(np.argmin(differences))]
        horizontal_m = haversine_distance_m(
            tx_lat,
            tx_lon,
            float(nearest["Lat_Deg"]),
            float(nearest["Lon_Deg"]),
        )
        altitude_difference_m = float(nearest["Alt_M"]) - tx_alt
        distance_3d_m = math.hypot(horizontal_m, altitude_difference_m)
        record = {
            "measurement_utc_seconds": utc_seconds,
            "gnss_utc_seconds": float(nearest["UTC_Seconds"]),
            "gnss_time_error_s": float(np.min(differences)),
            "rx_lat_deg": float(nearest["Lat_Deg"]),
            "rx_lon_deg": float(nearest["Lon_Deg"]),
            "rx_alt_m": float(nearest["Alt_M"]),
            "horizontal_distance_m": horizontal_m,
            "distance_3d_m": distance_3d_m,
            "geometric_los_delay_ns": distance_3d_m / C_M_PER_S * 1e9,
        }
        for optional in ("RTK_Status", "Sats", "Std_Lat", "Std_Lon", "Std_Alt"):
            if optional in nearest:
                record[f"rx_{optional.lower()}"] = float(nearest[optional])
        gnss_rows.append(record)
    result = pd.concat([result.reset_index(drop=True), pd.DataFrame(gnss_rows)], axis=1)
    result["candidate_system_delay_ns"] = (
        result["detected_first_delay_uncalibrated_ns"] - result["geometric_los_delay_ns"]
    )
    result["prediction_minus_uncalibrated_ns"] = (
        result["pred_first_path_delay_mean_ns"] - result["detected_first_delay_uncalibrated_ns"]
    )

    eligible = (
        np.isfinite(result["candidate_system_delay_ns"])
        & (result["first_path_tap"] >= 0)
        & (result["gnss_time_error_s"] <= args.max_gnss_time_error_s)
    )
    selection_method: str
    if args.los_timestamps_csv is not None:
        manual_los = read_manual_los(args.los_timestamps_csv)
        selected = eligible & result["timestamp"].isin(manual_los)
        selection_method = "manual_confirmed_los"
    else:
        eligible_offsets = result.loc[eligible, "candidate_system_delay_ns"]
        if eligible_offsets.empty:
            raise ValueError("No points passed the GNSS and finite-value checks")
        cutoff = float(eligible_offsets.quantile(args.lower_envelope_quantile))
        selected = eligible & (result["candidate_system_delay_ns"] <= cutoff)
        selection_method = "provisional_lower_envelope_not_confirmed_los"
        result["lower_envelope_cutoff_ns"] = cutoff

    selected_offsets = result.loc[selected, "candidate_system_delay_ns"].to_numpy(float)
    if selected_offsets.size < args.minimum_calibration_points:
        raise ValueError(
            f"Only {selected_offsets.size} calibration points selected; "
            f"need at least {args.minimum_calibration_points}"
        )
    system_delay_ns = float(np.median(selected_offsets))
    absolute_deviation = np.abs(selected_offsets - system_delay_ns)
    result["calibration_selected"] = selected
    result["detected_first_delay_calibrated_ns"] = (
        result["detected_first_delay_uncalibrated_ns"] - system_delay_ns
    )
    result["calibrated_minus_geometric_ns"] = (
        result["detected_first_delay_calibrated_ns"] - result["geometric_los_delay_ns"]
    )

    uncalibrated_metrics = finite_metrics(
        result["detected_first_delay_uncalibrated_ns"].to_numpy(float),
        result["pred_first_path_delay_mean_ns"].to_numpy(float),
    )
    calibrated_metrics = finite_metrics(
        result["detected_first_delay_calibrated_ns"].to_numpy(float),
        result["pred_first_path_delay_mean_ns"].to_numpy(float),
    )
    summary = {
        "selection_method": selection_method,
        "warning": (
            "Lower-envelope points are only LOS candidates. Use a manually reviewed "
            "timestamp CSV or a back-to-back calibration for a defensible physical offset."
            if args.los_timestamps_csv is None
            else "System delay was estimated from user-supplied LOS timestamps."
        ),
        "processed_mat_count": int(len(result)),
        "eligible_point_count": int(np.count_nonzero(eligible)),
        "calibration_point_count": int(selected_offsets.size),
        "system_delay_ns": system_delay_ns,
        "system_delay_samples_at_200mhz": system_delay_ns / 5.0,
        "calibration_std_ns": float(np.std(selected_offsets, ddof=1)) if selected_offsets.size > 1 else 0.0,
        "calibration_mad_ns": float(np.median(absolute_deviation)),
        "calibration_min_ns": float(np.min(selected_offsets)),
        "calibration_max_ns": float(np.max(selected_offsets)),
        "median_gnss_time_error_s": float(result.loc[eligible, "gnss_time_error_s"].median()),
        "tx_position": {"lat_deg": tx_lat, "lon_deg": tx_lon, "alt_m": tx_alt},
        "prediction_vs_uncalibrated_detection": uncalibrated_metrics,
        "prediction_vs_calibrated_detection": calibrated_metrics,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "first_path_gnss_per_mat.csv", index=False)
    result.loc[selected].to_csv(args.output_dir / "selected_calibration_points.csv", index=False)
    with (args.output_dir / "system_delay_calibration.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=True)
    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"Per-MAT diagnostics: {args.output_dir / 'first_path_gnss_per_mat.csv'}")
    print(f"Calibration JSON: {args.output_dir / 'system_delay_calibration.json'}")


if __name__ == "__main__":
    main()
