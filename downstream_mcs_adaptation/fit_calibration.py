from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream_mcs_adaptation.mcs import (  # noqa: E402
    file_sha256,
    load_json,
    subcarrier_power_weights,
    validate_uncalibrated_config,
)


DEFAULT_REPO = Path("/home/yrh/CSI_model/CSI_model")
DEFAULT_SELECTION = DEFAULT_REPO / "downstream_mcs_adaptation/artifacts/calibration_seed23421/selection.pt"
DEFAULT_RESULTS = DEFAULT_REPO / "downstream_mcs_adaptation/artifacts/calibration_seed23421/simulation_results.csv"
DEFAULT_OUTPUT = DEFAULT_REPO / "downstream_mcs_adaptation/artifacts/calibration_seed23421/calibrated_mcs.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit per-MCS EESM beta and BLER curves from training-link full-simulation results."
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--mcs-table", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--beta-min-db", type=float, default=-15.0)
    parser.add_argument("--beta-max-db", type=float, default=30.0)
    parser.add_argument("--min-observations-per-mcs", type=int, default=30)
    return parser.parse_args()


def samples_from_pt(path: Path) -> list:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    samples = payload if isinstance(payload, list) else payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Expected samples in {path}")
    return samples


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line, raw in enumerate(csv.DictReader(stream), start=2):
            try:
                row = {
                    "group_id": str(raw["group_id"]),
                    "mcs_index": int(raw["mcs_index"]),
                    "average_snr_db": float(raw["average_snr_db"]),
                    "block_errors": int(raw["block_errors"]),
                    "total_blocks": int(raw["total_blocks"]),
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid calibration CSV row {line}: {raw}") from exc
            if (
                not math.isfinite(row["average_snr_db"])
                or row["total_blocks"] <= 0
                or not 0 <= row["block_errors"] <= row["total_blocks"]
            ):
                raise ValueError(f"Invalid calibration values at row {line}: {row}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No simulation results in {path}")
    return rows


def fit_logistic(effective_snr: np.ndarray, errors: np.ndarray, blocks: np.ndarray) -> tuple[float, float, float]:
    observed = errors / blocks
    transition = effective_snr[int(np.argmin(np.abs(observed - 0.5)))]

    def objective(parameters: np.ndarray) -> float:
        intercept, slope = float(parameters[0]), float(parameters[1])
        logits = intercept + slope * effective_snr
        probability = np.clip(expit(logits), 1e-9, 1.0 - 1e-9)
        return float(-np.sum(errors * np.log(probability) + (blocks - errors) * np.log1p(-probability)))

    initial_slope = -1.0
    result = minimize(
        objective,
        x0=np.asarray([-initial_slope * transition, initial_slope]),
        method="L-BFGS-B",
        bounds=((-1000.0, 1000.0), (-100.0, -1e-5)),
    )
    if not result.success:
        raise RuntimeError(f"Logistic fit failed: {result.message}")
    return float(result.fun), float(result.x[0]), float(result.x[1])


def batch_effective_snr_db(
    average_snr_db: np.ndarray,
    power_weights: np.ndarray,
    beta_db: float,
) -> np.ndarray:
    """Vectorized EESM for one MCS over all calibration observations."""

    normalized = power_weights / power_weights.mean(axis=1, keepdims=True)
    gamma = np.power(10.0, average_snr_db[:, None] / 10.0) * normalized
    beta = float(np.power(10.0, beta_db / 10.0))
    log_mean_exp = logsumexp(-gamma / beta, axis=1) - math.log(gamma.shape[1])
    effective_linear = np.maximum(-beta * log_mean_exp, np.finfo(np.float64).tiny)
    return 10.0 * np.log10(effective_linear)


def main() -> None:
    args = parse_args()
    if args.beta_max_db <= args.beta_min_db or args.min_observations_per_mcs <= 0:
        raise ValueError("Invalid beta interval or minimum observation count")
    selection = torch.load(args.selection, map_location="cpu", weights_only=False)
    if not isinstance(selection, dict) or selection.get("format") != "csi_mcs_calibration_selection_v1":
        raise ValueError("Unsupported calibration selection")
    table_path = args.mcs_table or Path(selection["mcs_table"])
    config = load_json(table_path)
    validate_uncalibrated_config(config)
    expected_table_hash = selection.get("mcs_table_sha256")
    if expected_table_hash is not None and file_sha256(table_path) != expected_table_hash:
        raise ValueError("MCS table changed after calibration-link selection")
    samples = samples_from_pt(Path(selection["source_train_pt"]))
    selected_indices = torch.as_tensor(selection["source_indices"], dtype=torch.long).tolist()
    selected_by_group = {str(samples[index].group_id): samples[index] for index in selected_indices}
    if len(selected_by_group) != len(selected_indices):
        raise ValueError("Calibration selection has duplicate group IDs")
    mode = str(selection["effective_channel_mode"])
    weights = {
        group_id: subcarrier_power_weights(sample, mode)
        for group_id, sample in selected_by_group.items()
    }

    rows_by_mcs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    valid_mcs_indices = {int(row["index"]) for row in config["mcs"]}
    for row in read_results(args.results):
        if row["group_id"] not in selected_by_group:
            raise ValueError(f"Simulation result uses unselected/test link {row['group_id']}")
        if row["mcs_index"] not in valid_mcs_indices:
            raise ValueError(f"Unknown MCS index in simulation results: {row['mcs_index']}")
        rows_by_mcs[row["mcs_index"]].append(row)

    target_bler = float(config["system"]["target_bler"])
    target_logit = math.log(target_bler / (1.0 - target_bler))
    fitted_rows = []
    diagnostics = []
    for mcs in config["mcs"]:
        index = int(mcs["index"])
        rows = rows_by_mcs[index]
        if len(rows) < args.min_observations_per_mcs:
            raise ValueError(
                f"MCS {index} has only {len(rows)} observations; need {args.min_observations_per_mcs}"
            )
        errors = np.asarray([row["block_errors"] for row in rows], dtype=np.float64)
        blocks = np.asarray([row["total_blocks"] for row in rows], dtype=np.float64)
        average_snr = np.asarray([row["average_snr_db"] for row in rows], dtype=np.float64)
        weight_matrix = np.stack([weights[row["group_id"]] for row in rows]).astype(
            np.float64, copy=False
        )
        if errors.sum() <= 0 or errors.sum() >= blocks.sum():
            raise ValueError(f"MCS {index} results do not contain both successes and failures")

        def beta_objective(beta_db: float) -> float:
            effective = batch_effective_snr_db(average_snr, weight_matrix, beta_db)
            loss, _, _ = fit_logistic(effective, errors, blocks)
            return loss

        beta_fit = minimize_scalar(
            beta_objective,
            bounds=(args.beta_min_db, args.beta_max_db),
            method="bounded",
            options={"xatol": 0.02},
        )
        if not beta_fit.success:
            raise RuntimeError(f"EESM beta fit failed for MCS {index}: {beta_fit.message}")
        beta_db = float(beta_fit.x)
        effective = batch_effective_snr_db(average_snr, weight_matrix, beta_db)
        nll, intercept, logistic_coefficient = fit_logistic(effective, errors, blocks)
        threshold_db = (target_logit - intercept) / logistic_coefficient
        slope_db = -1.0 / logistic_coefficient
        fitted_rows.append(
            {
                **mcs,
                "eesm_beta_db": beta_db,
                "snr_threshold_db": float(threshold_db),
                "bler_slope_db": float(slope_db),
            }
        )
        diagnostics.append(
            {
                "mcs_index": index,
                "observations": len(rows),
                "simulated_blocks": int(blocks.sum()),
                "simulated_block_errors": int(errors.sum()),
                "negative_log_likelihood": nll,
                "eesm_beta_db": beta_db,
                "snr_threshold_db_at_target_bler": float(threshold_db),
                "bler_slope_db": float(slope_db),
                "effective_snr_min_db": float(effective.min()),
                "effective_snr_max_db": float(effective.max()),
            }
        )

    thresholds = [float(row["snr_threshold_db"]) for row in fitted_rows]
    if any(right <= left for left, right in zip(thresholds, thresholds[1:])):
        raise RuntimeError(
            "Fitted 10%-BLER thresholds are not strictly increasing with MCS: "
            f"{thresholds}. Increase simulation coverage or inspect the PHY setup."
        )
    output = {
        **config,
        "format": "csi_mcs_calibrated_v1",
        "effective_channel": {
            **dict(config.get("effective_channel", {})),
            "mode": mode,
        },
        "mcs": fitted_rows,
        "calibration": {
            "selection": str(args.selection.resolve()),
            "simulation_results": str(args.results.resolve()),
            "simulation_results_sha256": file_sha256(args.results),
            "uses_training_links_only": True,
            "source_train_pt": str(Path(selection["source_train_pt"]).resolve()),
            "effective_channel_mode": mode,
            "fit_method": "EESM beta search plus binomial logistic BLER fit",
            "diagnostics": diagnostics,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    print(f"calibrated_config={args.output.resolve()}")


if __name__ == "__main__":
    main()
