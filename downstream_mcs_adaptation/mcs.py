from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class CalibratedMCS:
    index: int
    name: str
    modulation: str
    coding_rate: float
    spectral_efficiency: float
    eesm_beta_db: float
    snr_threshold_db: float
    bler_slope_db: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def validate_uncalibrated_config(config: dict[str, Any]) -> None:
    system = config.get("system")
    rows = config.get("mcs")
    if not isinstance(system, dict) or not isinstance(rows, list) or not rows:
        raise ValueError("MCS config must contain system and a non-empty mcs list")
    expected = list(range(len(rows)))
    actual = [int(row.get("index", -1)) for row in rows]
    if actual != expected:
        raise ValueError(f"MCS indices must be contiguous from zero; found {actual}")
    efficiencies = [float(row["spectral_efficiency"]) for row in rows]
    if any(value <= 0.0 for value in efficiencies) or any(
        right <= left for left, right in zip(efficiencies, efficiencies[1:])
    ):
        raise ValueError("MCS spectral efficiencies must be positive and strictly increasing")
    names = [str(row.get("name", "")).strip() for row in rows]
    if any(not name for name in names) or len(set(names)) != len(rows):
        raise ValueError("MCS names must be non-empty and unique")
    if any(not 0.0 < float(row["coding_rate"]) <= 1.0 for row in rows):
        raise ValueError("Every MCS coding rate must be in (0, 1]")
    target_bler = float(system.get("target_bler", math.nan))
    if not 0.0 < target_bler < 1.0:
        raise ValueError("system.target_bler must be in (0, 1)")
    bandwidth_hz = float(system.get("bandwidth_hz", math.nan))
    if not math.isfinite(bandwidth_hz) or bandwidth_hz <= 0.0:
        raise ValueError("system.bandwidth_hz must be positive")
    if int(system.get("subcarriers", 0)) <= 0:
        raise ValueError("system.subcarriers must be positive")
    for name in ("noise_figure_db", "thermal_noise_density_dbm_hz"):
        if not math.isfinite(float(system.get(name, math.nan))):
            raise ValueError(f"system.{name} must be finite")


def load_calibrated_config(path: Path) -> tuple[dict[str, Any], list[CalibratedMCS]]:
    config = load_json(path)
    validate_uncalibrated_config(config)
    if config.get("format") != "csi_mcs_calibrated_v1":
        raise ValueError(
            f"{path} is not calibrated. Run fit_calibration.py; do not invent EESM beta or SNR thresholds."
        )
    calibrated: list[CalibratedMCS] = []
    for row in config["mcs"]:
        required = ("eesm_beta_db", "snr_threshold_db", "bler_slope_db")
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(f"MCS {row.get('index')} lacks calibrated fields {missing}")
        calibrated.append(
            CalibratedMCS(
                index=int(row["index"]),
                name=str(row["name"]),
                modulation=str(row["modulation"]),
                coding_rate=float(row["coding_rate"]),
                spectral_efficiency=float(row["spectral_efficiency"]),
                eesm_beta_db=float(row["eesm_beta_db"]),
                snr_threshold_db=float(row["snr_threshold_db"]),
                bler_slope_db=float(row["bler_slope_db"]),
            )
        )
    if any(
        not math.isfinite(value)
        for item in calibrated
        for value in (item.eesm_beta_db, item.snr_threshold_db, item.bler_slope_db)
    ):
        raise ValueError("All calibrated parameters must be finite")
    if any(item.bler_slope_db <= 0.0 for item in calibrated):
        raise ValueError("Every calibrated BLER slope must be positive")
    thresholds = [item.snr_threshold_db for item in calibrated]
    if any(right <= left for left, right in zip(thresholds, thresholds[1:])):
        raise ValueError(
            "Calibrated 10%-BLER thresholds must increase strictly with MCS index; "
            f"found {thresholds}. Inspect the full-link simulation."
        )
    return config, calibrated


def noise_power_dbm(system: dict[str, Any]) -> float:
    return (
        float(system["thermal_noise_density_dbm_hz"])
        + 10.0 * math.log10(float(system["bandwidth_hz"]))
        + float(system["noise_figure_db"])
    )


def noise_power_dbw(system: dict[str, Any]) -> float:
    return noise_power_dbm(system) - 30.0


def sample_bandwidth_hz(sample: Any) -> float | None:
    try:
        value = float(getattr(sample, "bandwidth_hz", math.nan))
    except (TypeError, ValueError):
        value = math.nan
    if math.isfinite(value) and value > 0.0:
        return value
    try:
        spacing = float(sample.subcarrier_spacing_hz)
        source_n_freq = int(getattr(sample, "source_n_freq", 0) or 0)
        if source_n_freq <= 0:
            source_n_freq = int(sample.tokens.shape[-1])
        value = spacing * source_n_freq
    except (AttributeError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def complex_beam_csi(sample: Any) -> torch.Tensor:
    """Recover [beam, subcarrier] complex CSI from beam-patch tokens.

    Beamspace tokenization stores all real patch entries followed by the matching
    imaginary entries. Zero-padded edge beams are harmless and remain zero.
    """

    tokens = torch.as_tensor(sample.tokens[: int(sample.n_tokens)]).float()
    if tokens.ndim != 3 or tokens.shape[1] == 0 or tokens.shape[1] % 2:
        raise ValueError(
            f"Expected tokens [beam_patch, even_d_token, subcarrier], got {tuple(tokens.shape)}"
        )
    half = tokens.shape[1] // 2
    return torch.complex(tokens[:, :half], tokens[:, half:]).reshape(-1, tokens.shape[-1])


def effective_frequency_response(sample: Any, mode: str = "best_beam") -> np.ndarray:
    beam_csi = complex_beam_csi(sample)
    beam_power = beam_csi.abs().square()
    nonempty = beam_power.sum(dim=1) > 0.0
    if not bool(nonempty.any()):
        raise ValueError(f"No nonzero CSI beam in group_id={getattr(sample, 'group_id', '?')}")
    beam_power = beam_power[nonempty]
    beam_csi = beam_csi[nonempty]
    if mode == "best_beam":
        selected = int(beam_power.mean(dim=1).argmax())
        response = beam_csi[selected]
    elif mode == "mrc":
        # Equivalent post-combining scalar channel; its squared magnitude is the
        # sum of branch powers under perfect MRC.
        response = torch.complex(torch.sqrt(beam_power.sum(dim=0)), torch.zeros_like(beam_power[0]))
    else:
        raise ValueError(f"Unknown effective-channel mode {mode!r}; use best_beam or mrc")
    mean_power = response.abs().square().double().mean()
    if not bool(torch.isfinite(mean_power)) or float(mean_power) <= 0.0:
        raise ValueError(f"Invalid CSI power for group_id={getattr(sample, 'group_id', '?')}")
    normalized = response / torch.sqrt(mean_power.to(response.real.dtype))
    return normalized.cpu().numpy().astype(np.complex64, copy=False)


def subcarrier_power_weights(sample: Any, mode: str = "best_beam") -> np.ndarray:
    response = effective_frequency_response(sample, mode)
    frequency_power = np.abs(response).astype(np.float64) ** 2
    weights = frequency_power / float(np.mean(frequency_power))
    return np.asarray(weights, dtype=np.float64)


def frequency_selectivity_db(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    values = values[np.isfinite(values) & (values >= 0.0)]
    if values.size == 0:
        return math.nan
    values = np.maximum(values, 1e-12)
    return float(10.0 * np.log10(np.percentile(values, 90) / np.percentile(values, 10)))


def per_subcarrier_snr_linear(average_snr_db: float, weights: np.ndarray) -> np.ndarray:
    if not math.isfinite(float(average_snr_db)):
        raise ValueError("average_snr_db must be finite")
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or not values.size or np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("subcarrier weights must be a finite nonnegative vector")
    mean = float(values.mean())
    if mean <= 0.0:
        raise ValueError("subcarrier weights have zero mean")
    return np.power(10.0, float(average_snr_db) / 10.0) * (values / mean)


def eesm_effective_snr_db(
    average_snr_db: float,
    weights: np.ndarray,
    beta_db: float,
) -> float:
    gamma = per_subcarrier_snr_linear(average_snr_db, weights)
    beta = float(np.power(10.0, float(beta_db) / 10.0))
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("EESM beta must be positive in linear units")
    scaled = -gamma / beta
    maximum = float(np.max(scaled))
    log_mean_exp = maximum + math.log(float(np.exp(scaled - maximum).mean()))
    effective_linear = max(-beta * log_mean_exp, np.finfo(np.float64).tiny)
    return float(10.0 * math.log10(effective_linear))


def estimate_bler(
    effective_snr_db: float,
    threshold_db: float,
    slope_db: float,
    target_bler: float,
) -> float:
    """Calibrated logistic curve parameterized at the target-BLER crossing."""

    target_logit = math.log(target_bler / (1.0 - target_bler))
    logit = target_logit - (float(effective_snr_db) - float(threshold_db)) / float(slope_db)
    if logit >= 0.0:
        return float(1.0 / (1.0 + math.exp(-min(logit, 700.0))))
    exp_logit = math.exp(max(logit, -700.0))
    return float(exp_logit / (1.0 + exp_logit))


def label_link(
    sample: Any,
    calibrated: list[CalibratedMCS],
    system: dict[str, Any],
    effective_channel_mode: str,
    power_offset_db: float = 0.0,
) -> dict[str, Any]:
    base_total_power_dbw = float(getattr(sample, "total_received_power_dbw", math.nan))
    if not math.isfinite(base_total_power_dbw):
        raise ValueError(
            f"Missing total_received_power_dbw for group_id={getattr(sample, 'group_id', '?')}"
        )
    if not math.isfinite(float(power_offset_db)):
        raise ValueError("power_offset_db must be finite")
    total_power_dbw = base_total_power_dbw + float(power_offset_db)
    weights = subcarrier_power_weights(sample, effective_channel_mode)
    expected_subcarriers = int(system.get("subcarriers", len(weights)))
    if len(weights) != expected_subcarriers:
        raise ValueError(
            f"CSI has {len(weights)} subcarriers but system calibration expects "
            f"{expected_subcarriers} for group_id={getattr(sample, 'group_id', '?')}"
        )
    average_snr_db = total_power_dbw - noise_power_dbw(system)
    target_bler = float(system["target_bler"])
    effective = np.asarray(
        [eesm_effective_snr_db(average_snr_db, weights, item.eesm_beta_db) for item in calibrated],
        dtype=np.float32,
    )
    bler = np.asarray(
        [
            estimate_bler(value, item.snr_threshold_db, item.bler_slope_db, target_bler)
            for value, item in zip(effective, calibrated)
        ],
        dtype=np.float32,
    )
    feasible = np.flatnonzero(bler <= target_bler + 1e-7)
    oracle_mcs = int(feasible[-1]) if feasible.size else -1
    return {
        "base_total_received_power_dbw": base_total_power_dbw,
        "power_offset_db": float(power_offset_db),
        "total_received_power_dbw": total_power_dbw,
        "noise_power_dbw": noise_power_dbw(system),
        "average_snr_db": average_snr_db,
        "frequency_selectivity_db": frequency_selectivity_db(weights),
        "effective_snr_per_mcs": effective,
        "estimated_bler_per_mcs": bler,
        "oracle_mcs": oracle_mcs,
        "class_index": oracle_mcs + 1,
    }
