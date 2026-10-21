from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
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

from downstream_localization.data_io import LocalizationCorpus  # noqa: E402
from downstream_localization.run_physical_consistency import (  # noqa: E402
    extract_predictions,
    finite_mask,
    pearson,
    sample_for_plot,
    selected_indices,
    spearman,
)


EPS = 1.0e-30


@dataclass(frozen=True)
class PDPOptions:
    n_fft_factor: int = 4
    pdp_threshold_db: float = 20.0
    max_delay_ns: float = 3000.0
    power_window_resolution: float = 1.0
    patch_1d: int = 4
    patch_rows: int = 2
    patch_cols: int = 2


@dataclass(frozen=True)
class PDPEstimate:
    pdp_first_delay_ns: float
    pdp_peak_delay_ns: float
    pdp_delay_spread_ns: float
    pdp_peak_power_db: float
    pdp_retained_power_db: float
    pdp_peak_to_tail_db: float
    raw_csi_energy_db: float
    retained_bin_count: int
    peak_bin_index: int
    first_bin_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Second-layer real-CSI validation: compare frozen model physical outputs "
            "with independent classical estimators from IFFT-PDP / raw CSI energy. "
            "This is an agreement test, not path-level accuracy."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        choices=("train", "val", "test", "all"),
        help="Corpus splits to analyze. Default: test.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=23421)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--n-fft-factor",
        type=int,
        default=4,
        help="Zero-padding factor before IFFT-PDP estimation.",
    )
    parser.add_argument(
        "--pdp-threshold-db",
        type=float,
        default=20.0,
        help=(
            "Retain PDP bins within this many dB below the peak. The first retained "
            "bin is reported as PDP first delay; retained bins define delay spread."
        ),
    )
    parser.add_argument(
        "--max-delay-ns",
        type=float,
        default=3000.0,
        help="Ignore PDP bins after this delay when estimating first delay/spread.",
    )
    parser.add_argument(
        "--power-window-resolution",
        type=float,
        default=1.0,
        help="Peak-power integration half-window in native delay-resolution units.",
    )
    parser.add_argument("--patch-1d", type=int, default=4)
    parser.add_argument("--patch-rows", type=int, default=2)
    parser.add_argument("--patch-cols", type=int, default=2)
    parser.add_argument(
        "--scatter-samples",
        type=int,
        default=4000,
        help="Maximum points per scatter panel.",
    )
    return parser.parse_args()


def complex_token_chunks(sample: Any) -> np.ndarray:
    tokens = sample.tokens[: int(sample.n_tokens)].detach().cpu().float()
    if tokens.ndim != 3 or tokens.shape[1] % 2:
        raise ValueError(
            "Expected tokens with shape [K, 2*C, F] and real/imaginary halves, "
            f"got {tuple(tokens.shape)} for group_id={getattr(sample, 'group_id', '')!r}."
        )
    half = tokens.shape[1] // 2
    return (
        tokens[:, :half].numpy().astype(np.float64, copy=False)
        + 1j * tokens[:, half:].numpy().astype(np.float64, copy=False)
    )


def reconstruct_beam_csi(sample: Any, options: PDPOptions) -> np.ndarray:
    chunks = complex_token_chunks(sample)
    array_type = str(sample.array_type).upper()
    rows = int(sample.array_rows)
    cols = int(sample.array_cols)
    n_freq = int(chunks.shape[-1])

    if array_type == "ULA":
        n_antennas = rows * cols
        patch = int(options.patch_1d)
        if chunks.shape[1] != patch:
            raise ValueError(
                f"ULA token has {chunks.shape[1]} complex channels but --patch-1d={patch}."
            )
        expected_tokens = math.ceil(n_antennas / patch)
        if chunks.shape[0] != expected_tokens:
            raise ValueError(
                f"ULA token count mismatch: got {chunks.shape[0]}, expected {expected_tokens}."
            )
        beam = np.zeros((n_antennas, n_freq), dtype=np.complex128)
        for token_index in range(expected_tokens):
            start = token_index * patch
            count = min(patch, n_antennas - start)
            beam[start : start + count] = chunks[token_index, :count]
        return beam

    if array_type != "UPA":
        raise ValueError(f"Unsupported array_type={array_type!r}; expected ULA or UPA.")

    patch_rows = int(options.patch_rows)
    patch_cols = int(options.patch_cols)
    patch_area = patch_rows * patch_cols
    if chunks.shape[1] != patch_area:
        raise ValueError(
            "UPA token complex-channel count does not match --patch-rows * --patch-cols: "
            f"{chunks.shape[1]} != {patch_rows} * {patch_cols}."
        )
    token_rows = math.ceil(rows / patch_rows)
    token_cols = math.ceil(cols / patch_cols)
    if chunks.shape[0] != token_rows * token_cols:
        raise ValueError(
            f"UPA token count mismatch: got {chunks.shape[0]}, "
            f"expected {token_rows * token_cols} for {rows}x{cols}."
        )

    beam_grid = np.zeros((rows, cols, n_freq), dtype=np.complex128)
    token_index = 0
    for token_row in range(token_rows):
        for token_col in range(token_cols):
            patch = chunks[token_index].reshape(patch_rows, patch_cols, n_freq)
            row_start = token_row * patch_rows
            col_start = token_col * patch_cols
            valid_rows = min(patch_rows, rows - row_start)
            valid_cols = min(patch_cols, cols - col_start)
            beam_grid[
                row_start : row_start + valid_rows,
                col_start : col_start + valid_cols,
            ] = patch[:valid_rows, :valid_cols]
            token_index += 1
    return beam_grid.reshape(rows * cols, n_freq)


def beam_to_antenna_csi(sample: Any, beam_csi: np.ndarray) -> np.ndarray:
    rows = int(sample.array_rows)
    cols = int(sample.array_cols)
    if str(sample.array_type).upper() == "ULA":
        return np.fft.ifft(beam_csi, axis=0)
    beam_grid = beam_csi.reshape(rows, cols, beam_csi.shape[-1])
    return np.fft.ifft2(beam_grid, axes=(0, 1)).reshape(rows * cols, beam_csi.shape[-1])


def frequency_to_delay(antenna_csi: np.ndarray, n_fft_factor: int) -> np.ndarray:
    n_freq = int(antenna_csi.shape[-1])
    if n_fft_factor <= 0:
        raise ValueError("--n-fft-factor must be positive")
    n_fft = n_freq * int(n_fft_factor)
    left = (n_fft - n_freq) // 2
    centered = np.zeros((*antenna_csi.shape[:-1], n_fft), dtype=np.complex128)
    centered[..., left : left + n_freq] = antenna_csi
    return (
        np.fft.ifft(np.fft.ifftshift(centered, axes=-1), axis=-1)
        * (float(n_fft) / float(n_freq))
    )


def sample_bandwidth_hz(sample: Any, n_freq: int) -> float:
    bandwidth_hz = float(getattr(sample, "bandwidth_hz", math.nan))
    if math.isfinite(bandwidth_hz) and bandwidth_hz > 0.0:
        return bandwidth_hz
    spacing_hz = float(getattr(sample, "subcarrier_spacing_hz", math.nan))
    if math.isfinite(spacing_hz) and spacing_hz > 0.0:
        source_n_freq = int(getattr(sample, "source_n_freq", 0) or n_freq)
        return spacing_hz * float(source_n_freq)
    raise ValueError(
        f"Missing positive bandwidth/subcarrier spacing for group_id={sample.group_id!r}"
    )


def estimate_pdp(sample: Any, options: PDPOptions) -> PDPEstimate:
    beam_csi = reconstruct_beam_csi(sample, options)
    antenna_csi = beam_to_antenna_csi(sample, beam_csi)
    delay_response = frequency_to_delay(antenna_csi, options.n_fft_factor)
    pdp = np.sum(np.abs(delay_response) ** 2, axis=0).real
    if not np.isfinite(pdp).any() or float(np.nanmax(pdp)) <= 0.0:
        raise ValueError(f"Zero/non-finite PDP for group_id={sample.group_id!r}")

    n_antennas = max(int(antenna_csi.shape[0]), 1)
    n_freq = int(antenna_csi.shape[-1])
    bandwidth_hz = sample_bandwidth_hz(sample, n_freq)
    spacing_hz = bandwidth_hz / float(n_freq)
    n_fft = int(delay_response.shape[-1])
    delay_step_ns = 1.0e9 / (float(n_fft) * spacing_hz)
    native_resolution_ns = 1.0e9 / (float(n_freq) * spacing_hz)
    delays_ns = np.arange(n_fft, dtype=np.float64) * delay_step_ns

    search_mask = delays_ns <= float(options.max_delay_ns)
    if not bool(np.any(search_mask)):
        raise ValueError("--max-delay-ns excludes every delay bin")
    search_count = int(np.flatnonzero(search_mask)[-1]) + 1
    profile = pdp[:search_count]
    peak_bin_index = int(np.argmax(profile))
    peak_power = float(profile[peak_bin_index])
    threshold = peak_power * 10.0 ** (-float(options.pdp_threshold_db) / 10.0)
    retained = np.zeros_like(pdp, dtype=bool)
    retained[:search_count] = profile >= threshold
    if not bool(np.any(retained)):
        retained[peak_bin_index] = True
    first_bin_index = int(np.flatnonzero(retained)[0])

    retained_power_bins = pdp[retained]
    retained_delays = delays_ns[retained]
    retained_sum = float(np.sum(retained_power_bins))
    if retained_sum <= 0.0:
        delay_spread_ns = 0.0
    else:
        mean_delay = float(np.sum(retained_power_bins * retained_delays) / retained_sum)
        variance = float(
            np.sum(retained_power_bins * (retained_delays - mean_delay) ** 2)
            / retained_sum
        )
        delay_spread_ns = math.sqrt(max(variance, 0.0))

    half_window_ns = (
        0.5 * float(options.power_window_resolution) * native_resolution_ns
    )
    half_window_bins = max(0, int(math.ceil(half_window_ns / delay_step_ns)))
    start = max(0, peak_bin_index - half_window_bins)
    stop = min(len(pdp), peak_bin_index + half_window_bins + 1)
    peak_integrated_power = float(np.sum(pdp[start:stop])) / (
        float(n_antennas) * float(options.n_fft_factor)
    )
    retained_integrated_power = retained_sum / (
        float(n_antennas) * float(options.n_fft_factor)
    )
    tail_power = max(retained_integrated_power - peak_integrated_power, EPS)
    raw_csi_energy = float(np.mean(np.abs(antenna_csi) ** 2))

    return PDPEstimate(
        pdp_first_delay_ns=float(delays_ns[first_bin_index]),
        pdp_peak_delay_ns=float(delays_ns[peak_bin_index]),
        pdp_delay_spread_ns=float(delay_spread_ns),
        pdp_peak_power_db=10.0 * math.log10(max(peak_integrated_power, EPS)),
        pdp_retained_power_db=10.0 * math.log10(max(retained_integrated_power, EPS)),
        pdp_peak_to_tail_db=10.0
        * math.log10(max(peak_integrated_power, EPS) / tail_power),
        raw_csi_energy_db=10.0 * math.log10(max(raw_csi_energy, EPS)),
        retained_bin_count=int(np.count_nonzero(retained)),
        peak_bin_index=peak_bin_index,
        first_bin_index=first_bin_index,
    )


def estimate_classical_for_indices(
    corpus: LocalizationCorpus,
    indices: np.ndarray,
    options: PDPOptions,
) -> dict[str, np.ndarray]:
    rows: dict[str, list[float]] = {
        "pdp_first_delay_ns": [],
        "pdp_peak_delay_ns": [],
        "pdp_delay_spread_ns": [],
        "pdp_peak_power_db": [],
        "pdp_retained_power_db": [],
        "pdp_peak_to_tail_db": [],
        "raw_csi_energy_db": [],
        "retained_bin_count": [],
        "peak_bin_index": [],
        "first_bin_index": [],
    }
    for order, index in enumerate(indices, start=1):
        estimate = estimate_pdp(corpus.samples[int(index)], options)
        record = asdict(estimate)
        for key in rows:
            rows[key].append(record[key])
        if order % 1000 == 0 or order == len(indices):
            print(f"classical PDP samples={order}/{len(indices)}")
    return {key: np.asarray(values) for key, values in rows.items()}


def safe_linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = finite_mask(x, y)
    count = int(mask.sum())
    result = {
        "count": count,
        "slope": float("nan"),
        "intercept": float("nan"),
        "r2": float("nan"),
        "pearson": pearson(x, y),
        "spearman": spearman(x, y),
    }
    if count < 2:
        return result

    x_fit = x[mask].astype(np.float64)
    y_fit = y[mask].astype(np.float64)
    if float(np.nanmax(x_fit) - np.nanmin(x_fit)) <= 0.0:
        result["fit_status"] = "skipped_constant_reference"
        return result
    if float(np.nanmax(y_fit) - np.nanmin(y_fit)) <= 0.0:
        result["fit_status"] = "skipped_constant_model"
        return result

    try:
        slope, intercept = np.polyfit(x_fit, y_fit, deg=1)
        fitted = slope * x_fit + intercept
        ss_res = float(np.square(y_fit - fitted).sum())
        ss_tot = float(np.square(y_fit - y_fit.mean()).sum())
    except np.linalg.LinAlgError:
        result["fit_status"] = "failed_linalg"
        return result
    except FloatingPointError:
        result["fit_status"] = "failed_floating_point"
        return result

    result["slope"] = float(slope)
    result["intercept"] = float(intercept)
    result["r2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    result["fit_status"] = "ok"
    return result


def agreement_stats(
    reference: np.ndarray,
    model: np.ndarray,
    *,
    include_error: bool,
) -> dict[str, Any]:
    stats = safe_linear_fit(reference, model)
    stats["reference_mean"] = float(np.nanmean(reference)) if len(reference) else float("nan")
    stats["model_mean"] = float(np.nanmean(model)) if len(model) else float("nan")
    if include_error:
        mask = finite_mask(reference, model)
        if int(mask.sum()) > 0:
            error = model[mask].astype(np.float64) - reference[mask].astype(np.float64)
            stats["mae"] = float(np.mean(np.abs(error)))
            stats["median_abs_error"] = float(np.median(np.abs(error)))
            stats["median_bias"] = float(np.median(error))
        else:
            stats["mae"] = float("nan")
            stats["median_abs_error"] = float("nan")
            stats["median_bias"] = float("nan")
    return stats


def metrics_for_mask(
    predictions: dict[str, np.ndarray],
    classical: dict[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, Any]:
    total_power = predictions["total_power_dbw"]
    model_power_key = (
        "total_power_dbw" if np.isfinite(total_power[mask]).any() else "first_power_dbw"
    )
    return {
        "first_delay_agreement": agreement_stats(
            classical["pdp_first_delay_ns"][mask],
            predictions["first_delay_ns"][mask],
            include_error=False,
        ),
        "delay_spread_agreement": agreement_stats(
            classical["pdp_delay_spread_ns"][mask],
            predictions["delay_spread_ns"][mask],
            include_error=True,
        ),
        "k_factor_vs_peak_to_tail_agreement": agreement_stats(
            classical["pdp_peak_to_tail_db"][mask],
            predictions["k_factor_db"][mask],
            include_error=True,
        ),
        "power_vs_raw_csi_energy_agreement": {
            "model_power_key": model_power_key,
            **agreement_stats(
                classical["raw_csi_energy_db"][mask],
                predictions[model_power_key][mask],
                include_error=True,
            ),
        },
        "first_power_vs_pdp_peak_power_agreement": agreement_stats(
            classical["pdp_peak_power_db"][mask],
            predictions["first_power_dbw"][mask],
            include_error=True,
        ),
    }


def build_metrics(
    predictions: dict[str, np.ndarray],
    classical: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    labels = predictions["los_label"]
    groups: dict[str, np.ndarray] = {
        "all": np.ones(len(labels), dtype=bool),
        "los": labels == "los",
        "nlos": labels == "nlos",
    }
    return {
        "sample_count": int(len(labels)),
        "label_counts": {
            "los": int((labels == "los").sum()),
            "nlos": int((labels == "nlos").sum()),
        },
        "note": (
            "These are agreement metrics with independent PDP/raw-CSI estimators, "
            "not path-level ground-truth accuracy."
        ),
        "pdp_options": {
            "n_fft_factor": int(args.n_fft_factor),
            "pdp_threshold_db": float(args.pdp_threshold_db),
            "max_delay_ns": float(args.max_delay_ns),
            "power_window_resolution": float(args.power_window_resolution),
        },
        "groups": {
            name: metrics_for_mask(predictions, classical, mask)
            for name, mask in groups.items()
            if int(mask.sum()) > 0
        },
    }


def flatten_metrics_for_csv(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, group_metrics in metrics["groups"].items():
        for comparison, values in group_metrics.items():
            row = {"group": group, "comparison": comparison}
            for key, value in values.items():
                if isinstance(value, (str, int, float)):
                    row[key] = value
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scatter_panel(
    axis: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    scatter_samples: int,
    seed: int,
) -> None:
    x_plot, y_plot = sample_for_plot(x, y, scatter_samples, seed)
    mask = finite_mask(x, y)
    if len(x_plot):
        axis.scatter(x_plot, y_plot, s=8, alpha=0.35)
    if int(mask.sum()) >= 2 and len(x_plot):
        x_fit = x[mask].astype(np.float64)
        y_fit = y[mask].astype(np.float64)
        if (
            float(np.nanmax(x_fit) - np.nanmin(x_fit)) > 0.0
            and float(np.nanmax(y_fit) - np.nanmin(y_fit)) > 0.0
        ):
            try:
                slope, intercept = np.polyfit(x_fit, y_fit, 1)
                grid = np.linspace(float(np.nanmin(x_plot)), float(np.nanmax(x_plot)), 100)
                axis.plot(grid, slope * grid + intercept, color="tab:red", linewidth=2)
            except np.linalg.LinAlgError:
                pass
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(f"{title}\nSpearman={spearman(x, y):.3f}, Pearson={pearson(x, y):.3f}")
    axis.grid(alpha=0.3)


def plot_agreement(
    predictions: dict[str, np.ndarray],
    classical: dict[str, np.ndarray],
    output: Path,
    scatter_samples: int,
    seed: int,
) -> None:
    total_power = predictions["total_power_dbw"]
    model_power_key = "total_power_dbw" if np.isfinite(total_power).any() else "first_power_dbw"

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    scatter_panel(
        axes[0, 0],
        classical["pdp_first_delay_ns"],
        predictions["first_delay_ns"],
        xlabel="PDP first delay (ns)",
        ylabel="Model first delay (ns)",
        title="First-delay agreement",
        scatter_samples=scatter_samples,
        seed=seed,
    )
    scatter_panel(
        axes[0, 1],
        classical["pdp_delay_spread_ns"],
        predictions["delay_spread_ns"],
        xlabel="PDP RMS delay spread (ns)",
        ylabel="Model delay spread (ns)",
        title="Delay-spread agreement",
        scatter_samples=scatter_samples,
        seed=seed + 1,
    )
    scatter_panel(
        axes[1, 0],
        classical["pdp_peak_to_tail_db"],
        predictions["k_factor_db"],
        xlabel="PDP peak-to-tail ratio (dB)",
        ylabel="Model K-factor (dB)",
        title="K-factor proxy agreement",
        scatter_samples=scatter_samples,
        seed=seed + 2,
    )
    scatter_panel(
        axes[1, 1],
        classical["raw_csi_energy_db"],
        predictions[model_power_key],
        xlabel="Raw CSI energy (dB)",
        ylabel=f"Model {model_power_key} (dB)",
        title="Power agreement",
        scatter_samples=scatter_samples,
        seed=seed + 3,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.n_fft_factor <= 0:
        raise ValueError("--n-fft-factor must be positive")
    if args.pdp_threshold_db <= 0.0:
        raise ValueError("--pdp-threshold-db must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corpus = LocalizationCorpus.load(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    predictions = extract_predictions(corpus, checkpoint, args.checkpoint, args)
    indices = predictions["index"].astype(np.int64)
    expected_indices = selected_indices(corpus, args).detach().cpu().numpy().astype(np.int64)
    if not np.array_equal(indices, expected_indices):
        raise RuntimeError("Prediction order does not match selected corpus indices")

    options = PDPOptions(
        n_fft_factor=int(args.n_fft_factor),
        pdp_threshold_db=float(args.pdp_threshold_db),
        max_delay_ns=float(args.max_delay_ns),
        power_window_resolution=float(args.power_window_resolution),
        patch_1d=int(args.patch_1d),
        patch_rows=int(args.patch_rows),
        patch_cols=int(args.patch_cols),
    )
    classical = estimate_classical_for_indices(corpus, indices, options)
    metrics = build_metrics(predictions, classical, args)

    np.savez_compressed(
        args.output_dir / "classical_agreement_predictions.npz",
        **{f"model_{key}": value for key, value in predictions.items()},
        **{f"classical_{key}": value for key, value in classical.items()},
    )
    (args.output_dir / "classical_agreement_metrics.json").write_text(
        json.dumps(
            {
                "data": str(args.data.resolve()),
                "checkpoint": str(args.checkpoint.resolve()),
                "splits": args.splits,
                "metrics": metrics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_csv(
        args.output_dir / "classical_agreement_summary.csv",
        flatten_metrics_for_csv(metrics),
    )
    plot_agreement(
        predictions,
        classical,
        args.output_dir / "classical_agreement_scatter.png",
        args.scatter_samples,
        args.seed,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"outputs={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
