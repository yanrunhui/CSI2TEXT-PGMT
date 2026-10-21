from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from preprocess_bupt_cir import (
    compensate_system_delay,
    legacy_upa_1x1_tokens,
    native_siso_tokens,
    phase_align,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply one calibrated fixed system delay to existing BUPT NPZ CFRs. "
            "This does not need the original large MAT files."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    delay = parser.add_mutually_exclusive_group(required=True)
    delay.add_argument("--system-delay-ns", type=float)
    delay.add_argument("--calibration-json", type=Path)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def calibrated_delay(args: argparse.Namespace) -> tuple[float, str]:
    if args.calibration_json is None:
        assert args.system_delay_ns is not None
        value = float(args.system_delay_ns)
        source = "command_line"
    else:
        with args.calibration_json.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        if "system_delay_ns" not in document:
            raise ValueError(f"system_delay_ns missing from {args.calibration_json}")
        value = float(document["system_delay_ns"])
        source = str(args.calibration_json)
    if not math.isfinite(value):
        raise ValueError(f"System delay must be finite, got {value}")
    return value, source


def scalar(data: dict[str, np.ndarray], name: str, default: float) -> float:
    if name not in data:
        return default
    values = np.asarray(data[name]).reshape(-1)
    return float(values[0]) if values.size else default


def process_file(
    input_path: Path,
    output_path: Path,
    system_delay_ns: float,
    calibration_source: str,
) -> dict[str, object]:
    with np.load(input_path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}

    if "selected_frequency_hz" not in data:
        raise KeyError(f"selected_frequency_hz missing from {input_path}")
    # New shards preserve the pre-compensation CFR explicitly. Older shards were
    # generated with zero compensation, so their cfr_raw is the same base value.
    base_cfr = data.get("cfr_before_system_delay_compensation", data.get("cfr_raw"))
    if base_cfr is None:
        raise KeyError(f"Neither base CFR field exists in {input_path}")
    frequencies_hz = np.asarray(data["selected_frequency_hz"], dtype=np.float64)
    calibrated_cfr = compensate_system_delay(
        np.asarray(base_cfr),
        frequencies_hz=frequencies_hz,
        system_delay_ns=system_delay_ns,
    )
    aligned_cfr = phase_align(calibrated_cfr)
    layout = str(np.asarray(data.get("token_layout", "native_siso")).reshape(-1)[0])
    if layout == "native_siso":
        tokens = native_siso_tokens(aligned_cfr)
    elif layout == "legacy_upa_2x2_padded":
        tokens = legacy_upa_1x1_tokens(aligned_cfr)
    else:
        raise ValueError(f"Unsupported token_layout={layout!r} in {input_path}")

    uncalibrated_first_ns = scalar(
        data, "first_path_delay_uncalibrated_ns", float("nan")
    )
    sample_rate_mhz = scalar(data, "sampleRateMHz", float("nan"))
    first_path_tap = scalar(data, "first_path_tap", float("nan"))
    sample_period_ns = 1000.0 / sample_rate_mhz if sample_rate_mhz > 0.0 else float("nan")
    data.update(
        {
            "cfr_before_system_delay_compensation": np.asarray(base_cfr),
            "cfr_raw": calibrated_cfr,
            "cfr_phase_aligned": aligned_cfr,
            "tokens": tokens,
            "system_delay_compensation_ns": np.asarray(system_delay_ns, dtype=np.float64),
            "system_delay_calibration_source": np.asarray(calibration_source),
            "first_path_delay_calibrated_ns": np.asarray(
                uncalibrated_first_ns - system_delay_ns, dtype=np.float64
            ),
            "first_path_tap_calibrated_float": np.asarray(
                first_path_tap - system_delay_ns / sample_period_ns,
                dtype=np.float64,
            ),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data)
    os.replace(temporary, output_path)
    return {
        "timestamp": input_path.stem,
        "input_file": str(input_path),
        "processed_file": str(output_path),
        "system_delay_compensation_ns": system_delay_ns,
        "first_path_delay_uncalibrated_ns": uncalibrated_first_ns,
        "first_path_delay_calibrated_ns": uncalibrated_first_ns - system_delay_ns,
        "token_shape": "x".join(str(value) for value in tokens.shape),
        "token_layout": layout,
    }


def main() -> None:
    args = parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError("input-dir and output-dir must differ; preserve the uncalibrated A/B data")
    system_delay_ns, source = calibrated_delay(args)
    paths = sorted(args.input_dir.glob("*.npz"))
    if args.max_files is not None:
        if args.max_files <= 0:
            raise ValueError("max-files must be positive")
        paths = paths[: args.max_files]
    if not paths:
        raise ValueError(f"No NPZ files found in {args.input_dir}")

    rows: list[dict[str, object]] = []
    for index, input_path in enumerate(paths):
        output_path = args.output_dir / input_path.name
        if output_path.exists() and not args.overwrite:
            print(f"[{index + 1:03d}/{len(paths):03d}] existing: {output_path.name}")
            continue
        rows.append(process_file(input_path, output_path, system_delay_ns, source))
        print(f"[{index + 1:03d}/{len(paths):03d}] calibrated: {output_path.name}")

    if rows:
        manifest = args.output_dir / "system_delay_manifest.csv"
        temporary = manifest.with_name(manifest.name + ".part")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, manifest)
    print(f"System delay removed: {system_delay_ns:g} ns")
    print(f"Calibration source: {source}")
    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
