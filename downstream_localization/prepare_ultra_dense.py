from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.caption import CaptionGenerator  # noqa: E402
from data.dataset import (  # noqa: E402
    PreprocessedSample,
    empty_delay_power_map,
    empty_delay_power_profile,
)
from data.preprocess import preprocess_sample  # noqa: E402
from data.semantic_key import SemanticKey  # noqa: E402


DEFAULT_DATA_ROOT = Path("/Volumes/One Touch/ultra_dense/ultra_dense/URA_lab_nLoS")
LIGHT_SPEED_MPS = 299_792_458.0


def derive_bw_bin(n_subcarriers: int) -> int:
    if n_subcarriers <= 64:
        return 0
    if n_subcarriers <= 128:
        return 1
    return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert KU Leuven ultra-dense measured CSI .npy files into the "
            "localization corpus format used by downstream_localization.run_experiment."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Keep every N-th existing CSI file after sorting by measurement index.",
    )
    parser.add_argument("--seed", type=int, default=23421)
    parser.add_argument(
        "--split-mode",
        choices=("spatial_tiles", "index_random", "block_holdout"),
        default="spatial_tiles",
        help=(
            "spatial_tiles hashes coarse position tiles; index_random hashes each "
            "measurement index; block_holdout uses one 251x251 quadrant for test."
        ),
    )
    parser.add_argument(
        "--tile-size-m",
        type=float,
        default=0.25,
        help="Tile side length for --split-mode spatial_tiles.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--test-block",
        type=int,
        default=3,
        choices=(0, 1, 2, 3),
        help="Held-out block for --split-mode block_holdout.",
    )
    parser.add_argument(
        "--val-block",
        type=int,
        default=2,
        choices=(0, 1, 2, 3),
        help="Validation block for --split-mode block_holdout.",
    )
    parser.add_argument("--array-rows", type=int, default=8)
    parser.add_argument("--array-cols", type=int, default=8)
    parser.add_argument("--target-nf", type=int, default=100)
    parser.add_argument("--patch-1d", type=int, default=4)
    parser.add_argument("--patch-2d", type=int, nargs=2, default=(2, 2))
    parser.add_argument("--carrier-hz", type=float, default=2.61e9)
    parser.add_argument("--bandwidth-hz", type=float, default=20e6)
    parser.add_argument("--freq-bin", type=int, default=0)
    parser.add_argument(
        "--los-status",
        choices=("auto", "los", "nlos"),
        default="auto",
        help="Propagation label stored in metadata/captions; auto infers it from data-root.",
    )
    parser.add_argument(
        "--reference",
        choices=("array_center", "origin"),
        default="array_center",
        help=(
            "Coordinate reference used as tx_xyz_m. array_center makes labels user "
            "position relative to the measured 8x8 array center."
        ),
    )
    return parser.parse_args()


def stable_unit_interval(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") / float(2**64)


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_stride <= 0:
        raise ValueError("--sample-stride must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided")
    if args.array_rows <= 0 or args.array_cols <= 0:
        raise ValueError("array dimensions must be positive")
    if args.array_rows * args.array_cols != 64:
        raise ValueError("This measured dataset stores 64 antenna CSI snapshots")
    if args.target_nf <= 0 or args.bandwidth_hz <= 0.0 or args.carrier_hz <= 0.0:
        raise ValueError("frequency and bandwidth arguments must be positive")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be in (0, 1)")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1)")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("--train-ratio + --val-ratio must be < 1")
    if args.tile_size_m <= 0.0:
        raise ValueError("--tile-size-m must be positive")
    if args.split_mode == "block_holdout" and args.test_block == args.val_block:
        raise ValueError("--test-block and --val-block must differ")


def measurement_files(samples_dir: Path) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in samples_dir.iterdir():
        name = path.name
        if not name.startswith("channel_measurement_") or not name.endswith(".npy"):
            continue
        try:
            index = int(name[len("channel_measurement_") : -4])
        except ValueError:
            continue
        files.append((index, path))
    files.sort(key=lambda item: item[0])
    return files


def block_id_from_index(index: int) -> int:
    block_size = 251 * 251
    block = index // block_size
    if not 0 <= block <= 3:
        raise ValueError(f"Measurement index {index} is outside the expected 4-block layout")
    return int(block)


def split_name(
    *,
    index: int,
    position_m: np.ndarray,
    args: argparse.Namespace,
) -> str:
    if args.split_mode == "block_holdout":
        block = block_id_from_index(index)
        if block == args.test_block:
            return "test"
        if block == args.val_block:
            return "val"
        return "train"

    if args.split_mode == "spatial_tiles":
        tile = np.floor(position_m[:2] / args.tile_size_m).astype(np.int64)
        key = f"tile:{int(tile[0])}:{int(tile[1])}"
    else:
        key = f"index:{index}"

    value = stable_unit_interval(key, args.seed)
    if value < args.train_ratio:
        return "train"
    if value < args.train_ratio + args.val_ratio:
        return "val"
    return "test"


def measured_antenna_coordinates_wavelengths(
    antenna_positions_mm: np.ndarray,
    carrier_hz: float,
) -> torch.Tensor:
    wavelength_m = LIGHT_SPEED_MPS / carrier_hz
    positions_m = antenna_positions_mm.astype(np.float64) / 1000.0
    centered_m = positions_m - positions_m.mean(axis=0, keepdims=True)
    return torch.from_numpy((centered_m / wavelength_m).astype(np.float32))


def infer_los_status(data_root: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    lowered = str(data_root).lower()
    if "nlos" in lowered or "n_los" in lowered or "nonlos" in lowered:
        return "nlos"
    if "los" in lowered:
        return "los"
    raise ValueError(
        "Could not infer LoS/NLoS status from --data-root; pass --los-status los or nlos."
    )


def make_measured_sample(
    *,
    index: int,
    csi_path: Path,
    antenna_coordinates_wavelengths: torch.Tensor,
    semantic_key: SemanticKey,
    prop_caption: str,
    args: argparse.Namespace,
) -> PreprocessedSample:
    channel = np.load(csi_path)
    if channel.shape != (args.array_rows * args.array_cols, 100):
        raise ValueError(f"Unexpected CSI shape in {csi_path}: {channel.shape}")
    raw_csi = torch.as_tensor(channel)
    tokens, beam_positions, metadata = preprocess_sample(
        raw_csi=raw_csi,
        array_type="UPA",
        n_row=args.array_rows,
        n_col=args.array_cols,
        n_rx=1,
        target_nf=args.target_nf,
        patch_1d=args.patch_1d,
        patch_2d=tuple(args.patch_2d),
        rx_index=None,
    )
    subcarrier_spacing_hz = float(args.bandwidth_hz) / float(channel.shape[-1])
    bw_bin = derive_bw_bin(int(metadata["target_nf"]))
    return PreprocessedSample(
        tokens=tokens,
        beam_positions=beam_positions,
        config_key=f"UPA-{args.array_rows}x{args.array_cols}",
        n_tokens=int(metadata["n_tokens"]),
        freq_bin=int(args.freq_bin),
        bw_bin=int(bw_bin),
        subcarrier_spacing_hz=subcarrier_spacing_hz,
        group_id=f"ultra_dense_{semantic_key.los_status}:{index}",
        semantic_key=semantic_key,
        config_label=0,
        obs_array_label=1,
        obs_ant_label=1,
        obs_freq_label=int(args.freq_bin),
        obs_bw_label=int(bw_bin),
        prop_caption=prop_caption,
        instance_caption=(
            f"Measured indoor {semantic_key.los_status.upper()} CSI snapshot from the KU Leuven ultra-dense "
            "URA laboratory campaign."
        ),
        n_paths=0,
        delay_spread_s=0.0,
        azimuth_spread_deg=0.0,
        k_factor_db=0.0,
        first_path_delay_s=math.nan,
        los_delay_s=math.nan,
        los_aoa_az_deg=math.nan,
        first_path_power_dbw=math.nan,
        total_received_power_dbw=math.nan,
        first_path_aoa_az_deg=math.nan,
        first_path_aod_az_deg=math.nan,
        reflection_count=0,
        diffraction_count=0,
        reflection_path_count=0,
        diffraction_path_count=0,
        direct_path_count=0,
        delay_power_map=empty_delay_power_map(),
        delay_power_profile=empty_delay_power_profile(),
        array_type="UPA",
        array_rows=int(args.array_rows),
        array_cols=int(args.array_cols),
        antenna_spacing_wavelengths=float("nan"),
        source_n_freq=int(channel.shape[-1]),
        bandwidth_hz=float(args.bandwidth_hz),
        antenna_coordinates_wavelengths=antenna_coordinates_wavelengths.clone(),
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    los_status = infer_los_status(args.data_root, args.los_status)

    positions_mm = np.load(args.data_root / "user_positions.npy")
    antenna_positions_mm = np.load(args.data_root / "antenna_positions.npy")
    files = measurement_files(args.data_root / "samples")
    if not files:
        raise FileNotFoundError(f"No measured CSI files found under {args.data_root / 'samples'}")
    available_csi_count = len(files)

    files = files[:: args.sample_stride]
    if args.max_samples is not None:
        files = files[: args.max_samples]
    if not files:
        raise RuntimeError("No measured CSI files remain after stride/max-samples filtering")

    reference_m = (
        antenna_positions_mm.mean(axis=0).astype(np.float32) / 1000.0
        if args.reference == "array_center"
        else np.zeros(3, dtype=np.float32)
    )
    antenna_coordinates = measured_antenna_coordinates_wavelengths(
        antenna_positions_mm, args.carrier_hz
    )
    semantic_key = SemanticKey(
        env_type="indoor",
        los_status=los_status,
        path_richness="any",
        ds_bin="any",
        as_az_bin="any",
        k_factor_bin="any",
        first_delay_bin="any",
        first_power_bin="unknown",
        first_angle_bin="any",
        reflection_bin="any",
        diffraction_bin="any",
    )
    prop_caption = CaptionGenerator().generate(semantic_key)

    samples: list[PreprocessedSample] = []
    tx_xyz: list[np.ndarray] = []
    rx_xyz: list[np.ndarray] = []
    split_lists: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    original_indices: list[int] = []

    for output_index, (measurement_index, csi_path) in enumerate(files):
        if not 0 <= measurement_index < len(positions_mm):
            raise ValueError(f"Measurement index {measurement_index} has no position row")
        user_position_m = positions_mm[measurement_index].astype(np.float32) / 1000.0
        sample = make_measured_sample(
            index=measurement_index,
            csi_path=csi_path,
            antenna_coordinates_wavelengths=antenna_coordinates,
            semantic_key=semantic_key,
            prop_caption=prop_caption,
            args=args,
        )
        samples.append(sample)
        tx_xyz.append(reference_m.copy())
        rx_xyz.append(user_position_m)
        original_indices.append(int(measurement_index))
        split_lists[
            split_name(index=measurement_index, position_m=user_position_m, args=args)
        ].append(output_index)
        if output_index % 5000 == 0 and output_index > 0:
            print(f"processed={output_index}/{len(files)}")

    split_indices = {
        name: torch.tensor(indices, dtype=torch.long) for name, indices in split_lists.items()
    }
    if any(len(indices) == 0 for indices in split_indices.values()):
        raise RuntimeError(
            "Empty split produced: "
            + ", ".join(f"{name}:{len(indices)}" for name, indices in split_indices.items())
        )

    tx = torch.from_numpy(np.stack(tx_xyz)).float()
    rx = torch.from_numpy(np.stack(rx_xyz)).float()
    relative_xy = rx[:, :2] - tx[:, :2]
    payload = {
        "format": "csi2text_localization_v1",
        "samples": samples,
        "tx_xyz_m": tx,
        "rx_xyz_m": rx,
        "relative_xy_m": relative_xy,
        "split_indices": split_indices,
        "metadata": {
            "source": f"ku_leuven_ultra_dense_ura_lab_{los_status}",
            "los_status": los_status,
            "data_root": str(args.data_root),
            "seed": args.seed,
            "split_mode": args.split_mode,
            "tile_size_m": args.tile_size_m,
            "sample_stride": args.sample_stride,
            "max_samples": args.max_samples,
            "original_position_count": int(len(positions_mm)),
            "available_csi_count": int(available_csi_count),
            "selected_csi_count": int(len(samples)),
            "original_measurement_indices": original_indices,
            "array_shape": [args.array_rows, args.array_cols],
            "target_nf": args.target_nf,
            "source_n_freq": 100,
            "carrier_hz": args.carrier_hz,
            "bandwidth_hz": args.bandwidth_hz,
            "subcarrier_spacing_hz": args.bandwidth_hz / 100.0,
            "reference": args.reference,
            "reference_xyz_m": reference_m.tolist(),
            "coordinate_unit": "meter",
            "position_note": (
                "rx_xyz_m stores measured user positions; tx_xyz_m is a fixed "
                "reference point so relative_xy_m is user_xy minus reference_xy."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    print(f"saved={args.output.resolve()}")
    print(f"samples={len(samples)}")
    print("splits=" + ",".join(f"{name}:{len(indices)}" for name, indices in split_indices.items()))


if __name__ == "__main__":
    main()
