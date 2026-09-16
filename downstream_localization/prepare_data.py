from __future__ import annotations

import argparse
import gzip
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# These helpers are deliberately reused so the localization CSI is bit-for-bit
# consistent with the project's pretraining preprocessing.
from scripts.preprocess_all import (  # noqa: E402
    LoadedScenarioDataset,
    _interactions_as_reflection_code,
    _iter_d2los_propbin_files,
    _map_uniform_file_allocations,
    _pad_rows,
    _selected_rx_indices,
    _source_group_id,
    _steering_vector,
    _uniform_file_allocations,
    preprocess_deepmimo_dataset,
    read_rayverse_propbin,
)


DEFAULT_D2LOS_ROOT = Path(
    "/home/yrh/CSI_model/CSI_model/deepmimo_scenarios/D2Los_Data"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build D2Los UPA localization samples with TX/RX coordinates."
    )
    parser.add_argument("--d2los-root", type=Path, default=DEFAULT_D2LOS_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--max-maps", type=int)
    parser.add_argument("--max-sources-per-map", type=int)
    parser.add_argument("--max-rx-per-source", type=int)
    parser.add_argument(
        "--sampling",
        choices=("sequential", "uniform", "map_uniform"),
        default="map_uniform",
    )
    parser.add_argument("--seed", type=int, default=23421)
    parser.add_argument("--array-rows", type=int, default=8)
    parser.add_argument("--array-cols", type=int, default=8)
    parser.add_argument("--antenna-spacing", type=float, default=0.5)
    parser.add_argument("--bandwidth-hz", type=float, default=100e6)
    parser.add_argument("--subcarriers", type=int, default=128)
    parser.add_argument("--tx-power-dbm", type=float, default=23.0)
    parser.add_argument("--freq-bin", type=int, default=0)
    parser.add_argument("--env-type", choices=("indoor", "outdoor", "O2I"), default="outdoor")
    parser.add_argument(
        "--exclude-seen-pt",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional pretraining .pt file. Links present in it are excluded so the "
            "localization set is unseen by CSI2TEXT-PGMT. May be repeated."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.array_rows <= 0 or args.array_cols <= 0:
        raise ValueError("array dimensions must be positive")
    if args.bandwidth_hz <= 0 or args.subcarriers <= 0:
        raise ValueError("bandwidth and subcarrier count must be positive")
    if args.max_rx_per_source is not None and args.max_rx_per_source <= 0:
        raise ValueError("--max-rx-per-source must be positive")


def read_source_xyz(path: Path) -> np.ndarray:
    """Read the source position stored as three little-endian float32 values."""
    opener = gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
    with opener as stream:
        header = stream.read(20)
    if len(header) < 20:
        raise ValueError(f"Incomplete propbin header: {path}")
    xyz = np.frombuffer(header, dtype="<f4", count=3, offset=8).astype(np.float32)
    if not np.isfinite(xyz).all():
        raise ValueError(f"Non-finite source coordinate in {path}: {xyz.tolist()}")
    return xyz


def load_seen_links(paths: list[Path]) -> set[str]:
    seen: set[str] = set()
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if isinstance(payload, dict) and "samples" in payload:
            samples = payload["samples"]
        elif isinstance(payload, list):
            samples = payload
        else:
            raise ValueError(f"Unsupported sample payload in {path}")
        seen.update(str(sample.group_id) for sample in samples)
    return seen


def split_name(group_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "little") / float(2**64)
    if value < 0.8:
        return "train"
    if value < 0.9:
        return "val"
    return "test"


def synthesize_channel(
    power_dbw: np.ndarray,
    phase_deg: np.ndarray,
    delay_s: np.ndarray,
    aod_az_deg: np.ndarray,
    aod_el_deg: np.ndarray,
    tx_shape: tuple[int, int],
    tx_spacing: float,
    bandwidth_hz: float,
    subcarriers: int,
) -> np.ndarray:
    """Synthesize the same single-RX channel used by the main preprocessor."""
    channel = np.zeros((1, tx_shape[0] * tx_shape[1], subcarriers), dtype=np.complex64)
    centered = np.arange(subcarriers, dtype=np.float32)
    centered -= centered.mean()
    frequencies = centered * (bandwidth_hz / subcarriers)
    valid = np.isfinite(power_dbw) & np.isfinite(delay_s)
    for path_idx in np.flatnonzero(valid):
        gain = np.sqrt(10.0 ** (power_dbw[path_idx] / 10.0)) * np.exp(
            1j * np.deg2rad(phase_deg[path_idx])
        )
        tx_sv = _steering_vector(
            tx_shape[0], tx_shape[1], tx_spacing, aod_az_deg[path_idx], aod_el_deg[path_idx]
        )
        delay_phase = np.exp(-1j * 2.0 * np.pi * frequencies * delay_s[path_idx])
        channel[0] += gain * np.conj(tx_sv)[:, None] * delay_phase[None, :]
    return channel


def source_batch(
    root: Path,
    propbin_path: Path,
    propbin,
    rx_indices: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list, list[np.ndarray], list[np.ndarray]]:
    source_xyz = read_source_xyz(propbin_path)
    carrier_hz = propbin.carrier_frequency_ghz * 1e9
    wavelength_m = 299_792_458.0 / max(carrier_hz, 1.0)

    channels: list[np.ndarray] = []
    delays: list[np.ndarray] = []
    aoa_azs: list[np.ndarray] = []
    aoa_els: list[np.ndarray] = []
    aod_azs: list[np.ndarray] = []
    aod_els: list[np.ndarray] = []
    powers: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    interactions: list[np.ndarray] = []
    los: list[int] = []
    path_counts: list[int] = []
    group_ids: list[str] = []
    tx_xyz: list[np.ndarray] = []
    rx_xyz: list[np.ndarray] = []

    for rx_idx in rx_indices:
        rx_record = propbin.rx_records[int(rx_idx)]
        count = int(rx_record["path_count"])
        offset = int(rx_record["path_offset"])
        paths = propbin.path_records[offset : offset + count]
        path_loss_db = paths["path_loss_db"].astype(np.float32)
        distance_m = paths["distance_m"].astype(np.float32)
        delay_s = paths["delay_ns"].astype(np.float32) * 1e-9
        power_dbw = args.tx_power_dbm - path_loss_db - 30.0
        phase_deg = np.rad2deg(-2.0 * np.pi * distance_m / wavelength_m).astype(np.float32)
        aoa_az = np.rad2deg(paths["aoa_az_rad"].astype(np.float32))
        aoa_el = np.rad2deg(paths["aoa_el_rad"].astype(np.float32))
        aod_az = np.rad2deg(paths["aod_az_rad"].astype(np.float32))
        aod_el = np.rad2deg(paths["aod_el_rad"].astype(np.float32))
        interaction_count = paths["interaction_count"].astype(np.int32)

        channels.append(
            synthesize_channel(
                power_dbw,
                phase_deg,
                delay_s,
                aod_az,
                aod_el,
                (args.array_rows, args.array_cols),
                args.antenna_spacing,
                args.bandwidth_hz,
                args.subcarriers,
            )
        )
        delays.append(delay_s)
        aoa_azs.append(aoa_az)
        aoa_els.append(aoa_el)
        aod_azs.append(aod_az)
        aod_els.append(aod_el)
        powers.append(power_dbw)
        phases.append(phase_deg)
        interactions.append(_interactions_as_reflection_code(interaction_count))
        los.append(int(np.any(interaction_count == 0)))
        path_counts.append(count)
        group_ids.append(_source_group_id(root, propbin_path, int(rx_idx)))
        tx_xyz.append(source_xyz.copy())
        rx_xyz.append(
            np.asarray([rx_record["x"], rx_record["y"], rx_record["z"]], dtype=np.float32)
        )

    dataset = LoadedScenarioDataset(
        channel=np.stack(channels),
        los=np.asarray(los, dtype=np.int64),
        num_paths=np.asarray(path_counts, dtype=np.int64),
        delay=_pad_rows(delays),
        aoa_az=_pad_rows(aoa_azs),
        aoa_el=_pad_rows(aoa_els),
        aod_az=_pad_rows(aod_azs),
        aod_el=_pad_rows(aod_els),
        power=_pad_rows(powers),
        phase=_pad_rows(phases),
        inter=_pad_rows(interactions, pad_value=-1),
        ch_params={
            "bs_antenna": {
                "shape": [args.array_rows, args.array_cols],
                "spacing": args.antenna_spacing,
            },
            "ue_antenna": {"shape": [1, 1], "spacing": args.antenna_spacing},
            "ofdm": {"bandwidth": args.bandwidth_hz, "subcarriers": args.subcarriers},
            "selected_subcarriers": np.arange(args.subcarriers),
        },
        group_ids=group_ids,
    )
    samples = preprocess_deepmimo_dataset(
        dataset=dataset,
        scenario=root.name,
        freq_bin=args.freq_bin,
        rx_index=0,
        env_type=args.env_type,
        max_samples=None,
        patch_1d=4,
        patch_2d=(2, 2),
        target_nf=args.subcarriers,
        include_empty_samples=False,
    )
    if len(samples) != len(tx_xyz):
        raise RuntimeError("Coordinate/sample count changed unexpectedly during preprocessing")
    return samples, tx_xyz, rx_xyz


def main() -> None:
    args = parse_args()
    validate_args(args)
    seen_links = load_seen_links(args.exclude_seen_pt)
    files = _iter_d2los_propbin_files(
        args.d2los_root, args.max_maps, args.max_sources_per_map
    )
    if not files:
        raise FileNotFoundError(f"No propbin files under {args.d2los_root}")

    if args.sampling == "map_uniform":
        allocations = _map_uniform_file_allocations(
            files, args.max_samples + len(seen_links), args.max_rx_per_source, args.seed
        )
    elif args.sampling == "uniform":
        allocations = _uniform_file_allocations(
            files, args.max_samples + len(seen_links), args.max_rx_per_source, args.seed
        )
    else:
        allocations = {}

    rng = np.random.default_rng(args.seed)
    all_samples: list = []
    all_tx: list[np.ndarray] = []
    all_rx: list[np.ndarray] = []
    for file_idx, path in enumerate(files, start=1):
        if len(all_samples) >= args.max_samples:
            break
        propbin = read_rayverse_propbin(path)
        remaining = args.max_samples - len(all_samples)
        requested = allocations.get(path) if allocations else remaining
        if requested is None or requested <= 0:
            continue
        indices = _selected_rx_indices(
            propbin.rx_records,
            max_rx_per_source=min(requested, args.max_rx_per_source or requested),
            remaining_samples=requested,
            sampling="uniform" if args.sampling != "sequential" else "sequential",
            rng=rng,
        )
        if seen_links:
            indices = np.asarray(
                [
                    idx
                    for idx in indices
                    if _source_group_id(args.d2los_root, path, int(idx)) not in seen_links
                ],
                dtype=np.int64,
            )
        if indices.size == 0:
            continue
        indices = indices[:remaining]
        samples, tx_xyz, rx_xyz = source_batch(
            args.d2los_root, path, propbin, indices, args
        )
        all_samples.extend(samples)
        all_tx.extend(tx_xyz)
        all_rx.extend(rx_xyz)
        if file_idx % 100 == 0 or len(all_samples) >= args.max_samples:
            print(f"files={file_idx}/{len(files)} samples={len(all_samples)}/{args.max_samples}")

    if not all_samples:
        raise RuntimeError("No localization samples were generated")
    tx = torch.from_numpy(np.stack(all_tx)).float()
    rx = torch.from_numpy(np.stack(all_rx)).float()
    relative_xy = rx[:, :2] - tx[:, :2]
    splits = [split_name(sample.group_id, args.seed) for sample in all_samples]
    split_indices = {
        name: torch.tensor([i for i, value in enumerate(splits) if value == name], dtype=torch.long)
        for name in ("train", "val", "test")
    }
    if any(len(indices) == 0 for indices in split_indices.values()):
        raise RuntimeError(f"Empty split produced: { {k: len(v) for k, v in split_indices.items()} }")

    payload = {
        "format": "csi2text_localization_v1",
        "samples": all_samples,
        "tx_xyz_m": tx,
        "rx_xyz_m": rx,
        "relative_xy_m": relative_xy,
        "split_indices": split_indices,
        "metadata": {
            "d2los_root": str(args.d2los_root),
            "seed": args.seed,
            "sampling": args.sampling,
            "array_shape": [args.array_rows, args.array_cols],
            "subcarriers": args.subcarriers,
            "bandwidth_hz": args.bandwidth_hz,
            "excluded_seen_links": len(seen_links),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    print(f"saved={args.output.resolve()}")
    print(f"samples={len(all_samples)}")
    print("splits=" + ",".join(f"{k}:{len(v)}" for k, v in split_indices.items()))


if __name__ == "__main__":
    main()
