from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.caption import CaptionGenerator  # noqa: E402
from data.semantic_key import SemanticKey  # noqa: E402
from downstream_localization.prepare_ultra_dense import (  # noqa: E402
    infer_los_status,
    make_measured_sample,
    measured_antenna_coordinates_wavelengths,
    measurement_files,
)


@dataclass(frozen=True)
class Candidate:
    index: int
    path: Path
    position_m: np.ndarray
    tile_key: tuple[int, int]
    cell_key: tuple[int, int]


@dataclass
class ScenarioCatalog:
    status: str
    root: Path
    positions_mm: np.ndarray
    antenna_coordinates_wavelengths: torch.Tensor
    reference_m: np.ndarray
    by_tile: dict[tuple[int, int], list[Candidate]]
    available_csi_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one balanced LoS/NLoS ultra-dense localization corpus with fixed "
            "train/val/test counts and spatially disjoint splits."
        )
    )
    parser.add_argument("--los-data-root", type=Path, required=True)
    parser.add_argument("--nlos-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=20_000)
    parser.add_argument("--val-per-class", type=int, default=2_500)
    parser.add_argument("--test-per-class", type=int, default=2_500)
    parser.add_argument("--seed", type=int, default=23421)
    parser.add_argument(
        "--split-tile-size-m",
        type=float,
        default=0.25,
        help="Train/val/test use disjoint coarse spatial tiles of this size.",
    )
    parser.add_argument(
        "--sample-cell-size-m",
        type=float,
        default=0.01,
        help="Within each class, keep at most one candidate per fine spatial cell.",
    )
    parser.add_argument(
        "--allow-cross-class-same-cell",
        action="store_true",
        help="Allow LoS and NLoS samples in the same split to use the same fine cell.",
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
        "--reference",
        choices=("array_center", "origin"),
        default="array_center",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("train_per_class", "val_per_class", "test_per_class"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.split_tile_size_m <= 0.0 or args.sample_cell_size_m <= 0.0:
        raise ValueError("tile/cell sizes must be positive")
    if args.sample_cell_size_m > args.split_tile_size_m:
        raise ValueError("--sample-cell-size-m must be <= --split-tile-size-m")


def load_catalog(
    *,
    root: Path,
    status: str,
    args: argparse.Namespace,
    rng: random.Random,
) -> ScenarioCatalog:
    positions_mm = np.load(root / "user_positions.npy")
    antenna_positions_mm = np.load(root / "antenna_positions.npy")
    antenna_coordinates = measured_antenna_coordinates_wavelengths(
        antenna_positions_mm, args.carrier_hz
    )
    reference_m = (
        antenna_positions_mm.mean(axis=0).astype(np.float32) / 1000.0
        if args.reference == "array_center"
        else np.zeros(3, dtype=np.float32)
    )

    files = measurement_files(root / "samples")
    by_cell: dict[tuple[tuple[int, int], tuple[int, int]], list[Candidate]] = {}
    for index, path in files:
        if not 0 <= index < len(positions_mm):
            continue
        position_m = positions_mm[index].astype(np.float32) / 1000.0
        tile_key = tuple(
            np.floor(position_m[:2] / args.split_tile_size_m).astype(np.int64).tolist()
        )
        cell_key = tuple(
            np.floor(position_m[:2] / args.sample_cell_size_m).astype(np.int64).tolist()
        )
        candidate = Candidate(
            index=int(index),
            path=path,
            position_m=position_m,
            tile_key=(int(tile_key[0]), int(tile_key[1])),
            cell_key=(int(cell_key[0]), int(cell_key[1])),
        )
        by_cell.setdefault((candidate.tile_key, candidate.cell_key), []).append(candidate)

    by_tile: dict[tuple[int, int], list[Candidate]] = {}
    for (tile_key, _cell_key), candidates in by_cell.items():
        chosen = rng.choice(candidates)
        by_tile.setdefault(tile_key, []).append(chosen)
    for candidates in by_tile.values():
        rng.shuffle(candidates)

    return ScenarioCatalog(
        status=status,
        root=root,
        positions_mm=positions_mm,
        antenna_coordinates_wavelengths=antenna_coordinates,
        reference_m=reference_m,
        by_tile=by_tile,
        available_csi_count=len(files),
    )


def allocate_tiles(
    *,
    los_catalog: ScenarioCatalog,
    nlos_catalog: ScenarioCatalog,
    quotas: dict[str, int],
    rng: random.Random,
    allow_cross_class_same_cell: bool,
) -> dict[str, list[tuple[int, int]]]:
    common_tiles = sorted(set(los_catalog.by_tile) & set(nlos_catalog.by_tile))
    rng.shuffle(common_tiles)
    allocations: dict[str, list[tuple[int, int]]] = {name: [] for name in quotas}
    used: set[tuple[int, int]] = set()

    for split_name, quota in quotas.items():
        capacity = 0
        for tile in common_tiles:
            if tile in used:
                continue
            tile_capacity = min(
                len(los_catalog.by_tile[tile]),
                len(nlos_catalog.by_tile[tile]),
            )
            if not allow_cross_class_same_cell:
                tile_capacity //= 2
            if tile_capacity <= 0:
                continue
            allocations[split_name].append(tile)
            used.add(tile)
            capacity += tile_capacity
            if capacity >= quota:
                break
        if capacity < quota:
            raise RuntimeError(
                f"Not enough spatially separated {split_name} candidates: "
                f"need {quota} per class, have {capacity}. Reduce --sample-cell-size-m "
                "or requested sample counts."
            )
    return allocations


def select_candidates(
    catalog: ScenarioCatalog,
    tiles: list[tuple[int, int]],
    count: int,
    rng: random.Random,
    forbidden_cells: set[tuple[int, int]] | None = None,
) -> list[Candidate]:
    forbidden_cells = forbidden_cells or set()
    selected: list[Candidate] = []
    shuffled_tiles = list(tiles)
    rng.shuffle(shuffled_tiles)
    for tile in shuffled_tiles:
        candidates = list(catalog.by_tile[tile])
        rng.shuffle(candidates)
        candidates = [
            candidate
            for candidate in candidates
            if candidate.cell_key not in forbidden_cells
        ]
        remaining = count - len(selected)
        selected.extend(candidates[:remaining])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"Could only select {len(selected)} {catalog.status} samples; requested {count}"
        )
    return selected


def semantic_key(status: str) -> SemanticKey:
    return SemanticKey(
        env_type="indoor",
        los_status=status,
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


def append_split(
    *,
    split_name: str,
    split_candidates: list[tuple[ScenarioCatalog, Candidate]],
    samples: list,
    tx_xyz: list[np.ndarray],
    rx_xyz: list[np.ndarray],
    split_indices: dict[str, list[int]],
    original_indices: dict[str, list[int]],
    args: argparse.Namespace,
) -> None:
    caption_generator = CaptionGenerator()
    for catalog, candidate in split_candidates:
        key = semantic_key(catalog.status)
        sample = make_measured_sample(
            index=candidate.index,
            csi_path=candidate.path,
            antenna_coordinates_wavelengths=catalog.antenna_coordinates_wavelengths,
            semantic_key=key,
            prop_caption=caption_generator.generate(key),
            args=args,
        )
        output_index = len(samples)
        samples.append(sample)
        tx_xyz.append(catalog.reference_m.copy())
        rx_xyz.append(candidate.position_m)
        split_indices[split_name].append(output_index)
        original_indices[catalog.status].append(candidate.index)
        if output_index % 5000 == 0 and output_index > 0:
            print(f"processed={output_index}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    rng = random.Random(args.seed)

    los_catalog = load_catalog(
        root=args.los_data_root,
        status=infer_los_status(args.los_data_root, "los"),
        args=args,
        rng=rng,
    )
    nlos_catalog = load_catalog(
        root=args.nlos_data_root,
        status=infer_los_status(args.nlos_data_root, "nlos"),
        args=args,
        rng=rng,
    )
    quotas = {
        "train": int(args.train_per_class),
        "val": int(args.val_per_class),
        "test": int(args.test_per_class),
    }
    tile_allocations = allocate_tiles(
        los_catalog=los_catalog,
        nlos_catalog=nlos_catalog,
        quotas=quotas,
        rng=rng,
        allow_cross_class_same_cell=args.allow_cross_class_same_cell,
    )

    samples: list = []
    tx_xyz: list[np.ndarray] = []
    rx_xyz: list[np.ndarray] = []
    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    original_indices: dict[str, list[int]] = {"los": [], "nlos": []}

    for split_name in ("train", "val", "test"):
        count = quotas[split_name]
        tiles = tile_allocations[split_name]
        los_selected = select_candidates(los_catalog, tiles, count, rng)
        forbidden_cells = (
            set()
            if args.allow_cross_class_same_cell
            else {candidate.cell_key for candidate in los_selected}
        )
        nlos_selected = select_candidates(
            nlos_catalog,
            tiles,
            count,
            rng,
            forbidden_cells=forbidden_cells,
        )
        paired = [(los_catalog, item) for item in los_selected]
        paired.extend((nlos_catalog, item) for item in nlos_selected)
        rng.shuffle(paired)
        append_split(
            split_name=split_name,
            split_candidates=paired,
            samples=samples,
            tx_xyz=tx_xyz,
            rx_xyz=rx_xyz,
            split_indices=split_indices,
            original_indices=original_indices,
            args=args,
        )

    tx = torch.from_numpy(np.stack(tx_xyz)).float()
    rx = torch.from_numpy(np.stack(rx_xyz)).float()
    payload = {
        "format": "csi2text_localization_v1",
        "samples": samples,
        "tx_xyz_m": tx,
        "rx_xyz_m": rx,
        "relative_xy_m": rx[:, :2] - tx[:, :2],
        "split_indices": {
            name: torch.tensor(indices, dtype=torch.long)
            for name, indices in split_indices.items()
        },
        "metadata": {
            "source": "ku_leuven_ultra_dense_ura_lab_balanced_los_nlos",
            "los_data_root": str(args.los_data_root),
            "nlos_data_root": str(args.nlos_data_root),
            "seed": args.seed,
            "train_per_class": args.train_per_class,
            "val_per_class": args.val_per_class,
            "test_per_class": args.test_per_class,
            "split_tile_size_m": args.split_tile_size_m,
            "sample_cell_size_m": args.sample_cell_size_m,
            "allow_cross_class_same_cell": args.allow_cross_class_same_cell,
            "available_csi_count": {
                "los": int(los_catalog.available_csi_count),
                "nlos": int(nlos_catalog.available_csi_count),
            },
            "selected_count": {
                "train": len(split_indices["train"]),
                "val": len(split_indices["val"]),
                "test": len(split_indices["test"]),
                "los_total": len(original_indices["los"]),
                "nlos_total": len(original_indices["nlos"]),
            },
            "split_tile_count": {
                name: len(tiles) for name, tiles in tile_allocations.items()
            },
            "original_measurement_indices": original_indices,
            "array_shape": [args.array_rows, args.array_cols],
            "target_nf": args.target_nf,
            "source_n_freq": 100,
            "carrier_hz": args.carrier_hz,
            "bandwidth_hz": args.bandwidth_hz,
            "subcarrier_spacing_hz": args.bandwidth_hz / 100.0,
            "reference": args.reference,
            "coordinate_unit": "meter",
            "position_note": (
                "LoS and NLoS are balanced per split. Train/val/test use disjoint "
                "coarse spatial tiles; each class keeps at most one sample per fine cell."
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
    print(
        "class_counts="
        f"los:{len(original_indices['los'])},nlos:{len(original_indices['nlos'])}"
    )


if __name__ == "__main__":
    main()
