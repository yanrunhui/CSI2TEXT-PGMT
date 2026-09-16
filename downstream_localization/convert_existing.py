from __future__ import annotations

import argparse
import gzip
import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preprocess_all import (  # noqa: E402
    PROP_HEADER_BYTES,
    PROP_MAGIC,
    PROP_RX_DTYPE,
    PROP_RX_RECORD_BYTES,
)

from downstream_localization.prepare_data import (  # noqa: E402
    DEFAULT_D2LOS_ROOT,
)


GROUP_ID_PATTERN = re.compile(
    r"^.+-map_(?P<map>\d+)-source_(?P<source>\d+)-rx_(?P<rx>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach D2Los TX/RX coordinates to existing preprocessed CSI splits. "
            "CSI is reused without resynthesis or tokenization."
        )
    )
    parser.add_argument("--train-pt", type=Path, required=True)
    parser.add_argument(
        "--val-pt",
        type=Path,
        help=(
            "Existing validation split. If omitted, --test-pt is deterministically "
            "split in half into validation and final test sets."
        ),
    )
    parser.add_argument("--test-pt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=23421)
    parser.add_argument("--d2los-root", type=Path, default=DEFAULT_D2LOS_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def samples_from_pt(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(payload, list):
        samples = payload
    elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        samples = payload["samples"]
    else:
        raise ValueError(f"Expected a sample list in {path}")
    if not samples:
        raise ValueError(f"Empty sample file: {path}")
    return samples


def split_heldout_samples(samples: list, seed: int) -> tuple[list, list]:
    if len(samples) < 2:
        raise ValueError("At least two held-out samples are needed to create val/test splits")
    ranked_indices = sorted(
        range(len(samples)),
        key=lambda index: hashlib.sha256(
            f"{seed}:{samples[index].group_id}".encode("utf-8")
        ).digest(),
    )
    val_indices = set(ranked_indices[: len(samples) // 2])
    validation = [sample for index, sample in enumerate(samples) if index in val_indices]
    test = [sample for index, sample in enumerate(samples) if index not in val_indices]
    return validation, test


def parse_group_id(group_id: str) -> tuple[int, int, int]:
    match = GROUP_ID_PATTERN.fullmatch(str(group_id))
    if match is None:
        raise ValueError(
            f"Cannot recover D2Los coordinates from group_id={group_id!r}; "
            "expected ...-map_<id>-source_<id>-rx_<id>"
        )
    return tuple(int(match.group(name)) for name in ("map", "source", "rx"))


def find_propbin(root: Path, map_id: int, source_id: int) -> Path:
    matches = sorted(
        (root / f"map_{map_id}").glob(
            f"special_points_propbin_*/source_{source_id}.propbin*"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one propbin for map={map_id}, source={source_id}; found {matches}"
        )
    return matches[0]


def read_coordinate_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read only the header and RX table, skipping all path/interaction payloads."""
    opener = gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
    with opener as stream:
        header = stream.read(PROP_HEADER_BYTES)
        if len(header) != PROP_HEADER_BYTES or header[:4] not in PROP_MAGIC:
            raise ValueError(f"Invalid propbin header: {path}")
        source_xyz = np.frombuffer(header, dtype="<f4", count=3, offset=8).copy()
        rx_count = int(np.frombuffer(header, dtype="<u4", count=1, offset=20)[0])
        table_bytes = stream.read(rx_count * PROP_RX_RECORD_BYTES)
    expected = rx_count * PROP_RX_RECORD_BYTES
    if len(table_bytes) != expected:
        raise ValueError(
            f"Incomplete RX coordinate table in {path}: got {len(table_bytes)}, expected {expected}"
        )
    records = np.frombuffer(table_bytes, dtype=PROP_RX_DTYPE, count=rx_count).copy()
    return source_xyz.astype(np.float32), records


def recover_coordinates(root: Path, samples: list) -> tuple[torch.Tensor, torch.Tensor]:
    parsed = [parse_group_id(sample.group_id) for sample in samples]
    links_by_source: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for sample_index, (map_id, source_id, rx_id) in enumerate(parsed):
        links_by_source.setdefault((map_id, source_id), []).append((sample_index, rx_id))

    tx_xyz = np.empty((len(samples), 3), dtype=np.float32)
    rx_xyz = np.empty((len(samples), 3), dtype=np.float32)
    for source_number, ((map_id, source_id), links) in enumerate(
        sorted(links_by_source.items()), start=1
    ):
        path = find_propbin(root, map_id, source_id)
        source_xyz, rx_records = read_coordinate_table(path)
        for sample_index, rx_id in links:
            if not 0 <= rx_id < len(rx_records):
                raise IndexError(
                    f"rx_{rx_id} is outside {path} with {len(rx_records)} records"
                )
            record = rx_records[rx_id]
            tx_xyz[sample_index] = source_xyz
            rx_xyz[sample_index] = [record["x"], record["y"], record["z"]]
        if source_number % 100 == 0 or source_number == len(links_by_source):
            print(f"coordinate sources={source_number}/{len(links_by_source)}")
    return torch.from_numpy(tx_xyz), torch.from_numpy(rx_xyz)


def validate_no_split_leakage(samples: list, ranges: dict[str, range]) -> None:
    groups = {
        name: {str(samples[index].group_id) for index in indices}
        for name, indices in ranges.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups[left] & groups[right]
        if overlap:
            raise ValueError(
                f"Existing splits leak {len(overlap)} map/source/rx links between {left} and {right}"
            )


def main() -> None:
    args = parse_args()
    train_samples = samples_from_pt(args.train_pt)
    test_file_samples = samples_from_pt(args.test_pt)
    if args.val_pt is None:
        val_samples, test_samples = split_heldout_samples(test_file_samples, args.seed)
        split_policy = "existing_train_plus_deterministic_half_split_of_test"
    else:
        val_samples = samples_from_pt(args.val_pt)
        test_samples = test_file_samples
        split_policy = "reused_existing_files"
    split_samples = {"train": train_samples, "val": val_samples, "test": test_samples}
    samples: list[Any] = []
    ranges: dict[str, range] = {}
    for name in ("train", "val", "test"):
        start = len(samples)
        samples.extend(split_samples[name])
        ranges[name] = range(start, len(samples))
    validate_no_split_leakage(samples, ranges)

    token_shapes = {tuple(sample.tokens.shape[1:]) for sample in samples}
    if len(token_shapes) != 1:
        raise ValueError(f"All splits must share [d_token, n_freq], found {sorted(token_shapes)}")
    tx_xyz, rx_xyz = recover_coordinates(args.d2los_root, samples)
    relative_xy = rx_xyz[:, :2] - tx_xyz[:, :2]
    split_indices = {
        name: torch.arange(indices.start, indices.stop, dtype=torch.long)
        for name, indices in ranges.items()
    }
    payload = {
        "format": "csi2text_localization_v1",
        "samples": samples,
        "tx_xyz_m": tx_xyz,
        "rx_xyz_m": rx_xyz,
        "relative_xy_m": relative_xy,
        "split_indices": split_indices,
        "metadata": {
            "d2los_root": str(args.d2los_root),
            "source_files": {
                "train": str(args.train_pt),
                "val": str(args.val_pt) if args.val_pt is not None else None,
                "test": str(args.test_pt),
            },
            "split_policy": split_policy,
            "split_seed": args.seed,
            "csi_reprocessed": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    print(f"saved={args.output.resolve()}")
    print(f"token_shape={next(iter(token_shapes))}")
    print("splits=" + ",".join(f"{name}:{len(value)}" for name, value in split_samples.items()))
    print("csi_reprocessed=False")


if __name__ == "__main__":
    main()
