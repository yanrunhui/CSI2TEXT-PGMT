from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import _ensure_continuous_fields  # noqa: E402
from downstream_beam_selection.beam_utils import (  # noqa: E402
    FFT_ORDER,
    NUM_BEAMS,
    beam_label_record,
    validate_beam_sample,
    verify_beam_unpack_roundtrip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 64-beam oracle labels and balanced train/val/test splits."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input", type=Path, help="Single candidate pool (legacy mode).")
    source_group.add_argument(
        "--train-input", type=Path,
        help="Existing pretraining train split; use with --holdout-input.",
    )
    parser.add_argument(
        "--holdout-input", type=Path,
        help="Existing disjoint holdout split; divided into validation and test.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selected-samples-output",
        type=Path,
        help="Optional 50k materialized sample file; defaults beside --output.",
    )
    parser.add_argument("--train-size", type=int, default=40_000)
    parser.add_argument("--val-size", type=int, default=5_000)
    parser.add_argument("--test-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=23_421)
    parser.add_argument("--acceptable-loss-db", type=float, default=1.0)
    parser.add_argument("--verify-examples", type=int, default=8)
    return parser.parse_args()


def load_samples(path: Path) -> list[Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty sample list in {path}")
    return payload


def status(sample: Any) -> str:
    value = str(getattr(getattr(sample, "semantic_key", None), "los_status", "")).lower()
    if value not in {"los", "nlos"}:
        raise ValueError(f"Unsupported LoS status {value!r}")
    return value


def select_balanced_indices(
    samples: list[Any], split_sizes: dict[str, int], seed: int
) -> dict[str, list[int]]:
    if any(size <= 0 or size % 2 for size in split_sizes.values()):
        raise ValueError("Every split size must be a positive even number for exact LoS/NLoS balance")
    by_status: dict[str, list[int]] = {"los": [], "nlos": []}
    seen_groups: set[str] = set()
    for index, sample in enumerate(samples):
        group_id = str(getattr(sample, "group_id", "")).strip()
        if not group_id:
            raise ValueError(f"Sample {index} has no group_id")
        if group_id in seen_groups:
            continue
        seen_groups.add(group_id)
        by_status[status(sample)].append(index)

    required_per_status = sum(split_sizes.values()) // 2
    for offset, name in enumerate(("los", "nlos")):
        random.Random(seed + offset).shuffle(by_status[name])
        if len(by_status[name]) < required_per_status:
            raise ValueError(
                f"Need {required_per_status} {name} samples, found {len(by_status[name])}"
            )

    result: dict[str, list[int]] = {}
    offsets = {"los": 0, "nlos": 0}
    for split_offset, (split_name, split_size) in enumerate(split_sizes.items()):
        per_status = split_size // 2
        chosen: list[int] = []
        for name in ("los", "nlos"):
            start = offsets[name]
            end = start + per_status
            chosen.extend(by_status[name][start:end])
            offsets[name] = end
        random.Random(seed + 100 + split_offset).shuffle(chosen)
        result[split_name] = chosen
    return result


def digest(values: list[str]) -> str:
    hasher = hashlib.sha256()
    for value in values:
        hasher.update(value.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def unique_group_ids(samples: list[Any], path: Path) -> set[str]:
    groups: set[str] = set()
    for index, sample in enumerate(samples):
        group_id = str(getattr(sample, "group_id", "")).strip()
        if not group_id:
            raise ValueError(f"Missing group_id at index {index} in {path}")
        if group_id in groups:
            raise ValueError(f"Duplicate physical link {group_id!r} in {path}")
        groups.add(group_id)
    return groups


def load_split_sources(args: argparse.Namespace) -> tuple[list[Any], dict[str, list[int]], dict[str, Any]]:
    split_sizes = {"train": args.train_size, "val": args.val_size, "test": args.test_size}
    if args.input is not None:
        if args.holdout_input is not None:
            raise ValueError("--holdout-input requires --train-input, not --input")
        samples = load_samples(args.input)
        unique_group_ids(samples, args.input)
        return samples, select_balanced_indices(samples, split_sizes, args.seed), {
            "split_source_mode": "single_pool",
            "original_source_file": str(args.input.resolve()),
        }

    if args.holdout_input is None:
        raise ValueError("--train-input requires --holdout-input")
    train_samples = load_samples(args.train_input)
    holdout_samples = load_samples(args.holdout_input)
    train_groups = unique_group_ids(train_samples, args.train_input)
    holdout_groups = unique_group_ids(holdout_samples, args.holdout_input)
    overlap = train_groups & holdout_groups
    if overlap:
        raise ValueError(f"Train/holdout physical-link overlap: {sorted(overlap)[:5]}")
    train_local = select_balanced_indices(
        train_samples, {"train": args.train_size}, args.seed
    )["train"]
    holdout_local = select_balanced_indices(
        holdout_samples,
        {"val": args.val_size, "test": args.test_size},
        args.seed + 1,
    )
    offset = len(train_samples)
    split_indices = {
        "train": train_local,
        "val": [offset + index for index in holdout_local["val"]],
        "test": [offset + index for index in holdout_local["test"]],
    }
    return train_samples + holdout_samples, split_indices, {
        "split_source_mode": "pretraining_train_plus_holdout",
        "pretraining_train_source": str(args.train_input.resolve()),
        "holdout_source": str(args.holdout_input.resolve()),
    }


def main() -> None:
    args = parse_args()
    samples, split_indices_in_source, source_metadata = load_split_sources(args)
    source_indices = [
        index
        for split_name in ("train", "val", "test")
        for index in split_indices_in_source[split_name]
    ]
    local_by_source = {source_index: local for local, source_index in enumerate(source_indices)}
    split_indices = {
        name: torch.tensor([local_by_source[index] for index in indices], dtype=torch.long)
        for name, indices in split_indices_in_source.items()
    }

    gains = []
    gains_linear = []
    oracle = []
    top5 = []
    oracle_gain = []
    acceptable = []
    group_ids = []
    los_status = []
    selected_samples = []
    for selected_index, source_index in enumerate(source_indices):
        sample = _ensure_continuous_fields(samples[source_index])
        validate_beam_sample(sample)
        if selected_index < args.verify_examples:
            verify_beam_unpack_roundtrip(sample.tokens)
        record = beam_label_record(sample.tokens, args.acceptable_loss_db)
        gains_linear.append(record["beam_gains"])
        gains.append(record["beam_gains_db"])
        oracle.append(record["oracle_beam"])
        top5.append(record["oracle_top5"])
        oracle_gain.append(record["oracle_gain_db"])
        acceptable.append(record["acceptable_beams_1db"])
        group_ids.append(str(sample.group_id))
        los_status.append(status(sample))
        selected_samples.append(sample)
        if selected_index < args.verify_examples:
            print(
                f"verified_group={sample.group_id} oracle_beam={record['oracle_beam']} "
                f"beam_row_col={divmod(record['oracle_beam'], 8)}"
            )

    selected_samples_output = args.selected_samples_output or args.output.with_name(
        f"{args.output.stem}_samples.pt"
    )
    selected_samples_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(selected_samples, selected_samples_output)

    payload = {
        "format": "csi_beam_selection_v1",
        "source_file": str(selected_samples_output.resolve()),
        "source_indices": torch.arange(len(selected_samples), dtype=torch.long),
        "group_ids": group_ids,
        "los_status": los_status,
        "beam_gains_db": torch.stack(gains),
        "beam_gains": torch.stack(gains_linear),
        "oracle_beam": torch.tensor(oracle, dtype=torch.long),
        "oracle_top5": torch.stack(top5),
        "oracle_gain_db": torch.tensor(oracle_gain, dtype=torch.float32),
        "acceptable_beams_1db": torch.stack(acceptable),
        "split_indices": split_indices,
        "metadata": {
            "seed": args.seed,
            "num_beams": NUM_BEAMS,
            "beam_grid_shape": [8, 8],
            "patch_shape": [2, 2],
            "fft_order": FFT_ORDER,
            "beam_id_rule": "beam_id = row * 8 + col",
            "gain_reduction": "mean unit-norm DFT beam power over subcarriers",
            "acceptable_loss_db": args.acceptable_loss_db,
            "source_sample_count": len(samples),
            **source_metadata,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    manifest = {
        "output": str(args.output),
        "source": str(args.input.resolve()) if args.input is not None else source_metadata,
        "selected_samples_output": str(selected_samples_output.resolve()),
        "selected_count": len(source_indices),
        "split_sizes": {name: len(indices) for name, indices in split_indices.items()},
        "los_status_counts": dict(Counter(los_status)),
        "split_los_status_counts": {
            name: dict(Counter(los_status[int(index)] for index in indices))
            for name, indices in split_indices.items()
        },
        "split_group_ids_sha256": {
            name: digest([group_ids[int(index)] for index in indices])
            for name, indices in split_indices.items()
        },
        "oracle_class_counts": {
            str(key): value for key, value in sorted(Counter(oracle).items())
        },
        "group_ids_sha256": digest(group_ids),
        "metadata": payload["metadata"],
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"saved_data={args.output}")
    print(f"saved_selected_samples={selected_samples_output}")
    print(f"saved_manifest={manifest_path}")
    print(f"selected={len(source_indices)} train={args.train_size} val={args.val_size} test={args.test_size}")


if __name__ == "__main__":
    main()
