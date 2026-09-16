from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preprocess_all import read_rayverse_propbin  # noqa: E402

from downstream_cp_selection.labeling import make_cp_label, validate_cp_lengths  # noqa: E402


DEFAULT_REPO = Path("/home/yrh/CSI_model/CSI_model")
DEFAULT_D2LOS_ROOT = DEFAULT_REPO / "deepmimo_scenarios/D2Los_Data"
DEFAULT_TRAIN_PT = (
    DEFAULT_REPO / "artifacts/d2los_50k_balanced_los_angle_seed23421_train40k.pt"
)
DEFAULT_TEST_PT = (
    DEFAULT_REPO / "artifacts/d2los_50k_balanced_los_angle_seed23421_test10k.pt"
)
DEFAULT_OUTPUT = DEFAULT_REPO / "downstream_cp_selection/artifacts/d2los_50k_cp_labels.pt"
GROUP_ID_PATTERN = re.compile(
    r"^.+-map_(?P<map>\d+)-source_(?P<source>\d+)-rx_(?P<rx>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build adaptive-CP labels from exact D2Los ray paths."
    )
    parser.add_argument("--train-pt", type=Path, default=DEFAULT_TRAIN_PT)
    parser.add_argument(
        "--val-pt",
        type=Path,
        help="Optional existing validation split; otherwise test10k is split 5k/5k.",
    )
    parser.add_argument("--test-pt", type=Path, default=DEFAULT_TEST_PT)
    parser.add_argument("--d2los-root", type=Path, default=DEFAULT_D2LOS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cp-ns", nargs="+", type=float, default=[40, 80, 160, 320, 640])
    parser.add_argument("--max-outside-energy-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=23421)
    parser.add_argument(
        "--overflow-policy",
        choices=("drop", "error", "last"),
        default="drop",
        help=(
            "What to do when even the longest CP leaves too much energy outside. "
            "drop preserves the exact label definition; last is an explicit clipping policy."
        ),
    )
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


def parse_group_id(group_id: str) -> tuple[int, int, int]:
    match = GROUP_ID_PATTERN.fullmatch(str(group_id))
    if match is None:
        raise ValueError(
            f"Cannot map group_id={group_id!r} to D2Los; expected "
            "...-map_<id>-source_<id>-rx_<id>"
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


def heldout_split(group_ids: list[str], seed: int) -> dict[int, str]:
    ranked = sorted(
        range(len(group_ids)),
        key=lambda index: hashlib.sha256(
            f"{seed}:{group_ids[index]}".encode("utf-8")
        ).digest(),
    )
    val = set(ranked[: len(ranked) // 2])
    return {index: ("val" if index in val else "test") for index in range(len(group_ids))}


def class_histogram(labels: list[int], cp_lengths: np.ndarray) -> dict[str, int]:
    counts = Counter(labels)
    return {
        f"CP-{index + 1}:{float(cp_ns):g}ns": int(counts.get(index, 0))
        for index, cp_ns in enumerate(cp_lengths)
    }


def main() -> None:
    args = parse_args()
    cp_lengths = validate_cp_lengths(args.cp_ns)
    if not args.d2los_root.exists():
        raise FileNotFoundError(args.d2los_root)

    source_samples: dict[str, list] = {"train": samples_from_pt(args.train_pt)}
    source_files: dict[str, str] = {"train": str(args.train_pt.resolve())}
    requested: list[dict[str, Any]] = []
    for local_index, sample in enumerate(source_samples["train"]):
        requested.append(
            {
                "source": "train",
                "local_index": local_index,
                "split": "train",
                "group_id": str(sample.group_id),
            }
        )

    if args.val_pt is not None:
        source_samples["val"] = samples_from_pt(args.val_pt)
        source_samples["test"] = samples_from_pt(args.test_pt)
        source_files["val"] = str(args.val_pt.resolve())
        source_files["test"] = str(args.test_pt.resolve())
        for source_name in ("val", "test"):
            for local_index, sample in enumerate(source_samples[source_name]):
                requested.append(
                    {
                        "source": source_name,
                        "local_index": local_index,
                        "split": source_name,
                        "group_id": str(sample.group_id),
                    }
                )
        split_policy = "existing_train_validation_test_files"
    else:
        source_samples["heldout"] = samples_from_pt(args.test_pt)
        source_files["heldout"] = str(args.test_pt.resolve())
        heldout_groups = [str(sample.group_id) for sample in source_samples["heldout"]]
        assignments = heldout_split(heldout_groups, args.seed)
        for local_index, sample in enumerate(source_samples["heldout"]):
            requested.append(
                {
                    "source": "heldout",
                    "local_index": local_index,
                    "split": assignments[local_index],
                    "group_id": str(sample.group_id),
                }
            )
        split_policy = "existing_train_plus_stable_half_split_of_test"

    group_ids = [entry["group_id"] for entry in requested]
    duplicate_count = len(group_ids) - len(set(group_ids))
    if duplicate_count:
        raise ValueError(f"Found {duplicate_count} duplicate/cross-split physical link IDs")

    links_by_source: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for entry_index, entry in enumerate(requested):
        map_id, source_id, rx_id = parse_group_id(entry["group_id"])
        links_by_source[(map_id, source_id)].append((entry_index, rx_id))

    labels_by_requested_index: dict[int, Any] = {}
    invalid_count = 0
    total_sources = len(links_by_source)
    for source_number, ((map_id, source_id), links) in enumerate(
        sorted(links_by_source.items()), start=1
    ):
        propbin_path = find_propbin(args.d2los_root, map_id, source_id)
        propbin = read_rayverse_propbin(propbin_path)
        for entry_index, rx_id in links:
            if not 0 <= rx_id < len(propbin.rx_records):
                raise IndexError(
                    f"rx_{rx_id} is outside {propbin_path} with "
                    f"{len(propbin.rx_records)} RX records"
                )
            rx_record = propbin.rx_records[rx_id]
            count = int(rx_record["path_count"])
            offset = int(rx_record["path_offset"])
            path_records = propbin.path_records[offset : offset + count]
            if len(path_records) != count:
                raise ValueError(f"Incomplete path slice for {requested[entry_index]['group_id']}")
            try:
                label = make_cp_label(
                    path_records["delay_ns"],
                    path_records["path_loss_db"],
                    cp_lengths.tolist(),
                    args.max_outside_energy_ratio,
                )
            except ValueError as exc:
                invalid_count += 1
                raise ValueError(
                    f"Cannot label {requested[entry_index]['group_id']}: {exc}"
                ) from exc
            labels_by_requested_index[entry_index] = label
        if source_number % 100 == 0 or source_number == total_sources:
            print(f"ray sources={source_number}/{total_sources}")

    retained: list[dict[str, Any]] = []
    overflows: list[dict[str, Any]] = []
    for index, entry in enumerate(requested):
        label = labels_by_requested_index[index]
        class_index = label.class_index
        if class_index is None:
            overflows.append(entry)
            if args.overflow_policy == "drop" or args.overflow_policy == "error":
                continue
            class_index = len(cp_lengths) - 1
        retained.append({**entry, "label": label, "class_index": class_index})

    if args.overflow_policy == "error" and overflows:
        raise RuntimeError(
            f"{len(overflows)} links exceed the {args.max_outside_energy_ratio:.4g} "
            f"energy threshold even at {cp_lengths[-1]:g} ns. Add a longer CP or use "
            "--overflow-policy drop after reviewing the printed distribution."
        )
    if not retained:
        raise RuntimeError("No samples remain after CP labeling")

    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for retained_index, entry in enumerate(retained):
        split_indices[entry["split"]].append(retained_index)
    if any(not values for values in split_indices.values()):
        raise RuntimeError(f"An empty split remains after filtering: {split_indices}")

    outside = np.stack(
        [entry["label"].outside_energy_ratios for entry in retained]
    ).astype(np.float32)
    labels = [int(entry["class_index"]) for entry in retained]
    payload = {
        "format": "csi_cp_selection_v1",
        "source_files": source_files,
        "sample_refs": [
            (entry["source"], int(entry["local_index"])) for entry in retained
        ],
        "group_ids": [entry["group_id"] for entry in retained],
        "labels": torch.tensor(labels, dtype=torch.long),
        "outside_energy_ratios": torch.from_numpy(outside),
        "max_excess_delay_ns": torch.tensor(
            [entry["label"].max_excess_delay_ns for entry in retained],
            dtype=torch.float32,
        ),
        "energy_delay_ns": torch.tensor(
            [entry["label"].energy_delay_ns for entry in retained], dtype=torch.float32
        ),
        "path_count": torch.tensor(
            [entry["label"].path_count for entry in retained], dtype=torch.long
        ),
        "cp_lengths_ns": torch.tensor(cp_lengths, dtype=torch.float32),
        "max_outside_energy_ratio": float(args.max_outside_energy_ratio),
        "split_indices": {
            name: torch.tensor(indices, dtype=torch.long)
            for name, indices in split_indices.items()
        },
        "metadata": {
            "d2los_root": str(args.d2los_root.resolve()),
            "split_policy": split_policy,
            "split_seed": args.seed,
            "overflow_policy": args.overflow_policy,
            "overflow_count": len(overflows),
            "invalid_count": invalid_count,
            "requested_count": len(requested),
            "retained_count": len(retained),
            "label_definition": "shortest CP with true relative outside energy <= threshold",
            "strict_outside_comparison": "excess_delay_ns > cp_ns",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(args.output)

    report = {
        "output": str(args.output.resolve()),
        "requested": len(requested),
        "retained": len(retained),
        "overflow": len(overflows),
        "cp_lengths_ns": cp_lengths.tolist(),
        "threshold": args.max_outside_energy_ratio,
        "splits": {name: len(indices) for name, indices in split_indices.items()},
        "overall_classes": class_histogram(labels, cp_lengths),
        "split_classes": {
            name: class_histogram([labels[index] for index in indices], cp_lengths)
            for name, indices in split_indices.items()
        },
        "outside_at_longest_cp_pct": {
            "mean": float(100.0 * outside[:, -1].mean()),
            "p95": float(100.0 * np.percentile(outside[:, -1], 95)),
            "max": float(100.0 * outside[:, -1].max()),
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"summary={summary_path.resolve()}")


if __name__ == "__main__":
    main()
