from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream_mcs_adaptation.mcs import (  # noqa: E402
    file_sha256,
    label_link,
    load_calibrated_config,
    noise_power_dbm,
    noise_power_dbw,
    sample_bandwidth_hz,
)


DEFAULT_REPO = Path("/home/yrh/CSI_model/CSI_model")
DEFAULT_TRAIN = (
    DEFAULT_REPO
    / "artifacts/d2los_50k_upa8x8_bw100m_nf128_los_nlos_balanced/upa8x8_100mhz_train.pt"
)
DEFAULT_TEST = (
    DEFAULT_REPO
    / "artifacts/d2los_50k_upa8x8_bw100m_nf128_los_nlos_balanced/upa8x8_100mhz_test.pt"
)
DEFAULT_CALIBRATED = DEFAULT_REPO / "downstream_mcs_adaptation/artifacts/calibration_seed23421/calibrated_mcs.json"
DEFAULT_OUTPUT = DEFAULT_REPO / "downstream_mcs_adaptation/artifacts/d2los_50k_mcs_labels.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reuse preprocessed CSI and total-power labels to generate calibrated oracle MCS labels."
    )
    parser.add_argument("--train-pt", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test-pt", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--val-pt", type=Path)
    parser.add_argument("--calibrated-mcs", type=Path, default=DEFAULT_CALIBRATED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=23421)
    parser.add_argument(
        "--tx-power-levels-dbm",
        type=float,
        nargs="+",
        default=[23.0],
        help=(
            "Transmit-power profiles. Propagation CSI stays unchanged and transmit power is "
            "provided explicitly to every downstream classifier."
        ),
    )
    parser.add_argument("--allow-bandwidth-mismatch", action="store_true")
    parser.add_argument("--allow-missing-power", action="store_true")
    return parser.parse_args()


def samples_from_pt(path: Path) -> list:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    samples = payload if isinstance(payload, list) else payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Expected a non-empty sample list in {path}")
    return samples


def stable_heldout_assignment(group_ids: list[str], seed: int) -> list[str]:
    ranked = sorted(
        range(len(group_ids)),
        key=lambda index: hashlib.sha256(f"{seed}:{group_ids[index]}".encode("utf-8")).digest(),
    )
    validation = set(ranked[: len(ranked) // 2])
    return ["val" if index in validation else "test" for index in range(len(group_ids))]


def class_histogram(labels: list[int], mcs_count: int) -> dict[str, int]:
    counts = Counter(labels)
    result = {"Outage": int(counts.get(0, 0))}
    result.update({f"MCS{index}": int(counts.get(index + 1, 0)) for index in range(mcs_count)})
    return result


def main() -> None:
    args = parse_args()
    config, calibrated = load_calibrated_config(args.calibrated_mcs)
    system = config["system"]
    base_tx_power_dbm = float(system["transmit_power_dbm"])
    tx_power_levels = [float(value) for value in args.tx_power_levels_dbm]
    if (
        not tx_power_levels
        or any(not math.isfinite(value) for value in tx_power_levels)
        or len(set(tx_power_levels)) != len(tx_power_levels)
    ):
        raise ValueError("tx-power-levels-dbm must be finite and unique")
    expected_bandwidth = float(system["bandwidth_hz"])
    mode = str(config.get("effective_channel", {}).get("mode", "best_beam"))
    calibration_mode = str(config.get("calibration", {}).get("effective_channel_mode", mode))
    if mode != calibration_mode:
        raise ValueError(
            f"Label effective-channel mode {mode!r} differs from calibration mode {calibration_mode!r}"
        )
    calibrated_train = config.get("calibration", {}).get("source_train_pt")
    if calibrated_train is not None and Path(calibrated_train).resolve() != args.train_pt.resolve():
        raise ValueError(
            "Calibrated MCS parameters came from a different training file: "
            f"{calibrated_train}"
        )

    sources: dict[str, list] = {"train": samples_from_pt(args.train_pt)}
    source_files = {"train": str(args.train_pt.resolve())}
    base_entries: list[dict[str, Any]] = []
    for local_index, sample in enumerate(sources["train"]):
        base_entries.append(
            {"source": "train", "local_index": local_index, "split": "train", "sample": sample}
        )
    if args.val_pt is not None:
        sources["val"] = samples_from_pt(args.val_pt)
        sources["test"] = samples_from_pt(args.test_pt)
        source_files.update({"val": str(args.val_pt.resolve()), "test": str(args.test_pt.resolve())})
        for source_name in ("val", "test"):
            for local_index, sample in enumerate(sources[source_name]):
                base_entries.append(
                    {"source": source_name, "local_index": local_index, "split": source_name, "sample": sample}
                )
        split_policy = "existing_train_validation_test_files"
    else:
        sources["heldout"] = samples_from_pt(args.test_pt)
        source_files["heldout"] = str(args.test_pt.resolve())
        heldout_ids = [str(sample.group_id) for sample in sources["heldout"]]
        assignments = stable_heldout_assignment(heldout_ids, args.seed)
        for local_index, (sample, split) in enumerate(zip(sources["heldout"], assignments)):
            base_entries.append(
                {"source": "heldout", "local_index": local_index, "split": split, "sample": sample}
            )
        split_policy = "existing_train_plus_stable_half_split_of_test"

    physical_group_ids = [str(entry["sample"].group_id) for entry in base_entries]
    if any(not value for value in physical_group_ids):
        raise ValueError("Every sample must have a non-empty group_id")
    if len(set(physical_group_ids)) != len(physical_group_ids):
        raise ValueError("Duplicate physical-link group IDs exist within or across splits")

    entries: list[dict[str, Any]] = []
    for base_entry in base_entries:
        physical_group_id = str(base_entry["sample"].group_id)
        for tx_power_dbm in tx_power_levels:
            entries.append(
                {
                    **base_entry,
                    "physical_group_id": physical_group_id,
                    "profile_group_id": f"{physical_group_id}|tx={tx_power_dbm:.6g}dBm",
                    "transmit_power_dbm": tx_power_dbm,
                    "power_offset_db": tx_power_dbm - base_tx_power_dbm,
                }
            )

    records: list[dict[str, Any]] = []
    retained_entries: list[dict[str, Any]] = []
    skipped_missing_power = 0
    bandwidth_missing = 0
    bandwidth_mismatch = 0
    for number, entry in enumerate(entries, start=1):
        sample = entry["sample"]
        sample_bandwidth = sample_bandwidth_hz(sample)
        if sample_bandwidth is None:
            bandwidth_missing += 1
        elif not math.isclose(sample_bandwidth, expected_bandwidth, rel_tol=1e-6, abs_tol=1.0):
            bandwidth_mismatch += 1
            if not args.allow_bandwidth_mismatch:
                raise ValueError(
                    f"Bandwidth mismatch for {sample.group_id}: sample={sample_bandwidth}, "
                    f"calibration={expected_bandwidth}"
                )
        try:
            record = label_link(
                sample,
                calibrated,
                system,
                mode,
                power_offset_db=float(entry["power_offset_db"]),
            )
        except ValueError as exc:
            if args.allow_missing_power and "total_received_power_dbw" in str(exc):
                skipped_missing_power += 1
                continue
            raise
        records.append(record)
        retained_entries.append(entry)
        if number % 5000 == 0 or number == len(entries):
            print(f"labeled={number}/{len(entries)} retained={len(records)}", flush=True)
    if not records:
        raise RuntimeError("No links were labeled")

    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for index, entry in enumerate(retained_entries):
        split_indices[entry["split"]].append(index)
    if any(not values for values in split_indices.values()):
        raise RuntimeError(f"An empty split remains: {split_indices}")

    labels = [int(record["class_index"]) for record in records]
    effective = np.stack([record["effective_snr_per_mcs"] for record in records]).astype(np.float32)
    bler = np.stack([record["estimated_bler_per_mcs"] for record in records]).astype(np.float32)
    payload = {
        "format": "csi_mcs_adaptation_v1",
        "source_files": source_files,
        "sample_refs": [
            (entry["source"], int(entry["local_index"])) for entry in retained_entries
        ],
        "group_ids": [str(entry["profile_group_id"]) for entry in retained_entries],
        "physical_group_ids": [str(entry["physical_group_id"]) for entry in retained_entries],
        "transmit_power_dbm": torch.tensor(
            [entry["transmit_power_dbm"] for entry in retained_entries], dtype=torch.float32
        ),
        "power_offset_db": torch.tensor(
            [entry["power_offset_db"] for entry in retained_entries], dtype=torch.float32
        ),
        "labels": torch.tensor(labels, dtype=torch.long),
        "oracle_mcs": torch.tensor([record["oracle_mcs"] for record in records], dtype=torch.long),
        "total_received_power_dbw": torch.tensor(
            [record["total_received_power_dbw"] for record in records], dtype=torch.float32
        ),
        "noise_power_dbw": float(noise_power_dbw(system)),
        "average_snr_db": torch.tensor([record["average_snr_db"] for record in records], dtype=torch.float32),
        "frequency_selectivity_db": torch.tensor(
            [record["frequency_selectivity_db"] for record in records], dtype=torch.float32
        ),
        "effective_snr_per_mcs": torch.from_numpy(effective),
        "estimated_bler_per_mcs": torch.from_numpy(bler),
        "spectral_efficiency": torch.tensor(
            [item.spectral_efficiency for item in calibrated], dtype=torch.float32
        ),
        "mcs_names": [item.name for item in calibrated],
        "split_indices": {
            name: torch.tensor(indices, dtype=torch.long) for name, indices in split_indices.items()
        },
        "system": system,
        "calibrated_mcs": [item.__dict__ for item in calibrated],
        "metadata": {
            "calibrated_mcs_file": str(args.calibrated_mcs.resolve()),
            "calibrated_mcs_sha256": file_sha256(args.calibrated_mcs),
            "effective_channel_mode": mode,
            "split_policy": split_policy,
            "split_seed": args.seed,
            "base_transmit_power_dbm": base_tx_power_dbm,
            "tx_power_levels_dbm": tx_power_levels,
            "power_profiles_per_physical_link": len(tx_power_levels),
            "power_profile_input_mode": "explicit_transmit_power",
            "csi_amplitude_scaled_by_power_profile": False,
            "transmit_power_is_classifier_input": True,
            "predicted_total_power_adjustment": (
                "predicted_rx_at_base_tx_plus_tx_power_offset_db"
            ),
            "raw_d2los_reextracted": False,
            "oracle_uses_true_total_power": True,
            "classifier_physics_must_be_predictions": True,
            "bandwidth_missing_count": bandwidth_missing,
            "bandwidth_mismatch_count": bandwidth_mismatch,
            "skipped_missing_power_count": skipped_missing_power,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(args.output)

    split_histograms = {
        name: class_histogram([labels[index] for index in indices], len(calibrated))
        for name, indices in split_indices.items()
    }
    overall = class_histogram(labels, len(calibrated))
    largest_fraction = max(overall.values()) / len(labels)
    extreme_fraction = (overall["Outage"] + overall[f"MCS{len(calibrated) - 1}"]) / len(labels)
    smallest_nonzero_class = min(overall.values())
    report = {
        "output": str(args.output.resolve()),
        "requested_count": len(entries),
        "retained_count": len(records),
        "splits": {name: len(indices) for name, indices in split_indices.items()},
        "class_mapping": {"0": "Outage", **{str(i + 1): item.name for i, item in enumerate(calibrated)}},
        "overall_class_histogram": overall,
        "split_class_histograms": split_histograms,
        "largest_class_fraction_pct": 100.0 * largest_fraction,
        "extreme_class_fraction_pct": 100.0 * extreme_fraction,
        "smallest_class_count": smallest_nonzero_class,
        "class_imbalance_warning": largest_fraction >= 0.8 or extreme_fraction >= 0.8,
        "base_transmit_power_dbm": base_tx_power_dbm,
        "tx_power_levels_dbm": tx_power_levels,
        "power_profile_input_mode": "explicit_transmit_power",
        "csi_amplitude_scaled_by_power_profile": False,
        "transmit_power_is_classifier_input": True,
        "noise_power_dbm": noise_power_dbm(system),
        "noise_power_dbw": noise_power_dbw(system),
        "effective_channel_mode": mode,
        "raw_d2los_reextracted": False,
    }
    summary = args.output.with_suffix(".summary.json")
    summary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if largest_fraction >= 0.8 or extreme_fraction >= 0.8:
        print(
            "WARNING: at least 80% of examples occupy one class or the two extreme classes. "
            "Calibrate/check thresholds, or revise power profiles while preserving physical splits.",
            flush=True,
        )


if __name__ == "__main__":
    main()
