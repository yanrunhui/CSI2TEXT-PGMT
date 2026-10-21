from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream_mcs_adaptation.mcs import (  # noqa: E402
    effective_frequency_response,
    file_sha256,
    load_json,
    noise_power_dbw,
    subcarrier_power_weights,
    frequency_selectivity_db,
    validate_uncalibrated_config,
)


DEFAULT_REPO = Path("/home/yrh/CSI_model/CSI_model")
DEFAULT_TRAIN = (
    DEFAULT_REPO
    / "artifacts/d2los_50k_upa8x8_bw100m_nf128_los_nlos_balanced/upa8x8_100mhz_train.pt"
)
DEFAULT_TABLE = DEFAULT_REPO / "downstream_mcs_adaptation/mcs_table.json"
DEFAULT_OUTPUT = DEFAULT_REPO / "downstream_mcs_adaptation/artifacts/calibration_seed23421"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select representative training links and emit full-link BLER simulation requests."
    )
    parser.add_argument("--train-pt", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--mcs-table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-links", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=23421)
    parser.add_argument("--snr-min-db", type=float, default=-10.0)
    parser.add_argument("--snr-max-db", type=float, default=35.0)
    parser.add_argument("--snr-step-db", type=float, default=2.5)
    parser.add_argument("--effective-channel-mode", choices=("best_beam", "mrc"), default="best_beam")
    return parser.parse_args()


def samples_from_pt(path: Path) -> list:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    samples = payload if isinstance(payload, list) else payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Expected a non-empty sample list in {path}")
    return samples


def finite(value: Any, name: str, group_id: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {name} for {group_id}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {name} for {group_id}")
    return result


def quantile_bin(values: np.ndarray, count: int) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, count + 1)))
    if len(edges) <= 2:
        return np.zeros(len(values), dtype=np.int64)
    return np.searchsorted(edges[1:-1], values, side="right").astype(np.int64)


def stable_key(seed: int, group_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()


def choose_round_robin(
    buckets: dict[tuple[int, ...], list[int]],
    group_ids: list[str],
    count: int,
    seed: int,
) -> list[int]:
    for indices in buckets.values():
        indices.sort(key=lambda index: stable_key(seed, group_ids[index]))
    ordered_keys = sorted(buckets, key=lambda key: stable_key(seed + 1, repr(key)))
    offsets = {key: 0 for key in ordered_keys}
    selected: list[int] = []
    while len(selected) < count:
        progressed = False
        for key in ordered_keys:
            offset = offsets[key]
            if offset < len(buckets[key]):
                selected.append(buckets[key][offset])
                offsets[key] += 1
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.num_links <= 0 or args.snr_step_db <= 0.0 or args.snr_max_db < args.snr_min_db:
        raise ValueError("Invalid selection size or SNR sweep")
    config = load_json(args.mcs_table)
    validate_uncalibrated_config(config)
    samples = samples_from_pt(args.train_pt)
    if args.num_links > len(samples):
        raise ValueError(f"Requested {args.num_links} links from only {len(samples)} samples")

    group_ids: list[str] = []
    powers: list[float] = []
    delays: list[float] = []
    selectivities: list[float] = []
    los_values: list[int] = []
    for index, sample in enumerate(samples):
        group_id = str(getattr(sample, "group_id", "")).strip()
        if not group_id:
            raise ValueError(f"Sample {index} has no group_id")
        group_ids.append(group_id)
        powers.append(finite(getattr(sample, "total_received_power_dbw", math.nan), "total power", group_id))
        delay_s = finite(getattr(sample, "delay_spread_s", math.nan), "delay spread", group_id)
        delays.append(math.log1p(max(delay_s * 1e9, 0.0)))
        weights = subcarrier_power_weights(sample, args.effective_channel_mode)
        selectivities.append(frequency_selectivity_db(weights))
        los = str(getattr(getattr(sample, "semantic_key", None), "los_status", "")).lower()
        if los not in {"los", "nlos"}:
            raise ValueError(f"Invalid LoS status {los!r} for {group_id}")
        los_values.append(1 if los == "los" else 0)
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("Training data contains duplicate physical-link group IDs")

    power_bins = quantile_bin(np.asarray(powers), 8)
    delay_bins = quantile_bin(np.asarray(delays), 3)
    selectivity_bins = quantile_bin(np.asarray(selectivities), 3)
    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index in range(len(samples)):
        buckets[(los_values[index], int(power_bins[index]), int(delay_bins[index]), int(selectivity_bins[index]))].append(index)
    selected = choose_round_robin(buckets, group_ids, args.num_links, args.seed)
    if len(selected) != args.num_links:
        raise RuntimeError(f"Selected only {len(selected)} of {args.num_links} requested links")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for selection_index, source_index in enumerate(selected):
        summary_rows.append(
            {
                "selection_index": selection_index,
                "source_index": source_index,
                "group_id": group_ids[source_index],
                "los_status": "los" if los_values[source_index] else "nlos",
                "total_received_power_dbw": powers[source_index],
                "base_average_snr_db": powers[source_index] - noise_power_dbw(config["system"]),
                "delay_spread_ns": math.expm1(delays[source_index]),
                "frequency_selectivity_db": selectivities[source_index],
                "power_bin": int(power_bins[source_index]),
                "delay_bin": int(delay_bins[source_index]),
                "selectivity_bin": int(selectivity_bins[source_index]),
            }
        )
    write_csv(args.output_dir / "selected_links.csv", summary_rows)
    effective_channels = np.stack(
        [effective_frequency_response(samples[index], args.effective_channel_mode) for index in selected]
    ).astype(np.complex64)
    np.savez_compressed(
        args.output_dir / "effective_channels.npz",
        group_id=np.asarray([group_ids[index] for index in selected]),
        source_index=np.asarray(selected, dtype=np.int64),
        normalized_frequency_response=effective_channels,
        subcarrier_power_weights=np.abs(effective_channels).astype(np.float32) ** 2,
    )

    snr_grid = np.arange(
        args.snr_min_db,
        args.snr_max_db + 0.5 * args.snr_step_db,
        args.snr_step_db,
        dtype=np.float64,
    )
    request_rows = []
    for row in summary_rows:
        for mcs in config["mcs"]:
            for snr_db in snr_grid:
                request_rows.append(
                    {
                        "group_id": row["group_id"],
                        "source_index": row["source_index"],
                        "mcs_index": int(mcs["index"]),
                        "average_snr_db": float(snr_db),
                        "block_errors": "",
                        "total_blocks": "",
                    }
                )
    write_csv(args.output_dir / "simulation_requests.csv", request_rows)

    selection_payload = {
        "format": "csi_mcs_calibration_selection_v1",
        "source_train_pt": str(args.train_pt.resolve()),
        "source_indices": torch.tensor(selected, dtype=torch.long),
        "group_ids": [group_ids[index] for index in selected],
        "effective_channel_mode": args.effective_channel_mode,
        "mcs_table": str(args.mcs_table.resolve()),
        "mcs_table_sha256": file_sha256(args.mcs_table),
        "seed": args.seed,
        "snr_grid_db": torch.tensor(snr_grid, dtype=torch.float32),
    }
    torch.save(selection_payload, args.output_dir / "selection.pt")
    report = {
        "source_train_pt": str(args.train_pt.resolve()),
        "selected_links": len(selected),
        "occupied_strata": len(buckets),
        "los_links": int(sum(los_values[index] for index in selected)),
        "nlos_links": int(len(selected) - sum(los_values[index] for index in selected)),
        "simulation_requests": len(request_rows),
        "snr_grid_db": snr_grid.tolist(),
        "effective_channel_mode": args.effective_channel_mode,
        "test_data_used": False,
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
