from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


DEFAULT_REPO = Path("/home/yrh/CSI_model/CSI_model")
DEFAULT_DIR = DEFAULT_REPO / "downstream_mcs_adaptation/artifacts/test_phy_validation_seed23421"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fast EESM oracle labels with final-test full-PHY block errors."
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_DIR / "selection.pt")
    parser.add_argument("--results", type=Path, default=DEFAULT_DIR / "simulation_results.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR / "validation_metrics.json")
    return parser.parse_args()


def read_results(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for line, raw in enumerate(csv.DictReader(stream), start=2):
            try:
                row = {
                    "group_id": str(raw["group_id"]),
                    "mcs_index": int(raw["mcs_index"]),
                    "average_snr_db": float(raw["average_snr_db"]),
                    "block_errors": int(raw["block_errors"]),
                    "total_blocks": int(raw["total_blocks"]),
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid result row {line}: {raw}") from exc
            if (
                not math.isfinite(row["average_snr_db"])
                or row["total_blocks"] <= 0
                or not 0 <= row["block_errors"] <= row["total_blocks"]
            ):
                raise ValueError(f"Invalid result values at row {line}: {row}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No simulation results in {path}")
    return rows


def main() -> None:
    args = parse_args()
    selection = torch.load(args.selection, map_location="cpu", weights_only=False)
    if not isinstance(selection, dict) or selection.get("format") != "csi_mcs_test_phy_validation_v1":
        raise ValueError("Unsupported test-PHY validation selection")
    group_ids = [str(value) for value in selection["group_ids"]]
    average_snr = torch.as_tensor(selection["average_snr_db"], dtype=torch.float64)
    fast_oracle = torch.as_tensor(selection["fast_oracle_mcs"], dtype=torch.long)
    mcs_count = len(selection["mcs_names"])
    target_bler = float(selection["target_bler"])
    if len(group_ids) != len(set(group_ids)) or len(group_ids) != len(fast_oracle):
        raise ValueError("Invalid or duplicate selected group IDs")
    lookup = {group_id: index for index, group_id in enumerate(group_ids)}
    observations: dict[str, dict[int, tuple[int, int]]] = defaultdict(dict)
    for row in read_results(args.results):
        if row["group_id"] not in lookup:
            raise ValueError(f"Result contains a non-selected link: {row['group_id']}")
        link_index = lookup[row["group_id"]]
        if not 0 <= row["mcs_index"] < mcs_count:
            raise ValueError(f"Unknown MCS index {row['mcs_index']}")
        if not math.isclose(
            row["average_snr_db"], float(average_snr[link_index]), rel_tol=0.0, abs_tol=1e-4
        ):
            raise ValueError(f"SNR mismatch for {row['group_id']} MCS {row['mcs_index']}")
        if row["mcs_index"] in observations[row["group_id"]]:
            raise ValueError(f"Duplicate result for {row['group_id']} MCS {row['mcs_index']}")
        observations[row["group_id"]][row["mcs_index"]] = (
            row["block_errors"],
            row["total_blocks"],
        )

    fast_selected_bler = []
    per_link = []
    for index, group_id in enumerate(group_ids):
        if set(observations[group_id]) != set(range(mcs_count)):
            missing = sorted(set(range(mcs_count)) - set(observations[group_id]))
            raise ValueError(f"Incomplete MCS sweep for {group_id}; missing {missing}")
        empirical_bler = [
            observations[group_id][mcs][0] / observations[group_id][mcs][1]
            for mcs in range(mcs_count)
        ]
        feasible = [mcs for mcs, value in enumerate(empirical_bler) if value <= target_bler]
        phy_oracle = max(feasible, default=-1)
        fast = int(fast_oracle[index])
        selected_bler = empirical_bler[fast] if fast >= 0 else 0.0
        selected_errors, selected_blocks = observations[group_id][fast] if fast >= 0 else (0, 0)
        fast_selected_bler.append(selected_bler)
        per_link.append(
            {
                "group_id": group_id,
                "average_snr_db": float(average_snr[index]),
                "fast_oracle_mcs": fast,
                "full_phy_oracle_mcs": phy_oracle,
                "empirical_bler_per_mcs": json.dumps(empirical_bler),
                "fast_selected_empirical_bler": selected_bler,
                "fast_selected_block_errors": selected_errors,
                "fast_selected_total_blocks": selected_blocks,
                "match": fast == phy_oracle,
                "fast_aggressive": fast > phy_oracle,
                "fast_conservative": fast < phy_oracle,
            }
        )
    count = len(group_ids)
    metrics = {
        "evaluated_final_test_links": count,
        "target_bler": target_bler,
        "fast_oracle_accuracy_pct": 100.0 * sum(row["match"] for row in per_link) / count,
        "fast_oracle_aggressive_rate_pct": 100.0 * sum(row["fast_aggressive"] for row in per_link) / count,
        "fast_oracle_conservative_rate_pct": 100.0 * sum(row["fast_conservative"] for row in per_link) / count,
        "fast_oracle_bler_violation_rate_pct": 100.0
        * sum(value > target_bler for value in fast_selected_bler)
        / count,
        "mean_fast_selected_empirical_bler_pct": 100.0 * sum(fast_selected_bler) / count,
        "note": "Empirical BLER uncertainty depends on total_blocks; report block counts with these metrics.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metrics": metrics, "per_link": per_link}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_link[0]))
        writer.writeheader()
        writer.writerows(per_link)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
