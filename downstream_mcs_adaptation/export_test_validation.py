from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from downstream_mcs_adaptation.data_io import MCSCorpus  # noqa: E402
from downstream_mcs_adaptation.mcs import effective_frequency_response, file_sha256  # noqa: E402


DEFAULT_REPO = Path("/home/yrh/CSI_model/CSI_model")
DEFAULT_DATA = DEFAULT_REPO / "downstream_mcs_adaptation/artifacts/d2los_50k_mcs_labels.pt"
DEFAULT_OUTPUT = DEFAULT_REPO / "downstream_mcs_adaptation/artifacts/test_phy_validation_seed23421"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a final-test-only PHY sweep for independent oracle-label validation."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-links", type=int, default=500)
    parser.add_argument("--seed", type=int, default=23421)
    return parser.parse_args()


def stable_key(seed: int, group_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    corpus = MCSCorpus.load(args.data)
    test_indices = corpus.split_indices["test"].tolist()
    if not 0 < args.num_links <= len(test_indices):
        raise ValueError(
            f"num-links must be in [1, {len(test_indices)}], got {args.num_links}"
        )
    selected = sorted(
        test_indices,
        key=lambda index: stable_key(args.seed, corpus.group_ids[index]),
    )[: args.num_links]
    mode = str(corpus.metadata["effective_channel_mode"])
    responses = np.stack(
        [effective_frequency_response(corpus.samples[index], mode) for index in selected]
    ).astype(np.complex64)
    group_ids = [corpus.group_ids[index] for index in selected]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "effective_channels.npz",
        group_id=np.asarray(group_ids),
        corpus_index=np.asarray(selected, dtype=np.int64),
        normalized_frequency_response=responses,
        subcarrier_power_weights=np.abs(responses).astype(np.float32) ** 2,
        average_snr_db=corpus.average_snr_db[selected].numpy(),
        fast_oracle_mcs=corpus.oracle_mcs[selected].numpy(),
    )
    requests = []
    for index, group_id in zip(selected, group_ids):
        for mcs_index in range(len(corpus.mcs_names)):
            requests.append(
                {
                    "group_id": group_id,
                    "corpus_index": index,
                    "mcs_index": mcs_index,
                    "average_snr_db": float(corpus.average_snr_db[index]),
                    "fast_oracle_mcs": int(corpus.oracle_mcs[index]),
                    "block_errors": "",
                    "total_blocks": "",
                }
            )
    write_csv(args.output_dir / "simulation_requests.csv", requests)
    selection = {
        "format": "csi_mcs_test_phy_validation_v1",
        "source_data": str(args.data.resolve()),
        "source_data_sha256": file_sha256(args.data),
        "test_indices": torch.tensor(selected, dtype=torch.long),
        "group_ids": group_ids,
        "average_snr_db": corpus.average_snr_db[selected].clone(),
        "fast_oracle_mcs": corpus.oracle_mcs[selected].clone(),
        "mcs_names": list(corpus.mcs_names),
        "target_bler": float(corpus.system["target_bler"]),
        "effective_channel_mode": mode,
        "seed": args.seed,
    }
    torch.save(selection, args.output_dir / "selection.pt")
    summary = {
        "source_data": str(args.data.resolve()),
        "selected_final_test_links": len(selected),
        "simulation_requests": len(requests),
        "mcs_count": len(corpus.mcs_names),
        "target_bler": float(corpus.system["target_bler"]),
        "effective_channel_mode": mode,
        "training_or_validation_links_used": False,
    }
    (args.output_dir / "export_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
