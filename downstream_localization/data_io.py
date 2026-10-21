from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import (  # noqa: E402
    CONFIG_FEATURE_DIM,
    DELAY_POWER_MAP_SHAPE,
    DELAY_POWER_PROFILE_BINS,
    _ensure_continuous_fields,
    _sample_configuration_features,
)


class LocalizationCorpus:
    def __init__(self, payload: dict[str, Any]):
        if payload.get("format") != "csi2text_localization_v1":
            raise ValueError("Unsupported localization dataset format")
        self.samples = [_ensure_continuous_fields(sample) for sample in payload["samples"]]
        self.tx_xyz_m = torch.as_tensor(payload["tx_xyz_m"], dtype=torch.float32)
        self.rx_xyz_m = torch.as_tensor(payload["rx_xyz_m"], dtype=torch.float32)
        self.relative_xy_m = torch.as_tensor(payload["relative_xy_m"], dtype=torch.float32)
        self.split_indices = {
            name: torch.as_tensor(indices, dtype=torch.long)
            for name, indices in payload["split_indices"].items()
        }
        self.metadata = dict(payload.get("metadata", {}))
        count = len(self.samples)
        if self.relative_xy_m.shape != (count, 2):
            raise ValueError(
                f"relative_xy_m must be [{count}, 2], got {tuple(self.relative_xy_m.shape)}"
            )
        all_indices = torch.cat([self.split_indices[name] for name in ("train", "val", "test")])
        if (
            len(all_indices) != count
            or len(torch.unique(all_indices)) != count
            or not torch.equal(torch.sort(all_indices).values, torch.arange(count))
        ):
            raise ValueError("train/val/test indices must partition the dataset exactly once")
        group_sets = {
            name: {self.samples[int(index)].group_id for index in indices}
            for name, indices in self.split_indices.items()
        }
        if any(
            group_sets[left] & group_sets[right]
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise ValueError("A map/source/rx link appears in more than one split")

    @classmethod
    def load(cls, path: Path) -> "LocalizationCorpus":
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a dictionary in {path}")
        return cls(payload)


class LocalizationDataset(Dataset):
    def __init__(self, corpus: LocalizationCorpus, indices: torch.Tensor):
        self.corpus = corpus
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        return {
            "index": index,
            "sample": self.corpus.samples[index],
            "target_xy": self.corpus.relative_xy_m[index],
            "tx_xyz": self.corpus.tx_xyz_m[index],
            "rx_xyz": self.corpus.rx_xyz_m[index],
        }


def collate_localization(items: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [item["sample"] for item in items]
    batch_size = len(samples)
    max_tokens = max(sample.n_tokens for sample in samples)
    d_token = int(samples[0].tokens.shape[1])
    n_freq = int(samples[0].tokens.shape[2])
    if any(tuple(sample.tokens.shape[1:]) != (d_token, n_freq) for sample in samples):
        raise ValueError("All samples in a batch must share d_token and n_freq")

    tokens = torch.zeros(batch_size, max_tokens, d_token, n_freq, dtype=torch.float32)
    beam_positions = torch.zeros(batch_size, max_tokens, 2, dtype=torch.float32)
    token_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    config_features = torch.zeros(batch_size, CONFIG_FEATURE_DIM, dtype=torch.float32)
    delay_power_map = torch.zeros(batch_size, *DELAY_POWER_MAP_SHAPE, dtype=torch.float32)
    delay_power_profile = torch.zeros(batch_size, DELAY_POWER_PROFILE_BINS, dtype=torch.float32)

    max_antennas = max(sample.array_rows * sample.array_cols for sample in samples)
    antenna_coordinates = torch.zeros(batch_size, max_antennas, 3, dtype=torch.float32)
    antenna_mask = torch.zeros(batch_size, max_antennas, dtype=torch.bool)
    for row, sample in enumerate(samples):
        count = sample.n_tokens
        tokens[row, :count] = sample.tokens.float()
        beam_positions[row, :count] = sample.beam_positions.float()
        token_mask[row, :count] = True
        config_features[row] = _sample_configuration_features(sample)
        coordinates = sample.antenna_coordinates_wavelengths.float()
        antenna_coordinates[row, : len(coordinates)] = coordinates
        antenna_mask[row, : len(coordinates)] = True
        delay_power_map[row] = sample.delay_power_map.float()
        delay_power_profile[row] = sample.delay_power_profile.float()

    return {
        "indices": torch.tensor([item["index"] for item in items], dtype=torch.long),
        "tokens": tokens,
        "beam_positions": beam_positions,
        "token_mask": token_mask,
        "freq_bin": torch.tensor([sample.freq_bin for sample in samples], dtype=torch.long),
        "bw_bin": torch.tensor([sample.bw_bin for sample in samples], dtype=torch.long),
        "subcarrier_spacing": torch.tensor(
            [sample.subcarrier_spacing_hz for sample in samples], dtype=torch.float32
        ),
        "config_features": config_features,
        "antenna_coordinates": antenna_coordinates,
        "antenna_mask": antenna_mask,
        "delay_power_map": delay_power_map,
        "delay_power_profile": delay_power_profile,
        "target_xy": torch.stack([item["target_xy"] for item in items]),
        "tx_xyz": torch.stack([item["tx_xyz"] for item in items]),
        "rx_xyz": torch.stack([item["rx_xyz"] for item in items]),
        "group_ids": [sample.group_id for sample in samples],
    }


def move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def encoder_kwargs(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "tokens": batch["tokens"],
        "beam_positions": batch["beam_positions"],
        "token_mask": batch["token_mask"],
        "freq_bin": batch["freq_bin"],
        "bw_bin": batch["bw_bin"],
        "subcarrier_spacing": batch["subcarrier_spacing"],
        "config_features": batch["config_features"],
        "antenna_coordinates": batch["antenna_coordinates"],
        "antenna_mask": batch["antenna_mask"],
    }
