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


def _samples_from_pt(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if isinstance(payload, list):
        samples = payload
    elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        samples = payload["samples"]
    else:
        raise ValueError(f"Expected a sample list in {path}")
    return samples


class CPCorpus:
    def __init__(self, payload: dict[str, Any], manifest_path: Path):
        if payload.get("format") != "csi_cp_selection_v1":
            raise ValueError("Unsupported CP-selection dataset format")
        source_paths = {
            name: Path(path) for name, path in dict(payload["source_files"]).items()
        }
        # Relative source paths are resolved beside the manifest.  Absolute paths
        # (the normal server setup) remain unchanged.
        source_paths = {
            name: path if path.is_absolute() else manifest_path.parent / path
            for name, path in source_paths.items()
        }
        loaded = {name: _samples_from_pt(path) for name, path in source_paths.items()}
        refs = payload["sample_refs"]
        self.samples = []
        for source_name, local_index in refs:
            if source_name not in loaded:
                raise ValueError(f"Unknown sample source {source_name!r}")
            index = int(local_index)
            if not 0 <= index < len(loaded[source_name]):
                raise IndexError(f"Sample reference {source_name}[{index}] is out of range")
            self.samples.append(_ensure_continuous_fields(loaded[source_name][index]))

        count = len(self.samples)
        self.labels = torch.as_tensor(payload["labels"], dtype=torch.long)
        self.outside_energy_ratios = torch.as_tensor(
            payload["outside_energy_ratios"], dtype=torch.float32
        )
        self.max_excess_delay_ns = torch.as_tensor(
            payload["max_excess_delay_ns"], dtype=torch.float32
        )
        self.energy_delay_ns = torch.as_tensor(payload["energy_delay_ns"], dtype=torch.float32)
        self.path_count = torch.as_tensor(payload["path_count"], dtype=torch.long)
        self.group_ids = [str(value) for value in payload["group_ids"]]
        self.cp_lengths_ns = torch.as_tensor(payload["cp_lengths_ns"], dtype=torch.float32)
        self.max_outside_energy_ratio = float(payload["max_outside_energy_ratio"])
        self.split_indices = {
            name: torch.as_tensor(indices, dtype=torch.long)
            for name, indices in payload["split_indices"].items()
        }
        self.metadata = dict(payload.get("metadata", {}))

        if self.labels.shape != (count,):
            raise ValueError(f"labels must have shape [{count}]")
        expected_energy_shape = (count, len(self.cp_lengths_ns))
        if tuple(self.outside_energy_ratios.shape) != expected_energy_shape:
            raise ValueError(
                f"outside_energy_ratios must have shape {expected_energy_shape}"
            )
        if self.labels.numel() and (
            int(self.labels.min()) < 0 or int(self.labels.max()) >= len(self.cp_lengths_ns)
        ):
            raise ValueError("CP class index is outside cp_lengths_ns")
        all_indices = torch.cat(
            [self.split_indices[name] for name in ("train", "val", "test")]
        )
        if (
            len(all_indices) != count
            or len(torch.unique(all_indices)) != count
            or not torch.equal(torch.sort(all_indices).values, torch.arange(count))
        ):
            raise ValueError("train/val/test indices must partition retained samples once")
        groups = {
            name: {self.group_ids[int(index)] for index in indices}
            for name, indices in self.split_indices.items()
        }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            if groups[left] & groups[right]:
                raise ValueError(f"Physical-link leakage between {left} and {right}")

    @classmethod
    def load(cls, path: Path) -> "CPCorpus":
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a dictionary in {path}")
        return cls(payload, path.resolve())


class CPDataset(Dataset):
    def __init__(self, corpus: CPCorpus, indices: torch.Tensor):
        self.corpus = corpus
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        return {
            "index": index,
            "sample": self.corpus.samples[index],
            "label": self.corpus.labels[index],
            "outside_energy_ratios": self.corpus.outside_energy_ratios[index],
        }


def collate_cp(items: list[dict[str, Any]]) -> dict[str, Any]:
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
        "target_class": torch.stack([item["label"] for item in items]).long(),
        "outside_energy_ratios": torch.stack(
            [item["outside_energy_ratios"] for item in items]
        ).float(),
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
